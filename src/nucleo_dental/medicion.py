"""Distancia implante–canal mandibular, colisión, penetración y semáforo.

Requisitos: R-004, R-005, R-006 y R-009. Ver CONTEXT.md para el significado
de distancia, margen, colisión, penetración y semáforo.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import vtk
from vtk.util import numpy_support

from nucleo_dental.implante import Implante

MARGEN_POR_DEFECTO_MM = 2.0
# Margen implante–diente vecino (RP-001 → R-013, decisión clínica 2026-10-07).
MARGEN_DIENTES_POR_DEFECTO_MM = 1.5
# Estructuras cuyo margen es óseo: se evalúan solo bajo el plano de la
# plataforma, sin la corona (decisión clínica 2026-10-08, R-013).
ESTRUCTURAS_BAJO_PLATAFORMA = {"dientes"}
# Espesor óseo mínimo en las paredes laterales (R-014, decisión clínica 2026-10-08).
MARGEN_HUESO_POR_DEFECTO_MM = 1.5
# Estructuras que se evalúan por espesor alrededor del implante, no por distancia.
ESTRUCTURAS_ESPESOR = {"hueso"}
_PASO_RAYOS_MM = 0.25        # separación de los rayos en altura y contorno, y paso grueso a lo largo
_PASO_FINO_MM = 0.01         # resolución final del espesor
_ESPESOR_MAXIMO_MM = 10.0    # más allá, el espesor se informa como 10 mm

# Separación entre puntos muestreados sobre el implante. Con 0,05 mm el error
# de muestreo queda bajo los 0,05 mm que exige R-004.
PASO_MUESTREO_MM = 0.05


_REGISTRO_STL = np.dtype([("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("atributo", "<u2")])


def leer_stl(ruta) -> vtk.vtkPolyData:
    """Lee una malla STL en LPS y mm.

    Las normales guardadas en el archivo se ignoran: el núcleo las recalcula.
    Exportadores reales escriben normales NaN y el lector de VTK aborta con
    ellas, aunque la geometría esté sana.
    """
    ruta = Path(ruta)
    if not ruta.is_file():
        raise ValueError(f"No existe el archivo de malla: {ruta}")
    datos = ruta.read_bytes()
    if len(datos) >= 84:
        n = int.from_bytes(datos[80:84], "little")
        if n > 0 and len(datos) == 84 + 50 * n:
            return _malla_desde_stl_binario(np.frombuffer(datos, _REGISTRO_STL, n, 84), ruta)

    lector = vtk.vtkSTLReader()   # STL de texto (ASCII)
    lector.SetFileName(str(ruta))
    lector.Update()
    malla = lector.GetOutput()
    if malla.GetNumberOfPoints() == 0 or malla.GetNumberOfCells() == 0:
        raise ValueError(f"La malla está vacía o no es un STL válido: {ruta}")
    return malla


def _malla_desde_stl_binario(registros: np.ndarray, ruta: Path) -> vtk.vtkPolyData:
    """Construye la malla con los vértices de cada triángulo, uniendo los vértices repetidos."""
    vertices = registros["v"].reshape(-1, 3)
    if not np.all(np.isfinite(vertices)):
        malos = int((~np.isfinite(registros["v"]).all(axis=(1, 2))).sum())
        raise ValueError(f"El STL tiene {malos} triángulos con vértices no finitos (NaN o infinito): {ruta}")
    puntos, indices = np.unique(vertices, axis=0, return_inverse=True)
    triangulos = indices.reshape(-1, 3).astype(np.int64)

    vtk_puntos = vtk.vtkPoints()
    vtk_puntos.SetData(numpy_support.numpy_to_vtk(puntos.astype(float), deep=True))
    offsets = np.arange(0, 3 * len(triangulos) + 1, 3, dtype=np.int64)
    celdas = vtk.vtkCellArray()
    celdas.SetData(numpy_support.numpy_to_vtkIdTypeArray(offsets, deep=True),
                   numpy_support.numpy_to_vtkIdTypeArray(triangulos.ravel(), deep=True))
    malla = vtk.vtkPolyData()
    malla.SetPoints(vtk_puntos)
    malla.SetPolys(celdas)
    return malla


def medir(implante: Implante, malla_canal: vtk.vtkPolyData,
          margen: float = MARGEN_POR_DEFECTO_MM) -> dict:
    """Mide la distancia del implante al canal y emite el semáforo.

    La distancia se calcula en dos direcciones y se toma la menor:
      1. vértices del canal → implante (exacta, pero ciega a las caras);
      2. puntos sobre el implante → superficie del canal (ve las caras).
    """
    margen = float(margen)
    if not np.isfinite(margen) or margen < 0:
        raise ValueError(f"El margen debe ser un número mayor o igual que 0 mm (se recibió {margen}).")
    _exigir_superficie_cerrada(malla_canal)

    vertices = numpy_support.vtk_to_numpy(malla_canal.GetPoints().GetData()).astype(float)
    superficie = implante.puntos_superficie(PASO_MUESTREO_MM)
    distancia_al_canal = _distancia_con_signo(malla_canal)

    d_vertices = implante.distancia_a_puntos(vertices)
    d_superficie = distancia_al_canal(superficie)

    colision = bool(implante.contiene(vertices).any() or (d_superficie < 0).any())

    if colision:
        distancia = 0.0
        penetracion = _penetracion(implante, malla_canal, distancia_al_canal, d_superficie)
    else:
        distancia = float(min(d_vertices.min(), d_superficie.min()))
        penetracion = 0.0

    return {
        "distancia_mm": distancia,
        "colision": colision,
        "penetracion_mm": penetracion,
        "semaforo": "rojo" if colision or distancia < margen else "verde",
    }


def recortar_bajo_plataforma(malla: vtk.vtkPolyData, implante: Implante):
    """Parte de la malla bajo el plano de la plataforma, como sólido cerrado (R-013).

    El margen a los dientes es óseo (raíz), no protésico: la corona sobre la
    plataforma no cuenta. El corte se cierra con una tapa en el plano para que
    la colisión siga bien definida. Devuelve None si no queda nada bajo el plano.
    """
    planos = vtk.vtkPlaneCollection()
    plano = vtk.vtkPlane()
    plano.SetOrigin(*implante.plataforma)
    plano.SetNormal(*(-implante.eje))   # se conserva el lado hacia el ápice
    planos.AddItem(plano)

    # El recorte depende de la orientación de los triángulos: con normales
    # invertidas deja la malla abierta (RG-010). Se orientan antes de cortar.
    orientada = vtk.vtkPolyDataNormals()
    orientada.SetInputData(malla)
    orientada.ConsistencyOn()
    orientada.AutoOrientNormalsOn()
    orientada.SplittingOff()

    recorte = vtk.vtkClipClosedSurface()
    recorte.SetInputConnection(orientada.GetOutputPort())
    recorte.SetClippingPlanes(planos)
    recorte.GenerateFacesOn()
    triangulos = vtk.vtkTriangleFilter()
    triangulos.SetInputConnection(recorte.GetOutputPort())
    limpio = vtk.vtkCleanPolyData()
    limpio.SetInputConnection(triangulos.GetOutputPort())
    limpio.Update()

    resultado = vtk.vtkPolyData()
    resultado.DeepCopy(limpio.GetOutput())
    return resultado if resultado.GetNumberOfCells() > 0 else None


def espesor_oseo(implante: Implante, malla_hueso: vtk.vtkPolyData, margen: float = None) -> dict:
    """Espesor mínimo de hueso alrededor de las paredes laterales del implante (R-014).

    Desde puntos de la pared lateral (cada ~0,25 mm en altura y en contorno,
    del borde del ápice a la plataforma) se lanzan rayos perpendiculares al
    eje, hacia afuera, y se mide cuánto hueso atraviesa cada uno antes de
    salir. Bajo el ápice no se mide (decisión clínica 2026-10-08). Una pared
    fuera del hueso tiene espesor 0. El valor se redondea hacia abajo
    (conservador) con resolución de 0,01 mm.
    """
    margen = MARGEN_HUESO_POR_DEFECTO_MM if margen is None else float(margen)
    if not np.isfinite(margen) or margen < 0:
        raise ValueError(f"El margen debe ser un número mayor o igual que 0 mm (se recibió {margen}).")
    _exigir_superficie_cerrada(malla_hueso)

    e1, e2 = implante._base_perpendicular()
    n_direcciones = 4 * int(np.ceil(2 * np.pi * implante.radio / (4 * _PASO_RAYOS_MM)))
    angulos = 2 * np.pi * np.arange(n_direcciones) / n_direcciones
    direcciones = np.cos(angulos)[:, None] * e1 + np.sin(angulos)[:, None] * e2
    alturas = np.linspace(0.0, implante.largo, int(np.ceil(implante.largo / _PASO_RAYOS_MM)) + 1)

    origen = (implante.apice + alturas[:, None, None] * implante.eje
              + implante.radio * direcciones[None, :, :]).reshape(-1, 3)
    direccion = np.tile(direcciones, (len(alturas), 1))
    altura = np.repeat(alturas, n_direcciones)

    # 1) Muestreo grueso a lo largo de cada rayo: primer paso fuera del hueso.
    pasos = np.arange(0.0, _ESPESOR_MAXIMO_MM + _PASO_RAYOS_MM / 2, _PASO_RAYOS_MM)
    puntos = origen[:, None, :] + pasos[None, :, None] * direccion[:, None, :]
    dentro = _dentro(malla_hueso, puntos.reshape(-1, 3)).reshape(len(origen), len(pasos))
    sale = ~dentro
    tiene_salida = sale.any(axis=1)
    primer_fuera = np.argmax(sale, axis=1)
    espesor = np.where(tiene_salida, 0.0, _ESPESOR_MAXIMO_MM)

    # 2) Refinado fino entre el último paso dentro y el primero fuera.
    refinar = tiene_salida & (primer_fuera > 0)
    if refinar.any():
        base = pasos[primer_fuera[refinar] - 1]
        finos = np.arange(_PASO_FINO_MM, _PASO_RAYOS_MM + _PASO_FINO_MM / 2, _PASO_FINO_MM)
        s = base[:, None] + finos[None, :]
        puntos_finos = origen[refinar][:, None, :] + s[:, :, None] * direccion[refinar][:, None, :]
        dentro_fino = _dentro(malla_hueso, puntos_finos.reshape(-1, 3)).reshape(s.shape)
        # Último paso fino aún dentro (o la base si el primero ya sale): valor conservador.
        n_dentro = np.cumprod(dentro_fino, axis=1).sum(axis=1)
        espesor[refinar] = base + n_dentro * _PASO_FINO_MM

    i = int(np.argmin(espesor))
    minimo = float(espesor[i])
    return {
        "espesor_minimo_mm": minimo,
        "margen_mm": margen,
        "semaforo": "verde" if minimo >= margen else "rojo",
        "direccion_minimo": [float(v) for v in direccion[i]],
        "altura_minimo_sobre_apice_mm": float(altura[i]),
    }


def evaluar_plan(implante: Implante, estructuras: dict) -> dict:
    """Evalúa el implante contra cada estructura con su propio margen (R-013).

    estructuras: {nombre: (malla, margen_mm)}, p. ej. {"canal": (..., 2.0), "dientes": (..., 1.5)}.
    Los dientes se evalúan solo bajo la plataforma (ver recortar_bajo_plataforma).
    El semáforo global es rojo si alguna estructura está en rojo.
    """
    if not estructuras:
        raise ValueError("No hay ninguna estructura contra la cual evaluar el implante.")
    por_estructura = {}
    for nombre, (malla, margen) in estructuras.items():
        if nombre in ESTRUCTURAS_ESPESOR:
            por_estructura[nombre] = espesor_oseo(implante, malla, margen)
            continue
        if nombre in ESTRUCTURAS_BAJO_PLATAFORMA:
            _exigir_superficie_cerrada(malla)
            malla = recortar_bajo_plataforma(malla, implante)
            if malla is None:
                raise ValueError(f"La malla de {nombre} no tiene ninguna parte bajo la plataforma del implante; "
                                 "revisa la segmentación o la posición del implante.")
        resultado = medir(implante, malla, margen)
        resultado["margen_mm"] = float(margen)
        por_estructura[nombre] = resultado
    rojo = any(r["semaforo"] == "rojo" for r in por_estructura.values())
    return {"semaforo": "rojo" if rojo else "verde", "estructuras": por_estructura}


def _penetracion(implante, malla_canal, distancia_al_canal, d_superficie) -> float:
    """Profundidad máxima del implante dentro del canal (R-009).

    El punto más hondo puede estar en el interior del implante (p. ej. cuando
    el implante atraviesa el canal), por eso se muestrea también su volumen,
    limitado a la caja del canal para no evaluar puntos que no pueden estar dentro.
    """
    interiores = implante.puntos_interiores(PASO_MUESTREO_MM)
    caja = np.array(malla_canal.GetBounds()).reshape(3, 2)
    en_caja = np.all((interiores >= caja[:, 0] - PASO_MUESTREO_MM)
                     & (interiores <= caja[:, 1] + PASO_MUESTREO_MM), axis=1)
    d_interior = distancia_al_canal(interiores[en_caja]) if en_caja.any() else np.zeros(1)
    mas_hondo = min(d_superficie.min(), d_interior.min())
    return float(max(-mas_hondo, 0.0))


def _exigir_superficie_cerrada(malla: vtk.vtkPolyData) -> None:
    """Sin superficie cerrada no hay 'dentro' ni 'fuera': la colisión sería indecidible."""
    bordes = vtk.vtkFeatureEdges()
    bordes.SetInputData(malla)
    bordes.BoundaryEdgesOn()
    bordes.NonManifoldEdgesOn()
    bordes.FeatureEdgesOff()
    bordes.ManifoldEdgesOff()
    bordes.Update()
    n = bordes.GetOutput().GetNumberOfCells()
    if n:
        raise ValueError(
            f"La malla de la estructura (canal o dientes) no es una superficie cerrada ({n} aristas abiertas "
            "o no manifold). Ciérrala (tapas en los extremos) antes de medir.")


def _distancia_con_signo(malla: vtk.vtkPolyData):
    """Función vectorizada: distancia con signo a la superficie (negativa dentro).

    El valor absoluto sale de vtkImplicitPolyDataDistance; el signo (dentro o
    fuera) NO se toma de ahí: ese signo depende de las normales y se equivoca
    junto a bordes vivos, como la tapa que deja el recorte bajo la plataforma.
    La pertenencia se decide con vtkSelectEnclosedPoints (lanzamiento de rayos),
    que solo exige una superficie cerrada (RG-010, RG-014).
    """
    implicita = vtk.vtkImplicitPolyDataDistance()
    implicita.SetInput(malla)

    def evaluar(puntos: np.ndarray) -> np.ndarray:
        puntos = np.ascontiguousarray(puntos, dtype=float)
        entrada = numpy_support.numpy_to_vtk(puntos, deep=True)
        salida = vtk.vtkDoubleArray()
        implicita.FunctionValue(entrada, salida)
        distancia = np.abs(numpy_support.vtk_to_numpy(salida))
        return np.where(_dentro(malla, puntos), -distancia, distancia)

    return evaluar


def _dentro(malla: vtk.vtkPolyData, puntos: np.ndarray) -> np.ndarray:
    """True por cada punto (N×3) encerrado por la superficie cerrada (lanzamiento de rayos)."""
    vtk_puntos = vtk.vtkPoints()
    vtk_puntos.SetData(numpy_support.numpy_to_vtk(np.ascontiguousarray(puntos, dtype=float), deep=True))
    consulta = vtk.vtkPolyData()
    consulta.SetPoints(vtk_puntos)
    encerrados = vtk.vtkSelectEnclosedPoints()
    encerrados.SetInputData(consulta)
    encerrados.SetSurfaceData(malla)
    encerrados.SetTolerance(1e-9)
    encerrados.Update()
    return numpy_support.vtk_to_numpy(
        encerrados.GetOutput().GetPointData().GetArray("SelectedPoints")).astype(bool)
