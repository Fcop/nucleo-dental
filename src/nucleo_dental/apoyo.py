"""Región de apoyo de la guía dentosoportada sobre el escaneo intraoral (R-016).

Dos modos que producen la misma salida (una máscara por vértice del escaneo):

- automático: superficie dental a menos de R mm del eje del implante y a no
  menos de `margen_encia_mm` de la encía (decisiones clínicas 2026-10-08:
  2 dientes de apoyo por lado y hasta 1 mm del margen gingival);
- curva: interior de una curva cerrada que el usuario dibuja por puntos sobre
  el escaneo (aquí el usuario manda; puede incluir encía).

La máscara se convierte en parche con geometria.malla_guia.extraer_parche.
"""

from __future__ import annotations

import numpy as np
import vtk
from vtk.util import numpy_support

from nucleo_dental.geometria.malla_guia import ARRAY_REGION, extraer_parche

RADIO_APOYO_POR_DEFECTO_MM = 24.0   # cubre ~2 dientes por lado con la mayor superficie (decisión clínica 2026-10-08)
UMBRAL_DIENTE_MM = 0.5          # cubre el residuo del registro (p90 ≈ 0,31 mm en el caso real)
ANGULO_MAXIMO_GRADOS = 45.0     # orientación del escaneo frente a la superficie dental del CBCT
_DESPLAZAMIENTO_MM = 0.3        # se evalúa la orientación a esta distancia, fuera de la superficie
_PASO_GRADIENTE_MM = 0.05
_TOLERANCIA_MM = 1e-6


def clasificar_diente_encia(escaneo: vtk.vtkPolyData, dientes: vtk.vtkPolyData,
                            umbral_mm: float = UMBRAL_DIENTE_MM) -> np.ndarray:
    """True por cada vértice del escaneo que es superficie dental.

    Un vértice es diente si está a menos de `umbral_mm` de los dientes del CBCT
    y su superficie tiene la misma orientación que la del diente más cercano.
    La segunda condición separa la encía que abraza el cuello del diente: está
    cerca de la raíz del CBCT, pero mira hacia otro lado.
    """
    puntos = _puntos(escaneo)
    normales = _normales(escaneo)
    distancia = _distancia_a(dientes)

    es_diente = np.zeros(len(puntos), dtype=bool)
    candidatos = np.flatnonzero(distancia(puntos) < umbral_mm)
    if len(candidatos) == 0:
        return es_diente

    p, n = puntos[candidatos], normales[candidatos]
    # Punto de evaluación: a 0,3 mm de la superficie, hacia el lado más lejano del diente.
    lados = np.vstack([p + _DESPLAZAMIENTO_MM * n, p - _DESPLAZAMIENTO_MM * n])
    d_lados = distancia(lados).reshape(2, -1)
    q = np.where((d_lados[0] >= d_lados[1])[:, None], lados[: len(p)], lados[len(p):])

    # Gradiente del campo de distancia (dirección hacia afuera del diente más cercano).
    h = _PASO_GRADIENTE_MM
    ejes = np.eye(3)
    muestras = np.vstack([q + h * e for e in ejes] + [q - h * e for e in ejes])
    d = distancia(muestras).reshape(6, -1)
    gradiente = ((d[:3] - d[3:]) / (2 * h)).T
    coseno = np.abs(np.einsum("ij,ij->i", gradiente, n)) / (np.linalg.norm(gradiente, axis=1) + 1e-12)
    es_diente[candidatos] = coseno >= np.cos(np.radians(ANGULO_MAXIMO_GRADOS))
    return es_diente


def region_automatica(escaneo: vtk.vtkPolyData, dientes, punto_eje, direccion_eje,
                      radio_mm: float = RADIO_APOYO_POR_DEFECTO_MM, margen_encia_mm: float = 1.0,
                      tipo_soporte: str = "dentosoportada") -> dict:
    """Región de apoyo propuesta automáticamente, según el tipo de soporte (R-016, R-020).

    dentosoportada:     superficie dental a menos de `radio_mm` del eje y a no
                        menos de `margen_encia_mm` de la encía;
    dentomucosoportada: superficie dental y mucosa a menos de `radio_mm`;
    mucosoportada:      solo mucosa a menos de `radio_mm` (los dientes del CBCT
                        son opcionales; si se dan, se excluyen).
    """
    from nucleo_dental.guia import TIPOS_SOPORTE

    if tipo_soporte not in TIPOS_SOPORTE:
        raise ValueError(f"Tipo de soporte desconocido '{tipo_soporte}'; opciones: " + ", ".join(TIPOS_SOPORTE))
    for nombre, valor in (("radio_mm", radio_mm), ("margen_encia_mm", margen_encia_mm)):
        if not np.isfinite(valor) or valor < 0:
            raise ValueError(f"{nombre} debe ser un número mayor o igual que 0 (se recibió {valor}).")
    puntos = _puntos(escaneo)
    if dientes is None:
        if tipo_soporte != "mucosoportada":
            raise ValueError(f"Una guía {tipo_soporte} necesita los dientes del CBCT.")
        es_diente = np.zeros(len(puntos), dtype=bool)
    else:
        es_diente = clasificar_diente_encia(escaneo, dientes)
        if tipo_soporte != "mucosoportada" and not es_diente.any():
            raise ValueError("Ningún punto del escaneo coincide con los dientes del CBCT: revisa el registro.")

    a = np.asarray(punto_eje, dtype=float)
    u = np.asarray(direccion_eje, dtype=float)
    u = u / np.linalg.norm(u)
    relativo = puntos - a
    dentro_radio = np.linalg.norm(relativo - np.outer(relativo @ u, u), axis=1) <= radio_mm

    distancia_encia = np.full(len(puntos), np.inf)
    if tipo_soporte == "dentosoportada":
        encia = _submalla(escaneo, ~es_diente)
        if encia is not None:
            distancia_encia[es_diente] = _distancia_a(encia)(puntos[es_diente])
        mascara = es_diente & (distancia_encia >= margen_encia_mm - _TOLERANCIA_MM) & dentro_radio
    elif tipo_soporte == "dentomucosoportada":
        mascara = dentro_radio.copy()
    else:
        mascara = (~es_diente) & dentro_radio
    if not mascara.any():
        raise ValueError("La región de apoyo quedó vacía: aumenta el radio o reduce el margen a la encía.")
    altura = distancia_encia[mascara].min() if tipo_soporte == "dentosoportada" else 0.0
    return {
        "mascara": mascara,
        "es_diente": es_diente,
        "puntos_region": int(mascara.sum()),
        "altura_minima_sobre_encia_mm": float(altura),
        "radio_mm": float(radio_mm),
        "margen_encia_mm": float(margen_encia_mm),
        "tipo_soporte": tipo_soporte,
    }


def region_desde_curva(escaneo: vtk.vtkPolyData, puntos_curva, punto_interior=None) -> np.ndarray:
    """Máscara del interior de una curva cerrada dibujada por puntos sobre el escaneo.

    Los puntos se unen por el camino más corto sobre la malla (vtkSelectPolyData).
    punto_interior indica de qué lado de la curva está la región; por defecto
    el centro de los puntos de la curva. Los vértices sobre la curva se incluyen.
    """
    curva = np.atleast_2d(np.asarray(puntos_curva, dtype=float))
    if curva.shape[0] < 3 or curva.shape[1] != 3:
        raise ValueError("La curva cerrada necesita al menos 3 puntos (x, y, z).")
    lazo = vtk.vtkPoints()
    # Submuestreo: vtkSelectPolyData degrada con lazos muy densos (nota de malla_guia).
    lazo.SetData(numpy_support.numpy_to_vtk(np.ascontiguousarray(curva[:: max(1, len(curva) // 400)]), deep=True))

    seleccion = vtk.vtkSelectPolyData()
    seleccion.SetInputData(escaneo)
    seleccion.SetLoop(lazo)
    seleccion.GenerateSelectionScalarsOn()
    seleccion.SetSelectionModeToClosestPointRegion()
    seleccion.SetClosestPoint(*(curva.mean(axis=0) if punto_interior is None else np.asarray(punto_interior, float)))
    seleccion.Update()
    escalares = seleccion.GetOutput().GetPointData().GetScalars()
    if escalares is None:
        raise ValueError("No se pudo delimitar la región: la curva no está pegada a la superficie del escaneo.")
    # Convención de vtkSelectPolyData: negativo dentro, 0 sobre la curva.
    mascara = numpy_support.vtk_to_numpy(escalares) <= _TOLERANCIA_MM
    if not mascara.any():
        raise ValueError("La región dentro de la curva quedó vacía.")
    return mascara


def leer_puntos_slicer(ruta) -> np.ndarray:
    """Puntos de control (N×3, LPS) de un archivo de marcas de Slicer (.mrk.json).

    Se respeta el sistema de coordenadas que declara el archivo; si no lo
    declara no se adivina (RG-002).
    """
    import json
    from pathlib import Path

    ruta = Path(ruta)
    if not ruta.is_file():
        raise ValueError(f"No existe el archivo de puntos: {ruta}")
    try:
        datos = json.loads(ruta.read_text(encoding="utf-8"))
        marcas = datos["markups"][0]
        puntos = np.array([c["position"] for c in marcas["controlPoints"]], dtype=float)
    except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError):
        raise ValueError(f"{ruta} no es un archivo de marcas de Slicer válido (.mrk.json).") from None
    sistema = marcas.get("coordinateSystem")
    if sistema == "LPS":
        return puntos
    if sistema == "RAS":
        return puntos * np.array([-1.0, -1.0, 1.0])
    raise ValueError(f"{ruta}: falta coordinateSystem (LPS o RAS); no se adivina el sistema de coordenadas.")


def leer_cajas_slicer(ruta) -> list:
    """Cajas (ROI) de un archivo de marcas de Slicer (.mrk.json), en LPS: [{"centro", "ejes", "tamano"}].

    `ejes` es 3×3 con los ejes de la caja en las COLUMNAS (convención del
    esquema de marcas de Slicer: [o0, o3, o6] es el eje x de la caja) y
    `tamano` el largo de cada arista a lo largo de esos ejes. Se respeta el
    sistema que declara el archivo; si no lo declara no se adivina (RG-002).
    """
    import json
    from pathlib import Path

    ruta = Path(ruta)
    if not ruta.is_file():
        raise ValueError(f"No existe el archivo de cajas: {ruta}")
    try:
        marcas = json.loads(ruta.read_text(encoding="utf-8"))["markups"]
    except (json.JSONDecodeError, KeyError, TypeError):
        raise ValueError(f"{ruta} no es un archivo de marcas de Slicer válido (.mrk.json).") from None
    cajas = []
    for marca in marcas:
        if marca.get("type") != "ROI":
            continue
        sistema = marca.get("coordinateSystem")
        if sistema not in ("LPS", "RAS"):
            raise ValueError(f"{ruta}: falta coordinateSystem (LPS o RAS); no se adivina el sistema de coordenadas.")
        if marca.get("insideOut", False):
            raise ValueError(f"{ruta}: una caja 'insideOut' no se puede usar como ventana.")
        try:
            centro = np.array(marca["center"], dtype=float)
            ejes = np.array(marca.get("orientation", np.eye(3).ravel()), dtype=float).reshape(3, 3)
            tamano = np.array(marca["size"], dtype=float)
        except (KeyError, ValueError):
            raise ValueError(f"{ruta}: a una caja le falta center, size u orientation válidos.") from None
        if sistema == "RAS":                          # (x, y, z) -> (-x, -y, z), también para los ejes
            cambio = np.diag([-1.0, -1.0, 1.0])
            centro, ejes = cambio @ centro, cambio @ ejes
        if np.any(tamano <= 0) or not np.allclose(ejes.T @ ejes, np.eye(3), atol=1e-4):
            raise ValueError(f"{ruta}: caja con tamaño no positivo u orientación que no es una rotación.")
        cajas.append({"centro": centro, "ejes": ejes, "tamano": tamano})
    if not cajas:
        raise ValueError(f"{ruta} no contiene ninguna caja (ROI).")
    return cajas


def parche_de_apoyo(escaneo: vtk.vtkPolyData, mascara: np.ndarray, solo_mayor: bool = False,
                    datos: dict | None = None) -> vtk.vtkPolyData:
    """Parche abierto del escaneo con los triángulos cuyos tres vértices están en la región.

    `datos` ({nombre: valor por vértice del escaneo}) viaja con los vértices al parche.
    """
    copia = vtk.vtkPolyData()
    copia.DeepCopy(escaneo)
    region = numpy_support.numpy_to_vtk(mascara.astype(np.float32), deep=True)
    region.SetName(ARRAY_REGION)
    copia.GetPointData().AddArray(region)
    for nombre, valores in (datos or {}).items():
        arreglo = numpy_support.numpy_to_vtk(np.ascontiguousarray(valores, dtype=float), deep=True)
        arreglo.SetName(nombre)
        copia.GetPointData().AddArray(arreglo)
    return extraer_parche(copia, solo_mayor=solo_mayor)


def _puntos(malla: vtk.vtkPolyData) -> np.ndarray:
    return numpy_support.vtk_to_numpy(malla.GetPoints().GetData()).astype(float)


def _normales(malla: vtk.vtkPolyData) -> np.ndarray:
    filtro = vtk.vtkPolyDataNormals()
    filtro.SetInputData(malla)
    filtro.ComputePointNormalsOn()
    filtro.ComputeCellNormalsOff()
    filtro.ConsistencyOn()
    filtro.SplittingOff()
    filtro.Update()
    return numpy_support.vtk_to_numpy(filtro.GetOutput().GetPointData().GetNormals()).astype(float)


def _distancia_a(malla: vtk.vtkPolyData):
    implicita = vtk.vtkImplicitPolyDataDistance()
    implicita.SetInput(malla)

    def evaluar(puntos: np.ndarray) -> np.ndarray:
        salida = vtk.vtkDoubleArray()
        implicita.FunctionValue(numpy_support.numpy_to_vtk(np.ascontiguousarray(puntos, dtype=float), deep=True),
                                salida)
        return np.abs(numpy_support.vtk_to_numpy(salida))

    return evaluar


def _submalla(malla: vtk.vtkPolyData, mascara: np.ndarray):
    """Triángulos con sus tres vértices en la máscara, o None si no hay ninguno."""
    triangulos = numpy_support.vtk_to_numpy(malla.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    triangulos = triangulos[mascara[triangulos].all(axis=1)]
    if len(triangulos) == 0:
        return None
    offsets = np.arange(0, 3 * len(triangulos) + 1, 3, dtype=np.int64)
    celdas = vtk.vtkCellArray()
    celdas.SetData(numpy_support.numpy_to_vtkIdTypeArray(offsets, deep=True),
                   numpy_support.numpy_to_vtkIdTypeArray(triangulos.astype(np.int64).ravel(), deep=True))
    sub = vtk.vtkPolyData()
    sub.SetPoints(malla.GetPoints())
    sub.SetPolys(celdas)
    return sub
