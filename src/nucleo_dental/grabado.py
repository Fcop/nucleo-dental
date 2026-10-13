"""Texto grabado (hundido) o en relieve sobre la guía (R-026).

El texto es libre y se entrega al ejecutar (nunca queda en el repositorio).
El usuario lo ubica con una caja ROI de Slicer, como las ventanas
(decisiones clínicas 2026-10-13):

- el texto se acuesta en el plano de las dos aristas más largas de la caja,
  con la línea a lo largo de la más larga, y se escala para caber;
- la arista más corta es la profundidad: lo que se hunde (grabado) o lo que
  sobresale (relieve), medida desde la cara externa de la guía en cada punto:
  el texto SIGUE LA SUPERFICIE (en una caja plana sobre una superficie curva
  las letras de los extremos quedaban en el aire; probado 2026-10-13);
- "afuera" es la cara de la caja más lejana del escaneo: desde ahí se lee el
  texto derecho (sin espejo), con su "arriba" hacia el eje de inserción.

Validación: el trazo de las letras mide al menos 0,5 mm (TRAZO_MINIMO_MM) y,
si es grabado, bajo las letras queda al menos 1 mm de resina
(RESINA_BAJO_GRABADO_MM). Si no, la guía no es válida.
"""

from __future__ import annotations

import numpy as np

TRAZO_MINIMO_MM = 0.5            # decisión clínica 2026-10-13
RESINA_BAJO_GRABADO_MM = 1.0     # decisión clínica 2026-10-13
MODOS = ("grabado", "relieve")
_PASO_RASTER_MM = 0.05           # resolución de la imagen 2D de las letras
_SUAVE_RASTER_PX = 4             # semiancho del suavizado del borde (0,2 mm): el campo satura a ±0,2 mm
_LOTE_TRIANGULOS = 32


def _caja_suave(imagen: np.ndarray, radio: int) -> np.ndarray:
    """Promedio móvil (2·radio + 1) en ambos ejes, con sumas acumuladas (sin bucles por píxel)."""
    for eje in (0, 1):
        relleno = np.pad(imagen, [(radio + 1, radio) if e == eje else (0, 0) for e in (0, 1)], mode="edge")
        acumulada = np.cumsum(relleno, axis=eje)
        n = imagen.shape[eje]
        imagen = (np.take(acumulada, np.arange(2 * radio + 1, 2 * radio + 1 + n), axis=eje)
                  - np.take(acumulada, np.arange(0, n), axis=eje)) / (2 * radio + 1)
    return imagen


def _raster_de_letras(puntos_mm: np.ndarray, triangulos: np.ndarray, margen_mm: float):
    """Imagen 2D de la UNIÓN de los triángulos de las letras, con su borde suavizado, en mm con signo.

    vtkVectorText arma algunas letras (p. ej. el "1") con polígonos que se
    superponen: extruidas, la zona doble queda "fuera" por paridad y no se
    graba. En 2D la unión es un O lógico y la superposición no importa.
    """
    bajo = puntos_mm.min(axis=0) - margen_mm
    alto = puntos_mm.max(axis=0) + margen_mm
    xs = np.arange(bajo[0], alto[0] + _PASO_RASTER_MM, _PASO_RASTER_MM)
    ys = np.arange(bajo[1], alto[1] + _PASO_RASTER_MM, _PASO_RASTER_MM)
    px, py = np.meshgrid(xs, ys)
    pixeles = np.column_stack([px.ravel(), py.ravel()])
    dentro = np.zeros(len(pixeles), dtype=bool)
    for inicio in range(0, len(triangulos), _LOTE_TRIANGULOS):   # por lotes de triángulos, vectorizado por píxel
        t = puntos_mm[triangulos[inicio:inicio + _LOTE_TRIANGULOS]]          # (k, 3, 2)
        a, b, c = t[:, 0, None, :], t[:, 1, None, :], t[:, 2, None, :]
        d1 = (pixeles - b)[..., 0] * (a - b)[..., 1] - (a - b)[..., 0] * (pixeles - b)[..., 1]
        d2 = (pixeles - c)[..., 0] * (b - c)[..., 1] - (b - c)[..., 0] * (pixeles - c)[..., 1]
        d3 = (pixeles - a)[..., 0] * (c - a)[..., 1] - (c - a)[..., 0] * (pixeles - a)[..., 1]
        negativo = (d1 < 0) | (d2 < 0) | (d3 < 0)
        positivo = (d1 > 0) | (d2 > 0) | (d3 > 0)
        dentro |= (~(negativo & positivo)).any(axis=0)
    suave = _caja_suave(dentro.reshape(px.shape).astype(float), _SUAVE_RASTER_PX)
    valor = (suave - 0.5) * 2 * _SUAVE_RASTER_PX * _PASO_RASTER_MM          # ~mm con signo cerca del borde
    return xs, ys, valor


def _muestrear(xs, ys, valor, x, y, fuera: float):
    """Interpolación bilineal de la imagen en (x, y); `fuera` fuera de la imagen."""
    fx = (x - xs[0]) / (xs[1] - xs[0])
    fy = (y - ys[0]) / (ys[1] - ys[0])
    validos = (fx >= 0) & (fy >= 0) & (fx <= len(xs) - 1) & (fy <= len(ys) - 1)
    salida = np.full(len(x), fuera, dtype=float)
    i0 = np.clip(np.floor(fx[validos]).astype(int), 0, len(xs) - 2)
    j0 = np.clip(np.floor(fy[validos]).astype(int), 0, len(ys) - 2)
    tx, ty = fx[validos] - i0, fy[validos] - j0
    salida[validos] = ((1 - tx) * (1 - ty) * valor[j0, i0] + tx * (1 - ty) * valor[j0, i0 + 1]
                       + (1 - tx) * ty * valor[j0 + 1, i0] + tx * ty * valor[j0 + 1, i0 + 1])
    return salida


def _glifos(texto: str):
    """Polígonos del texto en el plano xy (unidades de la fuente) y el ancho del trazo de esa fuente."""
    import vtk

    letras = vtk.vtkVectorText()
    letras.SetText(texto)
    letras.Update()
    referencia = vtk.vtkVectorText()       # el ancho de la "I" es el trazo de la fuente
    referencia.SetText("I")
    referencia.Update()
    limites = referencia.GetOutput().GetBounds()
    return letras.GetOutput(), limites[1] - limites[0]


def _marco(caja: dict, escaneo, eje_insercion):
    """(centro, u = renglón, v = arriba del texto, n = afuera, largo, alto, profundidad) de la caja."""
    from nucleo_dental.apoyo import _distancia_a

    ejes = np.asarray(caja["ejes"], dtype=float)
    tamano = np.asarray(caja["tamano"], dtype=float)
    orden = np.argsort(tamano)[::-1]                      # de la arista más larga a la más corta
    a1, a2, a3 = (ejes[:, i] for i in orden)
    largo, alto, profundidad = tamano[orden]
    centro = np.asarray(caja["centro"], dtype=float)
    lados = np.array([centro + profundidad / 2 * a3, centro - profundidad / 2 * a3])
    distancia = _distancia_a(escaneo)(lados)
    n = a3 if distancia[0] >= distancia[1] else -a3
    v = a2
    if eje_insercion is not None and abs(np.dot(a2, eje_insercion)) > 0.2 and np.dot(a2, eje_insercion) < 0:
        v = -a2
    u = np.cross(v, n)                                    # u × v = n: se lee derecho mirando desde afuera
    return centro, u, v, n, largo, alto, profundidad


def texto_como_malla(texto: str, caja: dict, escaneo, eje_insercion=None) -> dict:
    """Prisma de las letras (su contorno extruido a lo largo de la normal de la caja), escala y trazo en mm.

    El prisma es largo a ambos lados de la caja: el ensamblaje lo corta a la
    capa de `profundidad_mm` junto a la cara externa de la guía, que en una
    superficie curva sube y baja a lo largo del renglón.
    """
    import vtk
    from vtk.util import numpy_support

    if not texto or not texto.strip():
        raise ValueError("El texto está vacío.")
    glifos, trazo_fuente = _glifos(texto)
    centro, u, v, n, largo, alto, profundidad = _marco(caja, escaneo, eje_insercion)
    limites = np.array(glifos.GetBounds())
    ancho_texto, alto_texto = limites[1] - limites[0], limites[3] - limites[2]
    escala = min(largo / ancho_texto, alto / alto_texto)
    medio = np.array([(limites[0] + limites[1]) / 2, (limites[2] + limites[3]) / 2, 0.0])

    extruido = vtk.vtkLinearExtrusionFilter()
    extruido.SetInputData(glifos)
    extruido.SetExtrusionTypeToVectorExtrusion()
    extruido.SetVector(0, 0, 1)
    extruido.SetScaleFactor(1.0)
    extruido.CappingOn()
    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(extruido.GetOutputPort())
    limpio = vtk.vtkCleanPolyData()
    limpio.SetInputConnection(tri.GetOutputPort())
    limpio.Update()
    malla = vtk.vtkPolyData()
    malla.DeepCopy(limpio.GetOutput())

    local = numpy_support.vtk_to_numpy(malla.GetPoints().GetData()).astype(float) - medio
    # x, y de la fuente escalados al renglón; z de 0 a 1 recorre el prisma a lo largo de la normal.
    medio_largo = max(largo, alto) / 2 + 3.0
    mundo = (centro + np.outer(local[:, 0] * escala, u) + np.outer(local[:, 1] * escala, v)
             + np.outer((local[:, 2] - 0.5) * 2 * medio_largo, n))
    malla.GetPoints().SetData(numpy_support.numpy_to_vtk(np.ascontiguousarray(mundo), deep=True))
    if np.dot(np.cross(u, v), n) < 0:                     # no ocurre (u = v × n), pero el sólido debe quedar bien orientado
        raise RuntimeError("Marco del texto mal orientado.")

    # Centros de los triángulos de los glifos sobre el plano medio de la caja (para ubicar letras).
    tri_glifos = vtk.vtkTriangleFilter()
    tri_glifos.SetInputData(glifos)
    tri_glifos.Update()
    puntos_glifos = numpy_support.vtk_to_numpy(tri_glifos.GetOutput().GetPoints().GetData()).astype(float)
    triangulos = numpy_support.vtk_to_numpy(tri_glifos.GetOutput().GetPolys().GetConnectivityArray()).reshape(-1, 3)
    centros = puntos_glifos[triangulos].mean(axis=1) - medio
    en_letras = centro + np.outer(centros[:, 0] * escala, u) + np.outer(centros[:, 1] * escala, v)

    # Campo del prisma (positivo dentro de las letras): imagen 2D de la unión, extruida a lo largo de n.
    xs, ys, valor = _raster_de_letras((puntos_glifos[:, :2] - medio[:2]) * escala, triangulos, 1.0)

    def evaluar(puntos):
        relativo = np.asarray(puntos, dtype=float) - centro
        en_plano = _muestrear(xs, ys, valor, relativo @ u, relativo @ v, fuera=-1.0)
        return np.minimum(en_plano, medio_largo - np.abs(relativo @ n))

    return {"malla": malla, "trazo_mm": float(trazo_fuente * escala), "alto_letra_mm": float(alto_texto * escala),
            "escala": float(escala), "afuera": n, "en_letras": en_letras, "profundidad_mm": float(profundidad),
            "evaluar": evaluar}


def revisar_texto(info: dict, modo: str, campo: dict, resultado: dict, etiqueta: str) -> dict:
    """Trazo mínimo, resina bajo el grabado y texto fuera de la guía (con lo que midió el ensamblaje)."""
    informe = {"modo": modo, "trazo_mm": info["trazo_mm"], "alto_letra_mm": info["alto_letra_mm"],
               "profundidad_mm": info["profundidad_mm"], **campo}
    if info["trazo_mm"] < TRAZO_MINIMO_MM:
        resultado["valida"] = False
        resultado["problemas"].append(
            f"El trazo del {etiqueta} mide {info['trazo_mm']:.2f} mm (mínimo {TRAZO_MINIMO_MM}): agranda la caja "
            f"(letras de al menos {TRAZO_MINIMO_MM / info['trazo_mm'] * info['alto_letra_mm']:.1f} mm de alto).")
    if modo == "grabado" and not campo["resina_bajo_grabado_suficiente"]:
        resultado["valida"] = False
        resultado["problemas"].append(
            f"Bajo el {etiqueta} grabado quedan menos de {RESINA_BAJO_GRABADO_MM} mm de resina: reduce la "
            "profundidad (la arista corta de la caja).")
    if campo["fraccion_sobre_la_guia"] < 0.999:
        resultado["avisos"].append(
            f"El {etiqueta} sale de la guía: solo el {100 * campo['fraccion_sobre_la_guia']:.0f} % de las letras "
            "cae sobre ella; muévelo o achica la caja.")
    return informe
