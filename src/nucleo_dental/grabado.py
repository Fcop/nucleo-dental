"""Texto grabado (hundido) o en relieve sobre la guía (R-026).

El texto es libre y se entrega al ejecutar (nunca queda en el repositorio).
El usuario lo ubica con una caja ROI de Slicer, como las ventanas
(decisiones clínicas 2026-10-13):

- el texto se acuesta en el plano de las dos aristas más largas de la caja,
  con la línea a lo largo de la más larga, y se escala para caber;
- la arista más corta es la profundidad: lo que se hunde (grabado) o lo que
  sobresale (relieve);
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
    """Sólido cerrado de las letras dentro de la caja, con su escala y el ancho del trazo en mm."""
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
    # x, y de la fuente escalados al renglón; z de 0 a 1 recorre la profundidad, de la cara interna a la externa.
    mundo = (centro + np.outer(local[:, 0] * escala, u) + np.outer(local[:, 1] * escala, v)
             + np.outer((local[:, 2] - 0.5) * profundidad, n))
    malla.GetPoints().SetData(numpy_support.numpy_to_vtk(np.ascontiguousarray(mundo), deep=True))
    if np.dot(np.cross(u, v), n) < 0:                     # no ocurre (u = v × n), pero el sólido debe quedar bien orientado
        raise RuntimeError("Marco del texto mal orientado.")

    # Puntos del fondo de las letras: centros de los triángulos de los glifos, en la cara interna.
    tri_glifos = vtk.vtkTriangleFilter()
    tri_glifos.SetInputData(glifos)
    tri_glifos.Update()
    puntos_glifos = numpy_support.vtk_to_numpy(tri_glifos.GetOutput().GetPoints().GetData()).astype(float)
    triangulos = numpy_support.vtk_to_numpy(tri_glifos.GetOutput().GetPolys().GetConnectivityArray()).reshape(-1, 3)
    centros = puntos_glifos[triangulos].mean(axis=1) - medio
    fondo = centro + np.outer(centros[:, 0] * escala, u) + np.outer(centros[:, 1] * escala, v) - profundidad / 2 * n
    return {"malla": malla, "trazo_mm": float(trazo_fuente * escala), "alto_letra_mm": float(alto_texto * escala),
            "escala": float(escala), "afuera": n, "fondo": fondo, "profundidad_mm": float(profundidad)}


def revisar_texto(info: dict, modo: str, guia, resultado: dict, etiqueta: str) -> dict:
    """Trazo mínimo y, si es grabado, resina bajo las letras. Agrega los problemas al resultado."""
    from nucleo_dental.medicion import _dentro

    informe = {"modo": modo, "trazo_mm": info["trazo_mm"], "alto_letra_mm": info["alto_letra_mm"],
               "profundidad_mm": info["profundidad_mm"]}
    if info["trazo_mm"] < TRAZO_MINIMO_MM:
        resultado["valida"] = False
        resultado["problemas"].append(
            f"El trazo del {etiqueta} mide {info['trazo_mm']:.2f} mm (mínimo {TRAZO_MINIMO_MM}): agranda la caja "
            f"(letras de al menos {TRAZO_MINIMO_MM / info['trazo_mm'] * info['alto_letra_mm']:.1f} mm de alto).")
    if modo == "grabado":
        pasos = np.linspace(0.1, RESINA_BAJO_GRABADO_MM, 4)
        debajo = np.vstack([info["fondo"] - d * info["afuera"] for d in pasos])
        con_resina = _dentro(guia, debajo)
        informe["resina_bajo_grabado_suficiente"] = bool(con_resina.all())
        if not con_resina.all():
            resultado["valida"] = False
            resultado["problemas"].append(
                f"Bajo el {etiqueta} grabado quedan menos de {RESINA_BAJO_GRABADO_MM} mm de resina, o el texto sale de "
                "la guía: reduce la profundidad de la caja o muévela.")
    return informe
