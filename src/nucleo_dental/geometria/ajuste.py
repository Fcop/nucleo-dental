"""Análisis de ajuste entre la guía y el escaneo, antes de imprimir (R-022).

Adaptado del ajuste.py del módulo GuiaCorteTraumatologica de Francisco (MIT).
Cambios respecto del original:

- el signo de la distancia se decide por lanzamiento de rayos
  (medicion._dentro), no por las normales del escaneo: junto a bordes vivos
  las normales invierten el signo (RG-014);
- el objetivo de separación es POR PUNTO: la tolerancia de ajuste sobre los
  dientes y, en una guía dentosoportada, la holgura sobre la encía (con su
  rampa). Un objetivo único marcaría como "holgada" toda la zona de encía;
- la distancia sin signo se evalúa de una vez en C++ (vtkImplicitPolyDataDistance
  sobre el array de puntos), sin bucles Python.

Convención de signos de la separación:
    negativa  la guía INVADE el escaneo (interferencia: no va a asentar)
    cero      contacto exacto
    positiva  hay separación

Desviación = separación − objetivo: negativa es más apretado que lo diseñado.
"""

from __future__ import annotations

import numpy as np
import vtk
from vtk.util import numpy_support

__all__ = ["separacion_con_signo", "analizar_ajuste", "histograma_ajuste", "mapa_de_ajuste"]

UMBRAL_ZONA_MM = 0.5            # sobre el objetivo: más lejos ya no es cara de asiento
ALINEACION_MINIMA = 0.5         # coseno normal–dirección al escaneo (descarta el canto de la cáscara)
INTERFERENCIA_NUMERICA_MM = 0.01
_PASO_NORMAL_MM = 0.05


def separacion_con_signo(guia: vtk.vtkPolyData, escaneo: vtk.vtkPolyData, puntos=None) -> np.ndarray:
    """Separación de cada punto (por defecto los de la guía) al escaneo cerrado; negativa dentro."""
    from nucleo_dental.medicion import _dentro

    if puntos is None:
        puntos = numpy_support.vtk_to_numpy(guia.GetPoints().GetData()).astype(float)
    implicita = vtk.vtkImplicitPolyDataDistance()
    implicita.SetInput(escaneo)
    salida = vtk.vtkDoubleArray()
    implicita.FunctionValue(numpy_support.numpy_to_vtk(np.ascontiguousarray(puntos, dtype=float), deep=True), salida)
    distancia = np.abs(numpy_support.vtk_to_numpy(salida))
    return np.where(_dentro(escaneo, puntos), -distancia, distancia)


def analizar_ajuste(guia: vtk.vtkPolyData, escaneo: vtk.vtkPolyData, tolerancia_ajuste_mm: float,
                    holgura_por_vertice=None, desvio_aceptable_mm: float = 0.10,
                    umbral_zona_mm: float = UMBRAL_ZONA_MM) -> dict:
    """Estadísticas de ajuste sobre la cara de asiento de la guía.

    tolerancia_ajuste_mm : separación diseñada sobre los dientes.
    holgura_por_vertice : holgura diseñada por vértice del ESCANEO (p. ej.
        ensamblaje.holgura_sobre_encia); el objetivo de cada punto de la guía
        es máx(tolerancia, holgura del vértice del escaneo más cercano).
    desvio_aceptable_mm : banda alrededor del objetivo que cuenta como
        "ideal" (confirmado por Francisco el 2026-10-09).
    """
    nf = vtk.vtkPolyDataNormals()
    nf.SetInputData(guia)
    nf.ComputePointNormalsOn()
    nf.ComputeCellNormalsOff()
    nf.SplittingOff()
    nf.ConsistencyOn()
    nf.AutoOrientNormalsOn()          # guía cerrada: normales hacia afuera
    nf.Update()
    con_normales = nf.GetOutput()
    puntos = numpy_support.vtk_to_numpy(con_normales.GetPoints().GetData()).astype(float)
    normales = numpy_support.vtk_to_numpy(con_normales.GetPointData().GetNormals()).astype(float)

    d = separacion_con_signo(guia, escaneo, puntos)
    objetivo = np.full(len(puntos), float(tolerancia_ajuste_mm))
    if holgura_por_vertice is not None:
        objetivo = np.maximum(objetivo, _valor_del_vertice_mas_cercano(escaneo, holgura_por_vertice, puntos))

    # Cara de asiento: la normal (hacia afuera de la guía) apunta al escaneo, es
    # decir, avanzar por ella acerca al escaneo; y está cerca del objetivo.
    d_avance = separacion_con_signo(guia, escaneo, puntos + _PASO_NORMAL_MM * normales)
    hacia_escaneo = -(d_avance - d) / _PASO_NORMAL_MM
    zona = ((hacia_escaneo > ALINEACION_MINIMA) & (d <= objetivo + umbral_zona_mm)) | (d < -INTERFERENCIA_NUMERICA_MM)
    if not zona.any():
        raise ValueError("Ningún punto de la guía asienta sobre el escaneo: revisa que estén en el mismo sistema de coordenadas.")

    dz, oz = d[zona], objetivo[zona]
    desvio = dz - oz
    interferencia = dz < -INTERFERENCIA_NUMERICA_MM
    apretado = ~interferencia & (desvio < -desvio_aceptable_mm)
    ideal = ~interferencia & (np.abs(desvio) <= desvio_aceptable_mm)
    holgado = desvio > desvio_aceptable_mm
    n = int(zona.sum())

    def pct(mascara):
        return 100.0 * float(mascara.sum()) / n

    return {
        "puntos_guia": int(len(d)),
        "puntos_cara_asiento": n,
        "tolerancia_ajuste_mm": float(tolerancia_ajuste_mm),
        "desvio_aceptable_mm": float(desvio_aceptable_mm),
        "separacion_mediana_mm": float(np.median(dz)),
        "separacion_min_mm": float(dz.min()),
        "separacion_max_mm": float(dz.max()),
        "desvio_mediano_mm": float(np.median(desvio)),
        "desvio_p05_mm": float(np.percentile(desvio, 5)),
        "desvio_p95_mm": float(np.percentile(desvio, 95)),
        "pct_interferencia": pct(interferencia),
        "pct_apretado": pct(apretado),
        "pct_ideal": pct(ideal),
        "pct_holgado": pct(holgado),
        "interferencia_max_mm": float(-dz.min()) if interferencia.any() else 0.0,
        "sin_interferencia": not bool(interferencia.any()),
        "_separacion": d,
        "_objetivo": objetivo,
        "_zona": zona,
    }


def histograma_ajuste(resultado: dict, ancho: int = 40) -> str:
    """Histograma de texto del desvío respecto del objetivo, en la cara de asiento."""
    zona = resultado["_zona"]
    desvio = (resultado["_separacion"] - resultado["_objetivo"])[zona]
    bordes = [-np.inf, -0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20, 0.50, np.inf]
    etiquetas = ["< -0,20", "-0,20 a -0,10", "-0,10 a -0,05", "-0,05 a 0,00", " 0,00 a 0,05",
                 " 0,05 a 0,10", " 0,10 a 0,20", " 0,20 a 0,50", "> 0,50"]
    conteos = np.histogram(desvio, bins=bordes)[0]
    tope = max(int(conteos.max()), 1)
    filas = ["Desvío respecto del objetivo (mm), cara de asiento:"]
    for etiqueta, c in zip(etiquetas, conteos):
        filas.append(f"  {etiqueta:>13} | {'#' * int(round(ancho * c / tope)):<{ancho}} {100.0 * c / len(desvio):5.1f} %")
    return "\n".join(filas)


def mapa_de_ajuste(guia: vtk.vtkPolyData, resultado: dict) -> vtk.vtkPolyData:
    """Copia de la guía con la separación, el objetivo y el desvío por punto (para colorear en Slicer)."""
    mapa = vtk.vtkPolyData()
    mapa.DeepCopy(guia)
    desvio = np.where(resultado["_zona"], resultado["_separacion"] - resultado["_objetivo"], np.nan)
    for nombre, valores in (("Separacion_mm", resultado["_separacion"]), ("Objetivo_mm", resultado["_objetivo"]),
                            ("Desvio_mm", desvio)):
        arreglo = numpy_support.numpy_to_vtk(np.ascontiguousarray(valores, dtype=np.float32), deep=True)
        arreglo.SetName(nombre)
        mapa.GetPointData().AddArray(arreglo)
    mapa.GetPointData().SetActiveScalars("Desvio_mm")
    return mapa


def _valor_del_vertice_mas_cercano(malla: vtk.vtkPolyData, valores, puntos: np.ndarray) -> np.ndarray:
    """Valor (por vértice de `malla`) del vértice más cercano a cada punto, resuelto en C++."""
    fuente = vtk.vtkPolyData()
    fuente.SetPoints(malla.GetPoints())
    arreglo = numpy_support.numpy_to_vtk(np.ascontiguousarray(valores, dtype=float), deep=True)
    arreglo.SetName("valor")
    fuente.GetPointData().AddArray(arreglo)
    localizador = vtk.vtkStaticPointLocator()
    localizador.SetDataSet(fuente)
    localizador.BuildLocator()
    destino = vtk.vtkPolyData()
    destino_puntos = vtk.vtkPoints()
    destino_puntos.SetData(numpy_support.numpy_to_vtk(np.ascontiguousarray(puntos, dtype=float), deep=True))
    destino.SetPoints(destino_puntos)
    interpolador = vtk.vtkPointInterpolator()
    interpolador.SetInputData(destino)
    interpolador.SetSourceData(fuente)
    interpolador.SetLocator(localizador)
    interpolador.SetKernel(vtk.vtkVoronoiKernel())
    interpolador.Update()
    return numpy_support.vtk_to_numpy(interpolador.GetOutput().GetPointData().GetArray("valor")).astype(float)
