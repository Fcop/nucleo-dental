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


def region_automatica(escaneo: vtk.vtkPolyData, dientes: vtk.vtkPolyData, punto_eje, direccion_eje,
                      radio_mm: float = 20.0, margen_encia_mm: float = 1.0) -> dict:
    """Región de apoyo propuesta automáticamente.

    Superficie dental del escaneo a menos de `radio_mm` del eje del implante
    (medido perpendicular al eje) y a no menos de `margen_encia_mm` de la encía.
    """
    for nombre, valor in (("radio_mm", radio_mm), ("margen_encia_mm", margen_encia_mm)):
        if not np.isfinite(valor) or valor < 0:
            raise ValueError(f"{nombre} debe ser un número mayor o igual que 0 (se recibió {valor}).")
    puntos = _puntos(escaneo)
    es_diente = clasificar_diente_encia(escaneo, dientes)
    if not es_diente.any():
        raise ValueError("Ningún punto del escaneo coincide con los dientes del CBCT: revisa el registro.")

    distancia_encia = np.full(len(puntos), np.inf)
    encia = _submalla(escaneo, ~es_diente)
    if encia is not None:
        distancia_encia[es_diente] = _distancia_a(encia)(puntos[es_diente])

    a = np.asarray(punto_eje, dtype=float)
    u = np.asarray(direccion_eje, dtype=float)
    u = u / np.linalg.norm(u)
    relativo = puntos - a
    radial = np.linalg.norm(relativo - np.outer(relativo @ u, u), axis=1)

    mascara = es_diente & (distancia_encia >= margen_encia_mm - _TOLERANCIA_MM) & (radial <= radio_mm)
    if not mascara.any():
        raise ValueError("La región de apoyo quedó vacía: aumenta el radio o reduce el margen a la encía.")
    return {
        "mascara": mascara,
        "es_diente": es_diente,
        "puntos_region": int(mascara.sum()),
        "altura_minima_sobre_encia_mm": float(distancia_encia[mascara].min()),
        "radio_mm": float(radio_mm),
        "margen_encia_mm": float(margen_encia_mm),
    }


def parche_de_apoyo(escaneo: vtk.vtkPolyData, mascara: np.ndarray, solo_mayor: bool = False) -> vtk.vtkPolyData:
    """Parche abierto del escaneo con los triángulos cuyos tres vértices están en la región."""
    copia = vtk.vtkPolyData()
    copia.DeepCopy(escaneo)
    region = numpy_support.numpy_to_vtk(mascara.astype(np.float32), deep=True)
    region.SetName(ARRAY_REGION)
    copia.GetPointData().AddArray(region)
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
