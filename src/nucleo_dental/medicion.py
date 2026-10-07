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

# Separación entre puntos muestreados sobre el implante. Con 0,05 mm el error
# de muestreo queda bajo los 0,05 mm que exige R-004.
PASO_MUESTREO_MM = 0.05


def leer_stl(ruta) -> vtk.vtkPolyData:
    """Lee una malla STL en LPS y mm."""
    ruta = Path(ruta)
    if not ruta.is_file():
        raise ValueError(f"No existe el archivo de malla: {ruta}")
    lector = vtk.vtkSTLReader()
    lector.SetFileName(str(ruta))
    lector.Update()
    malla = lector.GetOutput()
    if malla.GetNumberOfPoints() == 0 or malla.GetNumberOfCells() == 0:
        raise ValueError(f"La malla está vacía o no es un STL válido: {ruta}")
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
            f"La malla del canal no es una superficie cerrada ({n} aristas abiertas o no manifold). "
            "Ciérrala (tapas en los extremos) antes de medir.")


def _distancia_con_signo(malla: vtk.vtkPolyData):
    """Función vectorizada: distancia con signo a la superficie (negativa dentro)."""
    normales = vtk.vtkPolyDataNormals()
    normales.SetInputData(malla)
    normales.ConsistencyOn()
    normales.AutoOrientNormalsOn()
    normales.SplittingOff()
    normales.Update()

    implicita = vtk.vtkImplicitPolyDataDistance()
    implicita.SetInput(normales.GetOutput())

    def evaluar(puntos: np.ndarray) -> np.ndarray:
        entrada = numpy_support.numpy_to_vtk(np.ascontiguousarray(puntos, dtype=float), deep=True)
        salida = vtk.vtkDoubleArray()
        implicita.FunctionValue(entrada, salida)
        return numpy_support.vtk_to_numpy(salida).copy()

    return evaluar
