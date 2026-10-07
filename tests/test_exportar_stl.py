"""Exportación STL siempre en LPS."""

import numpy as np
import pytest
import vtk
from vtk.util import numpy_support

from nucleo_dental.geometria import malla_guia


def _triangulo() -> vtk.vtkPolyData:
    puntos = vtk.vtkPoints()
    for p in ([1.0, 2.0, 3.0], [4.0, 2.0, 3.0], [1.0, 5.0, 3.0]):
        puntos.InsertNextPoint(p)
    celdas = vtk.vtkCellArray()
    celdas.InsertNextCell(3, [0, 1, 2])
    pd = vtk.vtkPolyData()
    pd.SetPoints(puntos)
    pd.SetPolys(celdas)
    return pd


def _leer_puntos(ruta) -> np.ndarray:
    lector = vtk.vtkSTLReader()
    lector.SetFileName(str(ruta))
    lector.Update()
    return numpy_support.vtk_to_numpy(lector.GetOutput().GetPoints().GetData())


def test_exportar_lps_por_defecto_no_transforma(tmp_path):
    """Verifica R-002: una malla LPS del núcleo se exporta sin cambiar sus coordenadas."""
    ruta = tmp_path / "lps.stl"
    malla_guia.exportar_stl(_triangulo(), str(ruta))
    np.testing.assert_allclose(_leer_puntos(ruta)[0], [1.0, 2.0, 3.0])


def test_exportar_desde_ras_convierte_a_lps(tmp_path):
    """Verifica R-002: una malla declarada RAS se escribe en LPS, (x, y, z) -> (-x, -y, z)."""
    ruta = tmp_path / "ras.stl"
    malla_guia.exportar_stl(_triangulo(), str(ruta), sistema_entrada="RAS")
    np.testing.assert_allclose(_leer_puntos(ruta)[0], [-1.0, -2.0, 3.0])


def test_exportar_rechaza_sistema_desconocido(tmp_path):
    """Verifica R-002: un sistema de coordenadas no reconocido es un error, no se adivina."""
    with pytest.raises(ValueError):
        malla_guia.exportar_stl(_triangulo(), str(tmp_path / "x.stl"), sistema_entrada="XYZ")
