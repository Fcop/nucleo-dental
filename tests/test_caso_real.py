"""Tests con el caso real anonimizado de tests/casos_prueba/inferior_prueba.

Los datos NO están en el repositorio (.gitignore, regla 8 de CLAUDE.md): son
locales. Sin la carpeta, estos tests se saltan.
"""

from pathlib import Path

import numpy as np
import pytest
from vtk.util import numpy_support

from nucleo_dental.medicion import leer_stl

CASO = Path(__file__).parent / "casos_prueba" / "inferior_prueba"
MODELOS = sorted(CASO.glob("*.stl"))

pytestmark = pytest.mark.skipif(not (CASO / "DICOM").is_dir(), reason="caso real no disponible (datos locales)")


@pytest.fixture(scope="module")
def volumen():
    pytest.importorskip("SimpleITK")
    from nucleo_dental.cbct import leer_cbct

    return leer_cbct(CASO / "DICOM")


def _vertices(ruta):
    return numpy_support.vtk_to_numpy(leer_stl(ruta).GetPoints().GetData())


def test_cbct_real_tiene_geometria_coherente(volumen):
    """Verifica R-010: el CBCT real se lee como una sola serie con vóxel isotrópico de 0,15 mm."""
    assert volumen.tamano.tolist() == [601, 601, 601]
    np.testing.assert_allclose(volumen.espaciado, [0.15, 0.15, 0.15], atol=1e-6)


@pytest.mark.parametrize("modelo", MODELOS, ids=lambda p: p.name)
def test_modelo_real_se_lee_y_es_cerrado(modelo):
    """Verifica R-004: los STL reales (con ~4 % de normales NaN) se leen y son superficies cerradas."""
    import vtk

    malla = leer_stl(modelo)
    bordes = vtk.vtkFeatureEdges()
    bordes.SetInputData(malla)
    bordes.BoundaryEdgesOn()
    bordes.NonManifoldEdgesOn()
    bordes.FeatureEdgesOff()
    bordes.ManifoldEdgesOff()
    bordes.Update()
    assert malla.GetNumberOfCells() > 100_000
    assert bordes.GetOutput().GetNumberOfCells() == 0


@pytest.mark.parametrize("modelo", MODELOS, ids=lambda p: p.name)
def test_modelo_real_en_lps_cae_dentro_y_en_ras_fuera(volumen, modelo):
    """Verifica R-010 con datos reales: el modelo en LPS está dentro del CBCT; su versión RAS, fuera."""
    vertices = _vertices(modelo)
    assert volumen.contiene(vertices).all()
    assert not volumen.contiene(vertices * [-1, -1, 1]).any()
