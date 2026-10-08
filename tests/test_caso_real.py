"""Tests con el caso real anonimizado de tests/casos_prueba/inferior_prueba.

Los datos NO están en el repositorio (.gitignore, regla 8 de CLAUDE.md): son
locales. Sin la carpeta, estos tests se saltan.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from vtk.util import numpy_support

from nucleo_dental.medicion import leer_stl

CASO = Path(__file__).parent / "casos_prueba" / "inferior_prueba"
MODELOS = sorted(p for p in CASO.glob("*.stl") if p.name != "Maxilla  Upper Skull.stl")
# El maxilar superior se excluye: su segmentación toca el borde del FOV (14 aristas
# abiertas, 0,2 % de vértices fuera del CBCT) y no se usa en la planificación inferior.

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
    """Verifica R-004: los STL reales (algunos con ~4 % de normales NaN) se leen y son superficies cerradas."""
    import vtk

    malla = leer_stl(modelo)
    bordes = vtk.vtkFeatureEdges()
    bordes.SetInputData(malla)
    bordes.BoundaryEdgesOn()
    bordes.NonManifoldEdgesOn()
    bordes.FeatureEdgesOff()
    bordes.ManifoldEdgesOff()
    bordes.Update()
    assert malla.GetNumberOfCells() > 0
    assert bordes.GetOutput().GetNumberOfCells() == 0


@pytest.mark.skipif(not (CASO / "esperado.json").is_file(), reason="sin medición manual de referencia")
def test_caso_real_contra_medicion_manual():
    """Verifica R-004, R-010 y R-011: el comando completo con --cbct reproduce la medición manual en Slicer.

    Implante planificado en 3D Slicer (D 4 mm, L 8 mm, inclinado ~10°) sobre el
    canal derecho. Referencia: distancia mínima medida a mano, 2,330 mm, desde el
    borde anterior del ápice; desde el centro del ápice la regla daba 2,733 mm.
    """
    from test_medir import ejecutar

    esperado = json.loads((CASO / "esperado.json").read_text(encoding="utf-8"))
    codigo, salida, stderr = ejecutar("medir", "--caso", CASO / "caso.json", "--cbct", CASO / "DICOM")
    assert salida is not None, stderr
    r = salida["resultado"]
    tol = esperado["tolerancia_mm"]
    assert r["distancia_mm"] == pytest.approx(esperado["distancia_mm"], abs=tol)
    assert r["colision"] is esperado["colision"]
    assert r["penetracion_mm"] == pytest.approx(esperado["penetracion_mm"], abs=tol)
    assert r["semaforo"] == esperado["semaforo"]
    assert codigo == esperado["codigo_salida"]
    assert salida["cbct"]["tamano"] == [601, 601, 601]


@pytest.mark.parametrize("modelo", MODELOS, ids=lambda p: p.name)
def test_modelo_real_en_lps_cae_dentro_y_en_ras_fuera(volumen, modelo):
    """Verifica R-010 con datos reales: el modelo en LPS está dentro del CBCT; su versión RAS, fuera."""
    vertices = _vertices(modelo)
    assert volumen.contiene(vertices).all()
    assert not volumen.contiene(vertices * [-1, -1, 1]).any()
