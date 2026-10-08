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
def _comparar_con_medicion_manual(salida, codigo):
    from dorados import comparar, leer_esperado

    esperado = leer_esperado(CASO)
    comparar(salida["resultado"], esperado)
    assert codigo == esperado["codigo_salida"]


@pytest.mark.skipif(not (CASO / "esperado.json").is_file(), reason="sin medición manual de referencia")
def test_caso_real_contra_medicion_manual():
    """Verifica R-004, R-010, R-011 y R-013: el comando completo con --cbct reproduce las mediciones manuales en Slicer.

    Implante planificado en 3D Slicer (D 4 mm, L 8 mm, inclinado ~10°) sobre el
    canal derecho. Referencias medidas a mano: canal 2,330 mm desde el borde
    anterior del ápice (desde el centro daba 2,733); dientes 7,529 mm al
    premolar bajo la plataforma (a la corona del molar daba 5,95-6,43); hueso:
    la pared vestibular sobresale 0,216 mm a nivel de la plataforma (rojo).
    La cavidad interna mesial de ~2,6 mm³ se informa sin dar rojo.
    """
    from test_medir import ejecutar

    codigo, salida, stderr = ejecutar("medir", "--caso", CASO / "caso.json", "--cbct", CASO / "DICOM")
    assert salida is not None, stderr
    _comparar_con_medicion_manual(salida, codigo)
    assert salida["cbct"]["tamano"] == [601, 601, 601]
    hueso = salida["resultado"]["estructuras"]["hueso"]
    assert hueso["altura_critica_sobre_apice_mm"] == pytest.approx(8.0, abs=0.3)       # en la plataforma
    assert hueso["direccion_critica"][0] < -0.9                                        # -x: vestibular (lado derecho)
    assert [round(c["volumen_mm3"]) for c in hueso["cavidades_en_contacto"]] == [3]    # cavidad mesial informada


@pytest.mark.skipif(not (CASO / "esperado.json").is_file(), reason="sin medición manual de referencia")
def test_caso_real_con_implante_stl():
    """Verifica R-011, R-012 y R-013: leyendo el implante desde su STL se reproducen las mediciones manuales."""
    from test_medir import ejecutar

    codigo, salida, stderr = ejecutar(
        "medir", "--canal", CASO / "Mandibular canal.stl", "--dientes", CASO / "Lower Teeth.stl",
        "--hueso", CASO / "Mandible.stl", "--implante-stl", CASO / "implante.stl", "--apice-hacia", "abajo", "--diametro", "4", "--largo", "8",
        "--cbct", CASO / "DICOM")
    assert salida is not None, stderr
    _comparar_con_medicion_manual(salida, codigo)
    assert salida["parametros"]["implante"]["largo"] == pytest.approx(8.0, abs=0.01)


def test_caso_real_largo_declarado_erroneo_se_rechaza():
    """Verifica R-012: el error real del 2026-10-08 (declarar 10 mm para un implante de 8 mm) se detecta."""
    from test_medir import ejecutar

    codigo, salida, stderr = ejecutar(
        "medir", "--canal", CASO / "Mandibular canal.stl", "--implante-stl", CASO / "implante.stl",
        "--apice-hacia", "abajo", "--diametro", "4", "--largo", "10")
    assert codigo == 1
    assert salida is None
    assert "largo" in stderr


@pytest.mark.parametrize("modelo", MODELOS, ids=lambda p: p.name)
def test_modelo_real_en_lps_cae_dentro_y_en_ras_fuera(volumen, modelo):
    """Verifica R-010 con datos reales: el modelo en LPS está dentro del CBCT; su versión RAS, fuera."""
    vertices = _vertices(modelo)
    assert volumen.contiene(vertices).all()
    assert not volumen.contiene(vertices * [-1, -1, 1]).any()
