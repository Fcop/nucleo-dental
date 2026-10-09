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


def test_caso_real_registro_del_escaneo_mejora_el_calce():
    """Verifica R-015 con datos reales: el refinamiento parte del calce manual en Slicer y lo mejora.

    El 2026-10-08 el calce manual dejaba las coronas a 0,29 mm (mediana) de los
    dientes del CBCT; el refinamiento las deja a ~0,12 mm y corrige ~0,45 mm en
    el ápice, con una estabilidad cercana a 0,07 mm entre variantes.
    """
    from nucleo_dental.implante import Implante
    from nucleo_dental.registro import refinar_registro

    apice = Implante.desde_malla(leer_stl(CASO / "implante.stl"), "abajo").apice
    r = refinar_registro(leer_stl(CASO / "Modelo_inf.stl"), leer_stl(CASO / "Lower Teeth.stl"), punto=apice)
    assert r["desviacion_despues_mm"]["mediana"] < 0.5 * r["desviacion_antes_mm"]["mediana"]
    assert r["desviacion_despues_mm"]["p90"] < r["desviacion_antes_mm"]["p90"]
    assert r["estabilidad_mm"] < 0.1
    assert r["correccion_en_punto_mm"] > 3 * r["estabilidad_mm"]   # la corrección no es ruido


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


@pytest.mark.skipif(not (CASO / "CC.mrk.json").is_file() or not (CASO / "puntos_plano.mrk.json").is_file(),
                    reason="sin la curva de límites o el plano oclusal de Francisco")
@pytest.mark.parametrize("soporte", ["dentomucosoportada", "dentosoportada"])
def test_guia_real_con_curva_y_plano_oclusal(soporte):
    """Verifica R-021 con el caso real: curva CC y plano oclusal de Francisco dan una guía válida, fuera del hueso, con Ø5,3."""
    import vtk

    from nucleo_dental.apoyo import leer_puntos_slicer, region_desde_curva
    from nucleo_dental.ensamblaje import eje_desde_plano_oclusal, guia_quirurgica
    from nucleo_dental.implante import Implante
    from nucleo_dental.kits import ONEGUIDE
    from nucleo_dental.medicion import _dentro

    escaneo = leer_stl(CASO / "escaneo_registrado.stl")
    implante = Implante.desde_malla(leer_stl(CASO / "implante.stl"), "abajo")
    eje = eje_desde_plano_oclusal(leer_puntos_slicer(CASO / "puntos_plano.mrk.json"), implante.eje)
    mascara = region_desde_curva(escaneo, leer_puntos_slicer(CASO / "CC.mrk.json"))
    r = guia_quirurgica(escaneo, leer_stl(CASO / "Lower Teeth.stl"), implante, ONEGUIDE, "impresa", mascara, eje,
                        tipo_soporte=soporte)
    assert r["valida"], r["problemas"]
    puntos = numpy_support.vtk_to_numpy(r["guia"].GetPoints().GetData())
    assert not _dentro(leer_stl(CASO / "Mandible.stl"), puntos).any()

    arbol = vtk.vtkOBBTree()
    arbol.SetDataSet(r["guia"])
    arbol.BuildLocator()
    cara_superior = np.array(r["puente_y_columna"]["geometria"]["cara_superior"])
    centro = cara_superior - 2.0 * implante.eje
    lado = np.cross(implante.eje, [1.0, 0.0, 0.0])
    lado /= np.linalg.norm(lado)
    cortes = vtk.vtkPoints()
    arbol.IntersectWithLine(centro, centro + 10 * lado, cortes, None)
    assert np.linalg.norm(np.array(cortes.GetPoint(0)) - centro) == pytest.approx(5.3 / 2, abs=0.05)


@pytest.mark.skipif(not all((CASO / n).is_file() for n in ("CC.mrk.json", "puntos_plano.mrk.json", "Roi_ventana.mrk.json")),
                    reason="sin la curva, el plano oclusal o la caja de ventana de Francisco")
def test_ventana_real_desde_roi_de_slicer():
    """Verifica R-024 con el caso real: la caja ROI que Francisco guardó en Slicer abre la guía y la guía sigue válida."""
    from nucleo_dental.apoyo import leer_cajas_slicer, leer_puntos_slicer, region_desde_curva
    from nucleo_dental.ensamblaje import eje_desde_plano_oclusal, guia_quirurgica
    from nucleo_dental.implante import Implante
    from nucleo_dental.kits import ONEGUIDE

    escaneo = leer_stl(CASO / "escaneo_registrado.stl")
    implante = Implante.desde_malla(leer_stl(CASO / "implante.stl"), "abajo")
    eje = eje_desde_plano_oclusal(leer_puntos_slicer(CASO / "puntos_plano.mrk.json"), implante.eje)
    mascara = region_desde_curva(escaneo, leer_puntos_slicer(CASO / "CC.mrk.json"))
    ventanas = leer_cajas_slicer(CASO / "Roi_ventana.mrk.json")
    r = guia_quirurgica(escaneo, leer_stl(CASO / "Lower Teeth.stl"), implante, ONEGUIDE, "impresa", mascara, eje,
                        tipo_soporte="dentosoportada", ventanas=ventanas)
    assert r["valida"], r["problemas"]
    assert [v["corta_la_guia"] for v in r["ventanas"]] == [True]
