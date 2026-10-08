"""Región de apoyo de la guía sobre el escaneo intraoral (R-016).

Caso dorado 012 calculado a mano por Francisco: con R = 20 mm desde el eje
(x = 0) quedan dentro los dientes 1 y 2 de cada lado y fuera el 3; el apoyo
empieza a 1 mm sobre la encía y la encía no forma parte de la región.
"""

import json

import numpy as np
import pytest
from vtk.util import numpy_support

from dorados import CASOS_DORADOS
from nucleo_dental.apoyo import clasificar_diente_encia, region_automatica
from nucleo_dental.medicion import leer_stl

CASO_012 = CASOS_DORADOS / "apoyo" / "caso_012_arcada_recta"


def _cargar_012():
    caso = json.loads((CASO_012 / "caso.json").read_text(encoding="utf-8"))
    esperado = json.loads((CASO_012 / "esperado.json").read_text(encoding="utf-8"))
    escaneo = leer_stl(CASO_012 / caso["escaneo"])
    dientes = leer_stl(CASO_012 / caso["dientes"])
    return caso, esperado, escaneo, dientes


@pytest.fixture(scope="module")
def region_012():
    caso, esperado, escaneo, dientes = _cargar_012()
    region = region_automatica(escaneo, dientes, caso["eje"]["punto"], caso["eje"]["direccion"],
                               radio_mm=caso["radio_mm"], margen_encia_mm=caso["margen_encia_mm"])
    puntos = numpy_support.vtk_to_numpy(escaneo.GetPoints().GetData()).astype(float)
    return esperado, puntos, region


def test_caso_dorado_012_dientes_dentro_y_fuera(region_012):
    """Verifica R-016: con R = 20 mm quedan dentro los dientes 1 y 2 de cada lado y fuera el 3."""
    esperado, puntos, region = region_012
    tol = esperado["tolerancia_mm"]
    sel = puntos[region["mascara"]]
    for x0, x1 in esperado["dientes_dentro_x"]:
        en_diente = (sel[:, 0] >= x0 - tol) & (sel[:, 0] <= x1 + tol)
        assert (en_diente & np.isclose(sel[:, 2], 8.0)).any(), f"falta la oclusal del diente {x0}..{x1}"
    for x0, x1 in esperado["dientes_fuera_x"]:
        assert not ((sel[:, 0] > x0 + tol) & (sel[:, 0] < x1 - tol)).any(), f"el diente {x0}..{x1} no debía entrar"


def test_caso_dorado_012_altura_minima_sobre_la_encia(region_012):
    """Verifica R-016: el apoyo empieza a 1 mm de la encía (las paredes bajo z = 1 quedan fuera)."""
    esperado, puntos, region = region_012
    tol = esperado["tolerancia_mm"]
    z = puntos[region["mascara"], 2]
    assert z.min() == pytest.approx(esperado["altura_minima_apoyo_mm"], abs=tol)
    assert region["altura_minima_sobre_encia_mm"] == pytest.approx(esperado["altura_minima_apoyo_mm"], abs=tol)


def test_caso_dorado_012_la_encia_no_entra(region_012):
    """Verifica R-016: en modo automático la encía (z = 0) no forma parte de la región."""
    esperado, puntos, region = region_012
    assert esperado["incluye_encia"] is False
    assert not (region["mascara"] & (puntos[:, 2] < 0.5)).any()


def test_encia_pegada_al_diente_no_se_confunde_con_diente():
    """Verifica R-016: la encía junto a la pared del diente (a 0,25 mm de la raíz del CBCT) se clasifica como encía.

    Por distancia sola sería "diente" (la raíz del CBCT sigue bajo la encía); la orientación
    de la superficie (encía horizontal, pared vertical) las separa.
    """
    _, _, escaneo, dientes = _cargar_012()
    puntos = numpy_support.vtk_to_numpy(escaneo.GetPoints().GetData()).astype(float)
    es_diente = clasificar_diente_encia(escaneo, dientes)
    junto_a_pared = (np.abs(puntos[:, 2]) < 1e-6) & (np.abs(puntos[:, 0] - 4.75) < 1e-6) & (np.abs(puntos[:, 1]) < 2)
    assert junto_a_pared.any()
    assert not es_diente[junto_a_pared].any()
    pared = (np.abs(puntos[:, 0] - 5.0) < 1e-6) & (puntos[:, 2] > 2) & (puntos[:, 2] < 7) & (np.abs(puntos[:, 1]) < 2)
    assert es_diente[pared].all()


CASO_014 = CASOS_DORADOS / "apoyo" / "caso_014_curva_oclusal"


def test_caso_dorado_014_curva_sobre_la_oclusal():
    """Verifica R-016: una curva cuadrada de 4 x 4 mm sobre la oclusal encierra x 6-10, y -2..2 (16 mm²), sin paredes."""
    from nucleo_dental.apoyo import parche_de_apoyo, region_desde_curva

    caso = json.loads((CASO_014 / "caso.json").read_text(encoding="utf-8"))
    esperado = json.loads((CASO_014 / "esperado.json").read_text(encoding="utf-8"))
    escaneo = leer_stl(CASO_014 / caso["escaneo"])
    mascara = region_desde_curva(escaneo, caso["curva"], caso["punto_interior"])
    puntos = numpy_support.vtk_to_numpy(escaneo.GetPoints().GetData()).astype(float)
    sel = puntos[mascara]
    tol = esperado["tolerancia_mm"]

    assert sel[:, 0].min() == pytest.approx(esperado["rango_x"][0], abs=tol)
    assert sel[:, 0].max() == pytest.approx(esperado["rango_x"][1], abs=tol)
    assert sel[:, 1].min() == pytest.approx(esperado["rango_y"][0], abs=tol)
    assert sel[:, 1].max() == pytest.approx(esperado["rango_y"][1], abs=tol)
    assert np.allclose(sel[:, 2], esperado["z"], atol=tol)          # sin paredes: todo en la oclusal
    assert esperado["incluye_paredes"] is False

    masa = vtk_mass(parche_de_apoyo(escaneo, mascara))
    assert masa == pytest.approx(esperado["area_mm2"], abs=esperado["tolerancia_area_mm2"])


def test_curva_puede_incluir_encia():
    """Verifica R-016: en modo curva manda el usuario; una curva sobre la encía la incluye."""
    from nucleo_dental.apoyo import region_desde_curva

    escaneo = leer_stl(CASOS_DORADOS / "escaneo_arcada_recta.stl")
    curva = [[-2, -6, 0], [2, -6, 0], [2, -4, 0], [-2, -4, 0]]
    mascara = region_desde_curva(escaneo, curva, [0, -5, 0])
    puntos = numpy_support.vtk_to_numpy(escaneo.GetPoints().GetData()).astype(float)
    assert mascara.any()
    assert np.allclose(puntos[mascara, 2], 0.0)


def test_curva_con_menos_de_3_puntos_es_error():
    """Verifica R-016: una curva cerrada necesita al menos 3 puntos."""
    from nucleo_dental.apoyo import region_desde_curva

    escaneo = leer_stl(CASOS_DORADOS / "escaneo_arcada_recta.stl")
    with pytest.raises(ValueError, match="3 puntos"):
        region_desde_curva(escaneo, [[0, 0, 8], [1, 0, 8]], [0.5, 0, 8])


@pytest.mark.parametrize("sistema, factor", [("LPS", (1, 1, 1)), ("RAS", (-1, -1, 1))])
def test_leer_curva_de_slicer_respeta_el_sistema(tmp_path, sistema, factor):
    """Verifica R-016 y R-002: los puntos de un .mrk.json de Slicer se leen en LPS según el sistema que declara."""
    from nucleo_dental.apoyo import leer_puntos_slicer

    puntos = [[6, -2, 8], [10, -2, 8], [10, 2, 8]]
    archivo = tmp_path / "curva.mrk.json"
    archivo.write_text(json.dumps({"markups": [{
        "type": "ClosedCurve", "coordinateSystem": sistema,
        "controlPoints": [{"label": f"C-{i}", "position": [p * f for p, f in zip(pt, factor)]}
                          for i, pt in enumerate(puntos)]}]}), encoding="utf-8")
    np.testing.assert_allclose(leer_puntos_slicer(archivo), puntos)


def test_leer_curva_sin_sistema_declarado_es_error(tmp_path):
    """Verifica R-016 y R-002: sin coordinateSystem no se adivina RAS o LPS; es error."""
    from nucleo_dental.apoyo import leer_puntos_slicer

    archivo = tmp_path / "curva.mrk.json"
    archivo.write_text(json.dumps({"markups": [{"type": "ClosedCurve",
                                                "controlPoints": [{"position": [0, 0, 0]}]}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="coordinateSystem"):
        leer_puntos_slicer(archivo)


def _implante_eje_x0(tmp_path):
    """STL de un implante vertical con eje en x = 0, y = 0 (bajo la encía de la arcada recta)."""
    import vtk

    from nucleo_dental.implante import Implante

    ruta = tmp_path / "implante.stl"
    escritor = vtk.vtkSTLWriter()
    escritor.SetInputData(Implante(4.1, 10.0, [0, 0, -12.0], [0, 0, 1]).como_malla())
    escritor.SetFileName(str(ruta))
    escritor.SetFileTypeToBinary()
    escritor.Write()
    return ruta


def test_comando_apoyo_automatico(tmp_path):
    """Verifica R-016 y R-007: `apoyo` automático guarda el parche, lo traza e informa la región (caso 012 con R = 20)."""
    from test_medir import _sha256, ejecutar

    salida_stl = tmp_path / "apoyo.stl"
    codigo, salida, stderr = ejecutar(
        "apoyo", "--escaneo", CASOS_DORADOS / "escaneo_arcada_recta.stl",
        "--dientes", CASOS_DORADOS / "dientes_arcada_recta.stl",
        "--implante-stl", _implante_eje_x0(tmp_path), "--apice-hacia", "abajo",
        "--radio", "20", "--salida", salida_stl)
    assert codigo == 0, stderr
    a = salida["apoyo"]
    assert a["modo"] == "automatico"
    assert a["altura_minima_sobre_encia_mm"] == pytest.approx(1.0, abs=0.05)
    assert a["area_mm2"] > 0
    assert salida["parametros"]["radio_mm"] == 20.0
    assert salida["parche"]["sha256"] == _sha256(salida_stl)


def test_comando_apoyo_con_curva_de_slicer(tmp_path):
    """Verifica R-016: `apoyo --curva` lee un .mrk.json (RAS) y delimita los 16 mm² del caso 014."""
    from test_medir import ejecutar

    caso = json.loads((CASO_014 / "caso.json").read_text(encoding="utf-8"))
    curva = tmp_path / "curva.mrk.json"
    curva.write_text(json.dumps({"markups": [{"type": "ClosedCurve", "coordinateSystem": "RAS", "controlPoints": [
        {"position": [-p[0], -p[1], p[2]]} for p in caso["curva"]]}]}), encoding="utf-8")
    codigo, salida, stderr = ejecutar(
        "apoyo", "--escaneo", CASOS_DORADOS / "escaneo_arcada_recta.stl", "--curva", curva,
        "--punto-interior", "8,0,8", "--salida", tmp_path / "apoyo.stl")
    assert codigo == 0, stderr
    assert salida["apoyo"]["modo"] == "curva"
    assert salida["apoyo"]["area_mm2"] == pytest.approx(16.0, abs=0.5)


@pytest.mark.parametrize("extra, texto",
                         [([], "--curva"),
                          (["--dientes", "X"], "--implante-stl"),
                          (["--curva", "no_existe.mrk.json"], "No existe")],
                         ids=["sin_modo", "automatico_sin_implante", "curva_inexistente"])
def test_comando_apoyo_errores(extra, texto):
    """Verifica R-008 y R-016: errores de entrada de `apoyo` devuelven 1 con mensaje claro."""
    from test_medir import ejecutar

    extra = [str(CASOS_DORADOS / "dientes_arcada_recta.stl") if e == "X" else e for e in extra]
    codigo, salida, stderr = ejecutar("apoyo", "--escaneo", CASOS_DORADOS / "escaneo_arcada_recta.stl", *extra)
    assert codigo == 1
    assert salida is None
    assert texto in stderr


def vtk_mass(parche):
    import vtk

    masa = vtk.vtkMassProperties()
    masa.SetInputData(parche)
    masa.Update()
    return masa.GetSurfaceArea()


def test_radio_por_defecto_es_24_mm():
    """Verifica R-016: el radio por defecto del modo automático es 24 mm (decisión clínica 2026-10-08)."""
    from nucleo_dental.apoyo import RADIO_APOYO_POR_DEFECTO_MM

    assert RADIO_APOYO_POR_DEFECTO_MM == 24.0


def test_radio_y_margen_configurables():
    """Verifica R-016: con R = 28 entra el diente 3; con margen de encía 2 mm el apoyo empieza en z = 2."""
    caso, _, escaneo, dientes = _cargar_012()
    puntos = numpy_support.vtk_to_numpy(escaneo.GetPoints().GetData()).astype(float)
    amplia = region_automatica(escaneo, dientes, caso["eje"]["punto"], caso["eje"]["direccion"],
                               radio_mm=28.0, margen_encia_mm=2.0)
    sel = puntos[amplia["mascara"]]
    assert ((sel[:, 0] > 21.1) & np.isclose(sel[:, 2], 8.0)).any()
    assert sel[:, 2].min() == pytest.approx(2.0, abs=0.05)
