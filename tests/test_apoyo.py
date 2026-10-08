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


def test_radio_y_margen_configurables():
    """Verifica R-016: con R = 28 entra el diente 3; con margen de encía 2 mm el apoyo empieza en z = 2."""
    caso, _, escaneo, dientes = _cargar_012()
    puntos = numpy_support.vtk_to_numpy(escaneo.GetPoints().GetData()).astype(float)
    amplia = region_automatica(escaneo, dientes, caso["eje"]["punto"], caso["eje"]["direccion"],
                               radio_mm=28.0, margen_encia_mm=2.0)
    sel = puntos[amplia["mascara"]]
    assert ((sel[:, 0] > 21.1) & np.isclose(sel[:, 2], 8.0)).any()
    assert sel[:, 2].min() == pytest.approx(2.0, abs=0.05)
