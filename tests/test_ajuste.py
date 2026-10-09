"""Informe de ajuste guía–escaneo antes de imprimir (R-022).

Usa la guía ensamblada del caso dorado 017: su cara interna quedó a 0,2 mm
del diente (cara interna en |x| = 4,2 frente a la corona en 4, y techo en
10,2 sobre la cima en 10, valores de Francisco).
"""

import json

import numpy as np
import pytest
import vtk
from vtk.util import numpy_support

from dorados import CASOS_DORADOS
from nucleo_dental.ensamblaje import ensamblar
from nucleo_dental.geometria.ajuste import analizar_ajuste, histograma_ajuste, mapa_de_ajuste
from nucleo_dental.medicion import leer_stl

CASO_017 = CASOS_DORADOS / "guia" / "caso_017_undercut"


@pytest.fixture(scope="module")
def guia_017():
    caso = json.loads((CASO_017 / "caso.json").read_text(encoding="utf-8"))
    diente = leer_stl(CASO_017 / caso["escaneo"])
    r = ensamblar([leer_stl(CASO_017 / caso["guia_sin_recortar"])], diente, caso["eje_insercion"],
                  caso["tolerancia_ajuste_mm"])
    return r["guia"], diente, caso["tolerancia_ajuste_mm"]


def _bajada(malla, mm: float):
    transformada = vtk.vtkTransform()
    transformada.Translate(0, 0, -mm)
    filtro = vtk.vtkTransformPolyDataFilter()
    filtro.SetInputData(malla)
    filtro.SetTransform(transformada)
    filtro.Update()
    return filtro.GetOutput()


def test_guia_017_asienta_a_la_tolerancia(guia_017):
    """Verifica R-022: la cara de asiento de la guía 017 queda a 0,2 mm del diente, toda dentro del desvío aceptable."""
    guia, diente, tolerancia = guia_017
    a = analizar_ajuste(guia, diente, tolerancia)
    assert a["separacion_mediana_mm"] == pytest.approx(0.2, abs=0.01)
    assert a["sin_interferencia"] and a["pct_interferencia"] == 0
    assert a["pct_ideal"] > 99.0


def test_detecta_la_interferencia(guia_017):
    """Verifica R-022: la misma guía bajada 0,3 mm invade el diente 0,1 mm (0,2 − 0,3) en el techo."""
    guia, diente, tolerancia = guia_017
    a = analizar_ajuste(_bajada(guia, 0.3), diente, tolerancia)
    assert not a["sin_interferencia"]
    assert a["interferencia_max_mm"] == pytest.approx(0.1, abs=0.02)


def test_objetivo_por_vertice(guia_017):
    """Verifica R-022: si el objetivo es 1 mm en todo el escaneo, cada punto que asienta a 0,2 mm queda "apretado" (−0,8)."""
    guia, diente, tolerancia = guia_017
    a = analizar_ajuste(guia, diente, tolerancia, holgura_por_vertice=np.full(diente.GetNumberOfPoints(), 1.0))
    assert np.all(a["_objetivo"] == 1.0) and a["sin_interferencia"]
    zona, separacion = a["_zona"], a["_separacion"]
    a_02 = zona & (np.abs(separacion - 0.2) < 0.02)       # techo y pared junto a la corona
    np.testing.assert_allclose((separacion - a["_objetivo"])[a_02], -0.8, atol=0.02)
    assert a["pct_apretado"] >= 100.0 * a_02.sum() / zona.sum() - 1e-9


def test_mapa_e_histograma(guia_017):
    """Verifica R-022: el mapa lleva separación, objetivo y desvío por punto; el histograma suma 100 %."""
    guia, diente, tolerancia = guia_017
    a = analizar_ajuste(guia, diente, tolerancia)
    mapa = mapa_de_ajuste(guia, a)
    for nombre in ("Separacion_mm", "Objetivo_mm", "Desvio_mm"):
        assert mapa.GetPointData().GetArray(nombre).GetNumberOfTuples() == guia.GetNumberOfPoints()
    desvio = numpy_support.vtk_to_numpy(mapa.GetPointData().GetArray("Desvio_mm"))
    assert np.isnan(desvio).sum() == a["puntos_guia"] - a["puntos_cara_asiento"]
    porcentajes = [float(f.split()[-2]) for f in histograma_ajuste(a).splitlines()[1:]]
    assert sum(porcentajes) == pytest.approx(100.0, abs=0.5)


def test_interferencia_invalida_la_guia(guia_017):
    """Verifica R-022: una guía que invade el escaneo queda no válida, con el motivo en los problemas."""
    from nucleo_dental.ensamblaje import _agregar_ajuste

    guia, diente, tolerancia = guia_017
    resultado = {"guia": _bajada(guia, 0.3), "valida": True, "problemas": []}
    _agregar_ajuste(resultado, diente, tolerancia, None)
    assert resultado["valida"] is False
    assert "invade el escaneo" in resultado["problemas"][0]


def test_la_holgura_rige_para_todas_las_piezas():
    """Verifica R-020 y R-022: sobre una encía inclinada 10°, una pieza de base plana queda a ≥ 1 mm en todo punto, no solo en el eje."""
    from nucleo_dental.geometria.ajuste import separacion_con_signo

    cubo = vtk.vtkCubeSource()
    cubo.SetBounds(-10, 10, -10, 10, -5, 0)
    girar = vtk.vtkTransform()
    girar.RotateY(10)                                     # la cara superior sube hacia -x
    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(cubo.GetOutputPort())
    girado = vtk.vtkTransformPolyDataFilter()
    girado.SetInputConnection(tri.GetOutputPort())
    girado.SetTransform(girar)
    limpio = vtk.vtkCleanPolyData()
    limpio.SetInputConnection(girado.GetOutputPort())
    limpio.Update()
    encia = limpio.GetOutput()
    pieza = vtk.vtkCubeSource()
    pieza.SetBounds(-3, 3, -3, 3, 1, 4)                   # base plana 1 mm sobre la encía solo en el eje
    tri_pieza = vtk.vtkTriangleFilter()
    tri_pieza.SetInputConnection(pieza.GetOutputPort())
    limpia = vtk.vtkCleanPolyData()
    limpia.SetInputConnection(tri_pieza.GetOutputPort())
    limpia.Update()

    sin = ensamblar([limpia.GetOutput()], encia, [0, 0, 1], 0.2)
    con = ensamblar([limpia.GetOutput()], encia, [0, 0, 1], 0.2,
                    holgura_por_vertice=np.full(encia.GetNumberOfPoints(), 1.0))
    assert separacion_con_signo(sin["guia"], encia).min() < 0.8         # sin la regla, el lado alto queda cerca
    assert separacion_con_signo(con["guia"], encia).min() == pytest.approx(1.0, abs=0.03)
