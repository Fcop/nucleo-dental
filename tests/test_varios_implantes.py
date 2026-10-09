"""Guías y medición con varios implantes (R-023).

Caso dorado 018: dos implantes verticales de Ø4,1 y largo 10, A centrado en
x = 0 (plataforma z = −2) y B en x = 8 (plataforma z = −3), con OneGuide
impresa (ver su esperado.json para el origen de cada valor).
"""

import json

import numpy as np
import pytest
import vtk
from vtk.util import numpy_support

from dorados import CASOS_DORADOS
from nucleo_dental.ensamblaje import guia_quirurgica
from nucleo_dental.implante import Implante
from nucleo_dental.kits import obtener_kit
from nucleo_dental.medicion import distancia_entre_implantes, evaluar_implantes

CASO_018 = CASOS_DORADOS / "guia" / "caso_018_dos_implantes"


def _leer():
    caso = json.loads((CASO_018 / "caso.json").read_text(encoding="utf-8"))
    esperado = json.loads((CASO_018 / "esperado.json").read_text(encoding="utf-8"))
    implantes = {n: Implante(d["diametro"], d["largo"], d["apice"], d["eje"]) for n, d in caso["implantes"].items()}
    return caso, esperado, implantes


def _cortes(malla, desde, hasta) -> np.ndarray:
    arbol = vtk.vtkOBBTree()
    arbol.SetDataSet(malla)
    arbol.BuildLocator()
    puntos = vtk.vtkPoints()
    arbol.IntersectWithLine(desde, hasta, puntos, None)
    return np.array([puntos.GetPoint(i) for i in range(puntos.GetNumberOfPoints())]).reshape(-1, 3)


def test_caso_dorado_018_distancia_y_semaforo():
    """Verifica R-023: entre las superficies de A y B hay 3,9 mm; con el margen de 3 mm el semáforo es verde."""
    _, e, implantes = _leer()
    r = evaluar_implantes(implantes)
    par = r["pares"]["A-B"]
    assert par["distancia_mm"] == pytest.approx(e["distancia_superficies_mm"], abs=e["tolerancia_mm"])
    assert par["semaforo"] == e["semaforo"] == r["semaforo"]
    assert par["margen_mm"] == 3.0


def test_caso_dorado_018_umbral_del_semaforo():
    """Verifica R-023: B pasa a rojo cuando su centro queda en x < 7,1; en 7,1 exacto la distancia es 3,0 (verde)."""
    _, e, implantes = _leer()
    a, b = implantes["A"], implantes["B"]
    umbral = e["umbral_rojo_x_centro_B"]

    def semaforo(x):
        movido = Implante(b.diametro, b.largo, [x, 0, b.apice[2]], b.eje)
        return evaluar_implantes({"A": a, "B": movido})["semaforo"]

    assert semaforo(umbral) == "verde"
    assert semaforo(umbral - 0.01) == "rojo"


def test_colision_entre_implantes():
    """Verifica R-023: dos implantes que se cruzan tienen distancia 0, colisión y semáforo rojo."""
    a = Implante(4.1, 10, [0, 0, -12], [0, 0, 1])
    b = Implante(4.1, 10, [3, 0, -12], [0.3, 0, 1])
    d = distancia_entre_implantes(a, b)
    assert d["colision"] and d["distancia_mm"] == 0.0
    assert evaluar_implantes({"a": a, "b": b})["semaforo"] == "rojo"


def test_distancia_entre_implantes_inclinados():
    """Verifica R-023: implantes cruzados en planos distintos: distancia entre ejes 6 menos los radios = 1,9 mm."""
    a = Implante(4.1, 20, [0, 0, -10], [0, 0, 1])                  # eje z, centro de altura en z = 0
    b = Implante(4.1, 20, [-10, 6, 0], [1, 0, 0])                  # eje x a y = 6, cruza sobre el eje de A
    assert distancia_entre_implantes(a, b)["distancia_mm"] == pytest.approx(6 - 4.1, abs=0.01)


def _encia_plana():
    cubo = vtk.vtkCubeSource()
    cubo.SetBounds(-12, 20, -10, 10, -5, 0)
    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(cubo.GetOutputPort())
    limpio = vtk.vtkCleanPolyData()
    limpio.SetInputConnection(tri.GetOutputPort())
    sub = vtk.vtkLinearSubdivisionFilter()             # vértices densos: el parche de apoyo necesita resolución
    sub.SetInputConnection(limpio.GetOutputPort())
    sub.SetNumberOfSubdivisions(5)
    sub.Update()
    return sub.GetOutput()


@pytest.fixture(scope="module")
def guia_018():
    caso, e, implantes = _leer()
    encia = _encia_plana()
    z = numpy_support.vtk_to_numpy(encia.GetPoints().GetData())[:, 2]
    r = guia_quirurgica(encia, None, [implantes["A"], implantes["B"]], obtener_kit(caso["kit"]), caso["fabricacion"],
                        z > -0.01, caso["eje_insercion"], tipo_soporte=caso["tipo_soporte"])
    return e, r


def test_caso_dorado_018_caras_superiores(guia_018):
    """Verifica R-023: cada anillo tiene su cara superior según su plataforma: A en z = 8,5 y B en z = 7,5."""
    e, r = guia_018
    tol = e["tolerancia_mm"]
    assert r["valida"], r["problemas"]
    assert _cortes(r["guia"], [-4, 0, 30], [-4, 0, 0])[0][2] == pytest.approx(e["cara_superior_A_z"], abs=tol)
    assert _cortes(r["guia"], [12, 0, 30], [12, 0, 0])[0][2] == pytest.approx(e["cara_superior_B_z"], abs=tol)


def test_caso_dorado_018_pared_entre_orificios_y_anillos_fundidos(guia_018):
    """Verifica R-023: a z = 7 la pared entre orificios mide 2,7 mm y los dos anillos forman un solo bloque."""
    e, r = guia_018
    tol = e["tolerancia_mm"]
    cortes = _cortes(r["guia"], [0, 0, 7], [20, 0, 7])[:, 0]
    assert cortes[1] - cortes[0] == pytest.approx(e["pared_entre_orificios_mm"], abs=tol)
    assert r["paredes_entre_orificios"][0]["pared_mm"] == pytest.approx(e["pared_entre_orificios_mm"], abs=tol)
    assert len(cortes) == 4                      # borde orificio A, orificio B (entrada y salida) y borde externo de B
    assert (r["metricas"]["piezas"] == 1) is e["anillos_fundidos"]
    assert r["avisos"] == []


def test_pared_delgada_es_aviso_y_solape_no_es_valido():
    """Verifica R-023: con la pared entre orificios bajo 1 mm hay aviso; si los orificios se solapan la guía no es válida."""
    from nucleo_dental.kits import ONEGUIDE

    encia = _encia_plana()
    z = numpy_support.vtk_to_numpy(encia.GetPoints().GetData())[:, 2]
    a = Implante(4.1, 10, [0, 0, -12], [0, 0, 1])
    for x_b, aviso, valida in ((5.8, True, True), (5.0, False, False)):      # paredes de 0,5 mm y de −0,3 (solape)
        b = Implante(4.1, 10, [x_b, 0, -12], [0, 0, 1])
        r = guia_quirurgica(encia, None, [a, b], ONEGUIDE, "impresa", z > -0.01, [0, 0, 1], tipo_soporte="mucosoportada")
        assert bool(r["avisos"]) is aviso
        assert r["valida"] is valida
        if not valida:
            assert any("se solapan" in p for p in r["problemas"])
