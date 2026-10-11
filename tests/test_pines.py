"""Pines de fijación (R-025): lectura desde STL, profundidad en el hueso, seguridad y geometría en la guía."""

import json

import numpy as np
import pytest
import vtk
from vtk.util import numpy_support

from nucleo_dental.ensamblaje import guia_quirurgica
from nucleo_dental.implante import Implante
from nucleo_dental.kits import ONEGUIDE
from nucleo_dental.pines import Pin, evaluar_pin, profundidad_en_hueso, refuerzo_y_agujero

U45 = np.array([1.0, 0.0, 1.0]) / np.sqrt(2)            # de la punta a la cabeza, 45° en el plano x–z


def _caja(limites, subdivisiones=0):
    cubo = vtk.vtkCubeSource()
    cubo.SetBounds(*limites)
    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(cubo.GetOutputPort())
    limpio = vtk.vtkCleanPolyData()
    limpio.SetInputConnection(tri.GetOutputPort())
    limpio.Update()
    malla = limpio.GetOutput()
    if subdivisiones:
        sub = vtk.vtkLinearSubdivisionFilter()
        sub.SetInputData(malla)
        sub.SetNumberOfSubdivisions(subdivisiones)
        sub.Update()
        malla = sub.GetOutput()
    return malla


@pytest.fixture(scope="module")
def escena():
    """Encía (escaneo) con su cara superior en z = 0 y hueso 1 mm más abajo (z = −1)."""
    escaneo = _caja((-15, 20, -10, 10, -20, 0), 5)
    hueso = _caja((-15, 20, -10, 10, -20, -1))
    return escaneo, hueso


def test_pin_desde_stl_reconoce_la_punta_en_el_hueso(escena):
    """Verifica R-025: del STL de un cilindro inclinado 45° sale el pin, con la punta en el extremo dentro del hueso."""
    _, hueso = escena
    original = Pin(1.5, 12.0, [6, 0, -7], U45)
    for malla in (original.como_malla(), Pin(1.5, 12.0, original.cabeza, -U45).como_malla()):   # ambos sentidos
        pin = Pin.desde_malla(malla, hueso)
        np.testing.assert_allclose(pin.punta, [6, 0, -7], atol=1e-6)
        np.testing.assert_allclose(pin.eje, U45, atol=1e-6)
        assert pin.diametro == pytest.approx(1.5, abs=0.01) and pin.largo == pytest.approx(12.0, abs=1e-6)


@pytest.mark.parametrize("punta, eje, mensaje", [([6, 0, 5], U45, "Ningún extremo"),
                                                 ([6, 0, -15.0], [0, 0, 1], "entero dentro")])
def test_pin_fuera_o_entero_en_el_hueso_es_error(escena, punta, eje, mensaje):
    """Verifica R-025: sin ningún extremo en el hueso, o con los dos, no se puede saber cuál es la punta."""
    _, hueso = escena
    with pytest.raises(ValueError, match=mensaje):
        Pin.desde_malla(Pin(1.5, 8.0, punta, eje).como_malla(), hueso)


def test_profundidad_en_hueso(escena):
    """Verifica R-025: de la punta en z = −7 hasta salir del hueso en z = −1 a 45° hay 6·√2 = 8,485 mm."""
    _, hueso = escena
    assert profundidad_en_hueso(Pin(1.5, 12.0, [6, 0, -7], U45), hueso) == pytest.approx(6 * np.sqrt(2), abs=1e-6)


def test_refuerzo_y_agujero(escena):
    """Verifica R-025: entrada en la encía en (13, 0, 0); agujero Ø1,8 (1,5 + 0,3 impresa) y refuerzo Ø5,8 (pared 2)."""
    escaneo, _ = escena
    r = refuerzo_y_agujero(Pin(1.5, 12.0, [6, 0, -7], U45), escaneo, ONEGUIDE, "impresa")
    np.testing.assert_allclose(r["entrada"], [13, 0, 0], atol=1e-6)
    assert r["diametro_agujero_mm"] == pytest.approx(1.8)
    assert r["diametro_refuerzo_mm"] == pytest.approx(5.8)
    assert r["alto_refuerzo_mm"] == ONEGUIDE.alto_refuerzo_pin_mm
    fresada = refuerzo_y_agujero(Pin(1.5, 12.0, [6, 0, -7], U45), escaneo, ONEGUIDE, "fresada")
    assert fresada["diametro_agujero_mm"] == pytest.approx(1.6)


def test_seguridad_del_pin_frente_a_un_implante():
    """Verifica R-025: con el margen de 2 mm, el pin lejos del implante es verde y pegado a él es rojo."""
    implante = {"A": Implante(4.1, 10, [0, 0, -12], [0, 0, 1])}
    assert evaluar_pin(Pin(1.5, 12.0, [6, 0, -7], U45), {}, implante)["semaforo"] == "verde"
    assert evaluar_pin(Pin(1.5, 12.0, [3.5, 0, -7], U45), {}, implante)["semaforo"] == "rojo"
    assert evaluar_pin(Pin(1.5, 12.0, [3.5, 0, -7], U45), {}, implante, margen=0.5)["semaforo"] == "verde"


def test_guia_con_pin(escena):
    """Verifica R-025: el refuerzo del pin se suma a la guía y su agujero la atraviesa; la guía sigue siendo una pieza."""
    escaneo, _ = escena
    z = numpy_support.vtk_to_numpy(escaneo.GetPoints().GetData())[:, 2]
    pin = Pin(1.5, 12.0, [6, 0, -7], U45)
    r = guia_quirurgica(escaneo, None, Implante(4.1, 10, [0, 0, -12], [0, 0, 1]), ONEGUIDE, "impresa", z > -0.01,
                        [0, 0, 1], tipo_soporte="mucosoportada", pines=[pin])
    assert r["valida"], r["problemas"]
    assert r["metricas"]["piezas"] == 1
    arbol = vtk.vtkOBBTree()
    arbol.SetDataSet(r["guia"])
    arbol.BuildLocator()
    cortes = vtk.vtkPoints()
    arbol.IntersectWithLine(pin.punta + 7.5 * np.sqrt(2) * U45, pin.cabeza + 10 * U45, cortes, None)   # por el eje
    assert cortes.GetNumberOfPoints() == 0                                                        # el agujero está libre
    assert [p["entre"] for p in r["paredes_entre_orificios"]] == [["orificio 1", "pin 1"]]


def test_comando_guia_pin_sin_hueso_es_error(capsys):
    """Verifica R-025: --pin-stl sin --hueso es error de entrada (código 1)."""
    from nucleo_dental.cli import main

    codigo = main(["guia", "--escaneo", "e.stl", "--dientes", "d.stl", "--implante-stl", "i.stl", "--apice-hacia",
                   "abajo", "--curva", "c.mrk.json", "--plano-oclusal", "p.mrk.json", "--pin-stl", "pin.stl",
                   "--salida", "g.stl"])
    assert codigo == 1
    assert "--hueso" in capsys.readouterr().err


def test_medir_con_pines(tmp_path, capsys, escena):
    """Verifica R-025: `medir --caso` con "pines" informa profundidad y semáforo de cada pin; un pin rojo pone rojo el global."""
    from nucleo_dental.cli import main

    _, hueso = escena
    for nombre, malla in (("hueso.stl", hueso), ("canal.stl", _caja((-15, 20, -1.5, 1.5, -40, -37))),
                          ("pin_lejos.stl", Pin(1.5, 12.0, [6, 0, -7], U45).como_malla()),
                          ("pin_cerca.stl", Pin(1.5, 12.0, [3.5, 0, -7], U45).como_malla())):
        escritor = vtk.vtkSTLWriter()
        escritor.SetInputData(malla)
        escritor.SetFileName(str(tmp_path / nombre))
        escritor.Write()
    for pin, codigo, semaforo in (("pin_lejos.stl", 0, "verde"), ("pin_cerca.stl", 2, "rojo")):
        caso = {"canal": "canal.stl", "margen": 2.0, "hueso": "hueso.stl",
                "implante": {"diametro": 4.1, "largo": 10.0, "apice": [0, 0, -12.0], "eje": [0, 0, 1]},
                "pines": {"p1": {"stl": pin}}}
        ruta = tmp_path / f"caso_{pin}.json"
        ruta.write_text(json.dumps(caso), encoding="utf-8")
        assert main(["medir", "--caso", str(ruta)]) == codigo
        resultado = json.loads(capsys.readouterr().out)["resultado"]
        p1 = resultado["pines"]["p1"]
        assert p1["semaforo"] == semaforo
        assert p1["profundidad_en_hueso_mm"] == pytest.approx(6 * np.sqrt(2), abs=1e-6)
        assert p1["estructuras"]["implante implante"]["margen_mm"] == 2.0
        if semaforo == "rojo":
            assert resultado["semaforo"] == "rojo"
