"""Ensamblaje de la guía quirúrgica (R-021).

Caso dorado 017: diente con la corona más ancha que el cuello y una guía que
rellena el hueco bajo la corona (ver su esperado.json para el origen de cada valor).
"""

import json

import numpy as np
import pytest
import vtk
from vtk.util import numpy_support

from dorados import CASOS_DORADOS
from nucleo_dental.cli import main
from nucleo_dental.ensamblaje import ensamblar, eje_desde_plano_oclusal, guia_quirurgica
from nucleo_dental.implante import Implante
from nucleo_dental.kits import ONEGUIDE
from nucleo_dental.medicion import leer_stl

CASO_017 = CASOS_DORADOS / "guia" / "caso_017_undercut"


def _cortes(malla, desde, hasta) -> np.ndarray:
    """Puntos donde el segmento atraviesa la malla, ordenados desde `desde`."""
    arbol = vtk.vtkOBBTree()
    arbol.SetDataSet(malla)
    arbol.BuildLocator()
    puntos = vtk.vtkPoints()
    arbol.IntersectWithLine(desde, hasta, puntos, None)
    return np.array([puntos.GetPoint(i) for i in range(puntos.GetNumberOfPoints())]).reshape(-1, 3)


@pytest.fixture(scope="module")
def caso_017():
    caso = json.loads((CASO_017 / "caso.json").read_text(encoding="utf-8"))
    esperado = json.loads((CASO_017 / "esperado.json").read_text(encoding="utf-8"))
    diente = leer_stl(CASO_017 / caso["escaneo"])
    guia = leer_stl(CASO_017 / caso["guia_sin_recortar"])
    r = ensamblar([guia], diente, caso["eje_insercion"], caso["tolerancia_ajuste_mm"])
    return caso, esperado, diente, guia, r


def test_caso_dorado_017_antes_del_recorte_no_se_inserta(caso_017):
    """Verifica R-021: la guía que rellena el hueco bajo la corona no puede insertarse (caso 017)."""
    from nucleo_dental.geometria.undercut_core import verificar_insercion

    caso, e, diente, guia, _ = caso_017
    assert verificar_insercion(guia, diente, eje=caso["eje_insercion"], spacing=0.2)["ok"] is e["insercion_antes_del_recorte"]


def test_caso_dorado_017_cara_interna_y_techo(caso_017):
    """Verifica R-021: tras recortar y aplicar 0,2 mm, la cara interna queda en |x| = 4,2 y el techo interno en z = 10,2."""
    _, e, _, _, r = caso_017
    tol = e["tolerancia_mm"]
    for z in (8.0, 5.0):                    # junto a la corona y bajo ella: la holgura se mantiene en todo el recorrido
        for signo in (1, -1):
            primero = _cortes(r["guia"], [0, 0, z], [signo * 20, 0, z])[0]
            assert abs(primero[0]) == pytest.approx(e["cara_interna_pared_x_abs"], abs=tol)
    techo = _cortes(r["guia"], [0, 0, 0], [0, 0, 30])[0]
    assert techo[2] == pytest.approx(e["techo_interno_z"], abs=tol)


def test_caso_dorado_017_se_quita_la_zona_que_choca(caso_017):
    """Verifica R-021: no queda guía en |x| de 3 a 4 bajo la corona; la pared externa sigue llegando a z = 4."""
    _, e, _, _, r = caso_017
    zona, tol = e["zona_que_choca"], e["tolerancia_mm"]
    for x in np.linspace(zona["x_abs_min"] + tol, zona["x_abs_max"] - tol, 5):
        for signo in (1, -1):
            cortes = _cortes(r["guia"], [signo * x, 0, zona["z_min"] - 1], [signo * x, 0, zona["z_max"]])
            assert len(cortes) == 0
    borde = _cortes(r["guia"], [5, 0, 0], [5, 0, 30])[0]
    assert borde[2] == pytest.approx(e["borde_inferior_pared_externa_z"], abs=tol)


def test_caso_dorado_017_la_guia_ensamblada_es_valida(caso_017):
    """Verifica R-021: la guía ensamblada es una sola pieza cerrada y se inserta sin colisión a lo largo del eje."""
    *_, r = caso_017
    assert r["valida"], r["problemas"]
    assert r["metricas"]["piezas"] == 1
    assert r["metricas"]["insercion"]["colision_mm3"] == 0


def test_tolerancia_segun_fabricacion():
    """Verifica R-021: la tolerancia de ajuste depende del método de fabricación (provisional, configurable)."""
    assert ONEGUIDE.tolerancia_ajuste("impresa") == 0.2
    assert ONEGUIDE.tolerancia_ajuste("fresada") == 0.1
    assert "tolerancia_ajuste_mm" in ONEGUIDE.provisionales
    assert ONEGUIDE.con(tolerancia_ajuste_mm={"impresa": 0.3}).tolerancia_ajuste("impresa") == 0.3
    with pytest.raises(ValueError, match="fabricación"):
        ONEGUIDE.tolerancia_ajuste("colada")


def test_tolerancia_mayor_separa_mas(caso_017):
    """Verifica R-021: con tolerancia 0,3 la cara interna se aleja a |x| = 4,3."""
    caso, _, diente, guia, _ = caso_017
    r = ensamblar([guia], diente, caso["eje_insercion"], 0.3)
    assert _cortes(r["guia"], [0, 0, 8], [20, 0, 8])[0][0] == pytest.approx(4.3, abs=0.05)


def test_eje_desde_plano_oclusal():
    """Verifica R-021: el eje de inserción es la normal al plano de 3 puntos, hacia el lado en que se retira la guía."""
    puntos = [[0, 0, 5], [10, 0, 5], [0, 10, 5]]
    np.testing.assert_allclose(eje_desde_plano_oclusal(puntos, [0, 0, 1]), [0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(eje_desde_plano_oclusal(puntos, [0.2, 0, -1]), [0, 0, -1], atol=1e-12)
    inclinado = eje_desde_plano_oclusal([[0, 0, 0], [10, 0, 10], [0, 10, 0]], [0, 0, 1])
    np.testing.assert_allclose(inclinado, [-np.sqrt(0.5), 0, np.sqrt(0.5)], atol=1e-12)


@pytest.mark.parametrize("puntos, referencia, mensaje", [
    ([[0, 0, 0], [1, 1, 1], [2, 2, 2]], [0, 0, 1], "alineados"),
    ([[0, 0, 0], [1, 0, 0]], [0, 0, 1], "3 puntos"),
    ([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [1, 0, 0], "paralela"),
])
def test_plano_oclusal_invalido_es_error(puntos, referencia, mensaje):
    """Verifica R-021: puntos alineados, un número distinto de 3 o una referencia sobre el plano son errores."""
    with pytest.raises(ValueError, match=mensaje):
        eje_desde_plano_oclusal(puntos, referencia)


def test_escaneo_abierto_es_error(caso_017):
    """Verifica R-021: el escaneo debe ser cerrado; abierto no se puede decidir dentro/fuera."""
    *_, guia, _ = caso_017
    with pytest.raises(ValueError, match="cerrada"):
        ensamblar([guia], leer_stl(CASOS_DORADOS / "escaneo_arcada_recta.stl"), [0, 0, 1], 0.2)


def _caja(limites):
    cubo = vtk.vtkCubeSource()
    cubo.SetBounds(*limites)
    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(cubo.GetOutputPort())
    limpio = vtk.vtkCleanPolyData()                 # vtkCubeSource repite vértices por cara: se unen para cerrar
    limpio.SetInputConnection(tri.GetOutputPort())
    limpio.Update()
    return limpio.GetOutput()


@pytest.fixture(scope="module")
def guia_arcada():
    """Arcada recta cerrada: zócalo de encía y un diente a cada lado de la brecha (x de ±5 a ±11)."""
    from nucleo_dental.geometria.malla_guia import booleana

    escaneo = _caja((-15, 15, -8, 8, -3, 0))
    for limites in ((5, 11, -3, 3, -1, 8), (-11, -5, -3, 3, -1, 8)):
        escaneo = booleana(escaneo, _caja(limites), "union", spacing=0.15)
    puntos = numpy_support.vtk_to_numpy(escaneo.GetPoints().GetData())
    # Como una curva del usuario: los dos dientes y la brecha entre ellos.
    limites_usuario = (np.abs(puntos[:, 0]) <= 12) & (puntos[:, 2] > -0.5)
    implante = Implante(4.1, 10.0, [0, 0, -12.0], [0, 0, 1])
    r = guia_quirurgica(escaneo, leer_stl(CASOS_DORADOS / "dientes_arcada_recta.stl"), implante, ONEGUIDE,
                        "impresa", limites_usuario, [0, 0, 1])
    return escaneo, r


def test_guia_completa_es_valida(guia_arcada):
    """Verifica R-021: carcasa de los límites del usuario + anillo, sin puente, dan una sola pieza cerrada que se inserta."""
    _, r = guia_arcada
    assert r["valida"], r["problemas"]
    assert r["metricas"]["piezas"] == 1
    assert r["con_puente"] is False


def test_dentosoportada_levanta_la_carcasa_sobre_la_encia(guia_arcada):
    """Verifica R-020 y R-021: en la brecha la cara interna queda 1 mm sobre la encía (holgura del kit) y sobre los dientes a la tolerancia."""
    escaneo, r = guia_arcada
    encia = _cortes(escaneo, [0, -4, 30], [0, -4, -5])[0][2]
    assert _cortes(r["guia"], [0, -4, 0.0], [0, -4, 30])[0][2] - encia == pytest.approx(1.0, abs=0.05)
    diente = _cortes(escaneo, [8, 0, 30], [8, 0, -5])[0][2]
    assert _cortes(r["guia"], [8, 0, diente], [8, 0, 30])[0][2] - diente == pytest.approx(0.2, abs=0.05)


def test_guia_completa_conserva_anillo_y_orificio(guia_arcada):
    """Verifica R-021: tras ensamblar, la cara superior sigue en z = 8,5 (caso 016) y el orificio mide Ø5,3 y el anillo Ø11,3 (caso 015)."""
    _, r = guia_arcada
    g = r["guia"]
    assert _cortes(g, [0, -4, 30], [0, -4, 0])[0][2] == pytest.approx(8.5, abs=0.05)
    for direccion in ([0, -1, 0], [0, 1, 0]):              # hacia ±y: lejos de la carcasa de los dientes
        cortes = _cortes(g, [0, 0, 7], 20 * np.array(direccion) + [0, 0, 7])
        radios = np.linalg.norm(cortes[:, :2], axis=1)
        assert radios[0] == pytest.approx(5.3 / 2, abs=0.05)
        assert radios[1] == pytest.approx(11.3 / 2, abs=0.08)


def test_guia_completa_no_entra_en_el_escaneo(guia_arcada):
    """Verifica R-021: ningún punto de la guía queda dentro del escaneo (la tolerancia solo resta)."""
    from nucleo_dental.medicion import _dentro

    escaneo, r = guia_arcada
    puntos = numpy_support.vtk_to_numpy(r["guia"].GetPoints().GetData())
    assert not _dentro(escaneo, puntos).any()


def test_holgura_sobre_encia_en_rampa():
    """Verifica R-020 y R-021: 1 mm sobre la encía lejos de los dientes, 0 sobre los dientes y valores intermedios en la transición."""
    from nucleo_dental.ensamblaje import holgura_sobre_encia

    escaneo = leer_stl(CASOS_DORADOS / "escaneo_arcada_recta.stl")
    puntos = numpy_support.vtk_to_numpy(escaneo.GetPoints().GetData())
    h = holgura_sobre_encia(escaneo, leer_stl(CASOS_DORADOS / "dientes_arcada_recta.stl"), 1.0)
    lejos = (np.abs(puntos[:, 0]) < 3) & (puntos[:, 2] == 0)                          # encía de la brecha
    oclusal = (puntos[:, 0] > 6) & (puntos[:, 0] < 10) & (np.abs(puntos[:, 1]) < 2) & (puntos[:, 2] == 8)
    assert np.allclose(h[lejos], 1.0) and np.allclose(h[oclusal], 0.0)
    assert ((h > 0.05) & (h < 0.95)).any()
    assert h.min() >= 0 and h.max() <= 1.0


def test_parche_a_solido_con_desfase_por_vertice():
    """Verifica R-021: el desfase puede variar por vértice (array del parche): cada punto sube lo suyo."""
    from nucleo_dental.geometria.malla_guia import parche_a_solido

    plano = vtk.vtkPlaneSource()
    plano.SetOrigin(0, 0, 0)
    plano.SetPoint1(10, 0, 0)
    plano.SetPoint2(0, 10, 0)
    plano.SetResolution(10, 10)
    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(plano.GetOutputPort())
    tri.Update()
    parche = tri.GetOutput()
    x = numpy_support.vtk_to_numpy(parche.GetPoints().GetData())[:, 0]
    arreglo = numpy_support.numpy_to_vtk(np.where(x < 5, 0.0, 1.0), deep=True)
    arreglo.SetName("Holgura")
    parche.GetPointData().AddArray(arreglo)
    solido = parche_a_solido(parche, 3.0, desfase="Holgura")
    assert _cortes(solido, [2, 5, -5], [2, 5, 10])[0][2] == pytest.approx(0.0, abs=1e-6)
    assert _cortes(solido, [8, 5, -5], [8, 5, 10])[0][2] == pytest.approx(1.0, abs=1e-6)
    assert _cortes(solido, [8, 5, 10], [8, 5, -5])[0][2] == pytest.approx(4.0, abs=1e-6)


def test_astillas_sueltas_se_descartan():
    """Verifica R-021: una pieza suelta menor que el volumen mínimo se descarta y se informa; una mayor queda y la guía no es válida."""
    from nucleo_dental.geometria.malla_guia import booleana

    diente = _caja((-3, 3, -3, 3, -5, 0))
    base = _caja((-3, 3, -3, 3, 0.2, 3.2))
    con_astilla = ensamblar([base, _caja((5, 5.5, 0, 0.5, 1, 1.5))], diente, [0, 0, 1], 0.2)
    assert con_astilla["valida"], con_astilla["problemas"]
    assert con_astilla["metricas"]["astillas_descartadas"] == 1
    assert con_astilla["metricas"]["volumen_astillas_mm3"] == pytest.approx(0.125, abs=0.05)
    con_pieza = ensamblar([base, _caja((5, 7, 0, 2, 1, 3))], diente, [0, 0, 1], 0.2)
    assert not con_pieza["valida"] and con_pieza["metricas"]["piezas"] == 2


def test_comando_guia_faltan_parametros(capsys):
    """Verifica R-021: `guia` exige la curva del usuario y el plano oclusal; sin ellos es error de entrada (código 1)."""
    codigo = main(["guia", "--escaneo", "e.stl", "--dientes", "d.stl", "--implante-stl", "i.stl",
                   "--apice-hacia", "abajo", "--salida", "g.stl"])
    assert codigo == 1
    error = capsys.readouterr().err
    assert "--curva" in error and "--plano-oclusal" in error
