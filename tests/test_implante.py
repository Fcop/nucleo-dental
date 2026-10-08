"""Modelo del implante (R-003).

Los valores esperados fueron calculados a mano por Francisco sobre el
implante de referencia: ápice (0, 0, 0), eje (0, 0, 1), diámetro 4,1 y
largo 10 (radio 2,05), salvo donde se indica otro implante.
"""

import numpy as np
import pytest

from nucleo_dental.implante import Implante


def implante_referencia() -> Implante:
    return Implante(diametro=4.1, largo=10.0, apice=[0, 0, 0], eje=[0, 0, 1])


def test_plataforma_esta_a_un_largo_del_apice_segun_el_eje():
    """Verifica R-003: la plataforma es ápice + largo × eje unitario."""
    imp = Implante(diametro=4.1, largo=10.0, apice=[0, 0, 4.0], eje=[0, 0, 1])
    np.testing.assert_allclose(imp.plataforma, [0, 0, 14.0])


@pytest.mark.parametrize(
    "punto, esperado",
    [
        ([5, 0, 5], 2.95),     # caso 1: costado
        ([0, 0, -3], 3.0),     # caso 2: tapa, bajo el ápice
        ([1, 0, 12], 2.0),     # caso 3: tapa, sobre la plataforma
        ([5.05, 0, 14], 5.0),  # caso 4: esquina, triángulo 3-4-5
    ],
    ids=["costado", "bajo_apice", "sobre_plataforma", "esquina"],
)
def test_distancia_a_puntos_fuera(punto, esperado):
    """Verifica R-003: distancia exacta al cilindro sólido en las tres zonas exteriores."""
    d = implante_referencia().distancia_a_puntos([punto])
    assert d.shape == (1,)
    assert d[0] == pytest.approx(esperado, abs=1e-9)


def test_distancia_a_puntos_es_vectorizada():
    """Verifica R-003: varios puntos a la vez, un resultado por punto, en orden."""
    d = implante_referencia().distancia_a_puntos([[5, 0, 5], [0, 0, -3], [5.05, 0, 14]])
    np.testing.assert_allclose(d, [2.95, 3.0, 5.0], atol=1e-9)


@pytest.mark.parametrize(
    "punto",
    [
        [1, 1, 5],     # caso 5: dentro, a 1,41 del eje
        [2.05, 0, 5],  # caso 6: justo sobre la pared
    ],
    ids=["dentro", "sobre_pared"],
)
def test_punto_dentro_o_en_superficie_esta_contenido_y_a_distancia_cero(punto):
    """Verifica R-003: un punto dentro o sobre la superficie está contenido y su distancia es 0."""
    imp = implante_referencia()
    assert imp.contiene([punto]).tolist() == [True]
    assert imp.distancia_a_puntos([punto])[0] == pytest.approx(0.0, abs=1e-9)


def test_contiene_es_vectorizado():
    """Verifica R-003: contiene responde por cada punto, en orden."""
    resultado = implante_referencia().contiene([[1, 1, 5], [5, 0, 5], [0, 0, -3], [2.05, 0, 5]])
    assert resultado.tolist() == [True, False, False, True]


def test_implante_acostado_respeta_el_eje():
    """Verifica R-003: con eje (1, 0, 0) el implante se extiende hacia +X desde el ápice (caso 7)."""
    imp = Implante(diametro=4.1, largo=10.0, apice=[10, 0, 0], eje=[1, 0, 0])
    np.testing.assert_allclose(imp.plataforma, [20, 0, 0])
    assert imp.distancia_a_puntos([[5, 0, 0]])[0] == pytest.approx(5.0, abs=1e-9)
    assert imp.contiene([[5, 0, 0]]).tolist() == [False]


def test_eje_invertido():
    """Verifica R-003: con eje (0, 0, -1) el implante baja desde el ápice (caso 8).

    Si el código ignorara el eje, el punto (0, 0, -5) quedaría a 5 mm bajo
    el ápice; lo correcto es que esté dentro.
    """
    imp = Implante(diametro=4.1, largo=10.0, apice=[0, 0, 0], eje=[0, 0, -1])
    np.testing.assert_allclose(imp.plataforma, [0, 0, -10])
    assert imp.contiene([[0, 0, -5]]).tolist() == [True]
    assert imp.distancia_a_puntos([[0, 0, -5]])[0] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize(
    "parametros, texto",
    [
        (dict(diametro=0, largo=10, apice=[0, 0, 0], eje=[0, 0, 1]), "diámetro"),
        (dict(diametro=-4.1, largo=10, apice=[0, 0, 0], eje=[0, 0, 1]), "diámetro"),
        (dict(diametro=4.1, largo=0, apice=[0, 0, 0], eje=[0, 0, 1]), "largo"),
        (dict(diametro=4.1, largo=-10, apice=[0, 0, 0], eje=[0, 0, 1]), "largo"),
        (dict(diametro=4.1, largo=10, apice=[0, 0, 0], eje=[0, 0, 0]), "eje"),
        (dict(diametro=4.1, largo=10, apice=[0, 0], eje=[0, 0, 1]), "ápice"),
        (dict(diametro=4.1, largo=10, apice=[0, 0, 0], eje=[0, 1]), "eje"),
        (dict(diametro=float("nan"), largo=10, apice=[0, 0, 0], eje=[0, 0, 1]), "diámetro"),
        (dict(diametro=4.1, largo=10, apice=[0, float("inf"), 0], eje=[0, 0, 1]), "ápice"),
    ],
    ids=["diametro_cero", "diametro_negativo", "largo_cero", "largo_negativo", "eje_nulo",
         "apice_2d", "eje_2d", "diametro_nan", "apice_infinito"],
)
def test_parametros_invalidos_se_rechazan_con_mensaje_claro(parametros, texto):
    """Verifica R-003: entradas inválidas se rechazan con un error en español que nombra el parámetro."""
    with pytest.raises(ValueError, match=texto):
        Implante(**parametros)


def _aristas_abiertas(pd) -> int:
    import vtk

    bordes = vtk.vtkFeatureEdges()
    bordes.SetInputData(pd)
    bordes.BoundaryEdgesOn()
    bordes.NonManifoldEdgesOn()
    bordes.FeatureEdgesOff()
    bordes.ManifoldEdgesOff()
    bordes.Update()
    return bordes.GetOutput().GetNumberOfCells()


def _puntos_y_triangulos(pd):
    from vtk.util import numpy_support

    puntos = numpy_support.vtk_to_numpy(pd.GetPoints().GetData())
    tri = numpy_support.vtk_to_numpy(pd.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    return puntos, tri


def test_como_malla_es_cerrada_y_coincide_con_el_cilindro():
    """Verifica R-003: la malla es cerrada, ocupa x, y en [-2,05; 2,05] y z en [0, 10], y sus vértices están sobre el cilindro."""
    imp = implante_referencia()
    pd = imp.como_malla()

    assert pd.GetNumberOfPolys() > 0
    assert _aristas_abiertas(pd) == 0
    np.testing.assert_allclose(pd.GetBounds(), [-2.05, 2.05, -2.05, 2.05, 0, 10], atol=1e-9)

    puntos, _ = _puntos_y_triangulos(pd)
    np.testing.assert_allclose(imp.distancia_a_puntos(puntos), 0.0, atol=1e-9)


def test_como_malla_tiene_normales_hacia_afuera():
    """Verifica R-003: la malla está orientada hacia afuera (volumen con signo positivo)."""
    puntos, tri = _puntos_y_triangulos(implante_referencia().como_malla())
    v0, v1, v2 = puntos[tri[:, 0]], puntos[tri[:, 1]], puntos[tri[:, 2]]
    volumen = np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0
    assert volumen > 0


@pytest.mark.parametrize("eje", [[0, 0, -1], [1, 1, 1], [0, 1, 0], [0, -1, 0]],
                         ids=["abajo", "diagonal", "y_positivo", "y_negativo"])
def test_como_malla_sigue_el_eje(eje):
    """Verifica R-003: con cualquier eje, la malla queda sobre el cilindro y contiene ápice y plataforma en sus tapas."""
    imp = Implante(diametro=4.1, largo=10.0, apice=[1, 2, 3], eje=eje)
    pd = imp.como_malla()
    assert _aristas_abiertas(pd) == 0
    puntos, _ = _puntos_y_triangulos(pd)
    np.testing.assert_allclose(imp.distancia_a_puntos(puntos), 0.0, atol=1e-9)
    t = (puntos - imp.apice) @ imp.eje
    assert t.min() == pytest.approx(0.0, abs=1e-9)
    assert t.max() == pytest.approx(10.0, abs=1e-9)


@pytest.mark.parametrize(
    "apice, eje, apice_hacia",
    [([0, 0, 4.0], [0, 0, 1], "abajo"),          # caso 001: mandíbula, ápice inferior
     ([1, 2, 3], [0, 0, -1], "arriba"),          # maxilar: ápice superior, eje hacia abajo
     ([1, 2, 3], [0.3, 0.2, 1.0], "abajo")],     # inclinado ~20°
    ids=["mandibula", "maxilar", "inclinado"],
)
def test_desde_malla_reconstruye_el_implante(apice, eje, apice_hacia):
    """Verifica R-012: desde la malla de un cilindro se recuperan diámetro, largo, ápice y eje."""
    original = Implante(diametro=4.1, largo=10.0, apice=apice, eje=eje)
    reconstruido = Implante.desde_malla(original.como_malla(), apice_hacia=apice_hacia)
    assert reconstruido.diametro == pytest.approx(4.1, abs=1e-6)
    assert reconstruido.largo == pytest.approx(10.0, abs=1e-6)
    np.testing.assert_allclose(reconstruido.apice, original.apice, atol=1e-6)
    np.testing.assert_allclose(reconstruido.eje, original.eje, atol=1e-6)


def test_desde_malla_apice_hacia_decide_el_extremo():
    """Verifica R-003 y R-012: el mismo cilindro con 'arriba' en vez de 'abajo' invierte ápice y plataforma."""
    malla = Implante(diametro=4.1, largo=10.0, apice=[0, 0, 4.0], eje=[0, 0, 1]).como_malla()
    invertido = Implante.desde_malla(malla, apice_hacia="arriba")
    np.testing.assert_allclose(invertido.apice, [0, 0, 14.0], atol=1e-6)
    np.testing.assert_allclose(invertido.eje, [0, 0, -1], atol=1e-6)


def test_desde_malla_rechaza_implante_casi_horizontal():
    """Verifica R-012: a más de 60° de la vertical, 'abajo' o 'arriba' no definen el ápice; se rechaza."""
    malla = Implante(diametro=4.1, largo=10.0, apice=[0, 0, 0], eje=[1, 0, 0.5]).como_malla()  # ~63°
    with pytest.raises(ValueError, match="60"):
        Implante.desde_malla(malla, apice_hacia="abajo")


def test_desde_malla_rechaza_valor_desconocido_de_apice_hacia():
    """Verifica R-012: apice_hacia solo acepta 'abajo' o 'arriba'."""
    malla = implante_referencia().como_malla()
    with pytest.raises(ValueError, match="abajo"):
        Implante.desde_malla(malla, apice_hacia="izquierda")


def test_desde_malla_rechaza_lo_que_no_es_cilindro():
    """Verifica R-012: una malla que no es un cilindro (aquí, un cono) se rechaza."""
    import vtk

    cono = vtk.vtkConeSource()
    cono.SetResolution(64)
    cono.SetHeight(10)
    cono.SetRadius(2)
    cono.SetDirection(0, 0, 1)
    cono.Update()
    with pytest.raises(ValueError, match="cilindro"):
        Implante.desde_malla(cono.GetOutput(), apice_hacia="abajo")


def test_eje_se_normaliza():
    """Verifica R-003: un eje no unitario se normaliza; (0, 0, 2) equivale a (0, 0, 1)."""
    imp = Implante(diametro=4.1, largo=10.0, apice=[0, 0, 0], eje=[0, 0, 2])
    np.testing.assert_allclose(imp.eje, [0, 0, 1])
    np.testing.assert_allclose(imp.plataforma, [0, 0, 10.0])
