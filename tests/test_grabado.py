"""Texto grabado o en relieve sobre la guía (R-026).

Escena plana: carcasa mucosoportada de 3 mm sobre una encía plana en z = 0,
así que la cara externa de la guía está en z = 3. Escena curva: la misma
carcasa sobre una encía cilíndrica. Reglas de Francisco (2026-10-13): trazo
mínimo 0,5 mm y al menos 1 mm de resina bajo el grabado. El texto sigue la
superficie: la profundidad se mide desde la cara externa en cada punto.
"""

import numpy as np
import pytest
import vtk
from vtk.util import numpy_support

from nucleo_dental.ensamblaje import guia_quirurgica
from nucleo_dental.grabado import texto_como_malla
from nucleo_dental.implante import Implante
from nucleo_dental.kits import ONEGUIDE


def _encia_plana():
    cubo = vtk.vtkCubeSource()
    cubo.SetBounds(-15, 20, -12, 10, -20, 0)
    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(cubo.GetOutputPort())
    limpio = vtk.vtkCleanPolyData()
    limpio.SetInputConnection(tri.GetOutputPort())
    sub = vtk.vtkLinearSubdivisionFilter()
    sub.SetInputConnection(limpio.GetOutputPort())
    sub.SetNumberOfSubdivisions(5)
    sub.Update()
    return sub.GetOutput()


def _encia_cilindrica(radio, largo=30.0, paso=0.5):
    """Cilindro cerrado de eje y, con la cima en z = 0, en una grilla regular de ~`paso` mm."""
    n, m = int(np.ceil(2 * np.pi * radio / paso)), int(np.ceil(largo / paso)) + 1
    angulo, y = np.meshgrid(2 * np.pi * np.arange(n) / n, np.linspace(-largo / 2, largo / 2, m))
    lado = np.column_stack([radio * np.sin(angulo).ravel(), y.ravel(), radio * np.cos(angulo).ravel() - radio])
    puntos = np.vstack([lado, [[0, -largo / 2, -radio], [0, largo / 2, -radio]]])
    c0, c1 = len(lado), len(lado) + 1
    i, j = (a.ravel() for a in np.meshgrid(np.arange(m - 1), np.arange(n), indexing="ij"))
    j1 = (j + 1) % n
    a, b, c, d = i * n + j, i * n + j1, (i + 1) * n + j, (i + 1) * n + j1
    k = np.arange(n)
    k1 = (k + 1) % n
    triangulos = np.vstack([np.column_stack([a, c, b]), np.column_stack([b, c, d]),
                            np.column_stack([np.full(n, c0), k, k1]),
                            np.column_stack([np.full(n, c1), (m - 1) * n + k1, (m - 1) * n + k])])
    malla = vtk.vtkPolyData()
    vtk_puntos = vtk.vtkPoints()
    vtk_puntos.SetData(numpy_support.numpy_to_vtk(puntos, deep=True))
    malla.SetPoints(vtk_puntos)
    celdas = vtk.vtkCellArray()
    celdas.SetData(numpy_support.numpy_to_vtkIdTypeArray(np.arange(0, 3 * len(triangulos) + 1, 3, dtype=np.int64), deep=True),
                   numpy_support.numpy_to_vtkIdTypeArray(triangulos.astype(np.int64).ravel(), deep=True))
    malla.SetPolys(celdas)
    return malla


ENCIA = _encia_plana()


def _guia(caja, modo="grabado", texto="C12 OG", encia=ENCIA, mascara=None):
    z = numpy_support.vtk_to_numpy(encia.GetPoints().GetData())[:, 2]
    return guia_quirurgica(encia, None, Implante(4.1, 10, [0, 0, -12], [0, 0, 1]), ONEGUIDE, "impresa",
                           z > -0.01 if mascara is None else mascara, [0, 0, 1], tipo_soporte="mucosoportada",
                           textos=[{"texto": texto, "caja": caja, "modo": modo}])


def _caja(profundidad, largo=26.0, alto=5.5, centro=(2.0, -8.0, 2.6)):
    return {"centro": np.array(centro), "ejes": np.eye(3), "tamano": np.array([largo, alto, profundidad])}


def _dentro_del_trazo(info, superficie, minimo=0.15):
    """Puntos de las letras a más de `minimo` mm del borde del trazo (un vóxel), proyectados a `superficie`."""
    puntos = superficie(info["en_letras"])
    return info["en_letras"][info["evaluar"](puntos) > minimo]


def _primer_corte(guia, desde, hasta):
    arbol = vtk.vtkOBBTree()
    arbol.SetDataSet(guia)
    arbol.BuildLocator()
    cortes = vtk.vtkPoints()
    arbol.IntersectWithLine(desde, hasta, cortes, None)
    return np.array(cortes.GetPoint(0)) if cortes.GetNumberOfPoints() else None


def _techo(guia, x, y):
    corte = _primer_corte(guia, [x, y, 30], [x, y, -1])
    return None if corte is None else corte[2]


@pytest.fixture(scope="module")
def grabado():
    return _guia(_caja(0.8))


def test_grabado_valido_hunde_las_letras(grabado):
    """Verifica R-026: con trazo ≥ 0,5 mm y 2,2 mm de resina debajo, la guía es válida y las letras se hunden 0,8 mm."""
    r = grabado
    assert r["valida"], r["problemas"]
    texto = r["textos"][0]
    assert texto["trazo_mm"] >= 0.5 and texto["resina_bajo_grabado_suficiente"]
    assert texto["fraccion_sobre_la_guia"] == pytest.approx(1.0)
    info = texto_como_malla("C12 OG", _caja(0.8), ENCIA, np.array([0, 0, 1.0]))
    letra = _dentro_del_trazo(info, lambda p: p * [1, 1, 0] + [0, 0, 3.0])[0]
    assert _techo(r["guia"], letra[0], letra[1]) == pytest.approx(3.0 - 0.8, abs=0.05)    # dentro de una letra
    assert _techo(r["guia"], 12.0, 5.0) == pytest.approx(3.0, abs=0.05)                  # fuera de la caja y del anillo


def test_la_posicion_de_la_caja_a_lo_largo_de_la_normal_no_importa():
    """Verifica R-026: la profundidad se mide desde la cara externa: la misma caja más arriba graba igual."""
    r = _guia(_caja(0.8, centro=(2.0, -8.0, 6.0)))
    info = texto_como_malla("C12 OG", _caja(0.8), ENCIA, np.array([0, 0, 1.0]))
    letra = _dentro_del_trazo(info, lambda p: p * [1, 1, 0] + [0, 0, 3.0])[0]
    assert r["valida"], r["problemas"]
    assert _techo(r["guia"], letra[0], letra[1]) == pytest.approx(2.2, abs=0.05)


def test_trazo_menor_al_minimo_no_es_valido():
    """Verifica R-026: en una caja chica el trazo queda bajo 0,5 mm: no válida, con el alto de letra necesario."""
    r = _guia(_caja(0.8, largo=10.0, alto=2.0))
    assert not r["valida"]
    assert r["textos"][0]["trazo_mm"] < 0.5
    assert any("trazo" in p and "agranda la caja" in p for p in r["problemas"])


def test_grabado_que_deja_menos_de_1_mm_no_es_valido():
    """Verifica R-026: grabar 2,4 mm en una carcasa de 3 mm deja 0,6 mm de resina: no válida."""
    r = _guia(_caja(2.4))
    assert not r["valida"]
    assert r["textos"][0]["resina_bajo_grabado_suficiente"] is False
    assert any("menos de 1.0 mm de resina" in p for p in r["problemas"])


def test_relieve_suma_material():
    """Verifica R-026: en relieve las letras sobresalen 1,2 mm de la cara externa (z = 3 → 4,2)."""
    r = _guia(_caja(1.2), modo="relieve")
    assert r["valida"], r["problemas"]
    info = texto_como_malla("C12 OG", _caja(1.2), ENCIA, np.array([0, 0, 1.0]))
    letra = _dentro_del_trazo(info, lambda p: p * [1, 1, 0] + [0, 0, 3.0])[0]
    assert _techo(r["guia"], letra[0], letra[1]) == pytest.approx(4.2, abs=0.05)


def test_el_texto_se_lee_derecho_desde_afuera():
    """Verifica R-026: mirando la guía desde afuera (+z), la "L" tiene el trazo vertical a la izquierda y el pie hacia la derecha.

    La caja está en y = +6, vestibular respecto del centro del escaneo (y ≈ −1): el arriba del texto es +y.
    """
    info = texto_como_malla("L", _caja(0.8, centro=(2.0, 6.0, 2.6)), ENCIA, np.array([0, 0, 1.0]))
    puntos = numpy_support.vtk_to_numpy(info["malla"].GetPoints().GetData())
    np.testing.assert_allclose(info["afuera"], [0, 0, 1])
    y_medio = (puntos[:, 1].min() + puntos[:, 1].max()) / 2
    arriba, abajo = puntos[puntos[:, 1] > y_medio], puntos[puntos[:, 1] < y_medio]
    assert arriba[:, 0].max() < abajo[:, 0].max()          # el pie de la L se extiende a la derecha (+x)


def test_en_cara_oclusal_el_arriba_va_hacia_vestibular():
    """Verifica R-026: en una cara oclusal el arriba del texto apunta a vestibular (lejos del centro del escaneo).

    Con la caja en y = −8 (vestibular = −y), leído desde +z: arriba = −y y el renglón avanza hacia −x,
    así que el pie de la "L" (abajo, +y) se extiende hacia −x más allá del trazo vertical.
    """
    info = texto_como_malla("L", _caja(0.8), ENCIA, np.array([0, 0, 1.0]))
    puntos = numpy_support.vtk_to_numpy(info["malla"].GetPoints().GetData())
    y_medio = (puntos[:, 1].min() + puntos[:, 1].max()) / 2
    arriba, abajo = puntos[puntos[:, 1] < y_medio], puntos[puntos[:, 1] > y_medio]
    assert abajo[:, 0].min() < arriba[:, 0].min()


@pytest.fixture(scope="module")
def curva():
    encia = _encia_cilindrica(30.0)
    puntos = numpy_support.vtk_to_numpy(encia.GetPoints().GetData())
    mascara = (puntos[:, 2] > -12) & (np.abs(puntos[:, 1]) < 12)
    caja = _caja(0.8, largo=30.0, alto=4.0, centro=(0.0, -9.5, 2.6))     # lejos del anillo (radio 5,65)
    r = _guia(caja, encia=encia, mascara=mascara)
    return r, texto_como_malla("C12 OG", caja, encia, np.array([0, 0, 1.0]))


def test_el_grabado_sigue_una_superficie_curva(curva):
    """Verifica R-026: sobre una encía cilíndrica de radio 30 las letras se hunden 0,8 mm en todo el renglón, medido en dirección radial."""
    r, info = curva
    assert r["valida"], r["problemas"]
    centro_cilindro = np.array([0.0, 0.0, -30.0])
    radio_externo = 30.0 + 3.0                               # cara externa de la carcasa
    def en_la_cara(puntos):                                  # proyección radial a la cara externa (radio 33)
        radial = puntos - centro_cilindro
        radial[:, 1] = 0.0
        return centro_cilindro + radial / np.linalg.norm(radial, axis=1, keepdims=True) * radio_externo + puntos * [0, 1, 0]

    muestras = _dentro_del_trazo(info, en_la_cara)
    assert np.ptp(muestras[:, 0]) > 15                       # cubren casi todo el renglón (unos 20 mm)
    for letra in muestras[::10]:
        radial = letra - centro_cilindro
        radial[1] = 0.0
        radial /= np.linalg.norm(radial)
        base = centro_cilindro + np.array([0, letra[1], 0])
        fondo = _primer_corte(r["guia"], base + 40 * radial, base)
        assert radio_externo - np.linalg.norm((fondo - base)[[0, 2]]) == pytest.approx(0.8, abs=0.15)


def test_texto_que_sale_de_la_guia_es_aviso():
    """Verifica R-026: si parte de las letras cae fuera de la carcasa, la guía sigue válida pero hay un aviso."""
    encia = _encia_cilindrica(15.0)
    puntos = numpy_support.vtk_to_numpy(encia.GetPoints().GetData())
    mascara = (puntos[:, 2] > -6) & (np.abs(puntos[:, 1]) < 12)            # la carcasa llega a x = ±13,9
    r = _guia(_caja(0.8, largo=30.0, alto=6.0, centro=(0.0, -8.0, 2.6)), encia=encia, mascara=mascara)
    assert r["textos"][0]["fraccion_sobre_la_guia"] < 0.95
    assert any("sale de la guía" in a for a in r["avisos"])


def test_texto_y_caja_van_juntos(capsys):
    """Verifica R-026: `guia --texto` sin `--caja-texto` es error de entrada (código 1)."""
    from nucleo_dental.cli import main

    codigo = main(["guia", "--escaneo", "e.stl", "--dientes", "d.stl", "--implante-stl", "i.stl", "--apice-hacia",
                   "abajo", "--curva", "c.mrk.json", "--plano-oclusal", "p.mrk.json", "--texto", "C12",
                   "--salida", "g.stl"])
    assert codigo == 1
    assert "--caja-texto" in capsys.readouterr().err
