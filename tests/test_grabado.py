"""Texto grabado o en relieve sobre la guía (R-026).

Escena: carcasa mucosoportada de 3 mm sobre una encía plana en z = 0, así que
la cara externa de la guía está en z = 3. Reglas de Francisco (2026-10-13):
trazo mínimo 0,5 mm y al menos 1 mm de resina bajo el grabado.
"""

import numpy as np
import pytest
import vtk
from vtk.util import numpy_support

from nucleo_dental.ensamblaje import guia_quirurgica
from nucleo_dental.grabado import texto_como_malla
from nucleo_dental.implante import Implante
from nucleo_dental.kits import ONEGUIDE


def _encia():
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


ENCIA = _encia()


def _guia(caja, modo="grabado", texto="C12 OG"):
    z = numpy_support.vtk_to_numpy(ENCIA.GetPoints().GetData())[:, 2]
    return guia_quirurgica(ENCIA, None, Implante(4.1, 10, [0, 0, -12], [0, 0, 1]), ONEGUIDE, "impresa", z > -0.01,
                           [0, 0, 1], tipo_soporte="mucosoportada",
                           textos=[{"texto": texto, "caja": caja, "modo": modo}])


def _caja(centro_z, profundidad, largo=26.0, alto=5.5):
    return {"centro": np.array([2.0, -8.0, centro_z]), "ejes": np.eye(3), "tamano": np.array([largo, alto, profundidad])}


def _techo(guia, x, y):
    """z del primer corte de un rayo vertical que baja desde arriba."""
    arbol = vtk.vtkOBBTree()
    arbol.SetDataSet(guia)
    arbol.BuildLocator()
    cortes = vtk.vtkPoints()
    arbol.IntersectWithLine([x, y, 30], [x, y, -1], cortes, None)
    return cortes.GetPoint(0)[2] if cortes.GetNumberOfPoints() else None


@pytest.fixture(scope="module")
def grabado():
    return _guia(_caja(2.6, 0.8))                     # letras de z = 2,2 a 3,0: quedan 2,2 mm de resina debajo


def test_grabado_valido_hunde_las_letras(grabado):
    """Verifica R-026: con trazo ≥ 0,5 mm y 2,2 mm de resina debajo, la guía es válida y las letras se hunden 0,8 mm."""
    r = grabado
    assert r["valida"], r["problemas"]
    texto = r["textos"][0]
    assert texto["trazo_mm"] >= 0.5 and texto["resina_bajo_grabado_suficiente"]
    info = texto_como_malla("C12 OG", _caja(2.6, 0.8), ENCIA, np.array([0, 0, 1.0]))
    fondo = info["fondo"][0]
    assert fondo[2] == pytest.approx(2.2, abs=1e-9)
    assert _techo(r["guia"], fondo[0], fondo[1]) == pytest.approx(2.2, abs=0.05)      # dentro de una letra
    assert _techo(r["guia"], 12.0, 5.0) == pytest.approx(3.0, abs=0.05)               # fuera de la caja y del anillo


def test_trazo_menor_al_minimo_no_es_valido():
    """Verifica R-026: en una caja chica el trazo queda bajo 0,5 mm: no válida, con el alto de letra necesario."""
    r = _guia(_caja(2.6, 0.8, largo=10.0, alto=2.0))
    assert not r["valida"]
    assert r["textos"][0]["trazo_mm"] < 0.5
    assert any("trazo" in p and "agranda la caja" in p for p in r["problemas"])


def test_grabado_que_deja_menos_de_1_mm_no_es_valido():
    """Verifica R-026: grabar 2,4 mm en una carcasa de 3 mm deja 0,6 mm de resina: no válida."""
    r = _guia(_caja(1.8, 2.4))
    assert not r["valida"]
    assert r["textos"][0]["resina_bajo_grabado_suficiente"] is False
    assert any("menos de 1.0 mm de resina" in p for p in r["problemas"])


def test_relieve_suma_material():
    """Verifica R-026: en relieve las letras sobresalen de la cara externa (z = 3) hasta el borde de la caja."""
    r = _guia(_caja(3.4, 1.2), modo="relieve")          # caja de z = 2,8 a 4,0
    assert r["valida"], r["problemas"]
    info = texto_como_malla("C12 OG", _caja(3.4, 1.2), ENCIA, np.array([0, 0, 1.0]))
    centro_letra = info["fondo"][0]
    assert _techo(r["guia"], centro_letra[0], centro_letra[1]) == pytest.approx(4.0, abs=0.05)


def test_el_texto_se_lee_derecho_desde_afuera():
    """Verifica R-026: mirando la guía desde afuera (+z), la "L" tiene el trazo vertical a la izquierda y el pie hacia la derecha."""
    info = texto_como_malla("L", _caja(2.6, 0.8), ENCIA, np.array([0, 0, 1.0]))
    puntos = numpy_support.vtk_to_numpy(info["malla"].GetPoints().GetData())
    np.testing.assert_allclose(info["afuera"], [0, 0, 1])
    y_medio = (puntos[:, 1].min() + puntos[:, 1].max()) / 2
    arriba, abajo = puntos[puntos[:, 1] > y_medio], puntos[puntos[:, 1] < y_medio]
    assert arriba[:, 0].max() < abajo[:, 0].max()          # el pie de la L se extiende a la derecha (+x)


def test_texto_y_caja_van_juntos(capsys):
    """Verifica R-026: `guia --texto` sin `--caja-texto` es error de entrada (código 1)."""
    from nucleo_dental.cli import main

    codigo = main(["guia", "--escaneo", "e.stl", "--dientes", "d.stl", "--implante-stl", "i.stl", "--apice-hacia",
                   "abajo", "--curva", "c.mrk.json", "--plano-oclusal", "p.mrk.json", "--texto", "C12",
                   "--salida", "g.stl"])
    assert codigo == 1
    assert "--caja-texto" in capsys.readouterr().err
