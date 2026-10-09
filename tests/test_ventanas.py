"""Ventanas de inspección: cajas que el usuario ubica y dimensiona, restadas de la guía (R-024)."""

import json

import numpy as np
import pytest
import vtk
from vtk.util import numpy_support

from nucleo_dental.apoyo import leer_cajas_slicer
from nucleo_dental.ensamblaje import guia_quirurgica
from nucleo_dental.guia import caja_como_malla
from nucleo_dental.implante import Implante
from nucleo_dental.kits import ONEGUIDE

GIRO_30 = np.array([[np.cos(np.pi / 6), -np.sin(np.pi / 6), 0], [np.sin(np.pi / 6), np.cos(np.pi / 6), 0], [0, 0, 1]])


def _roi(tmp_path, sistema="LPS", **extra):
    marca = {"type": "ROI", "coordinateSystem": sistema, "roiType": "Box", "center": [1.0, 2.0, 3.0],
             "orientation": GIRO_30.ravel().tolist(), "size": [3.0, 4.0, 8.0], **extra}
    ruta = tmp_path / f"roi_{sistema}.mrk.json"
    ruta.write_text(json.dumps({"markups": [marca]}), encoding="utf-8")
    return ruta


def test_leer_caja_de_slicer_lps_y_ras(tmp_path):
    """Verifica R-024: la ROI se lee con los ejes en las columnas; si viene en RAS se pasa a LPS (centro y ejes)."""
    lps = leer_cajas_slicer(_roi(tmp_path))[0]
    np.testing.assert_allclose(lps["centro"], [1, 2, 3])
    np.testing.assert_allclose(lps["ejes"][:, 0], [np.cos(np.pi / 6), np.sin(np.pi / 6), 0], atol=1e-12)
    np.testing.assert_allclose(lps["tamano"], [3, 4, 8])
    ras = leer_cajas_slicer(_roi(tmp_path, "RAS"))[0]
    np.testing.assert_allclose(ras["centro"], [-1, -2, 3])
    np.testing.assert_allclose(ras["ejes"][:, 0], [-np.cos(np.pi / 6), -np.sin(np.pi / 6), 0], atol=1e-12)


@pytest.mark.parametrize("extra, mensaje", [({"insideOut": True}, "insideOut"), ({"size": [3, 0, 8]}, "tamaño")])
def test_caja_invalida_es_error(tmp_path, extra, mensaje):
    """Verifica R-024: una caja invertida o con una arista nula no se acepta."""
    with pytest.raises(ValueError, match=mensaje):
        leer_cajas_slicer(_roi(tmp_path, **extra))


def test_caja_sin_sistema_o_sin_rois_es_error(tmp_path):
    """Verifica R-024: sin coordinateSystem no se adivina; un archivo sin ROI no sirve como ventanas."""
    ruta = _roi(tmp_path)
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    del datos["markups"][0]["coordinateSystem"]
    ruta.write_text(json.dumps(datos), encoding="utf-8")
    with pytest.raises(ValueError, match="coordinateSystem"):
        leer_cajas_slicer(ruta)
    datos["markups"][0]["type"] = "Fiducial"
    ruta.write_text(json.dumps(datos), encoding="utf-8")
    with pytest.raises(ValueError, match="ninguna caja"):
        leer_cajas_slicer(ruta)


def test_caja_como_malla():
    """Verifica R-024: la caja es un sólido cerrado de 3 × 4 × 8 = 96 mm³, girado según sus ejes."""
    caja = caja_como_malla([1, 2, 3], GIRO_30, [3, 4, 8])
    masa = vtk.vtkMassProperties()
    masa.SetInputData(caja)
    masa.Update()
    assert masa.GetVolume() == pytest.approx(96.0, rel=1e-9)
    puntos = numpy_support.vtk_to_numpy(caja.GetPoints().GetData())
    local = (puntos - [1, 2, 3]) @ GIRO_30
    np.testing.assert_allclose(np.abs(local), np.tile([1.5, 2.0, 4.0], (len(puntos), 1)), atol=1e-12)


@pytest.fixture(scope="module")
def guia_con_ventanas():
    cubo = vtk.vtkCubeSource()
    cubo.SetBounds(-12, 20, -10, 10, -5, 0)
    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(cubo.GetOutputPort())
    limpio = vtk.vtkCleanPolyData()
    limpio.SetInputConnection(tri.GetOutputPort())
    sub = vtk.vtkLinearSubdivisionFilter()
    sub.SetInputConnection(limpio.GetOutputPort())
    sub.SetNumberOfSubdivisions(5)
    sub.Update()
    encia = sub.GetOutput()
    z = numpy_support.vtk_to_numpy(encia.GetPoints().GetData())[:, 2]
    ventanas = [{"centro": np.array([-8.0, 6.0, 1.5]), "ejes": GIRO_30, "tamano": np.array([3.0, 3.0, 8.0])},
                {"centro": np.array([-8.0, 6.0, 40.0]), "ejes": np.eye(3), "tamano": np.array([3.0, 3.0, 3.0])}]
    return guia_quirurgica(encia, None, Implante(4.1, 10, [0, 0, -12], [0, 0, 1]), ONEGUIDE, "impresa", z > -0.01,
                           [0, 0, 1], tipo_soporte="mucosoportada", ventanas=ventanas)


def _atraviesa(guia, x, y) -> bool:
    arbol = vtk.vtkOBBTree()
    arbol.SetDataSet(guia)
    arbol.BuildLocator()
    puntos = vtk.vtkPoints()
    arbol.IntersectWithLine([x, y, 30], [x, y, -1], puntos, None)
    return puntos.GetNumberOfPoints() == 0


def test_la_ventana_abre_la_guia(guia_con_ventanas):
    """Verifica R-024: dentro de la caja girada 30° la guía queda abierta de lado a lado; fuera de ella hay resina."""
    r = guia_con_ventanas
    assert r["valida"], r["problemas"]

    def mundo(u, v):
        return np.array([-8.0, 6.0]) + GIRO_30[:2, :2] @ [u, v]

    for u, v in ((0, 0), (1.3, 1.3), (-1.3, 0.5)):                      # dentro (media arista 1,5)
        assert _atraviesa(r["guia"], *mundo(u, v))
    for u, v in ((1.8, 0), (0, -1.8)):                                  # fuera
        assert not _atraviesa(r["guia"], *mundo(u, v))


def test_ventana_que_no_toca_la_guia_es_aviso(guia_con_ventanas):
    """Verifica R-024: una caja lejos de la guía no corta nada y queda un aviso; la guía sigue válida."""
    r = guia_con_ventanas
    assert [v["corta_la_guia"] for v in r["ventanas"]] == [True, False]
    assert r["avisos"] == ["La ventana 2 no toca la guía: revisa su posición."]
