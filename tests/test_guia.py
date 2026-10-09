"""Perfil de kit y anillo guía con su orificio (R-018).

Caso dorado 015: implante del caso 001 con OneGuide impresa (ver su
esperado.json para el origen de cada valor).
"""

import json

import numpy as np
import pytest
import vtk
from vtk.util import numpy_support

from dorados import CASOS_DORADOS
from nucleo_dental.guia import anillo_y_orificio, geometria_orificio
from nucleo_dental.implante import Implante
from nucleo_dental.kits import ONEGUIDE, PerfilKit, obtener_kit

CASO_015 = CASOS_DORADOS / "guia" / "caso_015_anillo_oneguide"


def _implante(datos):
    return Implante(datos["diametro"], datos["largo"], datos["apice"], datos["eje"])


def _caso_015():
    caso = json.loads((CASO_015 / "caso.json").read_text(encoding="utf-8"))
    esperado = json.loads((CASO_015 / "esperado.json").read_text(encoding="utf-8"))
    return _implante(caso["implante"]), obtener_kit(caso["kit"]), caso["fabricacion"], esperado


def test_caso_dorado_015_alturas_y_diametros():
    """Verifica R-018: plataforma, caras del anillo, orificio, anillo y fresa del caso 015."""
    implante, kit, fabricacion, e = _caso_015()
    g = geometria_orificio(implante, kit, fabricacion)
    tol = e["tolerancia_mm"]
    assert g["plataforma"][2] == pytest.approx(e["plataforma_z"], abs=tol)
    assert g["cara_superior"][2] == pytest.approx(e["cara_superior_z"], abs=tol)
    assert g["cara_inferior_anillo"][2] == pytest.approx(e["cara_inferior_anillo_z"], abs=tol)
    assert g["diametro_orificio_mm"] == pytest.approx(e["diametro_orificio_mm"], abs=tol)
    assert g["diametro_externo_anillo_mm"] == pytest.approx(e["diametro_externo_anillo_mm"], abs=tol)
    assert g["punta_fresa"][2] == pytest.approx(e["punta_fresa_z"], abs=tol)
    assert g["largo_trabajo_fresa_mm"] == pytest.approx(e["largo_trabajo_fresa_mm"], abs=tol)


def test_perfil_oneguide():
    """Verifica R-018: valores del perfil OneGuide (catálogo Hiossen y decisiones clínicas 2026-10-08)."""
    assert ONEGUIDE.contacto_mm == 3.0
    assert ONEGUIDE.offset_mm == 10.5
    assert ONEGUIDE.espesor_plantilla_mm == 3.0
    assert ONEGUIDE.pared_anillo_mm == 3.0
    assert ONEGUIDE.holgura_encia_mm == 1.0
    assert ONEGUIDE.sobrefresado_mm == 0.3
    assert "sobrefresado_mm" in ONEGUIDE.provisionales
    assert ONEGUIDE.con_camisa is False
    assert ONEGUIDE.ajuste_fabricacion_mm == {"impresa": 0.3, "fresada": 0.1}


@pytest.mark.parametrize("diametro, guia", [(3.5, 5.0), (4.0, 5.0), (4.5, 5.0), (5.0, 5.7)])
def test_oneguide_elige_el_orificio_por_diametro(diametro, guia):
    """Verifica R-018: OneGuide usa Ø5,0 para fijaciones F3.5-F4.5 y Ø5,7 para F5.0 (catálogo)."""
    assert ONEGUIDE.diametro_guia_para(diametro) == guia


def test_implante_sin_orificio_en_el_kit_es_error():
    """Verifica R-018: un implante más ancho que lo que admite el kit se rechaza."""
    with pytest.raises(ValueError, match="kit"):
        ONEGUIDE.diametro_guia_para(6.0)


def test_fresada_usa_su_propio_ajuste():
    """Verifica R-018: fresada agrega 0,1 mm (Ø5,1); un método desconocido es error."""
    implante, kit, _, _ = _caso_015()
    assert geometria_orificio(implante, kit, "fresada")["diametro_orificio_mm"] == pytest.approx(5.1)
    with pytest.raises(ValueError, match="fabricación"):
        geometria_orificio(implante, kit, "tallada")


def test_perfil_configurable():
    """Verifica R-018: todos los parámetros del kit se pueden cambiar sin tocar el código."""
    implante, _, _, _ = _caso_015()
    otro = ONEGUIDE.con(offset_mm=9.0, contacto_mm=4.0)
    g = geometria_orificio(implante, otro, "impresa")
    assert g["cara_superior"][2] == pytest.approx(23.0)
    assert g["cara_inferior_anillo"][2] == pytest.approx(19.0)
    assert ONEGUIDE.offset_mm == 10.5            # el perfil original no cambia
    assert isinstance(otro, PerfilKit)


def test_con_camisa_el_orificio_es_el_externo_de_la_camisa():
    """Verifica R-018: con camisa, la resina aloja el diámetro externo (interno + 2 paredes) más el ajuste."""
    implante, _, _, _ = _caso_015()
    con_camisa = ONEGUIDE.con(con_camisa=True, pared_camisa_mm=1.0)
    g = geometria_orificio(implante, con_camisa, "impresa")
    assert g["diametro_interno_camisa_mm"] == pytest.approx(5.0)
    assert g["diametro_externo_camisa_mm"] == pytest.approx(7.0)
    assert g["diametro_orificio_mm"] == pytest.approx(7.3)


def _cerrada(malla) -> bool:
    bordes = vtk.vtkFeatureEdges()
    bordes.SetInputData(malla)
    bordes.BoundaryEdgesOn()
    bordes.NonManifoldEdgesOn()
    bordes.FeatureEdgesOff()
    bordes.ManifoldEdgesOff()
    bordes.Update()
    return bordes.GetOutput().GetNumberOfCells() == 0


def test_parche_a_solido_cuadrado_plano():
    """Verifica R-018: un parche plano de 10 x 10 mm con grosor 3 da un sólido cerrado de 300 mm³ (z de 0 a 3)."""
    from nucleo_dental.geometria.malla_guia import parche_a_solido

    plano = vtk.vtkPlaneSource()
    plano.SetOrigin(0, 0, 0)
    plano.SetPoint1(10, 0, 0)
    plano.SetPoint2(0, 10, 0)
    plano.SetResolution(20, 20)
    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(plano.GetOutputPort())
    tri.Update()
    solido = parche_a_solido(tri.GetOutput(), 3.0)
    assert _cerrada(solido)
    masa = vtk.vtkMassProperties()
    masa.SetInputData(solido)
    masa.Update()
    assert masa.GetVolume() == pytest.approx(300.0, rel=1e-6)
    z = np.array(solido.GetBounds())[4:]
    assert abs(z[1] - z[0]) == pytest.approx(3.0)


def test_caso_dorado_016_puente_y_columna():
    """Verifica R-019: piso del puente 1 mm sobre la encía, techo a 3 mm, columna hasta la cara superior (8,5), alto 4,5."""
    from nucleo_dental.guia import puente_y_columna
    from nucleo_dental.medicion import leer_stl

    carpeta = CASOS_DORADOS / "guia" / "caso_016_puente"
    caso = json.loads((carpeta / "caso.json").read_text(encoding="utf-8"))
    e = json.loads((carpeta / "esperado.json").read_text(encoding="utf-8"))
    tol = e["tolerancia_mm"]
    r = puente_y_columna(leer_stl(carpeta / caso["escaneo"]), leer_stl(carpeta / caso["dientes"]),
                         _implante(caso["implante"]), obtener_kit(caso["kit"]), caso["fabricacion"])

    assert r["geometria"]["plataforma"][2] == pytest.approx(e["plataforma_z"], abs=tol)
    assert r["geometria"]["cara_superior"][2] == pytest.approx(e["cara_superior_z"], abs=tol)
    zp = np.array(r["puente"].GetBounds())[4:]
    assert zp[0] == pytest.approx(e["piso_puente_z"], abs=tol)
    assert zp[1] == pytest.approx(e["techo_puente_z"], abs=tol)
    assert r["alto_columna_mm"] == pytest.approx(e["alto_columna_mm"], abs=tol)
    zc = np.array(r["columna"].GetBounds())[4:]
    assert zc[1] == pytest.approx(e["cara_superior_z"], abs=tol)
    assert zc[0] <= e["techo_puente_z"] + tol                       # la columna se une al puente
    assert _cerrada(r["puente"]) and _cerrada(r["columna"])


def test_contacto_efectivo_sin_alivio():
    """Verifica R-019: sin alivio, el contacto guía–fresa va del piso del puente a la cara superior (8,5 - 1 = 7,5 mm)."""
    from nucleo_dental.guia import puente_y_columna
    from nucleo_dental.medicion import leer_stl

    carpeta = CASOS_DORADOS / "guia" / "caso_016_puente"
    caso = json.loads((carpeta / "caso.json").read_text(encoding="utf-8"))
    r = puente_y_columna(leer_stl(carpeta / caso["escaneo"]), leer_stl(carpeta / caso["dientes"]),
                         _implante(caso["implante"]), obtener_kit(caso["kit"]), caso["fabricacion"])
    assert r["contacto_efectivo_mm"] == pytest.approx(7.5, abs=0.05)


def test_mallas_del_anillo_y_del_orificio():
    """Verifica R-018: el anillo es un cilindro cerrado de Ø11,3 entre z 21,5 y 24,5; el orificio de Ø5,3 atraviesa más allá."""
    implante, kit, fabricacion, e = _caso_015()
    anillo, orificio = anillo_y_orificio(implante, kit, fabricacion)
    a = np.array(anillo.GetBounds())
    np.testing.assert_allclose(a[[4, 5]], [21.5, 24.5], atol=1e-6)
    np.testing.assert_allclose(a[1] - a[0], 11.3, atol=1e-6)
    o = np.array(orificio.GetBounds())
    np.testing.assert_allclose(o[1] - o[0], 5.3, atol=1e-6)
    assert o[4] < 21.5 - 3 and o[5] > 24.5 + 1          # atraviesa toda la guía
    for malla in (anillo, orificio):
        bordes = vtk.vtkFeatureEdges()
        bordes.SetInputData(malla)
        bordes.BoundaryEdgesOn()
        bordes.NonManifoldEdgesOn()
        bordes.FeatureEdgesOff()
        bordes.ManifoldEdgesOff()
        bordes.Update()
        assert bordes.GetOutput().GetNumberOfCells() == 0
    # ambos coaxiales con el implante
    for malla in (anillo, orificio):
        p = numpy_support.vtk_to_numpy(malla.GetPoints().GetData())
        assert np.allclose(p[:, :2].mean(axis=0), [0, 0], atol=1e-6)
