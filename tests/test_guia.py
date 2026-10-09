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


def test_holgura_y_espesor_no_entran_en_la_encia():
    """Verifica R-019 (RG-017): sobre un surco de encía más angosto que la holgura, ningún punto del puente queda bajo la encía.

    Antes la holgura se aplicaba primero y las normales se recalculaban sobre la capa
    desplazada: en el surco esa capa se pliega, las normales se invierten y el espesor
    se extruía hacia el tejido (en el caso real llegó 1,7 mm dentro del hueso).
    """
    from nucleo_dental.geometria.malla_guia import parche_a_solido

    def encia(x):
        return -3.0 * np.exp(-(x / 0.6) ** 2)          # surco de 3 mm de hondo y ~1,2 mm de ancho

    plano = vtk.vtkPlaneSource()
    plano.SetOrigin(-6, -4, 0)
    plano.SetPoint1(6, -4, 0)
    plano.SetPoint2(-6, 4, 0)
    plano.SetResolution(120, 40)
    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(plano.GetOutputPort())
    tri.Update()
    superficie = tri.GetOutput()
    puntos = numpy_support.vtk_to_numpy(superficie.GetPoints().GetData())
    puntos[:, 2] = encia(puntos[:, 0])
    superficie.GetPoints().Modified()

    solido = parche_a_solido(superficie, 3.0, desfase=1.0, suavizado_normales=10)
    p = numpy_support.vtk_to_numpy(solido.GetPoints().GetData())
    assert (p[:, 2] - encia(p[:, 0])).min() >= 1.0 - 1e-6           # todo queda al menos a la holgura
    assert _cerrada(solido)


def test_alisar_borde_quita_la_escalera_sin_tocar_el_interior():
    """Verifica R-019: el contorno del puente se alisa (borde más corto, sin escalera) y la superficie de apoyo no se mueve."""
    from nucleo_dental.apoyo import parche_de_apoyo
    from nucleo_dental.geometria.malla_guia import _aristas_borde_con_ids
    from nucleo_dental.guia import _alisar_borde

    plano = vtk.vtkPlaneSource()
    plano.SetOrigin(-10, -10, 0)
    plano.SetPoint1(10, -10, 0)
    plano.SetPoint2(-10, 10, 0)
    plano.SetResolution(40, 40)
    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(plano.GetOutputPort())
    tri.Update()
    puntos = numpy_support.vtk_to_numpy(tri.GetOutput().GetPoints().GetData())
    parche = parche_de_apoyo(tri.GetOutput(), np.linalg.norm(puntos[:, :2], axis=1) <= 7.0)   # círculo en escalera

    def largo_borde(malla):
        aristas = _aristas_borde_con_ids(malla)
        p = numpy_support.vtk_to_numpy(aristas.GetPoints().GetData())
        lineas = numpy_support.vtk_to_numpy(aristas.GetLines().GetConnectivityArray()).reshape(-1, 2)
        return np.linalg.norm(p[lineas[:, 0]] - p[lineas[:, 1]], axis=1).sum()

    alisado = _alisar_borde(parche, 10)
    antes = numpy_support.vtk_to_numpy(parche.GetPoints().GetData())
    despues = numpy_support.vtk_to_numpy(alisado.GetPoints().GetData())
    movidos = np.linalg.norm(despues - antes, axis=1) > 1e-9
    assert largo_borde(alisado) < 0.85 * largo_borde(parche)       # la escalera mide ~4/π del círculo
    assert movidos.sum() <= len(numpy_support.vtk_to_numpy(_aristas_borde_con_ids(parche).GetPoints().GetData()))
    assert np.abs(despues[:, 2]).max() < 1e-9                       # sigue sobre la superficie plana
    radio = np.linalg.norm(despues[movidos, :2], axis=1)
    assert radio.min() > 6.0 and radio.max() < 7.5                  # el contorno no encoge ni se va


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


def _puente_016(**opciones):
    from nucleo_dental.guia import puente_y_columna
    from nucleo_dental.medicion import leer_stl

    carpeta = CASOS_DORADOS / "guia" / "caso_016_puente"
    caso = json.loads((carpeta / "caso.json").read_text(encoding="utf-8"))
    kit = opciones.pop("kit", obtener_kit(caso["kit"]))
    return puente_y_columna(leer_stl(carpeta / caso["escaneo"]), leer_stl(carpeta / caso["dientes"]),
                            _implante(caso["implante"]), kit, caso["fabricacion"], **opciones)


@pytest.mark.parametrize("tipo", ["dentomucosoportada", "mucosoportada"])
def test_apoyo_en_mucosa_sin_holgura(tipo):
    """Verifica R-020: si la guía apoya en mucosa no hay holgura: piso del puente en z = 0, techo en 3, columna de 5,5 mm.

    Valores derivados de la regla de Francisco (2026-10-09): "si se apoya en mucosa, este alivio no debe existir".
    """
    r = _puente_016(tipo_soporte=tipo)
    zp = np.array(r["puente"].GetBounds())[4:]
    assert zp[0] == pytest.approx(0.0, abs=0.05)
    assert zp[1] == pytest.approx(3.0, abs=0.05)
    assert r["alto_columna_mm"] == pytest.approx(5.5, abs=0.05)
    assert r["contacto_efectivo_mm"] == pytest.approx(8.5, abs=0.05)
    assert r["tipo_soporte"] == tipo


def test_tipo_de_soporte_desconocido_es_error():
    """Verifica R-020: solo se aceptan los tres tipos de soporte."""
    with pytest.raises(ValueError, match="soporte"):
        _puente_016(tipo_soporte="implantosoportada")


def test_mucosoportada_no_necesita_dientes():
    """Verifica R-020: una guía mucosoportada se construye sobre un escaneo sin dientes (desdentado)."""
    from nucleo_dental.guia import puente_y_columna

    plano = vtk.vtkPlaneSource()
    plano.SetOrigin(-15, -15, 0)
    plano.SetPoint1(15, -15, 0)
    plano.SetPoint2(-15, 15, 0)
    plano.SetResolution(60, 60)
    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(plano.GetOutputPort())
    tri.Update()
    implante = Implante(4.1, 10.0, [0, 0, -12.0], [0, 0, 1])
    r = puente_y_columna(tri.GetOutput(), None, implante, ONEGUIDE, "impresa",
                         tipo_soporte="mucosoportada", radio_mucosa_mm=10.0)
    zp = np.array(r["puente"].GetBounds())
    np.testing.assert_allclose(zp[4:], [0.0, 3.0], atol=0.05)
    # radio 10 alrededor del eje: el alisado del borde lo retrae menos de una celda (0,5 mm) por lado
    assert 19.0 <= zp[1] - zp[0] <= 20.0 + 1e-6


def test_dentosoportada_necesita_dientes():
    """Verifica R-020: sin dientes del CBCT no se puede delimitar una guía dentosoportada."""
    from nucleo_dental.guia import puente_y_columna
    from nucleo_dental.medicion import leer_stl

    with pytest.raises(ValueError, match="dientes"):
        puente_y_columna(leer_stl(CASOS_DORADOS / "escaneo_arcada_recta.stl"), None,
                         Implante(4.1, 10.0, [0, 0, -12.0], [0, 0, 1]), ONEGUIDE, "impresa")


def test_alivio_configurable():
    """Verifica R-020: con alivio de 1 mm el contacto vuelve a los 3 mm del kit y se genera el alivio de Ø6,3 bajo el anillo."""
    r = _puente_016(kit=ONEGUIDE.con(alivio_mm=1.0))
    assert r["contacto_efectivo_mm"] == pytest.approx(3.0, abs=1e-6)
    alivio = np.array(r["alivio"].GetBounds())
    assert alivio[1] - alivio[0] == pytest.approx(6.3, abs=1e-6)
    assert alivio[5] == pytest.approx(5.5, abs=1e-6)                 # llega a la cara inferior del anillo
    assert alivio[4] <= 1.0                                           # desde el piso del puente
    assert _puente_016()["alivio"] is None                            # sin alivio por defecto (OneGuide)


def test_limitar_profundidad():
    """Verifica R-020: solo quedan los puntos hasta 6 mm bajo la cresta, medidos a lo largo del eje."""
    from nucleo_dental.guia import _margen_de_profundidad

    puntos = np.array([[0, 0, 0], [0, 5, -5.9], [0, 6, -6.1], [0, 8, -10]], dtype=float)
    margen = _margen_de_profundidad(puntos, cresta=[0, 0, 0], eje=[0, 0, 1], profundidad_mm=6.0)
    np.testing.assert_allclose(margen, [6.0, 0.1, -0.1, -4.0], atol=1e-9)
    assert (margen >= 0).tolist() == [True, True, False, False]


def test_perfil_tiene_profundidad_y_alivio():
    """Verifica R-020: el perfil OneGuide trae profundidad de puente 6 mm y alivio 0 (decisiones clínicas 2026-10-09)."""
    assert ONEGUIDE.profundidad_puente_mm == 6.0
    assert ONEGUIDE.alivio_mm == 0.0


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
