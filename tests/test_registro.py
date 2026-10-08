"""Registro del escaneo intraoral con los dientes del CBCT (R-015).

El caso dorado 011 fue calculado a mano por Francisco: un escaneo idéntico
a los dientes, corrido 0,5 mm en +x, se corrige con -0,5 mm en x, error 0,
y el punto (0,5, 0, 10) queda en (0, 0, 10).
"""

import json

import numpy as np
import pytest
import vtk

from dorados import CASOS_DORADOS
from nucleo_dental.medicion import leer_stl
from nucleo_dental.registro import aplicar_matriz, refinar_registro

CASO_011 = CASOS_DORADOS / "registro" / "caso_011_escaneo_corrido"
ARCADA = CASOS_DORADOS / "arcada_cbct.stl"


def _transformar(malla, angulo_z_grados=0.0, centro=(0, 0, 0), traslacion=(0, 0, 0)):
    tr = vtk.vtkTransform()
    tr.PostMultiply()
    tr.Translate(*(-np.asarray(centro)))
    tr.RotateZ(angulo_z_grados)
    tr.Translate(*centro)
    tr.Translate(*traslacion)
    f = vtk.vtkTransformPolyDataFilter()
    f.SetInputData(malla)
    f.SetTransform(tr)
    f.Update()
    return f.GetOutput()


def test_caso_dorado_011_escaneo_corrido():
    """Verifica R-015: el escaneo corrido 0,5 mm en +x se corrige con -0,5 mm, error 0, y el punto (0,5, 0, 10) va a (0, 0, 10)."""
    caso = json.loads((CASO_011 / "caso.json").read_text(encoding="utf-8"))
    esperado = json.loads((CASO_011 / "esperado.json").read_text(encoding="utf-8"))
    tol = esperado["tolerancia_mm"]

    r = refinar_registro(leer_stl(CASO_011 / caso["escaneo"]), leer_stl(CASO_011 / caso["dientes"]))
    matriz = np.array(r["matriz_lps"])

    np.testing.assert_allclose(matriz[:3, 3], esperado["traslacion_mm"], atol=tol)
    assert r["rotacion_grados"] == pytest.approx(esperado["rotacion_grados"], abs=0.1)
    assert r["desviacion_despues_mm"]["mediana"] == pytest.approx(esperado["error_residual_mm"], abs=tol)
    np.testing.assert_allclose(aplicar_matriz(matriz, [caso["punto"]])[0], esperado["punto_registrado"], atol=tol)


def test_la_encia_no_participa():
    """Verifica R-015: la lámina de "encía" del escaneo (z = 4), que no existe en el CBCT, queda fuera del ajuste."""
    caso = json.loads((CASO_011 / "caso.json").read_text(encoding="utf-8"))
    escaneo = leer_stl(CASO_011 / caso["escaneo"])
    arcada = leer_stl(ARCADA)
    r = refinar_registro(escaneo, arcada)
    # Tras el ajuste participan exactamente los vértices de las cúspides; ninguno de la encía.
    assert r["puntos_usados"] == arcada.GetNumberOfPoints()
    assert escaneo.GetNumberOfPoints() > arcada.GetNumberOfPoints()


def test_escaneo_ya_alineado_no_se_mueve():
    """Verifica R-015: si el escaneo ya calza, la corrección es nula."""
    arcada = leer_stl(ARCADA)
    r = refinar_registro(arcada, arcada)
    np.testing.assert_allclose(np.array(r["matriz_lps"]), np.eye(4), atol=1e-3)


def test_recupera_giro_y_traslacion():
    """Verifica R-015: un escaneo girado 1° en z (en torno a (5, 9, 10)) y corrido 0,3 mm en y vuelve a su sitio.

    La referencia sale de la construcción: el registro debe deshacer exactamente el movimiento aplicado.
    """
    arcada = leer_stl(ARCADA)
    escaneo = _transformar(arcada, 1.0, centro=(5, 9, 10), traslacion=(0, 0.3, 0))
    r = refinar_registro(escaneo, arcada)
    puntos = np.array([[0, 0, 10], [10, 10, 12], [5, 18, 9]], dtype=float)
    from vtk.util import numpy_support

    movidos = numpy_support.vtk_to_numpy(
        _transformar(_poli(puntos), 1.0, (5, 9, 10), (0, 0.3, 0)).GetPoints().GetData()).astype(float)
    np.testing.assert_allclose(aplicar_matriz(np.array(r["matriz_lps"]), movidos), puntos, atol=0.05)
    assert r["rotacion_grados"] == pytest.approx(1.0, abs=0.1)


def test_estabilidad_informada():
    """Verifica R-015: el registro informa su estabilidad (dispersión entre variantes) y en el caso dorado es menor que 0,05 mm."""
    caso = json.loads((CASO_011 / "caso.json").read_text(encoding="utf-8"))
    r = refinar_registro(leer_stl(CASO_011 / caso["escaneo"]), leer_stl(ARCADA), punto=caso["punto"])
    assert r["estabilidad_mm"] < 0.05
    assert r["correccion_en_punto_mm"] == pytest.approx(0.5, abs=0.05)


def test_sin_superposicion_es_error():
    """Verifica R-015: si el escaneo está lejos de los dientes (sin alineación previa), se rechaza con un mensaje claro."""
    arcada = leer_stl(ARCADA)
    with pytest.raises(ValueError, match="superposición"):
        refinar_registro(_transformar(arcada, traslacion=(30, 0, 0)), arcada)


def test_comando_registrar(tmp_path):
    """Verifica R-015, R-007 y R-008: `registrar` informa la corrección, guarda el escaneo registrado y lo traza."""
    from test_medir import _sha256, ejecutar

    caso = json.loads((CASO_011 / "caso.json").read_text(encoding="utf-8"))
    escaneo = (CASO_011 / caso["escaneo"]).resolve()
    salida_stl = tmp_path / "escaneo_registrado.stl"
    codigo, salida, stderr = ejecutar("registrar", "--escaneo", escaneo, "--dientes", ARCADA,
                                      "--punto", "0.5,0,10", "--salida", salida_stl)
    assert codigo == 0, stderr
    r = salida["registro"]
    np.testing.assert_allclose(r["traslacion_mm"], [-0.5, 0, 0], atol=0.05)
    assert r["correccion_en_punto_mm"] == pytest.approx(0.5, abs=0.05)
    assert salida["parametros"]["sistema_coordenadas"] == "LPS"

    rutas = {a["ruta"]: a["sha256"] for a in salida["trazabilidad"]["archivos_entrada"]}
    assert rutas[str(escaneo)] == _sha256(escaneo)
    assert salida["escaneo_registrado"]["ruta"] == str(salida_stl.resolve())
    assert salida["escaneo_registrado"]["sha256"] == _sha256(salida_stl)

    # El escaneo guardado ya calza: registrarlo de nuevo no pide corrección.
    _, salida2, stderr2 = ejecutar("registrar", "--escaneo", salida_stl, "--dientes", ARCADA)
    assert salida2 is not None, stderr2
    assert np.linalg.norm(salida2["registro"]["traslacion_mm"]) < 0.02


@pytest.mark.parametrize("argumentos, texto",
                         [(["registrar", "--dientes", ARCADA], "--escaneo"),
                          (["registrar", "--escaneo", ARCADA], "--dientes"),
                          (["registrar", "--escaneo", "no_existe.stl", "--dientes", ARCADA], "No existe"),
                          (["registrar", "--escaneo", ARCADA, "--dientes", ARCADA, "--punto", "1,2"], "--punto")],
                         ids=["sin_escaneo", "sin_dientes", "escaneo_inexistente", "punto_incompleto"])
def test_comando_registrar_errores(argumentos, texto):
    """Verifica R-008 y R-015: errores de entrada de `registrar` devuelven 1 con mensaje claro."""
    from test_medir import ejecutar

    codigo, salida, stderr = ejecutar(*argumentos)
    assert codigo == 1
    assert salida is None
    assert texto in stderr


def _poli(puntos):
    from vtk.util import numpy_support

    vp = vtk.vtkPoints()
    vp.SetData(numpy_support.numpy_to_vtk(np.ascontiguousarray(puntos), deep=True))
    pd = vtk.vtkPolyData()
    pd.SetPoints(vp)
    return pd
