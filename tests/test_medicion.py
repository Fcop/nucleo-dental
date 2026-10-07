"""Medición implante–canal mandibular (R-004, R-005, R-006, R-009).

Los resultados esperados vienen de tests/casos_dorados/*/esperado.json y de
valores calculados a mano por Francisco. Este archivo nunca los modifica.
"""

import json
from pathlib import Path

import pytest

from nucleo_dental.implante import Implante
from nucleo_dental.medicion import leer_stl, medir

CASOS_DORADOS = Path(__file__).parent / "casos_dorados"
CANAL_RECTO = CASOS_DORADOS / "canal_recto.stl"
CARPETAS_DORADAS = sorted(p.parent for p in CASOS_DORADOS.glob("*/esperado.json"))


def _cargar_caso(carpeta: Path):
    caso = json.loads((carpeta / "caso.json").read_text(encoding="utf-8"))
    esperado = json.loads((carpeta / "esperado.json").read_text(encoding="utf-8"))
    imp = caso["implante"]
    implante = Implante(imp["diametro"], imp["largo"], imp["apice"], imp["eje"])
    canal = leer_stl(carpeta / caso["canal"])
    return implante, canal, caso["margen"], esperado


def test_hay_casos_dorados():
    """Verifica R-004: existen los casos dorados escritos a mano (005 = malla gruesa, RG-004)."""
    assert [c.name for c in CARPETAS_DORADAS] == [
        "caso_001", "caso_002", "caso_003", "caso_004_eje_invertido", "caso_005_malla_gruesa"]


def _implante_caso_001() -> Implante:
    return Implante(diametro=4.1, largo=10.0, apice=[0, 0, 4.0], eje=[0, 0, 1])


@pytest.mark.parametrize(
    "margen, semaforo",
    [(2.5, "verde"),   # variante a: distancia igual al margen
     (3.0, "rojo")],   # variante b: la distancia no alcanza el margen
    ids=["igual_al_margen", "bajo_el_margen"],
)
def test_semaforo_en_el_limite_del_margen(margen, semaforo):
    """Verifica R-006: con distancia 2,5 mm, margen 2,5 es verde y margen 3,0 es rojo (calculado a mano)."""
    r = medir(_implante_caso_001(), leer_stl(CANAL_RECTO), margen)
    assert r["semaforo"] == semaforo


def test_margen_por_defecto_es_2_mm():
    """Verifica R-006: sin margen explícito se usa 2,0 mm (caso 001: 2,5 mm → verde; caso 002: 1,45 mm → rojo)."""
    canal = leer_stl(CANAL_RECTO)
    assert medir(_implante_caso_001(), canal)["semaforo"] == "verde"
    lateral = Implante(diametro=4.1, largo=10.0, apice=[0, 5, 0], eje=[0, 0, 1])
    assert medir(lateral, canal)["semaforo"] == "rojo"


def test_leer_stl_inexistente_da_error_claro(tmp_path):
    """Verifica R-008: un archivo de canal inexistente es un error de entrada con mensaje en español."""
    with pytest.raises(ValueError, match="No existe"):
        leer_stl(tmp_path / "no_existe.stl")


def test_canal_abierto_se_rechaza(tmp_path):
    """Verifica R-004: con un canal que no es una superficie cerrada no se puede decidir qué es dentro; se rechaza."""
    import vtk

    plano = vtk.vtkPlaneSource()
    plano.Update()
    with pytest.raises(ValueError, match="cerrada"):
        medir(_implante_caso_001(), plano.GetOutput())


@pytest.mark.parametrize("carpeta", CARPETAS_DORADAS, ids=lambda c: c.name)
def test_canal_con_normales_invertidas_da_el_mismo_resultado(carpeta):
    """Verifica R-004 y R-005: el resultado no depende de la orientación de los triángulos del STL."""
    import vtk

    implante, canal, margen, esperado = _cargar_caso(carpeta)
    invertir = vtk.vtkReverseSense()
    invertir.SetInputData(canal)
    invertir.ReverseCellsOn()
    invertir.Update()

    r = medir(implante, invertir.GetOutput(), margen)
    tol = esperado["tolerancia_mm"]
    assert r["colision"] is esperado["colision"]
    assert r["distancia_mm"] == pytest.approx(esperado["distancia_mm"], abs=tol)
    assert r["penetracion_mm"] == pytest.approx(esperado["penetracion_mm"], abs=tol)


@pytest.mark.parametrize("margen", [-1.0, float("nan")], ids=["negativo", "nan"])
def test_margen_invalido_se_rechaza(margen):
    """Verifica R-006: el margen debe ser un número mayor o igual que 0."""
    with pytest.raises(ValueError, match="margen"):
        medir(_implante_caso_001(), leer_stl(CANAL_RECTO), margen)


@pytest.mark.parametrize("carpeta", CARPETAS_DORADAS, ids=lambda c: c.name)
def test_caso_dorado(carpeta):
    """Verifica R-002, R-004, R-005, R-006 y R-009 contra los resultados calculados a mano."""
    implante, canal, margen, esperado = _cargar_caso(carpeta)
    r = medir(implante, canal, margen)
    tol = esperado["tolerancia_mm"]

    assert r["distancia_mm"] == pytest.approx(esperado["distancia_mm"], abs=tol)
    assert r["colision"] is esperado["colision"]
    assert r["penetracion_mm"] == pytest.approx(esperado["penetracion_mm"], abs=tol)
    assert r["semaforo"] == esperado["semaforo"]
