"""Medición implante–estructuras: canal mandibular y dientes (R-004, R-005, R-006, R-009, R-013).

Los resultados esperados vienen de tests/casos_dorados/*/esperado.json y de
valores calculados a mano por Francisco. Este archivo nunca los modifica.
"""

from pathlib import Path

import pytest

from dorados import CARPETAS_DORADAS, CASOS_DORADOS, comparar, leer_caso, leer_esperado
from nucleo_dental.implante import Implante
from nucleo_dental.medicion import (MARGEN_DIENTES_POR_DEFECTO_MM, MARGEN_HUESO_POR_DEFECTO_MM, espesor_oseo,
                                    evaluar_plan, leer_stl, medir)

CANAL_RECTO = CASOS_DORADOS / "canal_recto.stl"


def _cargar_caso(carpeta: Path):
    """Implante, estructuras {nombre: (malla, margen)} y resultado esperado de un caso dorado."""
    caso = leer_caso(carpeta)
    imp = caso["implante"]
    implante = Implante(imp["diametro"], imp["largo"], imp["apice"], imp["eje"])
    estructuras = {"canal": (leer_stl(carpeta / caso["canal"]), caso["margen"])}
    if "dientes" in caso:
        estructuras["dientes"] = (leer_stl(carpeta / caso["dientes"]),
                                  caso.get("margen_dientes", MARGEN_DIENTES_POR_DEFECTO_MM))
    if "hueso" in caso:
        estructuras["hueso"] = (leer_stl(carpeta / caso["hueso"]),
                                caso.get("margen_hueso", MARGEN_HUESO_POR_DEFECTO_MM))
    return implante, estructuras, leer_esperado(carpeta)


def test_hay_casos_dorados():
    """Verifica R-004, R-013 y R-014: existen los casos dorados escritos a mano (005 = malla gruesa; 006-008 = dientes; 009-010 = hueso)."""
    assert [c.name for c in CARPETAS_DORADAS] == [
        "caso_001", "caso_002", "caso_003", "caso_004_eje_invertido", "caso_005_malla_gruesa",
        "caso_006_diente_verde", "caso_007_diente_rojo", "caso_008_corona_no_cuenta",
        "caso_009_hueso_rojo", "caso_010_hueso_verde"]


def test_margen_hueso_por_defecto_es_1_5_mm():
    """Verifica R-014: el espesor óseo mínimo por defecto en las paredes laterales es 1,5 mm."""
    assert MARGEN_HUESO_POR_DEFECTO_MM == 1.5


def test_pared_fuera_del_hueso_da_espesor_cero():
    """Verifica R-014: si parte de la pared lateral queda fuera del hueso (dehiscencia), el espesor es 0 y es rojo.

    Hueso 010 (de z = 0 a 20) con el implante subido a ápice z = 15: las paredes sobre z = 20 quedan fuera.
    """
    implante = Implante(diametro=4.1, largo=10.0, apice=[0, 0, 15.0], eje=[0, 0, 1])
    r = espesor_oseo(implante, leer_stl(CASOS_DORADOS / "hueso_010.stl"), 1.5)
    assert r["espesor_minimo_mm"] == 0.0
    assert r["semaforo"] == "rojo"


def test_bajo_el_apice_no_cuenta():
    """Verifica R-014: el hueso bajo el ápice no se mide; con solo 0,25 mm de hueso bajo el ápice no da rojo.

    Hueso 010 va de z = 0 a 20; implante con ápice en z = 0,25: espesor lateral sigue siendo 1,95.
    """
    implante = Implante(diametro=4.1, largo=10.0, apice=[0, 0, 0.25], eje=[0, 0, 1])
    r = espesor_oseo(implante, leer_stl(CASOS_DORADOS / "hueso_010.stl"), 1.5)
    assert r["espesor_minimo_mm"] == pytest.approx(1.95, abs=0.05)
    assert r["semaforo"] == "verde"


def test_espesor_informa_donde_esta_el_minimo():
    """Verifica R-014: la salida indica la dirección (LPS) y la altura sobre el ápice del espesor mínimo (caso 009: hacia -y)."""
    implante, estructuras, _ = _cargar_caso(CASOS_DORADOS / "caso_009_hueso_rojo")
    malla, margen = estructuras["hueso"]
    r = espesor_oseo(implante, malla, margen)
    assert r["direccion_minimo"] == pytest.approx([0, -1, 0], abs=1e-6)
    assert 0.0 <= r["altura_minimo_sobre_apice_mm"] <= 10.0


def test_margen_dientes_por_defecto_es_1_5_mm():
    """Verifica R-013: el margen a los dientes por defecto es 1,5 mm (RP-001)."""
    assert MARGEN_DIENTES_POR_DEFECTO_MM == 1.5


def test_semaforo_global_es_rojo_si_alguna_estructura_es_roja():
    """Verifica R-013: canal verde (2,5 mm) y dientes rojos (1,25 mm) dan semáforo global rojo (caso 007)."""
    implante, estructuras, _ = _cargar_caso(CASOS_DORADOS / "caso_007_diente_rojo")
    r = evaluar_plan(implante, estructuras)
    assert r["estructuras"]["canal"]["semaforo"] == "verde"
    assert r["estructuras"]["dientes"]["semaforo"] == "rojo"
    assert r["semaforo"] == "rojo"
    assert r["estructuras"]["dientes"]["margen_mm"] == 1.5


def test_recorte_bajo_plataforma_deja_un_solido_cerrado_bajo_el_plano():
    """Verifica R-013: el diente recortado queda cerrado y no tiene nada sobre el plano de la plataforma."""
    import numpy as np
    import vtk
    from vtk.util import numpy_support

    from nucleo_dental.medicion import recortar_bajo_plataforma

    implante = _implante_caso_001()                       # plataforma en z = 14
    diente = leer_stl(CASOS_DORADOS / "diente_006.stl")   # z de 5 a 25
    recortado = recortar_bajo_plataforma(diente, implante)

    puntos = numpy_support.vtk_to_numpy(recortado.GetPoints().GetData())
    assert puntos[:, 2].max() == pytest.approx(14.0, abs=1e-6)
    assert puntos[:, 2].min() == pytest.approx(5.0, abs=1e-6)
    bordes = vtk.vtkFeatureEdges()
    bordes.SetInputData(recortado)
    bordes.BoundaryEdgesOn()
    bordes.NonManifoldEdgesOn()
    bordes.FeatureEdgesOff()
    bordes.ManifoldEdgesOff()
    bordes.Update()
    assert bordes.GetOutput().GetNumberOfCells() == 0


def test_recorte_bajo_plataforma_de_un_diente_entero_sobre_ella_queda_vacio():
    """Verifica R-013: si todo el diente está sobre la plataforma, el recorte queda vacío (no se evalúa)."""
    from nucleo_dental.medicion import recortar_bajo_plataforma

    alto = Implante(diametro=6.0, largo=5.0, apice=[0, 7, 20], eje=[0, 0, 1]).como_malla()  # z de 20 a 25
    assert recortar_bajo_plataforma(alto, _implante_caso_001()) is None


def test_dentro_o_fuera_es_correcto_junto_al_borde_del_recorte():
    """Verifica R-013 y RG-014: junto al borde vivo que deja el recorte, cada punto se clasifica bien.

    El diente 006 recortado es un cilindro de radio 3 con centro en (0, 7), de
    z = 5 a z = 14: dentro/fuera se conoce exactamente. El signo de la
    distancia basado en normales se equivocaba justo en este borde.
    """
    import numpy as np

    from nucleo_dental.medicion import _distancia_con_signo, recortar_bajo_plataforma

    recortado = recortar_bajo_plataforma(leer_stl(CASOS_DORADOS / "diente_006.stl"), _implante_caso_001())
    rng = np.random.default_rng(0)
    angulo = rng.uniform(0, 2 * np.pi, 400)
    radio = rng.choice([2.5, 2.8, 3.2, 3.5], 400)
    z = rng.choice([13.5, 13.8, 14.2, 14.5], 400)
    puntos = np.column_stack([radio * np.cos(angulo), 7 + radio * np.sin(angulo), z])
    dentro_real = (radio < 3) & (z < 14)

    signo = _distancia_con_signo(recortado)(puntos)
    assert np.array_equal(signo < 0, dentro_real)


def test_evaluar_plan_exige_al_menos_una_estructura():
    """Verifica R-013: sin estructuras no hay nada que evaluar; es error, no verde."""
    with pytest.raises(ValueError, match="estructura"):
        evaluar_plan(_implante_caso_001(), {})


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


def _stl_con_registros_alterados(tmp_path, nombre, campo, valor, cada=7):
    """Copia canal_recto.stl alterando `campo` ('normal' o 'v') en uno de cada `cada` triángulos."""
    import numpy as np

    registro = np.dtype([("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    datos = bytearray(CANAL_RECTO.read_bytes())
    n = int.from_bytes(datos[80:84], "little")
    triangulos = np.frombuffer(datos, registro, n, 84).copy()
    triangulos[campo][::cada] = valor
    datos[84:] = triangulos.tobytes()
    destino = tmp_path / nombre
    destino.write_bytes(bytes(datos))
    return destino


def test_stl_con_normales_no_finitas_se_lee_igual(tmp_path):
    """Verifica R-004: las normales guardadas en el STL se ignoran; un STL con normales NaN mide igual.

    Caso real: los modelos de mandíbula exportados para el caso de prueba traen
    ~4 % de normales NaN y el lector de VTK los rechazaba.
    """
    canal = leer_stl(_stl_con_registros_alterados(tmp_path, "nan.stl", "normal", float("nan")))
    original = leer_stl(CANAL_RECTO)
    assert canal.GetNumberOfPoints() == original.GetNumberOfPoints()
    assert canal.GetNumberOfCells() == original.GetNumberOfCells()
    r = medir(_implante_caso_001(), canal)
    assert r["distancia_mm"] == pytest.approx(2.5, abs=0.05)
    assert r["semaforo"] == "verde"


def test_stl_con_vertices_no_finitos_se_rechaza(tmp_path):
    """Verifica R-004: un vértice NaN no se puede ignorar sin cambiar la geometría; es error de entrada."""
    ruta = _stl_con_registros_alterados(tmp_path, "vertice_nan.stl", "v", float("nan"), cada=1000)
    with pytest.raises(ValueError, match="no finitos"):
        leer_stl(ruta)


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

    implante, estructuras, esperado = _cargar_caso(carpeta)
    invertidas = {}
    for nombre, (malla, margen) in estructuras.items():
        invertir = vtk.vtkReverseSense()
        invertir.SetInputData(malla)
        invertir.ReverseCellsOn()
        invertir.Update()
        invertidas[nombre] = (invertir.GetOutput(), margen)

    comparar(evaluar_plan(implante, invertidas), esperado)


@pytest.mark.parametrize("margen", [-1.0, float("nan")], ids=["negativo", "nan"])
def test_margen_invalido_se_rechaza(margen):
    """Verifica R-006: el margen debe ser un número mayor o igual que 0."""
    with pytest.raises(ValueError, match="margen"):
        medir(_implante_caso_001(), leer_stl(CANAL_RECTO), margen)


@pytest.mark.parametrize("carpeta", CARPETAS_DORADAS, ids=lambda c: c.name)
def test_caso_dorado(carpeta):
    """Verifica R-002, R-004, R-005, R-006, R-009 y R-013 contra los resultados calculados a mano."""
    implante, estructuras, esperado = _cargar_caso(carpeta)
    comparar(evaluar_plan(implante, estructuras), esperado)
