"""Comando `nucleo-dental medir` ejecutado de verdad (R-007, R-008 y casos dorados)."""

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
CASOS_DORADOS = RAIZ / "tests" / "casos_dorados"
CARPETAS_DORADAS = sorted(p.parent for p in CASOS_DORADOS.glob("*/esperado.json"))
CANAL_RECTO = CASOS_DORADOS / "canal_recto.stl"

_SCRIPTS = Path(sys.executable).parent
COMANDO = next((str(p) for p in (_SCRIPTS / "nucleo-dental.exe", _SCRIPTS / "nucleo-dental") if p.exists()), None)


def ejecutar(*argumentos):
    """Ejecuta el comando instalado y devuelve (código de salida, JSON de salida o None, stderr)."""
    if COMANDO is None:
        pytest.fail("No se encontró el comando nucleo-dental; ejecuta pip install -e .[dev]")
    proceso = subprocess.run([COMANDO, *map(str, argumentos)], capture_output=True, text=True,
                             encoding="utf-8", cwd=RAIZ, timeout=120)
    salida = json.loads(proceso.stdout) if proceso.stdout.strip() else None
    return proceso.returncode, salida, proceso.stderr


def _sha256(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


@pytest.mark.parametrize("carpeta", CARPETAS_DORADAS, ids=lambda c: c.name)
def test_caso_dorado_con_el_comando(carpeta):
    """Verifica R-002, R-004, R-005, R-006, R-008 y R-009 con el comando real sobre cada caso dorado."""
    esperado = json.loads((carpeta / "esperado.json").read_text(encoding="utf-8"))
    codigo, salida, stderr = ejecutar("medir", "--caso", carpeta / "caso.json")
    assert salida is not None, stderr
    r = salida["resultado"]
    tol = esperado["tolerancia_mm"]

    assert r["distancia_mm"] == pytest.approx(esperado["distancia_mm"], abs=tol)
    assert r["colision"] is esperado["colision"]
    assert r["penetracion_mm"] == pytest.approx(esperado["penetracion_mm"], abs=tol)
    assert r["semaforo"] == esperado["semaforo"]
    assert codigo == esperado["codigo_salida"]


def test_trazabilidad():
    """Verifica R-007: la salida incluye versión, commit, parámetros, SHA-256 de cada entrada y fecha-hora UTC."""
    caso = CASOS_DORADOS / "caso_001" / "caso.json"
    antes = datetime.now(timezone.utc)
    _, salida, stderr = ejecutar("medir", "--caso", caso)
    despues = datetime.now(timezone.utc)
    assert salida is not None, stderr
    t = salida["trazabilidad"]

    assert t["version_motor"] == version("nucleo-dental")

    commit_real = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                 cwd=RAIZ).stdout.strip()
    assert t["commit_git"] == commit_real
    assert isinstance(t["cambios_sin_commit"], bool)

    fecha = datetime.fromisoformat(t["fecha_hora_utc"])
    assert fecha.utcoffset() == timedelta(0)
    assert antes - timedelta(seconds=1) <= fecha <= despues + timedelta(seconds=1)

    sha_por_archivo = {Path(a["ruta"]).resolve(): a["sha256"] for a in t["archivos_entrada"]}
    assert sha_por_archivo == {caso.resolve(): _sha256(caso), CANAL_RECTO.resolve(): _sha256(CANAL_RECTO)}

    p = salida["parametros"]
    assert p["implante"] == {"diametro": 4.1, "largo": 10.0, "apice": [0, 0, 4.0], "eje": [0, 0, 1]}
    assert p["margen"] == 2.0
    assert p["sistema_coordenadas"] == "LPS"
    assert p["unidades"] == "mm"


def test_parametros_sueltos_equivalen_al_caso():
    """Verifica R-006 y R-008: sin --caso, con parámetros sueltos y margen por defecto, el caso 001 da 2,5 mm y verde."""
    codigo, salida, stderr = ejecutar(
        "medir", "--canal", CANAL_RECTO, "--diametro", 4.1, "--largo", 10,
        "--apice", "0,0,4", "--eje", "0,0,1")
    assert salida is not None, stderr
    assert salida["resultado"]["distancia_mm"] == pytest.approx(2.5, abs=0.05)
    assert salida["parametros"]["margen"] == 2.0
    assert codigo == 0
    assert [Path(a["ruta"]).resolve() for a in salida["trazabilidad"]["archivos_entrada"]] == [CANAL_RECTO.resolve()]


@pytest.mark.parametrize("forma", ["separado", "con_igual"])
def test_coordenadas_negativas(forma):
    """Verifica R-002: --apice y --eje aceptan coordenadas negativas, habituales en LPS.

    Caso 001 con el eje invertido escrito como 0,0,-1: el implante atraviesa el canal (rojo, 2).
    """
    eje = ["--eje", "0,0,-1"] if forma == "separado" else ["--eje=0,0,-1"]
    apice = ["--apice", "-0,0,4"] if forma == "separado" else ["--apice=-0,0,4"]
    codigo, salida, stderr = ejecutar(
        "medir", "--canal", CANAL_RECTO, "--diametro", 4.1, "--largo", 10, *apice, *eje)
    assert salida is not None, stderr
    assert salida["parametros"]["implante"]["eje"] == [0, 0, -1]
    assert salida["resultado"]["colision"] is True
    assert codigo == 2


def _escribir_implante_stl(ruta: Path, diametro=4.1, largo=10.0, apice=(0, 0, 4.0), eje=(0, 0, 1)) -> Path:
    import vtk

    from nucleo_dental.implante import Implante

    escritor = vtk.vtkSTLWriter()
    escritor.SetInputData(Implante(diametro, largo, list(apice), list(eje)).como_malla())
    escritor.SetFileName(str(ruta))
    escritor.SetFileTypeToBinary()
    escritor.Write()
    return ruta


def test_implante_stl_equivale_a_parametros(tmp_path):
    """Verifica R-012 y R-007: el implante del caso 001 leído desde su STL da 2,5 mm y verde, y el STL queda trazado."""
    stl = _escribir_implante_stl(tmp_path / "implante.stl")
    codigo, salida, stderr = ejecutar("medir", "--canal", CANAL_RECTO, "--implante-stl", stl, "--apice-hacia", "abajo")
    assert salida is not None, stderr
    assert salida["resultado"]["distancia_mm"] == pytest.approx(2.5, abs=0.05)
    assert codigo == 0

    imp = salida["parametros"]["implante"]
    assert imp["stl"] == str(stl)
    assert imp["apice_hacia"] == "abajo"
    assert imp["diametro"] == pytest.approx(4.1, abs=1e-4)
    assert imp["largo"] == pytest.approx(10.0, abs=1e-4)
    assert imp["apice"] == pytest.approx([0, 0, 4.0], abs=1e-4)
    rutas = {Path(a["ruta"]).resolve(): a["sha256"] for a in salida["trazabilidad"]["archivos_entrada"]}
    assert rutas[stl.resolve()] == _sha256(stl)


def test_implante_stl_contrasta_medidas_declaradas(tmp_path):
    """Verifica R-012: medidas declaradas que difieren del STL en más de 0,1 mm son error de entrada (1).

    Reproduce el error del caso real: se declaró largo 10 para un implante de 8 mm.
    """
    stl = _escribir_implante_stl(tmp_path / "implante8.stl", diametro=4.0, largo=8.0)
    base = ["medir", "--canal", CANAL_RECTO, "--implante-stl", stl, "--apice-hacia", "abajo"]

    codigo, _, stderr = ejecutar(*base, "--diametro", 4.0, "--largo", 10)
    assert codigo == 1
    assert "largo" in stderr and "10" in stderr and "8.0" in stderr

    codigo, salida, stderr = ejecutar(*base, "--diametro", 4.05, "--largo", 8.08)
    assert salida is not None, stderr
    assert codigo in (0, 2)


def test_implante_stl_desde_caso_json(tmp_path):
    """Verifica R-012: caso.json acepta {"stl", "apice_hacia"} con ruta relativa al caso."""
    _escribir_implante_stl(tmp_path / "implante.stl")
    caso = tmp_path / "caso.json"
    caso.write_text(json.dumps({"canal": str(CANAL_RECTO),
                                "implante": {"stl": "implante.stl", "apice_hacia": "abajo"},
                                "margen": 2.0}), encoding="utf-8")
    codigo, salida, stderr = ejecutar("medir", "--caso", caso)
    assert salida is not None, stderr
    assert salida["resultado"]["distancia_mm"] == pytest.approx(2.5, abs=0.05)
    assert codigo == 0


@pytest.mark.parametrize(
    "extra, texto",
    [(["--implante-stl", "IMPLANTE", "--apice", "0,0,4"], "no se combina"),
     (["--implante-stl", "IMPLANTE"], "--apice-hacia"),
     (["--implante-stl", "IMPLANTE", "--apice-hacia", "izquierda"], "abajo"),
     (["--apice-hacia", "abajo", "--diametro", "4.1", "--largo", "10", "--apice", "0,0,4", "--eje", "0,0,1"],
      "--implante-stl")],
    ids=["stl_mas_apice", "falta_apice_hacia", "apice_hacia_invalido", "apice_hacia_sin_stl"],
)
def test_implante_stl_errores_de_entrada(tmp_path, extra, texto):
    """Verifica R-008 y R-012: combinaciones inválidas con --implante-stl devuelven 1 con mensaje claro."""
    stl = str(_escribir_implante_stl(tmp_path / "implante.stl"))
    argumentos = [stl if a == "IMPLANTE" else a for a in extra]
    codigo, salida, stderr = ejecutar("medir", "--canal", CANAL_RECTO, *argumentos)
    assert codigo == 1
    assert salida is None
    assert texto in stderr


def test_codigos_salida():
    """Verifica R-008: 0 = verde, 2 = rojo."""
    assert ejecutar("medir", "--caso", CASOS_DORADOS / "caso_001" / "caso.json")[0] == 0
    assert ejecutar("medir", "--caso", CASOS_DORADOS / "caso_002" / "caso.json")[0] == 2


@pytest.mark.parametrize(
    "argumentos, texto",
    [
        (["medir", "--caso", "no_existe/caso.json"], "No existe"),
        (["medir", "--canal", "no_existe.stl", "--diametro", "4.1", "--largo", "10",
          "--apice", "0,0,4", "--eje", "0,0,1"], "No existe"),
        (["medir", "--canal", CANAL_RECTO, "--diametro", "4.1", "--largo", "10",
          "--apice", "0,0,4", "--eje", "0,0,0"], "eje"),
        (["medir", "--canal", CANAL_RECTO, "--diametro", "4.1", "--largo", "10",
          "--apice", "0,0", "--eje", "0,0,1"], "apice"),
        (["medir", "--canal", CANAL_RECTO, "--diametro", "4.1"], "faltan"),
        (["medir", "--caso", CASOS_DORADOS / "caso_001" / "caso.json", "--canal", CANAL_RECTO], "no se combina"),
        (["medir", "--canal", CANAL_RECTO, "--diametro", "4.1", "--largo", "10",
          "--apice", "0,0,4", "--eje", "0,0,1", "--margen", "-1"], "margen"),
        (["medir", "--canal", CANAL_RECTO, "--diametro", "cuatro", "--largo", "10",
          "--apice", "0,0,4", "--eje", "0,0,1"], "diametro"),
        (["medir"], "--caso"),
        ([], "medir"),
    ],
    ids=["caso_inexistente", "canal_inexistente", "eje_nulo", "apice_incompleto", "faltan_parametros",
         "caso_mas_sueltos", "margen_negativo", "diametro_no_numerico", "sin_entrada", "sin_subcomando"],
)
def test_error_de_entrada_devuelve_1(argumentos, texto, tmp_path):
    """Verifica R-008: todo error de entrada devuelve 1 (nunca 2, que significa rojo) con mensaje en español."""
    codigo, salida, stderr = ejecutar(*argumentos)
    assert codigo == 1
    assert salida is None
    assert texto.lower() in stderr.lower()


def test_caso_json_mal_formado_devuelve_1(tmp_path):
    """Verifica R-008: un caso.json ilegible o incompleto es error de entrada (1)."""
    roto = tmp_path / "caso.json"
    roto.write_text("{ esto no es json", encoding="utf-8")
    assert ejecutar("medir", "--caso", roto)[0] == 1

    incompleto = tmp_path / "incompleto.json"
    incompleto.write_text(json.dumps({"canal": str(CANAL_RECTO), "margen": 2.0}), encoding="utf-8")
    codigo, _, stderr = ejecutar("medir", "--caso", incompleto)
    assert codigo == 1
    assert "implante" in stderr
