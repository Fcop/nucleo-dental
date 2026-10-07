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
