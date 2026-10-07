"""Comando de terminal `nucleo-dental` (R-007, R-008).

    nucleo-dental medir --caso ruta/caso.json
    nucleo-dental medir --canal canal.stl --diametro 4.1 --largo 10 \\
                        --apice x,y,z --eje dx,dy,dz [--margen 2.0]

Imprime un JSON con el resultado, los parámetros y la trazabilidad.
Códigos de salida: 0 = verde, 2 = rojo, 1 = error de entrada.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from nucleo_dental.implante import Implante
from nucleo_dental.medicion import MARGEN_POR_DEFECTO_MM, leer_stl, medir

CODIGO_VERDE = 0
CODIGO_ERROR_ENTRADA = 1
CODIGO_ROJO = 2

PARAMETROS_SUELTOS = ("canal", "diametro", "largo", "apice", "eje", "margen")


class ErrorEntrada(Exception):
    """Error en los datos o argumentos que entrega quien usa el comando."""


class _Parser(argparse.ArgumentParser):
    # argparse sale con código 2 ante un error de argumentos; aquí 2 significa
    # semáforo rojo (R-008), así que todo error de argumentos debe ser 1.
    def error(self, message):
        raise ErrorEntrada(f"argumentos inválidos: {message}")


def main(argv=None) -> int:
    for flujo in (sys.stdout, sys.stderr):
        flujo.reconfigure(encoding="utf-8")
    try:
        args = _crear_parser().parse_args(argv)
        if args.comando is None:
            raise ErrorEntrada("falta el subcomando. Uso: nucleo-dental medir --caso ruta/caso.json")
        salida = _medir(args)
    except (ErrorEntrada, ValueError) as error:
        print(f"Error de entrada: {error}", file=sys.stderr)
        return CODIGO_ERROR_ENTRADA

    print(json.dumps(salida, ensure_ascii=False, indent=2))
    return CODIGO_VERDE if salida["resultado"]["semaforo"] == "verde" else CODIGO_ROJO


def _crear_parser() -> argparse.ArgumentParser:
    parser = _Parser(prog="nucleo-dental", description="Núcleo de planificación de implantes guiados.")
    sub = parser.add_subparsers(dest="comando", parser_class=_Parser)
    m = sub.add_parser("medir", help="Distancia mínima implante–canal mandibular y semáforo.")
    m.add_argument("--caso", help="caso.json con canal, implante y margen.")
    m.add_argument("--canal", help="STL cerrado del canal mandibular, en LPS y mm.")
    m.add_argument("--diametro", help="Diámetro del implante en mm.")
    m.add_argument("--largo", help="Largo del implante en mm.")
    m.add_argument("--apice", help="Ápice x,y,z en LPS y mm.")
    m.add_argument("--eje", help="Eje dx,dy,dz del ápice a la plataforma.")
    m.add_argument("--margen", help=f"Margen en mm (por defecto {MARGEN_POR_DEFECTO_MM}).")
    return parser


def _medir(args) -> dict:
    sueltos = [n for n in PARAMETROS_SUELTOS if getattr(args, n) is not None]
    if args.caso is not None:
        if sueltos:
            raise ErrorEntrada("--caso no se combina con parámetros sueltos (" + ", ".join(f"--{n}" for n in sueltos) + ")")
        ruta_caso = Path(args.caso)
        caso = _leer_caso(ruta_caso)
        ruta_canal = (ruta_caso.parent / caso["canal"])
        implante_dict = caso["implante"]
        margen = caso["margen"]
        archivos = [ruta_caso, ruta_canal]
    else:
        if not sueltos:
            raise ErrorEntrada("indica --caso ruta/caso.json o los parámetros sueltos (--canal, --diametro, --largo, --apice, --eje)")
        faltan = [f"--{n}" for n in PARAMETROS_SUELTOS[:-1] if getattr(args, n) is None]
        if faltan:
            raise ErrorEntrada("faltan parámetros: " + ", ".join(faltan))
        ruta_canal = Path(args.canal)
        implante_dict = {
            "diametro": _numero(args.diametro, "--diametro"),
            "largo": _numero(args.largo, "--largo"),
            "apice": _vector(args.apice, "--apice"),
            "eje": _vector(args.eje, "--eje"),
        }
        margen = MARGEN_POR_DEFECTO_MM if args.margen is None else _numero(args.margen, "--margen")
        archivos = [ruta_canal]

    implante = Implante(implante_dict["diametro"], implante_dict["largo"],
                        implante_dict["apice"], implante_dict["eje"])
    resultado = medir(implante, leer_stl(ruta_canal), margen)

    return {
        "resultado": resultado,
        "parametros": {
            "canal": str(ruta_canal),
            "implante": implante_dict,
            "margen": margen,
            "sistema_coordenadas": "LPS",
            "unidades": "mm",
        },
        "trazabilidad": _trazabilidad(archivos),
    }


def _leer_caso(ruta: Path) -> dict:
    if not ruta.is_file():
        raise ErrorEntrada(f"No existe el archivo de caso: {ruta}")
    try:
        caso = json.loads(ruta.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ErrorEntrada(f"{ruta} no es un JSON válido: {error}") from None
    faltan = [c for c in ("canal", "implante", "margen") if c not in caso]
    if faltan:
        raise ErrorEntrada(f"{ruta} no tiene los campos: {', '.join(faltan)}")
    faltan = [c for c in ("diametro", "largo", "apice", "eje") if c not in caso["implante"]]
    if faltan:
        raise ErrorEntrada(f"{ruta}: al implante le faltan los campos: {', '.join(faltan)}")
    return caso


def _numero(texto: str, nombre: str) -> float:
    try:
        return float(texto)
    except ValueError:
        raise ErrorEntrada(f"{nombre} debe ser un número (se recibió '{texto}')") from None


def _vector(texto: str, nombre: str) -> list[float]:
    partes = texto.split(",")
    if len(partes) != 3:
        raise ErrorEntrada(f"{nombre} debe tener 3 números separados por coma, p. ej. 0,0,4 (se recibió '{texto}')")
    return [_numero(p.strip(), nombre) for p in partes]


def _trazabilidad(archivos: list[Path]) -> dict:
    commit, cambios = _estado_git()
    return {
        "version_motor": _version(),
        "commit_git": commit,
        "cambios_sin_commit": cambios,
        "archivos_entrada": [{"ruta": str(a.resolve()), "sha256": _sha256(a)} for a in archivos if a.is_file()],
        "fecha_hora_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _version() -> str:
    try:
        return version("nucleo-dental")
    except PackageNotFoundError:
        return "desconocida"


def _estado_git() -> tuple[str, bool | None]:
    """Commit del código que se está ejecutando y si hay cambios sin commit."""
    carpeta = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=carpeta, capture_output=True,
                                text=True, timeout=10, check=True).stdout.strip()
        estado = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=carpeta,
                                capture_output=True, text=True, timeout=10, check=True).stdout
        return commit, bool(estado.strip())
    except (OSError, subprocess.SubprocessError):
        return "desconocido", None


def _sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with ruta.open("rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


if __name__ == "__main__":
    sys.exit(main())
