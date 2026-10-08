"""Comando de terminal `nucleo-dental` (R-007, R-008).

    nucleo-dental medir --caso ruta/caso.json
    nucleo-dental medir --canal canal.stl --diametro 4.1 --largo 10 \\
                        --apice x,y,z --eje dx,dy,dz [--margen 2.0]
    nucleo-dental registrar --escaneo escaneo.stl --dientes dientes.stl \\
                            [--punto x,y,z] [--salida escaneo_registrado.stl]

Imprime un JSON con el resultado, los parámetros y la trazabilidad.
Códigos de salida de `medir`: 0 = verde, 2 = rojo, 1 = error de entrada.
Códigos de salida de `registrar`: 0 = registrado, 1 = error de entrada.
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

from vtk.util import numpy_support

from nucleo_dental.implante import Implante
from nucleo_dental.medicion import (MARGEN_DIENTES_POR_DEFECTO_MM, MARGEN_HUESO_POR_DEFECTO_MM,
                                    MARGEN_POR_DEFECTO_MM, evaluar_plan, leer_stl)

CODIGO_VERDE = 0
CODIGO_ERROR_ENTRADA = 1
CODIGO_ROJO = 2

PARAMETROS_SUELTOS = ("canal", "diametro", "largo", "apice", "eje", "implante_stl", "apice_hacia", "margen",
                      "dientes", "margen_dientes", "hueso", "margen_hueso")

# Estructuras opcionales además del canal: nombre -> margen por defecto (mm).
ESTRUCTURAS_OPCIONALES = {"dientes": MARGEN_DIENTES_POR_DEFECTO_MM, "hueso": MARGEN_HUESO_POR_DEFECTO_MM}

# Diferencia admitida entre las medidas declaradas y las del STL del implante
# (R-012, decisión clínica 2026-10-08).
TOLERANCIA_MEDIDAS_IMPLANTE_MM = 0.1


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
        args = _crear_parser().parse_args(_unir_vectores_negativos(sys.argv[1:] if argv is None else argv))
        if args.comando is None:
            raise ErrorEntrada("falta el subcomando. Uso: nucleo-dental medir --caso ruta/caso.json "
                               "o nucleo-dental registrar --escaneo … --dientes …")
        salida = _registrar(args) if args.comando == "registrar" else _medir(args)
    except (ErrorEntrada, ValueError) as error:
        print(f"Error de entrada: {error}", file=sys.stderr)
        return CODIGO_ERROR_ENTRADA

    print(json.dumps(salida, ensure_ascii=False, indent=2))
    if args.comando == "registrar":
        return CODIGO_VERDE
    return CODIGO_VERDE if salida["resultado"]["semaforo"] == "verde" else CODIGO_ROJO


def _unir_vectores_negativos(argv: list[str]) -> list[str]:
    """Convierte `--apice -30,-40,4` en `--apice=-30,-40,4`.

    argparse toma un valor que empieza con '-' por otra opción. En LPS las
    coordenadas negativas son habituales, así que se unen al flag.
    """
    resultado, i = [], 0
    while i < len(argv):
        if argv[i] in ("--apice", "--eje", "--punto") and i + 1 < len(argv) and argv[i + 1].startswith("-"):
            resultado.append(f"{argv[i]}={argv[i + 1]}")
            i += 2
        else:
            resultado.append(argv[i])
            i += 1
    return resultado


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
    m.add_argument("--implante-stl", help="STL del implante planificado (cilindro, LPS): reemplaza --apice y --eje; "
                                          "--diametro y --largo, si se indican, se contrastan con el STL.")
    m.add_argument("--apice-hacia", help="Con --implante-stl: 'abajo' (ápice inferior, mandíbula) o 'arriba' "
                                         "(ápice superior, maxilar).")
    m.add_argument("--margen", help=f"Margen al canal en mm (por defecto {MARGEN_POR_DEFECTO_MM}).")
    m.add_argument("--dientes", help="STL cerrado de los dientes (p. ej. segmentación de la arcada), en LPS: "
                                     "agrega la distancia a los dientes al semáforo.")
    m.add_argument("--margen-dientes", help=f"Margen a los dientes en mm (por defecto {MARGEN_DIENTES_POR_DEFECTO_MM}).")
    m.add_argument("--hueso", help="STL cerrado del hueso (p. ej. Mandible.stl), en LPS: agrega el espesor óseo "
                                   "mínimo alrededor de las paredes laterales del implante al semáforo.")
    m.add_argument("--margen-hueso", help=f"Espesor óseo mínimo en mm (por defecto {MARGEN_HUESO_POR_DEFECTO_MM}).")
    m.add_argument("--cbct", help="Carpeta con la serie DICOM del CBCT: verifica que canal e implante "
                                  "caigan dentro del volumen (detecta RAS/LPS mezclados). Requiere SimpleITK.")

    r = sub.add_parser("registrar", help="Refina el registro del escaneo intraoral sobre los dientes del CBCT.")
    r.add_argument("--escaneo", help="STL del escaneo intraoral, ya alineado aproximadamente (p. ej. por puntos en Slicer), en LPS.")
    r.add_argument("--dientes", help="STL de los dientes segmentados del CBCT, en LPS.")
    r.add_argument("--punto", help="Punto x,y,z (LPS) donde informar cuánto mueve la corrección, p. ej. el ápice.")
    r.add_argument("--salida", help="Ruta donde guardar el escaneo registrado (STL).")
    return parser


def _registrar(args) -> dict:
    """Subcomando `registrar` (R-015)."""
    from nucleo_dental.registro import refinar_registro, transformar_malla

    faltan = [f"--{n}" for n in ("escaneo", "dientes") if getattr(args, n) is None]
    if faltan:
        raise ErrorEntrada("faltan parámetros: " + ", ".join(faltan))
    ruta_escaneo, ruta_dientes = Path(args.escaneo), Path(args.dientes)
    punto = _vector(args.punto, "--punto") if args.punto is not None else None
    escaneo = leer_stl(ruta_escaneo)
    registro = refinar_registro(escaneo, leer_stl(ruta_dientes), punto=punto)

    salida = {
        "registro": registro,
        "parametros": {"escaneo": str(ruta_escaneo), "dientes": str(ruta_dientes), "punto": punto,
                       "sistema_coordenadas": "LPS", "unidades": "mm"},
        "trazabilidad": _trazabilidad([ruta_escaneo, ruta_dientes]),
    }
    if args.salida is not None:
        import vtk

        ruta_salida = Path(args.salida)
        escritor = vtk.vtkSTLWriter()
        escritor.SetInputData(transformar_malla(escaneo, registro["matriz_lps"]))
        escritor.SetFileName(str(ruta_salida))
        escritor.SetFileTypeToBinary()
        if not escritor.Write():
            raise ErrorEntrada(f"no se pudo escribir {ruta_salida}")
        salida["escaneo_registrado"] = {"ruta": str(ruta_salida.resolve()), "sha256": _sha256(ruta_salida)}
    return salida


def _medir(args) -> dict:
    sueltos = [n for n in PARAMETROS_SUELTOS if getattr(args, n) is not None]
    if args.caso is not None:
        if sueltos:
            raise ErrorEntrada("--caso no se combina con parámetros sueltos ("
                               + ", ".join(f"--{n.replace('_', '-')}" for n in sueltos) + ")")
        ruta_caso = Path(args.caso)
        caso = _leer_caso(ruta_caso)
        ruta_canal = (ruta_caso.parent / caso["canal"])
        especificacion = dict(caso["implante"])
        if "stl" in especificacion:
            especificacion["stl"] = str(ruta_caso.parent / especificacion["stl"])
        margen = caso["margen"]
        opcionales = {n: (ruta_caso.parent / caso[n], caso.get(f"margen_{n}", defecto))
                      for n, defecto in ESTRUCTURAS_OPCIONALES.items() if n in caso}
        archivos = [ruta_caso, ruta_canal]
    else:
        if not sueltos:
            raise ErrorEntrada("indica --caso ruta/caso.json o los parámetros sueltos (--canal y el implante: "
                               "--diametro --largo --apice --eje, o --implante-stl --apice-hacia)")
        especificacion = _implante_desde_argumentos(args)
        if args.canal is None:
            raise ErrorEntrada("faltan parámetros: --canal")
        ruta_canal = Path(args.canal)
        margen = MARGEN_POR_DEFECTO_MM if args.margen is None else _numero(args.margen, "--margen")
        opcionales = _opcionales_desde_argumentos(args)
        archivos = [ruta_canal]

    implante, implante_dict = _construir_implante(especificacion)
    if "stl" in implante_dict:
        archivos.append(Path(implante_dict["stl"]))
    estructuras = {"canal": (leer_stl(ruta_canal), margen)}
    for nombre, (ruta, margen_opcional) in opcionales.items():
        estructuras[nombre] = (leer_stl(ruta), margen_opcional)
        archivos.append(ruta)

    cbct = None
    if args.cbct is not None:
        from nucleo_dental.cbct import exigir_dentro_del_volumen, leer_cbct

        volumen = leer_cbct(args.cbct)
        vertices = {nombre: numpy_support.vtk_to_numpy(malla.GetPoints().GetData())
                    for nombre, (malla, _) in estructuras.items()}
        exigir_dentro_del_volumen(volumen, vertices, implante)
        cbct = volumen.descripcion()

    parametros = {"canal": str(ruta_canal), "implante": implante_dict, "margen": margen}
    for nombre, (ruta, margen_opcional) in opcionales.items():
        parametros.update({nombre: str(ruta), f"margen_{nombre}": margen_opcional})
    parametros.update({"sistema_coordenadas": "LPS", "unidades": "mm"})

    salida = {
        "resultado": evaluar_plan(implante, estructuras),
        "parametros": parametros,
        "trazabilidad": _trazabilidad(archivos),
    }
    if cbct is not None:
        salida["cbct"] = cbct
    return salida


def _opcionales_desde_argumentos(args) -> dict:
    """{nombre: (ruta, margen)} de las estructuras opcionales indicadas con --dientes, --hueso, etc."""
    opcionales = {}
    for nombre, defecto in ESTRUCTURAS_OPCIONALES.items():
        ruta, margen = getattr(args, nombre), getattr(args, f"margen_{nombre}")
        if margen is not None and ruta is None:
            raise ErrorEntrada(f"--margen-{nombre} solo se usa junto con --{nombre}")
        if ruta is not None:
            opcionales[nombre] = (Path(ruta), defecto if margen is None else _numero(margen, f"--margen-{nombre}"))
    return opcionales


def _implante_desde_argumentos(args) -> dict:
    """Especificación del implante a partir de los parámetros sueltos de la línea de comandos."""
    if args.implante_stl is None and args.apice_hacia is None:
        faltan = [f"--{n}" for n in ("canal", "diametro", "largo", "apice", "eje") if getattr(args, n) is None]
        if faltan:
            raise ErrorEntrada("faltan parámetros: " + ", ".join(faltan))
        return {"diametro": _numero(args.diametro, "--diametro"), "largo": _numero(args.largo, "--largo"),
                "apice": _vector(args.apice, "--apice"), "eje": _vector(args.eje, "--eje")}

    if args.implante_stl is None:
        raise ErrorEntrada("--apice-hacia solo se usa junto con --implante-stl")
    mezclados = [f"--{n}" for n in ("apice", "eje") if getattr(args, n) is not None]
    if mezclados:
        raise ErrorEntrada("--implante-stl no se combina con " + ", ".join(mezclados)
                           + ": la posición y el eje salen del STL")
    if args.apice_hacia is None:
        raise ErrorEntrada("falta --apice-hacia (abajo o arriba): el STL de un cilindro no dice cuál extremo es el ápice")
    especificacion = {"stl": args.implante_stl, "apice_hacia": args.apice_hacia}
    for nombre in ("diametro", "largo"):
        if getattr(args, nombre) is not None:
            especificacion[nombre] = _numero(getattr(args, nombre), f"--{nombre}")
    return especificacion


def _construir_implante(especificacion: dict) -> tuple[Implante, dict]:
    """Implante y su descripción para la salida, desde parámetros o desde un STL (R-012)."""
    if "stl" not in especificacion:
        implante = Implante(especificacion["diametro"], especificacion["largo"],
                            especificacion["apice"], especificacion["eje"])
        return implante, especificacion

    if "apice_hacia" not in especificacion:
        raise ErrorEntrada("al implante desde STL le falta 'apice_hacia' (abajo o arriba)")
    implante = Implante.desde_malla(leer_stl(especificacion["stl"]), especificacion["apice_hacia"])
    descripcion = {
        "stl": especificacion["stl"],
        "apice_hacia": especificacion["apice_hacia"],
        "diametro": implante.diametro,
        "largo": implante.largo,
        "apice": implante.apice.tolist(),
        "eje": implante.eje.tolist(),
    }
    for nombre, medido in (("diametro", implante.diametro), ("largo", implante.largo)):
        if nombre in especificacion:
            declarado = float(especificacion[nombre])
            descripcion[f"{nombre}_declarado"] = declarado
            if abs(declarado - medido) > TOLERANCIA_MEDIDAS_IMPLANTE_MM:
                raise ErrorEntrada(
                    f"el {nombre} declarado ({declarado} mm) difiere del medido en el STL ({medido:.3f} mm) "
                    f"en más de {TOLERANCIA_MEDIDAS_IMPLANTE_MM} mm; revisa el implante elegido o el STL")
    return implante, descripcion


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
    requeridos = ("stl", "apice_hacia") if "stl" in caso["implante"] else ("diametro", "largo", "apice", "eje")
    faltan = [c for c in requeridos if c not in caso["implante"]]
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
