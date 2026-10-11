"""Comando de terminal `nucleo-dental` (R-007, R-008).

    nucleo-dental medir --caso ruta/caso.json   (con "implante" o "implantes": {nombre: implante})
    nucleo-dental medir --canal canal.stl --diametro 4.1 --largo 10 \\
                        --apice x,y,z --eje dx,dy,dz [--margen 2.0]
    nucleo-dental registrar --escaneo escaneo.stl --dientes dientes.stl \\
                            [--punto x,y,z] [--salida escaneo_registrado.stl]

Imprime un JSON con el resultado, los parámetros y la trazabilidad.
Códigos de salida de `medir`: 0 = verde, 2 = rojo, 1 = error de entrada.
    nucleo-dental apoyo --escaneo escaneo.stl (--dientes dientes.stl --implante-stl i.stl
                        --apice-hacia abajo [--radio 24] [--margen-encia 1] | --curva curva.mrk.json)
                        --salida apoyo.stl

    nucleo-dental guia --escaneo escaneo.stl --dientes dientes.stl --implante-stl i.stl --apice-hacia abajo
                       --curva limites.mrk.json --plano-oclusal plano.mrk.json [--ventanas cajas.mrk.json]
                       [--fabricacion impresa]
                       [--soporte dentosoportada] [--kit oneguide] --salida guia.stl

Códigos de salida de `registrar` y `apoyo`: 0 = hecho, 1 = error de entrada.
Códigos de salida de `guia`: 0 = guía válida (se escribe --salida), 2 = guía no
válida (se escribe solo como <salida>_NO_VALIDA.stl para inspeccionarla), 1 = error de entrada.
`guia` escribe además <salida>_ajuste.vtp: la guía con la separación al escaneo por
punto (Desvio_mm), para colorearla en Slicer; el histograma va a la salida de error.
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

import numpy as np
from vtk.util import numpy_support

from nucleo_dental.implante import Implante
from nucleo_dental.pines import MARGEN_PIN_POR_DEFECTO_MM, Pin, evaluar_pin, profundidad_en_hueso
from nucleo_dental.medicion import (MARGEN_DIENTES_POR_DEFECTO_MM, MARGEN_HUESO_POR_DEFECTO_MM,
                                    MARGEN_IMPLANTES_POR_DEFECTO_MM, MARGEN_POR_DEFECTO_MM, evaluar_implantes,
                                    evaluar_plan, leer_stl)

CODIGO_VERDE = 0
CODIGO_ERROR_ENTRADA = 1
CODIGO_ROJO = 2

PARAMETROS_SUELTOS = ("canal", "diametro", "largo", "apice", "eje", "implante_stl", "apice_hacia", "margen",
                      "dientes", "margen_dientes", "hueso", "margen_hueso", "sobrefresado")

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
        salida = {"registrar": _registrar, "apoyo": _apoyo, "guia": _guia}.get(args.comando, _medir)(args)
    except (ErrorEntrada, ValueError) as error:
        print(f"Error de entrada: {error}", file=sys.stderr)
        return CODIGO_ERROR_ENTRADA

    print(json.dumps(salida, ensure_ascii=False, indent=2))
    if args.comando in ("registrar", "apoyo"):
        return CODIGO_VERDE
    if args.comando == "guia":
        return CODIGO_VERDE if salida["guia"]["valida"] else CODIGO_ROJO
    return CODIGO_VERDE if salida["resultado"]["semaforo"] == "verde" else CODIGO_ROJO


def _unir_vectores_negativos(argv: list[str]) -> list[str]:
    """Convierte `--apice -30,-40,4` en `--apice=-30,-40,4`.

    argparse toma un valor que empieza con '-' por otra opción. En LPS las
    coordenadas negativas son habituales, así que se unen al flag.
    """
    resultado, i = [], 0
    while i < len(argv):
        if argv[i] in ("--apice", "--eje", "--punto", "--punto-interior") and i + 1 < len(argv) and argv[i + 1].startswith("-"):
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
    m.add_argument("--sobrefresado", help="Lo que la fresa pasa más allá del ápice, en mm (por defecto 0); "
                                          "se suma al medir contra el canal (R-017).")
    m.add_argument("--cbct", help="Carpeta con la serie DICOM del CBCT: verifica que canal e implante "
                                  "caigan dentro del volumen (detecta RAS/LPS mezclados). Requiere SimpleITK.")

    r = sub.add_parser("registrar", help="Refina el registro del escaneo intraoral sobre los dientes del CBCT.")
    r.add_argument("--escaneo", help="STL del escaneo intraoral, ya alineado aproximadamente (p. ej. por puntos en Slicer), en LPS.")
    r.add_argument("--dientes", help="STL de los dientes segmentados del CBCT, en LPS.")
    r.add_argument("--punto", help="Punto x,y,z (LPS) donde informar cuánto mueve la corrección, p. ej. el ápice.")
    r.add_argument("--salida", help="Ruta donde guardar el escaneo registrado (STL).")

    a = sub.add_parser("apoyo", help="Región de apoyo de la guía sobre el escaneo (automática o por curva).")
    a.add_argument("--escaneo", help="STL del escaneo intraoral registrado con el CBCT, en LPS.")
    a.add_argument("--dientes", help="Modo automático: STL de los dientes segmentados del CBCT, en LPS.")
    a.add_argument("--implante-stl", help="Modo automático: STL del implante planificado (define el eje).")
    a.add_argument("--apice-hacia", help="Modo automático: 'abajo' o 'arriba' (ver medir).")
    a.add_argument("--radio", help="Modo automático: radio en mm desde el eje del implante (por defecto 24).")
    a.add_argument("--margen-encia", help="Modo automático: distancia mínima a la encía en mm (por defecto 1).")
    a.add_argument("--soporte", help="Modo automático: dentosoportada (por defecto), dentomucosoportada o "
                                      "mucosoportada (esta última no necesita --dientes).")
    a.add_argument("--curva", help="Modo curva: curva cerrada dibujada en Slicer (.mrk.json, LPS o RAS declarado).")
    a.add_argument("--punto-interior", help="Modo curva: punto x,y,z (LPS) dentro de la región; por defecto el centro de la curva.")
    a.add_argument("--salida", help="Ruta donde guardar el parche de apoyo (STL).")

    g = sub.add_parser("guia", help="Guía quirúrgica imprimible: apoyo dentro de la curva del usuario, puente, "
                                    "anillo, recorte por el eje de inserción y tolerancia (R-021).")
    g.add_argument("--escaneo", help="STL CERRADO del escaneo intraoral registrado con el CBCT, en LPS.")
    g.add_argument("--dientes", help="STL de los dientes segmentados del CBCT (no hace falta si es mucosoportada).")
    g.add_argument("--implante-stl", action="append", help="STL de un implante planificado; repetir por cada implante "
                                                              "(misma guía, mismo kit).")
    g.add_argument("--apice-hacia", help="'abajo' (mandíbula) o 'arriba' (maxilar).")
    g.add_argument("--curva", help="Límites de la guía: curva cerrada dibujada en Slicer (.mrk.json).")
    g.add_argument("--punto-interior", help="Punto x,y,z (LPS) dentro de la curva; por defecto su centro.")
    g.add_argument("--plano-oclusal", help="3 puntos del plano oclusal (.mrk.json): el eje de inserción es su normal.")
    g.add_argument("--ventanas", action="append", help="Ventanas de inspección: cajas (ROI) de Slicer (.mrk.json) "
                                                        "ubicadas y dimensionadas por el usuario; se restan de la guía. "
                                                        "Repetir por cada archivo.")
    g.add_argument("--pin-stl", action="append", help="Pin de fijación: STL de un cilindro ubicado en Slicer; repetir "
                                                       "por cada pin. Requiere --hueso (la punta es el extremo en el hueso).")
    g.add_argument("--hueso", help="STL cerrado del hueso (p. ej. Mandible.stl): con --pin-stl, define la punta del pin "
                                   "y cuánto entra en el hueso.")
    g.add_argument("--fabricacion", help="impresa (por defecto) o fresada: define ajuste del orificio y tolerancia.")
    g.add_argument("--soporte", help="dentosoportada (por defecto), dentomucosoportada o mucosoportada.")
    g.add_argument("--kit", help="Perfil del kit (por defecto oneguide).")
    g.add_argument("--salida", help="Ruta del STL de la guía.")
    return parser


def _guia(args) -> dict:
    """Subcomando `guia` (R-021 a R-025)."""
    import vtk

    from nucleo_dental.apoyo import leer_cajas_slicer, leer_puntos_slicer, region_desde_curva
    from nucleo_dental.ensamblaje import eje_desde_plano_oclusal, guia_quirurgica
    from nucleo_dental.kits import obtener_kit

    soporte = args.soporte or "dentosoportada"
    requeridos = ["escaneo", "implante_stl", "apice_hacia", "curva", "plano_oclusal", "salida"]
    if soporte != "mucosoportada":
        requeridos.append("dientes")
    faltan = [f"--{n.replace('_', '-')}" for n in requeridos if getattr(args, n) is None]
    if faltan:
        raise ErrorEntrada("faltan parámetros: " + ", ".join(faltan))
    rutas = {n: Path(getattr(args, n)) for n in ("escaneo", "curva", "plano_oclusal")}
    rutas_implantes = [Path(r) for r in args.implante_stl]
    rutas_ventanas = [Path(r) for r in (args.ventanas or [])]
    ventanas = [caja for ruta in rutas_ventanas for caja in leer_cajas_slicer(ruta)]
    ruta_dientes = Path(args.dientes) if args.dientes is not None else None
    rutas_pines = [Path(r) for r in (args.pin_stl or [])]
    if rutas_pines and args.hueso is None:
        raise ErrorEntrada("--pin-stl necesita --hueso: la punta del pin es el extremo que queda dentro del hueso")
    ruta_hueso = Path(args.hueso) if args.hueso is not None else None
    fabricacion = args.fabricacion or "impresa"
    kit = obtener_kit(args.kit or "oneguide")

    escaneo = leer_stl(rutas["escaneo"])
    implantes = [Implante.desde_malla(leer_stl(r), args.apice_hacia) for r in rutas_implantes]
    plano = leer_puntos_slicer(rutas["plano_oclusal"])
    eje = eje_desde_plano_oclusal(plano, np.mean([i.eje for i in implantes], axis=0))
    interior = _vector(args.punto_interior, "--punto-interior") if args.punto_interior is not None else None
    mascara = region_desde_curva(escaneo, leer_puntos_slicer(rutas["curva"]), interior)
    hueso = leer_stl(ruta_hueso) if ruta_hueso else None
    pines = [Pin.desde_malla(leer_stl(r), hueso) for r in rutas_pines]
    dientes = leer_stl(ruta_dientes) if ruta_dientes else None
    r = guia_quirurgica(escaneo, dientes, implantes, kit, fabricacion,
                        mascara, eje, tipo_soporte=soporte, ventanas=ventanas, pines=pines)

    ruta_salida = Path(args.salida)
    if not r["valida"]:
        ruta_salida = ruta_salida.with_name(ruta_salida.stem + "_NO_VALIDA" + ruta_salida.suffix)
    escritor = vtk.vtkSTLWriter()
    escritor.SetInputData(r["guia"])
    escritor.SetFileName(str(ruta_salida))
    escritor.SetFileTypeToBinary()
    if not escritor.Write():
        raise ErrorEntrada(f"no se pudo escribir {ruta_salida}")

    from nucleo_dental.geometria.ajuste import histograma_ajuste, mapa_de_ajuste

    ruta_mapa = ruta_salida.with_name(ruta_salida.stem + "_ajuste.vtp")
    escritor_mapa = vtk.vtkXMLPolyDataWriter()
    escritor_mapa.SetInputData(mapa_de_ajuste(r["guia"], r["ajuste"]))
    escritor_mapa.SetFileName(str(ruta_mapa))
    if not escritor_mapa.Write():
        raise ErrorEntrada(f"no se pudo escribir {ruta_mapa}")
    print(histograma_ajuste(r["ajuste"]), file=sys.stderr)
    ajuste = {k: v for k, v in r["ajuste"].items() if not k.startswith("_")}
    ajuste["mapa"] = {"ruta": str(ruta_mapa.resolve()), "sha256": _sha256(ruta_mapa),
                      "escalar": "Desvio_mm (separación − objetivo; negativo = más apretado)"}

    anillos = []
    for ruta, implante, pc in zip(rutas_implantes, implantes, r["anillos"]):
        angulo = float(np.degrees(np.arccos(np.clip(abs(np.dot(eje, implante.eje)), -1.0, 1.0))))
        anillos.append({"implante_stl": str(ruta), "angulo_eje_insercion_grados": angulo, **pc})
    por_nombre = {f"implante_{i + 1}": imp for i, imp in enumerate(implantes)}
    entre_implantes = evaluar_implantes(por_nombre) if len(implantes) > 1 else None
    informe_pines = []
    for ruta, pin, geometria in zip(rutas_pines, pines, r["pines"]):
        seguridad = evaluar_pin(pin, {"dientes": dientes} if dientes is not None else {}, por_nombre)
        informe_pines.append({"pin_stl": str(ruta), "diametro_mm": pin.diametro, "largo_mm": pin.largo,
                              "punta": pin.punta.tolist(), "cabeza": pin.cabeza.tolist(),
                              "profundidad_en_hueso_mm": profundidad_en_hueso(pin, hueso), **geometria,
                              "seguridad": seguridad})
    archivos = [rutas["escaneo"], *rutas_implantes, rutas["curva"], rutas["plano_oclusal"], *rutas_ventanas,
                *rutas_pines, *([ruta_hueso] if ruta_hueso else [])]
    return {
        "guia": {"valida": r["valida"], "problemas": r["problemas"], "avisos": r["avisos"],
                 "metricas": r["metricas"], "ruta": str(ruta_salida.resolve()), "sha256": _sha256(ruta_salida)},
        "ajuste": ajuste,
        "geometria": {"eje_insercion": r["eje_insercion"], "tolerancia_ajuste_mm": r["tolerancia_mm"],
                      "anillos": anillos, "paredes_entre_orificios": r["paredes_entre_orificios"],
                      "ventanas": r["ventanas"], "pines": informe_pines},
        "entre_implantes": entre_implantes,
        "parametros": {**{n: str(v) for n, v in rutas.items()}, "implantes_stl": [str(x) for x in rutas_implantes],
                       "ventanas": [str(x) for x in rutas_ventanas], "pines_stl": [str(x) for x in rutas_pines],
                       "hueso": str(ruta_hueso) if ruta_hueso else None,
                       "dientes": str(ruta_dientes) if ruta_dientes else None,
                       "apice_hacia": args.apice_hacia, "punto_interior": interior, "fabricacion": fabricacion,
                       "tipo_soporte": soporte, "kit": kit.nombre, "provisionales": sorted(kit.provisionales),
                       "sistema_coordenadas": "LPS", "unidades": "mm"},
        "trazabilidad": _trazabilidad(archivos + ([ruta_dientes] if ruta_dientes else [])),
    }


def _apoyo(args) -> dict:
    """Subcomando `apoyo` (R-016)."""
    import vtk

    from nucleo_dental.apoyo import (RADIO_APOYO_POR_DEFECTO_MM, leer_puntos_slicer, parche_de_apoyo,
                                     region_automatica, region_desde_curva)

    if args.escaneo is None:
        raise ErrorEntrada("faltan parámetros: --escaneo")
    ruta_escaneo = Path(args.escaneo)
    escaneo = leer_stl(ruta_escaneo)
    archivos = [ruta_escaneo]

    if args.curva is not None:
        ruta_curva = Path(args.curva)
        puntos_curva = leer_puntos_slicer(ruta_curva)
        interior = _vector(args.punto_interior, "--punto-interior") if args.punto_interior is not None else None
        mascara = region_desde_curva(escaneo, puntos_curva, interior)
        informe = {"modo": "curva", "puntos_curva": len(puntos_curva)}
        parametros = {"curva": str(ruta_curva), "punto_interior": interior}
        archivos.append(ruta_curva)
    elif args.dientes is not None or args.soporte == "mucosoportada":
        faltan = [f"--{n.replace('_', '-')}" for n in ("implante_stl", "apice_hacia") if getattr(args, n) is None]
        if faltan:
            raise ErrorEntrada("el modo automático necesita: " + ", ".join(faltan))
        ruta_dientes = Path(args.dientes) if args.dientes is not None else None
        ruta_implante = Path(args.implante_stl)
        soporte = args.soporte or "dentosoportada"
        implante = Implante.desde_malla(leer_stl(ruta_implante), args.apice_hacia)
        radio = RADIO_APOYO_POR_DEFECTO_MM if args.radio is None else _numero(args.radio, "--radio")
        margen_encia = 1.0 if args.margen_encia is None else _numero(args.margen_encia, "--margen-encia")
        region = region_automatica(escaneo, leer_stl(ruta_dientes) if ruta_dientes else None,
                                   implante.apice, implante.eje, radio_mm=radio, margen_encia_mm=margen_encia,
                                   tipo_soporte=soporte)
        mascara = region["mascara"]
        informe = {"modo": "automatico", "altura_minima_sobre_encia_mm": region["altura_minima_sobre_encia_mm"]}
        parametros = {"dientes": str(ruta_dientes) if ruta_dientes else None, "tipo_soporte": soporte,
                      "implante_stl": str(ruta_implante),
                      "apice_hacia": args.apice_hacia, "radio_mm": radio, "margen_encia_mm": margen_encia}
        archivos += [r for r in (ruta_dientes, ruta_implante) if r is not None]
    else:
        raise ErrorEntrada("indica el modo: --curva curva.mrk.json, o --dientes con --implante-stl y --apice-hacia")

    parche = parche_de_apoyo(escaneo, np.asarray(mascara))
    masa = vtk.vtkMassProperties()
    masa.SetInputData(parche)
    masa.Update()
    conectividad = vtk.vtkPolyDataConnectivityFilter()
    conectividad.SetInputData(parche)
    conectividad.SetExtractionModeToAllRegions()
    conectividad.Update()
    informe.update({"puntos_region": int(np.count_nonzero(mascara)), "area_mm2": float(masa.GetSurfaceArea()),
                    "partes": int(conectividad.GetNumberOfExtractedRegions())})

    salida = {"apoyo": informe,
              "parametros": {"escaneo": str(ruta_escaneo), **parametros, "sistema_coordenadas": "LPS", "unidades": "mm"},
              "trazabilidad": _trazabilidad(archivos)}
    if args.salida is not None:
        ruta_salida = Path(args.salida)
        escritor = vtk.vtkSTLWriter()
        escritor.SetInputData(parche)
        escritor.SetFileName(str(ruta_salida))
        escritor.SetFileTypeToBinary()
        if not escritor.Write():
            raise ErrorEntrada(f"no se pudo escribir {ruta_salida}")
        salida["parche"] = {"ruta": str(ruta_salida.resolve()), "sha256": _sha256(ruta_salida)}
    return salida


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
        especificaciones = {}
        for nombre, datos in (caso["implantes"] if "implantes" in caso else {"implante": caso["implante"]}).items():
            especificaciones[nombre] = dict(datos)
            if "stl" in datos:
                especificaciones[nombre]["stl"] = str(ruta_caso.parent / datos["stl"])
        varios = "implantes" in caso
        margen_implantes = caso.get("margen_implantes", MARGEN_IMPLANTES_POR_DEFECTO_MM)
        especificacion_pines = {n: str(ruta_caso.parent / d["stl"]) for n, d in caso.get("pines", {}).items()}
        margen_pines = caso.get("margen_pines", MARGEN_PIN_POR_DEFECTO_MM)
        margen = caso["margen"]
        sobrefresado = caso.get("sobrefresado_mm", 0.0)
        opcionales = {n: (ruta_caso.parent / caso[n], caso.get(f"margen_{n}", defecto))
                      for n, defecto in ESTRUCTURAS_OPCIONALES.items() if n in caso}
        archivos = [ruta_caso, ruta_canal]
    else:
        if not sueltos:
            raise ErrorEntrada("indica --caso ruta/caso.json o los parámetros sueltos (--canal y el implante: "
                               "--diametro --largo --apice --eje, o --implante-stl --apice-hacia)")
        especificaciones = {"implante": _implante_desde_argumentos(args)}
        varios = False
        margen_implantes = MARGEN_IMPLANTES_POR_DEFECTO_MM
        especificacion_pines, margen_pines = {}, MARGEN_PIN_POR_DEFECTO_MM
        if args.canal is None:
            raise ErrorEntrada("faltan parámetros: --canal")
        ruta_canal = Path(args.canal)
        margen = MARGEN_POR_DEFECTO_MM if args.margen is None else _numero(args.margen, "--margen")
        sobrefresado = 0.0 if args.sobrefresado is None else _numero(args.sobrefresado, "--sobrefresado")
        opcionales = _opcionales_desde_argumentos(args)
        archivos = [ruta_canal]

    implantes, descripciones = {}, {}
    for nombre, especificacion in especificaciones.items():
        implantes[nombre], descripciones[nombre] = _construir_implante(especificacion)
        if "stl" in descripciones[nombre]:
            archivos.append(Path(descripciones[nombre]["stl"]))
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
        for implante in implantes.values():
            exigir_dentro_del_volumen(volumen, vertices, implante)
        cbct = volumen.descripcion()

    parametros = {"canal": str(ruta_canal), "margen": margen, "sobrefresado_mm": sobrefresado}
    if varios:
        parametros.update({"implantes": descripciones, "margen_implantes": margen_implantes})
    else:
        parametros["implante"] = descripciones["implante"]
    for nombre, (ruta, margen_opcional) in opcionales.items():
        parametros.update({nombre: str(ruta), f"margen_{nombre}": margen_opcional})
    parametros.update({"sistema_coordenadas": "LPS", "unidades": "mm"})

    if varios:
        # R-023: cada implante contra las estructuras y cada par entre sí; rojo si algo es rojo.
        por_implante = {n: evaluar_plan(i, estructuras, sobrefresado_mm=sobrefresado) for n, i in implantes.items()}
        entre = evaluar_implantes(implantes, margen_implantes) if len(implantes) > 1 else None
        rojo = any(r["semaforo"] == "rojo" for r in por_implante.values()) or (entre and entre["semaforo"] == "rojo")
        resultado = {"semaforo": "rojo" if rojo else "verde", "implantes": por_implante, "entre_implantes": entre}
    else:
        resultado = evaluar_plan(implantes["implante"], estructuras, sobrefresado_mm=sobrefresado)
    if especificacion_pines:
        # R-025: cada pin contra canal y dientes completos y contra los implantes, con su margen.
        if "hueso" not in estructuras:
            raise ErrorEntrada("los pines necesitan \"hueso\" en el caso: la punta es el extremo dentro del hueso")
        hueso = estructuras["hueso"][0]
        resultado["pines"] = {}
        for nombre, ruta in especificacion_pines.items():
            pin = Pin.desde_malla(leer_stl(ruta), hueso)
            archivos.append(Path(ruta))
            seguridad = evaluar_pin(pin, {n: estructuras[n][0] for n in ("canal", "dientes") if n in estructuras},
                                    implantes, margen_pines)
            resultado["pines"][nombre] = {"stl": ruta, "diametro_mm": pin.diametro, "largo_mm": pin.largo,
                                          "punta": pin.punta.tolist(), "cabeza": pin.cabeza.tolist(),
                                          "profundidad_en_hueso_mm": profundidad_en_hueso(pin, hueso), **seguridad}
            if seguridad["semaforo"] == "rojo":
                resultado["semaforo"] = "rojo"
        parametros["margen_pines"] = margen_pines
    salida = {
        "resultado": resultado,
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
    faltan = [c for c in ("canal", "margen") if c not in caso]
    if ("implante" in caso) == ("implantes" in caso):
        faltan.append("implante (uno) o implantes ({nombre: implante}, varios)")
    if faltan:
        raise ErrorEntrada(f"{ruta} no tiene los campos: {', '.join(faltan)}")
    implantes = caso["implantes"] if "implantes" in caso else {"implante": caso["implante"]}
    if not isinstance(implantes, dict) or not implantes:
        raise ErrorEntrada(f"{ruta}: 'implantes' debe ser un objeto {{nombre: implante}} con al menos uno")
    for nombre, implante in implantes.items():
        requeridos = ("stl", "apice_hacia") if "stl" in implante else ("diametro", "largo", "apice", "eje")
        faltan = [c for c in requeridos if c not in implante]
        if faltan:
            raise ErrorEntrada(f"{ruta}: al implante '{nombre}' le faltan los campos: {', '.join(faltan)}")
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
