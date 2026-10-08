"""Lectura y comparación de los casos dorados (compartida por los tests).

Los esperado.json escritos antes de R-013 son planos y describen solo el
canal; los nuevos tienen un bloque por estructura ("canal", "dientes",
"hueso") y el semáforo global. Aquí se leen todos en el mismo formato, sin
modificarlos.
"""

import json
from pathlib import Path

import pytest

CASOS_DORADOS = Path(__file__).parent / "casos_dorados"
CARPETAS_DORADAS = sorted(p.parent for p in CASOS_DORADOS.glob("*/esperado.json"))
CAMPOS_ESTRUCTURA = ("distancia_mm", "colision", "penetracion_mm", "semaforo")
ESTRUCTURAS = ("canal", "dientes", "hueso")


def leer_caso(carpeta: Path) -> dict:
    return json.loads((carpeta / "caso.json").read_text(encoding="utf-8"))


def leer_esperado(carpeta: Path) -> dict:
    """{"estructuras": {nombre: {campos}}, "semaforo", "codigo_salida", "tolerancia_mm"}."""
    crudo = json.loads((carpeta / "esperado.json").read_text(encoding="utf-8"))
    if "canal" in crudo:
        estructuras = {n: crudo[n] for n in ESTRUCTURAS if n in crudo}
    else:
        estructuras = {"canal": {c: crudo[c] for c in CAMPOS_ESTRUCTURA}}
    return {"estructuras": estructuras, "semaforo": crudo["semaforo"],
            "codigo_salida": crudo["codigo_salida"], "tolerancia_mm": crudo["tolerancia_mm"]}


def comparar(resultado: dict, esperado: dict) -> None:
    """Compara cada campo esperado de cada estructura (números con tolerancia) y el semáforo global."""
    tol = esperado["tolerancia_mm"]
    assert set(resultado["estructuras"]) == set(esperado["estructuras"])
    for nombre, campos in esperado["estructuras"].items():
        obtenido = resultado["estructuras"][nombre]
        for campo, valor in campos.items():
            if isinstance(valor, bool) or isinstance(valor, str):
                assert obtenido[campo] == valor, f"{nombre}.{campo}"
            else:
                assert obtenido[campo] == pytest.approx(valor, abs=tol), f"{nombre}.{campo}"
    assert resultado["semaforo"] == esperado["semaforo"]
