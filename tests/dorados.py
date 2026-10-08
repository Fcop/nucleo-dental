"""Lectura de los casos dorados (compartida por los tests).

Los esperado.json escritos antes de R-013 son planos y describen solo el
canal; los nuevos tienen un bloque por estructura ("canal", "dientes") y el
semáforo global. Aquí ambos se leen en el mismo formato, sin modificarlos.
"""

import json
from pathlib import Path

CASOS_DORADOS = Path(__file__).parent / "casos_dorados"
CARPETAS_DORADAS = sorted(p.parent for p in CASOS_DORADOS.glob("*/esperado.json"))
CAMPOS_ESTRUCTURA = ("distancia_mm", "colision", "penetracion_mm", "semaforo")


def leer_caso(carpeta: Path) -> dict:
    return json.loads((carpeta / "caso.json").read_text(encoding="utf-8"))


def leer_esperado(carpeta: Path) -> dict:
    """{"estructuras": {nombre: {campos}}, "semaforo", "codigo_salida", "tolerancia_mm"}."""
    crudo = json.loads((carpeta / "esperado.json").read_text(encoding="utf-8"))
    if "canal" in crudo:
        estructuras = {n: crudo[n] for n in ("canal", "dientes") if n in crudo}
    else:
        estructuras = {"canal": {c: crudo[c] for c in CAMPOS_ESTRUCTURA}}
    return {"estructuras": estructuras, "semaforo": crudo["semaforo"],
            "codigo_salida": crudo["codigo_salida"], "tolerancia_mm": crudo["tolerancia_mm"]}
