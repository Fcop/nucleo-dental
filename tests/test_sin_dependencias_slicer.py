"""Protege el núcleo contra dependencias de 3D Slicer."""

import ast
from pathlib import Path

import pytest

RAIZ_SRC = Path(__file__).resolve().parents[1] / "src"
MODULOS_PROHIBIDOS = {"slicer", "qt", "ctk"}
ARCHIVOS = sorted(RAIZ_SRC.rglob("*.py"))


def _modulos_importados(arbol: ast.AST) -> set[str]:
    nombres = set()
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Import):
            nombres.update(alias.name.split(".")[0].lower() for alias in nodo.names)
        elif isinstance(nodo, ast.ImportFrom) and nodo.module and nodo.level == 0:
            nombres.add(nodo.module.split(".")[0].lower())
    return nombres


def test_hay_archivos_que_revisar():
    """Verifica R-001: el test recorre efectivamente el código de src/."""
    assert ARCHIVOS, f"No se encontraron .py en {RAIZ_SRC}"


@pytest.mark.parametrize("archivo", ARCHIVOS, ids=lambda p: str(p.relative_to(RAIZ_SRC)))
def test_sin_dependencias_slicer(archivo):
    """Verifica R-001: src/ no importa slicer, qt, ctk ni contiene "mrml"."""
    texto = archivo.read_text(encoding="utf-8")

    prohibidos = _modulos_importados(ast.parse(texto)) & MODULOS_PROHIBIDOS
    assert not prohibidos, f"{archivo.name} importa {sorted(prohibidos)}"

    assert "mrml" not in texto.lower(), f'{archivo.name} contiene "mrml"'
