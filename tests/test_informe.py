"""Informe del protocolo quirúrgico en PDF (R-027), recorriendo `nucleo-dental guia` de punta a punta."""

import json
import zlib

import numpy as np
import pytest
import vtk

from nucleo_dental.cli import main
from nucleo_dental.implante import Implante
from nucleo_dental.informe import escribir_pdf, lineas_del_informe


def _escribir_stl(malla, ruta):
    escritor = vtk.vtkSTLWriter()
    escritor.SetInputData(malla)
    escritor.SetFileName(str(ruta))
    escritor.SetFileTypeToBinary()
    escritor.Write()


def _marcas(ruta, tipo, puntos):
    datos = {"markups": [{"type": tipo, "coordinateSystem": "LPS",
                          "controlPoints": [{"position": list(map(float, p))} for p in puntos]}]}
    ruta.write_text(json.dumps(datos), encoding="utf-8")


@pytest.fixture(scope="module")
def corrida(tmp_path_factory):
    """Encía plana cerrada (cara superior en z = 0), un implante vertical, curva y plano oclusal; guía mucosoportada."""
    carpeta = tmp_path_factory.mktemp("informe")
    cubo = vtk.vtkCubeSource()
    cubo.SetBounds(-15, 20, -10, 10, -20, 0)
    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(cubo.GetOutputPort())
    limpio = vtk.vtkCleanPolyData()
    limpio.SetInputConnection(tri.GetOutputPort())
    sub = vtk.vtkLinearSubdivisionFilter()
    sub.SetInputConnection(limpio.GetOutputPort())
    sub.SetNumberOfSubdivisions(5)
    sub.Update()
    _escribir_stl(sub.GetOutput(), carpeta / "escaneo.stl")
    _escribir_stl(Implante(4.1, 10, [0, 0, -12], [0, 0, 1]).como_malla(), carpeta / "implante.stl")
    lado = np.linspace(0, 1, 8, endpoint=False)[:, None]
    esquinas = np.array([[-12, -8, 0], [17, -8, 0], [17, 8, 0], [-12, 8, 0]], dtype=float)
    curva = np.vstack([a + lado * (b - a) for a, b in zip(esquinas, np.roll(esquinas, -1, axis=0))])
    _marcas(carpeta / "curva.mrk.json", "ClosedCurve", curva)
    _marcas(carpeta / "plano.mrk.json", "Fiducial", [[0, 0, 5], [10, 0, 5], [0, 10, 5]])
    pdf = carpeta / "protocolo.pdf"
    argumentos = ["guia", "--escaneo", str(carpeta / "escaneo.stl"), "--implante-stl", str(carpeta / "implante.stl"),
                  "--apice-hacia", "abajo", "--curva", str(carpeta / "curva.mrk.json"),
                  "--plano-oclusal", str(carpeta / "plano.mrk.json"), "--soporte", "mucosoportada",
                  "--salida", str(carpeta / "guia.stl"), "--informe", str(pdf)]
    return carpeta, argumentos, pdf


def test_comando_guia_escribe_el_informe(corrida, capsys):
    """Verifica R-027: `guia --informe` escribe un PDF con el estado, el orificio y el largo de trabajo de la fresa."""
    carpeta, argumentos, pdf = corrida
    assert main(argumentos) == 0
    salida = json.loads(capsys.readouterr().out)
    (carpeta / "salida.json").write_text(json.dumps(salida), encoding="utf-8")
    contenido = pdf.read_bytes()
    assert contenido.startswith(b"%PDF-1.4") and contenido.rstrip().endswith(b"%%EOF")
    assert salida["informe"]["sha256"]
    # 20,80 mm: el largo de trabajo de este implante que calculó Francisco en el caso dorado 015.
    for texto in ("Protocolo quirúrgico", "VÁLIDA", "Ø 5,30 mm", "Largo de trabajo de la fresa", "20,80 mm",
                  "según el manual del kit", "sin validación clínica"):
        assert texto.encode("cp1252") in contenido, texto


def test_informe_de_una_guia_no_valida(corrida):
    """Verifica R-027: si la guía no es válida, el informe lo dice primero (NO IMPRIMIR) y lista los problemas."""
    carpeta, _, _ = corrida
    salida = json.loads((carpeta / "salida.json").read_text(encoding="utf-8"))
    salida["guia"]["valida"] = False
    salida["guia"]["problemas"] = ["La guía invade el escaneo hasta 0.30 mm."]
    lineas = lineas_del_informe(salida)
    estado = [texto for estilo, texto in lineas if estilo == "alerta"]
    assert "NO VÁLIDA — NO IMPRIMIR." in estado
    assert any("invade el escaneo" in t for t in estado)


def test_pdf_con_imagenes_y_varias_paginas(tmp_path):
    """Verifica R-027: el texto largo se reparte en páginas y las imágenes van comprimidas, dos por página."""
    lineas = [("titulo", "Prueba")] + [("normal", f"Renglón {i} con tildes: árbol, ñandú") for i in range(150)]
    imagen = np.zeros((40, 60, 3), dtype=np.uint8)
    imagen[:, :30] = (255, 128, 0)
    ruta = tmp_path / "prueba.pdf"
    escribir_pdf(lineas, ruta, [imagen, imagen, imagen])
    contenido = ruta.read_bytes()
    paginas = contenido.count(b"/Type /Page ")
    assert paginas == 3 + 2                                  # 150 renglones en 3 páginas + 3 imágenes en 2
    assert contenido.count(b"/Subtype /Image") == 3
    flujo = contenido.split(b"/Subtype /Image")[1].split(b"stream\n", 1)[1].split(b"\nendstream")[0]
    pixeles = np.frombuffer(zlib.decompress(flujo), dtype=np.uint8).reshape(40, 60, 3)
    np.testing.assert_array_equal(pixeles[0, 0], [255, 128, 0])
    assert "ñandú".encode("cp1252") in contenido


def test_guia_con_canal_mide_la_seguridad_del_plan(corrida, capsys):
    """Verifica R-027: con --canal, `guia` mide cada implante como `medir` (con el sobrefresado del kit) y el informe lo muestra."""
    carpeta, argumentos, pdf = corrida
    canal = vtk.vtkCubeSource()                          # techo del canal en z = −13: 1 mm bajo el ápice (z = −12)
    canal.SetBounds(-15, 20, -1.5, 1.5, -16, -13)
    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(canal.GetOutputPort())
    limpio = vtk.vtkCleanPolyData()
    limpio.SetInputConnection(tri.GetOutputPort())
    limpio.Update()
    _escribir_stl(limpio.GetOutput(), carpeta / "canal.stl")
    assert main(argumentos + ["--canal", str(carpeta / "canal.stl")]) == 0
    seguridad = json.loads(capsys.readouterr().out)["seguridad"]
    canal_1 = seguridad["implantes"]["implante_1"]["estructuras"]["canal"]
    assert canal_1["distancia_mm"] == pytest.approx(1.0 - 0.3, abs=0.01)        # sobrefresado de OneGuide (0,3)
    assert canal_1["semaforo"] == "rojo" and seguridad["semaforo"] == "rojo"
    contenido = pdf.read_bytes()
    for texto in ("Seguridad del plan: ROJO", "Canal: distancia 0,70 mm", "con sobrefresado de 0,30 mm"):
        assert texto.encode("cp1252") in contenido, texto
