"""Chequeo de que canal e implante caen dentro del CBCT (R-010, RG-002).

Los volúmenes son DICOM sintéticos creados en una carpeta temporal: nunca se
usa DICOM real en los tests (regla 8 de CLAUDE.md).
"""

import hashlib
from pathlib import Path

import pytest

sitk = pytest.importorskip("SimpleITK")
import vtk  # noqa: E402

from test_medir import CASOS_DORADOS, ejecutar  # noqa: E402

CANAL_RECTO = CASOS_DORADOS / "canal_recto.stl"
IDENTIDAD = (1, 0, 0, 0, 1, 0, 0, 0, 1)


def escribir_serie_dicom(carpeta: Path, origen, espaciado, tamano, direccion=IDENTIDAD) -> Path:
    """Escribe un volumen sintético como serie DICOM, un archivo por corte."""
    carpeta.mkdir(parents=True, exist_ok=True)
    imagen = sitk.Image([int(n) for n in tamano], sitk.sitkInt16)
    imagen.SetOrigin([float(v) for v in origen])
    imagen.SetSpacing([float(v) for v in espaciado])
    imagen.SetDirection([float(v) for v in direccion])
    d = imagen.GetDirection()
    orientacion = "\\".join(str(v) for v in (d[0], d[3], d[6], d[1], d[4], d[7]))

    escritor = sitk.ImageFileWriter()
    escritor.KeepOriginalImageUIDOn()
    for k in range(imagen.GetDepth()):
        corte = imagen[:, :, k]
        for etiqueta, valor in {
            "0010|0010": "SINTETICO^PRUEBA",
            "0008|0060": "CT",
            "0020|000d": "1.2.826.0.1.3680043.2.1125.1",
            "0020|000e": "1.2.826.0.1.3680043.2.1125.1.1",
            "0020|0037": orientacion,
            "0020|0032": "\\".join(str(v) for v in imagen.TransformIndexToPhysicalPoint((0, 0, k))),
            "0020|0013": str(k + 1),
        }.items():
            corte.SetMetaData(etiqueta, valor)
        escritor.SetFileName(str(carpeta / f"corte_{k:03d}.dcm"))
        escritor.Execute(corte)
    return carpeta


def trasladar_stl(origen: Path, destino: Path, x=0.0, y=0.0, z=0.0, escala=(1.0, 1.0, 1.0)) -> Path:
    """Copia un STL trasladado y luego escalado respecto del origen LPS.

    Escala (-1, -1, 1) después de trasladar simula un STL exportado en RAS sin
    convertir: el punto (x, y, z) queda en (-x, -y, z).
    """
    lector = vtk.vtkSTLReader()
    lector.SetFileName(str(origen))
    tr = vtk.vtkTransform()
    tr.PostMultiply()
    tr.Translate(x, y, z)
    tr.Scale(*escala)
    filtro = vtk.vtkTransformPolyDataFilter()
    filtro.SetInputConnection(lector.GetOutputPort())
    filtro.SetTransform(tr)
    escritor = vtk.vtkSTLWriter()
    escritor.SetInputConnection(filtro.GetOutputPort())
    escritor.SetFileName(str(destino))
    escritor.SetFileTypeToBinary()
    escritor.Write()
    return destino


def medir_con_cbct(canal, apice, cbct, eje="0,0,1"):
    return ejecutar("medir", "--canal", canal, "--diametro", "4.1", "--largo", "10",
                    "--apice", apice, "--eje", eje, "--cbct", cbct)


def test_dentro_del_volumen_mide_y_reporta_el_cbct(tmp_path):
    """Verifica R-010: con todo dentro del CBCT se mide normal y la salida describe el volumen."""
    cbct = escribir_serie_dicom(tmp_path / "cbct", origen=(-25, -10, -10), espaciado=(2.5, 2.0, 2.0),
                                tamano=(20, 10, 15))
    codigo, salida, stderr = medir_con_cbct(CANAL_RECTO, "0,0,4", cbct)
    assert codigo == 0, stderr
    assert salida["resultado"]["estructuras"]["canal"]["distancia_mm"] == pytest.approx(2.5, abs=0.05)

    c = salida["cbct"]
    assert c["tamano"] == [20, 10, 15]
    assert c["espaciado_mm"] == pytest.approx([2.5, 2.0, 2.0])
    assert c["origen_mm"] == pytest.approx([-25, -10, -10])
    assert c["n_archivos"] == 15
    primero = sorted(cbct.glob("*.dcm"))[0]
    assert c["sha256_primer_archivo"] == hashlib.sha256(primero.read_bytes()).hexdigest()
    assert len(c["sha256_serie"]) == 64


def test_fuera_de_volumen(tmp_path):
    """Verifica R-010: un canal en RAS sin convertir cae fuera del CBCT; código 1 y sugerencia RAS/LPS.

    El caso está desplazado lejos del origen (x = 10 a 50, y ≈ 40), como en un
    CBCT real. Pasar de LPS a RAS invierte x e y, y el canal sale del volumen.
    """
    cbct = escribir_serie_dicom(tmp_path / "cbct", origen=(5, 25, -10), espaciado=(2.5, 2.0, 2.0),
                                tamano=(20, 10, 15))
    canal_lps = trasladar_stl(CANAL_RECTO, tmp_path / "canal_lps.stl", x=30, y=40)
    canal_ras = trasladar_stl(CANAL_RECTO, tmp_path / "canal_ras.stl", x=30, y=40, escala=(-1, -1, 1))

    codigo, _, stderr = medir_con_cbct(canal_lps, "30,40,4", cbct)
    assert codigo == 0, stderr

    codigo, salida, stderr = medir_con_cbct(canal_ras, "30,40,4", cbct)
    assert codigo == 1
    assert salida is None
    assert "canal" in stderr.lower()
    assert "RAS" in stderr and "LPS" in stderr


def test_implante_fuera_de_volumen(tmp_path):
    """Verifica R-010: un implante con coordenadas RAS cae fuera del CBCT; código 1."""
    cbct = escribir_serie_dicom(tmp_path / "cbct", origen=(5, 25, -10), espaciado=(2.5, 2.0, 2.0),
                                tamano=(20, 10, 15))
    canal_lps = trasladar_stl(CANAL_RECTO, tmp_path / "canal_lps.stl", x=30, y=40)
    codigo, _, stderr = medir_con_cbct(canal_lps, "-30,-40,4", cbct)
    assert codigo == 1
    assert "implante" in stderr.lower()
    assert "RAS" in stderr


def test_volumen_oblicuo_usa_la_matriz_de_direccion(tmp_path):
    """Verifica R-010: la caja del CBCT se calcula con la matriz de dirección, no solo con origen y tamaño.

    El volumen está rotado 90° en z: su índice i avanza hacia +y y su índice j
    hacia -x. Ignorar la dirección pondría la caja en x = 24..74 y el canal
    (x = -20..20) parecería fuera.
    """
    rot_z_90 = (0, -1, 0, 1, 0, 0, 0, 0, 1)
    cbct = escribir_serie_dicom(tmp_path / "cbct", origen=(24, -7, -10), espaciado=(2.0, 2.0, 2.0),
                                tamano=(8, 25, 15), direccion=rot_z_90)
    codigo, salida, stderr = medir_con_cbct(CANAL_RECTO, "0,0,4", cbct)
    assert codigo == 0, stderr
    assert salida["cbct"]["direccion"] == pytest.approx(list(rot_z_90))


def test_carpeta_sin_dicom_es_error_de_entrada(tmp_path):
    """Verifica R-010 y R-008: una carpeta sin serie DICOM es error de entrada (1)."""
    vacia = tmp_path / "vacia"
    vacia.mkdir()
    codigo, _, stderr = medir_con_cbct(CANAL_RECTO, "0,0,4", vacia)
    assert codigo == 1
    assert "DICOM" in stderr
