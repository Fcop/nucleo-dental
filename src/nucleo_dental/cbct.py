"""Lectura del CBCT y chequeo de que canal e implante caen dentro (R-010, RG-002).

Si el canal o el implante vienen en RAS (el sistema de Slicer) en vez de LPS,
la medición se haría en otro lugar sin dar error. Este chequeo detecta el caso
más frecuente: con x e y invertidas, las coordenadas salen del volumen.

SimpleITK es una dependencia opcional (`pip install -e ".[cbct]"`) y solo se
importa al leer un CBCT. ITK trabaja en LPS: origen, espaciado y matriz de
dirección se leen ya en el sistema del núcleo.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class VolumenCBCT:
    carpeta: Path
    origen: np.ndarray       # centro del vóxel (0, 0, 0), LPS, mm
    espaciado: np.ndarray    # mm por vóxel en i, j, k
    tamano: np.ndarray       # vóxeles en i, j, k
    direccion: np.ndarray    # 3×3; la columna c es la dirección LPS del índice c
    archivos: tuple[Path, ...]

    def contiene(self, puntos) -> np.ndarray:
        """True por cada punto (N×3, LPS) dentro de la caja física del volumen.

        La caja abarca el vóxel completo: de -0,5 a tamaño - 0,5 en índices.
        """
        p = np.atleast_2d(np.asarray(puntos, dtype=float)) - self.origen
        indices = np.linalg.solve(self.direccion * self.espaciado, p.T).T
        return np.all((indices >= -0.5) & (indices <= self.tamano - 0.5), axis=1)

    def descripcion(self) -> dict:
        return {
            "carpeta": str(self.carpeta),
            "tamano": [int(n) for n in self.tamano],
            "espaciado_mm": [float(v) for v in self.espaciado],
            "origen_mm": [float(v) for v in self.origen],
            "direccion": [float(v) for v in self.direccion.ravel()],
            "n_archivos": len(self.archivos),
            "sha256_primer_archivo": _sha256(self.archivos[0]),
            "sha256_serie": _sha256_serie(self.archivos),
        }


def leer_cbct(carpeta) -> VolumenCBCT:
    """Lee la geometría de la serie DICOM de una carpeta (sin cargar los píxeles en el resultado)."""
    try:
        import SimpleITK as sitk
    except ImportError:
        raise ValueError('Para usar --cbct instala SimpleITK: pip install -e ".[cbct]"') from None

    carpeta = Path(carpeta)
    if not carpeta.is_dir():
        raise ValueError(f"No existe la carpeta del CBCT: {carpeta}")
    series = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(carpeta))
    if not series:
        raise ValueError(f"No se encontró una serie DICOM en {carpeta}")
    if len(series) > 1:
        raise ValueError(f"La carpeta {carpeta} tiene {len(series)} series DICOM; deja solo la del CBCT.")

    nombres = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(carpeta), series[0])
    lector = sitk.ImageSeriesReader()
    lector.SetFileNames(nombres)
    imagen = lector.Execute()

    return VolumenCBCT(
        carpeta=carpeta,
        origen=np.array(imagen.GetOrigin(), dtype=float),
        espaciado=np.array(imagen.GetSpacing(), dtype=float),
        tamano=np.array(imagen.GetSize(), dtype=int),
        direccion=np.array(imagen.GetDirection(), dtype=float).reshape(3, 3),
        archivos=tuple(Path(n) for n in nombres),
    )


def exigir_dentro_del_volumen(volumen: VolumenCBCT, vertices_canal, implante) -> None:
    """Error de entrada si algún vértice del canal o el ápice o la plataforma caen fuera del CBCT."""
    problemas = []

    fuera = ~volumen.contiene(vertices_canal)
    if fuera.any():
        problemas.append(f"{int(fuera.sum())} de {len(fuera)} vértices del canal")

    extremos = {"ápice": implante.apice, "plataforma": implante.plataforma}
    for nombre, dentro in zip(extremos, volumen.contiene(np.array(list(extremos.values())))):
        if not dentro:
            p = extremos[nombre]
            problemas.append(f"el {nombre} del implante ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f})")

    if problemas:
        raise ValueError(
            "Quedan fuera del volumen del CBCT: " + "; ".join(problemas) + ". "
            "Revisa el sistema de coordenadas: el núcleo trabaja en LPS y Slicer en RAS. "
            "Si exportaste desde Slicer, confirma que el diálogo de guardado indique LPS.")


def _sha256(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def _sha256_serie(archivos) -> str:
    """Huella de la serie completa: nombre y contenido de cada archivo, en orden."""
    h = hashlib.sha256()
    for ruta in archivos:
        h.update(ruta.name.encode("utf-8"))
        h.update(ruta.read_bytes())
    return h.hexdigest()
