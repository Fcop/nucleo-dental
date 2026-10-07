"""Modelo del implante (R-003).

El implante es un cilindro sólido de diámetro D y largo L. El ápice P es el
centro de la punta y el eje unitario u apunta del ápice a la plataforma.
Coordenadas LPS en mm.
"""

from __future__ import annotations

import numpy as np

# Margen numérico para considerar un punto "sobre la superficie". Absorbe solo
# el redondeo de punto flotante; no es una tolerancia clínica.
TOLERANCIA_SUPERFICIE_MM = 1e-9


def _positivo(valor, nombre: str) -> float:
    valor = float(valor)
    if not np.isfinite(valor) or valor <= 0:
        raise ValueError(f"El {nombre} del implante debe ser un número mayor que 0 mm (se recibió {valor}).")
    return valor


def _vector3(valor, nombre: str) -> np.ndarray:
    v = np.asarray(valor, dtype=float)
    if v.shape != (3,):
        raise ValueError(f"El {nombre} del implante debe tener 3 coordenadas (x, y, z); se recibió {valor}.")
    if not np.all(np.isfinite(v)):
        raise ValueError(f"El {nombre} del implante contiene valores no numéricos o infinitos: {valor}.")
    return v


class Implante:
    def __init__(self, diametro: float, largo: float, apice, eje):
        self.diametro = _positivo(diametro, "diámetro")
        self.largo = _positivo(largo, "largo")
        self.radio = self.diametro / 2.0
        self.apice = _vector3(apice, "ápice")
        eje = _vector3(eje, "eje")
        norma = np.linalg.norm(eje)
        if norma < 1e-12:
            raise ValueError("El eje del implante tiene largo cero; debe indicar una dirección.")
        self.eje = eje / norma

    @property
    def plataforma(self) -> np.ndarray:
        return self.apice + self.largo * self.eje

    def _coordenadas_axiales(self, puntos):
        """Altura t sobre el ápice (a lo largo del eje) y distancia r al eje, por punto."""
        p = np.atleast_2d(np.asarray(puntos, dtype=float)) - self.apice
        t = p @ self.eje
        r = np.linalg.norm(p - t[:, None] * self.eje, axis=1)
        return t, r

    def distancia_a_puntos(self, puntos) -> np.ndarray:
        """Distancia exacta de cada punto (N×3) al cilindro sólido; 0 si está dentro.

        Zonas: costado (solo exceso radial), tapas (solo exceso axial) y
        esquina (ambos, distancia al borde por Pitágoras).
        """
        t, r = self._coordenadas_axiales(puntos)
        exceso_axial = np.maximum.reduce([-t, t - self.largo, np.zeros_like(t)])
        exceso_radial = np.maximum(r - self.radio, 0.0)
        return np.hypot(exceso_axial, exceso_radial)

    def contiene(self, puntos) -> np.ndarray:
        """True por cada punto (N×3) dentro del cilindro o sobre su superficie."""
        return self.distancia_a_puntos(puntos) <= TOLERANCIA_SUPERFICIE_MM

    def puntos_superficie(self, paso: float) -> np.ndarray:
        """Puntos (N×3) sobre toda la superficie, separados a lo más `paso` mm.

        Incluye el centro de cada tapa y los bordes exactos (ápice y plataforma).
        """
        e1, e2 = self._base_perpendicular()
        n_vueltas = max(int(np.ceil(2 * np.pi * self.radio / paso)), 3)
        n_alturas = max(int(np.ceil(self.largo / paso)), 1) + 1

        angulos = 2 * np.pi * np.arange(n_vueltas) / n_vueltas
        alturas = np.linspace(0.0, self.largo, n_alturas)
        aro = self.radio * (np.cos(angulos)[:, None] * e1 + np.sin(angulos)[:, None] * e2)
        pared = (self.apice + alturas[:, None, None] * self.eje + aro[None, :, :]).reshape(-1, 3)

        disco = self._disco(paso, e1, e2)
        tapas = np.vstack([self.apice + disco, self.plataforma + disco])
        return np.vstack([pared, tapas])

    def puntos_interiores(self, paso: float) -> np.ndarray:
        """Puntos (N×3) en una grilla de lado `paso` que llena el volumen, incluido el eje."""
        e1, e2 = self._base_perpendicular()
        n_lado = int(np.ceil(self.radio / paso))
        coord = paso * np.arange(-n_lado, n_lado + 1)
        u1, u2 = np.meshgrid(coord, coord, indexing="ij")
        dentro = u1 ** 2 + u2 ** 2 <= self.radio ** 2
        seccion = u1[dentro][:, None] * e1 + u2[dentro][:, None] * e2
        alturas = np.linspace(0.0, self.largo, max(int(np.ceil(self.largo / paso)), 1) + 1)
        return (self.apice + alturas[:, None, None] * self.eje + seccion[None, :, :]).reshape(-1, 3)

    def _disco(self, paso: float, e1, e2) -> np.ndarray:
        """Puntos de un disco de radio R centrado en el origen, en anillos concéntricos."""
        n_anillos = max(int(np.ceil(self.radio / paso)), 1)
        radios = np.linspace(0.0, self.radio, n_anillos + 1)
        n_por_anillo = np.maximum(np.ceil(2 * np.pi * radios / paso).astype(int), 1)
        indice_anillo = np.repeat(np.arange(len(radios)), n_por_anillo)
        posicion = np.arange(n_por_anillo.sum()) - np.repeat(np.cumsum(n_por_anillo) - n_por_anillo, n_por_anillo)
        angulo = 2 * np.pi * posicion / n_por_anillo[indice_anillo]
        r = radios[indice_anillo]
        return (r * np.cos(angulo))[:, None] * e1 + (r * np.sin(angulo))[:, None] * e2

    def como_malla(self, lados: int = 64):
        """Superficie triangulada cerrada del implante, con normales hacia afuera.

        Los vértices quedan exactamente sobre el cilindro: el polígono de
        `lados` lados está inscrito en el círculo de radio R.
        """
        import vtk
        from vtk.util import numpy_support

        e1, e2 = self._base_perpendicular()
        angulos = 2.0 * np.pi * np.arange(lados) / lados
        circulo = self.radio * (np.cos(angulos)[:, None] * e1 + np.sin(angulos)[:, None] * e2)
        puntos = np.vstack([
            self.apice + circulo,        # anillo del ápice
            self.plataforma + circulo,   # anillo de la plataforma
            [self.apice, self.plataforma],
        ])
        centro_apice, centro_plataforma = 2 * lados, 2 * lados + 1

        k = np.arange(lados)
        k_sig = (k + 1) % lados
        pared = np.vstack([
            np.stack([k, k_sig, lados + k_sig], axis=1),
            np.stack([k, lados + k_sig, lados + k], axis=1),
        ])
        tapa_apice = np.stack([np.full(lados, centro_apice), k_sig, k], axis=1)
        tapa_plataforma = np.stack([np.full(lados, centro_plataforma), lados + k, lados + k_sig], axis=1)
        triangulos = np.vstack([pared, tapa_apice, tapa_plataforma]).astype(np.int64)

        vtk_puntos = vtk.vtkPoints()
        vtk_puntos.SetData(numpy_support.numpy_to_vtk(puntos, deep=True))
        offsets = np.arange(0, 3 * len(triangulos) + 1, 3, dtype=np.int64)
        vtk_celdas = vtk.vtkCellArray()
        vtk_celdas.SetData(numpy_support.numpy_to_vtkIdTypeArray(offsets, deep=True),
                           numpy_support.numpy_to_vtkIdTypeArray(triangulos.ravel(), deep=True))

        pd = vtk.vtkPolyData()
        pd.SetPoints(vtk_puntos)
        pd.SetPolys(vtk_celdas)
        return pd

    def _base_perpendicular(self):
        """Dos vectores unitarios e1, e2 perpendiculares al eje, con e1 × e2 = eje."""
        auxiliar = np.array([1.0, 0.0, 0.0]) if abs(self.eje[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = auxiliar - (auxiliar @ self.eje) * self.eje
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(self.eje, e1)
        return e1, e2
