"""Genera la malla sintética de los casos dorados (Paso 4).

Canal: tubo recto cerrado con tapas, a lo largo del eje X, de x = -20 a
x = 20 mm, centrado en y = 0, z = 0, radio 1,5 mm y 64 lados. Coordenadas
LPS en mm (R-002).

El tubo tiene anillos de vértices cada 1 mm en X, de modo que existen
vértices exactamente en el techo (0, 0, 1.5) y en el costado (0, 1.5, 0).

Este script NO escribe ningún esperado.json: los resultados esperados los
calcula y escribe una persona.

Uso:
    python scripts/generar_casos_dorados.py
"""

from pathlib import Path

import numpy as np
import vtk
from vtk.util import numpy_support

RADIO = 1.5
LADOS = 64
X_MIN, X_MAX = -20.0, 20.0
PASO_X = 1.0

SALIDA = Path(__file__).resolve().parents[1] / "tests" / "casos_dorados" / "canal_recto.stl"


def construir_canal() -> vtk.vtkPolyData:
    xs = np.linspace(X_MIN, X_MAX, int(round((X_MAX - X_MIN) / PASO_X)) + 1)
    angulos = 2.0 * np.pi * np.arange(LADOS) / LADOS
    y = RADIO * np.cos(angulos)
    z = RADIO * np.sin(angulos)
    # Anula los residuos de punto flotante (p. ej. cos(pi/2) = 6e-17) para que
    # los vértices del techo y del costado queden exactos.
    y[np.abs(y) < 1e-12] = 0.0
    z[np.abs(z) < 1e-12] = 0.0

    n_anillos = len(xs)
    anillos = np.empty((n_anillos, LADOS, 3))
    anillos[:, :, 0] = xs[:, None]
    anillos[:, :, 1] = y[None, :]
    anillos[:, :, 2] = z[None, :]
    puntos = np.vstack([
        anillos.reshape(-1, 3),
        [[X_MIN, 0.0, 0.0], [X_MAX, 0.0, 0.0]],   # centros de las tapas
    ])
    centro_ini = n_anillos * LADOS
    centro_fin = centro_ini + 1

    # Pared lateral: dos triángulos por cuadrilátero, normales hacia afuera.
    i = np.arange(n_anillos - 1)[:, None]
    k = np.arange(LADOS)[None, :]
    a = i * LADOS + k
    b = i * LADOS + (k + 1) % LADOS
    c = (i + 1) * LADOS + (k + 1) % LADOS
    d = (i + 1) * LADOS + k
    pared = np.concatenate([
        np.stack([a, b, c], axis=-1).reshape(-1, 3),
        np.stack([a, c, d], axis=-1).reshape(-1, 3),
    ])

    # Tapas en abanico: la de x_min mira a -X y la de x_max a +X.
    k = np.arange(LADOS)
    k_sig = (k + 1) % LADOS
    tapa_ini = np.stack([np.full(LADOS, centro_ini), k_sig, k], axis=-1)
    base_fin = (n_anillos - 1) * LADOS
    tapa_fin = np.stack([np.full(LADOS, centro_fin), base_fin + k, base_fin + k_sig], axis=-1)

    triangulos = np.vstack([pared, tapa_ini, tapa_fin]).astype(np.int64)

    pd = vtk.vtkPolyData()
    vtk_puntos = vtk.vtkPoints()
    vtk_puntos.SetData(numpy_support.numpy_to_vtk(puntos, deep=True))
    pd.SetPoints(vtk_puntos)

    offsets = np.arange(0, 3 * len(triangulos) + 1, 3, dtype=np.int64)
    vtk_celdas = vtk.vtkCellArray()
    vtk_celdas.SetData(numpy_support.numpy_to_vtkIdTypeArray(offsets, deep=True),
                       numpy_support.numpy_to_vtkIdTypeArray(triangulos.ravel(), deep=True))
    pd.SetPolys(vtk_celdas)
    return pd


def verificar(pd: vtk.vtkPolyData) -> None:
    """Comprueba que la malla es cerrada, con normales hacia afuera y los vértices pedidos."""
    bordes = vtk.vtkFeatureEdges()
    bordes.SetInputData(pd)
    bordes.BoundaryEdgesOn()
    bordes.NonManifoldEdgesOn()
    bordes.FeatureEdgesOff()
    bordes.ManifoldEdgesOff()
    bordes.Update()
    n_bordes = bordes.GetOutput().GetNumberOfCells()
    if n_bordes:
        raise RuntimeError(f"La malla no es cerrada: {n_bordes} aristas de borde o no manifold.")

    # Volumen con signo vía teorema de la divergencia: positivo si las normales miran afuera.
    puntos = numpy_support.vtk_to_numpy(pd.GetPoints().GetData())
    tri = numpy_support.vtk_to_numpy(pd.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    v0, v1, v2 = puntos[tri[:, 0]], puntos[tri[:, 1]], puntos[tri[:, 2]]
    volumen_con_signo = np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0
    if volumen_con_signo <= 0:
        raise RuntimeError("Las normales apuntan hacia adentro.")

    for objetivo in ([0.0, 0.0, RADIO], [0.0, RADIO, 0.0]):
        if not np.any(np.all(puntos == objetivo, axis=1)):
            raise RuntimeError(f"Falta el vértice exacto {objetivo}.")

    print(f"Volumen: {volumen_con_signo:.4f} mm3 (cilindro ideal: {np.pi * RADIO**2 * (X_MAX - X_MIN):.4f})")


def main() -> None:
    pd = construir_canal()
    verificar(pd)

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    escritor = vtk.vtkSTLWriter()
    escritor.SetInputData(pd)
    escritor.SetFileName(str(SALIDA))
    escritor.SetFileTypeToBinary()
    escritor.Write()

    x0, x1, y0, y1, z0, z1 = pd.GetBounds()
    print(f"Escrito: {SALIDA}")
    print(f"Vértices: {pd.GetNumberOfPoints()}  Triángulos: {pd.GetNumberOfCells()}")
    print(f"Límites (LPS, mm): x [{x0:g}, {x1:g}]  y [{y0:g}, {y1:g}]  z [{z0:g}, {z1:g}]")


if __name__ == "__main__":
    main()
