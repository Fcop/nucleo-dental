"""Genera las mallas sintéticas de los casos dorados.

Coordenadas LPS en mm (R-002). Ambos canales corren a lo largo del eje X, de
x = -20 a x = 20 mm, centrados en y = 0, z = 0.

- canal_recto.stl: tubo cerrado de radio 1,5 mm y 64 lados, con anillos de
  vértices cada 1 mm en X, de modo que existen vértices exactamente en el
  techo (0, 0, 1.5) y en el costado (0, 1.5, 0).
- canal_grueso.stl: caja cerrada de sección 3 × 3 mm (y, z en [-1,5; 1,5]),
  con solo 8 vértices y 12 triángulos. Es la malla más gruesa posible: los
  vértices quedan a 20 mm del centro, lejos de cualquier implante, mientras
  las caras planas pasan cerca (RG-004).

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

CARPETA = Path(__file__).resolve().parents[1] / "tests" / "casos_dorados"


def construir_canal_recto() -> vtk.vtkPolyData:
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

    return _malla(puntos, np.vstack([pared, tapa_ini, tapa_fin]))


def construir_canal_grueso() -> vtk.vtkPolyData:
    # Vértice (i, j, k) = (X[i], Y[j], Z[k]) con índice i*4 + j*2 + k.
    xs, ys, zs = (X_MIN, X_MAX), (-RADIO, RADIO), (-RADIO, RADIO)
    puntos = np.array([[x, y, z] for x in xs for y in ys for z in zs])
    # Cada cara como cuadrilátero en sentido antihorario visto desde afuera.
    caras = [
        (0, 1, 3, 2),  # x = -20
        (4, 6, 7, 5),  # x = +20
        (0, 4, 5, 1),  # y = -1,5
        (2, 3, 7, 6),  # y = +1,5
        (0, 2, 6, 4),  # z = -1,5
        (1, 5, 7, 3),  # z = +1,5 (techo)
    ]
    triangulos = [t for a, b, c, d in caras for t in ((a, b, c), (a, c, d))]
    return _malla(puntos, np.array(triangulos))


def _malla(puntos: np.ndarray, triangulos: np.ndarray) -> vtk.vtkPolyData:
    triangulos = np.asarray(triangulos, dtype=np.int64)
    pd = vtk.vtkPolyData()
    vtk_puntos = vtk.vtkPoints()
    vtk_puntos.SetData(numpy_support.numpy_to_vtk(np.asarray(puntos, dtype=float), deep=True))
    pd.SetPoints(vtk_puntos)

    offsets = np.arange(0, 3 * len(triangulos) + 1, 3, dtype=np.int64)
    vtk_celdas = vtk.vtkCellArray()
    vtk_celdas.SetData(numpy_support.numpy_to_vtkIdTypeArray(offsets, deep=True),
                       numpy_support.numpy_to_vtkIdTypeArray(triangulos.ravel(), deep=True))
    pd.SetPolys(vtk_celdas)
    return pd


def verificar(pd: vtk.vtkPolyData, volumen_ideal: float, vertices_exactos=()) -> None:
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

    for objetivo in vertices_exactos:
        if not np.any(np.all(puntos == objetivo, axis=1)):
            raise RuntimeError(f"Falta el vértice exacto {objetivo}.")

    print(f"  Volumen: {volumen_con_signo:.4f} mm3 (ideal: {volumen_ideal:.4f})")


def escribir(pd: vtk.vtkPolyData, nombre: str) -> None:
    ruta = CARPETA / nombre
    ruta.parent.mkdir(parents=True, exist_ok=True)
    escritor = vtk.vtkSTLWriter()
    escritor.SetInputData(pd)
    escritor.SetFileName(str(ruta))
    escritor.SetFileTypeToBinary()
    escritor.Write()

    x0, x1, y0, y1, z0, z1 = pd.GetBounds()
    print(f"  Escrito: {ruta}")
    print(f"  Vértices: {pd.GetNumberOfPoints()}  Triángulos: {pd.GetNumberOfCells()}")
    print(f"  Límites (LPS, mm): x [{x0:g}, {x1:g}]  y [{y0:g}, {y1:g}]  z [{z0:g}, {z1:g}]")


def main() -> None:
    largo = X_MAX - X_MIN

    print("canal_recto.stl")
    recto = construir_canal_recto()
    verificar(recto, np.pi * RADIO ** 2 * largo, vertices_exactos=([0.0, 0.0, RADIO], [0.0, RADIO, 0.0]))
    escribir(recto, "canal_recto.stl")

    print("canal_grueso.stl")
    grueso = construir_canal_grueso()
    verificar(grueso, (2 * RADIO) ** 2 * largo)
    escribir(grueso, "canal_grueso.stl")


if __name__ == "__main__":
    main()
