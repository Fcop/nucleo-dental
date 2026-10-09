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
- diente_006.stl y diente_007.stl: raíz simplificada, cilindro vertical de
  radio 3,0 mm entre z = 5 y z = 25, con 64 lados y centro en x = 0 e
  y = 7,0 o y = 6,3. Hay un vértice exacto en la pared que mira al implante
  (y = 4,0 y y = 3,3).
- diente_008.stl: dos piezas cerradas de radio 3,0 mm. "Raíz" con centro en
  y = 8,0 entre z = 5 y z = 13 (pared en y = 5,0) y "corona" con centro en
  y = 5,5 entre z = 15 y z = 25 (pared en y = 2,5), toda sobre la plataforma
  del implante del caso 001 (z = 14). Prueba que la corona no cuenta (R-013).
- hueso_009.stl y hueso_010.stl: tubo de hueso simplificado, cilindro vertical
  de z = 0 a z = 20 con centro en y = 0,5 y radio 4,0 o 4,5 (bordes en
  y = -3,5 / 4,5 y y = -4,0 / 5,0). Rodea al implante del caso 001 (R-014).
- arcada_cbct.stl: cinco "cúspides" esféricas de distintos radios (dientes del CBCT).
- escaneo_corrido.stl: la misma arcada corrida 0,5 mm en +x, más una lámina de
  "encía" en z = 4 que no existe en el CBCT (registro, R-015).
- escaneo_arcada_recta.stl y dientes_arcada_recta.stl: tres dientes cúbicos de
  6 mm por lado del eje x = 0 (bordes en x = 5/11, 13/19, 21/27), de z = 0 a 8,
  sobre una encía plana en z = 0; en el CBCT siguen como raíz hasta z = -10
  (región de apoyo, R-016).
- diente_017.stl y guia_017_sin_recortar.stl: corte frontal (x–z) extruido de
  y = -5 a 5. Diente sobre una base de encía (x de -10 a 10, z de -2 a 0):
  cuello de x = -3 a 3 hasta z = 6 y corona de x = -4 a 4 entre z = 6 y 10.
  Guía antes del recorte: caja de x = -6 a 6 y z = 4 a 12 menos el diente;
  rellena el hueco bajo la corona (|x| de 3 a 4, z de 4 a 6) (ensamblaje, R-021).

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
RADIO_DIENTE = 3.0

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


def construir_diente(centro_y: float, z_min: float = 5.0, z_max: float = 25.0,
                     radio: float = RADIO_DIENTE) -> vtk.vtkPolyData:
    """Cilindro vertical cerrado (raíz o tubo de hueso simplificado) con centro en x = 0."""
    from nucleo_dental.implante import Implante

    malla = Implante(diametro=2 * radio, largo=z_max - z_min, apice=[0.0, centro_y, z_min],
                     eje=[0.0, 0.0, 1.0]).como_malla(lados=LADOS)
    # Anula los residuos de punto flotante (cos(3π/2) = -1.8e-16) para que el
    # vértice de la pared que mira al implante quede exacto.
    puntos = numpy_support.vtk_to_numpy(malla.GetPoints().GetData()).copy()
    puntos[np.abs(puntos) < 1e-12] = 0.0
    malla.GetPoints().SetData(numpy_support.numpy_to_vtk(puntos, deep=True))
    return malla


# "Cúspides" de la arcada sintética: (centro LPS, radio). Tamaños y posiciones
# distintos para que ningún desplazamiento ni giro deje la arcada igual.
CUSPIDES = (((0.0, 0.0, 10.0), 2.0), ((10.0, 0.0, 10.0), 2.5), ((0.0, 10.0, 10.0), 1.5),
            ((10.0, 10.0, 12.0), 3.0), ((5.0, 18.0, 9.0), 2.2))
CORRIMIENTO_ESCANEO = (0.5, 0.0, 0.0)


def construir_arcada() -> vtk.vtkPolyData:
    """Dientes del "CBCT": esferas cerradas (cúspides) de distintos radios."""
    union = vtk.vtkAppendPolyData()
    for centro, radio in CUSPIDES:
        esfera = vtk.vtkSphereSource()
        esfera.SetCenter(*centro)
        esfera.SetRadius(radio)
        esfera.SetThetaResolution(48)
        esfera.SetPhiResolution(48)
        esfera.Update()
        union.AddInputData(esfera.GetOutput())
    limpio = vtk.vtkCleanPolyData()
    limpio.SetInputConnection(union.GetOutputPort())
    limpio.Update()
    return limpio.GetOutput()


def construir_escaneo_corrido(arcada: vtk.vtkPolyData) -> vtk.vtkPolyData:
    """El "escaneo intraoral": la misma arcada corrida 0,5 mm en +x, más una lámina de "encía"
    (z = 4, lejos de las cúspides) que no existe en el CBCT y que el registro debe descartar."""
    tr = vtk.vtkTransform()
    tr.Translate(*CORRIMIENTO_ESCANEO)
    mover = vtk.vtkTransformPolyDataFilter()
    mover.SetInputData(arcada)
    mover.SetTransform(tr)
    encia = vtk.vtkPlaneSource()
    encia.SetOrigin(-5.0, -5.0, 4.0)
    encia.SetPoint1(16.0, -5.0, 4.0)
    encia.SetPoint2(-5.0, 23.0, 4.0)
    encia.SetResolution(40, 50)
    triangulos = vtk.vtkTriangleFilter()
    triangulos.SetInputConnection(encia.GetOutputPort())
    union = vtk.vtkAppendPolyData()
    union.AddInputConnection(mover.GetOutputPort())
    union.AddInputConnection(triangulos.GetOutputPort())
    union.Update()
    return union.GetOutput()


# Arcada recta de la región de apoyo (R-016): eje del implante en x = 0; tres
# dientes cúbicos de 6 mm por lado, de z = 0 (encía) a z = 8.
DIENTES_ARCADA_RECTA = ((5.0, 11.0), (13.0, 19.0), (21.0, 27.0))
_RESOLUCION_CARA_MM = 0.25


def _cara(origen, punto1, punto2) -> vtk.vtkPolyData:
    """Rectángulo triangulado con vértices cada ~0,25 mm."""
    plano = vtk.vtkPlaneSource()
    plano.SetOrigin(*origen)
    plano.SetPoint1(*punto1)
    plano.SetPoint2(*punto2)
    plano.SetXResolution(max(1, int(round(np.linalg.norm(np.subtract(punto1, origen)) / _RESOLUCION_CARA_MM))))
    plano.SetYResolution(max(1, int(round(np.linalg.norm(np.subtract(punto2, origen)) / _RESOLUCION_CARA_MM))))
    triangulos = vtk.vtkTriangleFilter()
    triangulos.SetInputConnection(plano.GetOutputPort())
    triangulos.Update()
    return triangulos.GetOutput()


def construir_arcada_recta():
    """(escaneo, dientes del CBCT) de la arcada recta.

    Escaneo: encía plana en z = 0 más la cara oclusal y las cuatro paredes de
    cada diente (superficie abierta, como un escaneo real). CBCT: cada diente
    como caja cerrada que sigue bajo la encía (raíz) hasta z = -10.
    """
    union_escaneo = vtk.vtkAppendPolyData()
    union_escaneo.AddInputData(_cara((-35, -8, 0), (35, -8, 0), (-35, 8, 0)))
    union_cbct = vtk.vtkAppendPolyData()
    for signo in (1.0, -1.0):
        for x0, x1 in DIENTES_ARCADA_RECTA:
            a, b = sorted((signo * x0, signo * x1))
            caras = [((a, -3, 8), (b, -3, 8), (a, 3, 8)),     # oclusal
                     ((a, -3, 0), (b, -3, 0), (a, -3, 8)),     # pared -y
                     ((a, 3, 0), (b, 3, 0), (a, 3, 8)),        # pared +y
                     ((a, -3, 0), (a, 3, 0), (a, -3, 8)),      # pared -x
                     ((b, -3, 0), (b, 3, 0), (b, -3, 8))]      # pared +x
            for cara in caras:
                union_escaneo.AddInputData(_cara(*cara))
            caja = vtk.vtkCubeSource()
            caja.SetBounds(a, b, -3, 3, -10, 8)
            tri = vtk.vtkTriangleFilter()
            tri.SetInputConnection(caja.GetOutputPort())
            tri.Update()
            union_cbct.AddInputData(tri.GetOutput())
    salidas = []
    for union in (union_escaneo, union_cbct):
        limpio = vtk.vtkCleanPolyData()
        limpio.SetInputConnection(union.GetOutputPort())
        limpio.Update()
        salidas.append(limpio.GetOutput())
    return tuple(salidas)


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


def construir_perfil_extruido(perfil_xz, y_min: float, y_max: float) -> vtk.vtkPolyData:
    """Sólido cerrado: polígono simple del plano x–z extruido a lo largo de y."""
    perfil = np.asarray(perfil_xz, dtype=float)
    n = len(perfil)
    poligono = vtk.vtkPolyData()
    puntos = vtk.vtkPoints()
    puntos.SetData(numpy_support.numpy_to_vtk(np.column_stack([perfil[:, 0], np.zeros(n), perfil[:, 1]]), deep=True))
    poligono.SetPoints(puntos)
    celda = vtk.vtkCellArray()
    celda.InsertNextCell(n, list(range(n)))
    poligono.SetPolys(celda)
    tri = vtk.vtkTriangleFilter()                 # triangula la tapa (polígono cóncavo)
    tri.SetInputData(poligono)
    tri.Update()
    tapa = numpy_support.vtk_to_numpy(tri.GetOutput().GetPolys().GetConnectivityArray()).reshape(-1, 3)

    vertices = np.vstack([np.column_stack([perfil[:, 0], np.full(n, y_min), perfil[:, 1]]),
                          np.column_stack([perfil[:, 0], np.full(n, y_max), perfil[:, 1]])])
    # Tapas con el mismo sentido de giro que el polígono; así son coherentes con los lados.
    plano = vertices[:n]
    giro = np.cross(plano, np.roll(plano, -1, axis=0))[:, 1].sum()      # componente y de la normal del polígono
    normal_tapa = np.cross(plano[tapa[:, 1]] - plano[tapa[:, 0]], plano[tapa[:, 2]] - plano[tapa[:, 0]])[:, 1]
    tapa = np.where((normal_tapa * giro < 0)[:, None], tapa[:, ::-1], tapa)
    i = np.arange(n)
    j = (i + 1) % n
    lados = np.vstack([np.column_stack([i, j, j + n]), np.column_stack([i, j + n, i + n])])
    triangulos = np.vstack([tapa[:, ::-1], tapa + n, lados])
    pd = _malla(vertices, triangulos)
    v = numpy_support.vtk_to_numpy(pd.GetPoints().GetData())
    t = triangulos
    if np.einsum("ij,ij->i", v[t[:, 0]], np.cross(v[t[:, 1]], v[t[:, 2]])).sum() < 0:
        pd = _malla(vertices, triangulos[:, ::-1])
    return pd


DIENTE_017 = [(-10, -2), (10, -2), (10, 0), (3, 0), (3, 6), (4, 6), (4, 10),
              (-4, 10), (-4, 6), (-3, 6), (-3, 0), (-10, 0)]
GUIA_017 = [(-6, 4), (-3, 4), (-3, 6), (-4, 6), (-4, 10), (4, 10), (4, 6),
            (3, 6), (3, 4), (6, 4), (6, 12), (-6, 12)]


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

    for nombre, centro_y in (("diente_006.stl", 7.0), ("diente_007.stl", 6.3)):
        print(nombre)
        diente = construir_diente(centro_y)
        verificar(diente, np.pi * RADIO_DIENTE ** 2 * 20.0,
                  vertices_exactos=([0.0, centro_y - RADIO_DIENTE, 5.0],))
        escribir(diente, nombre)

    print("diente_008.stl")
    raiz = construir_diente(8.0, 5.0, 13.0)      # pared en y = 5,0; bajo la plataforma (z = 14)
    corona = construir_diente(5.5, 15.0, 25.0)   # pared en y = 2,5; toda sobre la plataforma
    union = vtk.vtkAppendPolyData()
    union.AddInputData(raiz)
    union.AddInputData(corona)
    union.Update()
    diente_008 = union.GetOutput()
    verificar(diente_008, np.pi * RADIO_DIENTE ** 2 * (8.0 + 10.0),
              vertices_exactos=([0.0, 5.0, 5.0], [0.0, 2.5, 15.0]))
    escribir(diente_008, "diente_008.stl")

    print("arcada_cbct.stl / escaneo_corrido.stl")
    arcada = construir_arcada()
    verificar(arcada, sum(4 / 3 * np.pi * r ** 3 for _, r in CUSPIDES) * 0.99)
    escribir(arcada, "arcada_cbct.stl")
    escaneo = construir_escaneo_corrido(arcada)
    escribir(escaneo, "escaneo_corrido.stl")

    print("escaneo_arcada_recta.stl / dientes_arcada_recta.stl")
    escaneo_recto, dientes_recto = construir_arcada_recta()
    verificar(dientes_recto, 6 * 6 * 6 * 18)
    escribir(escaneo_recto, "escaneo_arcada_recta.stl")
    escribir(dientes_recto, "dientes_arcada_recta.stl")

    for nombre, radio in (("hueso_009.stl", 4.0), ("hueso_010.stl", 4.5)):
        print(nombre)
        hueso = construir_diente(0.5, 0.0, 20.0, radio=radio)
        verificar(hueso, np.pi * radio ** 2 * 20.0,
                  vertices_exactos=([0.0, 0.5 - radio, 0.0], [0.0, 0.5 + radio, 0.0]))
        escribir(hueso, nombre)

    print("guia/caso_017_undercut/diente_017.stl y guia_017_sin_recortar.stl")
    diente_017 = construir_perfil_extruido(DIENTE_017, -5.0, 5.0)
    verificar(diente_017, (20 * 2 + 6 * 6 + 8 * 4) * 10.0)
    escribir(diente_017, "guia/caso_017_undercut/diente_017.stl")
    guia_017 = construir_perfil_extruido(GUIA_017, -5.0, 5.0)
    verificar(guia_017, (12 * 8 - 6 * 2 - 8 * 4) * 10.0)
    escribir(guia_017, "guia/caso_017_undercut/guia_017_sin_recortar.stl")


if __name__ == "__main__":
    main()
