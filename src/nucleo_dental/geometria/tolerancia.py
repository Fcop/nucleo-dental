"""
tolerancia.py — Separacion guia-hueso a una tolerancia objetivo
===============================================================

Toma una guia generada en contacto exacto con el hueso (tolerancia 0) y le
quita material hasta dejar una separacion uniforme de `tolerancia` mm.

Dependencias: numpy + vtk (ambos incluidos en 3D Slicer).
Licencia: MIT.

Restriccion dura
----------------
Solo resta material. Nunca anade. La garantia no es empirica: el solido de
salida se define como la INTERSECCION de la guia con el exterior del hueso
dilatado, y una interseccion no puede ser mayor que ninguno de sus operandos.
Ver `_interseccion` para el detalle de por que la garantia sobrevive a
Marching Cubes.

Por que no se usa una dilatacion binaria
----------------------------------------
El camino evidente es dilatar el hueso y restarlo de la guia en el espacio de
voxeles, como hace `recortar_guia()` en undercut_core. Ese camino no puede
alcanzar la precision que se pide aqui, y el motivo no es la conectividad de
la dilatacion sino la binarizacion misma.

Con mascaras binarias el borde del solido solo puede caer en las posiciones
que la grilla permite. Sobre una interfaz plana, el desplazamiento efectivo
que produce un umbral sobre una mascara binaria vale (k - 0.5) * spacing con
k entero: con voxel de 0.25 mm los unicos offsets representables son 0.125,
0.375, 0.625... Pedir 0.20 mm devuelve 0.125 mm. El error es de cuantizacion
y no baja aunque la distancia se calcule con una transformada euclidiana
exacta en vez de una dilatacion 6-conectada: cambiar L1 por L2 corrige la
anisotropia (el "diamante") pero deja intacto el escalon.

Suavizar el volumen antes de Marching Cubes tampoco lo arregla: una gaussiana
es simetrica y deja la isosuperficie 0.5 donde estaba.

Metodo
------
La unica forma de obtener precision sub-voxel es no binarizar nunca. Se
construyen dos campos escalares CONTINUOS sobre una grilla comun:

    d_guia   distancia con signo a la superficie de la guia   (+ = dentro)
    d_hueso  distancia con signo a la superficie del hueso    (+ = dentro)

y el resultado es la isosuperficie 0 de

    F = min( d_guia , -d_hueso - tolerancia )

El segundo termino es positivo donde se esta a mas de `tolerancia` mm por
fuera del hueso. Marching Cubes interpola F linealmente sobre cada arista, de
modo que el cruce por cero cae en posiciones sub-voxel y el offset deja de
estar cuantizado.

La magnitud de cada campo la calcula vtkImplicitPolyDataDistance, que da la
distancia exacta punto-triangulo, muestreada sobre toda la grilla en C++ por
vtkSampleFunction. Desde Python no se recorre nada.

El signo NO se toma de esa distancia, porque VTK lo deduce de las normales de
la malla y las normales no son fiables (ver ajuste.py, `_cara_interna`). Se
obtiene de una voxelizacion por estencil, que resuelve dentro/fuera por
paridad de rayos y no depende de ninguna convencion de orientacion.
"""

from __future__ import annotations

import numpy as np
import vtk
from vtk.util import numpy_support


__all__ = [
    "aplicar_tolerancia",
    "campo_distancia",
    "volumen_malla",
    "leer_stl",
    "escribir_stl",
]


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def volumen_malla(pd: vtk.vtkPolyData) -> float:
    """Volumen en mm3 de una malla cerrada."""
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(pd)
    tri.Update()
    mp = vtk.vtkMassProperties()
    mp.SetInputData(tri.GetOutput())
    mp.Update()
    return mp.GetVolume()


def _triangular(pd: vtk.vtkPolyData) -> vtk.vtkPolyData:
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(pd)
    tri.PassLinesOff()
    tri.PassVertsOff()
    tri.Update()
    return tri.GetOutput()


def _grilla(pd: vtk.vtkPolyData, spacing: float, pad_mm: float):
    """Origen y dimensiones de una grilla que envuelve `pd` con margen."""
    b = pd.GetBounds()
    origin = [b[0] - pad_mm, b[2] - pad_mm, b[4] - pad_mm]
    dims = [
        int(np.ceil((b[1] - b[0] + 2 * pad_mm) / spacing)) + 1,
        int(np.ceil((b[3] - b[2] + 2 * pad_mm) / spacing)) + 1,
        int(np.ceil((b[5] - b[4] + 2 * pad_mm) / spacing)) + 1,
    ]
    return origin, dims


def _imagen_vacia(origin, dims, spacing: float) -> vtk.vtkImageData:
    img = vtk.vtkImageData()
    img.SetOrigin(origin)
    img.SetSpacing(spacing, spacing, spacing)
    img.SetDimensions(dims)
    return img


def _dentro(pd: vtk.vtkPolyData, origin, dims, spacing: float) -> np.ndarray:
    """Mascara booleana (z, y, x): True donde el centro del voxel cae DENTRO.

    Usa el estencil de VTK, que resuelve la pertenencia por paridad de
    intersecciones. No depende de la orientacion de las normales.
    """
    img = _imagen_vacia(origin, dims, spacing)
    img.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    img.GetPointData().GetScalars().Fill(1)

    sten = vtk.vtkPolyDataToImageStencil()
    sten.SetInputData(pd)
    sten.SetOutputOrigin(origin)
    sten.SetOutputSpacing(spacing, spacing, spacing)
    sten.SetOutputWholeExtent(img.GetExtent())
    sten.SetTolerance(0.0)
    sten.Update()

    st = vtk.vtkImageStencil()
    st.SetInputData(img)
    st.SetStencilData(sten.GetOutput())
    st.ReverseStencilOff()
    st.SetBackgroundValue(0)
    st.Update()

    arr = numpy_support.vtk_to_numpy(st.GetOutput().GetPointData().GetScalars())
    return arr.reshape(dims[2], dims[1], dims[0]) > 0


def _recortar_a_caja(pd: vtk.vtkPolyData, origin, dims, spacing: float,
                     margen: float) -> vtk.vtkPolyData:
    """Deja solo las caras de `pd` cercanas a la grilla.

    El hueso suele ser una malla mucho mas grande que la guia (una mandibula
    entera contra una ferula de 3 cm). Construir el localizador sobre toda la
    malla y consultarlo millones de veces cuesta caro sin aportar nada: las
    caras lejanas nunca son las mas proximas a un punto de la grilla.

    Recortar introduce error solo mas alla de `margen` mm, donde el campo deja
    de ser exacto y pasa a ser una sobreestimacion. Como el umbral de trabajo
    son decimas de milimetro, es inofensivo.
    """
    caja = vtk.vtkBox()
    caja.SetBounds(
        origin[0] - margen, origin[0] + (dims[0] - 1) * spacing + margen,
        origin[1] - margen, origin[1] + (dims[1] - 1) * spacing + margen,
        origin[2] - margen, origin[2] + (dims[2] - 1) * spacing + margen)

    ex = vtk.vtkExtractPolyDataGeometry()
    ex.SetInputData(pd)
    ex.SetImplicitFunction(caja)
    ex.ExtractInsideOn()
    ex.ExtractBoundaryCellsOn()
    ex.Update()
    out = ex.GetOutput()

    # si el recorte deja la malla vacia, algo no cuadra: mejor el original
    return out if out.GetNumberOfCells() > 0 else pd


def _dilatar(m: np.ndarray, radio: int) -> np.ndarray:
    """Dilatacion binaria 6-conectada, `radio` iteraciones (distancia L1).

    Solo se usa para delimitar en que voxeles hace falta calcular el campo
    exacto. Ahi la anisotropia de la L1 no molesta, porque una bola L1 de
    radio r esta CONTENIDA en la bola euclidiana de radio r: pedir de mas es
    conservador y lo unico que cuesta es tiempo de calculo.
    """
    if radio <= 0:
        return m.copy()
    out = m.copy()
    for _ in range(int(radio)):
        p = out.copy()
        p[1:] |= out[:-1]
        p[:-1] |= out[1:]
        p[:, 1:] |= out[:, :-1]
        p[:, :-1] |= out[:, 1:]
        p[:, :, 1:] |= out[:, :, :-1]
        p[:, :, :-1] |= out[:, :, 1:]
        out = p
    return out


def _banda(m: np.ndarray, r_dentro: int, r_fuera: int) -> np.ndarray:
    """Corona alrededor de la superficie de `m`: r_fuera hacia afuera,
    r_dentro hacia adentro. Radios en voxeles, medidos en L1."""
    return _dilatar(m, r_fuera) & _dilatar(~m, r_dentro)


def _distancia_absoluta(pd: vtk.vtkPolyData, origin, dims, spacing: float,
                        necesario: np.ndarray = None,
                        lejos: float = 1000.0,
                        bloque: int = 8) -> np.ndarray:
    """Distancia sin signo de cada nodo de la grilla a la superficie de `pd`.

    vtkImplicitPolyDataDistance da la distancia exacta punto-triangulo y
    vtkSampleFunction la evalua dentro de C++, sin bucles desde Python.

    `necesario` limita el calculo a los voxeles donde el valor exacto influye
    en la isosuperficie final; el resto recibe `lejos`. Sin esto se paga el
    precio completo en todo el volumen, y la consulta al localizador domina el
    tiempo total: en una guia tipica menos del 15 % de los nodos importan.

    El recorrido va por bloques porque vtkSampleFunction solo sabe muestrear
    cajas. El objeto implicito se construye UNA vez y se reutiliza en todos
    los bloques: su localizador es lo caro de montar.
    """
    imp = vtk.vtkImplicitPolyDataDistance()
    imp.SetInput(pd)

    nx, ny, nz = int(dims[0]), int(dims[1]), int(dims[2])
    out = np.full((nz, ny, nx), lejos, dtype=np.float32)

    def _muestrear(x0, x1, y0, y1, z0, z1):
        s = vtk.vtkSampleFunction()
        s.SetImplicitFunction(imp)
        s.SetSampleDimensions(x1 - x0, y1 - y0, z1 - z0)
        s.SetModelBounds(
            origin[0] + x0 * spacing, origin[0] + (x1 - 1) * spacing,
            origin[1] + y0 * spacing, origin[1] + (y1 - 1) * spacing,
            origin[2] + z0 * spacing, origin[2] + (z1 - 1) * spacing)
        s.SetOutputScalarTypeToFloat()
        s.ComputeNormalsOff()
        s.CappingOff()
        s.Update()
        a = numpy_support.vtk_to_numpy(
            s.GetOutput().GetPointData().GetScalars())
        return np.abs(a.reshape(z1 - z0, y1 - y0, x1 - x0))

    if necesario is None:
        return _muestrear(0, nx, 0, ny, 0, nz)

    for z0 in range(0, nz, bloque):
        z1 = min(z0 + bloque, nz)
        for y0 in range(0, ny, bloque):
            y1 = min(y0 + bloque, ny)
            for x0 in range(0, nx, bloque):
                x1 = min(x0 + bloque, nx)
                if not necesario[z0:z1, y0:y1, x0:x1].any():
                    continue
                # vtkSampleFunction necesita al menos 2 nodos por eje
                a0, b0, c0 = x0, y0, z0
                if x1 - a0 < 2:
                    a0 = x1 - 2
                if y1 - b0 < 2:
                    b0 = y1 - 2
                if z1 - c0 < 2:
                    c0 = z1 - 2
                d = _muestrear(a0, x1, b0, y1, c0, z1)
                out[c0:z1, b0:y1, a0:x1] = d
    return out


def campo_distancia(pd: vtk.vtkPolyData, origin, dims, spacing: float,
                    margen_recorte: float = 5.0,
                    necesario: np.ndarray = None,
                    dentro: np.ndarray = None,
                    lejos: float = 1000.0) -> np.ndarray:
    """Campo de distancia con signo sobre la grilla. Positivo DENTRO de `pd`.

    Magnitud exacta (distancia punto-triangulo) y signo por estencil. Los dos
    ingredientes son independientes: ningun error de orientacion de normales
    puede invertir el campo.

    Fuera de `necesario` el campo vale +-`lejos` con el signo correcto. Es
    suficiente: Marching Cubes solo interpola aristas que cambian de signo, y
    la banda se dimensiona para contener todos esos cruces.
    """
    pd = _triangular(pd)
    if dentro is None:
        dentro = _dentro(pd, origin, dims, spacing)

    cercano = _recortar_a_caja(pd, origin, dims, spacing, margen_recorte)
    d = _distancia_absoluta(cercano, origin, dims, spacing, necesario, lejos)

    return np.where(dentro, d, -d).astype(np.float32)


def _superficie(campo: np.ndarray, origin, spacing: float,
                suavizado: int = 0) -> vtk.vtkPolyData:
    """Isosuperficie 0 del campo continuo.

    No hay antialiasing ni terraceo que corregir: el campo nunca fue binario,
    asi que la superficie ya sale suave y en posiciones sub-voxel. Por eso el
    suavizado de malla viene apagado por defecto — mover vertices despues solo
    puede degradar la separacion que se acaba de imponer.
    """
    nz, ny, nx = campo.shape

    # el borde de la grilla debe quedar "fuera" para que la malla cierre
    campo = campo.copy()
    campo[0, :, :] = campo[-1, :, :] = -spacing
    campo[:, 0, :] = campo[:, -1, :] = -spacing
    campo[:, :, 0] = campo[:, :, -1] = -spacing

    img = _imagen_vacia(origin, (nx, ny, nz), spacing)
    img.GetPointData().SetScalars(numpy_support.numpy_to_vtk(
        np.ascontiguousarray(campo.ravel(order="C"), dtype=np.float32),
        deep=True, array_type=vtk.VTK_FLOAT))

    fe = vtk.vtkFlyingEdges3D()
    fe.SetInputData(img)
    fe.SetValue(0, 0.0)
    fe.ComputeNormalsOn()
    fe.Update()
    out = fe.GetOutput()

    if suavizado > 0:
        sm = vtk.vtkWindowedSincPolyDataFilter()
        sm.SetInputData(out)
        sm.SetNumberOfIterations(int(suavizado))
        sm.SetPassBand(0.1)
        sm.BoundarySmoothingOff()
        sm.FeatureEdgeSmoothingOff()
        sm.NonManifoldSmoothingOn()
        sm.NormalizeCoordinatesOn()
        sm.Update()
        out = sm.GetOutput()

    return out


def _interseccion(d_guia: np.ndarray, d_hueso: np.ndarray,
                  tolerancia: float) -> np.ndarray:
    """F = min(d_guia, -d_hueso - tolerancia).

    Por que esto garantiza que solo se resta material, incluso despues de
    Marching Cubes: en cada nodo F <= d_guia. A lo largo de una arista de la
    grilla, la interpolacion lineal de un minimo es menor o igual que el
    minimo de las interpolaciones, de modo que el cruce por cero de F cae
    siempre del lado interior del cruce por cero de d_guia. La superficie de
    salida queda contenida en la de entrada por construccion, no por suerte.
    """
    return np.minimum(d_guia, -d_hueso - float(tolerancia))


# ---------------------------------------------------------------------------
# Funcion publica
# ---------------------------------------------------------------------------

def aplicar_tolerancia(guia: vtk.vtkPolyData,
                       hueso: vtk.vtkPolyData,
                       tolerancia: float = 0.20,
                       spacing: float = 0.20,
                       suavizado: int = 0,
                       margen_recorte: float = 5.0,
                       memoria_max_mb: float = 2000.0,
                       devolver_diagnostico: bool = False):
    """Separa la guia del hueso `tolerancia` mm quitando material.

    Es la tercera operacion de la familia, y conviene no confundirlas:

        eliminar_undercuts()  -> anade material al HUESO   (blockout previo)
        recortar_guia()       -> quita material de la GUIA (eje de insercion)
        aplicar_tolerancia()  -> quita material de la GUIA (juego de ajuste)

    Esta no sabe nada del eje de insercion: es un offset puro, igual en todas
    direcciones. Si la guia ademas traba por la falda, eso lo resuelve
    recortar_guia(); las dos operaciones son independientes y componibles.

    Parametros
    ----------
    tolerancia : mm de separacion buscada entre la cara de asiento y el hueso.
        0.10-0.15 para ajuste firme, 0.20-0.30 para uso clinico habitual con
        resina impresa. A diferencia de `holgura` en recortar_guia(), aqui no
        hay minimo tecnico de un voxel: el campo es continuo y una tolerancia
        menor que el voxel se representa igual de bien.
    spacing : tamano de voxel del campo, en mm. Controla la fidelidad con que
        se reconstruye la guia, NO la precision del offset. 0.20 es un buen
        compromiso; 0.15 si la guia lleva detalles finos.
    suavizado : iteraciones de windowed-sinc sobre la malla final. Dejar en 0.
        Cualquier valor > 0 mueve vertices despues de imponer la separacion y
        puede reintroducir contacto.
    margen_recorte : mm de hueso alrededor de la grilla que se conservan para
        calcular distancias. Subirlo solo hace el calculo mas lento.
    memoria_max_mb : corta con un error claro antes de intentar reservar una
        grilla que no cabe, en vez de dejar a Slicer sin memoria.

    Devuelve
    --------
    vtkPolyData de la guia separada. Con devolver_diagnostico=True devuelve
    (malla, dict).

    Nota sobre la geometria lejana
    ------------------------------
    La guia entera se reconstruye desde el campo, no solo la cara de asiento.
    Las superficies lejos del hueso se reproducen con un error del orden de
    spacing^2 por la interpolacion trilineal: irrelevante en la carcasa, pero
    perceptible en aristas vivas. Conviene aplicar la tolerancia ANTES de
    anadir camisas de fresado, o bajar `spacing` si ya estan puestas.
    """
    import time

    if tolerancia < 0:
        raise ValueError("La tolerancia no puede ser negativa: esta funcion "
                         "solo resta material.")
    if guia is None or guia.GetNumberOfPoints() == 0:
        raise ValueError("La guia esta vacia.")
    if hueso is None or hueso.GetNumberOfPoints() == 0:
        raise ValueError("El hueso esta vacio.")

    t0 = time.time()

    # la grilla envuelve solo a la GUIA: es lo unico que hay que reconstruir.
    # el hueso interviene a traves del campo de distancia, no de su tamano.
    pad = max(3 * spacing, 2 * spacing + tolerancia)
    origin, dims = _grilla(guia, spacing, pad)

    n = dims[0] * dims[1] * dims[2]
    mb = n * 4 / 1e6
    if 3 * mb > memoria_max_mb:
        raise MemoryError(
            f"La grilla seria de {dims[0]}x{dims[1]}x{dims[2]} = {n/1e6:.1f} M "
            f"nodos ({3*mb:.0f} MB). Sube `spacing` o sube `memoria_max_mb` "
            f"si el equipo lo aguanta.")

    # ---- donde hace falta el campo exacto.
    # el factor 2 convierte un radio euclidiano en uno L1 con holgura: la
    # dilatacion 6-conectada avanza mas despacio en las diagonales.
    m_guia = _dentro(_triangular(guia), origin, dims, spacing)
    m_hueso = _dentro(_triangular(hueso), origin, dims, spacing)

    n_tol = int(np.ceil(tolerancia / spacing))
    banda_guia = _banda(m_guia, 3, 3)
    # el campo del hueso solo influye donde puede quedar material de la guia,
    # y solo hasta `tolerancia` mas un par de voxeles por fuera del hueso
    banda_hueso = (_banda(m_hueso, 3, 2 * n_tol + 4)
                   & _dilatar(m_guia, 3))

    d_guia = campo_distancia(guia, origin, dims, spacing, margen_recorte,
                             banda_guia, m_guia)
    t_guia = time.time() - t0

    d_hueso = campo_distancia(hueso, origin, dims, spacing, margen_recorte,
                              banda_hueso, m_hueso)
    t_hueso = time.time() - t0 - t_guia
    frac = float(banda_guia.mean() + banda_hueso.mean()) / 2.0
    del m_guia, m_hueso, banda_guia, banda_hueso

    campo = _interseccion(d_guia, d_hueso, tolerancia)
    del d_guia

    pd_out = _superficie(campo, origin, spacing, suavizado)
    t_total = time.time() - t0

    if pd_out.GetNumberOfPoints() == 0:
        raise RuntimeError(
            f"La tolerancia de {tolerancia} mm no deja nada de la guia. "
            f"Comprueba que guia y hueso esten en el mismo sistema de "
            f"coordenadas y que la tolerancia sea razonable.")

    if not devolver_diagnostico:
        return pd_out

    # ---- verificacion interna: separacion medida sobre la malla de salida.
    # se sondea el campo del hueso en los vertices del resultado. Es un
    # control independiente del re-mallado (detecta errores de Marching
    # Cubes), no un sustituto de ajuste.analizar_ajuste().
    img = _imagen_vacia(origin, dims, spacing)
    img.GetPointData().SetScalars(numpy_support.numpy_to_vtk(
        np.ascontiguousarray((-d_hueso).ravel(order="C"), dtype=np.float32),
        deep=True, array_type=vtk.VTK_FLOAT))
    pr = vtk.vtkProbeFilter()
    pr.SetInputData(pd_out)
    pr.SetSourceData(img)
    pr.Update()
    sep = numpy_support.vtk_to_numpy(
        pr.GetOutput().GetPointData().GetScalars()).astype(np.float64)

    # la cara de asiento es la banda cercana al hueso; la externa esta a un
    # grosor de carcasa y solo diluiria la estadistica
    banda = sep <= tolerancia + 0.3
    v0 = volumen_malla(guia)
    v1 = volumen_malla(pd_out)

    diag = {
        "tolerancia_mm": float(tolerancia),
        "spacing_mm": float(spacing),
        "dims": tuple(int(x) for x in dims),
        "nodos_grilla": int(n),
        "fraccion_nodos_evaluados": round(frac, 4),
        "volumen_guia_original_mm3": v0,
        "volumen_resultado_mm3": v1,
        "volumen_removido_mm3": v0 - v1,
        "porcentaje_removido": 100.0 * (v0 - v1) / v0 if v0 > 0 else 0.0,
        "separacion_min_mm": float(sep.min()),
        "separacion_mediana_asiento_mm": float(np.median(sep[banda]))
                                          if banda.any() else float("nan"),
        "puntos_cara_asiento": int(banda.sum()),
        "solo_resta": bool(v1 <= v0),
        "segundos_campo_guia": round(t_guia, 2),
        "segundos_campo_hueso": round(t_hueso, 2),
        "segundos_total": round(t_total, 2),
    }
    return pd_out, diag


# ---------------------------------------------------------------------------
# E/S de conveniencia
# ---------------------------------------------------------------------------

def leer_stl(ruta: str) -> vtk.vtkPolyData:
    r = vtk.vtkSTLReader()
    r.SetFileName(ruta)
    r.Update()
    return r.GetOutput()


def escribir_stl(pd: vtk.vtkPolyData, ruta: str) -> None:
    w = vtk.vtkSTLWriter()
    w.SetFileName(ruta)
    w.SetInputData(pd)
    w.SetFileTypeToBinary()
    w.Write()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Separa una guia de su hueso una tolerancia dada, "
                    "quitando material.")
    ap.add_argument("guia", help="STL de la guia (contacto exacto)")
    ap.add_argument("hueso", help="STL del modelo oseo")
    ap.add_argument("salida", help="STL de salida")
    ap.add_argument("--tolerancia", type=float, default=0.20,
                    help="separacion buscada en mm (def: 0.20)")
    ap.add_argument("--spacing", type=float, default=0.20,
                    help="tamano de voxel del campo en mm (def: 0.20)")
    args = ap.parse_args()

    g, h = leer_stl(args.guia), leer_stl(args.hueso)
    out, d = aplicar_tolerancia(g, h, tolerancia=args.tolerancia,
                                spacing=args.spacing,
                                devolver_diagnostico=True)
    escribir_stl(out, args.salida)

    print(f"  tolerancia       : {d['tolerancia_mm']} mm")
    print(f"  voxel del campo  : {d['spacing_mm']} mm")
    print(f"  volumen guia     : {d['volumen_guia_original_mm3']:.1f} -> "
          f"{d['volumen_resultado_mm3']:.1f} mm3")
    print(f"  material removido: {d['volumen_removido_mm3']:.1f} mm3 "
          f"({d['porcentaje_removido']:.1f} %)")
    print(f"  separacion minima: {d['separacion_min_mm']:.3f} mm")
    print(f"  mediana asiento  : {d['separacion_mediana_asiento_mm']:.3f} mm")
    print(f"  tiempo           : {d['segundos_total']:.1f} s")
    print(f"  guardado en      : {args.salida}")
