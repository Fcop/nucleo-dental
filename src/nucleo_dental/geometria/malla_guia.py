"""
malla_guia.py — Construccion de la malla de la guia de corte
============================================================

Cubre las etapas que van desde el modelo oseo hasta el solido de la guia:

    1. seleccion de la region de apoyo   -> curva cerrada + pincel
    2. extraccion y suavizado del parche
    3. generacion del solido por offset  -> intrados + extrados + costillas
    4. ensamblado CSG de las features    -> ranuras, camisas, mango, grabado
    5. exportacion

Las etapas relacionadas con el eje de insercion y con el juego de ajuste NO se
implementan aqui: se delegan en los modulos ya existentes.

    undercut_core.eliminar_undercuts()  blockout previo sobre el HUESO
    undercut_core.recortar_guia()       desbloqueo posterior sobre la GUIA
    tolerancia.aplicar_tolerancia()     separacion guia-hueso sub-voxel
    ajuste.analizar_ajuste()            control de calidad previo a imprimir

Dependencias: numpy + vtk + los modulos citados. Sin dependencias de Slicer
(R-001): la recoleccion de features y la escena quedan en la capa de interfaz.
Licencia: MIT.

Convencion del eje
------------------
Se usa la misma que undercut_core: `eje` es la direccion en que la guia SE
RETIRA del hueso, apuntando hacia afuera.
"""

from __future__ import annotations

import numpy as np
import vtk
from vtk.util import numpy_support

from . import undercut_core as uc
from . import tolerancia as tl


__all__ = [
    "ATTR_ROL", "ATTR_ORDEN", "ATTR_GUIA", "ROL_POSITIVO", "ROL_NEGATIVO",
    "SelectorRegion",
    "extraer_parche",
    "suavizar_parche",
    "contar_bucles_borde",
    "parche_a_solido",
    "booleana",
    "ensamblar_guia",
    "validar_guia",
    "exportar_stl",
]


# ---------------------------------------------------------------------------
# CONTRATO DE INTEGRACION — features CSG
# ---------------------------------------------------------------------------
#
# ensamblar_guia() recibe las features como listas explicitas de
# (nombre, vtkPolyData), ya ordenadas. La capa de interfaz (p. ej. la extension
# de Slicer) es quien las recolecta; para hacerlo puede etiquetar sus modelos
# con estos atributos:
#
#     nodo.SetAttribute(ATTR_ROL,   ROL_POSITIVO | ROL_NEGATIVO)
#     nodo.SetAttribute(ATTR_ORDEN, "<entero>")      # opcional, def. 100
#     nodo.SetAttribute(ATTR_GUIA,  "<id de guia>")  # opcional, para filtrar
#
# ROL_POSITIVO -> se une antes de restar los negativos
#                 (collares de ranura, camisas de fresado, mango, aletas)
# ROL_NEGATIVO -> se resta al final
#                 (ranura de sierra, lumen de fresado, grabado de texto)
#
# El solido debe ser CERRADO. No hace falta que sea manifold perfecto: la
# booleana es voxelizada y tolera defectos.
#
# RESTRICCION IMPORTANTE: los positivos no deben tocar el hueso. La tolerancia
# se aplica antes de unirlos (ver ensamblar_guia), asi que cualquier positivo
# que invada el hueso lo hara sin juego de ajuste. Si necesitas un positivo en
# contacto, pasalo por tl.aplicar_tolerancia() individualmente antes de
# etiquetarlo.
# ---------------------------------------------------------------------------

ATTR_ROL = "GuiaCorte.Rol"
ATTR_ORDEN = "GuiaCorte.Orden"
ATTR_GUIA = "GuiaCorte.IdGuia"

ROL_POSITIVO = "positivo"
ROL_NEGATIVO = "negativo"

ARRAY_REGION = "RegionGuia"


# ---------------------------------------------------------------------------
# 1. REGION DE APOYO — curva cerrada y pincel sobre un unico campo escalar
# ---------------------------------------------------------------------------

class SelectorRegion:
    """Mantiene el campo escalar RegionGuia (0/1) sobre los puntos del hueso.

    La curva inicializa el campo; el pincel lo modifica. Ambos escriben sobre
    la misma estructura, de modo que se puede rehacer la curva y recuperar los
    retoques con repetir_trazos().
    """

    def __init__(self, nodo_hueso):
        self.nodo_hueso = nodo_hueso
        self.pd = nodo_hueso.GetPolyData()
        if self.pd is None or self.pd.GetNumberOfPoints() == 0:
            raise ValueError("El modelo de hueso no contiene geometria.")
        self._localizador = None
        self.trazos = []                       # (punto_mundo, radio, borrar)
        self._preparar_array()
        self._preparar_normales()

    # -- infraestructura ----------------------------------------------------

    def _preparar_array(self):
        pdata = self.pd.GetPointData()
        arr = pdata.GetArray(ARRAY_REGION)
        if arr is None:
            arr = vtk.vtkFloatArray()
            arr.SetName(ARRAY_REGION)
            arr.SetNumberOfComponents(1)
            arr.SetNumberOfTuples(self.pd.GetNumberOfPoints())
            arr.Fill(0.0)
            pdata.AddArray(arr)
        self.array_region = arr

    def _preparar_normales(self):
        if self.pd.GetPointData().GetNormals() is None:
            nf = vtk.vtkPolyDataNormals()
            nf.SetInputData(self.pd)
            nf.ComputePointNormalsOn()
            nf.ComputeCellNormalsOff()
            nf.ConsistencyOn()
            nf.AutoOrientNormalsOn()
            nf.SplittingOff()
            nf.Update()
            self.pd.GetPointData().SetNormals(
                nf.GetOutput().GetPointData().GetNormals())
        self._normales = numpy_support.vtk_to_numpy(
            self.pd.GetPointData().GetNormals())

    def _loc(self):
        if self._localizador is None:
            self._localizador = vtk.vtkPointLocator()
            self._localizador.SetDataSet(self.pd)
            self._localizador.BuildLocator()
        return self._localizador

    def _region(self):
        return numpy_support.vtk_to_numpy(self.array_region)

    # -- curva cerrada ------------------------------------------------------

    def region_desde_curva(self, nodo_curva, punto_interior=None):
        """Inicializa la region con el interior de una curva cerrada.

        La curva debe estar pegada a la superficie. Lo mas fiable es
        configurarla en Markups con:
            Curve type         = "shortest distance on surface"
            Constrain to Model = <modelo de hueso>

        punto_interior : coordenada mundo dentro de la region deseada. Si es
            None se toma la region de menor area, que en una mandibula
            completa casi nunca es la que quieres.
        """
        pts_curva = nodo_curva.GetCurvePointsWorld()
        if pts_curva is None or pts_curva.GetNumberOfPoints() < 3:
            raise ValueError("La curva necesita al menos 3 puntos.")

        # submuestreo: vtkSelectPolyData degrada mucho con loops muy densos
        n = pts_curva.GetNumberOfPoints()
        paso = max(1, n // 400)
        loop = vtk.vtkPoints()
        for i in range(0, n, paso):
            loop.InsertNextPoint(pts_curva.GetPoint(i))

        sel = vtk.vtkSelectPolyData()
        sel.SetInputData(self.pd)
        sel.SetLoop(loop)
        sel.GenerateSelectionScalarsOn()
        sel.GenerateUnselectedOutputOff()
        if punto_interior is not None:
            sel.SetSelectionModeToClosestPointRegion()
            sel.SetClosestPoint(punto_interior)
        else:
            sel.SetSelectionModeToSmallestRegion()
        sel.Update()

        escalares = sel.GetOutput().GetPointData().GetScalars()
        if escalares is None:
            raise RuntimeError(
                "vtkSelectPolyData no genero escalares. Causa habitual: la "
                "curva no esta suficientemente pegada a la superficie.")

        # convencion de vtkSelectPolyData: valores negativos = interior
        v = numpy_support.vtk_to_numpy(escalares)
        self._region()[:] = (v < 0.0).astype(np.float32)
        self.array_region.Modified()
        self.pd.Modified()

    # -- pincel -------------------------------------------------------------

    def pintar(self, punto_mundo, radio, borrar=False, registrar=True):
        """Trazo de pincel esferico restringido a la cara visible.

        El filtro por normal evita que el pincel atraviese cortical delgada y
        pinte la cara opuesta. Es un pincel euclideo, no geodesico; para
        geodesico real usa la herramienta "Select by points" de Dynamic
        Modeler con SelectionAlgorithm=GeodesicDistance, mas exacto en
        superficies de curvatura fuerte pero notablemente mas lento por trazo.
        """
        loc = self._loc()
        ids = vtk.vtkIdList()
        loc.FindPointsWithinRadius(radio, punto_mundo, ids)
        if ids.GetNumberOfIds() == 0:
            return 0

        n_semilla = self._normales[loc.FindClosestPoint(punto_mundo)]
        region = self._region()
        valor = 0.0 if borrar else 1.0

        idx = np.array([ids.GetId(k) for k in range(ids.GetNumberOfIds())])
        alineados = self._normales[idx] @ n_semilla > 0.0
        region[idx[alineados]] = valor

        if registrar:
            self.trazos.append(
                (tuple(punto_mundo), float(radio), bool(borrar)))
        self.array_region.Modified()
        self.pd.Modified()
        return int(alineados.sum())

    def repetir_trazos(self):
        """Reaplica el historial. Se usa tras rehacer la curva."""
        for punto, radio, borrar in list(self.trazos):
            self.pintar(punto, radio, borrar=borrar, registrar=False)

    def limpiar_trazos(self):
        self.trazos = []

    def parche(self, solo_mayor=True):
        return extraer_parche(self.pd, solo_mayor=solo_mayor)


# ---------------------------------------------------------------------------
# 2. EXTRACCION Y SUAVIZADO DEL PARCHE
# ---------------------------------------------------------------------------

def extraer_parche(pd_hueso: vtk.vtkPolyData,
                   solo_mayor: bool = True) -> vtk.vtkPolyData:
    """Extrae el parche abierto definido por el array RegionGuia."""
    arr = pd_hueso.GetPointData().GetArray(ARRAY_REGION)
    if arr is None:
        raise ValueError("El modelo no tiene el array %s." % ARRAY_REGION)
    if not (numpy_support.vtk_to_numpy(arr) > 0.5).any():
        raise ValueError("La region de apoyo esta vacia.")

    # Se selecciona en numpy y no con vtkThreshold a proposito. La API de
    # vtkThreshold cambio dos veces entre VTK 9.0 y 9.6 (ThresholdByUpper ->
    # SetLowerThreshold + SetThresholdFunction, y luego la semantica de los
    # enums), y en 9.6 la combinacion documentada devuelve cero celdas sin
    # emitir ningun error. Un criterio de tres comparaciones no justifica
    # depender de eso.
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(pd_hueso)
    tri.PassLinesOff()
    tri.PassVertsOff()
    tri.Update()
    pd = tri.GetOutput()

    region = numpy_support.vtk_to_numpy(
        pd.GetPointData().GetArray(ARRAY_REGION)) > 0.5

    conectividad = numpy_support.vtk_to_numpy(pd.GetPolys().GetData())
    if conectividad.size % 4 != 0:
        raise RuntimeError("La malla no quedo triangulada tras vtkTriangleFilter.")
    conectividad = conectividad.reshape(-1, 4)
    if not (conectividad[:, 0] == 3).all():
        raise RuntimeError("Se esperaban solo triangulos en la malla del hueso.")
    triangulos = conectividad[:, 1:]

    # solo celdas con SUS TRES puntos dentro de la region: deja el borde limpio
    dentro = region[triangulos].all(axis=1)
    if not dentro.any():
        raise ValueError(
            "Ningun triangulo tiene sus tres vertices dentro de la region. "
            "La seleccion es demasiado fina para la densidad de la malla.")
    seleccionados = triangulos[dentro].astype(np.int64)

    n = len(seleccionados)
    celdas = vtk.vtkCellArray()
    desplazamientos = (np.arange(n + 1, dtype=np.int64) * 3)
    celdas.SetData(
        numpy_support.numpy_to_vtkIdTypeArray(desplazamientos, deep=True),
        numpy_support.numpy_to_vtkIdTypeArray(
            np.ascontiguousarray(seleccionados.ravel()), deep=True))

    parche = vtk.vtkPolyData()
    parche.SetPoints(pd.GetPoints())
    parche.SetPolys(celdas)
    parche.GetPointData().PassData(pd.GetPointData())

    if solo_mayor:
        conn = vtk.vtkPolyDataConnectivityFilter()
        conn.SetInputData(parche)
        conn.SetExtractionModeToLargestRegion()
        conn.Update()
        parche = conn.GetOutput()

    lim = vtk.vtkCleanPolyData()
    lim.SetInputData(parche)
    lim.Update()
    return lim.GetOutput()


def suavizar_parche(parche: vtk.vtkPolyData, iteraciones: int = 20,
                    pass_band: float = 0.1,
                    suavizar_borde: bool = True) -> vtk.vtkPolyData:
    """Suavizado Taubin (windowed sinc). No encoge, a diferencia del Laplaciano.

    suavizar_borde=True redondea el contorno de la guia, mas comodo para el
    tejido blando. Desactivalo si el borde debe seguir exactamente la curva.

    Se aplica SIEMPRE antes de dar volumen: suavizar despues deforma el
    intrados y la guia deja de asentar.
    """
    sm = vtk.vtkWindowedSincPolyDataFilter()
    sm.SetInputData(parche)
    sm.SetNumberOfIterations(int(iteraciones))
    sm.SetPassBand(float(pass_band))
    sm.SetBoundarySmoothing(bool(suavizar_borde))
    sm.NonManifoldSmoothingOn()
    sm.NormalizeCoordinatesOn()
    sm.FeatureEdgeSmoothingOff()
    sm.Update()
    return sm.GetOutput()


def _normales(pd: vtk.vtkPolyData, auto_orientar: bool = False,
              invertir: bool = False) -> vtk.vtkPolyData:
    nf = vtk.vtkPolyDataNormals()
    nf.SetInputData(pd)
    nf.ComputePointNormalsOn()
    nf.ComputeCellNormalsOff()
    nf.ConsistencyOn()
    nf.SetAutoOrientNormals(auto_orientar)
    nf.SetFlipNormals(invertir)
    nf.SplittingOff()
    nf.Update()
    return nf.GetOutput()


def _generar_ids_punto(pd: vtk.vtkPolyData, nombre_array: str) -> vtk.vtkPolyData:
    """Copia de `pd` con un array de ids de punto originales.

    VTK 9.3 (Slicer 5.6+) deprecó vtkIdFilter en favor de vtkGenerateIds, y en
    algunos builds la clase vieja ya no queda expuesta en el wrapping de
    Python -- de ahi el AttributeError. Se prueba primero el filtro nuevo y
    se cae al viejo si no esta disponible, para que el modulo funcione igual
    en Slicer 5.4 y en 5.6+.
    """
    if hasattr(vtk, "vtkGenerateIds"):
        idf = vtk.vtkGenerateIds()
        idf.SetInputData(pd)
        idf.PointIdsOn()
        idf.CellIdsOff()
        idf.SetPointIdsArrayName(nombre_array)
    elif hasattr(vtk, "vtkIdFilter"):
        idf = vtk.vtkIdFilter()
        idf.SetInputData(pd)
        idf.PointIdsOn()
        idf.CellIdsOff()
        try:
            idf.SetPointIdsArrayName(nombre_array)
        except AttributeError:                  # VTK < 9
            idf.SetIdsArrayName(nombre_array)
    else:
        raise RuntimeError(
            "Ni vtkGenerateIds ni vtkIdFilter estan disponibles en esta "
            "build de VTK. Revisa la version de Slicer.")
    idf.Update()
    return idf.GetOutput()


def _aristas_borde_con_ids(pd: vtk.vtkPolyData) -> vtk.vtkPolyData:
    """Aristas de borde conservando los ids de punto originales."""
    con_ids = _generar_ids_punto(pd, "idsOrig")

    fe = vtk.vtkFeatureEdges()
    fe.SetInputData(con_ids)
    fe.BoundaryEdgesOn()
    fe.FeatureEdgesOff()
    fe.NonManifoldEdgesOff()
    fe.ManifoldEdgesOff()
    fe.ColoringOff()
    fe.Update()
    return fe.GetOutput()


def contar_bucles_borde(pd: vtk.vtkPolyData) -> int:
    """Numero de bucles de borde cerrados. Un parche sano tiene exactamente 1."""
    aristas = _aristas_borde_con_ids(pd)
    if aristas.GetNumberOfCells() == 0:
        return 0
    conn = vtk.vtkPolyDataConnectivityFilter()
    conn.SetInputData(aristas)
    conn.SetExtractionModeToAllRegions()
    conn.Update()
    return conn.GetNumberOfExtractedRegions()


# ---------------------------------------------------------------------------
# 3. PARCHE -> SOLIDO
# ---------------------------------------------------------------------------

def parche_a_solido(parche: vtk.vtkPolyData,
                    grosor: float) -> vtk.vtkPolyData:
    """Convierte un parche abierto en un solido cerrado de grosor constante.

        intrados  el parche original, con winding invertido
        extrados  el parche desplazado `grosor` a lo largo de las normales
        costillas dos triangulos por arista de borde, cosiendo ambas capas

    Como intrados y extrados comparten numeracion de puntos, el cosido es
    exacto y no depende de reordenar el bucle de borde.

    NOTA sobre socavados: esta funcion no sabe nada del eje de insercion. El
    desbloqueo se resuelve fuera, con undercut_core, por cualquiera de las dos
    vias descritas en ensamblar_guia().
    """
    if grosor <= 0:
        raise ValueError("El grosor debe ser positivo.")

    parche = _normales(parche, auto_orientar=False)

    n_bucles = contar_bucles_borde(parche)
    if n_bucles != 1:
        raise RuntimeError(
            "El parche tiene %d bucles de borde (se esperaba 1). Suele "
            "indicar agujeros interiores en la seleccion: usa el pincel en "
            "modo aditivo para cerrarlos." % n_bucles)

    # extrados: desplazamiento a lo largo de la normal
    parche.GetPointData().SetActiveVectors("Normals")
    warp = vtk.vtkWarpVector()
    warp.SetInputData(parche)
    warp.SetScaleFactor(float(grosor))
    warp.Update()
    externo = warp.GetOutput()

    n_pts = parche.GetNumberOfPoints()

    puntos = vtk.vtkPoints()
    puntos.SetNumberOfPoints(n_pts * 2)
    for i in range(n_pts):
        puntos.SetPoint(i, parche.GetPoint(i))
        puntos.SetPoint(i + n_pts, externo.GetPoint(i))

    celdas = vtk.vtkCellArray()

    for c in range(parche.GetNumberOfCells()):
        celda = parche.GetCell(c)
        if celda.GetNumberOfPoints() != 3:
            continue
        ids = [celda.GetPointId(k) for k in range(3)]
        # intrados: winding invertido para que la normal salga del solido
        celdas.InsertNextCell(3)
        for pid in reversed(ids):
            celdas.InsertCellPoint(pid)
        # extrados: winding original, desplazado
        celdas.InsertNextCell(3)
        for pid in ids:
            celdas.InsertCellPoint(pid + n_pts)

    aristas = _aristas_borde_con_ids(parche)
    ids_orig = aristas.GetPointData().GetArray("idsOrig")
    if ids_orig is None:
        raise RuntimeError("No se pudieron recuperar los ids de borde.")
    for c in range(aristas.GetNumberOfCells()):
        celda = aristas.GetCell(c)
        if celda.GetNumberOfPoints() != 2:
            continue
        a = int(ids_orig.GetTuple1(celda.GetPointId(0)))
        b = int(ids_orig.GetTuple1(celda.GetPointId(1)))
        celdas.InsertNextCell(3)
        celdas.InsertCellPoint(a)
        celdas.InsertCellPoint(b)
        celdas.InsertCellPoint(b + n_pts)
        celdas.InsertNextCell(3)
        celdas.InsertCellPoint(a)
        celdas.InsertCellPoint(b + n_pts)
        celdas.InsertCellPoint(a + n_pts)

    solido = vtk.vtkPolyData()
    solido.SetPoints(puntos)
    solido.SetPolys(celdas)

    lim = vtk.vtkCleanPolyData()
    lim.SetInputData(solido)
    lim.Update()

    # ya es cerrado, asi que AutoOrientNormals puede resolver la orientacion
    return _normales(lim.GetOutput(), auto_orientar=True)


# ---------------------------------------------------------------------------
# 4. BOOLEANA VOXELIZADA
# ---------------------------------------------------------------------------
#
# Se voxeliza en lugar de usar booleanas de malla (vtkbool) porque las mallas
# de origen anatomico traen auto-intersecciones y aristas no manifold que las
# hacen fallar.
#
# AQUI NO HAY PARAMETRO DE HOLGURA, a proposito. Desplazar una superficie
# dilatando una mascara binaria cuantiza el offset a (k-0.5)*spacing, tal como
# esta documentado en tolerancia.py. Todo juego de ajuste se resuelve con
# tl.aplicar_tolerancia(), que trabaja sobre campos continuos y alcanza
# precision sub-voxel.
# ---------------------------------------------------------------------------

def _grilla_comun(mallas, spacing: float, pad: float = None):
    if pad is None:
        pad = 3 * spacing
    lo = [1e30, 1e30, 1e30]
    hi = [-1e30, -1e30, -1e30]
    for pd in mallas:
        b = pd.GetBounds()
        for i in range(3):
            lo[i] = min(lo[i], b[2 * i] - pad)
            hi[i] = max(hi[i], b[2 * i + 1] + pad)
    dims = [int(np.ceil((hi[i] - lo[i]) / spacing)) + 1 for i in range(3)]
    return lo, dims


def booleana(a: vtk.vtkPolyData, b: vtk.vtkPolyData, operacion: str,
             spacing: float = 0.15, suavizado: int = 0,
             antialias: float = 0.8) -> vtk.vtkPolyData:
    """Booleana voxelizada entre dos solidos cerrados.

    operacion : "union" | "diferencia" | "interseccion"
    antialias : gaussiana sobre el volumen antes de Marching Cubes, en
        voxeles. Es lo que elimina el terraceo; suavizar la malla despues
        apenas lo reduce (ver undercut_core._superficie).
    """
    lo, dims = _grilla_comun([a, b], spacing)
    va = uc._voxelizar_en_grilla(a, lo, dims, spacing) > 0
    vb = uc._voxelizar_en_grilla(b, lo, dims, spacing) > 0

    if operacion == "union":
        r = va | vb
    elif operacion == "diferencia":
        r = va & ~vb
    elif operacion == "interseccion":
        r = va & vb
    else:
        raise ValueError("Operacion desconocida: %s" % operacion)

    if not r.any():
        raise RuntimeError(
            "El resultado de la booleana '%s' esta vacio." % operacion)

    return uc._superficie(r.astype(np.uint8), lo, spacing,
                          suavizado, antialias)


# ---------------------------------------------------------------------------
# 5. ENSAMBLADO
# ---------------------------------------------------------------------------

def ensamblar_guia(carcasa: vtk.vtkPolyData,
                   hueso: vtk.vtkPolyData,
                   eje=(0, 0, 1),
                   tolerancia: float = 0.20,
                   spacing: float = 0.15,
                   spacing_tolerancia: float = 0.20,
                   desbloqueo: str = "recortar",
                   positivos=(),
                   negativos=(),
                   callback=None) -> vtk.vtkPolyData:
    """Ejecuta la cadena completa en el orden correcto.

        1. desbloqueo del eje de insercion   (undercut_core)
        2. juego de ajuste                   (tolerancia)
        3. union de los positivos
        4. resta de los negativos

    Por que este orden y no otro
    ----------------------------
    (a) La tolerancia va ANTES de las features. aplicar_tolerancia()
        reconstruye la guia entera desde un campo continuo, con un error de
        orden spacing^2 en aristas vivas. Sobre la carcasa lisa es
        imperceptible; sobre una camisa de fresado ya montada, no. Este es el
        motivo por el que la version anterior de este modulo estaba mal: metia
        el clearance dentro de la booleana, despues de unir los refuerzos.

    (b) La tolerancia solo resta material, asi que no puede deshacer el
        desbloqueo del paso 1. El orden inverso si romperia: el desbloqueo es
        binario y su holgura minima de un voxel se comeria la separacion fina.

    (c) Los positivos van antes que los negativos. Unir el collar de refuerzo
        despues de abrir la ranura dejaria la pared de la ranura con el grosor
        original de la carcasa.

    desbloqueo : "recortar"  usa uc.recortar_guia() sobre la guia ya formada.
                             Es la via por defecto: solo toca la guia.
                 "ninguno"   asume que el hueso ya venia con blockout hecho
                             mediante uc.eliminar_undercuts() ANTES de extraer
                             el parche. Mas limpio geometricamente, pero
                             modifica todo el modelo oseo.

    positivos, negativos : secuencias de (nombre, vtkPolyData) ya ordenadas.
                           Los positivos se unen y luego se restan los
                           negativos, en el orden recibido.
    """
    import logging

    def avisar(msg):
        logging.info("malla_guia: %s", msg)
        if callback:
            callback(msg)

    r = carcasa

    if desbloqueo == "recortar":
        avisar("desbloqueando el eje de insercion")
        r = uc.recortar_guia(r, hueso, eje=eje, spacing=spacing)
    elif desbloqueo != "ninguno":
        raise ValueError("desbloqueo debe ser 'recortar' o 'ninguno'")

    if tolerancia > 0:
        avisar("aplicando tolerancia de %.2f mm" % tolerancia)
        r = tl.aplicar_tolerancia(r, hueso, tolerancia=tolerancia,
                                  spacing=spacing_tolerancia, suavizado=0)

    for nombre, pd in positivos:
        avisar("uniendo positivo '%s'" % nombre)
        r = booleana(r, pd, "union", spacing=spacing)

    for nombre, pd in negativos:
        avisar("restando negativo '%s'" % nombre)
        r = booleana(r, pd, "diferencia", spacing=spacing)

    avisar("ensamblado completo")
    return r


# ---------------------------------------------------------------------------
# 6. VALIDACION
# ---------------------------------------------------------------------------

def validar_guia(guia: vtk.vtkPolyData, hueso: vtk.vtkPolyData = None,
                 eje=(0, 0, 1), volumen_min: float = 100.0,
                 spacing_verif: float = 0.3) -> tuple:
    """Comprueba imprimibilidad y ajuste. Devuelve (ok, problemas, metricas).

    Las verificaciones geometricas (estanqueidad, piezas sueltas, volumen) son
    locales. Las de insercion y ajuste se delegan en undercut_core, que ya
    simula el barrido real a lo largo del eje.
    """
    problemas = []
    metricas = {}

    fe = vtk.vtkFeatureEdges()
    fe.SetInputData(guia)
    fe.BoundaryEdgesOn()
    fe.NonManifoldEdgesOn()
    fe.FeatureEdgesOff()
    fe.ManifoldEdgesOff()
    fe.Update()
    n_malas = fe.GetOutput().GetNumberOfCells()
    metricas["aristas_defectuosas"] = int(n_malas)
    if n_malas > 0:
        problemas.append(
            "Malla no estanca: %d aristas de borde o no manifold." % n_malas)

    conn = vtk.vtkPolyDataConnectivityFilter()
    conn.SetInputData(guia)
    conn.SetExtractionModeToAllRegions()
    conn.Update()
    n_piezas = conn.GetNumberOfExtractedRegions()
    metricas["piezas"] = int(n_piezas)
    if n_piezas > 1:
        problemas.append(
            "La guia esta partida en %d piezas inconexas. Causa habitual: una "
            "ranura atraviesa por completo un puente estrecho." % n_piezas)

    vol = uc.volumen_malla(guia)
    metricas["volumen_mm3"] = vol
    if vol < volumen_min:
        problemas.append(
            "Volumen de %.1f mm3, bajo el minimo de %.1f." % (vol, volumen_min))

    if hueso is not None:
        ins = uc.verificar_insercion(guia, hueso, eje=eje,
                                     spacing=spacing_verif)
        metricas["insercion"] = ins
        if not ins["ok"]:
            problemas.append(
                "La guia no puede insertarse: colision de %.1f mm3 a %.2f mm "
                "de recorrido." % (ins["colision_mm3"],
                                   ins["desplazamiento_mm"]))

    return (len(problemas) == 0), problemas, metricas


# ---------------------------------------------------------------------------
# 7. EXPORTACION
# ---------------------------------------------------------------------------

def exportar_stl(pd: vtk.vtkPolyData, ruta: str,
                 sistema_entrada: str = "LPS") -> str:
    """Exporta a STL binario. El archivo SIEMPRE queda en LPS (R-002).

    sistema_entrada : sistema de coordenadas en que viene `pd`.
        "LPS" (por defecto) -> se escribe tal cual; es el sistema del nucleo.
        "RAS"               -> se convierte a LPS antes de escribir. Usalo
                               solo con mallas que vienen de Slicer.

    La entrada RAS debe declararse de forma explicita: con el default
    anterior (convertir siempre) una malla del nucleo, ya en LPS, salia
    rotada 180 grados sin aviso (RG-002).

    RAS -> LPS es (x, y, z) -> (-x, -y, z), cuyo determinante vale +1: es una
    rotacion, no una reflexion. El winding de los triangulos NO se invierte y
    no hay que aplicar vtkReverseSense. (La version anterior de este modulo lo
    hacia y dejaba todas las normales al reves.)
    """
    sistema_entrada = sistema_entrada.upper()
    if sistema_entrada not in ("LPS", "RAS"):
        raise ValueError(
            "sistema_entrada debe ser 'LPS' o 'RAS', no '%s'." % sistema_entrada)

    a_escribir = pd
    if sistema_entrada == "RAS":
        tr = vtk.vtkTransform()
        tr.Scale(-1.0, -1.0, 1.0)
        f = vtk.vtkTransformPolyDataFilter()
        f.SetInputData(pd)
        f.SetTransform(tr)
        f.Update()
        a_escribir = f.GetOutput()

    w = vtk.vtkSTLWriter()
    w.SetInputData(a_escribir)
    w.SetFileName(ruta)
    w.SetFileTypeToBinary()
    w.Write()
    return ruta
