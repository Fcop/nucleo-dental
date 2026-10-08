"""Registro del escaneo intraoral con los dientes del CBCT (R-015).

Parte de una alineación previa (p. ej. por puntos en Slicer) y la refina con
las superficies de las coronas. Es un ICP rígido "recortado": en cada vuelta
solo participan los puntos del escaneo que ya están cerca de los dientes del
CBCT, así la encía y los tejidos que el CBCT no segmentó quedan fuera. El
umbral se va estrechando y cada etapa se repite hasta que la corrección deja
de cambiar.

La calidad no se mide con puntos marcados a mano (su incertidumbre, ~0,5 mm,
supera la precisión buscada) sino con la desviación residual de las coronas y
la estabilidad del resultado entre variantes del ajuste.
"""

from __future__ import annotations

import numpy as np
import vtk
from vtk.util import numpy_support

# Etapas del recorte (mm): solo cuentan los puntos del escaneo a menos de esta
# distancia de los dientes. Las variantes estiman la estabilidad del resultado.
_ETAPAS = (1.0, 0.6, 0.4)
_VARIANTES = ((1.5, 0.8, 0.5), (1.0, 0.5, 0.3))
_MAX_VUELTAS_POR_ETAPA = 6
_MAX_PUNTOS_ICP = 2000
_CONVERGENCIA_MM = 1e-3
_MIN_PUNTOS = 200
_ZONA_CORONAS_MM = 2.0
_MAX_CANDIDATOS = 20000


def refinar_registro(escaneo: vtk.vtkPolyData, dientes: vtk.vtkPolyData, punto=None) -> dict:
    """Matriz rígida 4×4 (LPS) que lleva el escaneo sobre los dientes del CBCT, con su informe de calidad.

    punto: opcional (p. ej. el ápice planificado); se informa cuánto lo mueve la corrección.
    """
    todos = numpy_support.vtk_to_numpy(escaneo.GetPoints().GetData()).astype(float)
    # Solo las coronas pueden calzar con el escaneo: se descartan las raíces
    # (lejos del escaneo) y el índice espacial se construye una sola vez.
    dientes = _zona_de_coronas(dientes, escaneo)
    distancia = _distancia_a(dientes)
    # Candidatos: puntos del escaneo que pueden llegar a participar (cerca de
    # las coronas), con una muestra regular acotada. La encía lejana y el resto
    # del escaneo no se vuelven a evaluar en cada vuelta.
    vertices = todos[distancia(todos) < _ZONA_CORONAS_MM]
    if len(vertices) > _MAX_CANDIDATOS:
        vertices = vertices[np.linspace(0, len(vertices) - 1, _MAX_CANDIDATOS).astype(int)]
    localizador = vtk.vtkCellLocator()
    localizador.SetDataSet(dientes)
    localizador.SetNumberOfCellsPerBucket(1)
    localizador.BuildLocator()
    dientes = (dientes, localizador)

    matriz = _ajustar(vertices, distancia, dientes, _ETAPAS)
    usados = distancia(aplicar_matriz(matriz, vertices)) < _ETAPAS[-1]
    antes = distancia(vertices[usados])
    despues = distancia(aplicar_matriz(matriz, vertices[usados]))

    referencia = np.asarray(punto, dtype=float) if punto is not None else vertices[usados].mean(axis=0)
    destinos = [aplicar_matriz(matriz, [referencia])[0]]
    for etapas in _VARIANTES:
        destinos.append(aplicar_matriz(_ajustar(vertices, distancia, dientes, etapas), [referencia])[0])
    destinos = np.array(destinos)
    estabilidad = max(np.linalg.norm(a - b) for a in destinos for b in destinos)

    resultado = {
        "matriz_lps": matriz.tolist(),
        "rotacion_grados": _angulo(matriz),
        "traslacion_mm": matriz[:3, 3].tolist(),
        "desviacion_antes_mm": _resumen(antes),
        "desviacion_despues_mm": _resumen(despues),
        "puntos_usados": int(usados.sum()),
        "fraccion_usada": float(usados.sum() / len(todos)),
        "estabilidad_mm": float(estabilidad),
    }
    if punto is not None:
        resultado["correccion_en_punto_mm"] = float(np.linalg.norm(destinos[0] - referencia))
    return resultado


def aplicar_matriz(matriz, puntos) -> np.ndarray:
    """Aplica una matriz 4×4 a puntos N×3."""
    matriz = np.asarray(matriz, dtype=float)
    puntos = np.atleast_2d(np.asarray(puntos, dtype=float))
    return puntos @ matriz[:3, :3].T + matriz[:3, 3]


def transformar_malla(malla: vtk.vtkPolyData, matriz) -> vtk.vtkPolyData:
    """Copia de la malla con la matriz 4×4 aplicada."""
    vtk_matriz = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            vtk_matriz.SetElement(i, j, float(np.asarray(matriz)[i, j]))
    tr = vtk.vtkTransform()
    tr.SetMatrix(vtk_matriz)
    filtro = vtk.vtkTransformPolyDataFilter()
    filtro.SetInputData(malla)
    filtro.SetTransform(tr)
    filtro.Update()
    salida = vtk.vtkPolyData()
    salida.DeepCopy(filtro.GetOutput())
    return salida


def _ajustar(vertices: np.ndarray, distancia, dientes: vtk.vtkPolyData, etapas) -> np.ndarray:
    matriz = np.eye(4)
    for umbral in etapas:
        for _ in range(_MAX_VUELTAS_POR_ETAPA):
            actuales = aplicar_matriz(matriz, vertices)
            cerca = actuales[distancia(actuales) < umbral]
            if len(cerca) < _MIN_PUNTOS:
                raise ValueError(
                    f"No hay suficiente superposición entre el escaneo y los dientes del CBCT "
                    f"({len(cerca)} puntos a menos de {umbral} mm). Alinéalos primero, por ejemplo "
                    "por puntos en Slicer, y vuelve a intentar.")
            paso = _icp(cerca, dientes)
            matriz = paso @ matriz
            movimiento = np.linalg.norm(aplicar_matriz(paso, cerca) - cerca, axis=1).max()
            if movimiento < _CONVERGENCIA_MM:
                break
    return matriz


def _zona_de_coronas(dientes: vtk.vtkPolyData, escaneo: vtk.vtkPolyData) -> vtk.vtkPolyData:
    """Triángulos de los dientes con sus tres vértices a menos de 2 mm del escaneo."""
    puntos = numpy_support.vtk_to_numpy(dientes.GetPoints().GetData()).astype(float)
    cerca = _distancia_a(escaneo)(puntos) < _ZONA_CORONAS_MM
    triangulos = numpy_support.vtk_to_numpy(dientes.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    triangulos = triangulos[cerca[triangulos].all(axis=1)]
    if len(triangulos) == 0:
        raise ValueError("No hay suficiente superposición entre el escaneo y los dientes del CBCT: "
                         "ninguna corona queda a menos de 2 mm del escaneo. Alinéalos primero.")
    offsets = np.arange(0, 3 * len(triangulos) + 1, 3, dtype=np.int64)
    celdas = vtk.vtkCellArray()
    celdas.SetData(numpy_support.numpy_to_vtkIdTypeArray(offsets, deep=True),
                   numpy_support.numpy_to_vtkIdTypeArray(triangulos.astype(np.int64).ravel(), deep=True))
    zona = vtk.vtkPolyData()
    zona.SetPoints(dientes.GetPoints())
    zona.SetPolys(celdas)
    return zona


def _icp(puntos: np.ndarray, dientes) -> np.ndarray:
    dientes, localizador = dientes
    if len(puntos) > _MAX_PUNTOS_ICP:   # submuestreo regular y reproducible
        puntos = puntos[np.linspace(0, len(puntos) - 1, _MAX_PUNTOS_ICP).astype(int)]
    vtk_puntos = vtk.vtkPoints()
    vtk_puntos.SetData(numpy_support.numpy_to_vtk(np.ascontiguousarray(puntos), deep=True))
    nube = vtk.vtkPolyData()
    nube.SetPoints(vtk_puntos)
    vertices = vtk.vtkVertexGlyphFilter()
    vertices.SetInputData(nube)
    vertices.Update()

    icp = vtk.vtkIterativeClosestPointTransform()
    icp.SetSource(vertices.GetOutput())
    icp.SetTarget(dientes)
    icp.SetLocator(localizador)
    icp.GetLandmarkTransform().SetModeToRigidBody()
    icp.StartByMatchingCentroidsOff()
    icp.SetMaximumNumberOfLandmarks(len(puntos))
    icp.SetMaximumNumberOfIterations(50)
    icp.CheckMeanDistanceOn()
    icp.SetMaximumMeanDistance(1e-4)   # se detiene cuando los puntos se mueven < 0,0001 mm por iteración
    icp.Update()
    m = icp.GetMatrix()
    return np.array([[m.GetElement(i, j) for j in range(4)] for i in range(4)])


def _distancia_a(malla: vtk.vtkPolyData):
    implicita = vtk.vtkImplicitPolyDataDistance()
    implicita.SetInput(malla)

    def evaluar(puntos: np.ndarray) -> np.ndarray:
        salida = vtk.vtkDoubleArray()
        implicita.FunctionValue(numpy_support.numpy_to_vtk(np.ascontiguousarray(puntos, dtype=float), deep=True),
                                salida)
        return np.abs(numpy_support.vtk_to_numpy(salida))

    return evaluar


def _angulo(matriz: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(matriz[:3, :3]) - 1) / 2, -1.0, 1.0))))


def _resumen(distancias: np.ndarray) -> dict:
    return {"mediana": float(np.median(distancias)), "p90": float(np.percentile(distancias, 90))}
