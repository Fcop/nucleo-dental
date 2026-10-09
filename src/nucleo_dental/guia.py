"""Geometría de la guía quirúrgica: anillo guía y orificio según el kit (R-018).

Todas las alturas se miden a lo largo del eje del implante:

    cara superior del orificio = plataforma + offset
    cara inferior del anillo   = cara superior − contacto
    punta de la fresa          = ápice − sobrefresado
    largo de trabajo de fresa  = cara superior → punta de la fresa
"""

from __future__ import annotations

import numpy as np

from nucleo_dental.implante import Implante
from nucleo_dental.kits import PerfilKit

# El orificio se prolonga más allá del anillo para atravesar toda la guía
# (anillo, puente y apoyo) al restarlo.
_PROLONGACION_ORIFICIO_MM = 20.0
# Pasadas de promedio de normales del puente: quita el ruido del escaneo sin
# perder la forma de la encía (caso real: 0 puntos en el hueso con 0 a 40).
_SUAVIZADO_NORMALES = 10
# Pasadas de alisado del contorno del puente (solo vértices del borde).
_PASADAS_BORDE = 10


def geometria_orificio(implante: Implante, kit: PerfilKit, fabricacion: str) -> dict:
    """Alturas y diámetros del anillo guía y del orificio para este implante y kit."""
    ajuste = kit.ajuste(fabricacion)
    guia = kit.diametro_guia_para(implante.diametro)
    plataforma = implante.plataforma
    cara_superior = plataforma + kit.offset_mm * implante.eje
    cara_inferior = cara_superior - kit.contacto_mm * implante.eje
    punta_fresa = implante.apice - kit.sobrefresado_mm * implante.eje

    resultado = {
        "kit": kit.nombre,
        "fabricacion": fabricacion,
        "diametro_guia_mm": guia,
        "plataforma": plataforma.tolist(),
        "cara_superior": cara_superior.tolist(),
        "cara_inferior_anillo": cara_inferior.tolist(),
        "punta_fresa": punta_fresa.tolist(),
        "largo_trabajo_fresa_mm": float(np.linalg.norm(cara_superior - punta_fresa)),
        "provisionales": sorted(kit.provisionales),
    }
    if kit.con_camisa:
        externo_camisa = guia + 2 * kit.pared_camisa_mm
        resultado.update({"diametro_interno_camisa_mm": guia,
                          "diametro_externo_camisa_mm": externo_camisa,
                          "diametro_orificio_mm": externo_camisa + ajuste})
    else:
        resultado["diametro_orificio_mm"] = guia + ajuste
    resultado["diametro_externo_anillo_mm"] = resultado["diametro_orificio_mm"] + 2 * kit.pared_anillo_mm
    return resultado


TIPOS_SOPORTE = ("dentosoportada", "dentomucosoportada", "mucosoportada")


def puente_y_columna(escaneo, dientes, implante: Implante, kit: PerfilKit, fabricacion: str,
                     solape_dientes_mm: float = 1.5, tipo_soporte: str = "dentosoportada",
                     radio_mucosa_mm: float = 24.0) -> dict:
    """Puente sobre la encía de la brecha y columna del orificio (R-019, R-020).

    Puente: la encía del escaneo alrededor del eje, hasta los dientes vecinos
    (más `solape_dientes_mm` para unirse al apoyo) o, en una guía
    mucosoportada, hasta `radio_mucosa_mm`; limitada a
    `kit.profundidad_puente_mm` bajo la cresta, a lo largo del eje. Se levanta
    `kit.holgura_encia_mm` solo si la guía es dentosoportada: si apoya en la
    mucosa no hay holgura (decisión clínica 2026-10-09). Espesor:
    `kit.espesor_plantilla_mm`.
    Columna: cilindro del diámetro externo del anillo desde el piso del puente
    hasta la cara superior del orificio. Contacto efectivo guía–fresa: el del
    kit si hay alivio (`kit.alivio_mm` > 0); si no, del piso del puente a la
    cara superior.
    """
    import vtk
    from vtk.util import numpy_support

    from nucleo_dental.apoyo import clasificar_diente_encia, parche_de_apoyo
    from nucleo_dental.geometria.malla_guia import parche_a_solido

    if tipo_soporte not in TIPOS_SOPORTE:
        raise ValueError(f"Tipo de soporte desconocido '{tipo_soporte}'; opciones: " + ", ".join(TIPOS_SOPORTE))
    g = geometria_orificio(implante, kit, fabricacion)
    puntos = numpy_support.vtk_to_numpy(escaneo.GetPoints().GetData()).astype(float)
    relativo = puntos - implante.apice
    radial = np.linalg.norm(relativo - np.outer(relativo @ implante.eje, implante.eje), axis=1)

    if tipo_soporte == "mucosoportada":
        es_diente = (clasificar_diente_encia(escaneo, dientes) if dientes is not None
                     else np.zeros(len(puntos), dtype=bool))
        alcance = float(radio_mucosa_mm)
    else:
        if dientes is None:
            raise ValueError(f"Una guía {tipo_soporte} necesita los dientes del CBCT para delimitarse.")
        es_diente = clasificar_diente_encia(escaneo, dientes)
        if not es_diente.any():
            raise ValueError("Ningún punto del escaneo coincide con los dientes del CBCT: revisa el registro.")
        alcance = radial[es_diente].min() + solape_dientes_mm
    holgura = kit.holgura_encia_mm if tipo_soporte == "dentosoportada" else 0.0

    mascara = (~es_diente) & (radial <= alcance)
    # La cresta es el primer cruce bajando por el eje desde la cara superior, y el
    # puente es la pieza de encía que la contiene: un escaneo cerrado también
    # tiene la cara inferior del zócalo, que no es encía de la brecha.
    cresta = _cruce_del_eje(parche_de_apoyo(escaneo, mascara), implante, np.array(g["cara_superior"]))
    mascara &= _margen_de_profundidad(puntos, cresta, implante.eje, kit.profundidad_puente_mm) >= 0
    encia = _orientar_hacia(_pieza_mas_cercana(parche_de_apoyo(escaneo, mascara), cresta), implante.eje)
    encia = _alisar_borde(encia, _PASADAS_BORDE)
    # Holgura y espesor con las mismas normales, calculadas sobre la encía (RG-017).
    puente = parche_a_solido(encia, kit.espesor_plantilla_mm, desfase=holgura,
                             suavizado_normales=_SUAVIZADO_NORMALES)

    return {"geometria": g, "tipo_soporte": tipo_soporte, "holgura_encia_mm": holgura, "puente": puente,
            "alcance_puente_mm": float(alcance), **_columna_y_alivio(g, cresta, holgura, implante, kit)}


def columna_del_anillo(escaneo, implante: Implante, kit: PerfilKit, fabricacion: str, holgura_mm: float) -> dict:
    """Columna del anillo (y alivio) sin puente, para una carcasa que ya cubre la brecha (R-021).

    La cresta es el primer cruce del escaneo bajando por el eje desde la cara
    superior; la columna nace `holgura_mm` sobre ella, donde queda la cara
    interna de la carcasa.
    """
    g = geometria_orificio(implante, kit, fabricacion)
    cresta = _cruce_del_eje(escaneo, implante, np.array(g["cara_superior"]))
    return {"geometria": g, "holgura_encia_mm": float(holgura_mm),
            **_columna_y_alivio(g, cresta, float(holgura_mm), implante, kit)}


def _columna_y_alivio(g: dict, cresta, holgura: float, implante: Implante, kit: PerfilKit) -> dict:
    """Columna del diámetro externo del anillo desde el piso hasta la cara superior, y el alivio si lo hay."""
    piso_en_eje = cresta + holgura * implante.eje
    techo_en_eje = piso_en_eje + kit.espesor_plantilla_mm * implante.eje
    cara_superior = np.array(g["cara_superior"])
    alto_piso = float((cara_superior - piso_en_eje) @ implante.eje)
    if alto_piso <= kit.espesor_plantilla_mm:
        raise ValueError("La cara superior del orificio queda dentro del puente: revisa el offset o la posición del implante.")
    columna = Implante(g["diametro_externo_anillo_mm"], alto_piso, piso_en_eje, implante.eje).como_malla()

    alivio = None
    contacto = alto_piso
    if kit.alivio_mm > 0:
        # Ensancha el orificio desde bajo el piso del puente hasta la cara inferior del anillo.
        inicio = piso_en_eje - 1.0 * implante.eje
        largo = float((np.array(g["cara_inferior_anillo"]) - inicio) @ implante.eje)
        alivio = Implante(g["diametro_orificio_mm"] + kit.alivio_mm, largo, inicio, implante.eje).como_malla()
        contacto = kit.contacto_mm

    return {
        "columna": columna,
        "alivio": alivio,
        "cresta_en_eje": cresta.tolist(),
        "piso_en_eje": piso_en_eje.tolist(),
        "techo_en_eje": techo_en_eje.tolist(),
        "alto_columna_mm": float((cara_superior - techo_en_eje) @ implante.eje),
        "contacto_efectivo_mm": float(contacto),
    }


def _cruce_del_eje(superficie, implante: Implante, desde) -> np.ndarray:
    """Primer punto donde el eje del implante, bajando de `desde` hacia el ápice, atraviesa la superficie."""
    import vtk

    arbol = vtk.vtkOBBTree()
    arbol.SetDataSet(superficie)
    arbol.BuildLocator()
    cortes = vtk.vtkPoints()
    arbol.IntersectWithLine(np.asarray(desde, float), implante.apice, cortes, None)
    if cortes.GetNumberOfPoints() == 0:
        raise ValueError("El eje del implante no atraviesa la encía de la brecha en el escaneo.")
    return np.array(cortes.GetPoint(0))


def _pieza_mas_cercana(superficie, punto):
    """La pieza conexa de la superficie más cercana a `punto`."""
    import vtk

    pieza = vtk.vtkPolyDataConnectivityFilter()
    pieza.SetInputData(superficie)
    pieza.SetExtractionModeToClosestPointRegion()
    pieza.SetClosestPoint(*np.asarray(punto, float))
    limpia = vtk.vtkCleanPolyData()
    limpia.SetInputConnection(pieza.GetOutputPort())
    limpia.Update()
    salida = vtk.vtkPolyData()
    salida.DeepCopy(limpia.GetOutput())
    return salida


def _alisar_borde(parche, pasadas: int):
    """Parche con su contorno alisado: quita la escalera de triángulos enteros del borde.

    Solo se mueven los vértices del borde, cada uno hacia el promedio de sus dos
    vecinos de borde (Taubin: avanza y retrocede para no encoger el contorno).
    El interior, que es la superficie de apoyo, queda intacto; por eso el borde
    nunca trepa por el diente vecino.
    """
    import vtk
    from vtk.util import numpy_support

    triangulos = numpy_support.vtk_to_numpy(parche.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    aristas = np.sort(np.vstack([triangulos[:, [0, 1]], triangulos[:, [1, 2]], triangulos[:, [2, 0]]]), axis=1)
    unicas, veces = np.unique(aristas, axis=0, return_counts=True)
    borde = unicas[veces == 1]
    if len(borde) == 0:
        return parche
    puntos = numpy_support.vtk_to_numpy(parche.GetPoints().GetData()).astype(float)
    vecinos = np.bincount(borde.ravel(), minlength=len(puntos)).astype(float)
    en_borde = vecinos > 0
    for _ in range(pasadas):
        for factor in (0.5, -0.53):
            suma = np.zeros_like(puntos)
            np.add.at(suma, borde[:, 0], puntos[borde[:, 1]])
            np.add.at(suma, borde[:, 1], puntos[borde[:, 0]])
            paso = np.zeros_like(puntos)
            paso[en_borde] = suma[en_borde] / vecinos[en_borde, None] - puntos[en_borde]
            puntos = puntos + factor * paso
    salida = vtk.vtkPolyData()
    salida.DeepCopy(parche)
    salida.GetPoints().SetData(numpy_support.numpy_to_vtk(puntos, deep=True))
    return salida


def _margen_de_profundidad(puntos, cresta, eje, profundidad_mm: float) -> np.ndarray:
    """Cuánto le falta a cada punto (mm) para quedar `profundidad_mm` bajo la cresta, a lo largo del eje.

    Positivo: el punto está dentro del límite; negativo: más profundo.
    """
    u = np.asarray(eje, float) / np.linalg.norm(eje)
    return (np.asarray(puntos, float) - np.asarray(cresta, float)) @ u + profundidad_mm


def _orientar_hacia(parche, direccion):
    """Parche con sus triángulos orientados para que las normales apunten, en promedio, hacia `direccion`."""
    import vtk
    from vtk.util import numpy_support

    from nucleo_dental.geometria.malla_guia import _normales

    normales = numpy_support.vtk_to_numpy(_normales(parche).GetPointData().GetNormals())
    if (normales @ np.asarray(direccion, float)).mean() >= 0:
        return parche
    invertir = vtk.vtkReverseSense()
    invertir.SetInputData(parche)
    invertir.ReverseCellsOn()
    invertir.ReverseNormalsOn()
    invertir.Update()
    salida = vtk.vtkPolyData()
    salida.DeepCopy(invertir.GetOutput())
    return salida


def anillo_y_orificio(implante: Implante, kit: PerfilKit, fabricacion: str):
    """(anillo, orificio) como mallas cerradas coaxiales con el implante.

    El anillo es un cilindro macizo (se suma a la guía) y el orificio un
    cilindro más largo que atraviesa toda la guía (se resta al final).
    """
    g = geometria_orificio(implante, kit, fabricacion)
    inferior = np.array(g["cara_inferior_anillo"])
    anillo = Implante(g["diametro_externo_anillo_mm"], kit.contacto_mm, inferior, implante.eje).como_malla()
    inicio = inferior - _PROLONGACION_ORIFICIO_MM * implante.eje
    largo = kit.contacto_mm + 2 * _PROLONGACION_ORIFICIO_MM
    orificio = Implante(g["diametro_orificio_mm"], largo, inicio, implante.eje).como_malla()
    return anillo, orificio


def caja_como_malla(centro, ejes, tamano):
    """Sólido cerrado de una caja: centro, ejes en las columnas de una matriz 3×3 y largo de cada arista."""
    import vtk
    from vtk.util import numpy_support

    cubo = vtk.vtkCubeSource()
    cubo.SetBounds(-0.5, 0.5, -0.5, 0.5, -0.5, 0.5)
    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(cubo.GetOutputPort())
    limpio = vtk.vtkCleanPolyData()                   # vtkCubeSource repite vértices por cara: se unen para cerrar
    limpio.SetInputConnection(tri.GetOutputPort())
    limpio.Update()
    malla = vtk.vtkPolyData()
    malla.DeepCopy(limpio.GetOutput())
    unidad = numpy_support.vtk_to_numpy(malla.GetPoints().GetData()).astype(float)
    puntos = (unidad * np.asarray(tamano, dtype=float)) @ np.asarray(ejes, dtype=float).T + np.asarray(centro, dtype=float)
    malla.GetPoints().SetData(numpy_support.numpy_to_vtk(np.ascontiguousarray(puntos), deep=True))
    return malla
