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


def puente_y_columna(escaneo, dientes, implante: Implante, kit: PerfilKit, fabricacion: str,
                     solape_dientes_mm: float = 1.5) -> dict:
    """Puente sobre la encía de la brecha y columna del orificio (R-019).

    Puente: la encía del escaneo alrededor del eje, hasta los dientes vecinos
    (más `solape_dientes_mm` para que se una al apoyo), levantada
    `kit.holgura_encia_mm` y con `kit.espesor_plantilla_mm` de espesor.
    Columna: cilindro del diámetro externo del anillo desde el piso del puente
    hasta la cara superior del orificio. Sin alivio (decisión clínica
    2026-10-09), la fresa roza resina desde el piso del puente hasta la cara
    superior: ese es el contacto efectivo.
    """
    import vtk
    from vtk.util import numpy_support

    from nucleo_dental.apoyo import clasificar_diente_encia, parche_de_apoyo
    from nucleo_dental.geometria.malla_guia import parche_a_solido

    g = geometria_orificio(implante, kit, fabricacion)
    puntos = numpy_support.vtk_to_numpy(escaneo.GetPoints().GetData()).astype(float)
    es_diente = clasificar_diente_encia(escaneo, dientes)
    if not es_diente.any():
        raise ValueError("Ningún punto del escaneo coincide con los dientes del CBCT: revisa el registro.")
    relativo = puntos - implante.apice
    radial = np.linalg.norm(relativo - np.outer(relativo @ implante.eje, implante.eje), axis=1)
    alcance = radial[es_diente].min() + solape_dientes_mm
    encia_brecha = parche_de_apoyo(escaneo, (~es_diente) & (radial <= alcance), solo_mayor=True)

    # Normales hacia afuera del tejido (mismo sentido que el eje, hacia la plataforma).
    encia_brecha = _orientar_hacia(encia_brecha, implante.eje)
    piso = _desplazar(encia_brecha, kit.holgura_encia_mm)
    puente = parche_a_solido(piso, kit.espesor_plantilla_mm)

    # Piso del puente sobre el eje: donde el eje corta la encía, más la holgura.
    arbol = vtk.vtkOBBTree()
    arbol.SetDataSet(encia_brecha)
    arbol.BuildLocator()
    cortes = vtk.vtkPoints()
    arbol.IntersectWithLine(implante.apice, np.array(g["cara_superior"]), cortes, None)
    if cortes.GetNumberOfPoints() == 0:
        raise ValueError("El eje del implante no atraviesa la encía de la brecha en el escaneo.")
    encia_en_eje = np.array(cortes.GetPoint(0))
    piso_en_eje = encia_en_eje + kit.holgura_encia_mm * implante.eje
    techo_en_eje = piso_en_eje + kit.espesor_plantilla_mm * implante.eje
    cara_superior = np.array(g["cara_superior"])
    alto_piso = float((cara_superior - piso_en_eje) @ implante.eje)
    if alto_piso <= kit.espesor_plantilla_mm:
        raise ValueError("La cara superior del orificio queda dentro del puente: revisa el offset o la posición del implante.")
    columna = Implante(g["diametro_externo_anillo_mm"], alto_piso, piso_en_eje, implante.eje).como_malla()

    return {
        "geometria": g,
        "puente": puente,
        "columna": columna,
        "alcance_puente_mm": float(alcance),
        "piso_en_eje": piso_en_eje.tolist(),
        "techo_en_eje": techo_en_eje.tolist(),
        "alto_columna_mm": float((cara_superior - techo_en_eje) @ implante.eje),
        "contacto_efectivo_mm": alto_piso,
    }


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


def _desplazar(parche, distancia: float):
    """Copia del parche desplazada `distancia` a lo largo de sus normales."""
    import vtk

    from nucleo_dental.geometria.malla_guia import _normales

    con_normales = _normales(parche)
    con_normales.GetPointData().SetActiveVectors("Normals")
    warp = vtk.vtkWarpVector()
    warp.SetInputData(con_normales)
    warp.SetScaleFactor(float(distancia))
    warp.Update()
    salida = vtk.vtkPolyData()
    salida.DeepCopy(warp.GetOutput())
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
