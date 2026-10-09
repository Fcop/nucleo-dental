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
    encia = _orientar_hacia(parche_de_apoyo(escaneo, mascara, solo_mayor=True), implante.eje)
    cresta = _cruce_del_eje(encia, implante, np.array(g["cara_superior"]))
    mascara &= _hasta_profundidad(puntos, cresta, implante.eje, kit.profundidad_puente_mm)
    encia = _orientar_hacia(parche_de_apoyo(escaneo, mascara, solo_mayor=True), implante.eje)
    piso = _desplazar(encia, holgura) if holgura > 0 else encia
    puente = parche_a_solido(piso, kit.espesor_plantilla_mm)

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
        "geometria": g,
        "tipo_soporte": tipo_soporte,
        "holgura_encia_mm": holgura,
        "puente": puente,
        "columna": columna,
        "alivio": alivio,
        "alcance_puente_mm": float(alcance),
        "cresta_en_eje": cresta.tolist(),
        "piso_en_eje": piso_en_eje.tolist(),
        "techo_en_eje": techo_en_eje.tolist(),
        "alto_columna_mm": float((cara_superior - techo_en_eje) @ implante.eje),
        "contacto_efectivo_mm": float(contacto),
    }


def _cruce_del_eje(superficie, implante: Implante, hasta) -> np.ndarray:
    """Punto donde el eje del implante (del ápice hacia `hasta`) atraviesa la superficie."""
    import vtk

    arbol = vtk.vtkOBBTree()
    arbol.SetDataSet(superficie)
    arbol.BuildLocator()
    cortes = vtk.vtkPoints()
    arbol.IntersectWithLine(implante.apice, np.asarray(hasta, float), cortes, None)
    if cortes.GetNumberOfPoints() == 0:
        raise ValueError("El eje del implante no atraviesa la encía de la brecha en el escaneo.")
    return np.array(cortes.GetPoint(0))


def _hasta_profundidad(puntos, cresta, eje, profundidad_mm: float) -> np.ndarray:
    """True por cada punto que no está más de `profundidad_mm` bajo la cresta, a lo largo del eje."""
    u = np.asarray(eje, float) / np.linalg.norm(eje)
    return (np.asarray(puntos, float) - np.asarray(cresta, float)) @ u >= -profundidad_mm


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
