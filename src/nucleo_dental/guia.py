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
