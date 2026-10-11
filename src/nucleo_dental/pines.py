"""Pines de fijación de la guía al hueso (R-025).

El usuario ubica cada pin en Slicer como un modelo STL (un cilindro), igual
que el implante (decisión clínica 2026-10-10). A diferencia del implante, el
pin suele ir inclinado (vestibular), así que la punta no se decide por
"abajo/arriba" sino por el hueso: es el extremo que queda dentro de él.

    punta   extremo dentro del hueso
    cabeza  extremo fuera, del lado de la guía
    eje     de la punta a la cabeza

Se informa cuánto entra en el hueso (sin mínimo) y se evalúa como un implante
contra canal, dientes e implantes, con margen de 2 mm configurable. En la guía
lleva un refuerzo de resina (pared y alto configurables en el kit) y un
agujero del diámetro del pin más el ajuste por fabricación.
"""

from __future__ import annotations

import numpy as np

from nucleo_dental.implante import Implante, cilindro_desde_malla
from nucleo_dental.kits import PerfilKit

MARGEN_PIN_POR_DEFECTO_MM = 2.0          # decisión clínica 2026-10-10, configurable
_PROLONGACION_AGUJERO_MM = 20.0          # el agujero sale de la guía por la cabeza


class Pin(Implante):
    """Cilindro del pin: `apice` es la punta (en el hueso) y `eje` va hacia la cabeza."""

    @property
    def punta(self) -> np.ndarray:
        return self.apice

    @property
    def cabeza(self) -> np.ndarray:
        return self.plataforma

    @classmethod
    def desde_malla(cls, malla, hueso) -> "Pin":
        """Pin desde la malla de un cilindro; la punta es el extremo que queda dentro del hueso."""
        from nucleo_dental.medicion import _dentro

        centro, eje, t_min, t_max, radio = cilindro_desde_malla(malla, "pin")
        extremos = np.array([centro + t_min * eje, centro + t_max * eje])
        dentro = _dentro(hueso, extremos)
        if dentro.all():
            raise ValueError("El pin queda entero dentro del hueso: no se sabe cuál extremo es la cabeza.")
        if not dentro.any():
            raise ValueError("Ningún extremo del pin está dentro del hueso: revisa su posición.")
        punta, cabeza = (extremos[0], extremos[1]) if dentro[0] else (extremos[1], extremos[0])
        return cls(diametro=2 * radio, largo=t_max - t_min, apice=punta, eje=cabeza - punta)


def profundidad_en_hueso(pin: Pin, hueso) -> float:
    """Largo del pin dentro del hueso, desde la punta hasta donde el eje sale del hueso (mm)."""
    cruce = _primer_cruce(hueso, pin.punta, pin.cabeza)
    if cruce is None:
        return float(pin.largo)
    return float(np.linalg.norm(cruce - pin.punta))


def evaluar_pin(pin: Pin, estructuras: dict, implantes: dict | None = None,
                margen: float = MARGEN_PIN_POR_DEFECTO_MM) -> dict:
    """Semáforo del pin contra cada estructura ({nombre: malla cerrada}) y cada implante, con un solo margen.

    Las estructuras se miden completas (con las raíces: el pin no tiene
    plataforma). Rojo si alguna distancia no alcanza el margen o hay colisión.
    """
    from nucleo_dental.medicion import distancia_entre_implantes, medir

    margen = float(margen)
    if not np.isfinite(margen) or margen < 0:
        raise ValueError(f"El margen del pin debe ser un número mayor o igual que 0 mm (se recibió {margen}).")
    resultados = {}
    for nombre, malla in estructuras.items():
        resultados[nombre] = {**medir(pin, malla, margen), "margen_mm": margen}
    for nombre, implante in (implantes or {}).items():
        d = distancia_entre_implantes(pin, implante)
        d.update({"margen_mm": margen, "semaforo": "rojo" if d["colision"] or d["distancia_mm"] < margen else "verde"})
        resultados[f"implante {nombre}"] = d
    rojo = any(r["semaforo"] == "rojo" for r in resultados.values())
    return {"semaforo": "rojo" if rojo else "verde", "estructuras": resultados}


def refuerzo_y_agujero(pin: Pin, escaneo, kit: PerfilKit, fabricacion: str) -> dict:
    """Refuerzo del pin (positivo) y su agujero (negativo) como mallas cerradas coaxiales con el pin.

    El refuerzo nace donde el eje, bajando desde la cabeza, entra en el escaneo,
    y sube `kit.alto_refuerzo_pin_mm` hacia la cabeza. El agujero atraviesa toda
    la guía: del lado de la punta hasta más allá de la cabeza.
    """
    entrada = _primer_cruce(escaneo, pin.cabeza, pin.punta)
    if entrada is None:
        raise ValueError("El eje del pin no atraviesa el escaneo: revisa su posición.")
    diametro_agujero = pin.diametro + kit.ajuste_pin(fabricacion)
    diametro_refuerzo = diametro_agujero + 2 * kit.pared_refuerzo_pin_mm
    refuerzo = Implante(diametro_refuerzo, kit.alto_refuerzo_pin_mm, entrada, pin.eje)
    largo_agujero = float(np.linalg.norm(pin.cabeza - pin.punta)) + 1.0 + _PROLONGACION_AGUJERO_MM
    agujero = Implante(diametro_agujero, largo_agujero, pin.punta - 1.0 * pin.eje, pin.eje)
    return {
        "entrada": entrada.tolist(),
        "diametro_agujero_mm": float(diametro_agujero),
        "diametro_refuerzo_mm": float(diametro_refuerzo),
        "alto_refuerzo_mm": float(kit.alto_refuerzo_pin_mm),
        "refuerzo": refuerzo.como_malla(),
        "agujero": agujero.como_malla(),
        "tramo_agujero": Implante(diametro_agujero, kit.alto_refuerzo_pin_mm, entrada, pin.eje),
    }


def _primer_cruce(malla, desde, hacia):
    """Primer punto donde el segmento `desde`→`hacia` atraviesa la superficie, o None."""
    import vtk

    arbol = vtk.vtkOBBTree()
    arbol.SetDataSet(malla)
    arbol.BuildLocator()
    cortes = vtk.vtkPoints()
    arbol.IntersectWithLine(np.asarray(desde, float), np.asarray(hacia, float), cortes, None)
    return np.array(cortes.GetPoint(0)) if cortes.GetNumberOfPoints() else None
