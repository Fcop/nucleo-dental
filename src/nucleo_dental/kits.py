"""Perfiles de kit de cirugía guiada (R-018).

Un perfil reúne los parámetros del kit y del diseño de la plantilla. Todos
son configurables: `perfil.con(campo=valor)` devuelve una copia modificada.
Los valores que no publica el fabricante se marcan como provisionales y se
informan como tales.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class PerfilKit:
    nombre: str
    # (diámetro máximo del implante, diámetro de la guía de la fresa), en orden creciente.
    orificios: tuple
    contacto_mm: float              # alto del anillo: contacto guía–fresa
    offset_mm: float                # cara superior del orificio → plataforma del implante
    espesor_plantilla_mm: float
    pared_anillo_mm: float          # pared de resina alrededor del orificio
    holgura_encia_mm: float         # separación del puente sobre la encía de la brecha
    sobrefresado_mm: float
    ajuste_fabricacion_mm: dict = field(default_factory=dict)   # holgura del orificio según método
    con_camisa: bool = False
    pared_camisa_mm: float = 1.0
    alivio_mm: float = 0.0                 # ensanche del orificio bajo el anillo (0 = sin alivio)
    profundidad_puente_mm: float = 6.0     # hasta dónde baja el puente bajo la cresta, a lo largo del eje
    tolerancia_ajuste_mm: dict = field(default_factory=dict)    # juego guía–diente según método
    ajuste_pin_mm: dict = field(default_factory=dict)           # holgura del agujero del pin según método
    pared_refuerzo_pin_mm: float = 2.0     # pared de resina alrededor del agujero del pin
    alto_refuerzo_pin_mm: float = 3.0      # alto del refuerzo, desde donde el pin entra en el escaneo
    provisionales: frozenset = frozenset()
    fuente: str = ""

    def diametro_guia_para(self, diametro_implante: float) -> float:
        for maximo, guia in self.orificios:
            if diametro_implante <= maximo + 1e-9:
                return guia
        raise ValueError(f"El kit {self.nombre} no tiene orificio para un implante de {diametro_implante} mm.")

    def ajuste(self, fabricacion: str) -> float:
        if fabricacion not in self.ajuste_fabricacion_mm:
            raise ValueError(f"Método de fabricación desconocido '{fabricacion}'; opciones: "
                             + ", ".join(self.ajuste_fabricacion_mm))
        return self.ajuste_fabricacion_mm[fabricacion]

    def tolerancia_ajuste(self, fabricacion: str) -> float:
        if fabricacion not in self.tolerancia_ajuste_mm:
            raise ValueError(f"Método de fabricación desconocido '{fabricacion}'; opciones: "
                             + ", ".join(self.tolerancia_ajuste_mm))
        return self.tolerancia_ajuste_mm[fabricacion]

    def ajuste_pin(self, fabricacion: str) -> float:
        if fabricacion not in self.ajuste_pin_mm:
            raise ValueError(f"Método de fabricación desconocido '{fabricacion}'; opciones: "
                             + ", ".join(self.ajuste_pin_mm))
        return self.ajuste_pin_mm[fabricacion]

    def con(self, **cambios) -> "PerfilKit":
        return replace(self, **cambios)


ONEGUIDE = PerfilKit(
    nombre="oneguide",
    orificios=((4.5, 5.0), (5.0, 5.7)),
    contacto_mm=3.0,
    offset_mm=10.5,
    espesor_plantilla_mm=3.0,
    pared_anillo_mm=3.0,
    holgura_encia_mm=1.0,
    sobrefresado_mm=0.3,
    ajuste_fabricacion_mm={"impresa": 0.3, "fresada": 0.1},
    con_camisa=False,
    alivio_mm=0.0,
    profundidad_puente_mm=6.0,
    tolerancia_ajuste_mm={"impresa": 0.2, "fresada": 0.1},
    ajuste_pin_mm={"impresa": 0.3, "fresada": 0.1},
    pared_refuerzo_pin_mm=2.0,
    alto_refuerzo_pin_mm=3.0,
    provisionales=frozenset({"sobrefresado_mm", "tolerancia_ajuste_mm", "alto_refuerzo_pin_mm"}),
    fuente=("Catálogo y manual Hiossen OneGuide (orificios Ø5,0/Ø5,7, contacto 3 mm, sin camisa); "
            "offset, espesor, pared, holgura, ajustes y sobrefresado: decisiones clínicas 2026-10-08; "
            "tolerancia de ajuste dentro del rango aceptado el 2026-10-09 (impresa 0,20–0,30; fresada 0,10–0,15); "
            "pines: ajuste como el del orificio y pared de refuerzo 2 mm (decisiones 2026-10-10), alto 3 mm provisional"),
)

_KITS = {ONEGUIDE.nombre: ONEGUIDE}


def obtener_kit(nombre: str) -> PerfilKit:
    try:
        return _KITS[nombre.lower()]
    except KeyError:
        raise ValueError(f"Kit desconocido '{nombre}'; disponibles: " + ", ".join(_KITS)) from None
