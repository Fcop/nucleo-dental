# nucleo-dental

Núcleo sin interfaz para planificar implantes dentales guiados y evaluar su seguridad respecto de estructuras anatómicas. Este glosario fija el significado de cada término en el código, los tests y la documentación.

## Geometría del implante

**Implante**:
Cilindro sólido definido por diámetro, largo, ápice y eje.
_Evitar_: fixture, tornillo

**Ápice**:
Centro de la punta del implante, el extremo que entra primero en el hueso.
_Evitar_: punta (como término técnico), extremo apical

**Plataforma**:
Centro del extremo coronal del implante, a un largo del ápice en la dirección del eje.
_Evitar_: cabeza, cuello

**Eje**:
Dirección unitaria que va del ápice a la plataforma.
_Evitar_: orientación, dirección de inserción

## Anatomía

**Canal mandibular**:
Superficie cerrada que delimita el conducto del nervio alveolar inferior.
_Evitar_: nervio (cuando se habla de la geometría), conducto dentario

**Dientes**:
Superficie cerrada de los dientes segmentados de la arcada (corona y raíz); el diente vecino al implante es el que importa para la seguridad.
_Evitar_: raíces (como estructura), piezas

**Hueso**:
Superficie cerrada del hueso segmentado (mandíbula) que debe rodear al implante.
_Evitar_: tejido óseo (como estructura), reborde

**Espesor óseo**:
Hueso que queda entre la pared lateral del implante y la superficie externa del hueso, medido perpendicular al eje; el mínimo alrededor del implante se compara con el margen de hueso. Bajo el ápice no se mide.
_Evitar_: tabla ósea (como término de software), ancho de hueso, distancia al hueso

**Exposición**:
Cuánto sobresale fuera del hueso la pared lateral del implante (dehiscencia o fenestración); el espesor óseo ahí es 0.
_Evitar_: espesor negativo, perforación (cuando se habla de la medida)

**Cavidad interna**:
Espacio sin hueso segmentado y rodeado de hueso por todos lados (p. ej. un espacio medular); no reduce el espesor óseo, pero se informa si el implante la toca.
_Evitar_: hueco, defecto (sin precisar)

**Estructura**:
Cada anatomía contra la que se evalúa el implante (canal mandibular, dientes), con su propio margen.
_Evitar_: objeto, órgano, malla (cuando se habla del concepto clínico)

## Evaluación de seguridad

**Distancia**:
Separación mínima medida entre la superficie del implante y la de una estructura; nunca es negativa y vale 0 cuando hay colisión.
_Evitar_: margen, holgura, distancia de seguridad

**Margen**:
Distancia mínima que el clínico exige entre el implante y una estructura para aceptar un plan; es una regla, no una medición. Cada estructura tiene el suyo (canal 2,0 mm, dientes 1,5 mm).
_Evitar_: distancia de seguridad, distancia, zona de seguridad

**Colisión**:
Situación en que el implante y una estructura comparten volumen o se tocan.
_Evitar_: choque, contacto, intersección

**Penetración**:
Profundidad máxima a la que el implante se mete en una estructura, medida desde su pared; vale 0 si no hay colisión.
_Evitar_: distancia negativa, invasión

**Semáforo**:
Veredicto por estructura: verde si la distancia alcanza su margen, rojo si no lo alcanza o si hay colisión. El semáforo global del plan es rojo si alguna estructura es roja.
_Evitar_: alerta, estado, resultado

**Sobrefresado**:
Tramo que la fresa avanza más allá del ápice del implante según el kit guiado.
_Evitar_: sobreperforación, exceso de fresa

## Registro

**Escaneo intraoral**:
Superficie de dientes y tejido blando tomada con escáner intraoral; es donde se apoya la guía.
_Evitar_: modelo (a secas), STL del paciente

**Registro**:
Movimiento rígido (rotación + traslación) que superpone el escaneo intraoral sobre los dientes del CBCT; su error pasa entero a la guía.
_Evitar_: calce, match, fusión

## Validación

**Caso dorado**:
Escenario sintético cuyo resultado esperado fue calculado a mano por una persona, independiente del código que lo verifica.
_Evitar_: fixture, caso de prueba (a secas)

**LPS**:
Sistema de coordenadas del proyecto: x crece hacia la izquierda del paciente, y hacia posterior, z hacia superior; unidades en mm.
_Evitar_: RAS (es otro sistema, el de Slicer)
