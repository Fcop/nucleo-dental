# Requisitos — nucleo-dental v0.0.1

| ID | Requisito | Verificación |
|---|---|---|
| R-001 | src/ no importa slicer, qt, ctk ni mrml | test_sin_dependencias_slicer |
| R-002 | Unidades mm; coordenadas LPS en entradas y salidas | casos dorados 001-003 |
| R-003 | Implante = cilindro sólido de diámetro D y largo L; ápice en P; eje unitario u apunta del ápice a la plataforma | test_eje_invertido |
| R-004 | `medir` reporta la distancia mínima entre la superficie del implante y la superficie triangulada del canal (no solo sus vértices), con error ≤ 0,05 mm | casos dorados 001-002 y 005 (malla gruesa) |
| R-005 | Si el implante interseca el canal: distancia 0 y colision = true | caso dorado 003 |
| R-006 | Semáforo: verde si distancia ≥ margen (una distancia igual al margen es verde); rojo si distancia < margen o colisión. Margen implante–canal mandibular por defecto 2,0 mm (confirmado clínicamente el 2026-10-07), configurable con --margen | casos dorados 001-003 |
| R-007 | Cada salida incluye versión del motor, commit de git, parámetros, SHA-256 de cada archivo de entrada y fecha-hora UTC | test_trazabilidad |
| R-008 | Código de salida: 0 = verde, 2 = rojo, 1 = error de entrada | test_codigos_salida |
| R-009 | Toda salida incluye `penetracion_mm`: la mayor distancia a la superficie del canal entre los puntos del implante que quedan dentro del canal; 0 si no hay colisión. Error ≤ 0,05 mm. La penetración nunca se reporta en `distancia_mm`, que no toma valores negativos | casos dorados 001-004 |

## Requisitos pendientes (fuera de v0.0.1)

Decisiones clínicas ya tomadas que aún no tienen código ni test. Al implementarlas se les asigna un ID R-0xx y un caso dorado.

| ID | Requisito | Origen |
|---|---|---|
| RP-001 | Margen implante–diente vecino: 1,5 mm de superficie del implante a superficie de la raíz | Decisión clínica 2026-10-07 |
| RP-002 | Margen implante–implante: 1,5 mm alrededor de cada implante (3 mm entre superficies). Fuera del MVP de implante unitario | Decisión clínica 2026-10-07 |
| RP-003 | La medición al canal considera el sobrefresado: lo que la fresa sobrepasa al ápice según el kit guiado. Se define junto con la especificación del kit | Decisión clínica 2026-10-07; RG-008 |
