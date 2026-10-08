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
| R-010 | Con --cbct, el comando verifica que todos los vértices del canal (y de los dientes, si se entregan), el ápice y la plataforma caen dentro de la caja física del CBCT (calculada con origen, espaciado, tamaño y matriz de dirección); si no, termina con código 1 y sugiere revisar RAS/LPS. La salida incluye tamaño, espaciado, origen, dirección, SHA-256 del primer archivo y de la serie completa | test_fuera_de_volumen, test_implante_fuera_de_volumen, test_volumen_oblicuo_usa_la_matriz_de_direccion |
| R-013 | Con --dientes (o `"dientes"` en caso.json), el implante se evalúa también contra los dientes con margen propio: 1,5 mm por defecto (RP-001), configurable con --margen-dientes. La salida reporta distancia, colisión, penetración, margen y semáforo por estructura; el semáforo global (y el código de salida) es rojo si alguna estructura es roja | casos dorados 006-007, test_semaforo_global_es_rojo_si_alguna_estructura_es_roja, test_dientes_* |
| R-012 | El implante puede leerse desde el STL planificado (`--implante-stl`, o `"stl"` en caso.json): se reconoce el cilindro y se obtienen diámetro, largo, ápice y eje; `--apice-hacia` (abajo/arriba en z LPS) es obligatorio y se rechaza un implante a más de 60° de la vertical o una malla que no sea cilindro. Si además se declaran diámetro o largo, una diferencia mayor que 0,1 mm con el STL es error de entrada (decisiones clínicas 2026-10-08) | test_desde_malla_*, test_implante_stl_*, test_caso_real_con_implante_stl, test_caso_real_largo_declarado_erroneo_se_rechaza |
| R-011 | En casos reales, `medir` concuerda con la distancia mínima medida a mano por el clínico en 3D Slicer dentro de 0,1 mm (tolerancia de medición manual, decisión clínica 2026-10-08). Los datos reales viven fuera del repositorio | test_caso_real_contra_medicion_manual (caso inferior_prueba: manual 2,330 mm) |

## Requisitos pendientes (fuera de v0.0.1)

Decisiones clínicas ya tomadas que aún no tienen código ni test. Al implementarlas se les asigna un ID R-0xx y un caso dorado.

| ID | Requisito | Origen |
|---|---|---|
| ~~RP-001~~ | Margen implante–diente vecino: 1,5 mm de superficie del implante a superficie de la raíz | Implementado como R-013 (2026-10-08) |
| RP-002 | Margen implante–implante: 1,5 mm alrededor de cada implante (3 mm entre superficies). Fuera del MVP de implante unitario | Decisión clínica 2026-10-07 |
| RP-003 | La medición al canal considera el sobrefresado: lo que la fresa sobrepasa al ápice según el kit guiado. Se define junto con la especificación del kit | Decisión clínica 2026-10-07; RG-008 |
