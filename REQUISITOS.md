# Requisitos — nucleo-dental v0.0.1

| ID | Requisito | Verificación |
|---|---|---|
| R-001 | src/ no importa slicer, qt, ctk ni mrml | test_sin_dependencias_slicer |
| R-002 | Unidades mm; coordenadas LPS en entradas y salidas | casos dorados 001-003 |
| R-003 | Implante = cilindro sólido de diámetro D y largo L; ápice en P; eje unitario u apunta del ápice a la plataforma | test_eje_invertido |
| R-004 | `medir` reporta la distancia mínima entre la superficie del implante y la superficie triangulada del canal (no solo sus vértices), con error ≤ 0,05 mm | casos dorados 001-002 |
| R-005 | Si el implante interseca el canal: distancia 0 y colision = true | caso dorado 003 |
| R-006 | Semáforo: verde si distancia ≥ margen; rojo si distancia < margen o colisión. Margen por defecto 2,0 mm, configurable con --margen | casos dorados 001-003 |
| R-007 | Cada salida incluye versión del motor, commit de git, parámetros, SHA-256 de cada archivo de entrada y fecha-hora UTC | test_trazabilidad |
| R-008 | Código de salida: 0 = verde, 2 = rojo, 1 = error de entrada | test_codigos_salida |
