# Registro de riesgos — nucleo-dental v0.0.1

| ID | Peligro | Consecuencia | Mitigación |
|---|---|---|---|
| RG-001 | Implante cerca del canal no detectado | Lesión del nervio alveolar inferior | R-004, R-006, casos 001-003 |
| RG-002 | Coordenadas RAS mezcladas con LPS | Medición en el lugar equivocado sin aviso | R-002; chequeo de bordes (Paso 7) |
| RG-003 | Eje del implante invertido | Ápice donde va la plataforma | R-003; test_eje_invertido |
| RG-004 | Medir solo contra vértices de una malla gruesa | Distancia sobreestimada | R-004 (superficie); pendiente caso con malla gruesa |
| RG-005 | Datos de pacientes en repo público | Incumplimiento Ley 21.719 | .gitignore; solo casos sintéticos; regla 8 de CLAUDE.md |
| RG-006 | Cambio de API de VTK altera resultados | Error silencioso | vtk fijado <9.7; casos dorados en cada cambio |
