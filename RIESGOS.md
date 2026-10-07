# Registro de riesgos — nucleo-dental v0.0.1

| ID | Peligro | Consecuencia | Mitigación |
|---|---|---|---|
| RG-001 | Implante cerca del canal no detectado | Lesión del nervio alveolar inferior | R-004, R-006, casos 001-003 |
| RG-002 | Coordenadas RAS mezcladas con LPS | Medición en el lugar equivocado sin aviso | R-002; chequeo de bordes (Paso 7) |
| RG-003 | Eje del implante invertido | Ápice donde va la plataforma | R-003; test_eje_invertido |
| RG-004 | Medir solo contra vértices de una malla gruesa | Distancia sobreestimada | R-004 (superficie); caso dorado 005: con solo vértices daría 18,07 mm (verde) en vez de 1,5 mm (rojo) |
| RG-005 | Datos de pacientes en repo público | Incumplimiento Ley 21.719 | .gitignore; solo casos sintéticos; regla 8 de CLAUDE.md |
| RG-006 | Cambio de API de VTK altera resultados | Error silencioso | vtk fijado <9.7; numpy fijado <2.5; casos dorados en cada cambio |
| RG-007 | Penetración en el canal leída como holgura, o sin gravedad visible | Plan inseguro aceptado, o corrección insuficiente | R-009: penetración en campo propio, nunca como distancia positiva; casos 003-004 |
| RG-008 | La fresa sobrepasa el ápice y el margen se mide solo contra el implante | Distancia real al nervio menor que la reportada | RP-003 (pendiente, ligado al kit); mientras tanto, considerar el sobrefresado al elegir la posición |
| RG-009 | Raíz del diente vecino no considerada en v0.0.1 | Lesión radicular sin alerta | RP-001 (pendiente); en v0.0.1 la cercanía al diente se revisa manualmente |
| RG-010 | STL del canal con normales invertidas o canal abierto | Punto dentro del canal leído como fuera: colisión no detectada | Reorientación automática de normales antes de medir; canal abierto se rechaza; test_canal_con_normales_invertidas_da_el_mismo_resultado, test_canal_abierto_se_rechaza |
