# Registro de riesgos — nucleo-dental v0.0.1

| ID | Peligro | Consecuencia | Mitigación |
|---|---|---|---|
| RG-001 | Implante cerca del canal no detectado | Lesión del nervio alveolar inferior | R-004, R-006, casos 001-003 |
| RG-002 | Coordenadas RAS mezcladas con LPS | Medición en el lugar equivocado sin aviso | R-002; R-010 (chequeo contra la caja del CBCT). Limitación: si la estructura queda simétrica respecto del origen (x, y) del sistema, la versión RAS también cae dentro y no se detecta; ver RG-011 |
| RG-012 | STL de exportadores reales con normales NaN | El lector de VTK aborta: no se puede medir un caso válido | `leer_stl` ignora las normales guardadas y rechaza solo vértices no finitos; test_stl_con_normales_no_finitas_se_lee_igual, test_caso_real |
| RG-011 | El chequeo de caja (R-010) no detecta un RAS/LPS cuya imagen invertida sigue dentro del volumen | Canal del lado contrario o desplazado, dentro del FOV, medido sin aviso | Pendiente: verificar contra la imagen (p. ej. que el canal coincida con un conducto de baja densidad rodeado de hueso). Mientras tanto, revisión visual del canal sobre el CBCT |
| RG-003 | Eje del implante invertido | Ápice donde va la plataforma | R-003; test_eje_invertido; con STL, `--apice-hacia` obligatorio y rechazo a más de 60° de la vertical (R-012) |
| RG-013 | Medidas del implante transcritas a mano (p. ej. 10 mm en vez de 8) | Plataforma y longitud erróneas en la planificación | R-012: leer el implante desde su STL y contrastar lo declarado con tolerancia 0,1 mm; ocurrió en el caso real del 2026-10-08 |
| RG-004 | Medir solo contra vértices de una malla gruesa | Distancia sobreestimada | R-004 (superficie); caso dorado 005: con solo vértices daría 18,07 mm (verde) en vez de 1,5 mm (rojo) |
| RG-005 | Datos de pacientes en repo público | Incumplimiento Ley 21.719 | .gitignore (`*.dcm`, `datos_locales/`, `tests/casos_prueba/`); casos dorados solo sintéticos; tests con datos reales se saltan si la carpeta no existe; regla 8 de CLAUDE.md |
| RG-006 | Cambio de API de VTK altera resultados | Error silencioso | vtk fijado <9.7; numpy fijado <2.5; casos dorados en cada cambio |
| RG-007 | Penetración en el canal leída como holgura, o sin gravedad visible | Plan inseguro aceptado, o corrección insuficiente | R-009: penetración en campo propio, nunca como distancia positiva; casos 003-004 |
| RG-008 | La fresa sobrepasa el ápice y el margen se mide solo contra el implante | Distancia real al nervio menor que la reportada | RP-003 (pendiente, ligado al kit); mientras tanto, considerar el sobrefresado al elegir la posición |
| RG-009 | Raíz del diente vecino no considerada | Lesión radicular sin alerta | R-013 con --dientes (casos 006-007). Riesgo residual: si no se entrega la segmentación de dientes, no se evalúa; la salida solo lista las estructuras recibidas |
| RG-010 | STL del canal con normales invertidas o canal abierto | Punto dentro del canal leído como fuera: colisión no detectada | Reorientación automática de normales antes de medir; canal abierto se rechaza; test_canal_con_normales_invertidas_da_el_mismo_resultado, test_canal_abierto_se_rechaza |
