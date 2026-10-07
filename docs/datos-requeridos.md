# Datos requeridos por etapa

Qué hay que preparar, cuándo y con qué características. Nada de esto se sube
al repositorio: todo dato real vive en `datos_locales/` (ignorado por Git).

## Etapa 1 — v0.0.1 (casos dorados): NO requiere datos reales

Solo geometría sintética. El canal (`canal_recto.stl`) lo genera
`scripts/generar_casos_dorados.py`; el implante es un cilindro definido por
parámetros. Lo único que aporta la persona son los JSON de cada caso, con los
resultados esperados calculados a mano.

## Etapa 2 — Paso 7: primer caso real de medición (CBCT + canal)

| Insumo | Características | Cómo obtenerlo |
|---|---|---|
| CBCT | Serie DICOM completa de un paciente con brecha unitaria posterior mandibular. Vóxel ≤ 0,3 mm (ideal 0,2). FOV que incluya el canal mandibular de la hemiarcada completa, del foramen mandibular al mentoniano. Sin artefacto de movimiento; los artefactos metálicos deben quedar lejos de la zona. | Archivo de tu práctica, con consentimiento y autorización para uso en investigación (Ley 20.584 y Ley 21.719). |
| Anonimización | Sin nombre, RUT, fecha de nacimiento, institución, médico tratante ni fechas identificables. Revisar también los tags privados. | Módulo DICOM de Slicer (anonimizar al exportar) o `gdcmanon`. Verificar el resultado abriendo los tags. |
| Canal mandibular | STL cerrado del canal segmentado en el CBCT, exportado en **LPS** (revisar el diálogo de guardado de Slicer). | Segmentación en Slicer (umbral + pintura, o curva con radio). |
| Implante de referencia | Ápice (x, y, z) en LPS, eje, diámetro y largo de una posición que conozcas. | Planificado en Slicer como cilindro, o tomado de un plan existente. |
| Medición manual de referencia | Distancia mínima implante-canal medida a mano en Slicer, con captura de pantalla. | Regla de Slicer en el corte donde se ven más cerca. |

Estructura local:

```text
datos_locales/caso_real_01/
├── cbct/          serie DICOM anonimizada
├── canal.stl      canal segmentado, LPS
├── implante.json  ápice, eje, diámetro, largo
└── referencia.md  medición manual y capturas
```

## Etapa 3 — Guía dentosoportada (unitario posterior)

| Insumo | Características |
|---|---|
| Escaneo intraoral o de modelo | STL de la arcada del mismo paciente del CBCT, con al menos los dos dientes vecinos a cada lado de la brecha. Debe ser cercano en el tiempo al CBCT, sin cambios dentarios entre ambos. |
| Registro CBCT ↔ escaneo | Al menos 3 puntos homólogos bien distribuidos (cúspides, bordes incisales) identificables en ambos, o registro hecho en Slicer y exportado ya transformado a LPS. |
| Especificación del implante | Marca, sistema, diámetro, largo y forma (cilíndrico o cónico). En el MVP se modela como cilindro (R-003). |
| Kit quirúrgico guiado | Diámetro interno y externo de la camisa, altura de la camisa, offset (distancia de la plataforma al tope de la camisa) y largos de las fresas. Son decisiones clínicas: van a REQUISITOS.md. |
| Parámetros de la guía | Grosor, tolerancia de ajuste (el código usa 0,20 mm por defecto), extensión del apoyo y ventanas de inspección. También son decisiones clínicas. |

## Etapa 4 — Validación en fantoma

| Insumo | Características |
|---|---|
| Fantoma | Mandíbula de hueso sintético (tipo Sawbones) o impresa a partir de un CBCT, con dientes vecinos o con un modelo dentado acoplado. |
| CBCT preoperatorio del fantoma | Mismo protocolo que en la etapa 2. |
| Escaneo de superficie del fantoma | STL, como en la etapa 3. |
| Guía impresa | Resina biocompatible para guías quirúrgicas, en la impresora que se usaría en clínica. |
| Implantes de prueba | Del mismo sistema del kit, colocados con el protocolo guiado. |
| CBCT postoperatorio | Para medir la desviación entre el plan y el resultado: entrada, ápice y ángulo. |
