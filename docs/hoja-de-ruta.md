# Hoja de ruta — nucleo-dental

Meta del MVP: planificar un **implante unitario posterior** y generar su **guía quirúrgica dentosoportada**, validada en fantoma. Cada tarea se cierra con tests en verde y requisitos trazables (REQUISITOS.md, RIESGOS.md).

Estado: ✅ hecho · ▶ en curso · ☐ pendiente

## Etapa 1 — Medir seguridad respecto del canal ✅

- ✅ Distancia mínima implante–canal en dos direcciones, colisión, penetración y semáforo (R-004 a R-006, R-009)
- ✅ Casos dorados 001–005 calculados a mano, incluida malla gruesa (RG-004)
- ✅ Comando `nucleo-dental medir` con trazabilidad y códigos de salida (R-007, R-008); etiqueta `v0.0.1`
- ✅ Chequeo RAS/LPS contra la caja del CBCT (R-010)
- ✅ Implante leído desde su STL (R-012)
- ✅ Primer caso real anonimizado validado contra medición manual: 2,330 mm (R-011)

## Etapa 2 — Completar la seguridad del plan ✅ (cerrada el 2026-10-08)

- ✅ **2a** Distancia al diente vecino bajo la plataforma, margen 1,5 mm (R-013): casos dorados 006-008 y caso real (programa 7,530 mm = manual 7,529 mm)
- ✅ **2b** Espesor óseo mínimo de 1,5 mm en las paredes laterales (R-014): casos dorados 009-010; caso real: dehiscencia vestibular en la plataforma (programa 0,229 mm = manual 0,216 mm); cavidades internas informadas
- → **2c** Sobrefresado: pasa a la etapa 3, porque depende de los datos del kit
- → **2d** Canal contra la imagen del CBCT (RG-011): pasa a *Pendientes de investigación*

## Etapa 3 — Guía quirúrgica dentosoportada ▶

Reutiliza los motores MIT `malla_guia`, `undercut_core` y `tolerancia`, y el escaneo intraoral.

- ☐ Verificar que el escaneo intraoral esté registrado con el CBCT
- ☐ Región de apoyo sobre los dientes vecinos
- ☐ Camisa (sleeve) según la especificación del kit
- ☐ Sobrefresado según el kit en la medición al canal (RP-003, RG-008)
- ☐ Eje de inserción, eliminación de undercuts y tolerancia de ajuste
- ☐ Validación geométrica y exportación de STL imprimible

## Etapa 4 — Validación en fantoma ☐

- ☐ Fantoma, guía impresa, implantes de prueba y CBCT posterior
- ☐ Desviación plan vs. resultado: entrada, ápice y ángulo

## Etapa 5 — Aplicación ☐

- ☐ Interfaz PySide6: asistente guiado y modo experto, delgada sobre el motor
- ☐ Segmentación automática de CBCT (canal, mandíbula, dientes) con [SlicerDentalSegmentator](https://github.com/gaudot/SlicerDentalSegmentator) (nnU-Net). Código Apache 2.0; falta revisar la licencia de los pesos del modelo. Sus salidas deben pasar los mismos chequeos (LPS, malla cerrada, R-010)
- ☐ Optimizar el espesor óseo con mallas grandes (hoy ~17 s con la mandíbula real)
- ☐ Servidor MCP que envuelva el motor, sin lógica nueva (al final)

## Pendientes de investigación

- ☐ Verificar el canal contra la imagen del CBCT (RG-011): detectar un RAS cuyo reflejo cae dentro del FOV
- ☐ Implantes cónicos de biblioteca (hoy solo cilindros, R-003)
