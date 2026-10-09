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

- ✅ Registro del escaneo intraoral con el CBCT (R-015): `registrar` con caso dorado 011; caso real: coronas 0,61 → 0,31 mm (p90), corrección de 0,45 mm en el ápice. Umbral clínico de aceptación: se decidirá con más casos
- ✅ Región de apoyo (R-016): modo automático (R = 24 mm, 1 mm de la encía) y modo curva por puntos; comando `nucleo-dental apoyo`
- ✅ Sobrefresado configurable en la medición al canal (R-017; OneGuide 0,3 mm provisional)
- ☐ Pincel y edición interactiva de la región (etapa 5, en la app)
- ✅ Perfil de kit configurable y anillo guía con orificio (R-018): OneGuide; caso dorado 015
- ✅ Puente sobre la brecha y columna del orificio (R-019): caso dorado 016; caso real generado para revisión visual
- ✅ Tipos de soporte (dento, dentomuco, muco), profundidad del puente (6 mm) y alivio configurables (R-020)
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
