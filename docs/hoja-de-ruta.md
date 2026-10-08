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

## Etapa 2 — Completar la seguridad del plan ▶

- ▶ **2a** Distancia a la raíz del diente vecino, margen 1,5 mm (R-013, RG-009): implementado con casos dorados 006-007; falta validar con el caso real contra medición manual
- ☐ **2b** Implante contenido en el hueso, sin perforar cortical (requiere definir el espesor mínimo)
- ☐ **2c** Sobrefresado según el kit (RP-003, RG-008), junto con la guía
- ☐ **2d** Verificación del canal contra la imagen del CBCT (RG-011)

## Etapa 3 — Guía quirúrgica dentosoportada ☐

Reutiliza los motores MIT `malla_guia`, `undercut_core` y `tolerancia`, y el escaneo intraoral.

- ☐ Verificar que el escaneo intraoral esté registrado con el CBCT
- ☐ Región de apoyo sobre los dientes vecinos
- ☐ Camisa (sleeve) según la especificación del kit
- ☐ Eje de inserción, eliminación de undercuts y tolerancia de ajuste
- ☐ Validación geométrica y exportación de STL imprimible

## Etapa 4 — Validación en fantoma ☐

- ☐ Fantoma, guía impresa, implantes de prueba y CBCT posterior
- ☐ Desviación plan vs. resultado: entrada, ápice y ángulo

## Etapa 5 — Aplicación ☐

- ☐ Interfaz PySide6: asistente guiado y modo experto, delgada sobre el motor
- ☐ Segmentación automática de CBCT (canal, mandíbula, dientes) con [SlicerDentalSegmentator](https://github.com/gaudot/SlicerDentalSegmentator) (nnU-Net). Código Apache 2.0; falta revisar la licencia de los pesos del modelo. Sus salidas deben pasar los mismos chequeos (LPS, malla cerrada, R-010)
- ☐ Servidor MCP que envuelva el motor, sin lógica nueva (al final)
