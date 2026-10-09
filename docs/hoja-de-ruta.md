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
- ✅ Corrección del puente (RG-017): holgura y espesor con las mismas normales (ya no entra en el hueso) y contorno alisado
- ✅ Límites de la guía definidos por el usuario con la curva cerrada por puntos (decisión 2026-10-09): `guia` exige `--curva`; lo automático (`apoyo`) queda como propuesta inicial
- ✅ Ensamblaje (R-021): eje de inserción normal al plano oclusal (3 puntos), recorte de undercuts, tolerancia de ajuste por fabricación, resta del orificio, validación y exportación STL; comando `nucleo-dental guia`; caso dorado 017
- ✅ `guia` en el caso real con la curva CC y el plano oclusal de Francisco: válida en dento y dentomucosoportada (eje de inserción a 18,2° del implante)
- ✅ Carcasa desde la curva sin puente; holgura de 1 mm sobre la encía solo en dentosoportada
- ✅ Informe de ajuste guía–escaneo antes de imprimir (R-022): objetivo por punto, histograma y mapa .vtp; una interferencia invalida la guía; desvío aceptable 0,10 mm (confirmado)
- ✅ Varios implantes (R-023): semáforo implante–implante (3 mm), un anillo por implante, pared entre orificios (aviso < 1 mm, solape = no válida); caso dorado 018
- ☐ `medir` con varios implantes en caso.json (hoy el semáforo implante–implante sale en `guia`)
- ☐ Ventanas de inspección para verificar el asiento de la guía (negativos sobre los dientes)
- ☐ Pines de fijación al hueso, con su dirección y su camisa (clave en dentomuco y mucosoportada)
- ☐ Texto grabado en la guía: caso, kit y posición del implante (trazabilidad)
- ☐ Informe del protocolo quirúrgico (PDF): kit, orificio, largo de trabajo de la fresa, offset
  (Ideas de funciones tomadas de B4D Guide3, software propietario: solo la idea, implementación propia)
- ☐ Otros métodos de eje de inserción en la app: plano de la vista, flecha, búsqueda del eje con menos undercut (`buscar_eje` de GuiaCorte) (etapa 5)
- ☐ Problemas de los módulos externos de Slicer, para corregirlos en sus proyectos: `docs/problemas-modulos-externos.md`

## Etapa 3b — Guías apilables (stackable) ☐

Sesión aparte (decisión 2026-10-09): son más complejas. Varias guías que se apilan sobre una base fijada al hueso con pines: base, reducción ósea, guía de implantes y prótesis provisional, cada una referenciada a la anterior. Referencia: https://www.3ddx.com/what-is-a-stackable-surgical-guide-and-when-do-you-need-one/

- ☐ Base con pines de fijación y sus referencias de encaje
- ☐ Guía de reducción ósea sobre la base
- ☐ Guía de implantes apilada sobre la base
- ☐ Transferencia a la prótesis provisional

## Etapa 4 — Validación en fantoma ☐

- ☐ Fantoma, guía impresa, implantes de prueba y CBCT posterior
- ☐ Desviación plan vs. resultado: entrada, ápice y ángulo
- ☐ Imprimir la guía del caso real (curva CC) y comprobar asiento e inserción

## Etapa 5 — Aplicación ☐

- ☐ Interfaz PySide6: asistente guiado y modo experto, delgada sobre el motor
- ☐ Segmentación automática de CBCT (canal, mandíbula, dientes) con [SlicerDentalSegmentator](https://github.com/gaudot/SlicerDentalSegmentator) (nnU-Net). Código Apache 2.0; falta revisar la licencia de los pesos del modelo. Sus salidas deben pasar los mismos chequeos (LPS, malla cerrada, R-010)
- ☐ Optimizar el espesor óseo con mallas grandes (hoy ~17 s con la mandíbula real)
- ☐ Servidor MCP que envuelva el motor, sin lógica nueva (al final)

## Pendientes de investigación

- ☐ Verificar el canal contra la imagen del CBCT (RG-011): detectar un RAS cuyo reflejo cae dentro del FOV
- ☐ Implantes cónicos de biblioteca (hoy solo cilindros, R-003)
