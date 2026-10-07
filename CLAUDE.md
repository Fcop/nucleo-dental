# nucleo-dental — reglas del proyecto

## Contexto
Núcleo sin interfaz para planificar implantes guiados. Destino final: uso
clínico tras validación. Trátalo como software médico.

## Reglas
1. Código, comentarios y documentación en español.
2. Prohibido importar slicer, qt, ctk o mrml en src/. Solo numpy y vtk.
3. Unidades en mm. Sistema de coordenadas LPS en toda entrada y salida.
4. Todo requisito de REQUISITOS.md tiene al menos un test que lo cite
   en su docstring (ej. "Verifica R-004").
5. NUNCA modifiques tests/casos_dorados/*/esperado.json ni sus valores.
   Si un test dorado falla, el error está en el código: avísame y detente.
6. Operaciones geométricas vectorizadas con numpy; nada de bucles Python
   sobre vértices o caras.
7. Antes de dar una tarea por terminada, ejecuta `pytest` y muéstrame
   el resultado completo.
8. Nunca agregues archivos DICOM ni datos de pacientes al repositorio.
9. Si una decisión clínica no está en REQUISITOS.md (márgenes, ejes,
   tolerancias), pregúntame; no la inventes.

## Agent skills

### Issue tracker

Los issues viven en GitHub Issues de `Fcop/nucleo-dental` (vía `gh`). Ver `docs/agents/issue-tracker.md`.

### Triage labels

Vocabulario por defecto: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. Ver `docs/agents/triage-labels.md`.

### Domain docs

Un solo contexto: `CONTEXT.md` y `docs/adr/` en la raíz. Ver `docs/agents/domain.md`.
