# Problemas encontrados en los módulos externos de Slicer

Módulos de Francisco, revisados el 2026-10-09 al compararlos con las copias de
`src/nucleo_dental/geometria/`. Viven en sus propios proyectos (fuera de este
repositorio, `.gitignore`); esta lista es para corregirlos allí.

| Módulo | Archivo |
|---|---|
| GuiaCorteTraumatologica | `GuiaCorte/GuiaCorteLib/{malla_guia,undercut_core,tolerancia}.py` |
| EliminarUndercuts | `EliminarUndercutsLib/undercut_core.py` (idéntico al de GuiaCorte) |

## 1. Las booleanas de vóxeles mueven caras que no deberían tocar

`malla_guia.booleana()` y `undercut_core.recortar_guia()` reconstruyen la guía
**entera** desde una máscara binaria (más el suavizado gaussiano del
antialias). Toda cara queda cuantizada al vóxel, incluso las que no participan
de la operación. Medido en una escena sintética con vóxel 0,15 mm:

- orificio de Ø5,3 restado como negativo: salió de Ø5,20–5,25 (hasta 0,1 mm menos);
- cara superior plana en z = 8,5: salió en 8,625;
- piso de un sólido en z = 1,0: salió en 0,675.

En la guía de corte esto afecta el **ancho de la ranura de la sierra** y la
posición de la pestaña. `ensamblar_guia()` aplica después
`aplicar_tolerancia()`, que es continua, pero no deshace lo que movió el paso
binario.

**Solución usada aquí** (`src/nucleo_dental/ensamblaje.py`): todo en una sola
grilla con campos de distancia continuos (los de `tolerancia.campo_distancia`):
`F = mín(máx(d_positivos), −d_hueso − T, −sombra, −d_negativos)`, donde la
sombra es el máximo acumulado de `d_hueso + T` a lo largo del eje. Las cotas
quedan sub-vóxel (Ø5,30, z = 8,50) y se reconstruye una sola vez.

## 2. `recortar_guia()` deja un undercut residual de hasta 1 vóxel

La sombra se calcula sobre el hueso **sin dilatar**
(`_sombra(ah > 0, gap_v)`), y la holgura solo se aplica al solape directo
(`ah_dil`). En un diente con la corona 1 mm más ancha que el cuello, la cara
interna de la guía bajo la corona quedó en |x| = 3,93 con la corona en |x| = 4:
0,07 mm dentro de la sombra, es decir, una traba al retirar la guía.

**Corrección mínima:** `bloquea = (_sombra(ah_dil > 0, gap_v) > 0) | (ah_dil > 0)`
(probada: la cara quedó en 4,17). Lo conservador es que la sombra también
lleve la holgura.

## 3. `exportar_stl()` rotaba 180° las mallas que ya estaban en LPS

El parámetro `sistema="LPS"` significaba "convertir a LPS", y lo hacía
siempre. Una malla que ya venía en LPS salía con x e y invertidas, sin aviso.
**Corregido aquí** como `sistema_entrada` (por defecto "LPS" = no tocar; "RAS"
= convertir), con error ante cualquier otro valor.

## 4. Bordes dentados de la carcasa

`suavizar_parche()` (windowed sinc con `BoundarySmoothing`) apenas suaviza el
contorno del parche: el borde sigue la escalera de triángulos enteros que deja
`extraer_parche()` (se ve en `ejemplo_malla_guia.stl`, borde inferior en
vestibular y lingual). Además mueve el interior, que es la superficie de
asiento.

**Solución usada aquí** (`guia._alisar_borde`): Taubin solo sobre los vértices
del borde, a lo largo del propio contorno; el interior queda intacto.

## 5. API de VTK obsoleta en 9.6

- `extraer_parche()` lee `GetPolys().GetData()` (obsoleto): usar
  `GetOffsetsArray()` y `GetConnectivityArray()`.
- `parche_a_solido()` tiene la rama `SetCells` para VTK < 9: usar
  `vtkCellArray.SetData(offsets, conectividad)`.

## 6. Rendimiento de `parche_a_solido()`

Arma intrados, extrados y costillas con bucles Python sobre caras y aristas.
**Vectorizado aquí** con numpy (mismo resultado, mucho más rápido en mallas
grandes).

## 7. Extrusión con normales recalculadas (riesgo si se agrega un desfase)

`parche_a_solido()` extruye a lo largo de las normales del parche original:
eso está bien. Pero si se desplaza primero el parche (p. ej. para una holgura)
y luego se extruye recalculando normales, éstas se invierten donde la capa
desplazada se pliega y el sólido entra en el tejido (aquí llegó a 1,7 mm
dentro del hueso, RG-017). **Solución usada aquí:** un solo campo de normales
para ambas capas (`parche_a_solido(..., desfase=, suavizado_normales=)`).
