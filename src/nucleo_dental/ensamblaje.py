"""Ensamblaje de la guía quirúrgica imprimible (R-021).

Todo se resuelve en UNA grilla, rotada para que el eje de inserción sea +z,
con campos de distancia continuos (positivos dentro de cada sólido):

    F = mín( máx(d_pieza ...),        unión de carcasa, columna (y puente, si lo hay)
             −d_escaneo − T,          tolerancia de ajuste T guía–escaneo
             −sombra,                 lo que choca al retirar la guía por el eje
             −d_negativo ... )        orificio de la fresa y alivio

La guía es la isosuperficie F = 0 (Flying Edges interpola cada arista, así que
las caras quedan en posiciones sub-vóxel). La sombra en un punto es el máximo
del campo del escaneo dilatado en T, (d_escaneo + T), sobre la columna que está
por encima del punto: es positiva si al subir la guía ese punto pasaría a
menos de T del escaneo. Por eso las caras laterales quedan a T del diente en
todo el recorrido de inserción, no solo en la posición final.

Por qué no se usan las booleanas de vóxeles de los motores (malla_guia,
undercut_core): reconstruyen la guía entera desde una máscara binaria y mueven
caras que no tocan nada (en la escena sintética, la cara superior del anillo
subía 0,125 mm, el piso del puente bajaba 0,3 mm y el orificio se achicaba
0,1 mm). Aquí cada término solo actúa donde es el mínimo.

El escaneo intraoral cumple el papel del "hueso" de los motores: es la
superficie cerrada sobre la que asienta la guía. Los límites del apoyo los
decide el usuario (curva cerrada) o la propuesta de apoyo.region_automatica.
"""

from __future__ import annotations

import numpy as np

from nucleo_dental.implante import Implante
from nucleo_dental.kits import PerfilKit

ESPACIADO_POR_DEFECTO_MM = 0.15   # lado del vóxel de la grilla de campos
_MARGEN_GRILLA_MM = 1.0
_PASADAS_NORMALES_APOYO = 10
_PASADAS_BORDE_APOYO = 10
_LEJOS_MM = 1000.0
_ARRAY_HOLGURA = "HolguraEncia"
# Piezas sueltas por debajo de este volumen se descartan como astillas: aparecen
# donde la cara externa se pliega sobre una fisura cóncava (caso real: 0,02 y
# 0,08 mm³). Una pieza real de una carcasa de 3 mm supera los 3 mm³.
# Confirmado por Francisco el 2026-10-09.
VOLUMEN_MINIMO_PIEZA_MM3 = 1.0
# Pasadas de promedio de la marca diente/encía: la carcasa sube a la holgura en
# una rampa de ~1 mm (aristas del escaneo de ~0,2–0,3 mm) en vez de un escalón.
_PASADAS_TRANSICION_ENCIA = 10


def eje_desde_plano_oclusal(puntos, referencia) -> np.ndarray:
    """Eje de inserción: normal unitaria al plano de 3 puntos, del lado de `referencia`.

    `referencia` es el sentido en que se retira la guía (p. ej. el eje del
    implante, que va del ápice a la plataforma, hacia oclusal).
    """
    p = np.asarray(puntos, dtype=float)
    if p.shape != (3, 3):
        raise ValueError("El plano oclusal se define con exactamente 3 puntos (x, y, z).")
    normal = np.cross(p[1] - p[0], p[2] - p[0])
    largo = np.linalg.norm(normal)
    escala = max(np.linalg.norm(p[1] - p[0]), np.linalg.norm(p[2] - p[0])) ** 2
    if largo <= 1e-6 * escala:
        raise ValueError("Los 3 puntos del plano oclusal están alineados: no definen un plano.")
    normal /= largo
    sentido = float(normal @ np.asarray(referencia, dtype=float))
    if abs(sentido) < 1e-9:
        raise ValueError("La referencia es paralela al plano oclusal: no indica hacia dónde se retira la guía.")
    return normal if sentido > 0 else -normal


def piezas_de_apoyo(escaneo, mascara, kit: PerfilKit, eje_insercion, holgura_por_vertice=None) -> list:
    """Sólidos del apoyo: cada pieza de la región con el espesor de la plantilla, hacia afuera.

    La región puede tener varias piezas (dientes separados por encía); se
    devuelven por separado y se unen en el ensamblaje. `holgura_por_vertice`
    (mm, un valor por vértice del escaneo) levanta la carcasa sobre esos
    vértices, p. ej. sobre la encía en una guía dentosoportada.
    """
    import vtk

    from nucleo_dental.apoyo import parche_de_apoyo
    from nucleo_dental.geometria.malla_guia import parche_a_solido
    from nucleo_dental.guia import _alisar_borde, _orientar_hacia

    n = escaneo.GetNumberOfPoints()
    holgura = np.zeros(n) if holgura_por_vertice is None else np.asarray(holgura_por_vertice, dtype=float)
    parche = parche_de_apoyo(escaneo, np.asarray(mascara, bool), datos={_ARRAY_HOLGURA: holgura})
    piezas = vtk.vtkPolyDataConnectivityFilter()
    piezas.SetInputData(parche)
    piezas.SetExtractionModeToAllRegions()
    piezas.Update()
    solidos = []
    for i in range(piezas.GetNumberOfExtractedRegions()):     # una vuelta por pieza, no por vértice
        una = vtk.vtkPolyDataConnectivityFilter()
        una.SetInputData(parche)
        una.SetExtractionModeToSpecifiedRegions()
        una.AddSpecifiedRegion(i)
        limpia = vtk.vtkCleanPolyData()
        limpia.SetInputConnection(una.GetOutputPort())
        limpia.Update()
        pieza = _alisar_borde(_orientar_hacia(limpia.GetOutput(), eje_insercion), _PASADAS_BORDE_APOYO)
        solidos.append(parche_a_solido(pieza, kit.espesor_plantilla_mm, desfase=_ARRAY_HOLGURA,
                                       suavizado_normales=_PASADAS_NORMALES_APOYO))
    return solidos


def holgura_sobre_encia(escaneo, dientes, holgura_mm: float) -> np.ndarray:
    """Holgura por vértice del escaneo: `holgura_mm` sobre la encía, 0 sobre los dientes.

    La marca diente/encía (apoyo.clasificar_diente_encia) se promedia con los
    vecinos para que la carcasa suba en rampa y no en escalón.
    """
    import vtk
    from vtk.util import numpy_support

    from nucleo_dental.apoyo import clasificar_diente_encia

    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(escaneo)
    tri.Update()
    triangulos = numpy_support.vtk_to_numpy(tri.GetOutput().GetPolys().GetConnectivityArray()).reshape(-1, 3)
    encia = (~clasificar_diente_encia(escaneo, dientes)).astype(float)
    for _ in range(_PASADAS_TRANSICION_ENCIA):
        suma = encia.copy()
        cuenta = np.ones(len(encia))
        for k in range(3):   # cada vértice recibe los valores de los otros dos vértices de cada triángulo
            np.add.at(suma, triangulos[:, k], encia[triangulos[:, (k + 1) % 3]] + encia[triangulos[:, (k + 2) % 3]])
            np.add.at(cuenta, triangulos[:, k], 2)
        encia = suma / cuenta
    return holgura_mm * encia


def ensamblar(positivos, escaneo, eje_insercion, tolerancia_mm: float, negativos=(),
              espaciado_mm: float = ESPACIADO_POR_DEFECTO_MM,
              volumen_minimo_pieza_mm3: float = VOLUMEN_MINIMO_PIEZA_MM3) -> dict:
    """Une los positivos, aplica tolerancia y desbloqueo por el eje, resta los negativos y valida.

    positivos, negativos: sólidos cerrados (vtkPolyData). Devuelve {"guia",
    "valida", "problemas", "metricas", ...}. Una guía no válida se devuelve
    igual, para inspeccionarla, pero nunca debe imprimirse.
    """
    from nucleo_dental.geometria import malla_guia as mg
    from nucleo_dental.geometria import tolerancia as tl
    from nucleo_dental.geometria import undercut_core as uc

    if not positivos:
        raise ValueError("No hay piezas que ensamblar.")
    if tolerancia_mm < 0:
        raise ValueError("La tolerancia de ajuste no puede ser negativa.")
    if not _es_cerrada(escaneo):
        raise ValueError("El escaneo debe ser una superficie cerrada: la tolerancia y el desbloqueo "
                         "deciden dentro/fuera del escaneo.")
    eje = np.asarray(eje_insercion, dtype=float)
    eje = eje / np.linalg.norm(eje)

    rotacion = uc.rotacion_a_z(eje)
    a_z = uc._matriz_vtk(rotacion)
    positivos_z = [tl._triangular(uc._aplicar_matriz(p, a_z)) for p in positivos]
    negativos_z = [tl._triangular(uc._aplicar_matriz(n, a_z)) for n in negativos]
    escaneo_z = tl._triangular(uc._aplicar_matriz(escaneo, a_z))

    origen, dims = _grilla_comun(positivos_z, espaciado_mm, _MARGEN_GRILLA_MM + tolerancia_mm)

    def campo(malla, cerca_fuera_mm: float = 0.0):
        dentro = tl._dentro(malla, origen, dims, espaciado_mm)
        fuera = 3 + int(np.ceil(cerca_fuera_mm / espaciado_mm))
        necesario = tl._banda(dentro, 3, fuera)
        return tl.campo_distancia(malla, origen, dims, espaciado_mm, necesario=necesario, dentro=dentro,
                                  lejos=_LEJOS_MM)

    union = np.max([campo(p) for p in positivos_z], axis=0)
    dilatado = campo(escaneo_z, cerca_fuera_mm=tolerancia_mm + 2 * espaciado_mm) + tolerancia_mm
    # Sombra: máximo de `dilatado` en los nodos estrictamente por encima (eje 0 de la grilla = z).
    acumulado = np.maximum.accumulate(dilatado[::-1], axis=0)[::-1]
    sombra = np.full_like(dilatado, -_LEJOS_MM)
    sombra[:-1] = acumulado[1:]
    f = np.minimum.reduce([union, -dilatado, -sombra] + [-campo(n) for n in negativos_z])

    guia_z = tl._superficie(f, origen, espaciado_mm)
    if guia_z.GetNumberOfPoints() == 0:
        raise RuntimeError("El ensamblaje no dejó nada de la guía: revisa el eje de inserción y los límites.")
    guia_z, cavidades, fragmentos, volumen_descartado = _limpiar_piezas(guia_z, volumen_minimo_pieza_mm3)
    guia = uc._aplicar_matriz(guia_z, uc._matriz_vtk(rotacion, inversa=True))
    valida, problemas, metricas = mg.validar_guia(guia, escaneo, eje=tuple(eje))
    metricas["cavidades_rellenas"] = cavidades
    metricas["astillas_descartadas"] = fragmentos
    metricas["volumen_astillas_mm3"] = volumen_descartado
    return {
        "guia": guia,
        "valida": bool(valida),
        "problemas": problemas,
        "metricas": metricas,
        "eje_insercion": eje.tolist(),
        "tolerancia_mm": float(tolerancia_mm),
        "espaciado_mm": float(espaciado_mm),
    }


def guia_quirurgica(escaneo, dientes, implante: Implante, kit: PerfilKit, fabricacion: str, mascara_apoyo,
                    eje_insercion, tipo_soporte: str = "dentosoportada", con_puente: bool = False,
                    espaciado_mm: float = ESPACIADO_POR_DEFECTO_MM) -> dict:
    """Guía completa: carcasa dentro de los límites + columna del anillo con su orificio.

    Con los límites del usuario (curva cerrada) la carcasa ya cubre la brecha:
    en una guía dentosoportada sube `kit.holgura_encia_mm` sobre la encía y
    apoya en los dientes; si apoya en mucosa no hay holgura (R-020).
    `con_puente` agrega el puente automático (R-019), pensado para una región
    automática que solo cubre dientes.
    """
    from nucleo_dental.guia import TIPOS_SOPORTE, anillo_y_orificio, columna_del_anillo, puente_y_columna

    if tipo_soporte not in TIPOS_SOPORTE:
        raise ValueError(f"Tipo de soporte desconocido '{tipo_soporte}'; opciones: " + ", ".join(TIPOS_SOPORTE))
    holgura = kit.holgura_encia_mm if tipo_soporte == "dentosoportada" else 0.0
    if con_puente:
        pc = puente_y_columna(escaneo, dientes, implante, kit, fabricacion, tipo_soporte=tipo_soporte)
        extras = [pc["puente"]]
    else:
        pc = columna_del_anillo(escaneo, implante, kit, fabricacion, holgura)
        extras = []
    por_vertice = None
    if holgura > 0:
        if dientes is None:
            raise ValueError("Una guía dentosoportada necesita los dientes del CBCT para separar diente de encía.")
        por_vertice = holgura_sobre_encia(escaneo, dientes, holgura)
    positivos = piezas_de_apoyo(escaneo, mascara_apoyo, kit, eje_insercion, por_vertice) + extras + [pc["columna"]]
    _, orificio = anillo_y_orificio(implante, kit, fabricacion)
    negativos = [orificio] + ([pc["alivio"]] if pc["alivio"] is not None else [])
    resultado = ensamblar(positivos, escaneo, eje_insercion, kit.tolerancia_ajuste(fabricacion),
                          negativos=negativos, espaciado_mm=espaciado_mm)
    resultado.update({"puente_y_columna": {k: v for k, v in pc.items() if k not in ("puente", "columna", "alivio")},
                      "tipo_soporte": tipo_soporte, "con_puente": con_puente, "holgura_encia_mm": holgura,
                      "fabricacion": fabricacion, "kit": kit.nombre})
    return resultado


def _limpiar_piezas(malla, volumen_minimo_mm3: float):
    """(malla limpia, cavidades rellenadas, astillas descartadas, volumen de las astillas).

    Cavidad: superficie cerrada entera dentro del cuerpo principal (burbuja
    donde una pieza se autointersecta y la paridad del estencil invierte el
    signo); descartarla deja ese volumen macizo. Astilla: pieza suelta de menos
    de `volumen_minimo_mm3`. Cualquier otra pieza suelta se conserva para que
    la validación la reporte.
    """
    import vtk
    from vtk.util import numpy_support

    from nucleo_dental.medicion import _dentro

    piezas = vtk.vtkPolyDataConnectivityFilter()
    piezas.SetInputData(malla)
    piezas.SetExtractionModeToAllRegions()
    piezas.ColorRegionsOn()
    piezas.Update()
    n = piezas.GetNumberOfExtractedRegions()
    if n <= 1:
        return _solo_triangulos(malla), 0, 0, 0.0
    salida = piezas.GetOutput()
    region_punto = numpy_support.vtk_to_numpy(salida.GetPointData().GetArray("RegionId"))
    puntos = numpy_support.vtk_to_numpy(salida.GetPoints().GetData()).astype(float)
    triangulos = numpy_support.vtk_to_numpy(salida.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    region_celda = region_punto[triangulos[:, 0]]
    # Volumen con signo de cada pieza (teorema de la divergencia), vectorizado por triángulo.
    v0, v1, v2 = puntos[triangulos[:, 0]], puntos[triangulos[:, 1]], puntos[triangulos[:, 2]]
    volumen = np.abs(np.bincount(region_celda, weights=np.einsum("ij,ij->i", v0, np.cross(v1, v2)) / 6.0,
                                 minlength=n))
    principal = int(np.argmax(volumen))
    cuerpo = vtk.vtkPolyDataConnectivityFilter()
    cuerpo.SetInputData(malla)
    cuerpo.SetExtractionModeToSpecifiedRegions()
    cuerpo.AddSpecifiedRegion(principal)
    cuerpo.Update()
    otras = np.array([r for r in range(n) if r != principal])
    primero = np.full(n, -1)
    primero[region_punto[::-1]] = np.arange(len(region_punto))[::-1]   # primer punto de cada región
    adentro = _dentro(cuerpo.GetOutput(), puntos[primero[otras]])
    fragmento = volumen[otras] < volumen_minimo_mm3
    conservar = [principal] + otras[~adentro & ~fragmento].tolist()
    final = vtk.vtkPolyDataConnectivityFilter()
    final.SetInputData(malla)
    final.SetExtractionModeToSpecifiedRegions()
    for r in conservar:
        final.AddSpecifiedRegion(int(r))
    limpia = vtk.vtkCleanPolyData()
    limpia.SetInputConnection(final.GetOutputPort())
    limpia.Update()
    astillas = fragmento & ~adentro
    return (_solo_triangulos(limpia.GetOutput()), int(adentro.sum()), int(astillas.sum()),
            float(volumen[otras][astillas].sum()))


def _solo_triangulos(malla):
    """Quita las líneas y vértices sueltos que deja vtkCleanPolyData al colapsar triángulos degenerados."""
    import vtk

    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(malla)
    tri.PassVertsOff()
    tri.PassLinesOff()
    tri.Update()
    return tri.GetOutput()


def _grilla_comun(mallas, espaciado: float, margen: float):
    limites = np.array([m.GetBounds() for m in mallas])
    bajo = limites[:, 0::2].min(axis=0) - margen
    alto = limites[:, 1::2].max(axis=0) + margen
    dims = [int(np.ceil((alto[i] - bajo[i]) / espaciado)) + 1 for i in range(3)]
    return bajo.tolist(), dims


def _es_cerrada(malla) -> bool:
    import vtk

    bordes = vtk.vtkFeatureEdges()
    bordes.SetInputData(malla)
    bordes.BoundaryEdgesOn()
    bordes.NonManifoldEdgesOn()
    bordes.FeatureEdgesOff()
    bordes.ManifoldEdgesOff()
    bordes.Update()
    return bordes.GetOutput().GetNumberOfCells() == 0
