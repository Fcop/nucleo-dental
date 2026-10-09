"""Ensamblaje de la guía quirúrgica imprimible (R-021).

Todo se resuelve en UNA grilla, rotada para que el eje de inserción sea +z,
con campos de distancia continuos (positivos dentro de cada sólido):

    F = mín( máx(d_pieza ...),        unión de apoyo, puente y columna
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


def piezas_de_apoyo(escaneo, mascara, kit: PerfilKit, eje_insercion) -> list:
    """Sólidos del apoyo: cada pieza de la región con el espesor de la plantilla, hacia afuera.

    La región puede tener varias piezas (dientes separados por encía); se
    devuelven por separado y se unen en el ensamblaje.
    """
    import vtk

    from nucleo_dental.apoyo import parche_de_apoyo
    from nucleo_dental.geometria.malla_guia import parche_a_solido
    from nucleo_dental.guia import _alisar_borde, _orientar_hacia

    parche = parche_de_apoyo(escaneo, np.asarray(mascara, bool))
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
        solidos.append(parche_a_solido(pieza, kit.espesor_plantilla_mm, suavizado_normales=_PASADAS_NORMALES_APOYO))
    return solidos


def ensamblar(positivos, escaneo, eje_insercion, tolerancia_mm: float, negativos=(),
              espaciado_mm: float = ESPACIADO_POR_DEFECTO_MM) -> dict:
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
    guia_z, cavidades, fragmentos = _limpiar_piezas(guia_z, espaciado_mm)
    guia = uc._aplicar_matriz(guia_z, uc._matriz_vtk(rotacion, inversa=True))
    valida, problemas, metricas = mg.validar_guia(guia, escaneo, eje=tuple(eje))
    metricas["cavidades_rellenas"] = cavidades
    metricas["fragmentos_descartados"] = fragmentos
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
                    eje_insercion, tipo_soporte: str = "dentosoportada",
                    espaciado_mm: float = ESPACIADO_POR_DEFECTO_MM) -> dict:
    """Guía completa: apoyo (máscara del usuario o automática) + puente + anillo con su orificio."""
    from nucleo_dental.guia import anillo_y_orificio, puente_y_columna

    pc = puente_y_columna(escaneo, dientes, implante, kit, fabricacion, tipo_soporte=tipo_soporte)
    positivos = piezas_de_apoyo(escaneo, mascara_apoyo, kit, eje_insercion) + [pc["puente"], pc["columna"]]
    _, orificio = anillo_y_orificio(implante, kit, fabricacion)
    negativos = [orificio] + ([pc["alivio"]] if pc["alivio"] is not None else [])
    resultado = ensamblar(positivos, escaneo, eje_insercion, kit.tolerancia_ajuste(fabricacion),
                          negativos=negativos, espaciado_mm=espaciado_mm)
    resultado.update({"puente_y_columna": {k: v for k, v in pc.items() if k not in ("puente", "columna", "alivio")},
                      "tipo_soporte": tipo_soporte, "fabricacion": fabricacion, "kit": kit.nombre})
    return resultado


def _limpiar_piezas(malla, espaciado: float):
    """(malla limpia, cavidades rellenadas, fragmentos descartados).

    Cavidad: superficie cerrada entera dentro del cuerpo principal (burbuja
    donde una pieza se autointersecta y la paridad del estencil invierte el
    signo); descartarla deja ese volumen macizo. Fragmento: pieza de menos de
    un vóxel de volumen, isla de muestreo de la grilla. Cualquier otra pieza
    suelta se conserva para que la validación la reporte.
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
        return malla, 0, 0
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
    fragmento = volumen[otras] < espaciado ** 3
    conservar = [principal] + otras[~adentro & ~fragmento].tolist()
    final = vtk.vtkPolyDataConnectivityFilter()
    final.SetInputData(malla)
    final.SetExtractionModeToSpecifiedRegions()
    for r in conservar:
        final.AddSpecifiedRegion(int(r))
    limpia = vtk.vtkCleanPolyData()
    limpia.SetInputConnection(final.GetOutputPort())
    limpia.Update()
    return limpia.GetOutput(), int(adentro.sum()), int((fragmento & ~adentro).sum())


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
