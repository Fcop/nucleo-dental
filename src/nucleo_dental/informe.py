"""Informe del protocolo quirúrgico en PDF (R-027).

Se arma con la salida del comando `guia` (la misma que se imprime en JSON):
estado de la guía, kit, cada implante con su orificio y el largo de trabajo
de la fresa, pines, ventanas, textos, ajuste, validación y trazabilidad.

El PDF se escribe a mano con la biblioteca estándar (regla 2: en src/ solo
numpy y vtk): texto en Helvetica con codificación WinAnsi (tildes y eñes) e
imágenes RGB comprimidas con zlib. La secuencia de fresas no se inventa: se
remite al manual del kit.
"""

from __future__ import annotations

import zlib

import numpy as np

ANCHO_PT, ALTO_PT = 595.0, 842.0           # A4
MARGEN_PT = 50.0
_ADVERTENCIA = ("Software en desarrollo, sin validación clínica: este informe no debe usarse para operar "
                "pacientes. Verifique cada valor contra la planificación y el manual del kit.")


def _texto_pdf(texto: str) -> bytes:
    """Cadena literal de PDF en WinAnsi (cp1252), con los caracteres especiales escapados."""
    crudo = texto.encode("cp1252", errors="replace")
    return b"(" + crudo.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)") + b")"


def _cortar(texto: str, ancho: int) -> list:
    """Corta el texto en renglones de a lo más `ancho` caracteres, por palabras."""
    renglones, actual = [], ""
    for palabra in texto.split(" "):
        if actual and len(actual) + 1 + len(palabra) > ancho:
            renglones.append(actual)
            actual = palabra
        else:
            actual = f"{actual} {palabra}" if actual else palabra
    return renglones + [actual]


def _mm(valor, decimales: int = 2) -> str:
    return "—" if valor is None else f"{float(valor):.{decimales}f}".replace(".", ",")


def _punto(p) -> str:
    return "(" + "; ".join(_mm(c) for c in p) + ")"


def lineas_del_informe(salida: dict) -> list:
    """Renglones del informe: (estilo, texto) con estilo "titulo", "seccion", "alerta" o "normal"."""
    g, geo, par = salida["guia"], salida["geometria"], salida["parametros"]
    traz = salida["trazabilidad"]
    lineas = [("titulo", "Protocolo quirúrgico — guía de implantes"),
              ("alerta", _ADVERTENCIA),
              ("normal", f"Fecha (UTC): {traz['fecha_hora_utc']}   Motor: {traz['version_motor']}   "
                         f"Commit: {traz['commit_git'] or '—'}" + ("  (con cambios sin commit)" if traz["cambios_sin_commit"] else "")),
              ("seccion", "Estado de la guía")]
    if g["valida"]:
        lineas.append(("normal", "VÁLIDA: una sola pieza cerrada, se inserta sin colisión y no invade el escaneo."))
    else:
        lineas.append(("alerta", "NO VÁLIDA — NO IMPRIMIR."))
        lineas += [("alerta", f"• {p}") for p in g["problemas"]]
    lineas += [("normal", f"Aviso: {a}") for a in g.get("avisos", [])]

    lineas += [("seccion", "Guía"),
               ("normal", f"Kit: {par['kit']}   Fabricación: {par['fabricacion']}   Soporte: {par['tipo_soporte']}"),
               ("normal", f"Tolerancia de ajuste guía–escaneo: {_mm(geo['tolerancia_ajuste_mm'])} mm   "
                          f"Eje de inserción (LPS): {_punto(geo['eje_insercion'])}"),
               ("normal", "Valores provisionales del kit: " + (", ".join(par.get("provisionales", [])) or "ninguno"))]

    for i, a in enumerate(geo["anillos"], start=1):
        gg = a["geometria"]
        lineas += [("seccion", f"Implante {i}" + (f" ({a['implante_stl']})" if a.get("implante_stl") else "")),
                   ("normal", f"Plataforma (LPS): {_punto(gg['plataforma'])}   Punta de la fresa: {_punto(gg['punta_fresa'])}"),
                   ("normal", f"Orificio: Ø {_mm(gg['diametro_orificio_mm'])} mm (guía de la fresa Ø {_mm(gg['diametro_guia_mm'])})"
                              f"   Anillo: Ø {_mm(gg['diametro_externo_anillo_mm'])} mm"),
                   ("normal", f"Largo de trabajo de la fresa (de la cara superior a la punta): {_mm(gg['largo_trabajo_fresa_mm'])} mm   "
                              f"Contacto efectivo guía–fresa: {_mm(a['contacto_efectivo_mm'])} mm"),
                   ("normal", f"Ángulo con el eje de inserción: {_mm(a.get('angulo_eje_insercion_grados'), 1)}°   "
                              "Secuencia de fresas: según el manual del kit."),
                   ("normal", "Distancias al canal, a los dientes y al hueso: ver el informe de `nucleo-dental medir`.")]

    entre = salida.get("entre_implantes")
    if entre:
        lineas.append(("seccion", f"Entre implantes: {entre['semaforo'].upper()}"))
        lineas += [("normal", f"{par_}: {_mm(d['distancia_mm'])} mm (margen {_mm(d['margen_mm'])}), {d['semaforo']}")
                   for par_, d in entre["pares"].items()]
    for p in geo.get("paredes_entre_orificios", []):
        lineas.append(("normal", f"Pared entre {p['entre'][0]} y {p['entre'][1]}: {_mm(p['pared_mm'])} mm"
                                 + (" (SOLAPADOS)" if p["solapados"] else "")))

    for i, p in enumerate(geo.get("pines", []), start=1):
        s = p["seguridad"]
        lineas += [("seccion", f"Pin {i}: {s['semaforo'].upper()}"),
                   ("normal", f"Ø {_mm(p['diametro_mm'])} × {_mm(p['largo_mm'])} mm   Profundidad en el hueso: "
                              f"{_mm(p['profundidad_en_hueso_mm'])} mm   Agujero Ø {_mm(p['diametro_agujero_mm'])}   "
                              f"Refuerzo Ø {_mm(p['diametro_refuerzo_mm'])}")]
        lineas += [("normal", f"  {nombre}: {_mm(r.get('distancia_mm'))} mm, {r['semaforo']}")
                   for nombre, r in s["estructuras"].items()]

    if geo.get("ventanas"):
        lineas.append(("seccion", "Ventanas de inspección"))
        lineas += [("normal", f"Ventana {v['ventana']}: {' × '.join(_mm(t, 1) for t in v['tamano'])} mm"
                              + ("" if v["corta_la_guia"] else " (NO TOCA LA GUÍA)")) for v in geo["ventanas"]]
    for i, t in enumerate(geo.get("textos", []), start=1):
        lineas.append(("normal", f"Texto {i} ({t['modo']}): letras de {_mm(t['alto_letra_mm'], 1)} mm, trazo "
                                 f"{_mm(t['trazo_mm'])} mm, profundidad {_mm(t['profundidad_mm'])} mm"))

    aj = salida.get("ajuste")
    if aj:
        lineas += [("seccion", "Ajuste guía–escaneo (cara de asiento)"),
                   ("normal", f"Ideal {_mm(aj['pct_ideal'], 1)} %   Apretado {_mm(aj['pct_apretado'], 1)} %   "
                              f"Holgado {_mm(aj['pct_holgado'], 1)} %   Interferencia {_mm(aj['pct_interferencia'], 1)} %"),
                   ("normal", f"Separación mediana {_mm(aj['separacion_mediana_mm'])} mm (mín. {_mm(aj['separacion_min_mm'])}); "
                              f"desvío aceptable ±{_mm(aj['desvio_aceptable_mm'])} mm")]

    m = g["metricas"]
    lineas += [("seccion", "Validación"),
               ("normal", f"Piezas: {m['piezas']}   Aristas defectuosas: {m['aristas_defectuosas']}   "
                          f"Volumen: {_mm(m['volumen_mm3'], 0)} mm³   Colisión al insertar: "
                          f"{_mm(m['insercion']['colision_mm3'])} mm³"),
               ("normal", f"Cavidades rellenadas: {m.get('cavidades_rellenas', 0)}   Astillas descartadas: "
                          f"{m.get('astillas_descartadas', 0)} ({_mm(m.get('volumen_astillas_mm3', 0), 3)} mm³)"),
               ("seccion", "Trazabilidad"),
               ("normal", f"STL de la guía: {g['ruta']}"),
               ("normal", f"SHA-256: {g['sha256']}")]
    lineas += [("normal", f"{a['ruta']}  SHA-256 {a['sha256'][:16]}…") for a in traz["archivos_entrada"]]
    return lineas


def escribir_pdf(lineas: list, ruta, imagenes=()) -> None:
    """PDF A4 con los renglones (paginados) y, al final, las imágenes RGB (arreglos alto × ancho × 3, uint8)."""
    estilos = {"titulo": (16, 1, 22), "seccion": (11, 1, 18), "alerta": (9, 1, 13), "normal": (9, 0, 12.5)}
    paginas, actual, y = [], [], ALTO_PT - MARGEN_PT
    for estilo, texto in lineas:
        tamano, negrita, salto = estilos[estilo]
        for renglon in _cortar(texto, int((ANCHO_PT - 2 * MARGEN_PT) / (tamano * 0.5))):
            if y - salto < MARGEN_PT:
                paginas.append(actual)
                actual, y = [], ALTO_PT - MARGEN_PT
            y -= salto
            color = b"0.75 0 0 rg " if estilo == "alerta" else b"0 0 0 rg "
            actual.append(color + b"BT /F%d %d Tf %.1f %.1f Td " % (2 if negrita else 1, tamano, MARGEN_PT, y)
                          + _texto_pdf(renglon) + b" Tj ET")
    paginas.append(actual)

    objetos = []                                              # cuerpo de cada objeto, numerados desde 1

    def agregar(cuerpo: bytes) -> int:
        objetos.append(cuerpo)
        return len(objetos)

    agregar(b"")                                             # 1: catálogo (se completa al final)
    agregar(b"")                                             # 2: árbol de páginas
    f1 = agregar(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
    f2 = agregar(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>")
    recursos = b"/Font << /F1 %d 0 R /F2 %d 0 R >>" % (f1, f2)
    hojas = []
    for contenido in paginas:
        datos = b"\n".join(contenido)
        flujo = agregar(b"<< /Length %d >>\nstream\n" % len(datos) + datos + b"\nendstream")
        hojas.append(agregar(b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %d %d] /Resources << %s >> /Contents %d 0 R >>"
                             % (ANCHO_PT, ALTO_PT, recursos, flujo)))
    alto_util = (ALTO_PT - 2 * MARGEN_PT - MARGEN_PT / 2) / 2           # dos imágenes por página, una sobre otra
    for inicio in range(0, len(imagenes), 2):
        dibujos, xobjetos = [], []
        for k, imagen in enumerate(imagenes[inicio:inicio + 2]):
            rgb = np.ascontiguousarray(np.asarray(imagen, dtype=np.uint8)[::-1])     # VTK da las filas de abajo hacia arriba
            alto, ancho = rgb.shape[:2]
            comprimido = zlib.compress(rgb.tobytes(), 6)
            xobjetos.append(agregar(b"<< /Type /XObject /Subtype /Image /Width %d /Height %d /ColorSpace /DeviceRGB "
                                    b"/BitsPerComponent 8 /Filter /FlateDecode /Length %d >>\nstream\n"
                                    % (ancho, alto, len(comprimido)) + comprimido + b"\nendstream"))
            escala = min((ANCHO_PT - 2 * MARGEN_PT) / ancho, alto_util / alto)
            arriba = ALTO_PT - MARGEN_PT - k * (alto_util + MARGEN_PT / 2)
            dibujos.append(b"q %.2f 0 0 %.2f %.1f %.1f cm /Im%d Do Q" % (ancho * escala, alto * escala, MARGEN_PT,
                                                                       arriba - alto * escala, k + 1))
        datos = b"\n".join(dibujos)
        flujo = agregar(b"<< /Length %d >>\nstream\n" % len(datos) + datos + b"\nendstream")
        nombres = b" ".join(b"/Im%d %d 0 R" % (k + 1, x) for k, x in enumerate(xobjetos))
        hojas.append(agregar(b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %d %d] /Resources << %s /XObject << %s >> >> "
                             b"/Contents %d 0 R >>" % (ANCHO_PT, ALTO_PT, recursos, nombres, flujo)))
    objetos[0] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objetos[1] = b"<< /Type /Pages /Kids [%s] /Count %d >>" % (b" ".join(b"%d 0 R" % h for h in hojas), len(hojas))

    salida = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    posiciones = []
    for numero, cuerpo in enumerate(objetos, start=1):
        posiciones.append(len(salida))
        salida += b"%d 0 obj\n" % numero + cuerpo + b"\nendobj\n"
    inicio_xref = len(salida)
    salida += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objetos) + 1)
    salida += b"".join(b"%010d 00000 n \n" % p for p in posiciones)
    salida += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objetos) + 1, inicio_xref)
    with open(ruta, "wb") as archivo:
        archivo.write(bytes(salida))


def vistas_de_la_guia(guia, escaneo, eje_insercion, tamano=(900, 600)) -> list:
    """Imágenes RGB de la guía sobre el escaneo: oclusal y lateral. Lista vacía si no hay render disponible."""
    import vtk
    from vtk.util import numpy_support

    try:
        imagenes = []
        eje = np.asarray(eje_insercion, dtype=float) / np.linalg.norm(eje_insercion)
        lateral = np.cross(eje, [1.0, 0.0, 0.0])
        if np.linalg.norm(lateral) < 1e-6:
            lateral = np.cross(eje, [0.0, 1.0, 0.0])
        lateral /= np.linalg.norm(lateral)
        centro = np.array(guia.GetCenter())
        radio = max(np.ptp(np.array(guia.GetBounds()).reshape(3, 2), axis=1)) * 1.6
        for direccion, arriba in ((eje, lateral), (lateral + 0.4 * eje, eje)):
            ventana = vtk.vtkRenderWindow()
            ventana.SetOffScreenRendering(1)
            ventana.SetSize(*tamano)
            render = vtk.vtkRenderer()
            render.SetBackground(1, 1, 1)
            for malla, color in ((escaneo, (0.88, 0.86, 0.72)), (guia, (0.95, 0.55, 0.25))):
                mapeo = vtk.vtkPolyDataMapper()
                mapeo.SetInputData(malla)
                actor = vtk.vtkActor()
                actor.SetMapper(mapeo)
                actor.GetProperty().SetColor(*color)
                render.AddActor(actor)
            ventana.AddRenderer(render)
            camara = render.GetActiveCamera()
            camara.SetFocalPoint(*centro)
            camara.SetPosition(*(centro + radio * direccion / np.linalg.norm(direccion)))
            camara.SetViewUp(*arriba)
            render.ResetCamera(guia.GetBounds())                 # encuadra la guía completa
            camara.Zoom(1.2)
            render.ResetCameraClippingRange()
            ventana.Render()
            captura = vtk.vtkWindowToImageFilter()
            captura.SetInput(ventana)
            captura.Update()
            datos = captura.GetOutput()
            ancho, alto, _ = datos.GetDimensions()
            rgb = numpy_support.vtk_to_numpy(datos.GetPointData().GetScalars()).reshape(alto, ancho, -1)[:, :, :3]
            imagenes.append(rgb)
            ventana.Finalize()
        return imagenes
    except Exception:                                         # sin OpenGL (p. ej. servidor sin pantalla): sin imágenes
        return []
