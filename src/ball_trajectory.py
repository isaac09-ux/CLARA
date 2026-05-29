"""
ball_trajectory.py — Reconstrucción de la trayectoria del balón para CLARA.

VballNet detecta ~1 de cada 3 frames del balón (recall ~34% en gym propio).
Entre dos contactos el balón está en vuelo libre: un arco parabólico. Los
frames perdidos NO son una adivinanza — están determinados por el vuelo que
los rodea. Aquí se reconstruyen ajustando el vuelo EN ESPACIO DE IMAGEN y
reproyectando los puntos rellenados a coordenadas de cancha.

¿Por qué en espacio de imagen y no en cancha?
  La homografía mapea el PLANO DEL PISO. Un balón en el aire se proyecta a
  donde su línea de visión pega contra el piso, así que su (court_x, court_y)
  traza una curva distorsionada, no una parábola limpia. En la imagen el
  vuelo sí es ~parabólico (gravedad constante en el eje y), así que el ajuste
  es fiel ahí; recién después se reproyecta a cancha.

Sutileza clave: el arco se REINICIA en cada contacto (saque, recepción,
colocación, ataque, bloqueo, defensa). No se puede ajustar una sola curva a
todo un rally. La segmentación en vuelos es por:
  1. Huecos temporales grandes (balón muerto / fuera de juego): nunca se
     puentean.
  2. Contactos: detectados como "donde una sola parábola deja de ajustar"
     (split-and-fit recursivo por residual), en vez de umbrales de ángulo
     frágiles que confunden el ápice de la parábola (gravedad, NO contacto)
     con un toque real.

Honestidad: los puntos interpolados van marcados con interp=True y NO cuentan
para el recall ni el score de CLARA. Esto sólo densifica la trayectoria para
zonas, tempo y visualización; las mediciones reales quedan intactas.

API:
    dense, summary = reconstruct_trajectory(ball_pts, fps, H, ppm, frame_w=W)
"""
import warnings

import numpy as np
import cv2


def _project_to_court(H, x, y, ppm):
    """img(px) -> cancha(m), igual que clara.project pero local (sin import
    circular). cv2.perspectiveTransform aplica la homografía homogénea."""
    pt = np.array([[[float(x), float(y)]]], dtype=np.float32)
    cx, cy = cv2.perspectiveTransform(pt, np.array(H, dtype=np.float64))[0][0]
    return float(cx) / ppm, float(cy) / ppm


def _dedup_by_frame(pts):
    """Un punto por frame: el de mayor confianza (o el primero si no hay conf).
    Un detector puede emitir varias cajas en el mismo frame; la trayectoria
    necesita una sola posición por instante."""
    best = {}
    for p in pts:
        f = p["frame"]
        if f not in best or p.get("conf", 0.0) > best[f].get("conf", 0.0):
            best[f] = p
    return [best[f] for f in sorted(best)]


def _split_temporal(pts, max_gap_frames):
    """Corta donde el hueco entre detecciones consecutivas excede el umbral.
    Un hueco grande significa balón fuera de juego entre rallies — puentearlo
    inventaría un vuelo que no existió."""
    if not pts:
        return []
    runs = [[pts[0]]]
    for p in pts[1:]:
        if p["frame"] - runs[-1][-1]["frame"] <= max_gap_frames:
            runs[-1].append(p)
        else:
            runs.append([p])
    return runs


def _fit_flight(pts):
    """Ajusta un vuelo en espacio de imagen por mínimos cuadrados:
        img_x = a*t + b           (velocidad horizontal ~constante -> recta)
        img_y = c*t^2 + d*t + e   (gravedad ~constante -> parábola)
    con t = frame - frame0 (centrado en el inicio para buen condicionamiento).

    Devuelve (fx, fy, rms_px, f0) donde fx(t), fy(t) son np.poly1d y rms_px es
    el error de reproyección RMS en píxeles; o None si faltan puntos.
    """
    if len(pts) < 3:
        return None
    f0 = pts[0]["frame"]
    t = np.array([p["frame"] - f0 for p in pts], dtype=np.float64)
    x = np.array([p["img_x"] for p in pts], dtype=np.float64)
    y = np.array([p["img_y"] for p in pts], dtype=np.float64)
    with warnings.catch_warnings():
        # polyfit avisa RankWarning si el ajuste queda mal condicionado. La
        # clase cambió de sitio en NumPy 2.0 (np.RankWarning -> np.exceptions),
        # así que suprimimos sin nombrarla — el bloque sólo hace dos polyfit.
        warnings.simplefilter("ignore")
        fx = np.poly1d(np.polyfit(t, x, 1))   # recta en x
        fy = np.poly1d(np.polyfit(t, y, 2))   # parábola en y
    dx = fx(t) - x
    dy = fy(t) - y
    rms = float(np.sqrt(np.mean(dx * dx + dy * dy)))
    return fx, fy, rms, f0


def _segment_flights(run, min_pts_fit, max_residual_px, depth=0, max_depth=6):
    """Split-and-fit recursivo. Ajusta una parábola al run; si el residual es
    alto, parte en el punto interior de peor residual (candidato a contacto) y
    recursa en cada mitad. Esto encuentra los contactos como "el lugar donde
    una sola parábola deja de explicar el vuelo", sin umbrales de ángulo.

    Devuelve lista de (segmento, fit) donde fit es None si el segmento es
    demasiado corto para reconstruir con confianza (passthrough sin relleno).
    """
    fit = _fit_flight(run)
    if fit is None or len(run) < min_pts_fit:
        return [(run, None)]
    fx, fy, rms, f0 = fit
    if rms <= max_residual_px or depth >= max_depth:
        return [(run, fit)]
    # Partir en el peor punto interior (no en los extremos: un contacto deja
    # al menos un par de muestras de cada lado).
    t = np.array([p["frame"] - f0 for p in run], dtype=np.float64)
    x = np.array([p["img_x"] for p in run], dtype=np.float64)
    y = np.array([p["img_y"] for p in run], dtype=np.float64)
    err = (fx(t) - x) ** 2 + (fy(t) - y) ** 2
    interior = list(range(1, len(run) - 1))
    if not interior:
        return [(run, fit)]
    split = max(interior, key=lambda i: err[i])
    # Corte sin solape: el punto de corte queda en la mitad izquierda. Así
    # ningún frame aparece en dos vuelos (sin duplicados en la salida).
    left, right = run[:split + 1], run[split + 1:]
    return (_segment_flights(left, min_pts_fit, max_residual_px,
                             depth + 1, max_depth)
            + _segment_flights(right, min_pts_fit, max_residual_px,
                               depth + 1, max_depth))


def _pt(frame, ix, iy, cxm, cym, interp, flight):
    return {"frame": int(frame),
            "img_x": round(float(ix), 1), "img_y": round(float(iy), 1),
            "court_x": round(float(cxm), 2), "court_y": round(float(cym), 2),
            "interp": bool(interp), "flight": int(flight)}


def reconstruct_trajectory(ball_pts, fps, H, ppm, frame_w=1280,
                           max_gap_s=1.0, min_pts_fit=4, residual_frac=0.02,
                           max_split_depth=6):
    """Densifica la trayectoria del balón rellenando los huecos por vuelo.

    ball_pts: detecciones REALES ya filtradas; cada una con frame, img_x,
              img_y, court_x, court_y y (opcional) conf.
    fps:      frames por segundo del video (para el umbral de hueco temporal).
    H, ppm:   homografía y píxeles-por-metro, para reproyectar los puntos
              interpolados a cancha.
    frame_w:  ancho del frame en px (escala el umbral de residual).

    Perillas:
      max_gap_s      hueco temporal que separa vuelos (balón fuera de juego).
      min_pts_fit    puntos mínimos para confiar en un ajuste (si no,
                     passthrough: se conservan los reales, no se rellena).
      residual_frac  umbral de residual como fracción del ancho del frame; por
                     encima, el segmento cruza un contacto y se parte.

    Devuelve (dense, summary):
      dense:   lista ordenada por frame de puntos
               {frame, img_x, img_y, court_x, court_y, interp, flight}.
               Los reales conservan su medición; los interpolados traen img
               ajustado, court reproyectado e interp=True.
      summary: {real, interpolated, total, flights, reconstructed_flights}.
    """
    pts = _dedup_by_frame([p for p in ball_pts
                           if "img_x" in p and "img_y" in p])
    summary = {"real": len(pts), "interpolated": 0, "total": len(pts),
               "flights": 0, "reconstructed_flights": 0}

    if len(pts) < 2:
        dense = [_pt(p["frame"], p["img_x"], p["img_y"],
                     p["court_x"], p["court_y"], False, 0) for p in pts]
        summary["flights"] = 1 if pts else 0
        return dense, summary

    max_gap_frames = max(int(round(max_gap_s * fps)), 1)
    max_residual_px = max(residual_frac * frame_w, 6.0)

    dense = []
    flight_id = 0
    interpolated = 0
    reconstructed = 0

    for run in _split_temporal(pts, max_gap_frames):
        for seg, fit in _segment_flights(run, min_pts_fit, max_residual_px,
                                         max_depth=max_split_depth):
            flight_id += 1
            real_frames = {p["frame"] for p in seg}
            # 1) Puntos reales del segmento (medición = verdad, no se tocan).
            for p in seg:
                dense.append(_pt(p["frame"], p["img_x"], p["img_y"],
                                 p["court_x"], p["court_y"], False, flight_id))
            # 2) Relleno de huecos sólo si el vuelo ajustó con confianza.
            if fit is None:
                continue
            fx, fy, _rms, f0 = fit
            reconstructed += 1
            for f in range(seg[0]["frame"], seg[-1]["frame"] + 1):
                if f in real_frames:
                    continue
                ix, iy = float(fx(f - f0)), float(fy(f - f0))
                cxm, cym = _project_to_court(H, ix, iy, ppm)
                dense.append(_pt(f, ix, iy, cxm, cym, True, flight_id))
                interpolated += 1

    dense.sort(key=lambda d: (d["frame"], d["flight"]))
    summary["interpolated"] = interpolated
    summary["total"] = len(dense)
    summary["flights"] = flight_id
    summary["reconstructed_flights"] = reconstructed
    return dense, summary
