"""
CLARA v0.8 — Multimodal scouting + auto-calibracion
Tentáculo de visión por computadora de LUCIA · Las Chispas.

Fusión v0.8 (ROI de jugadoras + parche del balón + reconstrucción):
  - ROI de jugadoras: el punto de pie se valida en PÍXELES contra el polígono
    de cancha proyectado por la homografía (court_roi_polygon). Descarta
    público/banca que is_in_court() deja pasar porque la homografía extiende
    el plano del piso.
  - Parche del balón: el balón vuela ENCIMA del piso, así que NO se filtra por
    su proyección a cancha (un balón aéreo "aterriza" fuera de línea y el
    filtro viejo mataba ~96% de lo detectado). Se filtra en PÍXELES contra la
    cancha-en-imagen estirada hacia arriba (ball_valid_region). Cada detección
    conserva img_x/img_y; filter_ball_tracks agrupa por píxeles.
  - UNIFICADO: un solo point_in_polygon (cv2) sirve a jugadoras y balón.
    Antes había point_in_roi (cv2) y point_in_polygon (ray-casting) duplicados.
  - ball_detection_rate con denominador honesto (frames realmente evaluados).
  - JSON: ball_track (detecciones reales) + ball_track_reconstructed
    (trayectoria densa por vuelo; ver ball_trajectory.py). Reconstrucción NO
    infla el score — recall y calidad se miden sobre detecciones reales.

Cambios v0.5.1 → v0.6:
  - INTEGRACIÓN: VballNet (TrackNetV4) como detector de balón opcional
    Motion-based, ~70% recall sin entrenamiento en gyms nuevos
  - INTEGRACIÓN: rtmlib (RTMPose) para keypoints de jugadoras
    17 puntos COCO, análisis biomecánico básico
  - Mantiene compatibilidad con clara_balon_v1.pt (YOLO custom)

Selectores de detector de balón:
  --ball-detector yolo       (default — usa YOLO11 base, clase 32, o --ball-model)
  --ball-detector vballnet   (usa modelo VballNet ONNX, requiere --vballnet-model)

Selectores de pose:
  --pose none      (default — solo bounding boxes)
  --pose rtmlib    (extrae keypoints por jugadora detectada)
"""
import cv2
import numpy as np
import json
import argparse
import sys
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO

# Reconstrucción de trayectoria del balón (rellena los huecos de VballNet por
# vuelo, en espacio de imagen). Módulo propio, sólo depende de numpy.
from ball_trajectory import reconstruct_trajectory

# En consolas Windows (cp1252) los caracteres de caja (┌─│), ✓, ⏳, ★ del banner
# revientan stdout con UnicodeEncodeError. Forzar utf-8 hace que CLARA corra
# igual en una PowerShell normal, no solo redirigida a archivo.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ============================================================
#  CONSTANTES
# ============================================================
PAL_BONE = (208, 224, 232)
PAL_OXBLOOD = (40, 40, 165)
PAL_OK = (122, 156, 90)
PAL_DARK = (84, 79, 68)
PAL_WARN = (60, 156, 220)
PAL_BALL = (60, 220, 230)
PAL_REJECT = (60, 60, 165)


# ── Región de validez del balón en píxeles ──────────────────
# El balón vuela ENCIMA del plano de cancha. Proyectarlo por la homografía
# del piso lo manda a coordenadas de cancha falsas (un balón aéreo "aterriza"
# fuera de las líneas). Por eso el balón NO se filtra con is_in_court sobre
# court_x/court_y — se filtra en PÍXELES contra el polígono de la cancha
# (pixel_corners de MIRA) estirado hacia arriba para darle aire al balón.
# Estas 3 constantes son las únicas perillas de criterio; se calibran mirando
# los resultados sobre tu propio video.
BALL_HEADROOM_FRAC = 0.50      # aire SOBRE la cancha que cuenta como válido,
                               # fracción del alto del frame. Generoso a
                               # propósito: un FP de más lo limpia
                               # filter_ball_tracks; matar un balón aéreo
                               # real reproduce justo el bug que arreglamos.
BALL_SIDE_MARGIN_FRAC = 0.03   # margen lateral, fracción del ancho del frame.
                               # Apretado: un balón muy fuera de banda es
                               # del público / otra cancha.
BALL_MAX_JUMP_PX_PER_FRAME = 0.09  # salto máximo del balón entre frames,
                                   # fracción del ancho. Escala con el gap
                                   # temporal dentro de filter_ball_tracks.


# ============================================================
#  GEOMETRÍA
# ============================================================
def project(H, x, y):
    pt = np.array([[[x, y]]], dtype=np.float32)
    return cv2.perspectiveTransform(pt, np.array(H))[0][0]


def zone_for_court_pos(x_m, y_m, court_w, court_h, half_court=False, margin=0.0):
    if (x_m < -margin or x_m > court_w + margin or
        y_m < -margin or y_m > court_h + margin):
        return None
    x_m = max(0, min(court_w, x_m))
    y_m = max(0, min(court_h, y_m))

    if half_court:
        front = y_m < court_h / 3
        col = "L" if x_m < court_w / 3 else ("M" if x_m < 2 * court_w / 3 else "R")
        zmap = {("L", True): 4, ("M", True): 3, ("R", True): 2,
                ("L", False): 5, ("M", False): 6, ("R", False): 1}
        return f"A{zmap[(col, front)]}"

    if y_m <= court_h / 2:
        side, local_y = "A", y_m
    else:
        side, local_y = "B", court_h - y_m
    front = local_y > court_h / 4
    col = "L" if x_m < court_w / 3 else ("M" if x_m < 2 * court_w / 3 else "R")
    if side == "A":
        zmap = {("L", True): 4, ("M", True): 3, ("R", True): 2,
                ("L", False): 5, ("M", False): 6, ("R", False): 1}
    else:
        zmap = {("R", True): 4, ("M", True): 3, ("L", True): 2,
                ("R", False): 5, ("M", False): 6, ("L", False): 1}
    return f"{side}{zmap[(col, front)]}"


def is_in_court(cx, cy, cw, ch, margin=0.5):
    return (cx is not None and cy is not None and
            -margin <= cx <= cw + margin and -margin <= cy <= ch + margin)


def court_roi_polygon(H, court_w, court_h, ppm, margin_m=2.0):
    """Polígono de la zona de juego en PÍXELES DE IMAGEN (no en cancha).

    is_in_court() filtra en coords proyectadas, pero el público detrás de la
    línea de fondo y el equipo del lado lejano (en media cancha) se PROYECTAN
    dentro del rango numérico de la cancha — la homografía extiende el plano
    del piso, así que gente parada en ese plano cae en coords [0..court].
    Por eso is_in_court() no los descarta.

    Este polígono trabaja en el espacio donde esa gente SÍ está separada: la
    imagen. Tomamos las 4 esquinas de la cancha (expandidas margin_m metros
    para no recortar jugadoras que persiguen balones fuera de línea o sacan
    desde atrás del fondo), las llevamos de metros → topdown(px) → imagen con
    H^-1. El margen se expande EN METROS (uniforme en cancha) y luego se
    proyecta, así respeta la perspectiva: cerca de cámara el margen se ve
    grande, lejos se ve chico — que es justo lo que se quiere.

    Devuelve un np.array (4,2) int32 listo para cv2.pointPolygonTest /
    cv2.polylines, o None si H no es invertible.

    Nota media cancha: un margen grande sobre la red (borde court_h) puede
    readmitir al equipo del lado lejano. Verifica el polígono en
    diagnostic.png y baja --roi-margin si se cuela gente del fondo.
    """
    try:
        Hinv = np.linalg.inv(np.array(H, dtype=np.float64))
    except np.linalg.LinAlgError:
        return None
    m = margin_m
    # esquinas de cancha expandidas, en metros
    corners_m = np.array([
        [-m,            -m],
        [court_w + m,   -m],
        [court_w + m,    court_h + m],
        [-m,             court_h + m],
    ], dtype=np.float64)
    # metros -> topdown(px) -> imagen(px)
    topdown = (corners_m * ppm).reshape(-1, 1, 2).astype(np.float32)
    img_pts = cv2.perspectiveTransform(topdown, Hinv.astype(np.float32))
    return img_pts.reshape(-1, 2).astype(np.int32)


def point_in_polygon(px, py, polygon):
    """True si (px,py) cae dentro del polígono (o si no hay polígono).

    UNIFICADO: única prueba punto-en-polígono de CLARA. La usan tanto el ROI
    de jugadoras (court_roi_polygon — trapecio en perspectiva) como el filtro
    del balón (ball_valid_region — bbox recto con headroom). Antes había dos
    implementaciones (point_in_roi con cv2 y point_in_polygon con ray-casting);
    esta las reemplaza. Acepta el polígono como np.array (N,2) o lista de
    (x,y), en float o int. polygon None => sin filtro => todo pasa.
    """
    if polygon is None:
        return True
    poly = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(poly, (float(px), float(py)), False) >= 0


def ball_valid_region(cal, frame_h, frame_w):
    """Polígono (en píxeles) donde un balón es válido: la cancha estirada
    hacia arriba para darle aire al balón aéreo.

    Variante del mismo concepto que court_roi_polygon (el footprint de la
    cancha en la imagen), pero con geometría distinta a propósito:
      - court_roi_polygon proyecta la cancha+margen por la homografía → trapecio
        en perspectiva. Correcto para el PIE de una jugadora (está en el piso).
      - aquí se usa el BOUNDING BOX recto de pixel_corners, expandido mucho
        hacia ARRIBA (headroom) y poco a los lados/abajo. Un bbox recto y no
        las bandas prolongadas porque prolongar las bandas convergentes hasta
        el techo las cruza en el punto de fuga y el polígono se autointersecta.
        El box recto siempre contiene la cancha y no tiene esa patología.

    Parte de pixel_corners de MIRA (las 4 esquinas de la cancha en píxeles).
    Devuelve 4 vértices [(x,y)...] o None si no hay pixel_corners utilizables
    (calibración vieja) — el llamador cae entonces al filtro por proyección.
    """
    pc = cal.get("pixel_corners")
    if not pc or len(pc) != 4:
        return None
    try:
        corners = [(float(x), float(y)) for x, y in pc]
    except (TypeError, ValueError):
        return None
    if not all(np.isfinite(v) for c in corners for v in c):
        return None

    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    side = BALL_SIDE_MARGIN_FRAC * frame_w
    bottom_m = BALL_SIDE_MARGIN_FRAC * frame_h     # tolerancia bajo la red
    headroom = BALL_HEADROOM_FRAC * frame_h        # aire sobre la cancha

    left = min(xs) - side
    right = max(xs) + side
    y_bottom = max(ys) + bottom_m
    y_top = max(0.0, min(ys) - headroom)

    return [
        (left, y_bottom),    # inferior izq (bajo la línea cercana)
        (right, y_bottom),   # inferior der
        (right, y_top),      # superior der (techo)
        (left, y_top),       # superior izq
    ]


def filter_ball_tracks(detections, stride, frame_w, max_gap=None,
                       max_jump_px_per_frame=None, min_track_len=2):
    """Agrupa detecciones de balón en tracks por cercanía temporal y espacial,
    descarta tracks cortos (FPs aislados: luces, banderines, balones del público).

    Una detección sólo sobrevive si tiene al menos otra dentro de max_gap
    frames Y de un salto en PÍXELES coherente con la velocidad del balón.

    La cercanía espacial se mide en píxeles, NO en metros de cancha: las
    coords de cancha de un balón aéreo son basura (la homografía mapea el
    piso). En píxeles el balón traza un arco suave cuadro a cuadro; un foco
    parpadeando no. El umbral escala con el gap temporal — un balón rápido
    puede saltar mucho entre dos detecciones separadas.
    """
    if len(detections) < min_track_len:
        return []
    if max_gap is None:
        # Un par de strides cubre saltos breves sin pegar tracks no relacionados.
        max_gap = max(stride * 3, 6)
    if max_jump_px_per_frame is None:
        max_jump_px_per_frame = BALL_MAX_JUMP_PX_PER_FRAME * frame_w

    dets = sorted(detections, key=lambda d: d["frame"])
    tracks = []
    current = [dets[0]]
    for d in dets[1:]:
        prev = current[-1]
        gap = d["frame"] - prev["frame"]
        dx = d["img_x"] - prev["img_x"]
        dy = d["img_y"] - prev["img_y"]
        max_jump = max_jump_px_per_frame * max(gap, 1)
        if gap <= max_gap and (dx * dx + dy * dy) ** 0.5 <= max_jump:
            current.append(d)
        else:
            tracks.append(current)
            current = [d]
    tracks.append(current)
    return [d for t in tracks if len(t) >= min_track_len for d in t]


def classify_detection(bbox, frame_h, frame_w, court_horizon_y=None,
                       max_height_ratio=0.55, max_width_ratio=0.40,
                       is_ball=False, edge_reject_height_ratio=0.35):
    """Filtro pre-homografía: descarta cajas obviamente inválidas por tamaño
    o por tocar el borde inferior siendo además grandes (público cercano).

    court_horizon_y queda aceptado por compatibilidad con calibraciones
    previas pero ya no se usa: la banda fija en píxeles (±5%/+10% del alto)
    asumía cámara casi frontal, donde la cancha ocupa una franja delgada.
    En ángulos elevados u oblicuos la cancha cubre cientos de píxeles
    verticales y la banda rechaza jugadoras reales (un día rechazó 1911
    detecciones, dejando 1 sola jugadora). El filtro correcto es
    is_in_court() después de proyectar a coords de cancha, que ya se aplica
    al armar `filtered` y `ball_clean`.

    fg_at_edge (corregido v0.6.2): antes rechazaba CUALQUIER caja cuyo
    borde inferior tocara el frame. En video de baja resolución (848x478)
    las jugadoras llenan buena parte del alto y muchas tocan el borde en
    algún frame — el filtro mataba miles de detecciones válidas (3699 en
    un caso real, dejando 0 tracks). Ahora solo rechaza si la caja toca el
    borde Y ADEMÁS es grande (>35% del alto), que es el patrón de un
    espectador/entrenador cortado en primer plano. Una jugadora lejana
    cuyos pies se cortan un poco en el borde es una detección válida y se
    conserva; si quedara fuera de cancha, is_in_court() la filtra después.
    """
    x1, y1, x2, y2 = bbox
    bbox_h = y2 - y1
    bbox_w = x2 - x1
    if not is_ball:
        if bbox_h / frame_h > max_height_ratio:
            return "fg_too_large", f"altura {bbox_h/frame_h:.0%}"
        if bbox_w / frame_w > max_width_ratio:
            return "fg_too_large", f"ancho {bbox_w/frame_w:.0%}"
    if y2 >= frame_h - 5:
        # Solo rechazar si además es grande (primer plano cortado).
        # Cajas normales/chicas en el borde = jugadora lejana válida.
        if not is_ball and bbox_h / frame_h > edge_reject_height_ratio:
            return "fg_at_edge", f"borde + grande ({bbox_h/frame_h:.0%})"
    return "ok", None


# ============================================================
#  DETECCIÓN DE BALÓN VIA VBALLNET
# ============================================================
# ── Cache de la pasada de VballNet ──────────────────────────────
# La pasada de VballNet (stride 1 sobre todo el video) es la parte lenta.
# Se cachea su salida CRUDA (coords en píxeles) a disco junto al video. En
# re-corridas con el mismo video/modelo/stride se carga del cache y se salta
# la pasada. El cache NO depende de la calibración: la proyección a cancha y
# el filtro de polígono se rehacen al cargar, así que recalibrar o mover
# BALL_HEADROOM_FRAC no lo invalida. Cualquier fallo de cache cae a recomputar.

def _vballnet_cache_key(video_path, model_path, threshold, stride):
    p = Path(video_path)
    try:
        st = p.stat()
        sig = {"video": p.name, "size": st.st_size, "mtime": int(st.st_mtime)}
    except OSError:
        sig = {"video": p.name, "size": -1, "mtime": -1}
    sig["model"] = Path(model_path).name
    sig["threshold"] = round(float(threshold), 4)
    sig["stride"] = int(stride)
    return sig


def _vballnet_detect_cached(video_path, model_path, threshold, stride, verbose,
                            use_cache=True, cache_path=None):
    """detect_balls envuelto en cache de disco. Devuelve la lista cruda de
    detecciones (frame, x, y, radius, confidence)."""
    from ball_vballnet import detect_balls

    if cache_path is None:
        cache_path = Path(video_path).with_suffix(".vballnet_cache.json")
    cache_path = Path(cache_path)
    key = _vballnet_cache_key(video_path, model_path, threshold, stride)

    if use_cache and cache_path.exists():
        try:
            with open(cache_path) as f:
                cached = json.load(f)
            if cached.get("key") == key:
                dets = cached["detections"]
                print(f"[VballNet] cache HIT: {len(dets)} detecciones desde "
                      f"{cache_path.name} — saltando la pasada")
                return dets
            print("[VballNet] cache existe pero cambió video/modelo/stride — "
                  "recomputando")
        except (json.JSONDecodeError, KeyError, OSError) as e:
            print(f"[VballNet] cache ilegible ({e}) — recomputando")

    raw = detect_balls(video_path, model_path,
                       threshold=threshold, stride=stride, verbose=verbose)

    if use_cache:
        try:
            tmp = cache_path.with_name(cache_path.name + ".tmp")
            with open(tmp, "w") as f:
                json.dump({"key": key, "detections": raw}, f)
            tmp.replace(cache_path)   # escritura atómica: un crash no corrompe
            print(f"[VballNet] cache guardado → {cache_path.name} "
                  f"({len(raw)} detecciones)")
        except OSError as e:
            print(f"[VballNet] no se pudo guardar cache ({e}) — continúo")

    return raw


def detect_balls_vballnet(video_path, model_path, H, ppm,
                          court_w, court_h,
                          frame_h, frame_w, court_horizon_y=None,
                          max_h_ratio=0.55, max_w_ratio=0.40,
                          rejected_counts=None,
                          threshold=0.5, stride=1, verbose=True,
                          use_cache=True, cache_path=None):
    """VballNet adapter. Returns CLARA-compatible ball detection list.

    Applies the same foreground filter (bottom-edge guard) used for YOLO
    ball detections so detecciones de tribuna/banca se descartan.

    stride > 1 alimenta el buffer de 9 frames sólo cada N frames del video.
    Reduce inferencias y RAM ~stride×; el balón salta más entre frames de
    la secuencia, así que el recall baja proporcionalmente.
    """
    raw = _vballnet_detect_cached(video_path, model_path, threshold, stride,
                                  verbose, use_cache=use_cache,
                                  cache_path=cache_path)
    out = []
    for d in raw:
        # bbox sintético desde (x, y, radius) para reusar classify_detection.
        # Sólo importa el borde inferior (y2) para el filtro de balón —
        # is_ball=True salta los checks de altura/ancho relativos al frame.
        r = max(d.get("radius", 5.0), 5.0)
        bbox = [d["x"] - r, d["y"] - r, d["x"] + r, d["y"] + r]
        status, _ = classify_detection(
            bbox, frame_h, frame_w, court_horizon_y,
            max_h_ratio, max_w_ratio, is_ball=True,
        )
        if status != "ok":
            if rejected_counts is not None:
                rejected_counts[f"ball_{status}"] += 1
            continue
        cx_m, cy_m = project(H, d["x"], d["y"])
        out.append({
            "frame": d["frame"],
            # Posición en PÍXELES — la verdad principal del balón (la
            # proyección a cancha sólo es fiable cerca del piso).
            "img_x": float(d["x"]),
            "img_y": float(d["y"]),
            "court_x": float(cx_m) / ppm,
            "court_y": float(cy_m) / ppm,
            "conf": d["confidence"],
        })
    return out


# ============================================================
#  TRACK STITCHING
# ============================================================
def stitch_tracks(raw_tracks, fps, stride, max_gap_s=1.5, max_jump_m=2.5):
    """
    Cose tracks fragmentados. ByteTrack parte una jugadora en varios IDs
    cuando se ocluye; esta funcion une los pedazos.

    Dos tracks A y B se unen si:
      - B empieza DESPUES de que A termina (sin solaparse)
      - el hueco temporal es <= max_gap_s segundos
      - la distancia entre el ultimo punto de A y el primero de B es
        <= max_jump_m metros (la jugadora no se teletransporta)

    Args:
        raw_tracks: dict {tid: [samples]} — cada sample tiene 'frame',
                    'court_x', 'court_y'
        fps: frames por segundo del video
        stride: cada cuantos frames se muestreo
        max_gap_s: hueco temporal maximo para considerar union
        max_jump_m: salto espacial maximo permitido

    Returns:
        dict {tid: [samples]} con los tracks cosidos.
    """
    if len(raw_tracks) < 2:
        return raw_tracks

    # Ordenar samples de cada track por frame
    tracks = {}
    for tid, samples in raw_tracks.items():
        tracks[tid] = sorted(samples, key=lambda s: s["frame"])

    max_gap_frames = max_gap_s * fps

    # Repetir hasta que no haya mas uniones (un track puede coser
    # varios pedazos en cadena)
    merged_any = True
    while merged_any:
        merged_any = False
        tids = list(tracks.keys())

        for i in range(len(tids)):
            tid_a = tids[i]
            if tid_a not in tracks:
                continue
            a = tracks[tid_a]
            a_end_frame = a[-1]["frame"]
            a_end_pos = (a[-1]["court_x"], a[-1]["court_y"])

            best_match = None
            best_gap = None

            for j in range(len(tids)):
                tid_b = tids[j]
                if tid_b == tid_a or tid_b not in tracks:
                    continue
                b = tracks[tid_b]
                b_start_frame = b[0]["frame"]
                # B debe empezar despues de que A termina
                gap = b_start_frame - a_end_frame
                if gap <= 0 or gap > max_gap_frames:
                    continue
                # distancia espacial entre fin de A e inicio de B
                b_start_pos = (b[0]["court_x"], b[0]["court_y"])
                dist = np.hypot(a_end_pos[0] - b_start_pos[0],
                                a_end_pos[1] - b_start_pos[1])
                if dist > max_jump_m:
                    continue
                # de los candidatos, preferir el de menor hueco temporal
                if best_gap is None or gap < best_gap:
                    best_gap = gap
                    best_match = tid_b

            if best_match is not None:
                # coser B dentro de A
                tracks[tid_a] = a + tracks[best_match]
                del tracks[best_match]
                merged_any = True

    return tracks


# ============================================================
#  SEGMENTACION DE RALLIES (heuristica sobre el balon)
# ============================================================
def segment_rallies(ball_pts, fps, max_gap_s=2.0, min_rally_s=1.0,
                    min_points=3, court_h=18.0, half_court=False):
    """Agrupa las detecciones de balon en rallies (periodos de juego continuo).

    Idea: entre rallies el balon esta quieto/sostenido y VballNet (motion-based)
    no lo detecta -> hueco temporal grande. Dentro del rally el balon se mueve y
    se detecta casi continuo (huecos chicos por misses). Un hueco > max_gap_s
    corta el rally. Sin entrenamiento, solo geometria + tiempo.

    ball_pts: lista de {'frame','court_x','court_y'} (cualquier orden).
    Devuelve (rallies, summary). Cada rally trae start/end frame, duracion,
    n_points, posicion de inicio y el lado que saca (aprox, por la y de inicio).
    """
    if not ball_pts:
        return [], {"n_rallies": 0, "n_serves": 0, "avg_rally_s": 0,
                    "longest_rally_s": 0, "total_play_s": 0, "serves_by_side": {}}

    pts = sorted(ball_pts, key=lambda b: b["frame"])
    max_gap_f = max_gap_s * fps
    groups = [[pts[0]]]
    for p in pts[1:]:
        if p["frame"] - groups[-1][-1]["frame"] <= max_gap_f:
            groups[-1].append(p)
        else:
            groups.append([p])

    rallies = []
    for g in groups:
        dur = (g[-1]["frame"] - g[0]["frame"]) / fps
        if len(g) < min_points or dur < min_rally_s:
            continue
        sx, sy = g[0]["court_x"], g[0]["court_y"]
        # El saque nace detras de una linea de fondo. En cancha completa la y de
        # inicio dice quien saco; en media cancha solo vemos un lado.
        side = ("A" if half_court else
                "A" if sy <= court_h / 2 else "B")
        rallies.append({
            "start_frame": g[0]["frame"],
            "end_frame": g[-1]["frame"],
            "duration_s": round(dur, 1),
            "n_points": len(g),
            "serve_side": side,
            "serve_pos_m": [round(sx, 2), round(sy, 2)],
        })

    durs = [r["duration_s"] for r in rallies]
    by_side = {}
    for r in rallies:
        by_side[r["serve_side"]] = by_side.get(r["serve_side"], 0) + 1
    summary = {
        "n_rallies": len(rallies),
        "n_serves": len(rallies),          # ~1 saque por rally
        "avg_rally_s": round(sum(durs) / len(durs), 1) if durs else 0,
        "longest_rally_s": max(durs) if durs else 0,
        "total_play_s": round(sum(durs), 1),
        "serves_by_side": by_side,
    }
    return rallies, summary


# ============================================================
#  PIPELINE PRINCIPAL
# ============================================================
def run(video_path, calibration_path, output_dir="out", stride=5,
        person_conf=0.4, ball_conf=0.10,
        ball_detector="yolo", ball_model_path=None, vballnet_model=None,
        vballnet_stride=1,
        vballnet_cache=True,
        pose_mode="none",
        court_model=None,
        court_seg_model=None,
        court_motion=False,
        roi_margin_m=2.0,
        use_roi=True,
        save_diagnostic=True):

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Auto-calibracion: si se pasa court_model (keypoints) y/o court_seg_model
    # (segmentacion pre-entrenada, sin entrenar), intentar calibrar solo.
    # Si funciona, se usa esa calibracion. Si falla, cae al cal.json manual
    # (si existe) o aborta pidiendo MIRA.
    if court_model is not None or court_seg_model is not None or court_motion:
        auto_cal = None

        if court_model is not None:
            from court_keypoints import auto_calibrate
            print("[•] Intentando auto-calibracion con modelo de cancha (keypoints)...")
            auto_cal = auto_calibrate(video_path, court_model)

        # Fallback sin entrenar: segmentacion de cancha pre-entrenada. Produce
        # el mismo cal.json y la misma convencion de coordenadas que
        # court_keypoints, asi que zone_for_court_pos sigue siendo valido.
        if auto_cal is None and court_seg_model is not None:
            from court_segmentation import auto_calibrate_seg
            print("[•] Intentando auto-calibracion por segmentacion (sin entrenar)...")
            auto_cal = auto_calibrate_seg(video_path, court_seg_model,
                                          qc_path=str(out / "cal_check.jpg"))

        # Fallback por MOVIMIENTO: deduce la cancha de donde se mueven las
        # jugadoras. Robusto en gimnasios multiuso/oblicuos donde la
        # segmentacion falla. Mismo cal.json y convencion de coordenadas.
        if auto_cal is None and court_motion:
            from court_motion_calibration import auto_calibrate_motion
            print("[•] Intentando auto-calibracion por movimiento (sin entrenar)...")
            auto_cal = auto_calibrate_motion(video_path,
                                             vballnet_model=vballnet_model,
                                             qc_path=str(out / "cal_check.jpg"))

        if auto_cal is not None:
            cal = auto_cal
            (out / "cal_auto.json").write_text(json.dumps(cal, indent=2))
            print(f"[•] Auto-calibracion OK — guardada en {out/'cal_auto.json'}")
        elif calibration_path is not None:
            print("[•] Auto-calibracion fallo — usando cal.json manual.")
            cal = json.loads(Path(calibration_path).read_text())
        else:
            raise RuntimeError(
                "Auto-calibracion fallo y no se dio --calibration manual. "
                "Calibra con MIRA y pasa el cal.json."
            )
    else:
        if calibration_path is None:
            raise RuntimeError(
                "Falta calibracion: pasa --calibration cal.json o "
                "--court-model modelo.pt"
            )
        cal = json.loads(Path(calibration_path).read_text())
    H = np.array(cal["homography_matrix"])
    court_w, court_h = cal["court_size_m"]
    ppm = cal["pixels_per_meter"]
    half_court = cal.get("half_court", False)
    court_horizon_y = cal.get("court_horizon_y", None)

    # ROI de zona de juego en píxeles de imagen. Descarta público/banca/equipo
    # lejano que se proyecta dentro del rango de cancha pero está fuera de la
    # cancha EN LA IMAGEN. Ver court_roi_polygon() para el porqué.
    roi_poly = court_roi_polygon(H, court_w, court_h, ppm, roi_margin_m) if use_roi else None
    if use_roi and roi_poly is None:
        print("⚠ ROI desactivado: homografía no invertible.")
    max_h_ratio = cal.get("max_person_height_ratio", 0.55)
    max_w_ratio = cal.get("max_person_width_ratio", 0.40)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"No pude abrir video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    Hf = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print(f"\n┌─ CLARA v0.8 ─────────────────────────────")
    print(f"│ Video: {Path(video_path).name}")
    print(f"│ {W}x{Hf} @ {fps:.1f}fps | {total_frames/fps/60:.2f} min")
    print(f"│ Cancha: {court_w}x{court_h}m {'[HALF]' if half_court else '[FULL]'}")
    print(f"│ Balón: {ball_detector}")
    print(f"│ Pose:  {pose_mode}")
    print(f"│ Stride: {stride}")
    print(f"└──────────────────────────────────────────\n")

    # Detector de personas: yolo11m (medium). CLARA es offline, no hay
    # presion de tiempo real, asi que no se usa el modelo nano: detecta
    # peor a jugadoras chicas/lejanas/borrosas (footage de celular) y cada
    # deteccion perdida es un hueco por donde se fragmenta un track.
    # 'm' es el punto medio para CPU; en GPU se puede subir a 'yolo11x'.
    person_model = YOLO("yolo11m.pt")

    pose_estimator = None
    if pose_mode == "rtmlib":
        try:
            from pose_rtmlib import PoseEstimator
            pose_estimator = PoseEstimator(mode="balanced", device="cpu")
        except ImportError:
            print("⚠ rtmlib no disponible. pip install rtmlib")
            pose_mode = "none"

    raw_tracks = defaultdict(list)
    rejected_counts = defaultdict(int)
    ball_detections = []
    samples_processed = 0

    classes_to_track = [0] if (ball_detector != "yolo" or ball_model_path) else [0, 32]

    # Usar la config de ByteTrack tuneada de CLARA (track_buffer alto para
    # aguantar oclusiones). Si el archivo no esta junto al script, cae al
    # bytetrack.yaml default de Ultralytics.
    tracker_cfg = Path(__file__).parent / "bytetrack_clara.yaml"
    tracker_arg = str(tracker_cfg) if tracker_cfg.exists() else "bytetrack.yaml"

    # IMPORTANTE: el tracker procesa TODOS los frames del video.
    # ByteTrack predice la posicion de cada jugadora con un filtro de
    # Kalman y asocia por IoU asumiendo frames consecutivos. Si se le
    # alimentan frames salteados (el viejo vid_stride=stride), una
    # jugadora se desplaza ~stride veces mas en pixeles entre frame y
    # frame, el IoU cae por debajo de match_thresh y ByteTrack crea un
    # ID nuevo -> tracks fragmentados. El submuestreo para analitica se
    # hace DESPUES, en el loop, no aqui.
    results = person_model.track(
        source=video_path,
        persist=True,
        tracker=tracker_arg,
        classes=classes_to_track,
        conf=min(person_conf, ball_conf),
        verbose=False,
        stream=True,
    )

    for frame_idx, r in enumerate(results):
        # ByteTrack ya proceso este frame y actualizo sus IDs internamente
        # (basta con consumir `r` del generador). Solo recolectamos un
        # sample para analitica cada `stride` frames: el tracking necesita
        # continuidad total, la analitica de zonas/heatmap no.
        if frame_idx % stride != 0:
            continue
        actual_frame = frame_idx

        if r.boxes is not None:
            cls = r.boxes.cls.int().cpu().tolist()
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().tolist()
            ids = (r.boxes.id.int().cpu().tolist()
                   if r.boxes.id is not None else [None] * len(cls))

            valid_persons = []
            for c, box, conf, tid in zip(cls, xyxy, confs, ids):
                x1, y1, x2, y2 = box
                cx = (x1 + x2) / 2.0
                ground_y = y2 if c == 0 else (y1 + y2) / 2.0
                cx_m, cy_m = project(H, cx, ground_y)
                court_x = float(cx_m) / ppm
                court_y = float(cy_m) / ppm

                if c == 0:
                    if conf < person_conf or tid is None:
                        continue
                    status, _ = classify_detection(
                        box, Hf, W, court_horizon_y,
                        max_h_ratio, max_w_ratio, is_ball=False,
                    )
                    if status != "ok":
                        rejected_counts[status] += 1
                        continue
                    # ROI en espacio de imagen: el punto de pie debe caer
                    # dentro de la zona de juego. Filtra público/banca/equipo
                    # lejano que is_in_court() deja pasar (ver court_roi_polygon).
                    if not point_in_polygon(cx, ground_y, roi_poly):
                        rejected_counts["fg_outside_roi"] += 1
                        continue
                    sample = {
                        "frame": actual_frame,
                        "court_x": court_x, "court_y": court_y,
                        "conf": conf,
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    }
                    raw_tracks[tid].append(sample)
                    if pose_estimator:
                        valid_persons.append((tid, sample,
                                              [float(x1), float(y1),
                                               float(x2), float(y2)]))
                elif c == 32 and ball_detector == "yolo" and conf >= ball_conf:
                    status, _ = classify_detection(
                        box, Hf, W, court_horizon_y,
                        max_h_ratio, max_w_ratio, is_ball=True,
                    )
                    if status != "ok":
                        rejected_counts[f"ball_{status}"] += 1
                        continue
                    ball_detections.append({
                        "frame": actual_frame,
                        # cx / ground_y = centro del bbox del balón (px).
                        "img_x": float(cx), "img_y": float(ground_y),
                        "court_x": court_x, "court_y": court_y,
                        "conf": conf,
                    })

            if pose_estimator and valid_persons:
                # ultralytics ya tiene el frame decodificado; usarlo directo
                # es O(1) vs. seek aleatorio con cap.set(POS_FRAMES) que
                # invalida el cache del decodificador en cada muestra.
                frame_img = getattr(r, "orig_img", None)
                if frame_img is not None:
                    bboxes = [vp[2] for vp in valid_persons]
                    try:
                        pose_results = pose_estimator.estimate_batch(
                            frame_img, bboxes)
                        for (tid, sample, _), pose_data in zip(
                                valid_persons, pose_results):
                            if pose_data["avg_score"] >= 0.3:
                                sample["pose"] = {
                                    "kp": pose_data["keypoints"],
                                    "sc": pose_data["scores"],
                                    "avg": pose_data["avg_score"],
                                }
                    except Exception as e:
                        if samples_processed % 50 == 0:
                            print(f"⚠ Pose error frame {actual_frame}: {e}")

        samples_processed += 1
        if samples_processed % 200 == 0:
            pct = actual_frame / total_frames * 100
            print(f"  ⏳ {samples_processed} muestras ({pct:.0f}%)")

    # ─── Detección de balón ───
    if ball_detector == "yolo" and ball_model_path and Path(ball_model_path).exists():
        print(f"\n[•] Detectando balón con modelo custom YOLO11...")
        ball_model = YOLO(ball_model_path)
        cap = cv2.VideoCapture(video_path)
        for sample_idx in range(samples_processed):
            fi = sample_idx * stride
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                break
            ball_res = ball_model(frame, verbose=False, conf=ball_conf)[0]
            if ball_res.boxes is not None and len(ball_res.boxes) > 0:
                for box, conf in zip(ball_res.boxes.xyxy.cpu().numpy(),
                                     ball_res.boxes.conf.cpu().tolist()):
                    x1, y1, x2, y2 = box
                    status, _ = classify_detection(
                        box, Hf, W, court_horizon_y,
                        max_h_ratio, max_w_ratio, is_ball=True,
                    )
                    if status != "ok":
                        rejected_counts[f"ball_{status}"] += 1
                        continue
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    cx_m, cy_m = project(H, cx, cy)
                    ball_detections.append({
                        "frame": fi,
                        "img_x": float(cx), "img_y": float(cy),
                        "court_x": float(cx_m) / ppm,
                        "court_y": float(cy_m) / ppm,
                        "conf": conf,
                    })
        cap.release()

    elif ball_detector == "vballnet":
        if vballnet_model is None or not Path(vballnet_model).exists():
            raise FileNotFoundError(
                "Para --ball-detector vballnet necesitas --vballnet-model <ruta .onnx>"
            )
        print(f"\n[•] Detectando balón con VballNet (stride={vballnet_stride})...")
        ball_detections.extend(
            detect_balls_vballnet(video_path, vballnet_model, H, ppm,
                                   court_w, court_h,
                                   frame_h=Hf, frame_w=W,
                                   court_horizon_y=court_horizon_y,
                                   max_h_ratio=max_h_ratio,
                                   max_w_ratio=max_w_ratio,
                                   rejected_counts=rejected_counts,
                                   stride=vballnet_stride,
                                   use_cache=vballnet_cache,
                                   verbose=True)
        )

    # ─── Track stitching: cose tracks fragmentados ───
    # ByteTrack parte una jugadora en varios IDs cuando hay oclusion.
    # Si un track termina donde/cuando otro empieza, y la posicion es
    # coherente, son la misma jugadora -> se unen.
    n_tracks_before_stitch = len(raw_tracks)
    # Volcar los tracks crudos (pre-cosido) para poder iterar los parametros
    # de stitch offline sin re-trackear, que es la parte cara del pipeline.
    (out / "raw_tracks.json").write_text(json.dumps(
        {str(tid): s for tid, s in raw_tracks.items()}, indent=2))
    # Oclusiones de voleibol (red, bloqueos) pasan facil de 1.5 s y la jugadora
    # se mueve mas de 2.5 m durante una larga. Barrido sobre raw_tracks.json de
    # test.mp4: 3s/4m deja 14 tracks (6 jugadoras reales), 5s/6m baja a 8 SIN
    # subir el spread espacial maximo (4.08 m, identico) -> fusiona fragmentos
    # sin unir jugadoras distintas; 8s/8m ya colapsa a 4 (over-merge). 5/6 es
    # el optimo medido.
    raw_tracks = stitch_tracks(raw_tracks, fps, stride,
                               max_gap_s=5.0, max_jump_m=6.0)
    n_stitched = n_tracks_before_stitch - len(raw_tracks)

    # ─── Filtrado de tracks ───
    min_samples = 15
    filtered = {}
    zone_visits = defaultdict(int)
    zone_first = defaultdict(int)
    zone_second = defaultdict(int)
    half_frame = total_frames // 2

    for tid, samples in raw_tracks.items():
        if len(samples) < min_samples:
            continue
        in_court = [s for s in samples
                    if is_in_court(s["court_x"], s["court_y"], court_w, court_h)]
        if not in_court:
            continue
        if (len(in_court) / len(samples) < 0.5 or
            len(in_court) < min_samples // 2):
            continue
        filtered[tid] = in_court
        for s in in_court:
            z = zone_for_court_pos(s["court_x"], s["court_y"],
                                    court_w, court_h, half_court, margin=0.5)
            if z:
                zone_visits[z] += 1
                if s["frame"] < half_frame:
                    zone_first[z] += 1
                else:
                    zone_second[z] += 1

    # ─── Filtrado de balón: en PÍXELES contra el polígono de cancha ───
    # El balón vuela encima del piso; proyectarlo por la homografía del piso
    # da coords de cancha falsas (un balón aéreo "aterriza" fuera de las
    # líneas, e is_in_court lo mataba — se perdía ~96% de lo detectado).
    # Se filtra con el píxel del balón contra la cancha-en-imagen estirada
    # hacia arriba. court_x/court_y se conservan: son fiables cuando el balón
    # está cerca del piso (para saber en qué zona cayó la jugada).
    ball_region = ball_valid_region(cal, Hf, W)
    if ball_region is not None:
        ball_clean = [b for b in ball_detections
                      if point_in_polygon(b["img_x"], b["img_y"], ball_region)]
        ball_filter_mode = "pixel_polygon"
    else:
        # Calibración sin pixel_corners (cal viejo): cae al filtro anterior.
        print("⚠ cal.json sin pixel_corners — balón filtrado por proyección "
              "(modo previo). Recalibra con MIRA para el filtro en píxeles.")
        ball_clean = [b for b in ball_detections
                      if is_in_court(b["court_x"], b["court_y"],
                                     court_w, court_h, margin=2)]
        ball_filter_mode = "court_projection_fallback"
    ball_before_tracking = len(ball_clean)
    ball_clean = filter_ball_tracks(ball_clean, stride=stride, frame_w=W)
    ball_isolated_dropped = ball_before_tracking - len(ball_clean)
    if ball_isolated_dropped > 0:
        rejected_counts["ball_isolated"] = ball_isolated_dropped

    # Frames únicos con al menos una detección de balón limpia.
    ball_frames_oncourt = len({b["frame"] for b in ball_clean})
    # Denominador honesto: frames que el detector de balón REALMENTE evaluó.
    # vballnet corre sobre el video completo / vballnet_stride; yolo va en el
    # loop de personas, con su stride. (El bug previo dividía frames de balón
    # a tasa completa entre samples_processed, que va con stride: unidades
    # distintas, tasa subestimada.)
    if ball_detector == "vballnet":
        ball_frames_evaluated = max(1, total_frames // max(vballnet_stride, 1))
    else:
        ball_frames_evaluated = max(1, samples_processed)

    # ─── Reconstrucción de trayectoria del balón (rellena huecos) ───
    # VballNet detecta ~1 de cada 3 frames; entre dos contactos el balón vuela
    # en arco, así que los frames perdidos se reconstruyen ajustando el vuelo
    # en espacio de imagen (img_x/img_y) y reproyectando a cancha. Los puntos
    # interpolados van marcados (interp=True) y NO cuentan para el recall ni el
    # score — esto sólo densifica la trayectoria para zonas/tempo/visualización.
    ball_trajectory_dense, recon_summary = reconstruct_trajectory(
        ball_clean, fps, H, ppm, frame_w=W)

    # ─── Segmentación de rallies / saques (heurística sobre el balón) ───
    rallies, play_summary = segment_rallies(
        ball_clean, fps, court_h=court_h, half_court=half_court)

    # ─── Score de calidad ───
    expected_tracks = 6 if half_court else 12
    score, score_breakdown = compute_quality_score(
        filtered, zone_visits, ball_frames_oncourt, rejected_counts,
        samples_processed, expected_tracks, half_court,
        ball_frames_evaluated=ball_frames_evaluated,
    )

    # ─── Pose analytics ───
    pose_stats = {}
    if pose_estimator:
        try:
            from pose_rtmlib import torso_lean_angle, stance_width, knee_flexion
            for tid, samples in filtered.items():
                lean_vals, stance_vals, knee_vals = [], [], []
                for s in samples:
                    p = s.get("pose")
                    if not p:
                        continue
                    lean = torso_lean_angle(p["kp"], p["sc"])
                    if lean is not None:
                        lean_vals.append(lean)
                    sw = stance_width(p["kp"], p["sc"])
                    if sw is not None:
                        stance_vals.append(sw)
                    # Promediar las dos rodillas — con la jugadora de perfil
                    # una pierna suele estar ocluida; si sólo medimos la
                    # izquierda perdemos la mitad de las muestras útiles.
                    kf_sides = [
                        v for v in (
                            knee_flexion(p["kp"], p["sc"], side="left"),
                            knee_flexion(p["kp"], p["sc"], side="right"),
                        ) if v is not None
                    ]
                    if kf_sides:
                        knee_vals.append(float(np.mean(kf_sides)))
                if lean_vals or stance_vals or knee_vals:
                    pose_stats[tid] = {
                        "samples_with_pose": len(lean_vals),
                        "torso_lean_deg_avg": (round(float(np.mean(lean_vals)), 1)
                                                if lean_vals else None),
                        "torso_lean_deg_max": (round(float(np.max(lean_vals)), 1)
                                                if lean_vals else None),
                        "stance_width_px_avg": (round(float(np.mean(stance_vals)), 1)
                                                 if stance_vals else None),
                        "knee_flexion_deg_avg": (round(float(np.mean(knee_vals)), 1)
                                                  if knee_vals else None),
                    }
        except ImportError:
            pass

    # Trayectoria del balón — DETECCIONES REALES (para la máquina de toques).
    # img_x/img_y es la verdad principal; court_x/court_y es fiable sólo con
    # el balón cerca del piso.
    ball_track = sorted(
        ({"frame": b["frame"],
          "img_x": round(b["img_x"], 1), "img_y": round(b["img_y"], 1),
          "court_x": round(b["court_x"], 2), "court_y": round(b["court_y"], 2),
          "conf": round(float(b["conf"]), 3)}
         for b in ball_clean),
        key=lambda b: b["frame"])

    metrics = {
        "clara_version": "0.8",
        "video": Path(video_path).name,
        "duration_s": round(total_frames / fps, 1),
        "duration_min": round(total_frames / fps / 60, 2),
        "stride": stride,
        "samples_processed": samples_processed,
        "court_size_m": [court_w, court_h],
        "half_court": half_court,
        "ball_detector": ball_detector,
        "pose_mode": pose_mode,
        "raw_tracks": n_tracks_before_stitch,
        "tracks_after_stitch": len(raw_tracks),
        "tracks_stitched": n_stitched,
        "filtered_tracks": len(filtered),
        "rejected_detections": dict(rejected_counts),
        "ball_detections_oncourt": len(ball_clean),
        "ball_frames_oncourt": ball_frames_oncourt,
        "ball_frames_evaluated": ball_frames_evaluated,
        "ball_detection_rate": round(
            ball_frames_oncourt / max(ball_frames_evaluated, 1), 3),
        "ball_filter": ball_filter_mode,
        "quality_score": score,
        "quality_breakdown": score_breakdown,
        "play_summary": play_summary,
        "rallies": rallies,
        # Legacy: trayectoria en coords de cancha de las detecciones reales.
        # Se conserva por compatibilidad con clara_report; para análisis usa
        # ball_track (real) y ball_track_reconstructed (densa).
        "ball_trajectory": [[b["frame"], round(b["court_x"], 2),
                             round(b["court_y"], 2)]
                            for b in sorted(ball_clean, key=lambda b: b["frame"])],
        "ball_track": ball_track,
        # Trayectoria densa: detecciones reales + frames interpolados por vuelo
        # (interp=True). NO afecta recall ni score; densifica para zonas/tempo.
        "ball_track_reconstructed": ball_trajectory_dense,
        "ball_reconstruction": recon_summary,
        "zone_visits_total": dict(zone_visits),
        "zone_visits_first_half": dict(zone_first),
        "zone_visits_second_half": dict(zone_second),
        "tracks": [],
        # Serie temporal cruda por jugadora (court coords) para post-procesadores
        # como touch_machine.py. Va aparte de `tracks` (resumen coach-facing) para
        # NO pisar ese esquema: aquí están las muestras frame a frame.
        "track_samples": [],
        "pose_stats": pose_stats,
    }

    for tid, samples in filtered.items():
        xs = np.array([s["court_x"] for s in samples])
        ys = np.array([s["court_y"] for s in samples])
        dist_m = float(np.sum(np.hypot(np.diff(xs), np.diff(ys))))
        t_span = (samples[-1]["frame"] - samples[0]["frame"]) / fps
        speed = round(dist_m / t_span, 2) if t_span > 0 else None
        zones = defaultdict(int)
        for s in samples:
            z = zone_for_court_pos(s["court_x"], s["court_y"],
                                    court_w, court_h, half_court, margin=0.5)
            if z:
                zones[z] += 1
        dom = max(zones.items(), key=lambda x: x[1])[0] if zones else None
        # Perfil de zonas (% del tiempo en cada zona): el coach lee la
        # tendencia posicional de la jugadora, no solo su zona dominante.
        ztot = sum(zones.values())
        zone_profile = ({z: round(c / ztot * 100) for z, c in
                         sorted(zones.items(), key=lambda x: -x[1])}
                        if ztot else {})
        side = "A" if half_court or np.median(ys) <= court_h / 2 else "B"
        # Fiabilidad del track: distingue una jugadora bien seguida de un
        # fragmento corto, para que el coach sepa de cuales datos fiarse.
        reliability = ("alta" if len(samples) >= 50 else
                       "media" if len(samples) >= 25 else "baja")
        metrics["tracks"].append({
            "id": tid, "samples": len(samples),
            "seconds_tracked": round(t_span, 1),
            "reliability": reliability,
            "distance_m": round(dist_m, 1),
            "avg_speed_m_per_s": speed,
            "side": side, "dominant_zone": dom,
            "zone_profile_pct": zone_profile,
            "avg_court_pos_m": [round(float(xs.mean()), 2),
                                round(float(ys.mean()), 2)],
            "pose_stats": pose_stats.get(tid),
        })
    metrics["tracks"].sort(key=lambda t: -t["samples"])

    # Muestras frame a frame de cada track (court coords). touch_machine.py las
    # usa para atribuir cada toque del balón a la jugadora más cercana; sólo
    # necesita frame + court_x/court_y, así que el resto del sample se omite.
    metrics["track_samples"] = [
        {"id": tid,
         "samples": [{"frame": s["frame"],
                      "court_x": round(s["court_x"], 2),
                      "court_y": round(s["court_y"], 2)} for s in ss]}
        for tid, ss in filtered.items()
    ]

    # ─── Topdowns ───
    # El topdown es un mapa del PISO: sólo tiene sentido dibujar balones que
    # estaban cerca del piso, donde su proyección a cancha es fiable. Los
    # balones aéreos siguen en ball_track (JSON) para la máquina de toques.
    ball_floor = [b for b in ball_clean
                  if is_in_court(b["court_x"], b["court_y"],
                                 court_w, court_h, margin=1.0)]
    save_topdown(filtered, ball_floor, court_w, court_h, ppm,
                 out / "topdown.png", title="Total",
                 metrics=metrics, half_court=half_court)
    first = {tid: [s for s in ss if s["frame"] < half_frame]
             for tid, ss in filtered.items()}
    first = {k: v for k, v in first.items() if len(v) >= 5}
    save_topdown(first, [b for b in ball_floor if b["frame"] < half_frame],
                 court_w, court_h, ppm,
                 out / "topdown_first_half.png", title="Primera mitad",
                 half_court=half_court)
    second = {tid: [s for s in ss if s["frame"] >= half_frame]
              for tid, ss in filtered.items()}
    second = {k: v for k, v in second.items() if len(v) >= 5}
    save_topdown(second, [b for b in ball_floor if b["frame"] >= half_frame],
                 court_w, court_h, ppm,
                 out / "topdown_second_half.png", title="Segunda mitad",
                 half_court=half_court)

    if save_diagnostic:
        save_diagnostic_frame(video_path, cal, rejected_counts,
                              out / "diagnostic.png", H, ppm, roi_poly=roi_poly)
        if pose_estimator:
            save_pose_sample(video_path, filtered,
                              out / "pose_sample.png")

    with open(out / "scouting_data.json", "w") as f:
        json.dump(metrics, f, indent=2)

    quality_label = ("EXCELENTE" if score >= 80 else
                     "BUENO" if score >= 60 else
                     "REGULAR" if score >= 40 else
                     "BAJO — interpreta con cuidado")

    print(f"\n┌─ Resultado ────────────────────────────")
    print(f"│ Tracks limpios: {len(filtered)} "
          f"(de {n_tracks_before_stitch} crudos → {len(raw_tracks)} tras cosido)")
    if n_stitched > 0:
        print(f"│ Tracks cosidos: {n_stitched} fragmentos unidos")
    print(f"│ Balones: {len(ball_clean)} ({metrics['ball_detection_rate']*100:.1f}%) "
          f"[{ball_detector}/{ball_filter_mode}]")
    if recon_summary.get("interpolated", 0) > 0:
        print(f"│ Trayectoria: {recon_summary['total']} puntos "
              f"({recon_summary['real']} reales + "
              f"{recon_summary['interpolated']} interpolados, "
              f"{recon_summary['flights']} vuelos)")
    if play_summary["n_rallies"] > 0:
        print(f"│ Rallies: {play_summary['n_rallies']} "
              f"(~{play_summary['n_serves']} saques) · "
              f"promedio {play_summary['avg_rally_s']}s · "
              f"más largo {play_summary['longest_rally_s']}s")
    if pose_estimator:
        print(f"│ Pose: {len(pose_stats)} tracks con keypoints")
    else:
        print(f"│ Pose: deshabilitado")
    print(f"│ Rechazos: {sum(rejected_counts.values())}")
    print(f"│")
    print(f"│ ★ CALIDAD: {score}/100 — {quality_label}")
    for k, v in score_breakdown.items():
        print(f"│   {k}: {v}")
    print(f"└────────────────────────────────────────")
    return metrics


def compute_quality_score(tracks, zones, ball_frames, rejected,
                          total_samples, expected_tracks, half_court,
                          ball_frames_evaluated=None):
    """ball_frames: número de frames únicos con al menos una detección de balón.
    ball_frames_evaluated: frames que el detector de balón realmente evaluó —
    denominador correcto para la tasa de balón. Si no se pasa, cae a
    total_samples (compatibilidad)."""
    breakdown = {}
    # Simetrico: premia ACERTAR el numero de jugadoras, no tener "muchas".
    # 48 tracks para 6 jugadoras es tracking roto (espectadores/fragmentos),
    # no 30/30. Penaliza sobre-deteccion tanto como sub-deteccion. La vieja
    # `len/expected*30` topada daba 30/30 a cualquier conteo >= esperado.
    n_tr = len(tracks)
    track_ratio = min(n_tr, expected_tracks) / max(n_tr, expected_tracks, 1)
    track_pts = int(round(30 * track_ratio))
    breakdown["tracks"] = f"{track_pts}/30 ({n_tr} de {expected_tracks})"

    expected_zones = 6 if half_court else 12
    zones_with_data = len([v for v in zones.values() if v > 3])
    zone_pts = min(25, int(zones_with_data / expected_zones * 25))
    breakdown["zonas"] = f"{zone_pts}/25 ({zones_with_data} de {expected_zones})"

    ball_denom = ball_frames_evaluated if ball_frames_evaluated else total_samples
    ball_rate = ball_frames / max(ball_denom, 1)
    if ball_rate >= 0.50:
        ball_pts = 20
    elif ball_rate >= 0.10:
        ball_pts = int(4 + (ball_rate - 0.10) / 0.40 * 16)
    else:
        ball_pts = int(ball_rate / 0.10 * 4)
    breakdown["balon"] = f"{ball_pts}/20 ({ball_rate*100:.1f}%)"

    total_dets = sum(len(s) for s in tracks.values()) + sum(rejected.values())
    if total_dets > 0:
        accept_ratio = sum(len(s) for s in tracks.values()) / total_dets
        accept_pts = int(accept_ratio * 15)
    else:
        accept_pts = 0
    breakdown["filtrado"] = f"{accept_pts}/15 (rechazos: {sum(rejected.values())})"

    # Termometro honesto de continuidad de identidad. Blend 50/50 de dos
    # señales que NO se contradicen entre si:
    #   - persistencia: de las N tracks principales (las expected_tracks mas
    #     largas, ignorando la cola de fragmentos espurios), que fraccion del
    #     video duran. 50%+ = credito lleno.
    #   - consolidacion: que tan cerca quedo el #tracks del esperado. Penaliza
    #     tanto sobre-segmentar (IDs partidos) como sub-detectar.
    # El viejo `mean(duracion)/video` mezclaba ambos males y peleaba contra la
    # sub-metrica `tracks`: mas fragmentos subian `tracks` y hundian este.
    if tracks and total_samples > 0:
        lengths = sorted((len(s) for s in tracks.values()), reverse=True)
        top = lengths[:expected_tracks]
        persist_ratio = (sum(top) / len(top)) / total_samples
        persist_score = min(1.0, persist_ratio / 0.50)
        consol_score = max(0.0, 1.0 - abs(len(tracks) - expected_tracks)
                                       / expected_tracks)
        stability_pts = int(round(10 * (0.5 * persist_score
                                        + 0.5 * consol_score)))
    else:
        persist_score = consol_score = 0.0
        stability_pts = 0
    breakdown["estabilidad"] = (
        f"{stability_pts}/10 (persist {persist_score:.2f}, "
        f"consol {consol_score:.2f})")

    total = track_pts + zone_pts + ball_pts + accept_pts + stability_pts
    return total, breakdown


def save_topdown(tracks, ball, court_w, court_h, ppm, path,
                 title="", metrics=None, half_court=False):
    court_W = int(court_w * ppm)
    court_H = int(court_h * ppm)
    pad = 50
    stats_w = 220 if metrics else 0
    W = court_W + pad * 2 + stats_w
    Hi = court_H + pad * 2 + 60

    img = np.full((Hi, W, 3), 12, dtype=np.uint8)
    cx0, cy0 = pad, pad + 40

    cv2.rectangle(img, (cx0, cy0), (cx0 + court_W, cy0 + court_H),
                  (28, 26, 22), -1)
    cv2.rectangle(img, (cx0, cy0), (cx0 + court_W, cy0 + court_H),
                  PAL_BONE, 2)

    if not half_court:
        cv2.line(img, (cx0, cy0 + court_H // 2),
                 (cx0 + court_W, cy0 + court_H // 2), PAL_OXBLOOD, 3)
        att = int(3 * ppm)
        cv2.line(img, (cx0, cy0 + court_H // 2 - att),
                 (cx0 + court_W, cy0 + court_H // 2 - att), PAL_DARK, 1)
        cv2.line(img, (cx0, cy0 + court_H // 2 + att),
                 (cx0 + court_W, cy0 + court_H // 2 + att), PAL_DARK, 1)
        zones_pos = {
            "A4": (1.5, 7.5), "A3": (4.5, 7.5), "A2": (7.5, 7.5),
            "A5": (1.5, 1.5), "A6": (4.5, 1.5), "A1": (7.5, 1.5),
            "B4": (7.5, 10.5), "B3": (4.5, 10.5), "B2": (1.5, 10.5),
            "B5": (7.5, 16.5), "B6": (4.5, 16.5), "B1": (1.5, 16.5),
        }
    else:
        cv2.line(img, (cx0, cy0), (cx0 + court_W, cy0), PAL_OXBLOOD, 3)
        att = int(3 * ppm)
        cv2.line(img, (cx0, cy0 + att), (cx0 + court_W, cy0 + att),
                 PAL_DARK, 1)
        zones_pos = {
            "A4": (1.5, 1.5), "A3": (4.5, 1.5), "A2": (7.5, 1.5),
            "A5": (1.5, 7.5), "A6": (4.5, 7.5), "A1": (7.5, 7.5),
        }

    for zname, (mx, my) in zones_pos.items():
        px = int(cx0 + mx * ppm)
        py = int(cy0 + my * ppm)
        cv2.putText(img, zname, (px - 13, py + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 56, 50), 1)

    overlay = img.copy()
    palette = [(40, 40, 165), (60, 200, 220), (122, 156, 90),
               (198, 154, 60), (208, 100, 200), (100, 100, 250),
               (50, 180, 100), (220, 120, 90), (160, 80, 220),
               (220, 200, 60), (60, 60, 200), (200, 60, 100)]
    for i, (tid, samples) in enumerate(tracks.items()):
        color = palette[i % len(palette)]
        for s in samples:
            px = int(cx0 + s["court_x"] * ppm)
            py = int(cy0 + s["court_y"] * ppm)
            cv2.circle(overlay, (px, py), 4, color, -1)
    img = cv2.addWeighted(overlay, 0.55, img, 0.45, 0)

    for b in ball:
        px = int(cx0 + b["court_x"] * ppm)
        py = int(cy0 + b["court_y"] * ppm)
        cv2.circle(img, (px, py), 4, PAL_BALL, 2)

    cv2.putText(img, f"CLARA - {title}", (cx0, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, PAL_BONE, 1)

    leg_y = cy0 + court_H + 25
    cv2.circle(img, (cx0 + 8, leg_y), 4, PAL_OXBLOOD, -1)
    cv2.putText(img, "jugadora", (cx0 + 20, leg_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, PAL_BONE, 1)
    cv2.circle(img, (cx0 + 100, leg_y), 4, PAL_BALL, 2)
    cv2.putText(img, "balon", (cx0 + 112, leg_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, PAL_BONE, 1)

    if metrics and stats_w > 0:
        sx = cx0 + court_W + pad
        sy = cy0
        score = metrics.get("quality_score", 0)
        score_color = (PAL_OK if score >= 60 else
                       PAL_WARN if score >= 40 else PAL_REJECT)
        cv2.putText(img, "CALIDAD", (sx, sy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, PAL_BONE, 1)
        cv2.putText(img, f"{score}/100", (sx, sy + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, score_color, 2)

        lines = [
            "",
            f"Duracion: {metrics['duration_min']:.1f}m",
            f"Tracks: {metrics['filtered_tracks']}",
            f"Balones: {metrics['ball_detections_oncourt']}",
            f"Detector: {metrics.get('ball_detector', '?')}",
            f"Pose: {metrics.get('pose_mode', 'none')}",
        ]
        if metrics.get("half_court"):
            lines.append("MODO: HALF-COURT")
        for i, line in enumerate(lines):
            cv2.putText(img, line, (sx, sy + 50 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, PAL_BONE, 1)

    cv2.imwrite(str(path), img)


def save_diagnostic_frame(video_path, cal, rejected_counts, path, H, ppm,
                          roi_poly=None):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return
    Hf, Wf = frame.shape[:2]
    corners = cal.get("pixel_corners", [])
    if corners:
        pts = np.array(corners, dtype=np.int32)
        cv2.polylines(frame, [pts], True, (60, 220, 230), 2)
        for x, y in corners:
            cv2.circle(frame, (int(x), int(y)), 6, (60, 220, 230), -1)
    # ROI de zona de juego (verde): todo punto de pie fuera de aquí se descarta.
    if roi_poly is not None:
        cv2.polylines(frame, [roi_poly.reshape(-1, 1, 2)], True,
                      (90, 156, 122), 2, cv2.LINE_AA)
    hor = cal.get("court_horizon_y")
    if hor is not None:
        cv2.line(frame, (0, int(hor)), (Wf, int(hor)),
                 (60, 156, 220), 1, cv2.LINE_AA)
    y_off = 30
    cv2.putText(frame, "DIAGNOSTICO CLARA",
                (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (232, 224, 208), 1)
    y_off += 24
    cv2.putText(frame, f"Rechazos: {sum(rejected_counts.values())}",
                (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (232, 224, 208), 1)
    y_off += 18
    for reason, count in rejected_counts.items():
        cv2.putText(frame, f"  {reason}: {count}",
                    (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (60, 156, 220), 1)
        y_off += 16
    cv2.imwrite(str(path), frame)


def save_pose_sample(video_path, filtered_tracks, path):
    """Render frame with skeletons of detected players."""
    try:
        from pose_rtmlib import draw_pose
    except ImportError:
        return

    best_frame_idx = None
    best_count = 0
    for tid, samples in filtered_tracks.items():
        for s in samples:
            if "pose" in s:
                fi = s["frame"]
                count = sum(1 for tid2, samples2 in filtered_tracks.items()
                           for s2 in samples2
                           if s2["frame"] == fi and "pose" in s2)
                if count > best_count:
                    best_count = count
                    best_frame_idx = fi
                if best_count >= 6:
                    break

    if best_frame_idx is None:
        return

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, best_frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return

    for tid, samples in filtered_tracks.items():
        for s in samples:
            if s["frame"] == best_frame_idx and "pose" in s:
                draw_pose(frame, s["pose"]["kp"], s["pose"]["sc"])

    cv2.putText(frame, f"CLARA pose sample · frame {best_frame_idx}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (232, 224, 208), 2)
    cv2.imwrite(str(path), frame)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="CLARA v0.8 — multimodal scouting")
    p.add_argument("video")
    p.add_argument("--calibration", default=None,
                   help="cal.json manual (de MIRA). Opcional si se usa "
                        "--court-model.")
    p.add_argument("--court-model", default=None,
                   help="Modelo YOLO-pose de cancha para auto-calibracion. "
                        "Si se da, CLARA calibra sola; si falla, cae a "
                        "--calibration.")
    p.add_argument("--court-seg-model", default=None,
                   help="Modelo YOLO-seg de cancha (court/weights/best.pt de "
                        "volleyball_analytics) para auto-calibracion por "
                        "segmentacion SIN entrenar. Se usa como fallback si "
                        "--court-model no se da o falla; mismo cal.json.")
    p.add_argument("--court-motion", action="store_true",
                   help="Auto-calibracion por MOVIMIENTO (sin entrenar): deduce "
                        "la cancha de donde se mueven las jugadoras. Robusto en "
                        "gimnasios multiuso/oblicuos donde --court-seg-model "
                        "falla. Fallback tras keypoints/segmentacion; mismo "
                        "cal.json. Usa --vballnet-model si se da como pista extra.")
    p.add_argument("--out", default="out")
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--person-conf", type=float, default=0.4)
    p.add_argument("--ball-conf", type=float, default=0.10)
    p.add_argument("--ball-detector", choices=["yolo", "vballnet"], default="yolo")
    p.add_argument("--ball-model", default=None)
    p.add_argument("--vballnet-model", default=None)
    p.add_argument("--vballnet-stride", type=int, default=1,
                   help="Submuestreo del buffer de VballNet (1=todos los frames). "
                        "Subir a 2-3 reduce RAM e inferencias en clips largos a "
                        "costa de recall.")
    p.add_argument("--no-vballnet-cache", action="store_true",
                   help="No cachear ni leer la pasada de VballNet "
                        "(siempre recomputa). Por defecto cachea junto al video.")
    p.add_argument("--pose", choices=["none", "rtmlib"], default="none")
    p.add_argument("--roi-margin", type=float, default=2.0,
                   help="Margen en metros alrededor de la cancha para el ROI "
                        "de imagen (filtra público/banca/equipo lejano). "
                        "Súbelo si recorta jugadoras reales, bájalo si se "
                        "cuela gente del fondo. Verifica en diagnostic.png.")
    p.add_argument("--no-roi", action="store_true",
                   help="Desactiva el filtro ROI de imagen (vuelve al "
                        "comportamiento previo: solo is_in_court).")
    a = p.parse_args()
    if (a.calibration is None and a.court_model is None and
            a.court_seg_model is None and not a.court_motion):
        p.error("Se requiere --calibration cal.json, --court-model, "
                "--court-seg-model o --court-motion")
    run(a.video, a.calibration, a.out, a.stride,
        a.person_conf, a.ball_conf,
        ball_detector=a.ball_detector,
        ball_model_path=a.ball_model,
        vballnet_model=a.vballnet_model,
        vballnet_stride=a.vballnet_stride,
        vballnet_cache=not a.no_vballnet_cache,
        pose_mode=a.pose,
        court_model=a.court_model,
        court_seg_model=a.court_seg_model,
        court_motion=a.court_motion,
        roi_margin_m=a.roi_margin,
        use_roi=not a.no_roi)
