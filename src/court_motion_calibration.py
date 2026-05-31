"""
court_motion_calibration.py — Auto-calibración de CLARA por MOVIMIENTO (sin entrenar).
======================================================================================

Tercer backend de auto-calibración, hermano de court_segmentation.py y
court_keypoints.py. Pensado para gimnasios MULTIUSO / ángulos oblicuos donde la
segmentación de cancha pre-entrenada falla (la cancha sale con <6.5% del frame y
las líneas de basket confunden al modelo de apariencia).

IDEA: en vez de mirar cómo se VE la cancha, deducimos DÓNDE está a partir de
DÓNDE se mueven las jugadoras. Pasada ligera de YOLO11m sobre frames muestreados
→ punto de pie por jugadora → densidad 2D → la huella densa ES la cancha. Es
robusto a las líneas pintadas porque usa movimiento, no apariencia.

Produce EXACTAMENTE el mismo cal.json que court_segmentation.auto_calibrate_seg
(misma convención de coordenadas y mismas llaves), así que clara.py lo consume
igual y zone_for_court_pos sigue dando las zonas correctas. Reusa la geometría/QC
validada de court_segmentation.py (order_points, corners_from_mask, draw_qc_overlay).

Convención de coordenadas (idéntica a court_segmentation.py):
    cercana (abajo en imagen)  -> y = court_h   (cerca de cámara)
    lejana  (arriba en imagen) -> y = 0
    x: 0 lateral izquierda -> court_w lateral derecha

CPU y GPU: la pasada de YOLO11m auto-selecciona device (cuda si hay, si no cpu),
configurable con --device. Toda la geometría (densidad, esquinas, homografía,
Hough) es OpenCV/CPU e independiente del device.

Tamaño de cancha AUTO: detecta media (9×9) vs completa (9×18) por el ASPECTO
métrico del quad (rectificación desde un rectángulo con dos puntos de fuga). El
ancho es 9 m en ambos casos; el aspecto largo/corto (~1 media, ~2 completa) lo
discrimina. Funciona con cámara OBLICUA (el caso de este gimnasio); con cámara
centrada sobre el eje de la cancha el aspecto queda indeterminado y se asume FULL
avisando. Forzable siempre con --court-size {full,half}.

Uso standalone (cero cambios a clara.py):
    python court_motion_calibration.py video.mp4 --out cal_motion.json --qc cal_check.jpg
    python clara.py video.mp4 --calibration cal_motion.json --ball-model ...

LIMITACIÓN HONESTA: la envolvente puede incluir calentamiento/run-off (se mitiga
con el umbral por percentil), la línea de fondo cercana sale escorzada/ocluida, y
la detección media/completa es heurística (--court-size es el escape). SIEMPRE
revisa el cal_check.jpg: el perímetro verde debe pegar con la cancha real. Si la
auto falla, clara.py cae solo a la calibración manual.
"""

import json
import argparse
from pathlib import Path

import cv2
import numpy as np

# Reusa la geometría/QC validada del backend de segmentación. order_points y
# CORNER_METERS quedan disponibles por si se necesitan; corners_from_mask y
# draw_qc_overlay son el núcleo reutilizado.
from court_segmentation import (
    order_points,            # noqa: F401  (reexport / utilidad)
    corners_from_mask,
    draw_qc_overlay,
    CORNER_METERS,           # noqa: F401  (9x18, caso particular de _corner_meters)
    MIN_POINTS,
)

DEFAULT_CONF = 0.25
PERSON_CLASS = 0             # clase 'person' en COCO/yolo11
# La cancha (huella de movimiento) debe ocupar al menos ~6% del frame para no
# calibrar sobre ruido (mismo criterio que court_segmentation.MIN_AREA_FRAC).
MIN_AREA_FRAC = 0.06


# ============================================================
#  HOMOGRAFÍA PARAMETRIZADA POR TAMAÑO DE CANCHA
# ============================================================
def _corner_meters(court_w, court_h):
    """Esquinas de cancha en metros [TL,TR,BR,BL] para un tamaño dado.

    Generaliza CORNER_METERS de court_segmentation (que fija 9×18) a media
    cancha 9×9. Mismo mapeo: TL->(0,0) lejana-izq, BR->(court_w,court_h)
    cercana-der.
    """
    return np.array([[0, 0], [court_w, 0], [court_w, court_h], [0, court_h]],
                    dtype=np.float32)


def build_homography(corners, court_w, court_h, ppm=40, flip_vertical=False):
    """corners imagen [TL,TR,BR,BL] -> homography_matrix (lista) para court_w×court_h.

    Versión parametrizada de court_segmentation.homography_from_corners (que es
    el caso court_w=9, court_h=18). flip_vertical=True si la cámara tiene la zona
    CERCANA arriba en la imagen.
    """
    if corners is None or len(corners) < MIN_POINTS:
        return None
    meters = _corner_meters(court_w, court_h)
    if flip_vertical:                      # intercambia cercana<->lejana
        meters = meters[[3, 2, 1, 0]]
    dst = (meters * ppm).astype(np.float32)
    src = np.asarray(corners, np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    if H is None:
        return None
    return H.tolist()


# ============================================================
#  1) PUNTOS DE ACTIVIDAD (espacio imagen)
# ============================================================
def _resolve_device(device):
    """'auto' -> 'cuda' si hay GPU, si no 'cpu'. Otro valor se respeta tal cual."""
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def collect_foot_points(video_path, person_model="yolo11m.pt",
                        n_sample_frames=200, conf=DEFAULT_CONF,
                        device="auto", verbose=True):
    """Pasada ligera de YOLO11m sobre frames muestreados (sin ROI ni calibración).

    Devuelve (points, heights, frame_shape, qc_frame):
      - points:   (N,2) float32 punto de pie (bbox_cx, bbox_y2) por jugadora.
      - heights:  (N,)  alto del bbox en px (para inferir cercana/lejana).
      - frame_shape: (Hf, W).
      - qc_frame: un frame representativo (muestra central) para el overlay QC.
    """
    from ultralytics import YOLO   # import perezoso (igual que seg)
    dev = _resolve_device(device)
    model = YOLO(str(person_model))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"No pude abrir video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    Hf = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if verbose:
        print(f"[motion-cal] Video {W}x{Hf}, muestreando {n_sample_frames} "
              f"frames (device={dev})...")

    idxs = np.linspace(0, max(total - 1, 0), n_sample_frames, dtype=int)
    mid_idx = idxs[len(idxs) // 2] if len(idxs) else 0
    pts, hts, qc_frame = [], [], None
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok:
            continue
        if int(fi) == int(mid_idx):
            qc_frame = frame.copy()
        res = model.predict(frame, classes=[PERSON_CLASS], conf=conf,
                            device=dev, verbose=False)[0]
        if res.boxes is None:
            continue
        for b in res.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = b[:4]
            pts.append([(x1 + x2) / 2.0, y2])   # punto de pie
            hts.append(y2 - y1)
    cap.release()
    if qc_frame is None and len(idxs):           # fallback: primer frame legible
        cap = cv2.VideoCapture(str(video_path))
        ok, qc_frame = cap.read()
        cap.release()

    points = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    heights = np.asarray(hts, dtype=np.float32).reshape(-1)
    if verbose:
        print(f"[motion-cal] {len(points)} puntos de pie recolectados.")
    return points, heights, (Hf, W), qc_frame


def collect_ball_points(video_path, vballnet_model, foot_bbox=None,
                        stride=2, verbose=True):
    """Opcional: puntos de balón (img_x,img_y) de VballNet como actividad extra.

    Sólo se conservan los que caen dentro del bounding box de los pies
    (foot_bbox = (xmin,ymin,xmax,ymax)) para descartar balones aéreos cerca del
    techo, que inflarían la envolvente hacia arriba. Devuelve (M,2) float32.
    """
    try:
        from ball_vballnet import detect_balls
    except Exception as e:
        if verbose:
            print(f"[motion-cal] VballNet no disponible ({e}); sigo sin balón.")
        return np.empty((0, 2), np.float32)
    dets = detect_balls(str(video_path), str(vballnet_model),
                        stride=stride, verbose=verbose)
    pts = np.array([[d["x"], d["y"]] for d in dets], dtype=np.float32) \
        if dets else np.empty((0, 2), np.float32)
    if foot_bbox is not None and len(pts):
        x0, y0, x1, y1 = foot_bbox
        keep = ((pts[:, 0] >= x0) & (pts[:, 0] <= x1) &
                (pts[:, 1] >= y0) & (pts[:, 1] <= y1))
        pts = pts[keep]
    if verbose:
        print(f"[motion-cal] {len(pts)} puntos de balón (dentro de la huella).")
    return pts


# ============================================================
#  2) ENVOLVENTE DE CANCHA (densidad de pies -> blob binario)
# ============================================================
def motion_envelope_mask(points, frame_shape, percentile=75, sigma=25.0,
                         extra_points=None, extra_weight=0.3):
    """Acumula puntos de pie en un heatmap, lo suaviza y umbraliza por percentil.

    El umbral por percentil tira el calentamiento/run-off disperso y deja la
    huella densa de la cancha. extra_points (p.ej. balón filtrado a la huella) se
    suman con peso bajo. Devuelve máscara uint8 (0/255) del tamaño del frame.
    """
    Hf, W = frame_shape
    acc = np.zeros((Hf, W), np.float32)
    if len(points):
        xs = np.clip(points[:, 0].astype(int), 0, W - 1)
        ys = np.clip(points[:, 1].astype(int), 0, Hf - 1)
        np.add.at(acc, (ys, xs), 1.0)
    if extra_points is not None and len(extra_points):
        xs = np.clip(extra_points[:, 0].astype(int), 0, W - 1)
        ys = np.clip(extra_points[:, 1].astype(int), 0, Hf - 1)
        np.add.at(acc, (ys, xs), extra_weight)

    acc = cv2.GaussianBlur(acc, (0, 0), sigmaX=sigma, sigmaY=sigma)
    pos = acc[acc > 0]
    if pos.size == 0:
        return np.zeros((Hf, W), np.uint8)
    thr = np.percentile(pos, percentile)
    mask = (acc >= thr).astype(np.uint8) * 255
    return mask


# ============================================================
#  3) ORIENTACIÓN + TAMAÑO (media vs completa)
# ============================================================
def detect_orientation(points, heights, frame_shape):
    """flip_vertical: True si la zona CERCANA (jugadoras más grandes) está ARRIBA.

    Las jugadoras cercanas a la cámara tienen bbox más alto. Si la mitad
    superior de la imagen concentra bboxes más altos que la inferior, la cámara
    está invertida respecto a la convención (cercana abajo).
    """
    if len(points) < 4:
        return False
    Hf = frame_shape[0]
    top = heights[points[:, 1] < Hf / 2]
    bot = heights[points[:, 1] >= Hf / 2]
    if len(top) == 0 or len(bot) == 0:
        return False
    return float(top.mean()) > float(bot.mean())


def _line(p, q):
    """Recta homogénea por dos puntos (cross product)."""
    return np.cross([p[0], p[1], 1.0], [q[0], q[1], 1.0])


def _intersect(l1, l2):
    """Punto de fuga (intersección) de dos rectas homogéneas, o None si ~paralelas."""
    p = np.cross(l1, l2)
    if abs(p[2]) < 1e-6:
        return None                      # se cruzan en el infinito (fronto-paralelo)
    return np.array([p[0] / p[2], p[1] / p[2]], np.float64)


def rectangle_aspect(corners, frame_shape):
    """Aspecto métrico (lado largo / lado corto) de un rectángulo desde su imagen.

    Rectificación métrica desde un rectángulo: con píxeles cuadrados y punto
    principal al centro, los DOS puntos de fuga de los lados ortogonales fijan la
    focal vía (v1-p0)·(v2-p0)+f²=0; con K se reconstruyen los 4 vértices 3D salvo
    escala resolviendo el paralelogramo (lados opuestos = mismo vector), y el
    cociente de longitudes de lados adyacentes es el aspecto real. No usa
    apariencia ni densidad: sólo las 4 esquinas (en orden cíclico).

    Devuelve (aspect, reliable):
      - reliable=True sólo cuando AMBOS puntos de fuga son finitos y f²>0 (cámara
        OBLICUA — el caso del usuario). Entonces aspect es métrico y exacto.
      - reliable=False cuando una dirección es fronto-paralela (cámara centrada
        mirando por el eje de la cancha): el problema queda indeterminado y se
        devuelve un aspecto SÓLO en píxeles, que el llamador no debe usar para
        decidir tamaño. (None, False) si está degenerado.
    """
    if corners is None or len(corners) != 4:
        return None, False
    TL, TR, BR, BL = [np.asarray(corners[i], np.float64) for i in range(4)]
    Hf, W = frame_shape[:2]
    cx, cy = W / 2.0, Hf / 2.0

    v1 = _intersect(_line(TL, TR), _line(BL, BR))   # fuga dir. lados top/bottom
    v2 = _intersect(_line(TL, BL), _line(TR, BR))   # fuga dir. lados left/right
    if v1 is None or v2 is None:                    # alguna dir. fronto-paralela
        return _aspect_pixels(TL, TR, BR, BL), False
    f2 = -((v1[0] - cx) * (v2[0] - cx) + (v1[1] - cy) * (v2[1] - cy))
    if f2 <= 1e-6:
        return _aspect_pixels(TL, TR, BR, BL), False
    f = float(np.sqrt(f2))

    Kinv = np.array([[1 / f, 0, -cx / f], [0, 1 / f, -cy / f], [0, 0, 1.0]])
    r = {k: Kinv @ np.array([p[0], p[1], 1.0])
         for k, p in (("TL", TL), ("TR", TR), ("BR", BR), ("BL", BL))}
    # paralelogramo: λ_TR rTR - rTL = λ_BR rBR - λ_BL rBL  (λ_TL = 1)
    M = np.column_stack([r["TR"], -r["BR"], r["BL"]])
    try:
        lam = np.linalg.solve(M, r["TL"])
    except np.linalg.LinAlgError:
        return _aspect_pixels(TL, TR, BR, BL), False
    P_TL, P_TR, P_BL = r["TL"], lam[0] * r["TR"], lam[2] * r["BL"]
    side_w = np.linalg.norm(P_TR - P_TL)
    side_l = np.linalg.norm(P_BL - P_TL)
    if min(side_w, side_l) < 1e-6:
        return None, False
    # invariante a que order_points cambie ancho<->largo: lado largo / corto >= 1
    return float(max(side_w, side_l) / min(side_w, side_l)), True


def _aspect_pixels(TL, TR, BR, BL):
    """Aspecto SÓLO en píxeles (no métrico). Para el caso no-confiable."""
    width = 0.5 * (np.linalg.norm(TR - TL) + np.linalg.norm(BR - BL))
    length = 0.5 * (np.linalg.norm(BL - TL) + np.linalg.norm(BR - TR))
    lo = min(width, length)
    return float(max(width, length) / lo) if lo > 1e-6 else None


def detect_court_size(corners, frame_shape, court_size="auto", verbose=True):
    """Decide media (9×9) vs completa (9×18) por el ASPECTO métrico del quad.

    El ancho de cancha es 9 m en ambos casos; sólo cambia el largo (9 vs 18),
    así que el aspecto métrico largo/ancho discrimina (~1 media, ~2 completa).
    Se recupera con rectangle_aspect() — geometría pura, robusto a relleno
    uniforme. Funciona con cámara OBLICUA (el caso del usuario); con cámara
    centrada (eje de la cancha) el aspecto queda indeterminado y se asume FULL
    avisando que se use --court-size si es media cancha. 'full'/'half' fuerzan.

    Devuelve (court_w, court_h, half_court).
    """
    if court_size == "full":
        return 9.0, 18.0, False
    if court_size == "half":
        return 9.0, 9.0, True

    aspect, reliable = rectangle_aspect(corners, frame_shape)
    if not reliable or aspect is None:
        if verbose:
            print("[motion-cal] tamaño auto: aspecto indeterminado (cámara "
                  "centrada/fronto-paralela) -> asumo FULL 9x18. Si es media "
                  "cancha, pasa --court-size half.")
        return 9.0, 18.0, False
    is_full = aspect >= 1.5      # umbral entre ~1 (media) y ~2 (completa)
    if verbose:
        print(f"[motion-cal] tamaño auto: aspecto largo/corto={aspect:.2f} "
              f"-> {'FULL 9x18' if is_full else 'HALF 9x9'}")
    return (9.0, 18.0, False) if is_full else (9.0, 9.0, True)


# ============================================================
#  4) SNAP A LÍNEAS (opcional)
# ============================================================
def refine_corners_to_lines(frame, corners, band_px=40,
                            canny_lo=50, canny_hi=150, verbose=True):
    """Ajusta cada arista del quad a la línea de cancha real cercana (Hough).

    Para cada uno de los 4 bordes del quad, busca con HoughLinesP segmentos
    dentro de una banda estrecha alrededor del borde y de orientación parecida,
    ajusta una recta robusta y reemplaza el borde. Las esquinas finales son las
    intersecciones de los 4 bordes ajustados. Conservador: si algún borde no
    encuentra soporte, conserva el original. Devuelve corners [TL,TR,BR,BL].
    """
    if corners is None or len(corners) != 4:
        return corners
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny_lo, canny_hi)
    Hf, W = gray.shape[:2]

    def edge_line(p0, p1):
        """Recta (a,b,c) ax+by+c=0 ajustada a la línea de cancha cerca de p0-p1."""
        mask = np.zeros((Hf, W), np.uint8)
        cv2.line(mask, tuple(p0.astype(int)), tuple(p1.astype(int)), 255, band_px)
        local = cv2.bitwise_and(edges, edges, mask=mask)
        segs = cv2.HoughLinesP(local, 1, np.pi / 180, threshold=40,
                               minLineLength=int(0.2 * np.hypot(*(p1 - p0))),
                               maxLineGap=20)
        edge_dir = (p1 - p0) / (np.linalg.norm(p1 - p0) + 1e-9)
        xs, ys = [], []
        if segs is not None:
            for s in segs[:, 0, :]:
                d = np.array([s[2] - s[0], s[3] - s[1]], np.float32)
                d /= (np.linalg.norm(d) + 1e-9)
                if abs(float(np.dot(d, edge_dir))) > 0.85:   # orientación similar
                    xs += [s[0], s[2]]
                    ys += [s[1], s[3]]
        if len(xs) < 4:
            return None
        vx, vy, x0, y0 = cv2.fitLine(np.column_stack([xs, ys]).astype(np.float32),
                                     cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        # recta como ax+by+c=0
        return np.array([vy, -vx, vx * y0 - vy * x0], np.float32)

    TL, TR, BR, BL = [corners[i].astype(np.float32) for i in range(4)]
    edges_pts = [(TL, TR), (TR, BR), (BR, BL), (BL, TL)]   # top,right,bottom,left
    lines = [edge_line(a, b) for a, b in edges_pts]
    if any(l is None for l in lines):
        if verbose:
            print("[motion-cal] snap: soporte insuficiente, conservo quad rugoso.")
        return corners

    def intersect(l1, l2):
        a1, b1, c1 = l1
        a2, b2, c2 = l2
        d = a1 * b2 - a2 * b1
        if abs(d) < 1e-6:
            return None
        return np.array([(b1 * c2 - b2 * c1) / d, (c1 * a2 - c2 * a1) / d],
                        np.float32)

    top, right, bottom, left = lines
    new = [intersect(left, top), intersect(top, right),
           intersect(right, bottom), intersect(bottom, left)]
    if any(p is None for p in new):
        return corners
    return order_points(np.array(new, np.float32))


# ============================================================
#  AUTO-CALIBRACIÓN POR MOVIMIENTO
# ============================================================
def auto_calibrate_motion(video_path, person_model="yolo11m.pt",
                          vballnet_model=None, n_sample_frames=200,
                          conf=DEFAULT_CONF, ppm=40, court_size="auto",
                          density_percentile=75, snap_lines=False,
                          flip_vertical=None, device="auto",
                          qc_path=None, verbose=True):
    """Deduce la cancha de DÓNDE se mueven las jugadoras y arma la homografía.

    Devuelve un dict tipo cal.json compatible con clara.py (mismas llaves que
    court_segmentation.auto_calibrate_seg), o None si no se pudo calibrar
    (entonces clara.py cae a manual).
    """
    points, heights, frame_shape, qc_frame = collect_foot_points(
        video_path, person_model, n_sample_frames, conf, device, verbose)
    if len(points) < 8:
        if verbose:
            print("[motion-cal] ✗ Muy pocos puntos de movimiento. Usa manual.")
        return None

    # balón opcional, filtrado a la huella de pies para no inflar hacia el techo
    ball_pts = None
    if vballnet_model is not None:
        x0, y0 = points.min(axis=0)
        x1, y1 = points.max(axis=0)
        ball_pts = collect_ball_points(video_path, vballnet_model,
                                       foot_bbox=(x0, y0, x1, y1), verbose=verbose)

    mask = motion_envelope_mask(points, frame_shape, density_percentile,
                                extra_points=ball_pts)
    if mask.sum() < MIN_AREA_FRAC * frame_shape[0] * frame_shape[1] * 255:
        if verbose:
            print("[motion-cal] ✗ Envolvente de movimiento demasiado chica.")
        return None

    corners = corners_from_mask(mask)
    if corners is None:
        if verbose:
            print("[motion-cal] ✗ No pude sacar esquinas de la envolvente.")
        return None

    if snap_lines and qc_frame is not None:
        corners = refine_corners_to_lines(qc_frame, corners, verbose=verbose)

    flip = detect_orientation(points, heights, frame_shape) \
        if flip_vertical is None else flip_vertical

    court_w, court_h, half_court = detect_court_size(
        corners, frame_shape, court_size, verbose)

    H = build_homography(corners, court_w, court_h, ppm, flip)
    if H is None:
        if verbose:
            print("[motion-cal] ✗ No se pudo calcular homografía.")
        return None

    if verbose:
        print(f"[motion-cal] ✓ Calibrado: cancha {court_w:.0f}x{court_h:.0f}m "
              f"{'[HALF]' if half_court else '[FULL]'}, flip={flip}")
        print(f"[motion-cal]   Esquinas (px): {corners.astype(int).tolist()}")

    if qc_path is not None and qc_frame is not None:
        cv2.imwrite(str(qc_path),
                    draw_qc_overlay(qc_frame, H, ppm, (court_w, court_h)))
        if verbose:
            print(f"[motion-cal]   QC overlay guardado en {qc_path} — REVÍSALO.")

    return {
        "_comment": "Generado por court_motion_calibration.py (auto-cal por movimiento, sin entrenar)",
        "video_reference": Path(video_path).name,
        "frame_shape": [frame_shape[0], frame_shape[1]],
        "court_size_m": [court_w, court_h],
        "pixels_per_meter": ppm,
        "homography_matrix": H,
        "half_court": half_court,
        "court_horizon_y": None,
        # pixel_corners a nivel raíz: activa el filtro de balón por imagen
        # (ball_valid_region en clara.py). El backend de segmentación lo omite.
        "pixel_corners": corners.astype(int).tolist(),
        "max_person_height_ratio": 0.55,
        "max_person_width_ratio": 0.40,
        "auto_calibration": {
            "method": "motion",
            "n_foot_points": int(len(points)),
            "n_ball_points": int(0 if ball_pts is None else len(ball_pts)),
            "density_percentile": density_percentile,
            "snap_lines": bool(snap_lines),
            "corners_px": corners.astype(int).tolist(),
            "flip_vertical": bool(flip),
        },
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Auto-calibración de cancha por MOVIMIENTO (sin entrenar)")
    p.add_argument("video")
    p.add_argument("--out", default="cal_motion.json")
    p.add_argument("--qc", default="cal_check.jpg",
                   help="ruta del overlay de control de calidad")
    p.add_argument("--person-model", default="yolo11m.pt",
                   help="modelo YOLO de personas (default yolo11m.pt)")
    p.add_argument("--vballnet-model", default=None,
                   help="modelo VballNet .onnx (opcional, actividad de balón)")
    p.add_argument("--frames", type=int, default=200)
    p.add_argument("--conf", type=float, default=DEFAULT_CONF)
    p.add_argument("--ppm", type=int, default=40)
    p.add_argument("--court-size", choices=["auto", "full", "half"], default="auto")
    p.add_argument("--density-percentile", type=float, default=75,
                   help="percentil de densidad para la envolvente (sube para "
                        "recortar calentamiento/run-off, baja si recorta cancha)")
    p.add_argument("--snap-lines", action="store_true",
                   help="ajusta el quad a líneas reales con Hough (experimental)")
    p.add_argument("--flip-vertical", action="store_true",
                   help="fuerza zona CERCANA arriba (default: auto)")
    p.add_argument("--device", default="auto",
                   help="auto|cpu|cuda|0 para la pasada de YOLO")
    a = p.parse_args()

    cal = auto_calibrate_motion(
        a.video, person_model=a.person_model, vballnet_model=a.vballnet_model,
        n_sample_frames=a.frames, conf=a.conf, ppm=a.ppm, court_size=a.court_size,
        density_percentile=a.density_percentile, snap_lines=a.snap_lines,
        flip_vertical=True if a.flip_vertical else None,
        device=a.device, qc_path=a.qc)
    if cal is None:
        print("\n✗ Auto-calibración por movimiento falló. Usa calibración manual.")
        raise SystemExit(1)
    Path(a.out).write_text(json.dumps(cal, indent=2))
    print(f"\n✓ Calibración guardada en {a.out}")
    print("  Revisa el QC overlay antes de confiar en ella.")
