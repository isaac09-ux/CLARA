"""
court_segmentation.py — Auto-calibración de CLARA SIN entrenar.
================================================================

Backend de auto-calibración alternativo a court_keypoints.py.

En vez de un modelo YOLO11-pose de 6 puntos (que hay que ENTRENAR), usa un
modelo de SEGMENTACIÓN de cancha YA PRE-ENTRENADO y descargable hoy
(court/weights/best.pt del repo volleyball_analytics), saca las 4 esquinas
de la máscara, y construye la homografía.

Produce EXACTAMENTE el mismo cal.json que court_keypoints.auto_calibrate,
así que es 100% compatible con clara.py (run() lo consume igual) y respeta
la MISMA convención de coordenadas, por lo que zone_for_court_pos sigue dando
las zonas correctas.

Convención de coordenadas (idéntica a court_keypoints.py):
    cercana (abajo en imagen)  -> y = 18   (cerca de cámara)
    lejana  (arriba en imagen) -> y = 0
    centro                     -> y = 9
    x: 0 lateral izquierda -> 9 lateral derecha

Pesos: descarga el ZIP de volleyball_analytics (Google Drive) y usa
       weights/court/weights/best.pt

Uso standalone (lo más simple — cero cambios a clara.py):
    python court_segmentation.py video.mp4 --model court_best.pt --out cal_auto.json --qc cal_check.jpg
    python clara.py video.mp4 --calibration cal_auto.json --ball-model ...

LIMITACIÓN HONESTA: la segmentación->esquinas es frágil en gimnasios
multiuso, ángulos bajos/oblicuos o cuando los jugadores tapan las líneas del
fondo. SIEMPRE revisa el cal_check.jpg: el contorno proyectado debe pegar con
la cancha real. Para trípode fijo, calibra UNA vez sobre un clip limpio
(pocos jugadores en cancha) y reúsa el JSON. Si la auto falla, clara.py cae
solo a la calibración manual.
"""

import json
import argparse
from pathlib import Path

import cv2
import numpy as np

# ultralytics se importa de forma perezosa dentro de auto_calibrate_seg, para
# que las utilidades de geometría (esquinas/homografía/QC) sean usables y
# testeables sin tener YOLO instalado.


# ============================================================
#  CONVENCIÓN DE ESQUINAS (idéntica a court_keypoints.py)
# ============================================================
# order_points devuelve [TL, TR, BR, BL] en espacio imagen.
# Mapeo a metros de cancha (cámara con fondo arriba, cercana abajo):
#   TL (arriba-izq, lejana-izq) -> (0, 0)
#   TR (arriba-der, lejana-der) -> (9, 0)
#   BR (abajo-der, cercana-der) -> (9, 18)
#   BL (abajo-izq, cercana-izq) -> (0, 18)
CORNER_METERS = np.array([[0, 0], [9, 0], [9, 18], [0, 18]], dtype=np.float32)

DEFAULT_CONF = 0.25
MIN_AREA_FRAC = 0.06     # la cancha debe ocupar al menos ~6% del frame
MIN_POINTS = 4


# ============================================================
#  MÁSCARA -> ESQUINAS  (lógica validada del módulo clara_court)
# ============================================================
def order_points(pts):
    """Ordena 4 puntos a [TL, TR, BR, BL] en espacio imagen."""
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = pts[:, 0] - pts[:, 1]
    return np.array([
        pts[np.argmin(s)],   # TL
        pts[np.argmax(d)],   # TR
        pts[np.argmax(s)],   # BR
        pts[np.argmin(d)],   # BL
    ], dtype=np.float32)


def corners_from_mask(mask):
    """4 esquinas (TL,TR,BR,BL) del blob de cancha más grande, o None."""
    m = (mask > 0).astype(np.uint8) * 255
    k = np.ones((5, 5), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(cnt, True)
    for eps in np.linspace(0.01, 0.08, 8):
        approx = cv2.approxPolyDP(cnt, eps * peri, True)
        if len(approx) == 4:
            return order_points(approx.reshape(-1, 2))
    return order_points(cv2.boxPoints(cv2.minAreaRect(cnt)))


def court_mask_from_result(res, frame_shape):
    """Une todas las máscaras de cancha de una predicción YOLO-seg -> binaria."""
    if res.masks is None or len(res.masks) == 0:
        return None
    m = res.masks.data.cpu().numpy()           # (n, h, w) en 0..1
    mask = (m.sum(axis=0) > 0.5).astype(np.uint8)
    return cv2.resize(mask, (frame_shape[1], frame_shape[0]),
                      interpolation=cv2.INTER_NEAREST)


def mask_quality(mask, frame_shape):
    """
    Puntúa qué tan buena es la máscara para calibrar.
    Prefiere canchas grandes y CONVEXAS (sin huecos por jugadores que tapan
    las líneas). Devuelve (score, corners) o (0, None).
    """
    h, w = frame_shape[:2]
    cnts, _ = cv2.findContours((mask > 0).astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0, None
    cnt = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < MIN_AREA_FRAC * h * w:
        return 0.0, None
    corners = corners_from_mask(mask)
    if corners is None:
        return 0.0, None
    hull_area = cv2.contourArea(cv2.convexHull(cnt))
    solidity = area / hull_area if hull_area > 0 else 0.0
    quad_area = cv2.contourArea(corners)
    fit = min(area / quad_area, 1.0) if quad_area > 0 else 0.0
    area_frac = area / (h * w)
    score = solidity * fit * (0.5 + area_frac)   # convexa + bien ajustada + grande
    return float(score), corners


# ============================================================
#  HOMOGRAFÍA  (mismo formato/convención que court_keypoints)
# ============================================================
def homography_from_corners(corners, ppm=40, flip_vertical=False):
    """
    corners: 4 puntos imagen [TL,TR,BR,BL]. Devuelve homography_matrix (lista).
    flip_vertical=True si tu cámara tiene la zona CERCANA arriba en la imagen.
    """
    if corners is None or len(corners) < MIN_POINTS:
        return None
    meters = CORNER_METERS.copy()
    if flip_vertical:                      # intercambia cercana<->lejana
        meters = meters[[3, 2, 1, 0]]
    dst = (meters * ppm).astype(np.float32)
    src = np.asarray(corners, np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    if H is None:
        return None
    return H.tolist()


# ============================================================
#  QC OVERLAY  (proyecta la cancha de vuelta sobre el frame)
# ============================================================
def draw_qc_overlay(frame, H, ppm=40, court_size=(9, 18)):
    """Dibuja perímetro + red + líneas de ataque proyectadas (control de calidad)."""
    cw, ch = court_size
    Hinv = np.linalg.inv(np.array(H))

    def to_img(pts_m):
        pts = (np.array(pts_m, np.float32) * ppm).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, Hinv).reshape(-1, 2).astype(int)

    out = frame.copy()
    peri = to_img([[0, 0], [cw, 0], [cw, ch], [0, ch]])
    cv2.polylines(out, [peri.reshape(-1, 1, 2)], True, (0, 220, 0), 3)
    net = to_img([[0, ch / 2], [cw, ch / 2]])
    cv2.line(out, tuple(net[0]), tuple(net[1]), (90, 200, 255), 3)
    for ay in (ch / 2 - 3, ch / 2 + 3):     # líneas de ataque a 3 m de la red
        al = to_img([[0, ay], [cw, ay]])
        cv2.line(out, tuple(al[0]), tuple(al[1]), (0, 220, 0), 1)
    return out


# ============================================================
#  AUTO-CALIBRACIÓN
# ============================================================
def auto_calibrate_seg(video_path, model_path, conf=DEFAULT_CONF, ppm=40,
                       court_size_m=(9, 18), n_sample_frames=20,
                       flip_vertical=False, qc_path=None, verbose=True):
    """
    Muestrea frames, segmenta la cancha en cada uno, se queda con la máscara
    de mejor calidad (grande + convexa) y construye la homografía.

    Devuelve un dict tipo cal.json compatible con clara.py, o None si no se
    pudo calibrar (entonces clara.py cae a manual).
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo de cancha no encontrado: {model_path}")
    from ultralytics import YOLO   # import perezoso
    model = YOLO(str(model_path))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"No pude abrir video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    Hf = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if verbose:
        print(f"[seg-cal] Video {W}x{Hf}, muestreando {n_sample_frames} frames...")

    idxs = np.linspace(0, max(total - 1, 0), n_sample_frames, dtype=int)
    best_score, best = -1.0, None      # best = (frame_idx, corners, frame)
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok:
            continue
        res = model.predict(frame, conf=conf, verbose=False)[0]
        mask = court_mask_from_result(res, frame.shape)
        if mask is None:
            continue
        score, corners = mask_quality(mask, frame.shape)
        if score > best_score:
            best_score, best = score, (int(fi), corners, frame.copy())
    cap.release()

    if best is None or best_score <= 0:
        if verbose:
            print("[seg-cal] ✗ No se segmentó una cancha utilizable. "
                  "Usa calibración manual.")
        return None

    best_fi, corners, best_frame = best
    H = homography_from_corners(corners, ppm, flip_vertical)
    if H is None:
        if verbose:
            print("[seg-cal] ✗ No se pudo calcular homografía.")
        return None

    if verbose:
        print(f"[seg-cal] ✓ Calibrado con frame {best_fi} (score {best_score:.3f})")
        print(f"[seg-cal]   Esquinas (px): {corners.astype(int).tolist()}")

    if qc_path is not None:
        cv2.imwrite(str(qc_path), draw_qc_overlay(best_frame, H, ppm, court_size_m))
        if verbose:
            print(f"[seg-cal]   QC overlay guardado en {qc_path} — REVÍSALO.")

    return {
        "_comment": "Generado por court_segmentation.py (auto-cal por segmentacion, sin entrenar)",
        "video_reference": Path(video_path).name,
        "frame_shape": [Hf, W],
        "court_size_m": list(court_size_m),
        "pixels_per_meter": ppm,
        "homography_matrix": H,
        "half_court": False,
        "court_horizon_y": None,
        "max_person_height_ratio": 0.55,
        "max_person_width_ratio": 0.40,
        "auto_calibration": {
            "method": "segmentation",
            "source_frame": best_fi,
            "quality_score": round(best_score, 3),
            "corners_px": corners.astype(int).tolist(),
            "flip_vertical": flip_vertical,
        },
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Auto-calibración de cancha por SEGMENTACIÓN (sin entrenar)")
    p.add_argument("video")
    p.add_argument("--model", required=True,
                   help="court/weights/best.pt de volleyball_analytics")
    p.add_argument("--out", default="cal_auto.json")
    p.add_argument("--qc", default="cal_check.jpg",
                   help="ruta del overlay de control de calidad")
    p.add_argument("--conf", type=float, default=DEFAULT_CONF)
    p.add_argument("--ppm", type=int, default=40)
    p.add_argument("--flip-vertical", action="store_true",
                   help="usa si la zona cercana queda ARRIBA en tu imagen")
    p.add_argument("--frames", type=int, default=20)
    a = p.parse_args()

    cal = auto_calibrate_seg(a.video, a.model, conf=a.conf, ppm=a.ppm,
                             n_sample_frames=a.frames,
                             flip_vertical=a.flip_vertical, qc_path=a.qc)
    if cal is None:
        print("\n✗ Auto-calibración falló. Usa calibración manual.")
        raise SystemExit(1)
    Path(a.out).write_text(json.dumps(cal, indent=2))
    print(f"\n✓ Calibración guardada en {a.out}")
    print("  Revisa el QC overlay antes de confiar en ella.")
