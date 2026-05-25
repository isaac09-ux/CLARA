"""
court_keypoints.py — Auto-calibración de CLARA

Detecta los puntos de referencia de la cancha con un modelo YOLOv8-pose
entrenado, y calcula la homografía automáticamente. Reemplaza la
calibración manual de MIRA cuando el modelo detecta con confianza.

ESQUEMA DE 6 PUNTOS (Opción 2):

        P3 ───────────────── P2      <- LEJANAS (fondo, lejos de camara)
        │                     │
       C_izq ─────────────── C_der   <- LINEA CENTRAL (casi siempre visible)
        │                     │
        P0 ───────────────── P1      <- CERCANAS (cerca de camara)
                [ CAMARA ]

Coordenadas reales en cancha de 9 x 18 m (x, y) en metros:
    P0  cercana-izq   = (0, 18)
    P1  cercana-der   = (9, 18)
    P2  lejana-der    = (9, 0)
    P3  lejana-izq    = (0, 0)
    C_izq central-izq = (0, 9)
    C_der central-der = (9, 9)

El modelo puede entrenarse con los 6 puntos. La homografia necesita
minimo 4 detectados con confianza; si hay 5-6 usa todos (mas robusto).

IMPORTANTE: el orden de keypoints del modelo entrenado DEBE coincidir
con COURT_POINTS_ORDER de abajo. Si entrenaste con otro orden, ajustalo.
"""
import cv2
import numpy as np
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError as e:
    raise ImportError("ultralytics no instalado. pip install ultralytics") from e


# ============================================================
#  ESQUEMA DE PUNTOS
# ============================================================
# Orden de los keypoints tal como salen del modelo entrenado.
# El indice en esta lista = indice del keypoint en el modelo.
COURT_POINTS_ORDER = ["P0", "P1", "P2", "P3", "C_izq", "C_der"]

# Coordenada real en cancha (metros) de cada punto.
COURT_POINTS_METERS = {
    "P0":    (0.0, 18.0),   # cercana izquierda
    "P1":    (9.0, 18.0),   # cercana derecha
    "P2":    (9.0, 0.0),    # lejana derecha
    "P3":    (0.0, 0.0),    # lejana izquierda
    "C_izq": (0.0, 9.0),    # central izquierda
    "C_der": (9.0, 9.0),    # central derecha
}

# Confianza minima para aceptar un punto detectado.
DEFAULT_KP_CONF = 0.5
# Minimo de puntos validos para calcular homografia.
MIN_POINTS = 4


# ============================================================
#  DETECCIÓN
# ============================================================
def detect_court_points(frame, model, kp_conf=DEFAULT_KP_CONF):
    """
    Corre el modelo de cancha sobre un frame.

    Returns: dict {nombre_punto: (x_px, y_px, confianza)} solo con
    los puntos cuya confianza supera kp_conf. Puede devolver < 6.
    """
    res = model(frame, verbose=False)[0]
    if res.keypoints is None or len(res.keypoints.data) == 0:
        return {}

    # Tomar la deteccion de cancha con mayor confianza de caja
    best_idx = 0
    if res.boxes is not None and len(res.boxes) > 1:
        best_idx = int(np.argmax(res.boxes.conf.cpu().numpy()))

    kp_xy = res.keypoints.xy.cpu().numpy()[best_idx]
    kp_cf = (res.keypoints.conf.cpu().numpy()[best_idx]
             if res.keypoints.conf is not None
             else np.ones(len(kp_xy)))

    points = {}
    for i, (xy, cf) in enumerate(zip(kp_xy, kp_cf)):
        if i >= len(COURT_POINTS_ORDER):
            break
        if cf >= kp_conf:
            name = COURT_POINTS_ORDER[i]
            points[name] = (float(xy[0]), float(xy[1]), float(cf))
    return points


# ============================================================
#  HOMOGRAFÍA
# ============================================================
def homography_from_points(points, ppm=40):
    """
    Calcula la homografia a partir de los puntos detectados.

    Args:
        points: dict {nombre: (x_px, y_px, conf)} de detect_court_points
        ppm: pixels_per_meter para el topdown

    Returns:
        dict con homography_matrix y metadata, o None si no hay
        suficientes puntos.
    """
    if len(points) < MIN_POINTS:
        return None

    src = []   # puntos en pixeles del frame
    dst = []   # puntos en pixeles del topdown (metros * ppm)
    used = []
    for name, (px, py, cf) in points.items():
        mx, my = COURT_POINTS_METERS[name]
        src.append([px, py])
        dst.append([mx * ppm, my * ppm])
        used.append(name)

    src = np.array(src, dtype=np.float32)
    dst = np.array(dst, dtype=np.float32)

    if len(src) == 4:
        H = cv2.getPerspectiveTransform(src, dst)
    else:
        # 5-6 puntos: minimos cuadrados, mas robusto
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

    if H is None:
        return None

    return {
        "homography_matrix": H.tolist(),
        "points_used": used,
        "n_points": len(used),
        "avg_confidence": round(
            float(np.mean([points[n][2] for n in used])), 3),
    }


# ============================================================
#  AUTO-CALIBRACIÓN
# ============================================================
def auto_calibrate(video_path, model_path, kp_conf=DEFAULT_KP_CONF,
                   ppm=40, court_size_m=(9, 18), n_sample_frames=15,
                   verbose=True):
    """
    Auto-calibra un video. Muestrea varios frames, detecta la cancha en
    cada uno, y se queda con la mejor deteccion (mas puntos + mas conf).

    Returns: dict tipo cal.json listo para CLARA, o None si no se pudo
    calibrar con confianza (entonces el usuario debe usar MIRA).
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo de cancha no encontrado: {model_path}")

    model = YOLO(str(model_path))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"No pude abrir video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if verbose:
        print(f"[auto-cal] Video {W}x{H_frame}, muestreando {n_sample_frames} frames...")

    # Muestrear frames distribuidos por todo el video
    sample_idxs = np.linspace(0, total - 1, n_sample_frames, dtype=int)
    best = None
    best_score = -1

    for fi in sample_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok:
            continue
        points = detect_court_points(frame, model, kp_conf)
        if len(points) < MIN_POINTS:
            continue
        # score = numero de puntos + confianza promedio
        avg_cf = np.mean([p[2] for p in points.values()])
        score = len(points) + avg_cf
        if score > best_score:
            best_score = score
            best = (int(fi), points)

    cap.release()

    if best is None:
        if verbose:
            print(f"[auto-cal] ✗ No se detecto la cancha con confianza "
                  f">= {kp_conf} en ningun frame.")
            print(f"[auto-cal]   Usa calibracion manual (MIRA).")
        return None

    best_fi, best_points = best
    homo = homography_from_points(best_points, ppm)
    if homo is None:
        if verbose:
            print(f"[auto-cal] ✗ No se pudo calcular homografia.")
        return None

    if verbose:
        print(f"[auto-cal] ✓ Calibrado con frame {best_fi}")
        print(f"[auto-cal]   Puntos usados: {homo['points_used']}")
        print(f"[auto-cal]   Confianza media: {homo['avg_confidence']}")

    # Armar cal.json compatible con CLARA
    cal = {
        "_comment": "Generado por court_keypoints.py (auto-calibracion CLARA)",
        "video_reference": Path(video_path).name,
        "frame_shape": [H_frame, W],
        "court_size_m": list(court_size_m),
        "pixels_per_meter": ppm,
        "homography_matrix": homo["homography_matrix"],
        "half_court": False,
        "court_horizon_y": None,
        "max_person_height_ratio": 0.55,
        "max_person_width_ratio": 0.40,
        "auto_calibration": {
            "source_frame": best_fi,
            "points_used": homo["points_used"],
            "n_points": homo["n_points"],
            "avg_confidence": homo["avg_confidence"],
        },
    }
    return cal


if __name__ == "__main__":
    import argparse
    import json

    p = argparse.ArgumentParser(
        description="Auto-calibracion de cancha para CLARA")
    p.add_argument("video")
    p.add_argument("--model", required=True,
                   help="Modelo YOLOv8-pose de cancha (.pt)")
    p.add_argument("--out", default="cal_auto.json")
    p.add_argument("--kp-conf", type=float, default=DEFAULT_KP_CONF,
                   help="Confianza minima por keypoint (default 0.5)")
    p.add_argument("--ppm", type=int, default=40)
    a = p.parse_args()

    cal = auto_calibrate(a.video, a.model, kp_conf=a.kp_conf, ppm=a.ppm)
    if cal is None:
        print("\n✗ Auto-calibracion fallo. Usa MIRA para calibrar manual.")
        raise SystemExit(1)

    Path(a.out).write_text(json.dumps(cal, indent=2))
    print(f"\n✓ Calibracion guardada en {a.out}")
