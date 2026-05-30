"""
calibrate_assisted.py — Calibración ASISTIDA para CLARA (auto-seg + ajuste manual)
==================================================================================

La segmentación encuentra las 4 esquinas aproximadas de la cancha; tú solo
ARRASTRAS las que salieron mal hasta que el contorno proyectado pegue con las
líneas reales. La máquina hace el 80%, tú el 20%. Cero entrenamiento.

Producto: cal.json compatible con clara.py, MISMA convención que
court_segmentation.py (cercana abajo -> y alto; lejana arriba -> y=0).

Flujo:
  - Toma las esquinas iniciales del modelo de segmentación (court/best.pt).
  - Si la segmentación falla, empiezas desde 4 esquinas por defecto.
  - Ventana interactiva: arrastras esquinas, ves el overlay actualizarse.
  - Guardas -> escribe cal.json + cal_check.jpg.

Uso:
  python calibrate_assisted.py partido.mp4 --court-seg-model court_best.pt \
      --out cal_assisted.json
  python clara.py partido.mp4 --calibration cal_assisted.json --ball-model ...

Nota: requiere entorno con GUI (cv2.imshow) — es un paso local, no de Colab.
Teclas:  s = guardar   r = reset a las esquinas auto   q = cancelar
"""

import json
import argparse
from pathlib import Path

import cv2
import numpy as np

# Reutiliza la lógica ya validada de court_segmentation.py
from court_segmentation import (court_mask_from_result, mask_quality)

DEFAULT_PPM = 40
GRAB_RADIUS = 28          # px (coords originales) para agarrar una esquina
# Etiquetas en orden [TL, TR, BR, BL] (misma convención que CORNER_METERS)
LABELS = ["1 lejana-izq", "2 lejana-der", "3 cercana-der", "4 cercana-izq"]


# ============================================================
#  GEOMETRÍA
# ============================================================
def build_homography(corners, court_size, ppm):
    """corners [TL,TR,BR,BL] -> H (imagen px -> topdown px = metros*ppm)."""
    cw, ch = court_size
    meters = np.float32([[0, 0], [cw, 0], [cw, ch], [0, ch]])
    dst = (meters * ppm).astype(np.float32)
    src = np.asarray(corners, np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def draw_overlay(frame, corners, court_size, ppm, half, active=-1):
    """Cancha proyectada + handles sobre copia del frame (resolución original)."""
    out = frame.copy()
    cw, ch = court_size
    try:                                   # la homografía puede fallar a mitad de arrastre
        Hinv = np.linalg.inv(build_homography(corners, court_size, ppm))

        def to_img(pts_m):
            p = (np.array(pts_m, np.float32) * ppm).reshape(-1, 1, 2)
            return cv2.perspectiveTransform(p, Hinv).reshape(-1, 2).astype(int)

        peri = to_img([[0, 0], [cw, 0], [cw, ch], [0, ch]])
        cv2.polylines(out, [peri.reshape(-1, 1, 2)], True, (0, 220, 0), 2)
        if not half:                       # cancha completa: red al centro + líneas de ataque
            net = to_img([[0, ch / 2], [cw, ch / 2]])
            cv2.line(out, tuple(net[0]), tuple(net[1]), (90, 200, 255), 2)
            for ay in (ch / 2 - 3, ch / 2 + 3):
                al = to_img([[0, ay], [cw, ay]])
                cv2.line(out, tuple(al[0]), tuple(al[1]), (0, 180, 0), 1)
    except Exception:
        pass

    for i, (x, y) in enumerate(corners.astype(int)):
        col = (0, 0, 255) if i == active else (0, 255, 255)
        cv2.circle(out, (x, y), 11, col, -1)
        cv2.circle(out, (x, y), 11, (0, 0, 0), 1)
        cv2.putText(out, LABELS[i], (x + 14, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
    return out


def default_corners(w, h):
    """4 esquinas por defecto (rectángulo inset) en orden TL,TR,BR,BL."""
    return np.float32([[w * 0.30, h * 0.32], [w * 0.70, h * 0.32],
                       [w * 0.82, h * 0.85], [w * 0.18, h * 0.85]])


# ============================================================
#  EDITOR INTERACTIVO
# ============================================================
def edit_corners(frame, init_corners, court_size, ppm, half,
                 window="Calibracion asistida CLARA"):
    h, w = frame.shape[:2]
    corners = np.asarray(init_corners, np.float32).copy()
    scale = min(1.0, 1280.0 / w)           # escala de despliegue
    state = {"active": -1}

    def on_mouse(event, x, y, flags, param):
        ox, oy = x / scale, y / scale      # display -> original
        if event == cv2.EVENT_LBUTTONDOWN:
            d = np.hypot(corners[:, 0] - ox, corners[:, 1] - oy)
            j = int(np.argmin(d))
            if d[j] <= GRAB_RADIUS:
                state["active"] = j
        elif event == cv2.EVENT_MOUSEMOVE and state["active"] >= 0:
            corners[state["active"]] = [np.clip(ox, 0, w - 1), np.clip(oy, 0, h - 1)]
        elif event == cv2.EVENT_LBUTTONUP:
            state["active"] = -1

    cv2.namedWindow(window)                # AUTOSIZE: display px == array px
    cv2.setMouseCallback(window, on_mouse)
    while True:
        canvas = draw_overlay(frame, corners, court_size, ppm, half, state["active"])
        cv2.putText(canvas, "Arrastra las esquinas a las lineas reales",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(canvas, "s=guardar  r=reset auto  q=cancelar",
                    (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        disp = cv2.resize(canvas, (int(w * scale), int(h * scale))) if scale < 1 else canvas
        cv2.imshow(window, disp)
        k = cv2.waitKey(20) & 0xFF
        if k == ord('s'):
            cv2.destroyWindow(window)
            return corners
        if k == ord('r'):
            corners = np.asarray(init_corners, np.float32).copy()
        if k == ord('q'):
            cv2.destroyWindow(window)
            return None


# ============================================================
#  SELECCIÓN DE FRAME + ESQUINAS INICIALES
# ============================================================
def _seg_corners(frame, model, conf):
    res = model.predict(frame, conf=conf, verbose=False)[0]
    mask = court_mask_from_result(res, frame.shape)
    if mask is None:
        return None
    _, corners = mask_quality(mask, frame.shape)
    return corners


def pick_frame_and_corners(video, seg_model_path, conf, frame_idx, n_sample):
    """Devuelve (frame, corners). corners=None si no hubo segmentación."""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"No pude abrir el video: {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_idx is not None:              # frame específico
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise SystemExit("Frame fuera de rango.")
        if not seg_model_path:
            return frame, None
        from ultralytics import YOLO
        return frame, _seg_corners(frame, YOLO(seg_model_path), conf)

    if not seg_model_path:                 # sin modelo: primer frame
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise SystemExit("No pude leer el primer frame.")
        return frame, None

    from ultralytics import YOLO           # muestrear y elegir mejor frame por seg
    model = YOLO(seg_model_path)
    idxs = np.linspace(0, max(total - 1, 0), n_sample, dtype=int)
    best_score, best = -1.0, None
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
            best_score, best = score, (frame.copy(), corners)
    cap.release()
    return best if best is not None else (None, None)


# ============================================================
#  MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser(
        description="Calibración asistida CLARA (auto-segmentación + ajuste manual)")
    ap.add_argument("video")
    ap.add_argument("--court-seg-model",
                    help="court/best.pt para esquinas iniciales (opcional)")
    ap.add_argument("--out", default="cal_assisted.json")
    ap.add_argument("--qc", default="cal_check.jpg")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--ppm", type=int, default=DEFAULT_PPM)
    ap.add_argument("--frame", type=int, default=None,
                    help="índice de frame a calibrar (default: mejor por seg)")
    ap.add_argument("--frames", type=int, default=20,
                    help="frames a muestrear para elegir el mejor")
    ap.add_argument("--half", action="store_true",
                    help="media cancha 9x9 (default: cancha completa 9x18)")
    a = ap.parse_args()

    court_size = (9, 9) if a.half else (9, 18)

    frame, corners = pick_frame_and_corners(
        a.video, a.court_seg_model, a.conf, a.frame, a.frames)

    if frame is None:                      # seg no detectó nada en ningún frame
        cap = cv2.VideoCapture(a.video)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise SystemExit("No pude leer el video.")
        corners = None
        print("[asist] La segmentación no detectó cancha — empiezas desde defaults.")

    h, w = frame.shape[:2]
    if corners is None:
        corners = default_corners(w, h)
        print("[asist] Sin esquinas auto: arrastra las 4 a las esquinas reales.")
    else:
        print("[asist] Esquinas iniciales de la segmentación. Ajusta las que estén mal.")

    final = edit_corners(frame, corners, court_size, a.ppm, a.half)
    if final is None:
        print("Cancelado.")
        raise SystemExit(1)

    H = build_homography(final, court_size, a.ppm)
    cal = {
        "_comment": "Generado por calibrate_assisted.py (auto-seg + ajuste manual)",
        "video_reference": Path(a.video).name,
        "frame_shape": [h, w],
        "court_size_m": list(court_size),
        "pixels_per_meter": a.ppm,
        "homography_matrix": H.tolist(),
        # Top-level pixel_corners (orden TL,TR,BR,BL) — clara.py lo consume para
        # el filtro del balón EN PÍXELES (ball_valid_region -> modo pixel_polygon).
        # Sin esta clave, clara.py cae al fallback por proyección y pierde la
        # mayoría de las detecciones aéreas del balón. Mismas 4 esquinas que el
        # editor; el bbox del balón usa min/max, así que el orden no afecta.
        "pixel_corners": np.asarray(final).astype(int).tolist(),
        "half_court": bool(a.half),
        "court_horizon_y": None,
        "max_person_height_ratio": 0.55,
        "max_person_width_ratio": 0.40,
        "auto_calibration": {
            "method": "assisted_segmentation",
            "corners_px": np.asarray(final).astype(int).tolist(),
        },
    }
    Path(a.out).write_text(json.dumps(cal, indent=2))
    cv2.imwrite(a.qc, draw_overlay(frame, final, court_size, a.ppm, a.half))
    print(f"✓ Calibración guardada en {a.out}")
    print(f"✓ QC overlay en {a.qc} — confirma que el contorno pega.")
    print(f"  Uso: python clara.py {Path(a.video).name} --calibration {a.out} --ball-model ...")


if __name__ == "__main__":
    main()
