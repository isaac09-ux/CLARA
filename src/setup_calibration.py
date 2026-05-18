"""
Asistente interactivo de calibración para CLARA.

Uso:
    python setup_calibration.py video.mp4 --out my_calibration.json

Pide que hagas clic en las 4 esquinas de la cancha (o media cancha).
Convención: cercana_izq, cercana_der, lejana_der, lejana_izq.
"""
import cv2, numpy as np, json, argparse
from pathlib import Path

CORNER_NAMES = ["cercana_izq", "cercana_der", "lejana_der", "lejana_izq"]

def calibrate(video_path, out_path, half_court=False, frame_idx=None):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx or total // 2)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"No pude leer frame de {video_path}")

    points = []

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            print(f"  {CORNER_NAMES[len(points)-1]}: ({x}, {y})")

    cv2.namedWindow("Calibracion CLARA")
    cv2.setMouseCallback("Calibracion CLARA", on_click)

    print(f"\nHaz clic en las 4 esquinas en orden:")
    for n in CORNER_NAMES:
        print(f"  - {n}")
    print("\nPresiona 'r' para reiniciar, 'q' para terminar.")

    while True:
        vis = frame.copy()
        for i, (x, y) in enumerate(points):
            cv2.circle(vis, (x, y), 6, (40, 40, 255), -1)
            cv2.putText(vis, CORNER_NAMES[i][:4], (x+8, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 255), 1)
        if len(points) > 1:
            for i in range(len(points)):
                cv2.line(vis, points[i], points[(i+1) % len(points)],
                        (0, 255, 255), 1)
        cv2.imshow("Calibracion CLARA", vis)
        k = cv2.waitKey(20) & 0xFF
        if k == ord('q') and len(points) == 4: break
        if k == ord('r'): points.clear(); print("[reiniciado]")
    cv2.destroyAllWindows()

    src = np.float32(points)
    court_w, court_h = (9, 9) if half_court else (9, 18)
    ppm = 40
    cw, ch = court_w * ppm, court_h * ppm
    dst = (np.float32([[0, 0], [cw, 0], [cw, ch], [0, ch]]) if half_court
           else np.float32([[cw, 0], [cw, ch], [0, ch], [0, 0]]))
    H, _ = cv2.findHomography(src, dst)

    cal = {
        "video_reference": Path(video_path).name,
        "frame_shape": list(frame.shape[:2]),
        "court_size_m": [court_w, court_h],
        "pixels_per_meter": ppm,
        "pixel_corners": src.astype(int).tolist(),
        "homography_matrix": H.tolist(),
        "half_court": half_court,
    }
    Path(out_path).write_text(json.dumps(cal, indent=2))
    print(f"\n[✓] Calibración guardada en {out_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--out", required=True)
    p.add_argument("--half-court", action="store_true")
    p.add_argument("--frame", type=int, default=None)
    a = p.parse_args()
    calibrate(a.video, a.out, a.half_court, a.frame)
