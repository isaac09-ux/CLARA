"""
Asistente de calibración para CLARA.

Interactivo (clic en 4 esquinas, abre ventana):
    python setup_calibration.py video.mp4 --out cal.json [--half-court] [--frame N]

No interactivo (esquinas por CLI — para AUTOMATIZAR, sin GUI):
    python setup_calibration.py video.mp4 --out cal.json --half-court \
        --corners "x1,y1 x2,y2 x3,y3 x4,y4" [--frame N] [--qc qc.jpg]

Convención de esquinas (orden): cercana_izq, cercana_der, lejana_der, lejana_izq.
Con --qc se guarda un overlay de la cancha proyectada para verificar la
calibración SIN abrir ninguna ventana (útil en pipelines automáticos).
"""
import sys, cv2, numpy as np, json, argparse
from pathlib import Path

# Consolas Windows (cp1252) revientan con UnicodeEncodeError al imprimir ✓/•,
# aunque el JSON ya se haya escrito. Forzar utf-8 (mismo arreglo que clara.py).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CORNER_NAMES = ["cercana_izq", "cercana_der", "lejana_der", "lejana_izq"]


def _read_frame(video_path, frame_idx=None):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES,
            frame_idx if frame_idx is not None else total // 2)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"No pude leer frame de {video_path}")
    return frame


def _pick_points_gui(frame):
    points = []

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            print(f"  {CORNER_NAMES[len(points)-1]}: ({x}, {y})")

    cv2.namedWindow("Calibracion CLARA")
    cv2.setMouseCallback("Calibracion CLARA", on_click)
    print("\nHaz clic en las 4 esquinas en orden:")
    for n in CORNER_NAMES:
        print(f"  - {n}")
    print("\nPresiona 'r' para reiniciar, 'q' para terminar.")
    while True:
        vis = frame.copy()
        for i, (x, y) in enumerate(points):
            cv2.circle(vis, (x, y), 6, (40, 40, 255), -1)
            cv2.putText(vis, CORNER_NAMES[i][:4], (x + 8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 255), 1)
        if len(points) > 1:
            for i in range(len(points)):
                cv2.line(vis, points[i], points[(i + 1) % len(points)],
                         (0, 255, 255), 1)
        cv2.imshow("Calibracion CLARA", vis)
        k = cv2.waitKey(20) & 0xFF
        if k == ord('q') and len(points) == 4:
            break
        if k == ord('r'):
            points.clear()
            print("[reiniciado]")
    cv2.destroyAllWindows()
    return points


def _homography(points, half_court):
    src = np.float32(points)
    court_w, court_h = (9, 9) if half_court else (9, 18)
    ppm = 40
    cw, ch = court_w * ppm, court_h * ppm
    dst = (np.float32([[0, 0], [cw, 0], [cw, ch], [0, ch]]) if half_court
           else np.float32([[cw, 0], [cw, ch], [0, ch], [0, 0]]))
    H, _ = cv2.findHomography(src, dst)
    return H, court_w, court_h, ppm


def _qc_overlay(frame, H, court_w, court_h, ppm, out_path):
    """Proyecta el perímetro de la cancha (+ línea de ataque) de vuelta al
    frame para verificar la homografía sin abrir una ventana."""
    Hinv = np.linalg.inv(H)

    def to_img(pts_m):
        pts = (np.array(pts_m, np.float32) * ppm).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, Hinv).reshape(-1, 2).astype(int)

    vis = frame.copy()
    peri = to_img([[0, 0], [court_w, 0], [court_w, court_h], [0, court_h]])
    cv2.polylines(vis, [peri.reshape(-1, 1, 2)], True, (0, 220, 0), 2)
    # línea de ataque a 3 m de la red (la red es el borde lejano, y=court_h)
    al = to_img([[0, court_h - 3], [court_w, court_h - 3]])
    cv2.line(vis, tuple(al[0]), tuple(al[1]), (0, 220, 220), 1)
    for i, pt in enumerate(peri):
        cv2.circle(vis, tuple(pt), 5, (40, 40, 255), -1)
        cv2.putText(vis, CORNER_NAMES[i][:4], (pt[0] + 6, pt[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 255), 1)
    cv2.imwrite(str(out_path), vis)
    print(f"[qc] overlay guardado en {out_path}")


def calibrate(video_path, out_path, half_court=False, frame_idx=None,
              points=None, qc_path=None):
    frame = _read_frame(video_path, frame_idx)
    if points is None:
        points = _pick_points_gui(frame)
    if len(points) != 4:
        raise ValueError("Se requieren exactamente 4 esquinas")
    H, court_w, court_h, ppm = _homography(points, half_court)
    cal = {
        "video_reference": Path(video_path).name,
        "frame_shape": list(frame.shape[:2]),
        "court_size_m": [court_w, court_h],
        "pixels_per_meter": ppm,
        "pixel_corners": np.float32(points).astype(int).tolist(),
        "homography_matrix": H.tolist(),
        "half_court": half_court,
    }
    Path(out_path).write_text(json.dumps(cal, indent=2))
    print(f"\n[✓] Calibración guardada en {out_path}")
    if qc_path is not None:
        _qc_overlay(frame, H, court_w, court_h, ppm, qc_path)
    return cal


def _parse_corners(s):
    pts = []
    for pair in s.replace(";", " ").split():
        x, y = pair.split(",")
        pts.append((float(x), float(y)))
    if len(pts) != 4:
        raise argparse.ArgumentTypeError("Se requieren 4 pares x,y")
    return pts


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--out", required=True)
    p.add_argument("--half-court", action="store_true")
    p.add_argument("--frame", type=int, default=None)
    p.add_argument("--corners", type=_parse_corners, default=None,
                   help='No interactivo: "x1,y1 x2,y2 x3,y3 x4,y4" en orden '
                        'cercana_izq cercana_der lejana_der lejana_izq')
    p.add_argument("--qc", default=None,
                   help="Ruta para guardar overlay de verificación (sin GUI)")
    a = p.parse_args()
    calibrate(a.video, a.out, a.half_court, a.frame,
              points=a.corners, qc_path=a.qc)
