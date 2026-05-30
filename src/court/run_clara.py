"""
run_clara.py  (CLI del subpaquete court/)
=========================================
CLI para correr la integración de cancha basada en volleyball_analytics.

⚠️  Este CLI usa `CLARACourtAnalyzer` (analyzer.py), que corre su PROPIO
detector+tracker y DUPLICA el pipeline principal de CLARA (`src/clara.py`).
Úsalo como herramienta ALTERNATIVA / de prototipado, no como el flujo de
producción. Para el pipeline robusto (ROI, stitch de tracks, vballnet,
trayectoria, pose) usa `python src/clara.py ...`.

Ejecútalo como módulo desde `src/`:
    cd src
    python -m court.run_clara --video partido.mp4 --mode manual --calib cancha.json

Ejemplos
--------
# 1) Calibrar a mano (trípode fijo), guardar y procesar:
python -m court.run_clara --video partido.mp4 --mode manual --calib cancha.json \
    --ball-model weights/ball/weights/best.pt

# 2) Reusar calibración existente (mismo encuadre):
python -m court.run_clara --video partido2.mp4 --mode load --calib cancha.json \
    --ball-model weights/ball/weights/best.pt

# 3) Auto-calibración con el modelo de cancha (cámara que cambia de ángulo):
python -m court.run_clara --video broadcast.mp4 --mode auto \
    --court-model weights/court/weights/best.pt \
    --ball-model weights/ball/weights/best.pt
"""

import os
import argparse

import cv2

from .court_calibration import CourtCalibrator
from .analyzer import CLARACourtAnalyzer


def first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"No pude leer el primer frame de {video_path}")
    return frame


def build_calibrator(args):
    if args.mode == "load":
        if not args.calib or not os.path.exists(args.calib):
            raise SystemExit("--mode load requiere un --calib existente.")
        print(f"[calib] Cargando {args.calib}")
        return CourtCalibrator.load(args.calib)

    frame = first_frame(args.video)
    cal = CourtCalibrator()

    if args.mode == "auto":
        if not args.court_model:
            raise SystemExit("--mode auto requiere --court-model (court/best.pt)")
        from ultralytics import YOLO
        print("[calib] Auto-calibración con modelo de cancha...")
        corners = cal.from_court_model(frame, YOLO(args.court_model))
        print("[calib] Esquinas detectadas (px):\n", corners)
    else:  # manual
        names = (["A_corner_left", "A_corner_right", "B_corner_right",
                  "B_corner_left", "net_left", "net_right"]
                 if args.six_points else None)
        print("[calib] Clic en los puntos en el orden indicado en la ventana.")
        cal.from_clicks(frame, landmark_names=names)

    if args.calib:
        cal.save(args.calib)
        print(f"[calib] Guardada en {args.calib}")

    # control de calidad: guarda un frame con la cancha proyectada encima
    qc = cal.draw_overlay(frame)
    cv2.imwrite("calib_check.jpg", qc)
    print("[calib] Revisa calib_check.jpg: el contorno debe pegar con las líneas.")
    return cal


def main():
    ap = argparse.ArgumentParser(description="CLARA - integración cancha / homografía")
    ap.add_argument("--video", required=True)
    ap.add_argument("--mode", choices=["manual", "auto", "load"], default="manual")
    ap.add_argument("--calib", help="ruta del JSON de calibración (guardar/cargar)")
    ap.add_argument("--court-model", help="ruta a court/weights/best.pt (modo auto)")
    ap.add_argument("--ball-model", help="ruta a ball/weights/best.pt")
    ap.add_argument("--player-model", default="yolov8x.pt",
                    help="modelo de jugadores (COCO person por defecto)")
    ap.add_argument("--six-points", action="store_true",
                    help="calibración manual con 6 puntos (4 esquinas + red)")
    ap.add_argument("--flip-lr", action="store_true",
                    help="invierte izquierda/derecha de las zonas")
    ap.add_argument("--every", type=int, default=1, help="procesa 1 de cada N frames")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--out", default="clara_annotated.mp4")
    ap.add_argument("--topdown", default="clara_topdown.mp4")
    ap.add_argument("--csv", default="clara_scouting.csv")
    ap.add_argument("--json", default="clara_scouting.json")
    ap.add_argument("--device", default=None, help="cuda:0 / cpu / mps")
    args = ap.parse_args()

    cal = build_calibrator(args)

    an = CLARACourtAnalyzer(
        cal,
        player_model_path=args.player_model,
        ball_model_path=args.ball_model,
        flip_lr=args.flip_lr,
        device=args.device,
    )

    print("[run] Procesando video...")
    an.process_video(args.video, out_video=args.out, topdown_video=args.topdown,
                     show=False, every=args.every, max_frames=args.max_frames)

    an.export_csv(args.csv)
    an.export_json(args.json)

    print("\n[done]")
    print("  video anotado :", args.out)
    print("  top-down      :", args.topdown)
    print("  scouting CSV  :", args.csv)
    print("  scouting JSON :", args.json)

    occ = an.zone_occupancy()
    if occ:
        print("\n[ocupación de zona por jugador] (frames por zona)")
        for tid, zones in sorted(occ.items()):
            top = sorted(zones.items(), key=lambda kv: -kv[1])
            resumen = "  ".join(f"Z{z}:{n}" for z, n in top)
            print(f"  #{tid}: {resumen}")


if __name__ == "__main__":
    main()
