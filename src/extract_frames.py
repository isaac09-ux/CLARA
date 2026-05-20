"""
CLARA · Extracción de frames para entrenar detector de balón.

Extrae frames del video largo (11 min) cada N segundos para etiquetado en Roboflow.

Uso:
    python extract_frames_for_training.py video_completo.mp4 --every 1.5
"""
import cv2
import argparse
from pathlib import Path


def extract(video_path: str, every_seconds: float = 1.5,
            output_dir: str = "frames_balon"):
    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps

    step = int(fps * every_seconds)
    print(f"[•] Video: {duration:.1f}s · {total} frames · {fps:.1f}fps")
    print(f"[•] Extrayendo cada {every_seconds}s (paso de {step} frames)")

    count = 0
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step == 0:
            ts = frame_idx / fps
            fname = out / f"frame_{count:04d}_t{ts:06.1f}s.jpg"
            cv2.imwrite(str(fname), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            count += 1
        frame_idx += 1

    cap.release()
    print(f"[✓] {count} frames extraídos en {out}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--every", type=float, default=1.5,
                   help="Segundos entre frames (default 1.5)")
    p.add_argument("--out", default="frames_balon")
    args = p.parse_args()
    extract(args.video, args.every, args.out)
