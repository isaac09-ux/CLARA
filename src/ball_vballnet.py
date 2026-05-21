"""
VballNet ball detector wrapper for CLARA.

Adapter around asigatchov/fast-volleyball-tracking-inference's VballNet ONNX model.
Reads video, returns list of (frame_idx, x, y, radius) ball detections.

Architecture: TrackNetV4-based, takes 9 consecutive grayscale frames as input,
outputs heatmaps for ball positions. Motion-aware, ignores static objects
(ceiling lights, banners) that confuse single-frame detectors.

Pretrained model file: VballNetV1_seq9_grayscale_330_h288_w512.onnx
Download from: https://github.com/asigatchov/fast-volleyball-tracking-inference

Usage:
    from ball_vballnet import detect_balls
    detections = detect_balls("game.mp4", "VballNet.onnx")
    # detections: list of {"frame": int, "x": float, "y": float, "radius": float}
"""
import cv2
import numpy as np
from pathlib import Path


# Constantes del modelo VballNet preentrenado
SEQ_LEN = 9          # frames consecutivos en input
MODEL_H = 288        # altura del input al modelo
MODEL_W = 512        # ancho del input
HEATMAP_THRESHOLD = 0.5  # umbral de heatmap para detectar balón


def _preprocess_frame(frame_bgr):
    """Convierte un frame BGR a grayscale normalizado y reescalado al input del modelo."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (MODEL_W, MODEL_H), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


def _postprocess_heatmap(heatmap, threshold=HEATMAP_THRESHOLD,
                          orig_w=None, orig_h=None):
    """
    Convierte un heatmap del modelo en (x, y, radius) en coords originales.
    Returns None si no hay detección sobre el umbral.
    """
    if heatmap.max() < threshold:
        return None

    # Centroide del cluster más fuerte
    y_idx, x_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)

    # Estimar radio basado en cuántos píxeles superan threshold * 0.5
    mask = heatmap > (threshold * 0.5)
    if mask.sum() > 0:
        # Aproximamos el radio asumiendo un blob ~circular
        radius_model = max(2.0, np.sqrt(mask.sum() / np.pi))
    else:
        radius_model = 3.0

    # Reescalar a coords originales si se proveen
    if orig_w is not None and orig_h is not None:
        x = float(x_idx) * orig_w / MODEL_W
        y = float(y_idx) * orig_h / MODEL_H
        # El radio escala proporcionalmente al ancho (asumiendo aspect ratio similar)
        radius = float(radius_model) * orig_w / MODEL_W
    else:
        x = float(x_idx)
        y = float(y_idx)
        radius = float(radius_model)

    return {"x": x, "y": y, "radius": radius, "confidence": float(heatmap.max())}


def detect_balls(video_path, model_path, threshold=HEATMAP_THRESHOLD,
                 stride=1, verbose=True):
    """
    Corre VballNet en un video y devuelve detecciones de balón frame por frame.

    Args:
        video_path: ruta al video .mp4
        model_path: ruta al modelo VballNet .onnx
        threshold: umbral del heatmap (0-1). Default 0.5.
        stride: procesar 1 de cada N frames (default 1 = todos).
                Nota: VballNet requiere secuencias consecutivas, así que stride
                solo afecta cuántos frames se EVALÚAN, no la ventana de input.
        verbose: imprimir progreso.

    Returns:
        List of {"frame": int, "x": float, "y": float, "radius": float,
                 "confidence": float}
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo VballNet no encontrado en {model_path}. "
            f"Descárgalo de https://github.com/asigatchov/fast-volleyball-tracking-inference"
        )

    # Cargar modelo
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise ImportError(
            "onnxruntime no instalado. Instala con: pip install onnxruntime"
        ) from e
    providers = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers.insert(0, "CUDAExecutionProvider")
    session = ort.InferenceSession(str(model_path), providers=providers)
    input_name = session.get_inputs()[0].name

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"No pude abrir video: {video_path}")

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if verbose:
        print(f"[VballNet] Video: {orig_w}x{orig_h}, {total} frames")
        print(f"[VballNet] Modelo: {model_path.name}")

    detections = []
    # Buffer circular de SEQ_LEN frames preprocesados
    buffer = []
    frame_idx = 0
    evaluated = 0  # nº de frames donde corrimos inferencia (denominador correcto del rate)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        proc = _preprocess_frame(frame)
        buffer.append(proc)

        # Mantener solo los últimos SEQ_LEN frames
        if len(buffer) > SEQ_LEN:
            buffer.pop(0)

        # Solo correr inferencia cuando tengamos suficientes frames.
        # Stride se aplica al frame CENTRAL (el que se reporta) para que las
        # detecciones de balón caigan en los mismos índices que el stream de
        # personas (ultralytics vid_stride=N emite frames en 0, N, 2N, ...).
        center = SEQ_LEN // 2
        center_frame_idx = frame_idx - center
        if (len(buffer) == SEQ_LEN
                and center_frame_idx >= 0
                and center_frame_idx % stride == 0):
            # Stack en formato (1, SEQ_LEN, H, W) - channels-first
            input_tensor = np.stack(buffer, axis=0)[np.newaxis, ...]
            output = session.run(None, {input_name: input_tensor})[0]
            evaluated += 1

            # output shape: (1, SEQ_LEN, H, W) — un heatmap por cada frame de la
            # secuencia. El frame central tiene contexto pasado y futuro
            # balanceado; los extremos sólo ven medio contexto y dan peor recall.
            heatmap = output[0, center]

            det = _postprocess_heatmap(heatmap, threshold, orig_w, orig_h)
            if det is not None:
                det["frame"] = center_frame_idx
                detections.append(det)

        frame_idx += 1
        if verbose and frame_idx % 500 == 0:
            print(f"[VballNet] frame {frame_idx}/{total} | "
                  f"detecciones: {len(detections)}")

    cap.release()

    if verbose:
        rate = len(detections) / max(evaluated, 1) * 100
        print(f"[VballNet] ✓ {len(detections)} balones / {evaluated} frames "
              f"evaluados ({rate:.1f}%) — stride={stride}, video={frame_idx} frames")

    return detections


def load_balls_from_csv(csv_path):
    """
    Lee un ball.csv generado por el repo asigatchov directamente.
    Útil si ya pre-procesaste el video por separado.

    Formato esperado: Frame,Visibility,X,Y,Radius
    """
    import csv
    detections = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["Visibility"]) == 1:
                detections.append({
                    "frame": int(row["Frame"]),
                    "x": float(row["X"]),
                    "y": float(row["Y"]),
                    "radius": float(row["Radius"]),
                    "confidence": 1.0,
                })
    return detections


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="VballNet ball detector standalone")
    p.add_argument("video")
    p.add_argument("--model", required=True, help="Ruta al .onnx de VballNet")
    p.add_argument("--out", default="vballnet_detections.csv",
                   help="CSV de salida")
    p.add_argument("--threshold", type=float, default=HEATMAP_THRESHOLD)
    a = p.parse_args()

    dets = detect_balls(a.video, a.model, threshold=a.threshold)

    import csv
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "x", "y", "radius", "confidence"])
        w.writeheader()
        w.writerows(dets)
    print(f"✓ Guardado en {a.out}")
