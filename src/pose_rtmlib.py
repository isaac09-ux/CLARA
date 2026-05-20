"""
RTMPose wrapper for CLARA — body keypoints per tracked player.

Uses Tau-J/rtmlib (https://github.com/Tau-J/rtmlib): super lightweight pose
estimation based on RTMPose models. Apache 2.0.

Unlike YOLOv8-pose, RTMPose runs entirely via ONNX runtime without mmcv/mmpose
dependencies. CPU-friendly.

Body keypoints output: 17 COCO keypoints
  0=nose, 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear,
  5=left_shoulder, 6=right_shoulder, 7=left_elbow, 8=right_elbow,
  9=left_wrist, 10=right_wrist, 11=left_hip, 12=right_hip,
  13=left_knee, 14=right_knee, 15=left_ankle, 16=right_ankle

For CLARA we mainly use shoulders (5,6), wrists (9,10), hips (11,12),
knees (13,14), ankles (15,16) for biomechanical analysis:
  - Spike approach angle: torso lean = angle between shoulders and hips
  - Defensive stance: ankle-knee-hip alignment, base width
  - Set arm position: wrist relative to shoulders
"""
import cv2
import numpy as np

try:
    from rtmlib import RTMPose
except ImportError as e:
    raise ImportError(
        "rtmlib no instalado. Instala con: pip install rtmlib"
    ) from e


# COCO 17 keypoint names (índice → nombre)
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Pares de skeleton para visualización
SKELETON_PAIRS = [
    (5, 7), (7, 9), (6, 8), (8, 10),       # brazos
    (5, 11), (6, 12), (11, 12), (5, 6),    # torso
    (11, 13), (13, 15), (12, 14), (14, 16) # piernas
]


class PoseEstimator:
    """Wrapper de RTMPose con caché de modelo y configuración por modo."""

    def __init__(self, mode="balanced", device="cpu", backend="onnxruntime"):
        """
        mode: 'performance' (más preciso, más lento)
              'balanced' (default — recomendado)
              'lightweight' (más rápido, menos preciso)
        device: 'cpu', 'cuda', 'mps'
        """
        print(f"[RTMPose] Cargando modelo (mode={mode}, device={device})...")
        self.pose_model = RTMPose(
            onnx_model="https://download.openmmlab.com/mmpose/v1/projects/"
                       "rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_"
                       "420e-256x192-e48f03d0_20230504.zip",
            model_input_size=(192, 256),
            backend=backend,
            device=device,
        )

    def estimate_for_bbox(self, frame, bbox):
        """
        Estima keypoints para una persona dentro de un bbox.

        Args:
            frame: imagen BGR completa
            bbox: [x1, y1, x2, y2] de la persona

        Returns:
            dict con:
              keypoints: list de [x, y] (17 puntos en coords del frame original)
              scores: list de confianza por keypoint (0-1)
              avg_score: confianza promedio
        """
        x1, y1, x2, y2 = bbox
        # RTMPose espera bbox en formato [x1, y1, x2, y2] como ndarray
        bboxes = np.array([[x1, y1, x2, y2]], dtype=np.float32)

        keypoints, scores = self.pose_model(frame, bboxes)
        # keypoints: (1, 17, 2), scores: (1, 17)
        kps = keypoints[0]
        scs = scores[0]

        return {
            "keypoints": kps.tolist(),
            "scores": scs.tolist(),
            "avg_score": float(scs.mean()),
        }

    def estimate_batch(self, frame, bboxes):
        """Estima keypoints para múltiples personas en un solo frame."""
        if len(bboxes) == 0:
            return []
        bboxes_arr = np.array(bboxes, dtype=np.float32)
        keypoints, scores = self.pose_model(frame, bboxes_arr)
        out = []
        for kps, scs in zip(keypoints, scores):
            out.append({
                "keypoints": kps.tolist(),
                "scores": scs.tolist(),
                "avg_score": float(scs.mean()),
            })
        return out


# ============================================================
#  ANÁLISIS BIOMECÁNICO BÁSICO
# ============================================================
def torso_lean_angle(keypoints, scores, min_score=0.5):
    """
    Ángulo de inclinación del torso. Positivo = inclinado hacia adelante.
    Útil para evaluar aproximación al remate.

    Returns None si los keypoints no son confiables.
    """
    L_SH, R_SH, L_HIP, R_HIP = 5, 6, 11, 12
    needed = [L_SH, R_SH, L_HIP, R_HIP]
    if any(scores[i] < min_score for i in needed):
        return None

    shoulder_mid = np.array([
        (keypoints[L_SH][0] + keypoints[R_SH][0]) / 2,
        (keypoints[L_SH][1] + keypoints[R_SH][1]) / 2,
    ])
    hip_mid = np.array([
        (keypoints[L_HIP][0] + keypoints[R_HIP][0]) / 2,
        (keypoints[L_HIP][1] + keypoints[R_HIP][1]) / 2,
    ])
    # Vector del torso (hip → shoulder)
    torso_vec = shoulder_mid - hip_mid
    # Ángulo respecto a vertical (0 = vertical, positivo = inclinado adelante)
    angle_rad = np.arctan2(torso_vec[0], -torso_vec[1])
    return float(np.degrees(angle_rad))


def stance_width(keypoints, scores, min_score=0.5):
    """
    Ancho de base (distancia entre tobillos) en píxeles.
    Indicador de stance defensiva. Returns None si keypoints débiles.
    """
    L_ANK, R_ANK = 15, 16
    if scores[L_ANK] < min_score or scores[R_ANK] < min_score:
        return None
    dx = keypoints[L_ANK][0] - keypoints[R_ANK][0]
    dy = keypoints[L_ANK][1] - keypoints[R_ANK][1]
    return float(np.hypot(dx, dy))


def knee_flexion(keypoints, scores, side="left", min_score=0.5):
    """
    Ángulo de flexión de rodilla (180 = pierna estirada, <180 = flexionada).
    Indicador de stance defensiva profunda.
    """
    if side == "left":
        hip, knee, ankle = 11, 13, 15
    else:
        hip, knee, ankle = 12, 14, 16
    if any(scores[i] < min_score for i in [hip, knee, ankle]):
        return None

    p_hip = np.array(keypoints[hip])
    p_knee = np.array(keypoints[knee])
    p_ankle = np.array(keypoints[ankle])

    v1 = p_hip - p_knee
    v2 = p_ankle - p_knee
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    cos = np.clip(cos, -1, 1)
    return float(np.degrees(np.arccos(cos)))


def draw_pose(frame, keypoints, scores, min_score=0.5,
              kp_color=(60, 220, 230), bone_color=(40, 40, 165)):
    """Dibuja los keypoints y skeleton en el frame para visualización."""
    h, w = frame.shape[:2]

    for a, b in SKELETON_PAIRS:
        if scores[a] >= min_score and scores[b] >= min_score:
            pa = (int(keypoints[a][0]), int(keypoints[a][1]))
            pb = (int(keypoints[b][0]), int(keypoints[b][1]))
            cv2.line(frame, pa, pb, bone_color, 2)

    for i, (kp, sc) in enumerate(zip(keypoints, scores)):
        if sc >= min_score:
            cv2.circle(frame, (int(kp[0]), int(kp[1])), 4, kp_color, -1)

    return frame


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Test RTMPose on a single image")
    p.add_argument("image")
    p.add_argument("--out", default="pose_test.jpg")
    a = p.parse_args()

    pose = PoseEstimator(mode="balanced", device="cpu")
    frame = cv2.imread(a.image)
    h, w = frame.shape[:2]
    # bbox = whole image
    result = pose.estimate_for_bbox(frame, [0, 0, w, h])
    print(f"Avg keypoint score: {result['avg_score']:.2f}")
    drawn = draw_pose(frame, result["keypoints"], result["scores"])
    cv2.imwrite(a.out, drawn)
    print(f"✓ Guardado en {a.out}")
