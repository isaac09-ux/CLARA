"""Shared fixtures for CLARA tests."""
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def full_court_calibration():
    """Real example calibration from calibration/example_full_court.json."""
    path = REPO_ROOT / "calibration" / "example_full_court.json"
    return json.loads(path.read_text())


@pytest.fixture
def half_court_calibration():
    path = REPO_ROOT / "calibration" / "example_half_court.json"
    return json.loads(path.read_text())


@pytest.fixture
def identity_homography():
    """A homography that's the identity in pixels-per-meter=1 space.

    Pixel (x, y) projects to court (x, y) in meters when ppm=1.
    Lets tests assert geometric correctness without dealing with real perspective.
    """
    return np.eye(3, dtype=np.float64)


@pytest.fixture
def fake_frame_hd():
    """A 720p BGR frame with a known gradient. Useful for preprocessing tests."""
    rng = np.random.default_rng(seed=42)
    return rng.integers(0, 256, size=(720, 1280, 3), dtype=np.uint8)


@pytest.fixture
def rejected_counts():
    return defaultdict(int)


@pytest.fixture
def standing_keypoints():
    """COCO-17 keypoints for a person standing upright, facing camera.

    Coords chosen so:
      - shoulder midpoint directly above hip midpoint (zero torso lean)
      - ankles 40 px apart horizontally (stance width = 40)
      - hip-knee-ankle vertically aligned (knee flexion ≈ 180°)
    """
    kp = [[0.0, 0.0]] * 17
    # head/face — not used by biomechanics, set to plausible values
    kp[0] = [100.0, 50.0]
    kp[1] = [95.0, 45.0]
    kp[2] = [105.0, 45.0]
    kp[3] = [90.0, 50.0]
    kp[4] = [110.0, 50.0]
    # shoulders (5=left, 6=right) at y=100
    kp[5] = [80.0, 100.0]
    kp[6] = [120.0, 100.0]
    # elbows / wrists — neutral by side
    kp[7] = [75.0, 140.0]
    kp[8] = [125.0, 140.0]
    kp[9] = [70.0, 180.0]
    kp[10] = [130.0, 180.0]
    # hips (11=left, 12=right) at y=200 — directly below shoulders
    kp[11] = [80.0, 200.0]
    kp[12] = [120.0, 200.0]
    # knees at y=270 — directly below hips
    kp[13] = [80.0, 270.0]
    kp[14] = [120.0, 270.0]
    # ankles at y=340 — directly below knees (legs straight)
    kp[15] = [80.0, 340.0]
    kp[16] = [120.0, 340.0]
    return kp


@pytest.fixture
def high_scores():
    """All 17 keypoints confident."""
    return [0.95] * 17


@pytest.fixture
def sample_metrics_json(tmp_path):
    """Minimal scouting_data.json for report tests."""
    data = {
        "clara_version": "0.6",
        "video": "test_match.mp4",
        "duration_min": 12.3,
        "duration_s": 738.0,
        "stride": 5,
        "samples_processed": 800,
        "court_size_m": [9, 18],
        "half_court": False,
        "ball_detector": "vballnet",
        "pose_mode": "none",
        "raw_tracks": 25,
        "filtered_tracks": 10,
        "rejected_detections": {"fg_too_large": 5},
        "ball_detections_oncourt": 42,
        "ball_frames_oncourt": 40,
        "ball_detection_rate": 0.05,
        "quality_score": 72,
        "quality_breakdown": {
            "tracks": "25/30",
            "zonas": "20/25",
            "balon": "10/20",
            "filtrado": "12/15",
            "estabilidad": "5/10",
        },
        "zone_visits_total": {
            "A1": 30, "A2": 25, "A3": 40, "A4": 15, "A5": 10, "A6": 20,
            "B1": 18, "B2": 22, "B3": 35, "B4": 28, "B5": 12, "B6": 16,
        },
        "zone_visits_first_half": {"A3": 20, "B3": 15, "A1": 10},
        "zone_visits_second_half": {"A3": 20, "B3": 20, "A2": 15},
        "tracks": [
            {
                "id": 1, "samples": 200, "distance_m": 145.3,
                "avg_speed_m_per_s": 0.5, "side": "A",
                "dominant_zone": "A3",
                "avg_court_pos_m": [4.5, 5.0],
                "pose_stats": None,
            },
            {
                "id": 2, "samples": 180, "distance_m": 120.0,
                "avg_speed_m_per_s": 0.6, "side": "B",
                "dominant_zone": "B3",
                "avg_court_pos_m": [4.5, 13.0],
                "pose_stats": None,
            },
        ],
        "pose_stats": {},
    }
    json_path = tmp_path / "scouting_data.json"
    json_path.write_text(json.dumps(data, indent=2))
    return json_path


@pytest.fixture
def synthetic_topdown_png(tmp_path):
    """A tiny PNG file for embedding tests."""
    img = np.full((100, 200, 3), 30, dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (190, 90), (200, 200, 200), 2)
    out = tmp_path / "topdown.png"
    cv2.imwrite(str(out), img)
    return out
