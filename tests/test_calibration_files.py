"""Sanity tests for the example calibration JSONs shipped in calibration/.

These files are referenced in README quick-start and should always be valid
inputs for clara.run(...). We don't run the pipeline here — just validate
the schema and make sure the homography matrix is invertible.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from clara import project

CAL_DIR = Path(__file__).resolve().parent.parent / "calibration"


REQUIRED_KEYS = {
    "court_size_m",
    "pixels_per_meter",
    "pixel_corners",
    "homography_matrix",
    "half_court",
}


@pytest.mark.parametrize("filename", [
    "example_full_court.json",
    "example_half_court.json",
])
def test_calibration_has_required_keys(filename):
    cal = json.loads((CAL_DIR / filename).read_text())
    missing = REQUIRED_KEYS - cal.keys()
    assert not missing, f"{filename} missing keys: {missing}"


@pytest.mark.parametrize("filename,expected_size", [
    ("example_full_court.json", [9, 18]),
    ("example_half_court.json", [9, 9]),
])
def test_court_size_matches_mode(filename, expected_size):
    cal = json.loads((CAL_DIR / filename).read_text())
    assert cal["court_size_m"] == expected_size


@pytest.mark.parametrize("filename", [
    "example_full_court.json",
    "example_half_court.json",
])
def test_pixel_corners_are_four_points(filename):
    cal = json.loads((CAL_DIR / filename).read_text())
    corners = cal["pixel_corners"]
    assert len(corners) == 4
    for corner in corners:
        assert len(corner) == 2


@pytest.mark.parametrize("filename", [
    "example_full_court.json",
    "example_half_court.json",
])
def test_homography_is_invertible(filename):
    cal = json.loads((CAL_DIR / filename).read_text())
    H = np.array(cal["homography_matrix"])
    assert H.shape == (3, 3)
    det = np.linalg.det(H)
    assert abs(det) > 1e-9, "Homography matrix is singular"


def test_setup_calibration_round_trip_full_court():
    """Pin the contract that setup_calibration.py establishes via
    cv2.findHomography: a fresh H computed from 4 pixel corners projects
    those same corners onto the court corners (within sub-pixel tolerance).
    """
    import cv2
    # 4 pixel corners in setup_calibration order: cercana_izq, cercana_der,
    # lejana_der, lejana_izq
    pixel_corners = np.float32([
        [240, 620], [1085, 620], [1153, 437], [170, 437],
    ])
    court_w, court_h = 9, 18
    ppm = 30
    cw, ch = court_w * ppm, court_h * ppm
    # Full-court dst order (mirrors setup_calibration.py:60)
    dst = np.float32([[cw, 0], [cw, ch], [0, ch], [0, 0]])
    H, _ = cv2.findHomography(pixel_corners, dst)

    expected_court = [(c[0] / ppm, c[1] / ppm) for c in dst]
    for (px, py), (ex, ey) in zip(pixel_corners, expected_court):
        x, y = project(H, float(px), float(py))
        assert (x / ppm, y / ppm) == pytest.approx((ex, ey), abs=0.01)


def test_setup_calibration_round_trip_half_court():
    import cv2
    pixel_corners = np.float32([
        [245, 320], [1015, 320], [1070, 250], [200, 250],
    ])
    court_w, court_h = 9, 9
    ppm = 40
    cw, ch = court_w * ppm, court_h * ppm
    # Half-court dst order (mirrors setup_calibration.py:59)
    dst = np.float32([[0, 0], [cw, 0], [cw, ch], [0, ch]])
    H, _ = cv2.findHomography(pixel_corners, dst)

    expected_court = [(c[0] / ppm, c[1] / ppm) for c in dst]
    for (px, py), (ex, ey) in zip(pixel_corners, expected_court):
        x, y = project(H, float(px), float(py))
        assert (x / ppm, y / ppm) == pytest.approx((ex, ey), abs=0.01)
