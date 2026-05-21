"""Tests for pure biomechanics functions in pose_rtmlib.py.

Covers torso_lean_angle, stance_width, knee_flexion, and draw_pose with
hand-crafted COCO-17 keypoints. No rtmlib/onnxruntime needed — these are
pure NumPy.
"""
import cv2
import numpy as np
import pytest

from pose_rtmlib import (
    SKELETON_PAIRS,
    draw_pose,
    knee_flexion,
    stance_width,
    torso_lean_angle,
)


# ---------- torso_lean_angle ----------

class TestTorsoLeanAngle:
    def test_upright_gives_zero(self, standing_keypoints, high_scores):
        # Shoulders directly above hips → 0° lean
        angle = torso_lean_angle(standing_keypoints, high_scores)
        assert angle == pytest.approx(0.0, abs=0.5)

    def test_lean_forward_positive(self, standing_keypoints, high_scores):
        # Move shoulders forward (right in image, +x) relative to hips
        kp = [list(p) for p in standing_keypoints]
        # shift shoulders +30 in x
        kp[5][0] += 30.0
        kp[6][0] += 30.0
        angle = torso_lean_angle(kp, high_scores)
        assert angle > 0
        # atan2(30, 100) ≈ 16.7°
        assert angle == pytest.approx(16.7, abs=0.5)

    def test_lean_backward_negative(self, standing_keypoints, high_scores):
        kp = [list(p) for p in standing_keypoints]
        kp[5][0] -= 30.0
        kp[6][0] -= 30.0
        angle = torso_lean_angle(kp, high_scores)
        assert angle < 0

    def test_returns_none_when_low_confidence(self, standing_keypoints):
        scores = [0.95] * 17
        scores[5] = 0.2  # left shoulder unconfident
        assert torso_lean_angle(standing_keypoints, scores) is None

    def test_returns_none_when_any_torso_keypoint_low(self, standing_keypoints):
        for idx in (5, 6, 11, 12):
            scores = [0.95] * 17
            scores[idx] = 0.1
            assert torso_lean_angle(standing_keypoints, scores) is None, (
                f"Should reject when keypoint {idx} has low confidence"
            )


# ---------- stance_width ----------

class TestStanceWidth:
    def test_known_width(self, standing_keypoints, high_scores):
        # Ankles at x=80 and x=120, same y → distance 40
        width = stance_width(standing_keypoints, high_scores)
        assert width == pytest.approx(40.0)

    def test_includes_vertical_offset(self, standing_keypoints, high_scores):
        # If one ankle is also offset in y, distance is the hypotenuse
        kp = [list(p) for p in standing_keypoints]
        kp[15] = [80.0, 340.0]
        kp[16] = [110.0, 380.0]
        width = stance_width(kp, high_scores)
        expected = float(np.hypot(110 - 80, 380 - 340))  # 50.0
        assert width == pytest.approx(expected)

    def test_returns_none_when_ankle_unconfident(self, standing_keypoints):
        scores = [0.95] * 17
        scores[15] = 0.2
        assert stance_width(standing_keypoints, scores) is None

    def test_zero_width_when_ankles_overlap(self, standing_keypoints, high_scores):
        kp = [list(p) for p in standing_keypoints]
        kp[15] = [100.0, 340.0]
        kp[16] = [100.0, 340.0]
        assert stance_width(kp, high_scores) == pytest.approx(0.0)


# ---------- knee_flexion ----------

class TestKneeFlexion:
    def test_straight_leg_is_180(self, standing_keypoints, high_scores):
        # Hip-knee-ankle vertically aligned → 180°
        assert knee_flexion(standing_keypoints, high_scores, side="left") == \
            pytest.approx(180.0, abs=0.1)
        assert knee_flexion(standing_keypoints, high_scores, side="right") == \
            pytest.approx(180.0, abs=0.1)

    def test_90deg_bend(self, high_scores):
        # Hip at (0, 0), knee at (0, 100), ankle at (100, 100)
        # Vectors at knee: (0,-100) up to hip, (100, 0) to ankle → 90°
        kp = [[0.0, 0.0]] * 17
        kp[11] = [0.0, 0.0]   # hip
        kp[13] = [0.0, 100.0] # knee
        kp[15] = [100.0, 100.0]  # ankle
        angle = knee_flexion(kp, high_scores, side="left")
        assert angle == pytest.approx(90.0, abs=0.1)

    def test_acute_bend_below_90(self, high_scores):
        # Deep squat: hip near ankle in y, knee out front
        kp = [[0.0, 0.0]] * 17
        kp[11] = [0.0, 50.0]   # hip
        kp[13] = [50.0, 100.0] # knee
        kp[15] = [0.0, 100.0]  # ankle (close to hip in xy)
        angle = knee_flexion(kp, high_scores, side="left")
        assert 0 < angle < 90

    def test_right_side_uses_right_keypoints(self, high_scores):
        kp = [[0.0, 0.0]] * 17
        # Right leg bent 90°, left leg straight
        kp[11] = [0.0, 0.0]    # left hip
        kp[13] = [0.0, 100.0]  # left knee
        kp[15] = [0.0, 200.0]  # left ankle (straight)
        kp[12] = [50.0, 0.0]   # right hip
        kp[14] = [50.0, 100.0] # right knee
        kp[16] = [150.0, 100.0]# right ankle (bent 90°)
        assert knee_flexion(kp, high_scores, side="left") == pytest.approx(180.0, abs=0.1)
        assert knee_flexion(kp, high_scores, side="right") == pytest.approx(90.0, abs=0.1)

    def test_returns_none_when_keypoint_unconfident(self, standing_keypoints):
        scores = [0.95] * 17
        scores[13] = 0.2
        assert knee_flexion(standing_keypoints, scores, side="left") is None


# ---------- draw_pose ----------

class TestDrawPose:
    def test_draws_circles_and_skeleton(self, standing_keypoints, high_scores):
        frame = np.zeros((400, 400, 3), dtype=np.uint8)
        out = draw_pose(frame.copy(), standing_keypoints, high_scores)
        # Some pixels should now be non-zero
        assert (out != 0).any()

    def test_does_not_draw_low_confidence_keypoints(self, standing_keypoints):
        scores = [0.0] * 17  # all below default min_score=0.5
        frame = np.zeros((400, 400, 3), dtype=np.uint8)
        out = draw_pose(frame.copy(), standing_keypoints, scores)
        # Nothing should be drawn
        assert (out == 0).all()

    def test_skeleton_pairs_are_valid_indices(self):
        # Sanity check on the constant
        for a, b in SKELETON_PAIRS:
            assert 0 <= a < 17
            assert 0 <= b < 17
