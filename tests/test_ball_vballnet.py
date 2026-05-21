"""Tests for ball_vballnet.py adapter pure functions.

Covers _preprocess_frame, _postprocess_heatmap, load_balls_from_csv.
detect_balls() requires onnxruntime + a model file so is left to integration tests.
"""
import csv

import cv2
import numpy as np
import pytest

from ball_vballnet import (
    MODEL_H,
    MODEL_W,
    SEQ_LEN,
    _postprocess_heatmap,
    _preprocess_frame,
    load_balls_from_csv,
)


# ---------- _preprocess_frame ----------

class TestPreprocessFrame:
    def test_output_shape_matches_model_input(self, fake_frame_hd):
        proc = _preprocess_frame(fake_frame_hd)
        assert proc.shape == (MODEL_H, MODEL_W)

    def test_output_dtype_is_float32(self, fake_frame_hd):
        proc = _preprocess_frame(fake_frame_hd)
        assert proc.dtype == np.float32

    def test_output_values_in_unit_range(self, fake_frame_hd):
        proc = _preprocess_frame(fake_frame_hd)
        assert proc.min() >= 0.0
        assert proc.max() <= 1.0

    def test_solid_white_normalizes_to_one(self):
        frame = np.full((480, 640, 3), 255, dtype=np.uint8)
        proc = _preprocess_frame(frame)
        assert proc.max() == pytest.approx(1.0)
        assert proc.min() == pytest.approx(1.0)

    def test_solid_black_normalizes_to_zero(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        proc = _preprocess_frame(frame)
        assert proc.max() == pytest.approx(0.0)


# ---------- _postprocess_heatmap ----------

class TestPostprocessHeatmap:
    def test_below_threshold_returns_none(self):
        heatmap = np.full((MODEL_H, MODEL_W), 0.2, dtype=np.float32)
        assert _postprocess_heatmap(heatmap, threshold=0.5) is None

    def test_finds_single_hot_pixel(self):
        heatmap = np.zeros((MODEL_H, MODEL_W), dtype=np.float32)
        heatmap[100, 200] = 0.9
        det = _postprocess_heatmap(heatmap, threshold=0.5)
        assert det is not None
        # No rescale (orig_w/h not provided) → coords in model space
        assert det["x"] == pytest.approx(200.0)
        assert det["y"] == pytest.approx(100.0)
        assert det["confidence"] == pytest.approx(0.9)

    def test_rescales_to_original_coords(self):
        heatmap = np.zeros((MODEL_H, MODEL_W), dtype=np.float32)
        # Hot pixel at center of model grid
        heatmap[MODEL_H // 2, MODEL_W // 2] = 0.95
        det = _postprocess_heatmap(
            heatmap, threshold=0.5, orig_w=1920, orig_h=1080,
        )
        # Centre maps to centre
        assert det["x"] == pytest.approx(1920 / 2, abs=2.0)
        assert det["y"] == pytest.approx(1080 / 2, abs=2.0)

    def test_radius_grows_with_blob_size(self):
        small = np.zeros((MODEL_H, MODEL_W), dtype=np.float32)
        small[100:102, 100:102] = 0.9  # 4 pixels

        big = np.zeros((MODEL_H, MODEL_W), dtype=np.float32)
        big[100:110, 100:110] = 0.9    # 100 pixels

        det_small = _postprocess_heatmap(small, threshold=0.5)
        det_big = _postprocess_heatmap(big, threshold=0.5)
        assert det_big["radius"] > det_small["radius"]

    def test_radius_has_minimum_floor(self):
        heatmap = np.zeros((MODEL_H, MODEL_W), dtype=np.float32)
        heatmap[100, 100] = 0.9  # single pixel
        det = _postprocess_heatmap(heatmap, threshold=0.5)
        # Floor is 2.0 in model space
        assert det["radius"] >= 2.0

    def test_picks_strongest_blob_when_multiple(self):
        heatmap = np.zeros((MODEL_H, MODEL_W), dtype=np.float32)
        heatmap[50, 50] = 0.6        # weaker
        heatmap[200, 300] = 0.95     # stronger
        det = _postprocess_heatmap(heatmap, threshold=0.5)
        assert det["x"] == pytest.approx(300.0)
        assert det["y"] == pytest.approx(200.0)


# ---------- load_balls_from_csv ----------

class TestLoadBallsFromCsv:
    def _write_csv(self, path, rows):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["Frame", "Visibility", "X", "Y", "Radius"]
            )
            writer.writeheader()
            writer.writerows(rows)

    def test_loads_visible_rows(self, tmp_path):
        csv_path = tmp_path / "balls.csv"
        self._write_csv(csv_path, [
            {"Frame": 1, "Visibility": 1, "X": 100.5, "Y": 200.0, "Radius": 5.0},
            {"Frame": 2, "Visibility": 1, "X": 110.0, "Y": 205.0, "Radius": 5.0},
        ])
        out = load_balls_from_csv(csv_path)
        assert len(out) == 2
        assert out[0]["frame"] == 1
        assert out[0]["x"] == pytest.approx(100.5)
        assert out[0]["confidence"] == 1.0

    def test_skips_invisible_rows(self, tmp_path):
        csv_path = tmp_path / "balls.csv"
        self._write_csv(csv_path, [
            {"Frame": 1, "Visibility": 1, "X": 100.0, "Y": 200.0, "Radius": 5.0},
            {"Frame": 2, "Visibility": 0, "X": 0.0, "Y": 0.0, "Radius": 0.0},
            {"Frame": 3, "Visibility": 1, "X": 110.0, "Y": 210.0, "Radius": 5.0},
        ])
        out = load_balls_from_csv(csv_path)
        assert len(out) == 2
        assert [d["frame"] for d in out] == [1, 3]

    def test_empty_csv(self, tmp_path):
        csv_path = tmp_path / "balls.csv"
        self._write_csv(csv_path, [])
        assert load_balls_from_csv(csv_path) == []


# ---------- module-level constants ----------

class TestModelConstants:
    def test_seq_len_is_nine(self):
        # VballNetV1 architecture invariant — 9 consecutive frames
        assert SEQ_LEN == 9

    def test_model_input_size(self):
        assert (MODEL_H, MODEL_W) == (288, 512)
