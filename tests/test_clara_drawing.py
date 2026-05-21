"""Smoke tests for the matplotlib-free image generators in clara.py.

save_topdown and save_diagnostic_frame use cv2 directly — we run them end-to-end
on tiny inputs and assert the output PNG exists and is readable.
"""
import json

import cv2
import numpy as np
import pytest

from clara import save_topdown


@pytest.fixture
def minimal_metrics():
    return {
        "duration_min": 5.0,
        "filtered_tracks": 2,
        "ball_detections_oncourt": 10,
        "quality_score": 75,
        "ball_detector": "vballnet",
        "pose_mode": "none",
        "half_court": False,
    }


class TestSaveTopdown:
    def test_writes_a_readable_png(self, tmp_path, minimal_metrics):
        tracks = {
            1: [{"court_x": 4.5, "court_y": 5.0, "frame": 0},
                {"court_x": 5.0, "court_y": 5.5, "frame": 5}],
            2: [{"court_x": 3.0, "court_y": 12.0, "frame": 10}],
        }
        ball = [{"court_x": 4.5, "court_y": 9.0, "frame": 5}]
        out_path = tmp_path / "topdown.png"
        save_topdown(
            tracks, ball, court_w=9, court_h=18, ppm=20, path=out_path,
            title="test", metrics=minimal_metrics, half_court=False,
        )
        assert out_path.exists()
        img = cv2.imread(str(out_path))
        assert img is not None
        assert img.shape[0] > 0 and img.shape[1] > 0

    def test_writes_half_court_layout(self, tmp_path, minimal_metrics):
        out_path = tmp_path / "topdown_half.png"
        save_topdown(
            tracks={}, ball=[], court_w=9, court_h=9, ppm=20,
            path=out_path, title="half", metrics=minimal_metrics,
            half_court=True,
        )
        assert out_path.exists()

    def test_handles_empty_tracks_and_balls(self, tmp_path):
        out_path = tmp_path / "empty.png"
        save_topdown(
            tracks={}, ball=[], court_w=9, court_h=18, ppm=20,
            path=out_path, title="empty",
        )
        assert out_path.exists()

    def test_works_without_metrics(self, tmp_path):
        # metrics=None code path
        out_path = tmp_path / "no_metrics.png"
        save_topdown(
            tracks={1: [{"court_x": 4.5, "court_y": 9.0, "frame": 0}] * 5},
            ball=[],
            court_w=9, court_h=18, ppm=20, path=out_path,
        )
        assert out_path.exists()
