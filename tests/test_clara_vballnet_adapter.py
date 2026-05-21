"""Tests for the detect_balls_vballnet adapter in clara.py.

The adapter wraps ball_vballnet.detect_balls and applies CLARA's foreground
filter + homography projection. We mock the underlying detector so this stays
fast and dependency-free.
"""
from collections import defaultdict
from unittest.mock import patch

import numpy as np
import pytest

from clara import detect_balls_vballnet


@pytest.fixture
def identity_pipeline_args(identity_homography):
    """Args wired so that pixel (x, y) → court (x, y) meters when ppm=1."""
    return {
        "video_path": "fake.mp4",
        "model_path": "fake.onnx",
        "H": identity_homography,
        "ppm": 1.0,
        "court_w": 9.0,
        "court_h": 18.0,
        "frame_h": 720,
        "frame_w": 1280,
    }


def test_passes_detections_through_when_all_ok(identity_pipeline_args):
    fake_dets = [
        {"frame": 10, "x": 400.0, "y": 300.0, "radius": 8.0, "confidence": 0.7},
        {"frame": 20, "x": 600.0, "y": 350.0, "radius": 8.0, "confidence": 0.8},
    ]
    rejected = defaultdict(int)
    with patch("ball_vballnet.detect_balls", return_value=fake_dets):
        out = detect_balls_vballnet(
            **identity_pipeline_args,
            rejected_counts=rejected,
            verbose=False,
        )
    assert len(out) == 2
    assert out[0]["frame"] == 10
    assert out[0]["court_x"] == pytest.approx(400.0)
    assert out[0]["court_y"] == pytest.approx(300.0)
    assert out[0]["conf"] == pytest.approx(0.7)
    assert sum(rejected.values()) == 0


def test_horizon_filter_rejects_ball_below_court(identity_pipeline_args):
    # court_horizon_y=300, slack=10% of frame_h=72 → y2 > 372 rejects.
    # Detection at y=400 with radius=8 → y2 = 408 > 372 → rejected
    fake_dets = [
        {"frame": 5, "x": 400.0, "y": 400.0, "radius": 8.0, "confidence": 0.6},
    ]
    rejected = defaultdict(int)
    with patch("ball_vballnet.detect_balls", return_value=fake_dets):
        out = detect_balls_vballnet(
            **identity_pipeline_args,
            court_horizon_y=300,
            rejected_counts=rejected,
            verbose=False,
        )
    assert out == []
    assert rejected["ball_fg_below_horizon"] == 1


def test_bottom_edge_rejected(identity_pipeline_args):
    # Frame height 720, edge guard rejects y2 >= 715. Detection at y=712 r=5 → y2=717
    fake_dets = [
        {"frame": 5, "x": 400.0, "y": 712.0, "radius": 5.0, "confidence": 0.6},
    ]
    rejected = defaultdict(int)
    with patch("ball_vballnet.detect_balls", return_value=fake_dets):
        out = detect_balls_vballnet(
            **identity_pipeline_args,
            rejected_counts=rejected,
            verbose=False,
        )
    assert out == []
    assert rejected["ball_fg_at_edge"] == 1


def test_empty_input_returns_empty(identity_pipeline_args):
    with patch("ball_vballnet.detect_balls", return_value=[]):
        out = detect_balls_vballnet(
            **identity_pipeline_args,
            rejected_counts=defaultdict(int),
            verbose=False,
        )
    assert out == []


def test_missing_radius_uses_synthetic_minimum(identity_pipeline_args):
    """A detection without 'radius' key shouldn't crash — adapter floors at 5.0."""
    fake_dets = [
        {"frame": 1, "x": 400.0, "y": 300.0, "confidence": 0.7},  # no radius
    ]
    with patch("ball_vballnet.detect_balls", return_value=fake_dets):
        out = detect_balls_vballnet(
            **identity_pipeline_args,
            rejected_counts=defaultdict(int),
            verbose=False,
        )
    assert len(out) == 1


def test_projection_applies_ppm(identity_homography):
    """With ppm=2, pixel coords get divided by 2 to get meters."""
    fake_dets = [
        {"frame": 1, "x": 400.0, "y": 300.0, "radius": 5.0, "confidence": 0.7},
    ]
    with patch("ball_vballnet.detect_balls", return_value=fake_dets):
        out = detect_balls_vballnet(
            video_path="fake.mp4", model_path="fake.onnx",
            H=identity_homography, ppm=2.0,
            court_w=9.0, court_h=18.0,
            frame_h=720, frame_w=1280,
            rejected_counts=defaultdict(int),
            verbose=False,
        )
    assert out[0]["court_x"] == pytest.approx(200.0)
    assert out[0]["court_y"] == pytest.approx(150.0)
