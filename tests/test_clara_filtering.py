"""Tests for filtering logic in clara.py.

Covers:
  - classify_detection: foreground filter
  - filter_ball_tracks: temporal/spatial clustering that drops isolated FPs
"""
import pytest

from clara import classify_detection, filter_ball_tracks


# ---------- classify_detection ----------

class TestClassifyDetection:
    FRAME_H = 720
    FRAME_W = 1280

    def test_ok_for_normal_person(self):
        # 100×300 bbox in middle of frame
        bbox = [500, 200, 600, 500]
        status, reason = classify_detection(bbox, self.FRAME_H, self.FRAME_W)
        assert status == "ok"
        assert reason is None

    def test_too_tall_person_rejected(self):
        # bbox height = 500, frame_h = 720 → ratio 0.69 > 0.55
        bbox = [500, 100, 600, 600]
        status, reason = classify_detection(bbox, self.FRAME_H, self.FRAME_W)
        assert status == "fg_too_large"
        assert "altura" in reason

    def test_too_wide_person_rejected(self):
        # bbox width = 600, frame_w = 1280 → ratio 0.47 > 0.40
        bbox = [200, 200, 800, 500]
        status, reason = classify_detection(bbox, self.FRAME_H, self.FRAME_W)
        assert status == "fg_too_large"
        assert "ancho" in reason

    def test_below_horizon_rejected(self):
        # horizon at y=300, bottom of bbox at y=500 → > horizon + 10% frame (72)
        bbox = [500, 200, 600, 500]
        status, reason = classify_detection(
            bbox, self.FRAME_H, self.FRAME_W, court_horizon_y=300,
        )
        assert status == "fg_below_horizon"

    def test_horizon_allows_slack_within_10pct(self):
        # horizon=300, slack = frame_h * 0.1 = 72 → bbox y2=370 still ok
        bbox = [500, 200, 600, 370]
        status, _ = classify_detection(
            bbox, self.FRAME_H, self.FRAME_W, court_horizon_y=300,
        )
        assert status == "ok"

    def test_at_bottom_edge_rejected(self):
        # bbox touches bottom (y2 = 718 >= 720 - 5) but isn't too tall/wide
        # so the edge check is what fires, not the size check
        bbox = [500, 400, 600, 718]
        status, reason = classify_detection(bbox, self.FRAME_H, self.FRAME_W)
        assert status == "fg_at_edge"
        assert "borde" in reason

    def test_ball_skips_height_width_checks(self):
        """CHANGELOG v0.6: is_ball=True bypasses height/width ratio (balls
        are small anyway, the relevant rejection for them is horizon/edge)."""
        # Same too-tall bbox that would reject a person
        bbox = [500, 100, 600, 600]
        status, _ = classify_detection(
            bbox, self.FRAME_H, self.FRAME_W, is_ball=True,
        )
        assert status == "ok"

    def test_ball_still_rejected_at_bottom_edge(self):
        """CHANGELOG v0.5.1 fix: foreground filter applies to ball too."""
        bbox = [500, 700, 600, 718]
        status, _ = classify_detection(
            bbox, self.FRAME_H, self.FRAME_W, is_ball=True,
        )
        assert status == "fg_at_edge"

    def test_ball_still_rejected_below_horizon(self):
        bbox = [500, 380, 600, 420]
        status, _ = classify_detection(
            bbox, self.FRAME_H, self.FRAME_W, court_horizon_y=300, is_ball=True,
        )
        assert status == "fg_below_horizon"


# ---------- filter_ball_tracks ----------

def _det(frame, x, y, conf=0.5):
    return {"frame": frame, "court_x": x, "court_y": y, "conf": conf}


class TestFilterBallTracks:
    def test_empty_input_returns_empty(self):
        assert filter_ball_tracks([], stride=5) == []

    def test_single_detection_dropped_as_isolated(self):
        # min_track_len=2 by default → a lone detection has nothing to pair with
        assert filter_ball_tracks([_det(10, 4.5, 9.0)], stride=5) == []

    def test_two_close_detections_survive(self):
        dets = [_det(10, 4.5, 9.0), _det(15, 4.6, 9.1)]
        out = filter_ball_tracks(dets, stride=5)
        assert len(out) == 2

    def test_distant_pair_in_space_split_into_two_short_tracks_and_dropped(self):
        # Frames close (within gap) but positions 10m apart → different tracks,
        # each of length 1 < min_track_len → both dropped
        dets = [_det(10, 1.0, 1.0), _det(12, 8.0, 17.0)]
        out = filter_ball_tracks(dets, stride=5, max_dist_m=6.0)
        assert out == []

    def test_temporally_isolated_detection_split(self):
        # Two detections close in space but a huge frame gap → split → both lonely
        dets = [_det(10, 4.5, 9.0), _det(500, 4.5, 9.0)]
        out = filter_ball_tracks(dets, stride=5)
        # Default max_gap = max(stride*3, 6) = 15 < 490 → split → both dropped
        assert out == []

    def test_long_real_rally_preserved(self):
        # 8 detections moving smoothly across the court
        dets = [
            _det(10, 4.0, 9.0),
            _det(15, 4.5, 9.5),
            _det(20, 5.0, 10.0),
            _det(25, 5.5, 10.5),
            _det(30, 6.0, 11.0),
            _det(35, 6.5, 11.5),
            _det(40, 7.0, 12.0),
            _det(45, 7.5, 12.5),
        ]
        out = filter_ball_tracks(dets, stride=5)
        assert len(out) == 8

    def test_real_track_plus_isolated_fp(self):
        # Real moving ball + a single stray detection (e.g. ceiling light)
        real = [
            _det(10, 4.0, 9.0),
            _det(15, 4.5, 9.5),
            _det(20, 5.0, 10.0),
        ]
        fp = [_det(200, 0.5, 1.0)]
        out = filter_ball_tracks(real + fp, stride=5)
        assert len(out) == 3
        assert all(d["frame"] < 100 for d in out)

    def test_custom_min_track_len(self):
        # With min_track_len=3, a 2-det track gets dropped
        dets = [_det(10, 4.0, 9.0), _det(15, 4.5, 9.5)]
        assert filter_ball_tracks(dets, stride=5, min_track_len=3) == []

    def test_input_ordering_independent(self):
        # filter sorts by frame internally
        dets_sorted = [_det(10, 4.0, 9.0), _det(15, 4.5, 9.5), _det(20, 5.0, 10.0)]
        dets_shuffled = [dets_sorted[2], dets_sorted[0], dets_sorted[1]]
        out_a = filter_ball_tracks(dets_sorted, stride=5)
        out_b = filter_ball_tracks(dets_shuffled, stride=5)
        assert [d["frame"] for d in out_a] == [d["frame"] for d in out_b]
