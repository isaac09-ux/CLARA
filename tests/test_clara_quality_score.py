"""Tests for compute_quality_score.

CHANGELOG v0.5.1 explicitly flagged two regressions here:
  - Ball detection rate of 10% used to give full 20/20 (now should give ~4/20)
  - Stability score formula was wrong; 50%+ presence required for max
These tests pin both behaviors.
"""
from collections import defaultdict

import pytest

from clara import compute_quality_score


def _tracks(*sample_counts):
    """Build a {tid: [samples]} dict from a list of sample counts per track."""
    return {i: [{} for _ in range(n)] for i, n in enumerate(sample_counts)}


class TestTrackComponent:
    def test_no_tracks_gives_zero(self):
        score, breakdown = compute_quality_score(
            tracks={}, zones={}, ball_frames=0,
            rejected=defaultdict(int), total_samples=100,
            expected_tracks=12, half_court=False,
        )
        assert "0/30" in breakdown["tracks"]

    def test_full_expected_count_gives_max(self):
        score, breakdown = compute_quality_score(
            tracks=_tracks(*([10] * 12)), zones={}, ball_frames=0,
            rejected=defaultdict(int), total_samples=100,
            expected_tracks=12, half_court=False,
        )
        assert "30/30" in breakdown["tracks"]

    def test_half_expected_gives_half(self):
        score, breakdown = compute_quality_score(
            tracks=_tracks(*([10] * 6)), zones={}, ball_frames=0,
            rejected=defaultdict(int), total_samples=100,
            expected_tracks=12, half_court=False,
        )
        # 6/12 * 30 = 15
        assert "15/30" in breakdown["tracks"]

    def test_more_than_expected_capped_at_30(self):
        score, breakdown = compute_quality_score(
            tracks=_tracks(*([10] * 20)), zones={}, ball_frames=0,
            rejected=defaultdict(int), total_samples=100,
            expected_tracks=12, half_court=False,
        )
        assert "30/30" in breakdown["tracks"]


class TestZoneComponent:
    def test_zones_with_threshold_above_3_count(self):
        # Only zones with >3 visits count toward coverage
        zones = {"A1": 4, "A2": 5, "A3": 2, "A4": 1}
        score, breakdown = compute_quality_score(
            tracks=_tracks(10), zones=zones, ball_frames=0,
            rejected=defaultdict(int), total_samples=100,
            expected_tracks=12, half_court=False,
        )
        # 2 zones qualify, expected=12 (full court) → 2/12 * 25 = 4
        assert "4/25" in breakdown["zonas"]
        assert "2 de 12" in breakdown["zonas"]

    def test_half_court_uses_6_expected_zones(self):
        zones = {f"A{i}": 10 for i in range(1, 7)}
        score, breakdown = compute_quality_score(
            tracks=_tracks(10), zones=zones, ball_frames=0,
            rejected=defaultdict(int), total_samples=100,
            expected_tracks=6, half_court=True,
        )
        # All 6 zones qualify, expected=6 → 25/25
        assert "25/25" in breakdown["zonas"]


class TestBallComponent:
    """CHANGELOG v0.5.1: previously 10% ball rate gave full 20/20. Pin the
    rebalanced piecewise function:
      rate >= 0.50  →  20
      rate >= 0.10  →  4 + (rate - 0.10) / 0.40 * 16
      rate <  0.10  →  rate / 0.10 * 4
    """

    @pytest.mark.parametrize("ball_frames,total,expected_pts", [
        (0, 100, 0),       # 0%  → 0
        (5, 100, 2),       # 5%  → 2
        (10, 100, 4),      # 10% → 4 (NOT 20 — this is the regression fix)
        (50, 100, 20),     # 50% → 20 (max)
        (75, 100, 20),     # 75% → still 20 (cap)
        (30, 100, 12),     # 30% = 0.10+0.20 of the way to 0.50 → 4 + 0.5*16 = 12
    ])
    def test_piecewise_curve(self, ball_frames, total, expected_pts):
        score, breakdown = compute_quality_score(
            tracks=_tracks(10), zones={}, ball_frames=ball_frames,
            rejected=defaultdict(int), total_samples=total,
            expected_tracks=12, half_court=False,
        )
        assert f"{expected_pts}/20" in breakdown["balon"]

    def test_zero_samples_does_not_crash(self):
        score, breakdown = compute_quality_score(
            tracks={}, zones={}, ball_frames=0,
            rejected=defaultdict(int), total_samples=0,
            expected_tracks=12, half_court=False,
        )
        assert "0/20" in breakdown["balon"]


class TestFilterAcceptanceComponent:
    def test_no_rejects_gives_max(self):
        score, breakdown = compute_quality_score(
            tracks=_tracks(50), zones={}, ball_frames=0,
            rejected=defaultdict(int), total_samples=100,
            expected_tracks=12, half_court=False,
        )
        # All 50 detections accepted, 0 rejected → 15/15
        assert "15/15" in breakdown["filtrado"]

    def test_half_rejected_gives_half(self):
        rejected = defaultdict(int)
        rejected["fg_too_large"] = 50
        score, breakdown = compute_quality_score(
            tracks=_tracks(50), zones={}, ball_frames=0,
            rejected=rejected, total_samples=200,
            expected_tracks=12, half_court=False,
        )
        # 50 accepted / 100 total → 7/15
        assert "7/15" in breakdown["filtrado"]


class TestStabilityComponent:
    """CHANGELOG v0.5.1: 'Stability score formula corrected. 50%+ presence for max'"""

    def test_50pct_avg_presence_gives_max(self):
        # 2 tracks, each 50 samples, total_samples=100 → avg presence = 50%
        score, breakdown = compute_quality_score(
            tracks=_tracks(50, 50), zones={}, ball_frames=0,
            rejected=defaultdict(int), total_samples=100,
            expected_tracks=12, half_court=False,
        )
        assert "10/10" in breakdown["estabilidad"]

    def test_above_50pct_still_capped_at_10(self):
        score, breakdown = compute_quality_score(
            tracks=_tracks(80, 90), zones={}, ball_frames=0,
            rejected=defaultdict(int), total_samples=100,
            expected_tracks=12, half_court=False,
        )
        assert "10/10" in breakdown["estabilidad"]

    def test_25pct_presence_gives_half(self):
        # avg 25 samples / 100 total = 25% → 25/50 * 10 = 5
        score, breakdown = compute_quality_score(
            tracks=_tracks(25, 25), zones={}, ball_frames=0,
            rejected=defaultdict(int), total_samples=100,
            expected_tracks=12, half_court=False,
        )
        assert "5/10" in breakdown["estabilidad"]

    def test_no_tracks_gives_zero(self):
        score, breakdown = compute_quality_score(
            tracks={}, zones={}, ball_frames=0,
            rejected=defaultdict(int), total_samples=100,
            expected_tracks=12, half_court=False,
        )
        assert "0/10" in breakdown["estabilidad"]


class TestTotalScore:
    def test_total_is_sum_of_components(self):
        # All-max scenario
        score, breakdown = compute_quality_score(
            tracks=_tracks(*([50] * 12)),
            zones={f"A{i}": 10 for i in range(1, 7)} |
                  {f"B{i}": 10 for i in range(1, 7)},
            ball_frames=60, rejected=defaultdict(int), total_samples=100,
            expected_tracks=12, half_court=False,
        )
        assert score == 100

    def test_total_bounded_0_to_100(self):
        # Worst case
        score, _ = compute_quality_score(
            tracks={}, zones={}, ball_frames=0,
            rejected=defaultdict(int), total_samples=100,
            expected_tracks=12, half_court=False,
        )
        assert 0 <= score <= 100

    def test_realistic_copa_scenario_around_78(self):
        """CHANGELOG: 'Copa test: 14.4% balls → 78/100 (was 94/100)'
        Recreate a similar scenario and assert score is in the 'good' band."""
        tracks = _tracks(*([40] * 10))  # 10 of expected 12
        zones = {f"A{i}": 20 for i in range(1, 7)} | \
                {f"B{i}": 15 for i in range(1, 7)}
        score, _ = compute_quality_score(
            tracks=tracks, zones=zones,
            ball_frames=144, rejected=defaultdict(int), total_samples=1000,
            expected_tracks=12, half_court=False,
        )
        # 10/12*30=25 + 25 + ~6 + 15 + 10 = ~81 — solidly in "BUENO" band (60-79)
        # or low EXCELENTE. Assert it's not the broken 94+
        assert 70 <= score <= 90
