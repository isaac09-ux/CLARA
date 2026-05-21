"""Tests for pure geometry helpers in clara.py.

Covers:
  - project: homography projection
  - zone_for_court_pos: position → official volleyball zone label
  - is_in_court: bounds check with margin (CHANGELOG v0.5.1 flagged a
    margin inconsistency with zone_for_court_pos that we pin here)
"""
import numpy as np
import pytest

from clara import is_in_court, project, zone_for_court_pos


# ---------- project ----------

class TestProject:
    def test_identity_is_a_no_op(self, identity_homography):
        x, y = project(identity_homography, 100.0, 200.0)
        assert x == pytest.approx(100.0)
        assert y == pytest.approx(200.0)

    def test_translation(self):
        H = np.array([
            [1.0, 0.0, 50.0],
            [0.0, 1.0, -30.0],
            [0.0, 0.0, 1.0],
        ])
        x, y = project(H, 10.0, 20.0)
        assert x == pytest.approx(60.0)
        assert y == pytest.approx(-10.0)

    def test_scaling(self):
        H = np.diag([2.0, 3.0, 1.0])
        x, y = project(H, 5.0, 7.0)
        assert x == pytest.approx(10.0)
        assert y == pytest.approx(21.0)

    def test_accepts_python_list_homography(self, full_court_calibration):
        # run() passes np.array(H), but project() must also tolerate raw lists
        H = full_court_calibration["homography_matrix"]
        x, y = project(H, 600.0, 500.0)
        assert np.isfinite(x) and np.isfinite(y)


# ---------- zone_for_court_pos ----------

class TestZoneForCourtPos:
    def test_out_of_bounds_returns_none(self):
        assert zone_for_court_pos(-1.0, 5.0, 9, 18) is None
        assert zone_for_court_pos(10.0, 5.0, 9, 18) is None
        assert zone_for_court_pos(5.0, -1.0, 9, 18) is None
        assert zone_for_court_pos(5.0, 19.0, 9, 18) is None

    def test_within_margin_is_clamped_and_returns_zone(self):
        # x slightly negative but within margin
        z = zone_for_court_pos(-0.3, 1.5, 9, 18, margin=0.5)
        assert z is not None
        assert z.startswith("A")

    def test_full_court_side_A_front_zones(self):
        # Side A = top half (y < 9). Front row = y > court_h/4 = 4.5 (closer to net at y=9)
        # Left column (x < 3) front  → A4
        # Middle (3 <= x < 6) front  → A3
        # Right (x >= 6) front       → A2
        assert zone_for_court_pos(1.5, 7.5, 9, 18) == "A4"
        assert zone_for_court_pos(4.5, 7.5, 9, 18) == "A3"
        assert zone_for_court_pos(7.5, 7.5, 9, 18) == "A2"

    def test_full_court_side_A_back_zones(self):
        # Back row (y <= 4.5)
        # Left → A5, Middle → A6, Right → A1
        assert zone_for_court_pos(1.5, 1.5, 9, 18) == "A5"
        assert zone_for_court_pos(4.5, 1.5, 9, 18) == "A6"
        assert zone_for_court_pos(7.5, 1.5, 9, 18) == "A1"

    def test_full_court_side_B_mirrored(self):
        # Side B = bottom half. Zone numbering mirrors L/R from coach perspective.
        # docs/architecture.md:
        #   LADO B front: B2 (left)  B3 (mid)  B4 (right)   ← near net
        #   LADO B back:  B1 (left)  B6 (mid)  B5 (right)   ← near endline
        assert zone_for_court_pos(1.5, 10.5, 9, 18) == "B2"
        assert zone_for_court_pos(4.5, 10.5, 9, 18) == "B3"
        assert zone_for_court_pos(7.5, 10.5, 9, 18) == "B4"
        assert zone_for_court_pos(1.5, 16.5, 9, 18) == "B1"
        assert zone_for_court_pos(4.5, 16.5, 9, 18) == "B6"
        assert zone_for_court_pos(7.5, 16.5, 9, 18) == "B5"

    def test_half_court_uses_only_side_A_labels(self):
        # half_court has only one side (the visible one) → all zones prefix "A"
        # In half-court the y axis is 0..court_h with front = y < court_h/3
        # (court_h=9 → front when y < 3)
        assert zone_for_court_pos(1.5, 1.5, 9, 9, half_court=True) == "A4"
        assert zone_for_court_pos(4.5, 1.5, 9, 9, half_court=True) == "A3"
        assert zone_for_court_pos(7.5, 1.5, 9, 9, half_court=True) == "A2"
        # Back: y > 3
        assert zone_for_court_pos(1.5, 7.5, 9, 9, half_court=True) == "A5"
        assert zone_for_court_pos(4.5, 7.5, 9, 9, half_court=True) == "A6"
        assert zone_for_court_pos(7.5, 7.5, 9, 9, half_court=True) == "A1"

    @pytest.mark.parametrize("x,y,expected_side", [
        (4.5, 0.0, "A"),   # top edge
        (4.5, 8.99, "A"),  # just before mid
        (4.5, 9.01, "B"),  # just after mid
        (4.5, 18.0, "B"),  # bottom edge
    ])
    def test_full_court_side_split_at_midline(self, x, y, expected_side):
        z = zone_for_court_pos(x, y, 9, 18)
        assert z is not None
        assert z.startswith(expected_side)


# ---------- is_in_court ----------

class TestIsInCourt:
    def test_inside_returns_true(self):
        assert is_in_court(4.5, 9.0, 9, 18)

    def test_outside_no_margin_returns_false(self):
        assert not is_in_court(-1.0, 5.0, 9, 18, margin=0.0)
        assert not is_in_court(10.0, 5.0, 9, 18, margin=0.0)

    def test_default_margin_accepts_slight_negative(self):
        # default margin=0.5 in is_in_court
        assert is_in_court(-0.4, 5.0, 9, 18)
        assert not is_in_court(-0.6, 5.0, 9, 18)

    def test_none_coords_returns_false(self):
        assert not is_in_court(None, 5.0, 9, 18)
        assert not is_in_court(5.0, None, 9, 18)

    def test_margin_consistency_with_zone_for_court_pos(self):
        """CHANGELOG v0.5.1 fix: margins between is_in_court and zone_for_court_pos
        used to disagree. Pin behavior: when both are called with margin=0.5, a
        point that's in-court should also map to a zone."""
        for x, y in [(0.0, 0.0), (9.0, 18.0), (-0.5, 9.0), (4.5, 18.5)]:
            in_court = is_in_court(x, y, 9, 18, margin=0.5)
            zone = zone_for_court_pos(x, y, 9, 18, margin=0.5)
            assert in_court == (zone is not None), (
                f"Mismatch at ({x}, {y}): is_in_court={in_court}, zone={zone}"
            )
