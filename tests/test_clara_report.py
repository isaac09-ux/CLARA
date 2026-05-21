"""Tests for clara_report.py — HTML report generation."""
import json
from pathlib import Path

import pytest

from clara_report import embed_image_as_base64, generate_report, render_zone_bars


# ---------- render_zone_bars ----------

class TestRenderZoneBars:
    def test_empty_zones_returns_placeholder(self):
        html = render_zone_bars({})
        assert "Sin datos" in html

    def test_side_filter_restricts_zones(self):
        zones = {"A1": 10, "A2": 5, "B1": 20, "B3": 15}
        html_a = render_zone_bars(zones, side_filter="A")
        assert "A1" in html_a
        assert "A2" in html_a
        assert "B1" not in html_a
        assert "B3" not in html_a

    def test_side_A_renders_in_official_order(self):
        # Render order is 4-3-2 / 5-6-1 (volleyball convention)
        zones = {"A1": 1, "A2": 2, "A3": 3, "A4": 4, "A5": 5, "A6": 6}
        html = render_zone_bars(zones, side_filter="A")
        order_pos = [html.index(f">A{i}<") for i in [4, 3, 2, 5, 6, 1]]
        assert order_pos == sorted(order_pos)

    def test_bar_widths_proportional_to_max(self):
        zones = {"A1": 50, "A2": 100}  # A2 is the max
        html = render_zone_bars(zones, side_filter="A")
        # The 100% bar should appear (or close to it)
        assert "width: 100%" in html or "width: 100.0%" in html
        # The 50-value bar should be at 50%
        assert "width: 50" in html

    def test_explicit_max_val_overrides_data_max(self):
        zones = {"A1": 50}
        html = render_zone_bars(zones, side_filter="A", max_val=200)
        # 50/200 = 25%
        assert "width: 25" in html

    def test_unfiltered_renders_all(self):
        zones = {"A1": 10, "B1": 20}
        html = render_zone_bars(zones)
        assert "A1" in html and "B1" in html


# ---------- embed_image_as_base64 ----------

class TestEmbedImage:
    def test_returns_img_tag_when_exists(self, synthetic_topdown_png):
        html = embed_image_as_base64(synthetic_topdown_png)
        assert "<img" in html
        assert "data:image/png;base64," in html

    def test_returns_empty_when_missing(self, tmp_path):
        html = embed_image_as_base64(tmp_path / "nonexistent.png")
        assert html == ""


# ---------- generate_report ----------

class TestGenerateReport:
    def test_writes_html_file(self, sample_metrics_json):
        out = generate_report(sample_metrics_json)
        assert Path(out).exists()
        content = Path(out).read_text()
        assert "<html" in content
        assert "</html>" in content

    def test_embeds_metadata(self, sample_metrics_json):
        out = generate_report(sample_metrics_json)
        content = Path(out).read_text(encoding="utf-8")
        assert "test_match.mp4" in content
        assert "12.3" in content  # duration
        assert "0.6" in content   # version

    def test_includes_top_tracks(self, sample_metrics_json):
        out = generate_report(sample_metrics_json)
        content = Path(out).read_text(encoding="utf-8")
        assert "#1" in content and "#2" in content
        assert "A3" in content  # dominant zone

    def test_embeds_topdown_when_provided(self, sample_metrics_json,
                                          synthetic_topdown_png):
        out = generate_report(sample_metrics_json,
                              topdown_path=synthetic_topdown_png)
        content = Path(out).read_text(encoding="utf-8")
        assert "data:image/png;base64," in content

    def test_topdown_placeholder_when_absent(self, sample_metrics_json):
        out = generate_report(sample_metrics_json)
        content = Path(out).read_text(encoding="utf-8")
        assert "no disponible" in content

    def test_ball_warning_shown_when_zero_balls(self, tmp_path):
        data = {
            "clara_version": "0.6", "video": "test.mp4", "duration_min": 5,
            "stride": 5, "samples_processed": 100, "filtered_tracks": 0,
            "raw_tracks": 0, "ball_detections_oncourt": 0,
            "court_size_m": [9, 18],
            "zone_visits_total": {}, "zone_visits_first_half": {},
            "zone_visits_second_half": {}, "tracks": [],
        }
        json_path = tmp_path / "data.json"
        json_path.write_text(json.dumps(data))
        out = generate_report(json_path)
        content = Path(out).read_text(encoding="utf-8")
        assert "Cero detecciones" in content

    def test_no_ball_warning_when_balls_present(self, sample_metrics_json):
        out = generate_report(sample_metrics_json)
        content = Path(out).read_text(encoding="utf-8")
        assert "Cero detecciones" not in content

    def test_custom_output_path(self, sample_metrics_json, tmp_path):
        custom_out = tmp_path / "custom_report.html"
        out = generate_report(sample_metrics_json, output_path=custom_out)
        assert Path(out) == custom_out
        assert custom_out.exists()
