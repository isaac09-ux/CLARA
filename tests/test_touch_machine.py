"""Tests de no-regresion de touch_machine.py (post-procesador de toques).

Solo dependen de numpy + scipy (lo que ya pide touch_machine); NO importan
clara.py, asi que corren sin el stack pesado (cv2/ultralytics). Correr:
    python -m unittest tests.test_touch_machine
    pytest tests/test_touch_machine.py

Blindan: el contrato del JSON de CLARA (track_samples vs el resumen `tracks`),
la deteccion de quiebres de trayectoria, la atribucion al jugador mas cercano
y el corte de rallies por hueco temporal.
"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import touch_machine as tm  # noqa: E402


def mk_ball_track(start=(200.0, 200.0), ppm=40.0):
    """Trayectoria sintetica: 4 tramos rectos con quiebres en frames 30/60/90.

    Velocidad constante por tramo y giro de 90 entre tramos -> un pico de
    aceleracion limpio en cada frontera. court_x/court_y = img / ppm.
    """
    segs = [(12.0, 0.0), (0.0, 12.0), (-12.0, 0.0), (0.0, -12.0)]

    def vel(f):
        return segs[min(f // 30, 3)]

    x, y = start
    out = []
    for f in range(120):
        out.append({
            "frame": f,
            "img_x": round(x, 3), "img_y": round(y, 3),
            "court_x": round(x / ppm, 3), "court_y": round(y / ppm, 3),
            "conf": 0.9,
        })
        vx, vy = vel(f)
        x += vx
        y += vy
    return out


class TestExtractTracks(unittest.TestCase):
    def test_prefers_track_samples(self):
        # Con ambos campos presentes debe usar track_samples (la serie cruda),
        # nunca el resumen `tracks`.
        data = {
            "track_samples": [
                {"id": 3, "samples": [{"frame": 0, "court_x": 1.0, "court_y": 2.0}]}
            ],
            "tracks": [{"id": 9, "samples": 5, "reliability": "alta"}],
        }
        out = tm._extract_tracks(data)
        self.assertEqual(set(out), {3})
        self.assertEqual(out[3][0]["court_x"], 1.0)

    def test_summary_tracks_ignored_not_crash(self):
        # Regresion: el `tracks` coach-facing de CLARA trae 'samples' ENTERO.
        # Antes reventaba (sorted(int)); ahora se ignora y avisa con None.
        data = {"tracks": [{"id": 1, "samples": 50, "reliability": "alta"}]}
        self.assertIsNone(tm._extract_tracks(data))

    def test_dict_form(self):
        data = {"track_samples": {"7": [{"frame": 0, "court_x": 0.0, "court_y": 0.0}]}}
        out = tm._extract_tracks(data)
        self.assertIn(7, out)

    def test_load_scouting_rejects_int_samples(self):
        # End-to-end del contrato: ball_track presente pero solo resumen `tracks`
        # -> load_scouting debe abortar con mensaje claro, no con TypeError.
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "scouting_data.json")
            with open(p, "w") as f:
                json.dump({
                    "ball_track": mk_ball_track(),
                    "tracks": [{"id": 1, "samples": 50}],
                }, f)
            with self.assertRaises(ValueError):
                tm.load_scouting(p)


class TestDetectTouches(unittest.TestCase):
    def test_detects_trajectory_breaks(self):
        ball = tm.reconstruct_ball(mk_ball_track(), max_gap_interp=5, smooth_w=1)
        touches, accel, _ = tm.detect_touches(ball, fps=60)
        frames = sorted(t["frame"] for t in touches)
        # Los tres quiebres reales estan en 30/60/90; tolerancia de +-2 frames.
        self.assertGreaterEqual(len(touches), 2)
        for f in frames:
            self.assertTrue(min(abs(f - b) for b in (30, 60, 90)) <= 2,
                            f"toque en frame {f} no cae cerca de un quiebre")

    def test_flat_trajectory_no_touches(self):
        # Balon quieto -> aceleracion ~0 -> sin toques fantasma.
        flat = [{"frame": f, "img_x": 100.0, "img_y": 100.0,
                 "court_x": 2.5, "court_y": 2.5, "conf": 0.9} for f in range(60)]
        ball = tm.reconstruct_ball(flat, max_gap_interp=5, smooth_w=1)
        touches, _, _ = tm.detect_touches(ball, fps=60)
        self.assertEqual(touches, [])


class TestAttribution(unittest.TestCase):
    def test_picks_nearest_player(self):
        touches = [{"frame": 50}]
        tracks = {
            1: [{"frame": 40, "court_x": 1.0, "court_y": 1.0},
                {"frame": 60, "court_x": 1.0, "court_y": 1.0}],
            2: [{"frame": 40, "court_x": 8.0, "court_y": 8.0},
                {"frame": 60, "court_x": 8.0, "court_y": 8.0}],
        }
        ball_by_frame = {50: {"frame": 50, "court_x": 1.2, "court_y": 1.0}}
        out = tm.attribute_touches(touches, tracks, ball_by_frame, ppm=40)
        self.assertEqual(out[0]["track_id"], 1)
        self.assertLessEqual(out[0]["player_distance_m"], tm.MAX_ATTRIBUTION_DIST_M)

    def test_far_ball_is_unattributed(self):
        touches = [{"frame": 50}]
        tracks = {2: [{"frame": 40, "court_x": 8.0, "court_y": 8.0},
                      {"frame": 60, "court_x": 8.0, "court_y": 8.0}]}
        ball_by_frame = {50: {"frame": 50, "court_x": 50.0, "court_y": 50.0}}
        out = tm.attribute_touches(touches, tracks, ball_by_frame, ppm=40)
        self.assertIsNone(out[0]["track_id"])


class TestRallies(unittest.TestCase):
    def test_split_by_gap(self):
        fps = 60
        gap = int(tm.RALLY_GAP_SECONDS * fps)  # 210 frames
        touches = [
            {"frame": 10, "track_id": 1},
            {"frame": 40, "track_id": 2},
            {"frame": 40 + gap + 30, "track_id": 1},
            {"frame": 40 + gap + 60, "track_id": 2},
        ]
        rallies = tm.group_into_rallies(touches, fps)
        self.assertEqual(len(rallies), 2)
        self.assertEqual(rallies[0]["n_touches"], 2)
        self.assertEqual(rallies[1]["n_touches"], 2)
        # Los touches quedan anotados con su rally.
        self.assertEqual(touches[0]["rally_id"], 0)
        self.assertEqual(touches[0]["seq_in_rally"], 0)

    def test_small_gaps_stay_one_rally(self):
        touches = [{"frame": f, "track_id": 1} for f in range(0, 200, 20)]
        rallies = tm.group_into_rallies(touches, fps=60)
        self.assertEqual(len(rallies), 1)


class TestRunEndToEnd(unittest.TestCase):
    def test_full_pipeline_writes_enriched_json(self):
        scouting = {
            "duration_s": 2.0,
            "ball_track": mk_ball_track(),
            "track_samples": [
                {"id": 1, "samples": [{"frame": f, "court_x": 14.0, "court_y": 5.0}
                                      for f in range(0, 120, 5)]},
            ],
        }
        with tempfile.TemporaryDirectory() as d:
            inp = os.path.join(d, "scouting_data.json")
            outp = os.path.join(d, "enriched.json")
            with open(inp, "w") as f:
                json.dump(scouting, f)

            data = tm.run(inp, out_path=outp, plot=False)
            self.assertIsNotNone(data)
            self.assertTrue(os.path.exists(outp))

            with open(outp) as f:
                enriched = json.load(f)
            self.assertEqual(enriched["touch_machine_version"], tm.VERSION)
            self.assertGreaterEqual(len(enriched["touches"]), 2)
            self.assertGreaterEqual(len(enriched["rallies"]), 1)
            # rallies serializadas no arrastran la lista anidada de touches.
            self.assertNotIn("touches", enriched["rallies"][0])


if __name__ == "__main__":
    unittest.main()
