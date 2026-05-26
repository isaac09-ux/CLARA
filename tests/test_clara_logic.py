"""Tests de no-regresion de la logica pura de CLARA.

Sin dependencias extra (stdlib unittest). Corre con:
    .venv\\Scripts\\python.exe -m unittest discover tests

Blindan las formulas que rediseñamos a mano (score de calidad simetrico,
blend de estabilidad, stitch de tracks). Si alguien las toca y cambia el
comportamiento, estos tests truenan.
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import clara
import clara_report


def mk_track(n, x=1.0, y=1.0):
    """Track sintetico de n samples en una posicion fija."""
    return [{"frame": i * 5, "court_x": x, "court_y": y, "conf": 0.7} for i in range(n)]


class TestQualityScore(unittest.TestCase):
    def test_conteo_exacto_da_tracks_llenos(self):
        # 6 tracks para 6 jugadoras (media cancha) -> tracks 30/30
        tracks = {i: mk_track(100) for i in range(6)}
        zones = {f"A{z}": 50 for z in range(1, 7)}
        total, bd = clara.compute_quality_score(
            tracks, zones, ball_frames=0, rejected={},
            total_samples=200, expected_tracks=6, half_court=True)
        self.assertTrue(bd["tracks"].startswith("30/30"))

    def test_sobre_deteccion_tanquea_tracks(self):
        # 48 tracks para 6 jugadoras: el bug viejo daba 30/30. Ahora debe ser bajo.
        tracks = {i: mk_track(30) for i in range(48)}
        zones = {f"A{z}": 50 for z in range(1, 7)}
        total, bd = clara.compute_quality_score(
            tracks, zones, ball_frames=0, rejected={},
            total_samples=650, expected_tracks=6, half_court=True)
        track_pts = int(bd["tracks"].split("/")[0])
        self.assertLessEqual(track_pts, 6,
                             f"sobre-deteccion deberia tanquear tracks, dio {track_pts}")

    def test_sub_deteccion_tambien_penaliza(self):
        # 3 tracks para 6 esperadas -> ratio 3/6 = 0.5 -> 15/30
        tracks = {i: mk_track(100) for i in range(3)}
        total, bd = clara.compute_quality_score(
            tracks, {}, 0, {}, 200, 6, True)
        self.assertEqual(int(bd["tracks"].split("/")[0]), 15)

    def test_estabilidad_es_blend_con_componentes(self):
        tracks = {i: mk_track(100) for i in range(6)}
        total, bd = clara.compute_quality_score(
            tracks, {}, 0, {}, 200, 6, True)
        # 6/6 tracks, top6 duran 100/200=50% -> persist 1.0, consol 1.0 -> 10/10
        self.assertTrue(bd["estabilidad"].startswith("10/10"))
        self.assertIn("persist", bd["estabilidad"])
        self.assertIn("consol", bd["estabilidad"])

    def test_ball_rate_alto_da_20(self):
        tracks = {0: mk_track(50)}
        total, bd = clara.compute_quality_score(
            tracks, {}, ball_frames=100, rejected={},
            total_samples=100, expected_tracks=6, half_court=True)
        self.assertTrue(bd["balon"].startswith("20/20"))


class TestStitch(unittest.TestCase):
    def test_une_fragmentos_secuenciales_cercanos(self):
        raw = {
            1: [{"frame": f, "court_x": 1.0, "court_y": 1.0} for f in (0, 5, 10)],
            2: [{"frame": f, "court_x": 1.2, "court_y": 1.1} for f in (20, 25, 30)],
        }
        out = clara.stitch_tracks(raw, fps=30, stride=5, max_gap_s=5.0, max_jump_m=6.0)
        self.assertEqual(len(out), 1, "fragmentos contiguos y cercanos deben unirse")

    def test_no_une_si_el_hueco_temporal_es_grande(self):
        raw = {
            1: [{"frame": f, "court_x": 1.0, "court_y": 1.0} for f in (0, 5)],
            2: [{"frame": f, "court_x": 1.0, "court_y": 1.0} for f in (500, 505)],
        }
        out = clara.stitch_tracks(raw, fps=30, stride=5, max_gap_s=5.0, max_jump_m=6.0)
        self.assertEqual(len(out), 2, "hueco > max_gap no debe unir")

    def test_no_une_si_el_salto_espacial_es_grande(self):
        raw = {
            1: [{"frame": f, "court_x": 1.0, "court_y": 1.0} for f in (0, 5)],
            2: [{"frame": f, "court_x": 50.0, "court_y": 50.0} for f in (10, 15)],
        }
        out = clara.stitch_tracks(raw, fps=30, stride=5, max_gap_s=5.0, max_jump_m=6.0)
        self.assertEqual(len(out), 2, "salto > max_jump no debe unir")


class TestGeometry(unittest.TestCase):
    def test_is_in_court(self):
        self.assertTrue(clara.is_in_court(4.5, 4.5, 9, 9))
        self.assertFalse(clara.is_in_court(20, 20, 9, 9))
        # margen: justo afuera pero dentro del margen default (0.5)
        self.assertTrue(clara.is_in_court(9.3, 4.5, 9, 9))


class TestRallies(unittest.TestCase):
    @staticmethod
    def _run(start, end, step=5, x=2.0, y=5.0):
        return [{"frame": f, "court_x": x, "court_y": y}
                for f in range(start, end + 1, step)]

    def test_separa_rallies_por_hueco_grande(self):
        # rally A (0-150f) + hueco de 200f + rally B (350-450f) -> 2 rallies
        pts = self._run(0, 150) + self._run(350, 450)
        rallies, summ = clara.segment_rallies(pts, fps=30, court_h=18)
        self.assertEqual(summ["n_rallies"], 2)
        self.assertEqual(summ["n_serves"], 2)

    def test_huecos_chicos_no_cortan_el_rally(self):
        # un solo rally continuo (gaps de 5f << 60f) -> 1 rally
        pts = self._run(0, 300)
        rallies, summ = clara.segment_rallies(pts, fps=30, court_h=18)
        self.assertEqual(summ["n_rallies"], 1)
        self.assertAlmostEqual(rallies[0]["duration_s"], 10.0, places=1)

    def test_grupos_espurios_se_descartan(self):
        # 2 puntos aislados (< min_points) no son un rally
        pts = self._run(0, 150) + [{"frame": 800, "court_x": 1, "court_y": 1},
                                   {"frame": 803, "court_x": 1, "court_y": 1}]
        rallies, summ = clara.segment_rallies(pts, fps=30, court_h=18)
        self.assertEqual(summ["n_rallies"], 1)

    def test_lado_de_saque_por_posicion(self):
        # cancha completa: inicio en y<9 -> lado A, y>9 -> lado B
        a = self._run(0, 150, y=2.0)
        b = self._run(350, 500, y=15.0)
        rallies, summ = clara.segment_rallies(a + b, fps=30, court_h=18,
                                              half_court=False)
        sides = {r["serve_side"] for r in rallies}
        self.assertEqual(sides, {"A", "B"})

    def test_sin_balon_no_truena(self):
        rallies, summ = clara.segment_rallies([], fps=30)
        self.assertEqual(summ["n_rallies"], 0)
        self.assertEqual(rallies, [])


class TestReport(unittest.TestCase):
    def test_quality_verdict_umbrales(self):
        self.assertEqual(clara_report.quality_verdict(85)[0], "excelente")
        self.assertEqual(clara_report.quality_verdict(72)[0], "bueno")
        self.assertEqual(clara_report.quality_verdict(50)[0], "regular")
        self.assertEqual(clara_report.quality_verdict(20)[0], "bajo")

    def test_fmt_secs(self):
        self.assertEqual(clara_report.fmt_secs(45), "45s")
        self.assertEqual(clara_report.fmt_secs(89), "1m 29s")
        self.assertEqual(clara_report.fmt_secs(None), "0s")


if __name__ == "__main__":
    unittest.main()
