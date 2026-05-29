"""Tests de no-regresion de la logica pura de CLARA.

Sin dependencias extra (stdlib unittest). Corre con:
    .venv\\Scripts\\python.exe -m unittest discover tests

Blindan las formulas que rediseñamos a mano (score de calidad simetrico,
blend de estabilidad, stitch de tracks). Si alguien las toca y cambia el
comportamiento, estos tests truenan.
"""
import sys
import os
import types
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import clara
import clara_report
import ball_trajectory


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


class TestROI(unittest.TestCase):
    """ROI de imagen. Con H identidad y ppm=1, metros == píxeles de imagen,
    así que el polígono es la cancha expandida por el margen directamente."""
    IDENT = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    def test_poligono_es_cancha_mas_margen(self):
        poly = clara.court_roi_polygon(self.IDENT, 9, 9, ppm=1, margin_m=2.0)
        xs, ys = poly[:, 0], poly[:, 1]
        self.assertEqual((xs.min(), xs.max()), (-2, 11))
        self.assertEqual((ys.min(), ys.max()), (-2, 11))

    def test_punto_dentro_pasa(self):
        poly = clara.court_roi_polygon(self.IDENT, 9, 9, ppm=1, margin_m=2.0)
        self.assertTrue(clara.point_in_polygon(4.5, 4.5, poly))   # centro
        self.assertTrue(clara.point_in_polygon(10.0, 4.5, poly))  # dentro del margen

    def test_punto_fuera_se_descarta(self):
        poly = clara.court_roi_polygon(self.IDENT, 9, 9, ppm=1, margin_m=2.0)
        self.assertFalse(clara.point_in_polygon(50.0, 50.0, poly))  # tribuna lejana
        self.assertFalse(clara.point_in_polygon(4.5, 30.0, poly))   # fondo lejano

    def test_roi_none_no_filtra(self):
        # ROI desactivado (--no-roi): todo punto pasa
        self.assertTrue(clara.point_in_polygon(999, 999, None))

    def test_homografia_no_invertible_da_none(self):
        singular = [[1, 0, 0], [2, 0, 0], [0, 0, 1]]  # filas dependientes
        self.assertIsNone(
            clara.court_roi_polygon(singular, 9, 9, ppm=1, margin_m=2.0))


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


class TestBallTrajectory(unittest.TestCase):
    """Reconstrucción de trayectoria (ball_trajectory.py). H identidad y ppm=1
    => court == img, así se valida la lógica sin una homografía real."""
    IDENT = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    @staticmethod
    def _flight(frames, ax, bx, ay, by, cy):
        """Vuelo sintético: img_x = ax*t+bx (recta), img_y = ay*t^2+by*t+cy."""
        out = []
        for t in frames:
            x = ax * t + bx
            y = ay * t * t + by * t + cy
            out.append({"frame": t, "img_x": x, "img_y": y,
                        "court_x": x, "court_y": y, "conf": 0.8})
        return out

    def test_rellena_huecos_de_un_vuelo(self):
        # Parábola limpia, sólo 1 de cada 3 frames detectado (0..18).
        det = self._flight(range(0, 19, 3), ax=6, bx=120, ay=0.8, by=-14, cy=300)
        dense, summ = ball_trajectory.reconstruct_trajectory(
            det, fps=30, H=self.IDENT, ppm=1, frame_w=1000)
        self.assertEqual(summ["real"], 7)
        self.assertEqual(summ["reconstructed_flights"], 1)
        # Rellena los 12 frames faltantes -> 19 puntos continuos.
        self.assertEqual(summ["total"], 19)
        self.assertEqual(summ["interpolated"], 12)
        frames = [d["frame"] for d in dense]
        self.assertEqual(frames, list(range(0, 19)))  # continuo y ordenado
        # El punto interpolado en t=1 cae sobre la parábola verdadera.
        p1 = next(d for d in dense if d["frame"] == 1)
        self.assertTrue(p1["interp"])
        self.assertAlmostEqual(p1["img_x"], 126.0, delta=0.5)   # 6*1+120
        self.assertAlmostEqual(p1["img_y"], 286.8, delta=0.5)   # .8-14+300

    def test_court_reproyectado_con_identidad(self):
        det = self._flight(range(0, 19, 3), ax=6, bx=120, ay=0.8, by=-14, cy=300)
        dense, _ = ball_trajectory.reconstruct_trajectory(
            det, fps=30, H=self.IDENT, ppm=1, frame_w=1000)
        # Con H identidad y ppm=1, court == img también en los interpolados.
        interp = next(d for d in dense if d["interp"])
        self.assertAlmostEqual(interp["court_x"], interp["img_x"], delta=0.1)
        self.assertAlmostEqual(interp["court_y"], interp["img_y"], delta=0.1)

    def test_no_interpola_con_pocos_puntos(self):
        # 3 puntos (< min_pts_fit=4) -> passthrough, sin relleno.
        det = self._flight([0, 3, 6], ax=6, bx=120, ay=0.8, by=-14, cy=300)
        dense, summ = ball_trajectory.reconstruct_trajectory(
            det, fps=30, H=self.IDENT, ppm=1, frame_w=1000)
        self.assertEqual(summ["interpolated"], 0)
        self.assertEqual(summ["real"], 3)
        self.assertTrue(all(not d["interp"] for d in dense))

    def test_no_puentea_hueco_temporal(self):
        # Dos vuelos separados por un hueco enorme (>> max_gap_s*fps).
        a = self._flight(range(0, 19, 3), ax=6, bx=120, ay=0.8, by=-14, cy=300)
        b = self._flight(range(200, 219, 3), ax=-6, bx=900, ay=0.8, by=-14, cy=300)
        dense, summ = ball_trajectory.reconstruct_trajectory(
            a + b, fps=30, H=self.IDENT, ppm=1, frame_w=1000, max_gap_s=1.0)
        self.assertGreaterEqual(summ["flights"], 2)
        # Ningún frame inventado dentro del hueco muerto (19..199).
        gap_frames = [d["frame"] for d in dense if 19 <= d["frame"] <= 199]
        self.assertEqual(gap_frames, [])

    def test_corta_en_contacto(self):
        # Un solo run temporal, pero la x se INVIERTE a mitad (un toque): una
        # sola recta no ajusta -> split-and-fit debe partirlo en >=2 vuelos.
        f1 = self._flight([0, 2, 4, 6, 8], ax=10, bx=100, ay=0.5, by=-9, cy=300)
        # segundo vuelo: x baja (rebote/devolución), arranca donde quedó f1.
        f2 = []
        for t in [10, 12, 14, 16, 18]:
            x = 180 - 10 * (t - 8)        # invierte dirección horizontal
            y = 0.5 * (t - 9) ** 2 - 5 * (t - 9) + 250
            f2.append({"frame": t, "img_x": x, "img_y": y,
                       "court_x": x, "court_y": y, "conf": 0.8})
        dense, summ = ball_trajectory.reconstruct_trajectory(
            f1 + f2, fps=30, H=self.IDENT, ppm=1, frame_w=1000, max_gap_s=1.0)
        self.assertGreaterEqual(summ["flights"], 2,
                                "la inversión de dirección es un contacto")
        # No hay frames duplicados pese al corte.
        frames = [d["frame"] for d in dense]
        self.assertEqual(len(frames), len(set(frames)))

    def test_sin_balon_no_truena(self):
        dense, summ = ball_trajectory.reconstruct_trajectory(
            [], fps=30, H=self.IDENT, ppm=1)
        self.assertEqual(dense, [])
        self.assertEqual(summ["total"], 0)
        self.assertEqual(summ["interpolated"], 0)


class TestVballNetCache(unittest.TestCase):
    """Cache en disco de la pasada de VballNet (la parte lenta).

    Se stubea detect_balls para CONTAR cuantas veces corre la pasada: el cache
    debe saltarla en re-corridas y reinvalidarse al cambiar video/modelo/stride.
    """

    def setUp(self):
        self.calls = {"n": 0}
        fake = types.ModuleType("ball_vballnet")

        def detect_balls(video_path, model_path, threshold=0.5, stride=1,
                         verbose=True):
            self.calls["n"] += 1
            return [{"frame": 0, "x": 1.0, "y": 2.0, "radius": 3.0,
                     "confidence": 0.9}]

        fake.detect_balls = detect_balls
        self._patch = mock.patch.dict(sys.modules, {"ball_vballnet": fake})
        self._patch.start()
        self._tmp = tempfile.TemporaryDirectory()
        self.video = os.path.join(self._tmp.name, "match.mp4")
        with open(self.video, "wb") as f:
            f.write(b"x" * 128)
        self.cache = os.path.join(self._tmp.name, "c.json")

    def tearDown(self):
        self._patch.stop()
        self._tmp.cleanup()

    def _detect(self, **over):
        kw = dict(video_path=self.video, model_path="VballNetV1.onnx",
                  threshold=0.5, stride=1, verbose=False, cache_path=self.cache)
        kw.update(over)
        return clara._vballnet_detect_cached(**kw)

    def test_miss_then_hit(self):
        r1 = self._detect()
        self.assertEqual(self.calls["n"], 1)
        self.assertTrue(os.path.exists(self.cache))
        r2 = self._detect()
        self.assertEqual(self.calls["n"], 1, "cache HIT no debe recomputar")
        self.assertEqual(r1, r2)

    def test_stride_change_invalidates(self):
        self._detect()
        self._detect(stride=3)
        self.assertEqual(self.calls["n"], 2, "cambiar stride invalida el cache")

    def test_use_cache_false_always_recomputes(self):
        self._detect(use_cache=False)
        self._detect(use_cache=False)
        self.assertEqual(self.calls["n"], 2)

    def test_corrupt_cache_falls_back(self):
        self._detect()
        with open(self.cache, "w") as f:
            f.write("{ broken json")
        self._detect()  # no debe reventar, recomputa
        self.assertEqual(self.calls["n"], 2)

    def test_key_is_calibration_independent(self):
        # El cache guarda coords CRUDAS en pixeles; la proyeccion a cancha se
        # rehace al cargar, asi que recalibrar no debe invalidarlo.
        k = clara._vballnet_cache_key(self.video, "m.onnx", 0.5, 1)
        self.assertEqual(
            set(k), {"video", "size", "mtime", "model", "threshold", "stride"})


if __name__ == "__main__":
    unittest.main()
