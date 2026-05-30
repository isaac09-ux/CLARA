"""
court/ — Subpaquete de geometría y calibración de cancha (FIVB) para CLARA.

Integra la parte de homografía + geometría de cancha del proyecto
`volleyball_analytics`. Está DESACOPLADO del pipeline principal `src/clara.py`:
no se importa desde clara.py ni lo modifica.

Qué aporta (additivo, sin duplicar clara.py)
--------------------------------------------
  court_geometry   : dimensiones oficiales FIVB, landmarks con nombre,
                     clasificación de zonas 1-6 (enteros) y render top-down.
  CourtCalibrator  : homografía manual (4 ó 6 puntos por clic) y automática
                     (modelo de segmentación de cancha), con save/load JSON
                     y transformaciones imagen<->metros.

Qué DUPLICA el pipeline principal (no cableado en clara.py)
-----------------------------------------------------------
  analyzer.CLARACourtAnalyzer : corre su propio YOLO+ByteTrack y detección de
                     balón. clara.py ya hace esto con más robustez (ROI,
                     stitch_tracks, vballnet, trayectoria, pose). Se conserva
                     como pipeline ALTERNATIVO / de referencia. Ver analyzer.py.

Uso típico (solo geometría + calibración, sin re-detectar)
----------------------------------------------------------
    from court import court_geometry as cg
    from court.court_calibration import CourtCalibrator

    cal = CourtCalibrator.load("cancha.json")
    x_m, y_m = cal.image_to_court([(px, py)])[0]
    zona = cg.classify_zone(x_m, y_m, cg.side_of(y_m))
"""

from . import court_geometry
from .court_geometry import (
    COURT_WIDTH, COURT_LENGTH, NET_Y, ATTACK_LINE_A, ATTACK_LINE_B,
    COURT_CORNERS_M, LANDMARKS_M,
    in_court, side_of, classify_zone,
    m2px_render, make_topdown_canvas, draw_court,
)
from .court_calibration import CourtCalibrator, corners_from_mask, order_points

__all__ = [
    "court_geometry",
    "COURT_WIDTH", "COURT_LENGTH", "NET_Y", "ATTACK_LINE_A", "ATTACK_LINE_B",
    "COURT_CORNERS_M", "LANDMARKS_M",
    "in_court", "side_of", "classify_zone",
    "m2px_render", "make_topdown_canvas", "draw_court",
    "CourtCalibrator", "corners_from_mask", "order_points",
]
