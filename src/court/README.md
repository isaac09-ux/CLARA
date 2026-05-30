# Subpaquete `court/` · Geometría + homografía de cancha (FIVB)

Integra la parte de **homografía + geometría de cancha** de
[`volleyball_analytics`](https://github.com/masouduut94/volleyball_analytics)
como **subpaquete autocontenido** de CLARA. Transforma posiciones imagen →
**coordenadas reales (metros)**, asigna **zona oficial 1–6** y genera **vista
top-down**.

> **Desacoplado a propósito.** Este subpaquete **no se importa desde
> `src/clara.py` ni lo modifica**. El pipeline de producción sigue siendo
> `clara.py`. Aquí viven utilidades de geometría/calibración (additivas) y un
> pipeline alternativo de referencia (duplicado — ver más abajo).

---

## Archivos

| Archivo | Qué hace | ¿Duplica `clara.py`? |
|---|---|---|
| `court_geometry.py` | Dimensiones FIVB, landmarks, zonas 1–6 (enteros), render top-down. Solo matemática. | **No** — additivo |
| `court_calibration.py` | `CourtCalibrator`: homografía manual (clic) o auto (modelo de segmentación), save/load JSON, transformar imagen↔metros. | **No** — additivo |
| `analyzer.py` | `CLARACourtAnalyzer`: pipeline por frame (jugadores+ByteTrack, balón, proyección, zonas, top-down, export). | **Sí** — ver aviso |
| `run_clara.py` | CLI que une calibración + analyzer (pipeline alternativo). | Sí (usa `analyzer.py`) |
| `test_core.py` | Prueba la matemática (homografía, zonas, esquinas). No requiere modelos. | — |

---

## ⚠️ `analyzer.py` duplica tu detección/tracking

Revisado contra `src/clara.py`. `CLARACourtAnalyzer` corre **su propio**
detector + tracker y **duplica**, con menos robustez, lo que ya hace el
pipeline principal:

| Etapa | `analyzer.py` (volleyball_analytics) | `clara.py` (producción) |
|---|---|---|
| Jugadores + tracking | `YOLO.track(bytetrack.yaml, classes=[0])` | YOLO11m + `bytetrack_clara.yaml` tuneado + ROI por polígono + `stitch_tracks()` + pose |
| Balón | 1 `YOLO.predict` por frame, máx conf | backends `yolo`/`vballnet` + `filter_ball_tracks()` + `reconstruct_trajectory()` |
| Proyección | `cal.image_to_court()` | `project(H, ...)` |
| Zonas | `classify_zone()` → enteros 1–6 | `zone_for_court_pos()` → strings `"A4"` |
| Top-down / export | `render_topdown` / CSV-JSON | `save_topdown()` + export propio |

**Decisión:** no se cablea `analyzer.py` dentro de `clara.py`. Se conserva como
pipeline **alternativo / de prototipado** (p. ej. para probar rápido los pesos
`ball/court` de volleyball_analytics o validar la geometría FIVB aislada).

Si algún día quieres aprovechar **solo** la proyección + zonas FIVB sin
re-detectar, alimenta las cajas que ya produce `clara.py` a la homografía de
`CourtCalibrator` y a `court_geometry.classify_zone` — no vuelvas a correr
YOLO/ByteTrack.

---

## Uso recomendado (solo additivo, sin duplicar)

```python
from court import court_geometry as cg
from court.court_calibration import CourtCalibrator

cal = CourtCalibrator.load("cancha.json")          # o .from_clicks(frame)
x_m, y_m = cal.image_to_court([(px, py)])[0]        # pixel -> metros
zona = cg.classify_zone(x_m, y_m, cg.side_of(y_m))  # 1..6
```

### Tests (no requieren modelos)

```bash
cd src
python -m court.test_core
```

### Pipeline alternativo (CLI de referencia)

```bash
cd src
# manual (trípode fijo): clic en 4 esquinas (o 6 con --six-points)
python -m court.run_clara --video partido.mp4 --mode manual --calib cancha.json \
    --ball-model weights/ball/weights/best.pt
# reusar calibración:
python -m court.run_clara --video partido2.mp4 --mode load --calib cancha.json
# auto con modelo de segmentación de cancha:
python -m court.run_clara --video broadcast.mp4 --mode auto \
    --court-model weights/court/weights/best.pt
```

---

## Notas de compatibilidad

- **Dos convenciones de coordenadas independientes.** Este subpaquete usa
  lado A en `y=0` y zonas como enteros `1–6`. `clara.py` / `court_keypoints.py`
  usan otra convención (cámara cercana en `y=18`, zonas `"A4"`/`"B1"`). **No
  mezcles** una calibración de un sistema con el otro.
- **Formato de calibración distinto.** El JSON de `CourtCalibrator`
  (`src_pts`/`dst_pts`/`H`) **no** es el `data/cal.json` del pipeline principal
  (`homography_matrix`/`pixels_per_meter`/...). No los cargues cruzados.
- **Pesos.** Los modelos `ball/` y `court/` de volleyball_analytics son
  externos (licencia GPL-2.0 en ese repo). Para `--mode auto` necesitas
  `court/weights/best.pt`; para balón, `ball/weights/best.pt`.

---

## Caveats honestos (heredados)

1. **Balón en el aire (paralaje).** La homografía asume el punto sobre el plano;
   un balón alto se proyecta con error. Es exacto cerca del piso. Cada fila de
   balón lleva el flag `low`.
2. **Handedness de zonas.** Verifica con tu primer video; si está espejado, usa
   `flip_lr=True` (o `--flip-lr` en el CLI).
3. **Auto-calibración frágil con oclusión.** Si los jugadores tapan las líneas,
   el modelo de cancha puede dar esquinas imperfectas. Para trípode fijo, la
   calibración manual una vez es más robusta.
