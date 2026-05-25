# Changelog

## v0.7.0 (current) — Identificación de jugadora por número de jersey

- NEW: módulo `src/jersey_id.py`. Convierte un track anónimo de ByteTrack en
  una jugadora con nombre. Lee el número del jersey con OCR (easyocr) y cruza
  contra un roster conocido. Run con `--identify --roster roster.json`.
- Diseño: **votación por track, no por frame**. El número solo es legible en
  una fracción de los frames; cada lectura de OCR se acumula como un voto
  ponderado por confianza sobre el track, y el número con más peso gana. Una
  mala lectura aislada queda enterrada por las buenas.
- **Conjunto cerrado**: solo cuentan votos por números presentes en el roster.
  Un OCR que lee un número inexistente se descarta — filtra casi todos los
  errores sin entrenar nada. Consecuencia útil: las rivales (fuera del roster)
  se quedan como tracks anónimos, que es lo deseable para scouting.
- Reusa la pose de `--pose rtmlib` cuando está disponible: hombros (5,6) y
  caderas (11,12) recortan la zona del número con precisión. Sin pose, cae a
  un recorte por proporciones del bbox.
- Preprocesamiento con CLAHE para rescatar números de bajo contraste (kit Casa
  hueso/rosa palo).
- Costo de OCR acotado por `--id-stride` (cada N muestras) y por un tope
  interno de observaciones por track (`max_obs_per_track`).
- JSON: cada `tracks[]` gana un campo `identity` con `{number, name,
  confidence, votes, weight}`; nuevo campo raíz `identified_players`.
- ADITIVO: con `--identify` apagado, CLARA se comporta idéntico a v0.6.2.
- Nuevo CLI: `--identify`, `--roster <path>` (default `roster.json`),
  `--id-stride N` (default 6). Nueva dependencia opcional: `easyocr`.
- Limitación conocida: una rival con un número que coincide con el roster se
  etiqueta mal. Mitigación: filtrar por `side` (A/B) en el reporte.

## v0.6.2 — Filtro de borde inferior (fg_at_edge)

- FIX: `classify_detection` ya no rechaza toda caja cuyo borde inferior toque el frame. El check `y2 >= frame_h - 5` descartaba CUALQUIER detección pegada al borde, asumiendo que era público cortado. En video de baja resolución (p.ej. 848x478) las jugadoras llenan buena parte del alto y muchas tocan el borde en algún frame — el filtro tiró 3699 detecciones válidas en un caso real, dejando 0 tracks y score 0/100. Ahora `fg_at_edge` solo rechaza si la caja toca el borde **y además** es grande (>35% del alto), patrón de un espectador/entrenador en primer plano. Jugadoras lejanas cuyos pies se cortan un poco se conservan; si quedaran fuera de cancha, `is_in_court()` las filtra tras la proyección. Los balones tocando el borde ya no se descartan (antes: `ball_fg_at_edge`).
- Nuevo parámetro `edge_reject_height_ratio` en `classify_detection` (default 0.35).

## v0.6.1 — Filtro de horizonte & VballNet stride

- FIX: `classify_detection` ya no rechaza por banda de píxeles en torno a `court_horizon_y`. La banda fija (±5%/+10% del alto) asumía cámara casi frontal; en ángulos elevados u oblicuos rechazaba jugadoras reales (un día tiró 1911 detecciones y dejó 1 sola jugadora en pie). El filtrado válido se hace post-proyección con `is_in_court()`, que ya estaba activo aguas abajo. `court_horizon_y` sigue aceptado en la calibración para retrocompatibilidad pero es ignorado; se conservan los checks de altura/ancho relativos y el guard de borde inferior.
- NEW: `--vballnet-stride N` (default 1). Alimenta el buffer de 9 frames de VballNet sólo cada N frames del video — reduce ~N× las inferencias y el footprint de RAM del session ONNX que reventaba en clips largos. Recall baja proporcionalmente.

## v0.6 — Multimodal scouting

### New features
- **VballNet ball detector**: TrackNetV4-based motion detection. Run with `--ball-detector vballnet --vballnet-model VballNet.onnx`. ~70% recall in any gym without training. Wrapper: `src/ball_vballnet.py`.
- **rtmlib pose estimation**: RTMPose 17-keypoint extraction per player. Run with `--pose rtmlib`. Adds `torso_lean_deg`, `stance_width_px`, `knee_flexion_deg` per track. Wrapper: `src/pose_rtmlib.py`.
- **Pose sample image**: when `--pose rtmlib`, CLARA renders `pose_sample.png` with skeletons overlaid on a representative frame.
- **JSON output**: now includes `ball_detector`, `pose_mode`, and per-track `pose_stats`.

### CLI changes
- New: `--ball-detector {yolo,vballnet}` (default `yolo`)
- New: `--vballnet-model <path>` (required if `--ball-detector vballnet`)
- New: `--pose {none,rtmlib}` (default `none`)

## v0.5.1 — Bug fixes

- FIX: Quality score rebalanced. Ball detection rate of 10% no longer gives full 20/20. Now requires 50%+ for max. Copa test: 14.4% balls → 78/100 (was 94/100).
- FIX: Stability score formula corrected. 50%+ presence for max.
- FIX: Foreground filter now applied to ball detections too.
- FIX: `out.mkdir(parents=True)` for nested paths.
- FIX: Ball loop uses `cap.set(POS_FRAMES)` to skip frames efficiently.
- FIX: `zone_for_court_pos` and `is_in_court` margin consistency.

## v0.5

- Half-court mode for partial court views
- Foreground filter for spectators
- Quality score 0-100 with breakdown
- Diagnostic image showing rejected detections

## v0.4

- HTML report generator
- Topdown with zones A1-A6/B1-B6 marked
- Custom ball model support via `--ball-model`

## v0.3

- Python pipeline: YOLOv8 + ByteTrack + homography + zone analytics

## v0.1-0.2

- HTML/JS prototype
