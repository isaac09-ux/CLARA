# Changelog

## v0.6.1 (current) — Filtro de horizonte & VballNet stride

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
