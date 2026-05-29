# CLARA Architecture

## Pipeline

```
video.mp4
    │
    ▼
┌─────────────────────────────────────────────────┐
│  1. YOLO11 + ByteTrack                          │
│     - Detect persons (class 0)                  │
│     - Detect balls (class 32 or custom .pt)     │
│     - Assign persistent track IDs               │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  2. Foreground filter (pre-projection)          │
│     - Reject bboxes > 55% frame height          │
│     - Reject bboxes > 40% frame width           │
│     - Reject touching bottom edge               │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  3. Projection + in-court / ROI filter          │
│     - People: project feet -> court (m),        │
│       drop if outside court OR outside the      │
│       image play ROI (court_roi_polygon)        │
│     - Ball: kept/dropped in PIXELS, not court   │
│       coords (aerial balls project off-court)   │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  4. Track filtering                             │
│     - Drop tracks with <15 samples              │
│     - Drop tracks <50% in-court                 │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  5. Zone analytics                              │
│     - Map court positions to zones A1-A6/B1-B6  │
│     - Count visits per zone                     │
│     - Split first-half vs second-half           │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  6. Quality scoring (0-100)                     │
│     - Self-evaluation across 5 dimensions       │
│     - Honest "trust this run?" signal           │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
   scouting_data.json
   topdown.png + topdown_first/second_half.png
   diagnostic.png
   scouting_report.html (via clara_report.py)
```

## Calibration model

CLARA requires a one-time calibration per camera position. The calibration JSON maps:

- 4 pixel corners of the court (or half-court) → court coordinates in meters
- A 3×3 homography matrix `H` derived from the corners
- Optional: `half_court: true` for partial views
- `court_horizon_y` is accepted for legacy calibrations but ignored as of v0.6.1 — see Tradeoffs

The convention for full court (9×18 m):
- `cercana_izq, cercana_der` → bottom of frame (closest sideline)
- `lejana_der, lejana_izq` → top of frame (far sideline)

## Zone mapping

Volleyball uses 6 zones per side. The numbering is mirrored between teams (each team's "zone 4" is on their own left). CLARA's `zone_for_court_pos` encodes this directly so coaches see zones from each team's perspective.

```
LADO A (top half of topdown):
    A4   A3   A2     ← front row (near net)
    A5   A6   A1     ← back row (near endline)
─── NET ──────────
LADO B (bottom half of topdown):
    B2   B3   B4     ← front row (near net)
    B1   B6   B5     ← back row (near endline)
```

## Why a quality score?

CLARA processes amateur volleyball footage with imperfect conditions: handheld cameras, partial court visibility, spectators in frame, low resolution. Without self-assessment, coaches might trust contaminated data.

The score tells you which runs are publishable and which need re-grabbing. It's not a benchmark of CLARA — it's a transparency mechanism for the coach.

## Tradeoffs

- **Stride**: higher = faster but loses tracking continuity. Default 5 = 6 Hz sampling, sufficient for zone analytics, not for fast events.
- **Foreground filter**: bbox-size and bottom-edge guards only. The earlier `court_horizon_y` band assumed a flat-ish viewing angle; on elevated/oblique cameras the court spans ~240 px vertically while the band accepted only ~108 px, so it threw away most players. Player filtering now happens post-projection via `is_in_court()` **plus** an image-space play ROI (`court_roi_polygon` + `point_in_polygon`): the homography extends the floor plane, so spectators behind the endline and the far-side bench project *inside* the numeric court range and `is_in_court()` alone keeps them. The ROI is the court footprint (expanded `--roi-margin` metres, default 2) projected back into the image, where that crowd *is* separated. Disable with `--no-roi`; verify the green polygon in `diagnostic.png`.
- **VballNet RAM**: full-rate inference on long clips can run the ONNX session out of memory mid-video. Pass `--vballnet-stride 2` (or higher) to feed the 9-frame buffer only every Nth video frame — linear cost/RAM savings at the price of recall.
- **Ball geometry**: the single-camera homography maps the *floor* plane, so a ball high in the air projects to where its sight-line hits the floor — well outside the court lines. Filtering the ball by its court projection (the old behaviour) therefore killed ~96% of real aerial detections. CLARA now filters the ball in **pixels** against the court footprint stretched upward for headroom (`ball_valid_region`), keeping `court_x/court_y` only as a floor-reliable hint. The same pixel-space view drives trajectory reconstruction (`ball_trajectory.py`): each free flight is a parabola in the image (constant gravity in `y`), so missed frames between contacts are fit and reprojected — marked `interp=True`, they densify the track for zones/tempo but never count toward recall or the quality score.
