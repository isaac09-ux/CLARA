# CLARA v0.6 integrations

## VballNet (motion-based ball detector)

Source: https://github.com/asigatchov/fast-volleyball-tracking-inference

VballNet is a TrackNetV4-based model that detects the ball using 9 consecutive grayscale frames. Because it uses motion rather than appearance, it ignores static objects (ceiling lamps, banners) that confuse single-frame detectors.

### Setup

1. Download a pretrained ONNX model from the upstream repo:
   ```bash
   git clone https://github.com/asigatchov/fast-volleyball-tracking-inference
   cp fast-volleyball-tracking-inference/models/VballNetV1_seq9_grayscale_330_h288_w512.onnx .
   ```

2. Install ONNX runtime: `pip install onnxruntime`

3. Run CLARA with VballNet:
   ```bash
   python src/clara.py video.mp4 --calibration cal.json \
       --ball-detector vballnet \
       --vballnet-model VballNetV1_seq9_grayscale_330_h288_w512.onnx
   ```

### Benchmark on Las Chispas footage

| Gym | Recall | Notes |
|---|---|---|
| Copa (where YOLO custom was trained) | 34.6% | Outperforms custom YOLO (14.4%) |
| New gym AZUL (unseen) | 68.7% | Custom YOLO would need new labels |

### Architecture

- Input: tensor `(1, 9, 288, 512)` — 9 grayscale frames stacked
- Output: heatmaps per frame, peak = ball position
- Inference: ~30 FPS on CPU
- Model size: small (~200KB ONNX)

The `src/ball_vballnet.py` adapter handles the sequence buffering and heatmap-to-coordinates conversion. It returns CLARA-compatible ball detection dicts.

---

## rtmlib (RTMPose keypoints)

Source: https://github.com/Tau-J/rtmlib

rtmlib is a lightweight wrapper around RTMPose models. No mmcv/mmpose/mmdet dependencies. Apache 2.0.

### Setup

```bash
pip install rtmlib
```

First run downloads the model (~50MB) to `~/.cache/rtmlib/`.

### Usage

```bash
python src/clara.py video.mp4 --calibration cal.json --pose rtmlib
```

### Output

For each track that has valid pose detections, CLARA records in `scouting_data.json`:

```json
{
  "id": 5,
  "samples": 234,
  "pose_stats": {
    "samples_with_pose": 187,
    "torso_lean_deg_avg": 12.4,
    "torso_lean_deg_max": 38.7,
    "stance_width_px_avg": 45.2,
    "knee_flexion_deg_avg": 142.1
  }
}
```

### Biomechanical metrics

The wrapper `src/pose_rtmlib.py` computes 3 base metrics:

- **`torso_lean_deg`**: angle of torso from vertical. 0 = upright. Positive = leaning forward (typical of approach to spike). Useful for analyzing spike approach mechanics.
- **`stance_width_px`**: ankle-to-ankle distance in pixels. Wider = more defensive stance.
- **`knee_flexion_deg`**: hip-knee-ankle angle. 180 = straight leg. <150 = good defensive squat.

These are derived from the 17 COCO keypoints. Easy to extend in `pose_rtmlib.py`:

```python
def my_metric(keypoints, scores, min_score=0.5):
    # keypoints[5] = left shoulder, [9] = left wrist, etc.
    if scores[9] < min_score: return None
    # ... compute and return
```

### Performance cost

Pose adds ~3-5x to processing time (one inference per player per sampled frame). For long videos, expect tradeoffs:
- 30s clip @ stride 5 + 6 players: +10-15s
- 5min match @ stride 5 + 12 players: +3-5 min

For exploration runs, leave `--pose none`. Enable only when analyzing specific plays.

---

## touch_machine (touch & rally detection)

`src/touch_machine.py` is a **standalone post-processor**: it reads a
`scouting_data.json` produced by CLARA and writes an enriched copy with
per-touch and per-rally data. It never re-reads the video, so you can re-tune
thresholds and re-run it in seconds.

### What it produces

- **`touches[]`** — each frame where the ball trajectory breaks (a contact),
  with image/court position, the acceleration magnitude that triggered it, and
  the attributed player (`track_id`, `player_distance_m`).
- **`rallies[]`** — touches grouped into continuous play by ball silence
  (default gap 3.5 s), with per-player touch counts.

### How it works

1. Aligns `ball_track` into a per-frame array, interpolating short gaps.
2. Computes the **magnitude of the ball's acceleration** (px/frame²) — it
   spikes wherever the ball abruptly changes direction (a contact).
3. Picks peaks (`scipy.signal.find_peaks`): above an acceleration percentile,
   spaced apart, and prominent over the local baseline.
4. Attributes each touch to the **nearest player** in court coordinates,
   interpolating every track's position to the touch frame.
5. Groups touches into rallies by temporal gap.

### Prerequisites (auto-satisfied by CLARA v0.8.1+)

`scouting_data.json` must contain two fields, both emitted by `clara.py`
automatically:

| Field | Provides |
|---|---|
| `ball_track` | real ball detections in pixels (`img_x/img_y`) + `court_x/court_y` |
| `track_samples` | per-frame court positions of each player track |

`track_samples` is a **dedicated** field — it does **not** replace the
coach-facing `tracks` summary (which keeps `samples` as a count plus
reliability, zones, speed, etc.). The post-processor reads `track_samples`;
the HTML report keeps using `tracks`.

> Older JSONs (pre-0.8.1) lack `track_samples` — just re-run CLARA. For
> backward compatibility, `touch_machine` also accepts a `tracks` field that
> already contains the raw sample lists.

### Install & run

```bash
pip install scipy        # only required extra (matplotlib optional, for --plot)

python src/touch_machine.py results/scouting_data.json
# -> results/scouting_data_enriched.json

python src/touch_machine.py results/scouting_data.json --out enriched.json --plot
# --plot writes a PNG: ball-y track + acceleration with detected touches marked
```

### Tuning knobs

All near the top of `touch_machine.py`:

- `ACCEL_PCT_THRESHOLD` (92) — acceleration percentile to qualify as a peak.
- `MIN_TOUCH_SPACING_FRAMES` (12) — minimum frames between touches (~0.2 s @ 60 fps).
- `MIN_PEAK_PROMINENCE` (3.0) — how much a peak must stand out from baseline.
- `MAX_ATTRIBUTION_DIST_M` (1.5) — beyond this, a touch is left unattributed.
- `RALLY_GAP_SECONDS` (3.5) — ball-silence gap that ends a rally.

> Note: CLARA already emits its own ball-heuristic `rallies`/`play_summary`
> (segmented from ball motion). `touch_machine`'s `rallies[]` are a separate,
> touch-based view of play — keep the two distinct when reading the JSON.
