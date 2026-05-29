# CLARA

**Computer vision tentacle of LUCIA — Las Chispas volleyball analytics**

CLARA processes match video into structured scouting data: player tracks, ball trajectories, zone-by-zone heatmaps, biomechanical pose data, and a self-reported quality score.

```
┌────────────────────────────────────────────────────────┐
│  CLARA · TENTACULO DE LUCIA                            │
│                                                        │
│   video.mp4  →  CLARA  →  scouting_data.json           │
│                       →  topdown.png                   │
│                       →  scouting_report.html          │
│                       →  pose_sample.png (opcional)    │
│                       →  diagnostic.png                │
└────────────────────────────────────────────────────────┘
```

## What CLARA does

- **Player tracking** via YOLO11 + ByteTrack
- **Ball detection** in 3 modes:
  - YOLO11 base (no setup, low recall)
  - YOLO11 custom (high recall in trained gym, requires labeling)
  - **VballNet** (motion-based, ~70% recall in any gym, **no training needed**)
- **Pose estimation** via RTMPose (optional): 17 keypoints per player
- **Player identification** (optional): reads jersey numbers via OCR and
  matches them to a known roster — turns anonymous track IDs into named players
- **Court projection** via homography from 4 corner points
- **Zone analytics** mapped to official volleyball zones 1-6 per side
- **Foreground filtering** discards spectators between camera and court
- **Quality scoring** 0-100 self-evaluation
- **Biomechanical analytics**: torso lean, stance width, knee flexion

## Quick start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Calibrate (one-time per camera position)
python src/setup_calibration.py video.mp4 --out cal.json

# 3. Process video (3 modes)

# A — Solo YOLO11 base (rápido, low recall en balón)
python src/clara.py video.mp4 --calibration cal.json --out results/

# B — Con modelo YOLO11 custom entrenado
python src/clara.py video.mp4 --calibration cal.json \
    --ball-detector yolo --ball-model clara_balon_v1.pt \
    --out results/

# C — Con VballNet (recomendado: motion-based, sin entrenar)
python src/clara.py video.mp4 --calibration cal.json \
    --ball-detector vballnet --vballnet-model VballNetV1.onnx \
    --out results/

# D — Full multimodal: VballNet + pose
python src/clara.py video.mp4 --calibration cal.json \
    --ball-detector vballnet --vballnet-model VballNetV1.onnx \
    --pose rtmlib \
    --out results/

# E — Con identificación de jugadora por número de jersey
#     (copia roster.example.json -> roster.json y pon tus jugadoras primero)
python src/clara.py video.mp4 --calibration cal.json \
    --identify --roster roster.json \
    --out results/

# 4. Generate HTML report
python src/clara_report.py results/scouting_data.json --topdown results/topdown.png
```

## Player identification (`--identify`)

CLARA's ByteTrack IDs are anonymous and reset every video: track #5 today is
not track #5 tomorrow. With `--identify`, CLARA reads the jersey number of
each track via OCR and matches it to a roster, so `scouting_data.json` carries
a real player name per track.

1. Copy `roster.example.json` to `roster.json` and fill in your players
   (`{jersey_number: name}`).
2. Run with `--identify --roster roster.json`.

It works best alongside `--pose rtmlib` (pose keypoints crop the number region
precisely), but falls back to a bbox-ratio crop without pose.

**How it stays robust:** the number is only legible in a fraction of frames,
so CLARA votes across the whole track instead of trusting any single frame —
each OCR read is a confidence-weighted vote, the dominant number wins. Only
numbers present in the roster count, so OCR misreads are discarded and rival
players (not in the roster) stay anonymous on purpose. Output per track:
`identity = {number, name, confidence, votes, weight}`.

## Ball detector comparison

| Detector | Setup | Recall (own gym) | Recall (new gym) | Training |
|---|---|---|---|---|
| `yolo` (base) | None | ~0% | ~0% | None |
| `yolo --ball-model custom.pt` | Label + train | 14-30% | 5-15% | 2-4 hours |
| **`vballnet`** | Download .onnx | **34%** | **68%** | **None** |

VballNet uses 9-frame temporal context — ignores static lamps/banners, catches blurred/occluded balls.

Get the pretrained model: https://github.com/asigatchov/fast-volleyball-tracking-inference (models/ folder)

## Pose estimation (rtmlib)

When `--pose rtmlib`, CLARA extracts 17 COCO keypoints per detected player and computes:

- **Torso lean angle** — forward lean indicator (useful for spike approach analysis)
- **Stance width** — distance between ankles (defensive readiness)
- **Knee flexion** — depth of defensive stance

Output goes into `scouting_data.json` under each `tracks[].pose_stats`.

## Quality score

CLARA evaluates its own output across 5 dimensions:

| Component | Max | What it measures |
|---|---|---|
| Tracks identified | 30 | Tracks found vs. expected (12 full / 6 half) |
| Zone coverage | 25 | Zones with meaningful activity |
| Ball detection rate | 20 | % of frames with ball (50%+ = max) |
| Filter acceptance | 15 | Detections kept vs. rejected as foreground |
| Track stability | 10 | Average track duration |

**Interpretation:**
- **80-100:** scouting confiable, datos accionables
- **60-79:** scouting útil con caveats
- **40-59:** insight de zonas, no de jugadoras individuales
- **<40:** re-grabar o re-calibrar

## Touch & rally detection (post-processor)

`src/touch_machine.py` turns a `scouting_data.json` into per-touch and per-rally
data **without re-reading the video**. It finds ball contacts as spikes in the
ball's acceleration and attributes each to the nearest player:

```bash
pip install scipy
python src/touch_machine.py results/scouting_data.json   # -> *_enriched.json
```

It adds `touches[]` (frame, position, attributed `track_id`) and `rallies[]`
(grouped by ball silence). Runs on CLARA v0.8.1+ output as-is — it needs
`ball_track` + `track_samples`, both emitted by default. See
[docs/integrations.md](docs/integrations.md#touch_machine-touch--rally-detection).

## Project structure

```
clara/
├── src/
│   ├── clara.py              — main pipeline (v0.8.0)
│   ├── clara_report.py       — HTML report generator
│   ├── touch_machine.py      — touch & rally detection (post-processor)
│   ├── ball_vballnet.py      — VballNet adapter
│   ├── pose_rtmlib.py        — RTMPose wrapper + biomechanics
│   ├── jersey_id.py          — jersey number identification
│   ├── setup_calibration.py  — interactive calibration
│   └── extract_frames.py     — frame extraction for training
├── roster.example.json       — roster template for --identify
├── calibration/              — example calibration JSONs
├── docs/
│   ├── architecture.md
│   ├── training.md           — custom YOLO training guide
│   └── vballnet.md           — VballNet integration notes
└── clara_train_colab.ipynb   — Colab notebook for YOLO training
```

## Versioning

- **v0.1**: HTML prototype with TensorFlow.js
- **v0.2-0.3**: Python pipeline with YOLOv8 + ByteTrack + homography
- **v0.4**: HTML reports, zone-marked topdowns
- **v0.5**: Half-court mode, foreground filter, quality score
- **v0.5.1**: Bug fixes (score rebalancing, ball filter, efficiency)
- **v0.6**: VballNet integration, rtmlib pose estimation
- **v0.6.1-0.6.2**: Foreground filter fixes (horizon band, bottom edge)
- **v0.7.0**: Player identification by jersey number (`--identify`)
- **v0.8.0**: YOLO11m detector, honest quality metrics, coach-ready per-player scouting data, multi-gym validation

## License

Reserved — internal Las Chispas project. CLARA is the visual perception tentacle of the LUCIA analytics system.

> *"Until the last star falls."*
