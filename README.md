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

- **Player tracking** via YOLOv8 + ByteTrack
- **Ball detection** in 3 modes:
  - YOLOv8 base (no setup, low recall)
  - YOLOv8 custom (high recall in trained gym, requires labeling)
  - **VballNet** (motion-based, ~70% recall in any gym, **no training needed**)
- **Pose estimation** via RTMPose (optional): 17 keypoints per player
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

# A — Solo YOLOv8 base (rápido, low recall en balón)
python src/clara.py video.mp4 --calibration cal.json --out results/

# B — Con modelo YOLOv8 custom entrenado
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

# 4. Generate HTML report
python src/clara_report.py results/scouting_data.json --topdown results/topdown.png
```

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

## Project structure

```
clara/
├── src/
│   ├── clara.py              — main pipeline (v0.6)
│   ├── clara_report.py       — HTML report generator
│   ├── ball_vballnet.py      — VballNet adapter
│   ├── pose_rtmlib.py        — RTMPose wrapper + biomechanics
│   ├── setup_calibration.py  — interactive calibration
│   └── extract_frames.py     — frame extraction for training
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

## License

Reserved — internal Las Chispas project. CLARA is the visual perception tentacle of the LUCIA analytics system.

> *"Until the last star falls."*
