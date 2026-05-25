# Training a custom ball detector

YOLO11 base (COCO weights) does not reliably detect volleyballs — it tends to either miss them (low recall) or hallucinate on round objects like ceiling lights.

For accurate ball tracking, train a custom YOLO model on your specific gym.

## Workflow

### 1. Extract frames

```bash
python src/extract_frames.py game.mp4 --every 1.5 --out frames/
```

Extract every 1.5 seconds. For a 5-minute game, you get ~200 frames.

### 2. Label in Roboflow

1. Create a free [Roboflow](https://roboflow.com) account
2. New Project → Object Detection
3. Upload the `frames/` folder
4. Label each frame:
   - Tool: **Bounding Box** (not Smart Select / SAM3)
   - Class: `ball` (singular, lowercase)
   - Skip frames without a visible ball — they serve as negative examples
5. Generate a Dataset Version when done

### 3. Export the labeled dataset

In Roboflow → Versions → Download Dataset → **YOLOv8 format** → "Show download code"

Copy the snippet (4 lines starting with `from roboflow import Roboflow`).

### 4. Train in Google Colab (free GPU)

Use `clara_train_colab.ipynb` (in this repo).

Steps:
1. Upload notebook to [Google Colab](https://colab.research.google.com)
2. **Runtime → Change runtime type → T4 GPU**
3. Paste the Roboflow snippet in the dataset cell
4. Run all cells
5. Wait 20-40 minutes
6. The notebook auto-downloads `clara_balon_v1.pt` when done

### 5. Use the trained model

```bash
python src/clara.py video.mp4 --calibration cal.json --ball-model clara_balon_v1.pt --ball-conf 0.3
```

## How many frames do I need?

| Frames labeled | Expected mAP@50 | Use case |
|---|---|---|
| 100-150 | 40-55% | First experiment, one gym |
| 300-500 | 55-70% | Production for one gym |
| 800+ frames + multiple gyms | 70-85% | Robust across venues |

## Domain shift warning

A model trained on one gym may underperform in others (different lighting, walls, ball color, camera angles). The recommended approach:

1. Train v1 with your first gym (~150 frames)
2. Use v1 to auto-label frames from new gyms
3. Manually correct errors in Roboflow
4. Train v2 with the combined dataset (~400 frames, multiple gyms)
5. Repeat

Each iteration generalizes better.

## Alternative: VballNet

For ball detection specifically, [VballNet](https://github.com/asigatchov/fast-volleyball-tracking-inference) uses motion-based detection (TrackNetV4) and works without training across different gyms. See `docs/vballnet.md`.
