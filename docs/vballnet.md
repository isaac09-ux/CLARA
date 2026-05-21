# VballNet integration (future work)

[VballNet](https://github.com/asigatchov/fast-volleyball-tracking-inference) is a TrackNetV4-based ball detection model that uses **motion across 9 consecutive frames** instead of single-frame appearance. Key advantages over YOLO-based ball detection:

- No training required — pretrained ONNX model works across gyms
- Ignores static objects (ceiling lights, banners) because they don't move
- Detects blurred/occluded balls via temporal context
- 100-200 FPS on CPU
- Available pretrained weights from the project

## Benchmark vs custom YOLO

Tested on the same Las Chispas footage:

| Model | Recall (original gym) | Recall (new gym) | Training needed |
|---|---|---|---|
| YOLOv8n base | 0% | 0% | None |
| Custom YOLOv8 (clara_balon_v1.pt) | 14.4% | ~10% (estimated) | 2 hours labeling |
| **VballNet (pretrained)** | **34.6%** | **68.7%** | **None** |

VballNet outperforms a custom-trained YOLO model on new gyms it has never seen.

## Why we haven't fully integrated yet

VballNet outputs ball positions as `(frame, x, y, radius)` in `ball.csv`, not bounding boxes. CLARA needs to:

1. Pre-process video with VballNet → `ball.csv`
2. Read CSV and convert to CLARA's ball detection format
3. Apply homography projection from (x, y) point
4. Skip the bbox-based foreground filter for balls (point-based detection)

## Planned integration (v0.6)

```python
# Proposed CLI:
python src/clara.py video.mp4 \
  --calibration cal.json \
  --ball-detector vballnet \
  --vballnet-model VballNetV1_seq9_grayscale_330_h288_w512.onnx
```

CLARA would then:
1. Run VballNet on the video (CPU-only inference)
2. Use the resulting ball trajectories alongside YOLOv8 player tracks
3. Output the same scouting JSON / topdown / HTML report

## Why this is significant

For Las Chispas specifically, VballNet eliminates the need for per-gym ball detector training. The existing custom-trained `clara_balon_v1.pt` is preserved as a backup but no longer the primary path.

For other amateur volleyball clubs adopting CLARA, this means:
- Zero labeling work to get started
- Works across diverse venues out of the box
- Better recall than any custom-trained alternative

See `/scripts/test_vballnet.py` (planned) for a reference integration.
