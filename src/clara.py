"""
CLARA v0.8 — Multimodal scouting + auto-calibracion
Tentáculo de visión por computadora de LUCIA · Las Chispas.

Cambios v0.5.1 → v0.6:
  - INTEGRACIÓN: VballNet (TrackNetV4) como detector de balón opcional
    Motion-based, ~70% recall sin entrenamiento en gyms nuevos
  - INTEGRACIÓN: rtmlib (RTMPose) para keypoints de jugadoras
    17 puntos COCO, análisis biomecánico básico
  - Mantiene compatibilidad con clara_balon_v1.pt (YOLO custom)

Selectores de detector de balón:
  --ball-detector yolo       (default — usa YOLO11 base, clase 32, o --ball-model)
  --ball-detector vballnet   (usa modelo VballNet ONNX, requiere --vballnet-model)

Selectores de pose:
  --pose none      (default — solo bounding boxes)
  --pose rtmlib    (extrae keypoints por jugadora detectada)
"""
import cv2
import numpy as np
import json
import argparse
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO


# ============================================================
#  CONSTANTES
# ============================================================
PAL_BONE = (208, 224, 232)
PAL_OXBLOOD = (40, 40, 165)
PAL_OK = (122, 156, 90)
PAL_DARK = (84, 79, 68)
PAL_WARN = (60, 156, 220)
PAL_BALL = (60, 220, 230)
PAL_REJECT = (60, 60, 165)


# ============================================================
#  GEOMETRÍA
# ============================================================
def project(H, x, y):
    pt = np.array([[[x, y]]], dtype=np.float32)
    return cv2.perspectiveTransform(pt, np.array(H))[0][0]


def zone_for_court_pos(x_m, y_m, court_w, court_h, half_court=False, margin=0.0):
    if (x_m < -margin or x_m > court_w + margin or
        y_m < -margin or y_m > court_h + margin):
        return None
    x_m = max(0, min(court_w, x_m))
    y_m = max(0, min(court_h, y_m))

    if half_court:
        front = y_m < court_h / 3
        col = "L" if x_m < court_w / 3 else ("M" if x_m < 2 * court_w / 3 else "R")
        zmap = {("L", True): 4, ("M", True): 3, ("R", True): 2,
                ("L", False): 5, ("M", False): 6, ("R", False): 1}
        return f"A{zmap[(col, front)]}"

    if y_m <= court_h / 2:
        side, local_y = "A", y_m
    else:
        side, local_y = "B", court_h - y_m
    front = local_y > court_h / 4
    col = "L" if x_m < court_w / 3 else ("M" if x_m < 2 * court_w / 3 else "R")
    if side == "A":
        zmap = {("L", True): 4, ("M", True): 3, ("R", True): 2,
                ("L", False): 5, ("M", False): 6, ("R", False): 1}
    else:
        zmap = {("R", True): 4, ("M", True): 3, ("L", True): 2,
                ("R", False): 5, ("M", False): 6, ("L", False): 1}
    return f"{side}{zmap[(col, front)]}"


def is_in_court(cx, cy, cw, ch, margin=0.5):
    return (cx is not None and cy is not None and
            -margin <= cx <= cw + margin and -margin <= cy <= ch + margin)


def filter_ball_tracks(detections, stride, max_gap=None,
                       max_dist_m=6.0, min_track_len=2):
    """Agrupa detecciones de balón en tracks por cercanía temporal y espacial,
    descarta tracks cortos (FPs aislados: luces, banderines, balones del público).

    Una detección sólo sobrevive si tiene al menos otra detección dentro de
    max_gap frames Y max_dist_m metros en coords de cancha. Sin entrenar nada,
    elimina los falsos positivos puntuales que un buen detector frame-por-frame
    no puede distinguir de un balón real.
    """
    if len(detections) < min_track_len:
        return []
    if max_gap is None:
        # Un par de strides cubre saltos breves sin pegar tracks no relacionados.
        max_gap = max(stride * 3, 6)

    dets = sorted(detections, key=lambda d: d["frame"])
    tracks = []
    current = [dets[0]]
    for d in dets[1:]:
        prev = current[-1]
        gap = d["frame"] - prev["frame"]
        dx = d["court_x"] - prev["court_x"]
        dy = d["court_y"] - prev["court_y"]
        if gap <= max_gap and (dx * dx + dy * dy) ** 0.5 <= max_dist_m:
            current.append(d)
        else:
            tracks.append(current)
            current = [d]
    tracks.append(current)
    return [d for t in tracks if len(t) >= min_track_len for d in t]


def classify_detection(bbox, frame_h, frame_w, court_horizon_y=None,
                       max_height_ratio=0.55, max_width_ratio=0.40,
                       is_ball=False, edge_reject_height_ratio=0.35):
    """Filtro pre-homografía: descarta cajas obviamente inválidas por tamaño
    o por tocar el borde inferior siendo además grandes (público cercano).

    court_horizon_y queda aceptado por compatibilidad con calibraciones
    previas pero ya no se usa: la banda fija en píxeles (±5%/+10% del alto)
    asumía cámara casi frontal, donde la cancha ocupa una franja delgada.
    En ángulos elevados u oblicuos la cancha cubre cientos de píxeles
    verticales y la banda rechaza jugadoras reales (un día rechazó 1911
    detecciones, dejando 1 sola jugadora). El filtro correcto es
    is_in_court() después de proyectar a coords de cancha, que ya se aplica
    al armar `filtered` y `ball_clean`.

    fg_at_edge (corregido v0.6.2): antes rechazaba CUALQUIER caja cuyo
    borde inferior tocara el frame. En video de baja resolución (848x478)
    las jugadoras llenan buena parte del alto y muchas tocan el borde en
    algún frame — el filtro mataba miles de detecciones válidas (3699 en
    un caso real, dejando 0 tracks). Ahora solo rechaza si la caja toca el
    borde Y ADEMÁS es grande (>35% del alto), que es el patrón de un
    espectador/entrenador cortado en primer plano. Una jugadora lejana
    cuyos pies se cortan un poco en el borde es una detección válida y se
    conserva; si quedara fuera de cancha, is_in_court() la filtra después.
    """
    x1, y1, x2, y2 = bbox
    bbox_h = y2 - y1
    bbox_w = x2 - x1
    if not is_ball:
        if bbox_h / frame_h > max_height_ratio:
            return "fg_too_large", f"altura {bbox_h/frame_h:.0%}"
        if bbox_w / frame_w > max_width_ratio:
            return "fg_too_large", f"ancho {bbox_w/frame_w:.0%}"
    if y2 >= frame_h - 5:
        # Solo rechazar si además es grande (primer plano cortado).
        # Cajas normales/chicas en el borde = jugadora lejana válida.
        if not is_ball and bbox_h / frame_h > edge_reject_height_ratio:
            return "fg_at_edge", f"borde + grande ({bbox_h/frame_h:.0%})"
    return "ok", None


# ============================================================
#  DETECCIÓN DE BALÓN VIA VBALLNET
# ============================================================
def detect_balls_vballnet(video_path, model_path, H, ppm,
                          court_w, court_h,
                          frame_h, frame_w, court_horizon_y=None,
                          max_h_ratio=0.55, max_w_ratio=0.40,
                          rejected_counts=None,
                          threshold=0.5, stride=1, verbose=True):
    """VballNet adapter. Returns CLARA-compatible ball detection list.

    Applies the same foreground filter (bottom-edge guard) used for YOLO
    ball detections so detecciones de tribuna/banca se descartan.

    stride > 1 alimenta el buffer de 9 frames sólo cada N frames del video.
    Reduce inferencias y RAM ~stride×; el balón salta más entre frames de
    la secuencia, así que el recall baja proporcionalmente.
    """
    from ball_vballnet import detect_balls
    raw = detect_balls(video_path, model_path,
                       threshold=threshold, stride=stride, verbose=verbose)
    out = []
    for d in raw:
        # bbox sintético desde (x, y, radius) para reusar classify_detection.
        # Sólo importa el borde inferior (y2) para el filtro de balón —
        # is_ball=True salta los checks de altura/ancho relativos al frame.
        r = max(d.get("radius", 5.0), 5.0)
        bbox = [d["x"] - r, d["y"] - r, d["x"] + r, d["y"] + r]
        status, _ = classify_detection(
            bbox, frame_h, frame_w, court_horizon_y,
            max_h_ratio, max_w_ratio, is_ball=True,
        )
        if status != "ok":
            if rejected_counts is not None:
                rejected_counts[f"ball_{status}"] += 1
            continue
        cx_m, cy_m = project(H, d["x"], d["y"])
        out.append({
            "frame": d["frame"],
            "court_x": float(cx_m) / ppm,
            "court_y": float(cy_m) / ppm,
            "conf": d["confidence"],
        })
    return out


# ============================================================
#  TRACK STITCHING
# ============================================================
def stitch_tracks(raw_tracks, fps, stride, max_gap_s=1.5, max_jump_m=2.5):
    """
    Cose tracks fragmentados. ByteTrack parte una jugadora en varios IDs
    cuando se ocluye; esta funcion une los pedazos.

    Dos tracks A y B se unen si:
      - B empieza DESPUES de que A termina (sin solaparse)
      - el hueco temporal es <= max_gap_s segundos
      - la distancia entre el ultimo punto de A y el primero de B es
        <= max_jump_m metros (la jugadora no se teletransporta)

    Args:
        raw_tracks: dict {tid: [samples]} — cada sample tiene 'frame',
                    'court_x', 'court_y'
        fps: frames por segundo del video
        stride: cada cuantos frames se muestreo
        max_gap_s: hueco temporal maximo para considerar union
        max_jump_m: salto espacial maximo permitido

    Returns:
        dict {tid: [samples]} con los tracks cosidos.
    """
    if len(raw_tracks) < 2:
        return raw_tracks

    # Ordenar samples de cada track por frame
    tracks = {}
    for tid, samples in raw_tracks.items():
        tracks[tid] = sorted(samples, key=lambda s: s["frame"])

    max_gap_frames = max_gap_s * fps

    # Repetir hasta que no haya mas uniones (un track puede coser
    # varios pedazos en cadena)
    merged_any = True
    while merged_any:
        merged_any = False
        tids = list(tracks.keys())

        for i in range(len(tids)):
            tid_a = tids[i]
            if tid_a not in tracks:
                continue
            a = tracks[tid_a]
            a_end_frame = a[-1]["frame"]
            a_end_pos = (a[-1]["court_x"], a[-1]["court_y"])

            best_match = None
            best_gap = None

            for j in range(len(tids)):
                tid_b = tids[j]
                if tid_b == tid_a or tid_b not in tracks:
                    continue
                b = tracks[tid_b]
                b_start_frame = b[0]["frame"]
                # B debe empezar despues de que A termina
                gap = b_start_frame - a_end_frame
                if gap <= 0 or gap > max_gap_frames:
                    continue
                # distancia espacial entre fin de A e inicio de B
                b_start_pos = (b[0]["court_x"], b[0]["court_y"])
                dist = np.hypot(a_end_pos[0] - b_start_pos[0],
                                a_end_pos[1] - b_start_pos[1])
                if dist > max_jump_m:
                    continue
                # de los candidatos, preferir el de menor hueco temporal
                if best_gap is None or gap < best_gap:
                    best_gap = gap
                    best_match = tid_b

            if best_match is not None:
                # coser B dentro de A
                tracks[tid_a] = a + tracks[best_match]
                del tracks[best_match]
                merged_any = True

    return tracks


# ============================================================
#  PIPELINE PRINCIPAL
# ============================================================
def run(video_path, calibration_path, output_dir="out", stride=5,
        person_conf=0.4, ball_conf=0.10,
        ball_detector="yolo", ball_model_path=None, vballnet_model=None,
        vballnet_stride=1,
        pose_mode="none",
        court_model=None,
        save_diagnostic=True):

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Auto-calibracion: si se pasa court_model, intentar calibrar solo.
    # Si funciona, se usa esa calibracion. Si falla, cae al cal.json manual
    # (si existe) o aborta pidiendo MIRA.
    if court_model is not None:
        from court_keypoints import auto_calibrate
        print("[•] Intentando auto-calibracion con modelo de cancha...")
        auto_cal = auto_calibrate(video_path, court_model)
        if auto_cal is not None:
            cal = auto_cal
            (out / "cal_auto.json").write_text(json.dumps(cal, indent=2))
            print(f"[•] Auto-calibracion OK — guardada en {out/'cal_auto.json'}")
        elif calibration_path is not None:
            print("[•] Auto-calibracion fallo — usando cal.json manual.")
            cal = json.loads(Path(calibration_path).read_text())
        else:
            raise RuntimeError(
                "Auto-calibracion fallo y no se dio --calibration manual. "
                "Calibra con MIRA y pasa el cal.json."
            )
    else:
        if calibration_path is None:
            raise RuntimeError(
                "Falta calibracion: pasa --calibration cal.json o "
                "--court-model modelo.pt"
            )
        cal = json.loads(Path(calibration_path).read_text())
    H = np.array(cal["homography_matrix"])
    court_w, court_h = cal["court_size_m"]
    ppm = cal["pixels_per_meter"]
    half_court = cal.get("half_court", False)
    court_horizon_y = cal.get("court_horizon_y", None)
    max_h_ratio = cal.get("max_person_height_ratio", 0.55)
    max_w_ratio = cal.get("max_person_width_ratio", 0.40)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"No pude abrir video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    Hf = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print(f"\n┌─ CLARA v0.8 ─────────────────────────────")
    print(f"│ Video: {Path(video_path).name}")
    print(f"│ {W}x{Hf} @ {fps:.1f}fps | {total_frames/fps/60:.2f} min")
    print(f"│ Cancha: {court_w}x{court_h}m {'[HALF]' if half_court else '[FULL]'}")
    print(f"│ Balón: {ball_detector}")
    print(f"│ Pose:  {pose_mode}")
    print(f"│ Stride: {stride}")
    print(f"└──────────────────────────────────────────\n")

    # Detector de personas: yolo11m (medium). CLARA es offline, no hay
    # presion de tiempo real, asi que no se usa el modelo nano: detecta
    # peor a jugadoras chicas/lejanas/borrosas (footage de celular) y cada
    # deteccion perdida es un hueco por donde se fragmenta un track.
    # 'm' es el punto medio para CPU; en GPU se puede subir a 'yolo11x'.
    person_model = YOLO("yolo11m.pt")

    pose_estimator = None
    if pose_mode == "rtmlib":
        try:
            from pose_rtmlib import PoseEstimator
            pose_estimator = PoseEstimator(mode="balanced", device="cpu")
        except ImportError:
            print("⚠ rtmlib no disponible. pip install rtmlib")
            pose_mode = "none"

    raw_tracks = defaultdict(list)
    rejected_counts = defaultdict(int)
    ball_detections = []
    samples_processed = 0

    classes_to_track = [0] if (ball_detector != "yolo" or ball_model_path) else [0, 32]

    # Usar la config de ByteTrack tuneada de CLARA (track_buffer alto para
    # aguantar oclusiones). Si el archivo no esta junto al script, cae al
    # bytetrack.yaml default de Ultralytics.
    tracker_cfg = Path(__file__).parent / "bytetrack_clara.yaml"
    tracker_arg = str(tracker_cfg) if tracker_cfg.exists() else "bytetrack.yaml"

    # IMPORTANTE: el tracker procesa TODOS los frames del video.
    # ByteTrack predice la posicion de cada jugadora con un filtro de
    # Kalman y asocia por IoU asumiendo frames consecutivos. Si se le
    # alimentan frames salteados (el viejo vid_stride=stride), una
    # jugadora se desplaza ~stride veces mas en pixeles entre frame y
    # frame, el IoU cae por debajo de match_thresh y ByteTrack crea un
    # ID nuevo -> tracks fragmentados. El submuestreo para analitica se
    # hace DESPUES, en el loop, no aqui.
    results = person_model.track(
        source=video_path,
        persist=True,
        tracker=tracker_arg,
        classes=classes_to_track,
        conf=min(person_conf, ball_conf),
        verbose=False,
        stream=True,
    )

    for frame_idx, r in enumerate(results):
        # ByteTrack ya proceso este frame y actualizo sus IDs internamente
        # (basta con consumir `r` del generador). Solo recolectamos un
        # sample para analitica cada `stride` frames: el tracking necesita
        # continuidad total, la analitica de zonas/heatmap no.
        if frame_idx % stride != 0:
            continue
        actual_frame = frame_idx

        if r.boxes is not None:
            cls = r.boxes.cls.int().cpu().tolist()
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().tolist()
            ids = (r.boxes.id.int().cpu().tolist()
                   if r.boxes.id is not None else [None] * len(cls))

            valid_persons = []
            for c, box, conf, tid in zip(cls, xyxy, confs, ids):
                x1, y1, x2, y2 = box
                cx = (x1 + x2) / 2.0
                ground_y = y2 if c == 0 else (y1 + y2) / 2.0
                cx_m, cy_m = project(H, cx, ground_y)
                court_x = float(cx_m) / ppm
                court_y = float(cy_m) / ppm

                if c == 0:
                    if conf < person_conf or tid is None:
                        continue
                    status, _ = classify_detection(
                        box, Hf, W, court_horizon_y,
                        max_h_ratio, max_w_ratio, is_ball=False,
                    )
                    if status != "ok":
                        rejected_counts[status] += 1
                        continue
                    sample = {
                        "frame": actual_frame,
                        "court_x": court_x, "court_y": court_y,
                        "conf": conf,
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    }
                    raw_tracks[tid].append(sample)
                    if pose_estimator:
                        valid_persons.append((tid, sample,
                                              [float(x1), float(y1),
                                               float(x2), float(y2)]))
                elif c == 32 and ball_detector == "yolo" and conf >= ball_conf:
                    status, _ = classify_detection(
                        box, Hf, W, court_horizon_y,
                        max_h_ratio, max_w_ratio, is_ball=True,
                    )
                    if status != "ok":
                        rejected_counts[f"ball_{status}"] += 1
                        continue
                    ball_detections.append({
                        "frame": actual_frame,
                        "court_x": court_x, "court_y": court_y,
                        "conf": conf,
                    })

            if pose_estimator and valid_persons:
                # ultralytics ya tiene el frame decodificado; usarlo directo
                # es O(1) vs. seek aleatorio con cap.set(POS_FRAMES) que
                # invalida el cache del decodificador en cada muestra.
                frame_img = getattr(r, "orig_img", None)
                if frame_img is not None:
                    bboxes = [vp[2] for vp in valid_persons]
                    try:
                        pose_results = pose_estimator.estimate_batch(
                            frame_img, bboxes)
                        for (tid, sample, _), pose_data in zip(
                                valid_persons, pose_results):
                            if pose_data["avg_score"] >= 0.3:
                                sample["pose"] = {
                                    "kp": pose_data["keypoints"],
                                    "sc": pose_data["scores"],
                                    "avg": pose_data["avg_score"],
                                }
                    except Exception as e:
                        if samples_processed % 50 == 0:
                            print(f"⚠ Pose error frame {actual_frame}: {e}")

        samples_processed += 1
        if samples_processed % 200 == 0:
            pct = actual_frame / total_frames * 100
            print(f"  ⏳ {samples_processed} muestras ({pct:.0f}%)")

    # ─── Detección de balón ───
    if ball_detector == "yolo" and ball_model_path and Path(ball_model_path).exists():
        print(f"\n[•] Detectando balón con modelo custom YOLO11...")
        ball_model = YOLO(ball_model_path)
        cap = cv2.VideoCapture(video_path)
        for sample_idx in range(samples_processed):
            fi = sample_idx * stride
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                break
            ball_res = ball_model(frame, verbose=False, conf=ball_conf)[0]
            if ball_res.boxes is not None and len(ball_res.boxes) > 0:
                for box, conf in zip(ball_res.boxes.xyxy.cpu().numpy(),
                                     ball_res.boxes.conf.cpu().tolist()):
                    x1, y1, x2, y2 = box
                    status, _ = classify_detection(
                        box, Hf, W, court_horizon_y,
                        max_h_ratio, max_w_ratio, is_ball=True,
                    )
                    if status != "ok":
                        rejected_counts[f"ball_{status}"] += 1
                        continue
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    cx_m, cy_m = project(H, cx, cy)
                    ball_detections.append({
                        "frame": fi,
                        "court_x": float(cx_m) / ppm,
                        "court_y": float(cy_m) / ppm,
                        "conf": conf,
                    })
        cap.release()

    elif ball_detector == "vballnet":
        if vballnet_model is None or not Path(vballnet_model).exists():
            raise FileNotFoundError(
                "Para --ball-detector vballnet necesitas --vballnet-model <ruta .onnx>"
            )
        print(f"\n[•] Detectando balón con VballNet (stride={vballnet_stride})...")
        ball_detections.extend(
            detect_balls_vballnet(video_path, vballnet_model, H, ppm,
                                   court_w, court_h,
                                   frame_h=Hf, frame_w=W,
                                   court_horizon_y=court_horizon_y,
                                   max_h_ratio=max_h_ratio,
                                   max_w_ratio=max_w_ratio,
                                   rejected_counts=rejected_counts,
                                   stride=vballnet_stride,
                                   verbose=True)
        )

    # ─── Track stitching: cose tracks fragmentados ───
    # ByteTrack parte una jugadora en varios IDs cuando hay oclusion.
    # Si un track termina donde/cuando otro empieza, y la posicion es
    # coherente, son la misma jugadora -> se unen.
    n_tracks_before_stitch = len(raw_tracks)
    # Volcar los tracks crudos (pre-cosido) para poder iterar los parametros
    # de stitch offline sin re-trackear, que es la parte cara del pipeline.
    (out / "raw_tracks.json").write_text(json.dumps(
        {str(tid): s for tid, s in raw_tracks.items()}, indent=2))
    # Oclusiones de voleibol (red, bloqueos) pasan facil de 1.5 s y la jugadora
    # se mueve mas de 2.5 m durante una larga. Barrido sobre raw_tracks.json de
    # test.mp4: 3s/4m deja 14 tracks (6 jugadoras reales), 5s/6m baja a 8 SIN
    # subir el spread espacial maximo (4.08 m, identico) -> fusiona fragmentos
    # sin unir jugadoras distintas; 8s/8m ya colapsa a 4 (over-merge). 5/6 es
    # el optimo medido.
    raw_tracks = stitch_tracks(raw_tracks, fps, stride,
                               max_gap_s=5.0, max_jump_m=6.0)
    n_stitched = n_tracks_before_stitch - len(raw_tracks)

    # ─── Filtrado de tracks ───
    min_samples = 15
    filtered = {}
    zone_visits = defaultdict(int)
    zone_first = defaultdict(int)
    zone_second = defaultdict(int)
    half_frame = total_frames // 2

    for tid, samples in raw_tracks.items():
        if len(samples) < min_samples:
            continue
        in_court = [s for s in samples
                    if is_in_court(s["court_x"], s["court_y"], court_w, court_h)]
        if not in_court:
            continue
        if (len(in_court) / len(samples) < 0.5 or
            len(in_court) < min_samples // 2):
            continue
        filtered[tid] = in_court
        for s in in_court:
            z = zone_for_court_pos(s["court_x"], s["court_y"],
                                    court_w, court_h, half_court, margin=0.5)
            if z:
                zone_visits[z] += 1
                if s["frame"] < half_frame:
                    zone_first[z] += 1
                else:
                    zone_second[z] += 1

    ball_clean = [b for b in ball_detections
                  if is_in_court(b["court_x"], b["court_y"],
                                 court_w, court_h, margin=2)]
    ball_before_tracking = len(ball_clean)
    ball_clean = filter_ball_tracks(ball_clean, stride=stride)
    ball_isolated_dropped = ball_before_tracking - len(ball_clean)
    if ball_isolated_dropped > 0:
        rejected_counts["ball_isolated"] = ball_isolated_dropped

    # Frames únicos con al menos una detección de balón — métrica acotada a
    # [0, samples_processed]. Distinta de len(ball_clean) cuando un detector
    # emite varias cajas en el mismo frame.
    ball_frames_oncourt = len({b["frame"] for b in ball_clean})

    # ─── Score de calidad ───
    expected_tracks = 6 if half_court else 12
    score, score_breakdown = compute_quality_score(
        filtered, zone_visits, ball_frames_oncourt, rejected_counts,
        samples_processed, expected_tracks, half_court
    )

    # ─── Pose analytics ───
    pose_stats = {}
    if pose_estimator:
        try:
            from pose_rtmlib import torso_lean_angle, stance_width, knee_flexion
            for tid, samples in filtered.items():
                lean_vals, stance_vals, knee_vals = [], [], []
                for s in samples:
                    p = s.get("pose")
                    if not p:
                        continue
                    lean = torso_lean_angle(p["kp"], p["sc"])
                    if lean is not None:
                        lean_vals.append(lean)
                    sw = stance_width(p["kp"], p["sc"])
                    if sw is not None:
                        stance_vals.append(sw)
                    # Promediar las dos rodillas — con la jugadora de perfil
                    # una pierna suele estar ocluida; si sólo medimos la
                    # izquierda perdemos la mitad de las muestras útiles.
                    kf_sides = [
                        v for v in (
                            knee_flexion(p["kp"], p["sc"], side="left"),
                            knee_flexion(p["kp"], p["sc"], side="right"),
                        ) if v is not None
                    ]
                    if kf_sides:
                        knee_vals.append(float(np.mean(kf_sides)))
                if lean_vals or stance_vals or knee_vals:
                    pose_stats[tid] = {
                        "samples_with_pose": len(lean_vals),
                        "torso_lean_deg_avg": (round(float(np.mean(lean_vals)), 1)
                                                if lean_vals else None),
                        "torso_lean_deg_max": (round(float(np.max(lean_vals)), 1)
                                                if lean_vals else None),
                        "stance_width_px_avg": (round(float(np.mean(stance_vals)), 1)
                                                 if stance_vals else None),
                        "knee_flexion_deg_avg": (round(float(np.mean(knee_vals)), 1)
                                                  if knee_vals else None),
                    }
        except ImportError:
            pass

    metrics = {
        "clara_version": "0.8",
        "video": Path(video_path).name,
        "duration_s": round(total_frames / fps, 1),
        "duration_min": round(total_frames / fps / 60, 2),
        "stride": stride,
        "samples_processed": samples_processed,
        "court_size_m": [court_w, court_h],
        "half_court": half_court,
        "ball_detector": ball_detector,
        "pose_mode": pose_mode,
        "raw_tracks": n_tracks_before_stitch,
        "tracks_after_stitch": len(raw_tracks),
        "tracks_stitched": n_stitched,
        "filtered_tracks": len(filtered),
        "rejected_detections": dict(rejected_counts),
        "ball_detections_oncourt": len(ball_clean),
        "ball_frames_oncourt": ball_frames_oncourt,
        "ball_detection_rate": round(ball_frames_oncourt / max(samples_processed, 1), 3),
        "quality_score": score,
        "quality_breakdown": score_breakdown,
        "zone_visits_total": dict(zone_visits),
        "zone_visits_first_half": dict(zone_first),
        "zone_visits_second_half": dict(zone_second),
        "tracks": [],
        "pose_stats": pose_stats,
    }

    for tid, samples in filtered.items():
        xs = np.array([s["court_x"] for s in samples])
        ys = np.array([s["court_y"] for s in samples])
        dist_m = float(np.sum(np.hypot(np.diff(xs), np.diff(ys))))
        t_span = (samples[-1]["frame"] - samples[0]["frame"]) / fps
        speed = round(dist_m / t_span, 2) if t_span > 0 else None
        zones = defaultdict(int)
        for s in samples:
            z = zone_for_court_pos(s["court_x"], s["court_y"],
                                    court_w, court_h, half_court, margin=0.5)
            if z:
                zones[z] += 1
        dom = max(zones.items(), key=lambda x: x[1])[0] if zones else None
        # Perfil de zonas (% del tiempo en cada zona): el coach lee la
        # tendencia posicional de la jugadora, no solo su zona dominante.
        ztot = sum(zones.values())
        zone_profile = ({z: round(c / ztot * 100) for z, c in
                         sorted(zones.items(), key=lambda x: -x[1])}
                        if ztot else {})
        side = "A" if half_court or np.median(ys) <= court_h / 2 else "B"
        # Fiabilidad del track: distingue una jugadora bien seguida de un
        # fragmento corto, para que el coach sepa de cuales datos fiarse.
        reliability = ("alta" if len(samples) >= 50 else
                       "media" if len(samples) >= 25 else "baja")
        metrics["tracks"].append({
            "id": tid, "samples": len(samples),
            "seconds_tracked": round(t_span, 1),
            "reliability": reliability,
            "distance_m": round(dist_m, 1),
            "avg_speed_m_per_s": speed,
            "side": side, "dominant_zone": dom,
            "zone_profile_pct": zone_profile,
            "avg_court_pos_m": [round(float(xs.mean()), 2),
                                round(float(ys.mean()), 2)],
            "pose_stats": pose_stats.get(tid),
        })
    metrics["tracks"].sort(key=lambda t: -t["samples"])

    # ─── Topdowns ───
    save_topdown(filtered, ball_clean, court_w, court_h, ppm,
                 out / "topdown.png", title="Total",
                 metrics=metrics, half_court=half_court)
    first = {tid: [s for s in ss if s["frame"] < half_frame]
             for tid, ss in filtered.items()}
    first = {k: v for k, v in first.items() if len(v) >= 5}
    save_topdown(first, [b for b in ball_clean if b["frame"] < half_frame],
                 court_w, court_h, ppm,
                 out / "topdown_first_half.png", title="Primera mitad",
                 half_court=half_court)
    second = {tid: [s for s in ss if s["frame"] >= half_frame]
              for tid, ss in filtered.items()}
    second = {k: v for k, v in second.items() if len(v) >= 5}
    save_topdown(second, [b for b in ball_clean if b["frame"] >= half_frame],
                 court_w, court_h, ppm,
                 out / "topdown_second_half.png", title="Segunda mitad",
                 half_court=half_court)

    if save_diagnostic:
        save_diagnostic_frame(video_path, cal, rejected_counts,
                              out / "diagnostic.png", H, ppm)
        if pose_estimator:
            save_pose_sample(video_path, filtered,
                              out / "pose_sample.png")

    with open(out / "scouting_data.json", "w") as f:
        json.dump(metrics, f, indent=2)

    quality_label = ("EXCELENTE" if score >= 80 else
                     "BUENO" if score >= 60 else
                     "REGULAR" if score >= 40 else
                     "BAJO — interpreta con cuidado")

    print(f"\n┌─ Resultado ────────────────────────────")
    print(f"│ Tracks limpios: {len(filtered)} "
          f"(de {n_tracks_before_stitch} crudos → {len(raw_tracks)} tras cosido)")
    if n_stitched > 0:
        print(f"│ Tracks cosidos: {n_stitched} fragmentos unidos")
    print(f"│ Balones: {len(ball_clean)} ({metrics['ball_detection_rate']*100:.1f}%) "
          f"[{ball_detector}]")
    if pose_estimator:
        print(f"│ Pose: {len(pose_stats)} tracks con keypoints")
    else:
        print(f"│ Pose: deshabilitado")
    print(f"│ Rechazos: {sum(rejected_counts.values())}")
    print(f"│")
    print(f"│ ★ CALIDAD: {score}/100 — {quality_label}")
    for k, v in score_breakdown.items():
        print(f"│   {k}: {v}")
    print(f"└────────────────────────────────────────")
    return metrics


def compute_quality_score(tracks, zones, ball_frames, rejected,
                          total_samples, expected_tracks, half_court):
    """ball_frames: número de frames únicos con al menos una detección de balón
    (no el total de cajas) — acotado a [0, total_samples]."""
    breakdown = {}
    # Simetrico: premia ACERTAR el numero de jugadoras, no tener "muchas".
    # 48 tracks para 6 jugadoras es tracking roto (espectadores/fragmentos),
    # no 30/30. Penaliza sobre-deteccion tanto como sub-deteccion. La vieja
    # `len/expected*30` topada daba 30/30 a cualquier conteo >= esperado.
    n_tr = len(tracks)
    track_ratio = min(n_tr, expected_tracks) / max(n_tr, expected_tracks, 1)
    track_pts = int(round(30 * track_ratio))
    breakdown["tracks"] = f"{track_pts}/30 ({n_tr} de {expected_tracks})"

    expected_zones = 6 if half_court else 12
    zones_with_data = len([v for v in zones.values() if v > 3])
    zone_pts = min(25, int(zones_with_data / expected_zones * 25))
    breakdown["zonas"] = f"{zone_pts}/25 ({zones_with_data} de {expected_zones})"

    ball_rate = ball_frames / max(total_samples, 1)
    if ball_rate >= 0.50:
        ball_pts = 20
    elif ball_rate >= 0.10:
        ball_pts = int(4 + (ball_rate - 0.10) / 0.40 * 16)
    else:
        ball_pts = int(ball_rate / 0.10 * 4)
    breakdown["balon"] = f"{ball_pts}/20 ({ball_rate*100:.1f}%)"

    total_dets = sum(len(s) for s in tracks.values()) + sum(rejected.values())
    if total_dets > 0:
        accept_ratio = sum(len(s) for s in tracks.values()) / total_dets
        accept_pts = int(accept_ratio * 15)
    else:
        accept_pts = 0
    breakdown["filtrado"] = f"{accept_pts}/15 (rechazos: {sum(rejected.values())})"

    # Termometro honesto de continuidad de identidad. Blend 50/50 de dos
    # señales que NO se contradicen entre si:
    #   - persistencia: de las N tracks principales (las expected_tracks mas
    #     largas, ignorando la cola de fragmentos espurios), que fraccion del
    #     video duran. 50%+ = credito lleno.
    #   - consolidacion: que tan cerca quedo el #tracks del esperado. Penaliza
    #     tanto sobre-segmentar (IDs partidos) como sub-detectar.
    # El viejo `mean(duracion)/video` mezclaba ambos males y peleaba contra la
    # sub-metrica `tracks`: mas fragmentos subian `tracks` y hundian este.
    if tracks and total_samples > 0:
        lengths = sorted((len(s) for s in tracks.values()), reverse=True)
        top = lengths[:expected_tracks]
        persist_ratio = (sum(top) / len(top)) / total_samples
        persist_score = min(1.0, persist_ratio / 0.50)
        consol_score = max(0.0, 1.0 - abs(len(tracks) - expected_tracks)
                                       / expected_tracks)
        stability_pts = int(round(10 * (0.5 * persist_score
                                        + 0.5 * consol_score)))
    else:
        persist_score = consol_score = 0.0
        stability_pts = 0
    breakdown["estabilidad"] = (
        f"{stability_pts}/10 (persist {persist_score:.2f}, "
        f"consol {consol_score:.2f})")

    total = track_pts + zone_pts + ball_pts + accept_pts + stability_pts
    return total, breakdown


def save_topdown(tracks, ball, court_w, court_h, ppm, path,
                 title="", metrics=None, half_court=False):
    court_W = int(court_w * ppm)
    court_H = int(court_h * ppm)
    pad = 50
    stats_w = 220 if metrics else 0
    W = court_W + pad * 2 + stats_w
    Hi = court_H + pad * 2 + 60

    img = np.full((Hi, W, 3), 12, dtype=np.uint8)
    cx0, cy0 = pad, pad + 40

    cv2.rectangle(img, (cx0, cy0), (cx0 + court_W, cy0 + court_H),
                  (28, 26, 22), -1)
    cv2.rectangle(img, (cx0, cy0), (cx0 + court_W, cy0 + court_H),
                  PAL_BONE, 2)

    if not half_court:
        cv2.line(img, (cx0, cy0 + court_H // 2),
                 (cx0 + court_W, cy0 + court_H // 2), PAL_OXBLOOD, 3)
        att = int(3 * ppm)
        cv2.line(img, (cx0, cy0 + court_H // 2 - att),
                 (cx0 + court_W, cy0 + court_H // 2 - att), PAL_DARK, 1)
        cv2.line(img, (cx0, cy0 + court_H // 2 + att),
                 (cx0 + court_W, cy0 + court_H // 2 + att), PAL_DARK, 1)
        zones_pos = {
            "A4": (1.5, 7.5), "A3": (4.5, 7.5), "A2": (7.5, 7.5),
            "A5": (1.5, 1.5), "A6": (4.5, 1.5), "A1": (7.5, 1.5),
            "B4": (7.5, 10.5), "B3": (4.5, 10.5), "B2": (1.5, 10.5),
            "B5": (7.5, 16.5), "B6": (4.5, 16.5), "B1": (1.5, 16.5),
        }
    else:
        cv2.line(img, (cx0, cy0), (cx0 + court_W, cy0), PAL_OXBLOOD, 3)
        att = int(3 * ppm)
        cv2.line(img, (cx0, cy0 + att), (cx0 + court_W, cy0 + att),
                 PAL_DARK, 1)
        zones_pos = {
            "A4": (1.5, 1.5), "A3": (4.5, 1.5), "A2": (7.5, 1.5),
            "A5": (1.5, 7.5), "A6": (4.5, 7.5), "A1": (7.5, 7.5),
        }

    for zname, (mx, my) in zones_pos.items():
        px = int(cx0 + mx * ppm)
        py = int(cy0 + my * ppm)
        cv2.putText(img, zname, (px - 13, py + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 56, 50), 1)

    overlay = img.copy()
    palette = [(40, 40, 165), (60, 200, 220), (122, 156, 90),
               (198, 154, 60), (208, 100, 200), (100, 100, 250),
               (50, 180, 100), (220, 120, 90), (160, 80, 220),
               (220, 200, 60), (60, 60, 200), (200, 60, 100)]
    for i, (tid, samples) in enumerate(tracks.items()):
        color = palette[i % len(palette)]
        for s in samples:
            px = int(cx0 + s["court_x"] * ppm)
            py = int(cy0 + s["court_y"] * ppm)
            cv2.circle(overlay, (px, py), 4, color, -1)
    img = cv2.addWeighted(overlay, 0.55, img, 0.45, 0)

    for b in ball:
        px = int(cx0 + b["court_x"] * ppm)
        py = int(cy0 + b["court_y"] * ppm)
        cv2.circle(img, (px, py), 4, PAL_BALL, 2)

    cv2.putText(img, f"CLARA - {title}", (cx0, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, PAL_BONE, 1)

    leg_y = cy0 + court_H + 25
    cv2.circle(img, (cx0 + 8, leg_y), 4, PAL_OXBLOOD, -1)
    cv2.putText(img, "jugadora", (cx0 + 20, leg_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, PAL_BONE, 1)
    cv2.circle(img, (cx0 + 100, leg_y), 4, PAL_BALL, 2)
    cv2.putText(img, "balon", (cx0 + 112, leg_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, PAL_BONE, 1)

    if metrics and stats_w > 0:
        sx = cx0 + court_W + pad
        sy = cy0
        score = metrics.get("quality_score", 0)
        score_color = (PAL_OK if score >= 60 else
                       PAL_WARN if score >= 40 else PAL_REJECT)
        cv2.putText(img, "CALIDAD", (sx, sy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, PAL_BONE, 1)
        cv2.putText(img, f"{score}/100", (sx, sy + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, score_color, 2)

        lines = [
            "",
            f"Duracion: {metrics['duration_min']:.1f}m",
            f"Tracks: {metrics['filtered_tracks']}",
            f"Balones: {metrics['ball_detections_oncourt']}",
            f"Detector: {metrics.get('ball_detector', '?')}",
            f"Pose: {metrics.get('pose_mode', 'none')}",
        ]
        if metrics.get("half_court"):
            lines.append("MODO: HALF-COURT")
        for i, line in enumerate(lines):
            cv2.putText(img, line, (sx, sy + 50 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, PAL_BONE, 1)

    cv2.imwrite(str(path), img)


def save_diagnostic_frame(video_path, cal, rejected_counts, path, H, ppm):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return
    Hf, Wf = frame.shape[:2]
    corners = cal.get("pixel_corners", [])
    if corners:
        pts = np.array(corners, dtype=np.int32)
        cv2.polylines(frame, [pts], True, (60, 220, 230), 2)
        for x, y in corners:
            cv2.circle(frame, (int(x), int(y)), 6, (60, 220, 230), -1)
    hor = cal.get("court_horizon_y")
    if hor is not None:
        cv2.line(frame, (0, int(hor)), (Wf, int(hor)),
                 (60, 156, 220), 1, cv2.LINE_AA)
    y_off = 30
    cv2.putText(frame, "DIAGNOSTICO CLARA",
                (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (232, 224, 208), 1)
    y_off += 24
    cv2.putText(frame, f"Rechazos: {sum(rejected_counts.values())}",
                (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (232, 224, 208), 1)
    y_off += 18
    for reason, count in rejected_counts.items():
        cv2.putText(frame, f"  {reason}: {count}",
                    (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (60, 156, 220), 1)
        y_off += 16
    cv2.imwrite(str(path), frame)


def save_pose_sample(video_path, filtered_tracks, path):
    """Render frame with skeletons of detected players."""
    try:
        from pose_rtmlib import draw_pose
    except ImportError:
        return

    best_frame_idx = None
    best_count = 0
    for tid, samples in filtered_tracks.items():
        for s in samples:
            if "pose" in s:
                fi = s["frame"]
                count = sum(1 for tid2, samples2 in filtered_tracks.items()
                           for s2 in samples2
                           if s2["frame"] == fi and "pose" in s2)
                if count > best_count:
                    best_count = count
                    best_frame_idx = fi
                if best_count >= 6:
                    break

    if best_frame_idx is None:
        return

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, best_frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return

    for tid, samples in filtered_tracks.items():
        for s in samples:
            if s["frame"] == best_frame_idx and "pose" in s:
                draw_pose(frame, s["pose"]["kp"], s["pose"]["sc"])

    cv2.putText(frame, f"CLARA pose sample · frame {best_frame_idx}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (232, 224, 208), 2)
    cv2.imwrite(str(path), frame)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="CLARA v0.8 — multimodal scouting")
    p.add_argument("video")
    p.add_argument("--calibration", default=None,
                   help="cal.json manual (de MIRA). Opcional si se usa "
                        "--court-model.")
    p.add_argument("--court-model", default=None,
                   help="Modelo YOLO-pose de cancha para auto-calibracion. "
                        "Si se da, CLARA calibra sola; si falla, cae a "
                        "--calibration.")
    p.add_argument("--out", default="out")
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--person-conf", type=float, default=0.4)
    p.add_argument("--ball-conf", type=float, default=0.10)
    p.add_argument("--ball-detector", choices=["yolo", "vballnet"], default="yolo")
    p.add_argument("--ball-model", default=None)
    p.add_argument("--vballnet-model", default=None)
    p.add_argument("--vballnet-stride", type=int, default=1,
                   help="Submuestreo del buffer de VballNet (1=todos los frames). "
                        "Subir a 2-3 reduce RAM e inferencias en clips largos a "
                        "costa de recall.")
    p.add_argument("--pose", choices=["none", "rtmlib"], default="none")
    a = p.parse_args()
    if a.calibration is None and a.court_model is None:
        p.error("Se requiere --calibration cal.json o --court-model modelo.pt")
    run(a.video, a.calibration, a.out, a.stride,
        a.person_conf, a.ball_conf,
        ball_detector=a.ball_detector,
        ball_model_path=a.ball_model,
        vballnet_model=a.vballnet_model,
        vballnet_stride=a.vballnet_stride,
        pose_mode=a.pose,
        court_model=a.court_model)
