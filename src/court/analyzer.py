"""
analyzer.py  (originalmente clara_analyzer.py de volleyball_analytics)
=====================================================================

⚠️  AVISO DE DUPLICACIÓN — LEER ANTES DE USAR  ⚠️
-------------------------------------------------
Este `CLARACourtAnalyzer` corre su PROPIO pipeline de detección + tracking
y DUPLICA lo que el pipeline principal de CLARA (`src/clara.py`) ya hace,
con menos robustez:

  • Jugadores + ByteTrack:
        aquí -> YOLO(player_model).track(tracker="bytetrack.yaml", classes=[0])
        clara.py -> YOLO11m + `bytetrack_clara.yaml` (track_buffer tuneado) +
                    filtro de ROI por polígono + stitch_tracks() (cose IDs
                    fragmentados) + pose (rtmlib). MÁS ROBUSTO.
  • Balón:
        aquí -> un solo YOLO(ball_model).predict por frame, toma el de mayor conf.
        clara.py -> backends `yolo` y `vballnet` + filter_ball_tracks() (mata FPs)
                    + reconstruct_trajectory(). MÁS COMPLETO.
  • Proyección imagen->cancha:  duplica project() de clara.py.
  • Zonas:                      solapa con zone_for_court_pos() de clara.py
                                (aquí enteros 1-6 vía court_geometry; allá "A4").
  • Top-down / CSV / JSON:      solapa con save_topdown() y la exportación de clara.py.

Por eso este módulo NO está cableado dentro de `clara.py`. Se conserva como
pipeline ALTERNATIVO / DE REFERENCIA, autocontenido, útil para:
  - prototipado rápido sobre los pesos de volleyball_analytics (ball/court),
  - validar la geometría/zonas FIVB de `court_geometry` de forma aislada.

Si en el futuro quieres aprovechar SOLO la proyección+zonas FIVB de aquí sin
re-detectar, alimenta tus propias cajas (las de clara.py) a través de la
homografía de `CourtCalibrator` y de `court_geometry.classify_zone`, en vez de
volver a correr YOLO/ByteTrack en este módulo.

-------------------------------------------------
Núcleo de integración (descripción original):
Por cada frame:
  1. Detecta + trackea jugadores (YOLOv8 + ByteTrack vía Ultralytics).
  2. Detecta el balón (modelo ball/best.pt de volleyball_analytics).
  3. Proyecta posiciones imagen -> coordenadas de cancha (metros) con la
     homografía calibrada (CourtCalibrator).
  4. Asigna lado (A/B) y zona oficial (1-6) a cada jugador.
  5. Dibuja overlay sobre el video y una vista top-down limpia.
  6. Acumula filas de scouting -> CSV / JSON.

Punto de honestidad sobre el balón: la homografía asume que el punto está SOBRE
el plano de la cancha. El balón en el aire sufre paralaje -> su proyección al
piso es exacta solo cuando está cerca del suelo (saques, recepciones, defensas,
caídas/landing). Para esos eventos es oro; para trayectoria en vuelo es aproximada.
Por eso cada fila de balón lleva un flag `low` (heurístico de altura del bbox).
"""

import json
import csv
from collections import defaultdict, Counter

import numpy as np
import cv2

from .court_geometry import (classify_zone, side_of, in_court,
                             make_topdown_canvas, draw_court, m2px_render,
                             COURT_LENGTH)

# Ultralytics se importa de forma perezosa dentro de __init__ para que el resto
# del módulo (y los tests de geometría) no dependan de tenerlo instalado.


class CLARACourtAnalyzer:
    def __init__(self, calibrator,
                 player_model_path="yolov8x.pt",
                 ball_model_path=None,
                 player_conf=0.30, ball_conf=0.25,
                 tracker="bytetrack.yaml",
                 court_margin=1.0, flip_lr=False,
                 device=None):
        from ultralytics import YOLO  # import perezoso

        self.cal = calibrator
        self.player_model = YOLO(player_model_path)
        self.ball_model = YOLO(ball_model_path) if ball_model_path else None

        self.player_conf = player_conf
        self.ball_conf = ball_conf
        self.tracker = tracker
        self.court_margin = court_margin      # margen (m) para aceptar detecciones
        self.flip_lr = flip_lr
        self.device = device

        self.records = []   # filas de scouting (formato largo)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _foot_point(box):
        """Punto de contacto con el piso = centro inferior del bbox."""
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2.0, float(y2))

    @staticmethod
    def _center(box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    # ------------------------------------------------------------------
    # procesamiento de un frame
    # ------------------------------------------------------------------
    def process_frame(self, frame, frame_idx, t_sec=None):
        annotated = frame.copy()
        data = {"frame": frame_idx, "t": t_sec, "players": [], "ball": None}

        # ---------- jugadores + ByteTrack ----------
        kw = dict(persist=True, conf=self.player_conf, classes=[0],
                  tracker=self.tracker, verbose=False)
        if self.device is not None:
            kw["device"] = self.device
        res = self.player_model.track(frame, **kw)[0]

        if res.boxes is not None and res.boxes.id is not None:
            boxes = res.boxes.xyxy.cpu().numpy()
            ids = res.boxes.id.cpu().numpy().astype(int)
            for box, tid in zip(boxes, ids):
                foot = self._foot_point(box)
                cx, cy = self.cal.image_to_court([foot])[0]
                if not in_court(cx, cy, margin=self.court_margin):
                    continue
                side = side_of(cy)
                zone = classify_zone(cx, cy, side, flip_lr=self.flip_lr)
                player = {"track_id": int(tid),
                          "x_m": round(float(cx), 3), "y_m": round(float(cy), 3),
                          "side": side, "zone": int(zone),
                          "bbox": [float(v) for v in box]}
                data["players"].append(player)
                self._annotate_player(annotated, box, tid, side, zone)

        # ---------- balón ----------
        if self.ball_model is not None:
            bkw = dict(conf=self.ball_conf, verbose=False)
            if self.device is not None:
                bkw["device"] = self.device
            br = self.ball_model.predict(frame, **bkw)[0]
            if br.boxes is not None and len(br.boxes) > 0:
                bxy = br.boxes.xyxy.cpu().numpy()
                cfs = br.boxes.conf.cpu().numpy()
                i = int(np.argmax(cfs))
                box = bxy[i]
                center = self._center(box)
                cx, cy = self.cal.image_to_court([center])[0]
                # heurística: balón "bajo" si su bbox está en la mitad inferior
                low = bool(box[3] > frame.shape[0] * 0.5)
                data["ball"] = {"x_m": round(float(cx), 3),
                                "y_m": round(float(cy), 3),
                                "conf": round(float(cfs[i]), 3),
                                "low": low,
                                "bbox": [float(v) for v in box]}
                self._annotate_ball(annotated, box)

        self._accumulate(data)
        return annotated, data

    # ------------------------------------------------------------------
    # anotación sobre el frame original
    # ------------------------------------------------------------------
    def _annotate_player(self, img, box, tid, side, zone):
        x1, y1, x2, y2 = [int(v) for v in box]
        col = (255, 160, 80) if side == 'A' else (80, 255, 160)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        cv2.circle(img, (int((x1 + x2) / 2), y2), 4, (0, 0, 255), -1)
        cv2.putText(img, f"#{tid} {side}{zone}", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2, cv2.LINE_AA)

    def _annotate_ball(self, img, box):
        x1, y1, x2, y2 = [int(v) for v in box]
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        cv2.circle(img, (cx, cy), max(6, (x2 - x1) // 2), (0, 215, 255), 2)

    # ------------------------------------------------------------------
    # render top-down
    # ------------------------------------------------------------------
    def render_topdown(self, data, scale=40, margin=40):
        img = make_topdown_canvas(scale=scale, margin=margin)
        draw_court(img, scale=scale, margin=margin)
        for p in data["players"]:
            px, py = m2px_render(p["x_m"], p["y_m"], scale, margin)
            col = (255, 160, 80) if p["side"] == 'A' else (80, 255, 160)
            cv2.circle(img, (px, py), 9, col, -1)
            cv2.circle(img, (px, py), 9, (20, 20, 20), 1)
            cv2.putText(img, str(p["track_id"]), (px - 6, py + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 20), 1, cv2.LINE_AA)
        if data["ball"]:
            bx, by = m2px_render(data["ball"]["x_m"], data["ball"]["y_m"], scale, margin)
            cv2.circle(img, (bx, by), 6, (0, 215, 255), -1)
            cv2.circle(img, (bx, by), 6, (20, 20, 20), 1)
        return img

    # ------------------------------------------------------------------
    # acumulación de scouting
    # ------------------------------------------------------------------
    def _accumulate(self, data):
        for p in data["players"]:
            self.records.append({"frame": data["frame"], "t": data["t"],
                                 "kind": "player", "track_id": p["track_id"],
                                 "x_m": p["x_m"], "y_m": p["y_m"],
                                 "side": p["side"], "zone": p["zone"],
                                 "conf": ""})
        if data["ball"]:
            b = data["ball"]
            self.records.append({"frame": data["frame"], "t": data["t"],
                                 "kind": "ball", "track_id": -1,
                                 "x_m": b["x_m"], "y_m": b["y_m"],
                                 "side": "", "zone": "", "conf": b["conf"]})

    # ------------------------------------------------------------------
    # procesar un video completo
    # ------------------------------------------------------------------
    def process_video(self, video_path, out_video=None, topdown_video=None,
                      show=False, every=1, max_frames=None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"No pude abrir el video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        writer = cv2.VideoWriter(out_video, fourcc, fps, (w, h)) if out_video else None
        tdwriter = None

        idx, processed = 0, 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % every == 0:
                ann, data = self.process_frame(frame, idx, idx / fps)
                if writer is not None:
                    writer.write(ann)
                if topdown_video is not None:
                    td = self.render_topdown(data)
                    if tdwriter is None:
                        tdwriter = cv2.VideoWriter(topdown_video, fourcc,
                                                   fps / every,
                                                   (td.shape[1], td.shape[0]))
                    tdwriter.write(td)
                if show:
                    cv2.imshow("CLARA", ann)
                    if (cv2.waitKey(1) & 0xFF) == ord('q'):
                        break
                processed += 1
                if max_frames and processed >= max_frames:
                    break
            idx += 1

        cap.release()
        if writer:
            writer.release()
        if tdwriter:
            tdwriter.release()
        if show:
            cv2.destroyAllWindows()
        return self.records

    # ------------------------------------------------------------------
    # exportar
    # ------------------------------------------------------------------
    def export_csv(self, path):
        keys = ["frame", "t", "kind", "track_id", "x_m", "y_m", "side", "zone", "conf"]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self.records)

    def export_json(self, path):
        with open(path, "w") as f:
            json.dump(self.records, f, indent=2)

    # ------------------------------------------------------------------
    # KPIs rápidos de scouting
    # ------------------------------------------------------------------
    def zone_occupancy(self):
        """{track_id: {zona: nº de frames}}  -- dónde vive cada jugador."""
        per_player = defaultdict(Counter)
        for r in self.records:
            if r["kind"] == "player":
                per_player[r["track_id"]][r["zone"]] += 1
        return {tid: dict(c) for tid, c in per_player.items()}

    def ball_landing_points(self, only_low=True):
        """Lista de (x_m, y_m) del balón (por defecto solo cuando está bajo)."""
        pts = []
        for r in self.records:
            if r["kind"] == "ball":
                pts.append((r["x_m"], r["y_m"]))
        return pts
