"""
court_calibration.py
====================
Calibración de homografía  imagen <-> cancha (metros)  para CLARA.

Dos caminos:

  1) MANUAL  (recomendado para trípode fijo)
     Clic en las esquinas/puntos UNA sola vez -> se guarda en JSON y se
     reutiliza para todo el video y para toda grabación con el mismo encuadre.
     Cero costo de inferencia, máxima robustez, nunca falla a media grabación.

  2) AUTOMÁTICO  (cámara que cambia de ángulo)
     Usa el modelo de segmentación de cancha de volleyball_analytics
     (court/weights/best.pt) para extraer las 4 esquinas del frame.
     Menos robusto si las líneas están ocluidas por jugadores.

La homografía mapea pixeles -> METROS sobre el sistema canónico de
court_geometry. Su inversa mapea metros -> pixeles para dibujar overlays.

NOTA DE INTEGRACIÓN (CLARA):
El JSON que produce este `CourtCalibrator` (claves "src_pts"/"dst_pts"/"H")
NO es el mismo formato que el `data/cal.json` del pipeline principal
(claves "homography_matrix"/"pixels_per_meter"/...). Son dos sistemas de
calibración independientes; no los cargues cruzados.
"""

import json
import numpy as np
import cv2

from .court_geometry import COURT_CORNERS_M, LANDMARKS_M, COURT_WIDTH, COURT_LENGTH


def order_points(pts):
    """Ordena 4 puntos a [TL, TR, BR, BL] en espacio imagen (truco suma/diferencia)."""
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = (pts[:, 0] - pts[:, 1])
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmax(d)]
    bl = pts[np.argmin(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def corners_from_mask(mask):
    """
    Extrae 4 esquinas (orden TL,TR,BR,BL imagen) del blob de cancha más grande
    de una máscara binaria. Limpia ruido y busca el epsilon que da 4 vértices;
    si no, cae a rectángulo de área mínima.
    """
    m = (mask > 0).astype(np.uint8) * 255
    k = np.ones((5, 5), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)

    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(cnt, True)

    for eps in np.linspace(0.01, 0.08, 8):
        approx = cv2.approxPolyDP(cnt, eps * peri, True)
        if len(approx) == 4:
            return order_points(approx.reshape(-1, 2))

    box = cv2.boxPoints(cv2.minAreaRect(cnt))
    return order_points(box)


class CourtCalibrator:
    def __init__(self):
        self.H = None          # imagen -> metros
        self.H_inv = None      # metros -> imagen
        self.src_pts = None    # puntos imagen usados
        self.dst_pts = None    # puntos metros correspondientes

    # ------------------------------------------------------------------
    # construcción de la homografía
    # ------------------------------------------------------------------
    def _set(self, src_pts, dst_pts):
        src = np.asarray(src_pts, np.float32)
        dst = np.asarray(dst_pts, np.float32)
        if len(src) < 4:
            raise ValueError("Se necesitan al menos 4 puntos.")
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None:
            raise RuntimeError("findHomography falló: revisa que los puntos no "
                               "sean colineales ni estén mal ordenados.")
        self.H = H
        self.H_inv = np.linalg.inv(H)
        self.src_pts = src
        self.dst_pts = dst
        return H

    def from_corners(self, corners_img):
        """corners_img: 4 puntos en orden [A_izq, A_der, B_der, B_izq] (TL,TR,BR,BL)."""
        return self._set(corners_img, COURT_CORNERS_M)

    def from_points(self, img_pts, landmark_names):
        """
        Homografía a partir de N>=4 puntos imagen y sus nombres de landmark.
        Permite usar más de 4 puntos (mínimos cuadrados) para mayor precisión.
        """
        dst = np.array([LANDMARKS_M[n] for n in landmark_names], np.float32)
        return self._set(np.asarray(img_pts, np.float32), dst)

    # ------------------------------------------------------------------
    # camino automático (modelo de segmentación)
    # ------------------------------------------------------------------
    def from_court_model(self, frame, court_model, conf=0.25):
        """
        Estima la homografía con un modelo YOLOv8-seg de cancha
        (court/weights/best.pt de volleyball_analytics).
        Devuelve las 4 esquinas detectadas (para que las verifiques visualmente).

        Supuesto: la cámara está aprox. derecha (cancha cuasi alineada, lado A
        hacia abajo en la imagen). Si está muy rotada, usa calibración manual.
        """
        res = court_model.predict(frame, conf=conf, verbose=False)[0]
        if res.masks is None or len(res.masks) == 0:
            raise RuntimeError("El modelo no segmentó cancha en este frame.")

        masks = res.masks.data.cpu().numpy()           # (n, h, w) en 0..1
        mask = (masks.sum(axis=0) > 0.5).astype(np.uint8)
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

        corners = corners_from_mask(mask)
        if corners is None:
            raise RuntimeError("No se pudieron extraer esquinas de la máscara.")
        self.from_corners(corners)
        return corners

    # ------------------------------------------------------------------
    # camino manual (clics)
    # ------------------------------------------------------------------
    def from_clicks(self, frame, landmark_names=None, window="Calibracion CLARA"):
        """
        Abre una ventana y pide clicar los puntos EN ORDEN.
        landmark_names: claves de LANDMARKS_M. Default = 4 esquinas.
        RECOMENDADO para más precisión: las 4 esquinas + la red:
          ["A_corner_left","A_corner_right","B_corner_right","B_corner_left",
           "net_left","net_right"]
        Teclas: u = deshacer último, q = cancelar.
        """
        if landmark_names is None:
            landmark_names = ["A_corner_left", "A_corner_right",
                              "B_corner_right", "B_corner_left"]
        clicks = []
        base = frame.copy()

        def redraw():
            disp = base.copy()
            for i, (px, py) in enumerate(clicks):
                cv2.circle(disp, (px, py), 6, (0, 255, 255), -1)
                cv2.putText(disp, landmark_names[i], (px + 8, py - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            idx = len(clicks)
            msg = (f"Clic: {landmark_names[idx]}   (u=deshacer  q=cancelar)"
                   if idx < len(landmark_names) else "Listo. Cierra la ventana.")
            cv2.putText(disp, msg, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)
            cv2.imshow(window, disp)

        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < len(landmark_names):
                clicks.append((x, y))
                redraw()

        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window, on_mouse)
        redraw()
        while True:
            key = cv2.waitKey(20) & 0xFF
            if key == ord('u') and clicks:
                clicks.pop()
                redraw()
            elif key == ord('q'):
                cv2.destroyWindow(window)
                raise RuntimeError("Calibración cancelada por el usuario.")
            if len(clicks) == len(landmark_names):
                cv2.waitKey(300)
                break
        cv2.destroyWindow(window)
        self.from_points(clicks, landmark_names)
        return np.array(clicks, np.float32)

    # ------------------------------------------------------------------
    # persistencia
    # ------------------------------------------------------------------
    def save(self, path):
        if self.H is None:
            raise RuntimeError("No hay homografía que guardar.")
        data = {
            "src_pts": self.src_pts.tolist(),
            "dst_pts": self.dst_pts.tolist(),
            "H": self.H.tolist(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        c = cls()
        c._set(np.array(data["src_pts"], np.float32),
               np.array(data["dst_pts"], np.float32))
        return c

    # ------------------------------------------------------------------
    # transformaciones
    # ------------------------------------------------------------------
    def image_to_court(self, pts):
        """(N,2) pixeles -> (N,2) metros."""
        pts = np.asarray(pts, np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)

    def court_to_image(self, pts):
        """(N,2) metros -> (N,2) pixeles."""
        pts = np.asarray(pts, np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H_inv).reshape(-1, 2)

    def draw_overlay(self, frame, color=(0, 220, 0)):
        """Dibuja el contorno de la cancha sobre el frame original (control de calidad)."""
        out = frame.copy()
        corners = self.court_to_image(COURT_CORNERS_M).astype(np.int32)
        cv2.polylines(out, [corners.reshape(-1, 1, 2)], True, color, 2)
        # red proyectada
        net = self.court_to_image([[0, 9.0], [COURT_WIDTH, 9.0]]).astype(np.int32)
        cv2.line(out, tuple(net[0]), tuple(net[1]), (90, 200, 255), 2)
        return out
