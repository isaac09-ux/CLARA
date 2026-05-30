"""
jersey_id.py — Identificación de jugadora por número de jersey.

CLARA ya asigna IDs de track persistentes (ByteTrack), pero esos IDs son
anónimos y se reinician en cada video: el track #5 de hoy no es el track #5
de mañana. Este módulo convierte un track anónimo en una jugadora con nombre
leyendo el número del jersey y cruzándolo contra un roster conocido.

POR QUÉ NÚMERO Y NO CARA
------------------------
Reconocimiento facial no sirve en voleibol amateur: blur de movimiento,
jugadoras de espaldas la mitad del tiempo, caras de pocos píxeles en video de
baja resolución. El número de jersey es texto grande y de alto contraste
diseñado precisamente para leerse a distancia. Lo tratamos como un problema
de scene text recognition (igual que la literatura de SoccerNet jersey number).

EL TRUCO: VOTACIÓN POR TRACK, NO POR FRAME
------------------------------------------
El número solo es legible en una fracción de los frames (espalda de frente,
sin oclusión, sin blur). Un solo frame es poco confiable. Pero ByteTrack ya
mantiene el ID del track durante toda la jugada — así que acumulamos cada
lectura de OCR como un VOTO ponderado por confianza sobre ese track, y al
final el número con más peso gana. Una mala lectura aislada queda enterrada
por las buenas.

CONJUNTO CERRADO
----------------
Solo cuentan votos por números que están en el roster. Si el OCR lee "33" y
nadie en Las Chispas usa el 33, el voto se descarta. Esto filtra casi todos
los errores de OCR sin entrenar nada. Consecuencia útil: las jugadoras rival
(que no están en el roster) se quedan como tracks anónimos — exactamente lo
que quieres para scouting (identificas a las tuyas, las rivales se analizan
por zona).

LIMITACIONES CONOCIDAS (v0.7.0)
-------------------------------
- Si una rival usa un número que coincide con uno del roster, se etiqueta
  mal. Mitigación: el track trae `side` (A/B) — filtra por lado en el reporte.
- Kit Casa (hueso con números rosa palo): contraste bajo para OCR. El
  preprocesamiento aplica CLAHE para rescatarlo, pero el kit Visitante
  (oxblood) y el Third (negro/oro) leen mejor. En partido de local, espera
  más lecturas fallidas — la votación por track lo compensa parcialmente.
- OCR es costoso. El llamador debe espaciar las observaciones (id_stride) y
  este módulo además topa las observaciones por track (max_obs_per_track).

Dependencia: easyocr (Apache 2.0). No añade peso pesado nuevo — easyocr usa
torch, que CLARA ya instala para YOLO.
"""
import json
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

# Índices de keypoints COCO usados para recortar la zona del número.
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12


def load_roster(path):
    """Lee roster.json -> dict {int numero: str nombre}.

    Formato esperado:
        {"team": "...", "players": {"5": "Jess", "7": "...", ...}}
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"No encontré el roster: {path}. Copia roster.example.json "
            f"a roster.json y pon tus jugadoras."
        )
    data = json.loads(p.read_text())
    players = data.get("players", {})
    roster = {}
    for num, name in players.items():
        try:
            roster[int(num)] = str(name)
        except (ValueError, TypeError):
            continue
    if not roster:
        raise ValueError(f"El roster {path} no tiene jugadoras válidas.")
    return roster


def torso_crop(frame, bbox, pose=None, pad=0.18):
    """Recorta la región del número del jersey.

    Con pose disponible usa hombros (5,6) y caderas (11,12): el número va
    en ese rectángulo (pecho al frente, espalda alta atrás). Sin pose, cae
    a proporciones del bbox — el número se ubica en el torso medio-alto.

    Devuelve el recorte BGR, o None si la geometría es inválida.
    """
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    if bw < 4 or bh < 4:
        return None

    cx1 = cy1 = cx2 = cy2 = None
    if pose is not None:
        kp = pose.get("kp")
        sc = pose.get("sc")
        if kp is not None and sc is not None and len(kp) > R_HIP:
            idx = [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]
            if all(sc[i] >= 0.3 for i in idx):
                xs = [kp[i][0] for i in idx]
                ys = [kp[i][1] for i in idx]
                cx1, cx2 = min(xs), max(xs)
                cy1, cy2 = min(ys), max(ys)

    if cx1 is None:  # fallback por proporciones del bbox
        cx1 = x1 + 0.12 * bw
        cx2 = x2 - 0.12 * bw
        cy1 = y1 + 0.18 * bh
        cy2 = y1 + 0.55 * bh

    # padding y recorte seguro a los límites del frame
    px, py = pad * (cx2 - cx1), pad * (cy2 - cy1)
    ix1 = max(0, int(cx1 - px));  iy1 = max(0, int(cy1 - py))
    ix2 = min(W, int(cx2 + px));  iy2 = min(H, int(cy2 + py))
    if ix2 - ix1 < 6 or iy2 - iy1 < 6:
        return None
    return frame[iy1:iy2, ix1:ix2]


def preprocess_for_ocr(crop, min_side=96):
    """Gris + CLAHE + upscale. CLAHE rescata números de bajo contraste
    (kit Casa hueso/rosa); el upscale ayuda a easyocr con recortes chicos."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    h, w = gray.shape
    short = min(h, w)
    if short < min_side:
        scale = min_side / short
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


class JerseyIdentifier:
    """Lee números de jersey y resuelve identidades por votación por track.

    Uso:
        ident = JerseyIdentifier("roster.json")
        # en el loop, cada id_stride muestras:
        ident.observe(track_id, frame_img, bbox, pose=sample.get("pose"))
        # al final:
        identities = ident.resolve()   # {track_id: {...}}
    """

    def __init__(self, roster_path, min_crop_px=40,
                 vote_threshold=1.5, min_dominance=0.5,
                 max_obs_per_track=30, ocr_conf_min=0.3, gpu=False):
        """
        min_crop_px       lado mínimo del recorte; más chico = OCR basura, se salta
        vote_threshold    peso acumulado mínimo (suma de confianzas) para nombrar
        min_dominance     fracción del peso total que debe llevarse el ganador
        max_obs_per_track tope de observaciones por track (acota costo de OCR)
        ocr_conf_min      confianza mínima de easyocr para contar una lectura
        """
        self.roster = load_roster(roster_path)
        self.valid_numbers = set(self.roster)
        self.min_crop_px = min_crop_px
        self.vote_threshold = vote_threshold
        self.min_dominance = min_dominance
        self.max_obs_per_track = max_obs_per_track
        self.ocr_conf_min = ocr_conf_min

        # votos[tid] = {numero: peso_acumulado};  obs[tid] = lecturas intentadas
        self._votes = defaultdict(lambda: defaultdict(float))
        self._obs = defaultdict(int)

        # Import perezoso: easyocr es pesado de cargar y descarga modelos la
        # primera vez. Así clara.py puede capturar el ImportError con gracia.
        try:
            import easyocr
        except ImportError as e:
            raise ImportError(
                "easyocr no instalado. Instala con: pip install easyocr"
            ) from e
        print("[jersey_id] Cargando easyocr (descarga modelos la 1a vez)...")
        self._reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)

    def observe(self, tid, frame, bbox, pose=None):
        """Intenta leer el número de un track en un frame. Acumula votos.

        Es barato salir temprano: si el track ya tiene suficientes
        observaciones, no corremos OCR otra vez.
        """
        if self._obs[tid] >= self.max_obs_per_track:
            return
        crop = torso_crop(frame, bbox, pose=pose)
        if crop is None or min(crop.shape[:2]) < self.min_crop_px:
            return
        self._obs[tid] += 1

        proc = preprocess_for_ocr(crop)
        try:
            results = self._reader.readtext(
                proc, allowlist="0123456789", detail=1, paragraph=False)
        except Exception:
            return

        for _, text, conf in results:
            text = text.strip()
            if not text.isdigit() or not (1 <= len(text) <= 2):
                continue
            num = int(text)
            if conf < self.ocr_conf_min or num not in self.valid_numbers:
                continue
            self._votes[tid][num] += float(conf)

    def resolve(self):
        """Cierra la votación. Devuelve {tid: identidad}.

        identidad = {number, name, confidence, votes, weight}
        number es None cuando no hay evidencia suficiente -> 'desconocida'.
        """
        out = {}
        for tid, obs in self._obs.items():
            votes = self._votes.get(tid, {})
            total = sum(votes.values())
            if not votes or total <= 0:
                out[tid] = {"number": None, "name": "desconocida",
                            "confidence": 0.0, "votes": obs, "weight": 0.0}
                continue
            best_num = max(votes, key=votes.get)
            best_w = votes[best_num]
            dominance = best_w / total
            named = (best_w >= self.vote_threshold and
                     dominance >= self.min_dominance)
            out[tid] = {
                "number": best_num if named else None,
                "name": self.roster[best_num] if named else "desconocida",
                "confidence": round(dominance, 2),
                "votes": obs,
                "weight": round(best_w, 2),
            }
        return out
