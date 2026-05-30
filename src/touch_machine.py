#!/usr/bin/env python3
"""
touch_machine.py v0.1 — Detección de toques sobre la trayectoria del balón.

Lee scouting_data.json producido por CLARA v0.8+ y escupe un JSON enriquecido
con touches[] (frame, jugadora, posición) y rallies[] (agrupados por silencio
del balón). Diseñado como post-procesador independiente: tuneas thresholds y
re-corres sin reprocesar el video.

Pipeline:
  1. Interpola huecos cortos en ball_track (lineal, <=MAX_GAP_INTERP).
  2. Suaviza ruido píxel a píxel con ventana corta.
  3. Calcula magnitud de aceleración en píxeles/frame².
  4. Detecta picos: aceleración alta + separación mínima + prominencia.
  5. Atribuye cada toque al jugador más cercano (interpolando su track).
  6. Agrupa toques en rallies por gap temporal.

Prereq: scouting_data.json (CLARA v0.8.1+) ya trae todo lo necesario:
  - ball_track     detecciones reales del balón en píxeles (+ court_x/y).
  - track_samples  serie temporal por jugadora en court coords.
Ambos campos los emite clara.py automáticamente. Si tu JSON viene de una
versión vieja sin `track_samples`, vuelve a correr CLARA. (Por compat se
sigue aceptando un `tracks` que ya contenga las muestras crudas.)

Uso:
    python touch_machine.py scouting_data.json
    python touch_machine.py scouting_data.json --out enriched.json --plot
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.signal import find_peaks


# ─── Perillas a calibrar contra tu video real ──────────────────────────
VERSION = "0.1"

# Reconstrucción de trayectoria
MAX_GAP_INTERP_FRAMES = 5   # rellena huecos <=5 frames interpolando lineal
SMOOTH_WINDOW = 3           # ventana de suavizado (impar; 3 = casi nada)

# Detección de toques (los dos más importantes)
ACCEL_PCT_THRESHOLD = 92    # percentil de magnitud de aceleración para ser pico
MIN_TOUCH_SPACING_FRAMES = 12  # min entre toques (12 @ 60fps = 0.2s)
MIN_PEAK_PROMINENCE = 3.0   # px/frame² — destaque sobre baseline local

# Atribución a jugador
MAX_ATTRIBUTION_DIST_M = 1.5  # >1.5m del balón = no es tu toque
PLAYER_INTERP_MAX_GAP = 30    # frames máx de interp para posición de jugadora

# Rallies
RALLY_GAP_SECONDS = 3.5


# ─── 1. Carga ──────────────────────────────────────────────────────────

def load_scouting(path):
    with open(path) as f:
        data = json.load(f)

    ball_track = data.get("ball_track")
    if not ball_track:
        raise ValueError(
            "scouting_data.json no tiene ball_track. ¿Corriste CLARA v0.8+ "
            "con el parche del balón aplicado?"
        )

    tracks = _extract_tracks(data)
    if not tracks:
        raise ValueError(
            "scouting_data.json no incluye muestras por jugadora "
            "(track_samples). Vuelve a correr CLARA v0.8.1+ — clara.py las "
            "emite frame a frame junto al resumen `tracks`."
        )
    return data, ball_track, tracks


def _extract_tracks(data):
    """Saca las muestras por jugadora del JSON de CLARA.

    Prefiere `track_samples` (el campo dedicado de v0.8.1+). Cae a `tracks`
    SÓLO si éste contiene las muestras crudas — el `tracks` resumen de CLARA
    (donde 'samples' es un entero, no una lista) se ignora sin reventar.
    Acepta lista [{'id': N, 'samples': [...]}] o dict {tid: [samples]}.
    """
    raw = data.get("track_samples")
    if raw is None:
        raw = data.get("tracks")
    if raw is None:
        return None
    if isinstance(raw, list):
        # [{'id': N, 'samples': [...]}]
        out = {}
        for t in raw:
            samples = t.get("samples") or t.get("track") or []
            if isinstance(samples, list) and samples:
                out[int(t["id"])] = samples
        return out or None
    if isinstance(raw, dict):
        # {tid: [samples]}
        out = {int(k): v for k, v in raw.items()
               if isinstance(v, list) and v}
        return out or None
    return None


# ─── 2. Reconstrucción de trayectoria ──────────────────────────────────

def reconstruct_ball(ball_track, max_gap_interp, smooth_w):
    """ball_track (lista de dets) → arrays alineados por frame.

    Devuelve dict con arrays numpy:
      frames, x, y, present (True donde había detección real, False interp).
    Frames perdidos en huecos grandes quedan como NaN — la aceleración
    pinta NaN ahí y find_peaks los ignora (los huecos no generan toques fantasma).
    """
    if not ball_track:
        return None

    bt = sorted(ball_track, key=lambda d: d["frame"])
    f_min = bt[0]["frame"]
    f_max = bt[-1]["frame"]
    n = f_max - f_min + 1

    x = np.full(n, np.nan)
    y = np.full(n, np.nan)
    conf = np.zeros(n)
    present = np.zeros(n, dtype=bool)

    for d in bt:
        i = d["frame"] - f_min
        # Si hay varias detecciones en el mismo frame, nos quedamos con la más confiada.
        if not present[i] or d["conf"] > conf[i]:
            x[i] = d["img_x"]
            y[i] = d["img_y"]
            conf[i] = d["conf"]
            present[i] = True

    # Interpolar huecos pequeños
    _interp_gaps(x, max_gap_interp)
    _interp_gaps(y, max_gap_interp)

    # Suavizado: media móvil ignorando NaN
    if smooth_w >= 3:
        x = _smooth_nan(x, smooth_w)
        y = _smooth_nan(y, smooth_w)

    frames = np.arange(f_min, f_max + 1)
    return {"frames": frames, "x": x, "y": y, "present": present, "f_min": f_min}


def _interp_gaps(arr, max_gap):
    """In-place: rellena tramos de NaN cuya longitud <= max_gap."""
    n = len(arr)
    i = 0
    while i < n:
        if np.isnan(arr[i]):
            j = i
            while j < n and np.isnan(arr[j]):
                j += 1
            gap = j - i
            if gap <= max_gap and i > 0 and j < n:
                # interpolar entre arr[i-1] y arr[j]
                a, b = arr[i - 1], arr[j]
                for k in range(gap):
                    arr[i + k] = a + (b - a) * (k + 1) / (gap + 1)
            i = j
        else:
            i += 1


def _smooth_nan(arr, w):
    """Media móvil que ignora NaN. Centro de la ventana."""
    half = w // 2
    out = arr.copy()
    for i in range(len(arr)):
        lo = max(0, i - half)
        hi = min(len(arr), i + half + 1)
        window = arr[lo:hi]
        valid = window[~np.isnan(window)]
        if len(valid) >= 2:
            out[i] = valid.mean()
    return out


# ─── 3. Aceleración + picos = toques candidatos ────────────────────────

def detect_touches(ball, fps):
    """Devuelve lista de frames donde la trayectoria se quiebra."""
    x, y = ball["x"], ball["y"]

    # Velocidad (px/frame). El sign cambia abrupto en un toque, así que la
    # MAGNITUD de la aceleración (np.diff dos veces, luego norma) marca el quiebre.
    vx = np.diff(x)
    vy = np.diff(y)
    ax = np.diff(vx)
    ay = np.diff(vy)
    accel = np.sqrt(ax ** 2 + ay ** 2)
    # accel[i] corresponde a la transición entre frames i+1 y i+2
    # Alineamos al frame del medio:
    accel_frames = ball["frames"][1:-1]

    valid = ~np.isnan(accel)
    if valid.sum() < 10:
        return [], accel, accel_frames

    thresh = np.percentile(accel[valid], ACCEL_PCT_THRESHOLD)
    peaks, props = find_peaks(
        np.where(valid, accel, 0.0),
        height=thresh,
        distance=MIN_TOUCH_SPACING_FRAMES,
        prominence=MIN_PEAK_PROMINENCE,
    )

    touches = []
    for pk in peaks:
        frame = int(accel_frames[pk])
        touches.append({
            "frame": frame,
            "img_x": float(x[pk + 1]),
            "img_y": float(y[pk + 1]),
            "accel": float(accel[pk]),
            "accel_z": float(accel[pk] / (thresh + 1e-6)),
        })
    return touches, accel, accel_frames


# ─── 4. Atribución al jugador más cercano ──────────────────────────────

def attribute_touches(touches, tracks, ball_track_by_frame, ppm):
    """Para cada toque, encuentra la jugadora más cercana en court coords.

    Usa el court_x/court_y del balón en ese frame (es fiable cerca del piso/jugadora;
    en el momento del contacto el balón está pegado a la jugadora, así que la
    proyección sí ubica bien dónde fue el toque).
    """
    # Index de samples por jugadora ordenado por frame
    track_idx = {tid: sorted(ss, key=lambda s: s["frame"]) for tid, ss in tracks.items()}

    out = []
    for t in touches:
        frame = t["frame"]
        ball_pos = _ball_court_at_frame(frame, ball_track_by_frame)
        if ball_pos is None:
            t["track_id"] = None
            t["player_distance_m"] = None
            out.append(t)
            continue
        bx, by = ball_pos
        t["court_x"] = bx
        t["court_y"] = by

        best_tid, best_dist = None, float("inf")
        for tid, samples in track_idx.items():
            pos = _player_pos_at_frame(frame, samples, PLAYER_INTERP_MAX_GAP)
            if pos is None:
                continue
            dx = pos[0] - bx
            dy = pos[1] - by
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_tid = tid

        if best_tid is not None and best_dist <= MAX_ATTRIBUTION_DIST_M:
            t["track_id"] = int(best_tid)
            t["player_distance_m"] = round(best_dist, 2)
        else:
            t["track_id"] = None
            t["player_distance_m"] = round(best_dist, 2) if best_dist != float("inf") else None
        out.append(t)
    return out


def _ball_court_at_frame(frame, ball_by_frame):
    """Devuelve (court_x, court_y) del balón en el frame, o None."""
    d = ball_by_frame.get(frame)
    if d and "court_x" in d:
        return float(d["court_x"]), float(d["court_y"])
    # buscar el más cercano dentro de ±3 frames
    for delta in range(1, 4):
        for f in (frame - delta, frame + delta):
            d = ball_by_frame.get(f)
            if d and "court_x" in d:
                return float(d["court_x"]), float(d["court_y"])
    return None


def _player_pos_at_frame(frame, samples, max_gap):
    """Interpola la posición de la jugadora en el frame entre samples vecinos."""
    if not samples:
        return None
    # binary-search-ish lineal (samples son pocos)
    before, after = None, None
    for s in samples:
        if s["frame"] <= frame:
            before = s
        elif after is None and s["frame"] > frame:
            after = s
            break
    if before is None and after is None:
        return None
    if before is None:
        return (after["court_x"], after["court_y"]) if after["frame"] - frame <= max_gap else None
    if after is None:
        return (before["court_x"], before["court_y"]) if frame - before["frame"] <= max_gap else None
    gap = after["frame"] - before["frame"]
    if gap > max_gap:
        return None
    u = (frame - before["frame"]) / gap
    return (
        before["court_x"] + u * (after["court_x"] - before["court_x"]),
        before["court_y"] + u * (after["court_y"] - before["court_y"]),
    )


# ─── 5. Rallies ─────────────────────────────────────────────────────────

def group_into_rallies(touches, fps, gap_seconds=RALLY_GAP_SECONDS):
    if not touches:
        return []
    gap_frames = int(gap_seconds * fps)
    rallies = []
    current = [touches[0]]
    for t in touches[1:]:
        if t["frame"] - current[-1]["frame"] > gap_frames:
            rallies.append(_finalize_rally(current, len(rallies), fps))
            current = [t]
        else:
            current.append(t)
    rallies.append(_finalize_rally(current, len(rallies), fps))

    # Anotar los touches con rally_id y seq_in_rally
    for r in rallies:
        for i, t in enumerate(r["touches"]):
            t["rally_id"] = r["id"]
            t["seq_in_rally"] = i
    return rallies


def _finalize_rally(touches_in_rally, rid, fps):
    by_player = defaultdict(int)
    for t in touches_in_rally:
        if t["track_id"] is not None:
            by_player[t["track_id"]] += 1
    return {
        "id": rid,
        "start_frame": touches_in_rally[0]["frame"],
        "end_frame": touches_in_rally[-1]["frame"],
        "duration_s": round((touches_in_rally[-1]["frame"] - touches_in_rally[0]["frame"]) / fps, 2),
        "n_touches": len(touches_in_rally),
        "touches_by_player": dict(by_player),
        "touches": touches_in_rally,
    }


# ─── 6. Diagnóstico visual (opcional) ───────────────────────────────────

def save_diagnostic_plot(ball, accel, accel_frames, touches, fps, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠ matplotlib no instalado, saltando plot")
        return

    touch_frames = np.array([t["frame"] for t in touches])
    f0 = ball["f_min"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    ax1.plot(ball["frames"], ball["y"], color="#666", linewidth=0.8)
    ax1.scatter(ball["frames"][ball["present"]], ball["y"][ball["present"]],
                s=2, color="#0aa", alpha=0.4, label="detección real")
    for tf in touch_frames:
        ax1.axvline(tf, color="#d33", linewidth=0.7, alpha=0.7)
    ax1.invert_yaxis()  # imagen: y aumenta hacia abajo
    ax1.set_ylabel("ball y (px, invertido)")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(alpha=0.2)

    ax2.plot(accel_frames, accel, color="#444", linewidth=0.6)
    if len(accel) and not np.isnan(accel).all():
        thr = np.percentile(accel[~np.isnan(accel)], ACCEL_PCT_THRESHOLD)
        ax2.axhline(thr, color="#888", linestyle="--", linewidth=0.7,
                    label=f"thresh p{ACCEL_PCT_THRESHOLD}={thr:.1f}")
    for tf in touch_frames:
        ax2.axvline(tf, color="#d33", linewidth=0.7, alpha=0.7)
    ax2.set_ylabel("|accel| (px/frame²)")
    ax2.set_xlabel("frame")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(alpha=0.2)

    fig.suptitle(f"Touch machine v{VERSION} — {len(touches)} toques detectados",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ─── 7. Orquestación ────────────────────────────────────────────────────

def run(scouting_path, out_path=None, plot=False):
    scouting_path = Path(scouting_path)
    data, ball_track, tracks = load_scouting(scouting_path)

    duration_s = data.get("duration_s", 0)
    n_frames = max((b["frame"] for b in ball_track), default=0) + 1
    fps = n_frames / duration_s if duration_s > 0 else 60.0
    ppm = (data.get("pixels_per_meter")
           or (data.get("cal", {}) if isinstance(data.get("cal"), dict) else {}).get("pixels_per_meter")
           or 40)

    print(f"▸ ball_track:  {len(ball_track)} detecciones")
    print(f"▸ tracks:      {len(tracks)} jugadoras")
    print(f"▸ fps inferido: {fps:.1f}")

    ball = reconstruct_ball(ball_track, MAX_GAP_INTERP_FRAMES, SMOOTH_WINDOW)
    if ball is None:
        print("⚠ no hay datos de balón suficientes")
        return None

    touches, accel, accel_frames = detect_touches(ball, fps)
    print(f"▸ toques candidatos:  {len(touches)}")

    ball_by_frame = {d["frame"]: d for d in ball_track}
    touches = attribute_touches(touches, tracks, ball_by_frame, ppm)
    attributed = sum(1 for t in touches if t["track_id"] is not None)
    print(f"▸ toques atribuidos:  {attributed} / {len(touches)}")

    rallies = group_into_rallies(touches, fps)
    print(f"▸ rallies:  {len(rallies)}")

    data["touch_machine_version"] = VERSION
    data["touches"] = touches
    data["rallies"] = [{k: v for k, v in r.items() if k != "touches"} for r in rallies]

    if out_path is None:
        out_path = scouting_path.with_name(scouting_path.stem + "_enriched.json")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"✓ {out_path}")

    if plot:
        plot_path = Path(out_path).with_suffix(".png")
        save_diagnostic_plot(ball, accel, accel_frames, touches, fps, plot_path)
        print(f"✓ {plot_path}")

    # Resumen por jugadora
    by_player = defaultdict(int)
    for t in touches:
        if t["track_id"] is not None:
            by_player[t["track_id"]] += 1
    if by_player:
        print("\n┌─ Toques por jugadora ────────")
        for tid, n in sorted(by_player.items(), key=lambda kv: -kv[1]):
            print(f"│  track {tid}:  {n} toques")
        print("└──────────────────────────────")

    return data


def main():
    p = argparse.ArgumentParser(description=f"touch_machine v{VERSION}")
    p.add_argument("scouting_json", help="ruta al scouting_data.json de CLARA")
    p.add_argument("--out", help="ruta del JSON enriquecido (default: ..._enriched.json)")
    p.add_argument("--plot", action="store_true", help="diagnóstico visual (PNG)")
    args = p.parse_args()
    run(args.scouting_json, args.out, args.plot)


if __name__ == "__main__":
    main()
