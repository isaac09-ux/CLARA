"""
CLARA · Generador de reporte HTML
Toma scouting_data.json y produce un dashboard HTML navegable.

Uso:
    python clara_report.py out_v4/scouting_data.json --topdown out_v4/topdown.png
"""
import json
import argparse
import base64
import sys
from pathlib import Path

# Consolas Windows (cp1252) revientan al imprimir ✓ con UnicodeEncodeError
# aunque el HTML ya se haya escrito. Forzar utf-8 evita el falso "fallo".
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>CLARA · Scouting Report — {video_name}</title>
<style>
  :root {{
    --bg: #0a0a0a;
    --bg-2: #121212;
    --bone: #e8e0d0;
    --bone-dim: #a8a294;
    --bone-faint: #544f44;
    --oxblood: #a52828;
    --line: #2a2824;
    --ok: #7a9c5a;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg);
    color: var(--bone);
    font-family: 'SF Mono', 'Monaco', 'Inconsolata', monospace;
    line-height: 1.5;
    padding: 24px;
    max-width: 1200px;
    margin: 0 auto;
  }}
  header {{
    border-bottom: 1px solid var(--line);
    padding-bottom: 20px;
    margin-bottom: 28px;
  }}
  .brand {{
    font-size: 11px;
    color: var(--bone-faint);
    letter-spacing: 0.25em;
    text-transform: uppercase;
    margin-bottom: 6px;
  }}
  h1 {{
    font-size: 22px;
    font-weight: 600;
    letter-spacing: -0.01em;
  }}
  h1 .v {{
    color: var(--oxblood);
    font-weight: 400;
  }}
  .meta {{
    color: var(--bone-dim);
    font-size: 12px;
    margin-top: 8px;
    letter-spacing: 0.05em;
  }}
  h2 {{
    font-size: 14px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--bone-dim);
    margin: 32px 0 14px;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--line);
  }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px;
    margin-bottom: 20px;
  }}
  .card {{
    background: var(--bg-2);
    border: 1px solid var(--line);
    padding: 14px 16px;
  }}
  .card .label {{
    font-size: 10px;
    letter-spacing: 0.2em;
    color: var(--bone-faint);
    text-transform: uppercase;
    margin-bottom: 6px;
  }}
  .card .value {{
    font-size: 22px;
    font-weight: 500;
    color: var(--bone);
  }}
  .card .sub {{
    font-size: 11px;
    color: var(--bone-dim);
    margin-top: 2px;
  }}
  .zone-bars {{
    background: var(--bg-2);
    border: 1px solid var(--line);
    padding: 16px;
  }}
  .zone-row {{
    display: grid;
    grid-template-columns: 40px 1fr 50px;
    align-items: center;
    margin-bottom: 4px;
    font-size: 12px;
  }}
  .zone-name {{ color: var(--bone-dim); }}
  .zone-bar {{
    background: var(--bg);
    height: 14px;
    position: relative;
    overflow: hidden;
  }}
  .zone-bar-fill {{
    background: var(--oxblood);
    height: 100%;
    transition: width 0.3s;
  }}
  .zone-bar-fill.dim {{ background: var(--bone-faint); }}
  .zone-count {{ text-align: right; color: var(--bone); }}
  .topdown-wrap {{
    background: var(--bg-2);
    border: 1px solid var(--line);
    padding: 20px;
    display: flex;
    justify-content: center;
  }}
  .topdown-wrap img {{ max-width: 100%; height: auto; }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
  }}
  th, td {{
    text-align: left;
    padding: 8px 10px;
    border-bottom: 1px solid var(--line);
  }}
  th {{
    color: var(--bone-faint);
    font-weight: 400;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    font-size: 10px;
  }}
  td.id {{ color: var(--oxblood); font-weight: 500; }}
  .side-A {{ color: var(--ok); }}
  .side-B {{ color: var(--bone); }}
  .two-col {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
  }}
  footer {{
    margin-top: 40px;
    padding-top: 20px;
    border-top: 1px solid var(--line);
    color: var(--bone-faint);
    font-size: 10px;
    letter-spacing: 0.1em;
    text-align: center;
  }}
  .warning {{
    background: rgba(165, 40, 40, 0.1);
    border-left: 3px solid var(--oxblood);
    padding: 12px 16px;
    color: var(--bone-dim);
    font-size: 12px;
    margin: 14px 0;
  }}
  .quality {{
    display: flex; align-items: center; gap: 22px;
    background: var(--bg-2); border: 1px solid var(--line);
    border-left: 5px solid var(--bone-faint);
    padding: 18px 24px; margin: 6px 0 4px;
  }}
  .quality.excelente {{ border-left-color: var(--ok); }}
  .quality.bueno {{ border-left-color: #b8a34a; }}
  .quality.regular {{ border-left-color: var(--oxblood); }}
  .quality.bajo {{ border-left-color: var(--oxblood); }}
  .q-score {{ font-size: 46px; font-weight: 600; color: var(--bone); line-height: 1; white-space: nowrap; }}
  .q-score span {{ font-size: 18px; color: var(--bone-faint); }}
  .q-label {{ font-size: 12px; letter-spacing: 0.2em; text-transform: uppercase; color: var(--bone-dim); margin-bottom: 5px; }}
  .q-interp {{ font-size: 13px; color: var(--bone); }}
  td.rel-alta {{ color: var(--ok); }}
  td.rel-media {{ color: var(--bone); }}
  td.rel-baja {{ color: var(--bone-faint); }}
  td.zprofile {{ color: var(--bone-dim); font-size: 11px; }}
</style>
</head>
<body>
<header>
  <div class="brand">CLARA · TENTACULO DE LUCIA</div>
  <h1>Scouting Report <span class="v">v{version}</span></h1>
  <div class="meta">
    {video_name} · {duration_min} min · procesado con stride {stride}
  </div>
</header>

<div class="quality {quality_tier}">
  <div class="q-score">{quality_score}<span>/100</span></div>
  <div class="q-body">
    <div class="q-label">Calidad del scouting · {quality_label}</div>
    <div class="q-interp">{quality_interp}</div>
  </div>
</div>
{quality_caveat}

<h2>Resumen</h2>
<div class="grid">
  <div class="card">
    <div class="label">Duración</div>
    <div class="value">{duration_min} min</div>
    <div class="sub">{samples_processed} muestras analizadas</div>
  </div>
  <div class="card">
    <div class="label">Tracks identificados</div>
    <div class="value">{filtered_tracks}</div>
    <div class="sub">de {raw_tracks} detecciones crudas</div>
  </div>
  <div class="card">
    <div class="label">Balones detectados</div>
    <div class="value">{ball_detections_oncourt}</div>
    <div class="sub">modelo: {ball_model}</div>
  </div>
  <div class="card">
    <div class="label">Cancha</div>
    <div class="value">{court_w} × {court_h} m</div>
    <div class="sub">homografía activa</div>
  </div>
</div>

{ball_warning}

<h2>Ritmo de juego</h2>
{play_html}

<h2>Vista Cenital</h2>
<div class="topdown-wrap">
  {topdown_html}
</div>

<h2>Distribución por Zonas</h2>
<div class="two-col">
  <div>
    <div class="zone-bars">
      <div class="zone-row" style="margin-bottom: 10px;">
        <div></div>
        <div style="color: var(--bone-faint); font-size: 10px; letter-spacing: 0.15em;">LADO A</div>
        <div></div>
      </div>
      {zones_a_html}
    </div>
  </div>
  <div>
    <div class="zone-bars">
      <div class="zone-row" style="margin-bottom: 10px;">
        <div></div>
        <div style="color: var(--bone-faint); font-size: 10px; letter-spacing: 0.15em;">LADO B</div>
        <div></div>
      </div>
      {zones_b_html}
    </div>
  </div>
</div>

<h2>Jugadoras seguidas (top 10 por tiempo en cancha)</h2>
<table>
  <thead>
    <tr>
      <th>Jugadora</th>
      <th>Fiabilidad</th>
      <th>Tiempo en cancha</th>
      <th>Zona principal</th>
      <th>Reparto de zonas</th>
      <th>Distancia (m)</th>
      <th>Velocidad (m/s)</th>
    </tr>
  </thead>
  <tbody>
    {tracks_html}
  </tbody>
</table>

<h2>Comparación por mitades</h2>
<div class="two-col">
  <div class="zone-bars">
    <div style="color: var(--bone-faint); font-size: 10px; letter-spacing: 0.15em; margin-bottom: 10px;">PRIMERA MITAD</div>
    {first_half_html}
  </div>
  <div class="zone-bars">
    <div style="color: var(--bone-faint); font-size: 10px; letter-spacing: 0.15em; margin-bottom: 10px;">SEGUNDA MITAD</div>
    {second_half_html}
  </div>
</div>

<footer>
  Las Chispas · CLARA es el tentáculo de visión por computadora de LUCIA
</footer>
</body>
</html>
"""


def render_zone_bars(zones, side_filter=None, max_val=None):
    """Renderiza barras horizontales por zona."""
    if side_filter:
        zones = {k: v for k, v in zones.items() if k.startswith(side_filter)}
    if not zones:
        return '<div style="color: var(--bone-faint); font-size: 11px;">Sin datos</div>'

    if max_val is None:
        max_val = max(zones.values()) if zones else 1

    # Ordenar zonas oficiales: 4-3-2 arriba, 5-6-1 abajo
    order = (["A4", "A3", "A2", "A5", "A6", "A1"] if side_filter == "A"
             else ["B4", "B3", "B2", "B5", "B6", "B1"] if side_filter == "B"
             else sorted(zones.keys()))

    html = []
    for z in order:
        if z not in zones:
            continue
        v = zones[z]
        pct = v / max_val * 100
        html.append(f'''
            <div class="zone-row">
              <div class="zone-name">{z}</div>
              <div class="zone-bar"><div class="zone-bar-fill" style="width: {pct}%"></div></div>
              <div class="zone-count">{v}</div>
            </div>
        ''')
    return "\n".join(html)


def fmt_secs(s):
    """Segundos -> 'Xm YYs' o 'Xs'. El coach piensa en tiempo, no en muestras."""
    s = int(round(s or 0))
    return f"{s // 60}m {s % 60:02d}s" if s >= 60 else f"{s}s"


def quality_verdict(score):
    """Score 0-100 -> (tier_css, etiqueta, interpretacion en lenguaje de coach)."""
    if score >= 80:
        return ("excelente", "Excelente",
                "Datos confiables — úsalos con confianza, incluso por jugadora.")
    if score >= 60:
        return ("bueno", "Bueno",
                "Datos útiles con reservas: el detalle por jugadora es aproximado; "
                "el heatmap de zonas y el balón son sólidos.")
    if score >= 40:
        return ("regular", "Regular",
                "Sirve para leer zonas del equipo, no jugadoras individuales.")
    return ("bajo", "Bajo",
            "Re-grabar o re-calibrar — no bases decisiones en estos datos.")


def embed_image_as_base64(path):
    """Embed image inline en el HTML."""
    if not Path(path).exists():
        return ""
    data = Path(path).read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f'<img src="data:image/png;base64,{b64}" alt="topdown">'


def generate_report(json_path, topdown_path=None, output_path=None):
    data = json.loads(Path(json_path).read_text())

    if output_path is None:
        output_path = Path(json_path).parent / "scouting_report.html"

    # Topdown embedded
    topdown_html = ""
    if topdown_path and Path(topdown_path).exists():
        topdown_html = embed_image_as_base64(topdown_path)
    else:
        topdown_html = '<div style="color: var(--bone-faint);">Topdown no disponible</div>'

    # Zonas
    zone_total = data.get("zone_visits_total", {})
    max_zone = max(zone_total.values()) if zone_total else 1

    zones_a_html = render_zone_bars(zone_total, side_filter="A", max_val=max_zone)
    zones_b_html = render_zone_bars(zone_total, side_filter="B", max_val=max_zone)

    # Mitades
    first_zones = data.get("zone_visits_first_half", {})
    second_zones = data.get("zone_visits_second_half", {})
    max_half = max(list(first_zones.values()) + list(second_zones.values()) + [1])
    first_html = render_zone_bars(first_zones, max_val=max_half)
    second_html = render_zone_bars(second_zones, max_val=max_half)

    # Tracks — lenguaje de coach: nombre (si hay identidad) o #track, fiabilidad,
    # tiempo en cancha, zona principal + reparto, distancia, velocidad.
    tracks_html = []
    for t in data.get("tracks", [])[:10]:
        ident = t.get("identity")
        name = ident.get("name") if isinstance(ident, dict) else None
        player = name or f"#{t['id']}"
        rel = t.get("reliability", "-")
        prof = t.get("zone_profile_pct", {}) or {}
        prof_str = " · ".join(f"{z} {p}%" for z, p in list(prof.items())[:3]) or "—"
        tracks_html.append(f'''
          <tr>
            <td class="id">{player}</td>
            <td class="rel-{rel}">{rel}</td>
            <td>{fmt_secs(t.get('seconds_tracked'))}</td>
            <td>{t.get('dominant_zone') or '—'}</td>
            <td class="zprofile">{prof_str}</td>
            <td>{t.get('distance_m', 0)}</td>
            <td>{t.get('avg_speed_m_per_s') or '—'}</td>
          </tr>
        ''')

    # Ritmo de juego: rallies / saques (Sprint 2, heuristica sobre el balon)
    ps = data.get("play_summary") or {}
    if ps.get("n_rallies", 0) > 0:
        sides = ps.get("serves_by_side", {})
        sides_txt = (" · ".join(f"lado {k}: {v}" for k, v in sorted(sides.items()))
                     if len(sides) > 1 else "≈ uno por rally")
        play_html = f'''<div class="grid">
      <div class="card"><div class="label">Rallies</div>
        <div class="value">{ps['n_rallies']}</div>
        <div class="sub">periodos de juego continuo</div></div>
      <div class="card"><div class="label">Saques</div>
        <div class="value">~{ps['n_serves']}</div>
        <div class="sub">{sides_txt}</div></div>
      <div class="card"><div class="label">Rally promedio</div>
        <div class="value">{ps['avg_rally_s']}s</div>
        <div class="sub">más largo: {ps['longest_rally_s']}s</div></div>
      <div class="card"><div class="label">Tiempo de juego</div>
        <div class="value">{ps['total_play_s']}s</div>
        <div class="sub">balón en movimiento</div></div>
    </div>'''
        # Los rallies salen del balon: con deteccion baja se subestiman duraciones
        # y se pierden rallies. Avisar para que el coach no lea de mas.
        if data.get("ball_detection_rate", 0) < 0.30:
            play_html += ('<div class="warning">⚠ Detección de balón baja '
                          f'({data.get("ball_detection_rate", 0)*100:.0f}%): los rallies '
                          'y duraciones son un <b>piso aproximado</b> (se cuentan del '
                          'primer al último balón visto). Con mejor detección de balón '
                          'estos números suben en fiabilidad.</div>')
    else:
        play_html = ('<div class="warning">Sin rallies detectados. Requiere '
                     'detección de balón — corre con <code>--ball-detector '
                     'vballnet</code>.</div>')

    # Calidad: veredicto en lenguaje de coach
    score = data.get("quality_score", 0)
    quality_tier, quality_label, quality_interp = quality_verdict(score)

    # Caveat de sobre-deteccion: si hay muchos mas tracks que jugadoras esperadas,
    # advertir al coach que el dato por jugadora es aproximado (sin matar el resto).
    half = data.get("half_court")
    expected = 6 if half else 12
    n_tr = data.get("filtered_tracks", 0)
    quality_caveat = ""
    if n_tr > expected * 1.5:
        cancha_txt = "media cancha" if half else "cancha completa"
        quality_caveat = f'''
        <div class="warning">
          ⚠ CLARA siguió <b>{n_tr}</b> tracks, pero en {cancha_txt} se esperan ~{expected}
          jugadoras. Probablemente incluye banca, árbitros, público cercano o ambos
          equipos. <b>Trata los datos por jugadora como aproximados</b> — el heatmap
          de zonas y el balón siguen siendo útiles.
        </div>
        '''

    # Ball warning: cero o baja deteccion -> recomendar VballNet (no Roboflow)
    ball_warning = ""
    ball_rate = data.get("ball_detection_rate", 0)
    if data.get("ball_detections_oncourt", 0) == 0:
        ball_warning = '''
        <div class="warning">
          ⚠ Cero detecciones de balón. El detector YOLO base no sirve para voleibol.
          Re-procesa con <code>--ball-detector vballnet --vballnet-model &lt;modelo.onnx&gt;</code>
          (detección por movimiento, sin entrenar).
        </div>
        '''
    elif ball_rate < 0.15:
        ball_warning = f'''
        <div class="warning">
          ⚠ Detección de balón baja ({ball_rate*100:.0f}%). El balón proyecta fuera de
          cancha cuando va por el aire, así que el conteo es orientativo, no exhaustivo.
        </div>
        '''

    # Compose
    html = HTML_TEMPLATE.format(
        version=data.get("clara_version", "0.4"),
        quality_tier=quality_tier,
        quality_score=score,
        quality_label=quality_label,
        quality_interp=quality_interp,
        quality_caveat=quality_caveat,
        play_html=play_html,
        video_name=data.get("video", "—"),
        duration_min=data.get("duration_min", 0),
        stride=data.get("stride", 1),
        samples_processed=data.get("samples_processed", 0),
        filtered_tracks=data.get("filtered_tracks", 0),
        raw_tracks=data.get("raw_tracks", 0),
        ball_detections_oncourt=data.get("ball_detections_oncourt", 0),
        ball_model=data.get("ball_detector", data.get("ball_model", "—")),
        court_w=data.get("court_size_m", [0, 0])[0],
        court_h=data.get("court_size_m", [0, 0])[1],
        topdown_html=topdown_html,
        zones_a_html=zones_a_html,
        zones_b_html=zones_b_html,
        first_half_html=first_html,
        second_half_html=second_html,
        tracks_html="\n".join(tracks_html),
        ball_warning=ball_warning,
    )

    Path(output_path).write_text(html, encoding="utf-8")
    print(f"[✓] Reporte HTML generado: {output_path}")
    return output_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("json_path")
    p.add_argument("--topdown", default=None,
                   help="Ruta al topdown.png para embeber")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    generate_report(a.json_path, a.topdown, a.out)
