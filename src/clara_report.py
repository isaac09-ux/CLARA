"""
CLARA · Generador de reporte HTML
Toma scouting_data.json y produce un dashboard HTML navegable.

Uso:
    python clara_report.py out_v4/scouting_data.json --topdown out_v4/topdown.png
"""
import json
import argparse
import base64
from pathlib import Path


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

<h2>Top Tracks (10 con más presencia)</h2>
<table>
  <thead>
    <tr>
      <th>ID</th>
      <th>Lado</th>
      <th>Zona dominante</th>
      <th>Muestras</th>
      <th>Distancia (m)</th>
      <th>Velocidad media (m/s)</th>
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

    # Tracks
    tracks_html = []
    for t in data.get("tracks", [])[:10]:
        side_class = f"side-{t.get('side', 'A')}"
        tracks_html.append(f'''
          <tr>
            <td class="id">#{t['id']}</td>
            <td class="{side_class}">{t.get('side', '-')}</td>
            <td>{t.get('dominant_zone') or '-'}</td>
            <td>{t.get('samples', 0)}</td>
            <td>{t.get('distance_m', 0)}</td>
            <td>{t.get('avg_speed_m_per_s') or '-'}</td>
          </tr>
        ''')

    # Ball warning si no hay
    ball_warning = ""
    if data.get("ball_detections_oncourt", 0) == 0:
        ball_warning = '''
        <div class="warning">
          ⚠ Cero detecciones de balón. YOLOv8n base no es confiable para
          balones de voleibol. Cuando entrenes el modelo custom en Roboflow,
          vuelve a procesar con <code>--ball-model balon.pt</code>.
        </div>
        '''

    # Compose
    html = HTML_TEMPLATE.format(
        version=data.get("clara_version", "0.4"),
        video_name=data.get("video", "—"),
        duration_min=data.get("duration_min", 0),
        stride=data.get("stride", 1),
        samples_processed=data.get("samples_processed", 0),
        filtered_tracks=data.get("filtered_tracks", 0),
        raw_tracks=data.get("raw_tracks", 0),
        ball_detections_oncourt=data.get("ball_detections_oncourt", 0),
        ball_model=data.get("ball_model", "—"),
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
