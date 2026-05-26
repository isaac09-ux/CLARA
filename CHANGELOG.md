# Changelog

## Unreleased

### Datos para el coach + validacion multi-gimnasio

Se corrio CLARA en 4 clips reales (3 gimnasios, media y cancha completa, 720p y
848x478) para validar que las mejoras de tracking no estuvieran sobre-ajustadas a
un solo video. Hallazgos y cambios:

- **Score `tracks` arreglado (era el ultimo sub-score que premiaba cantidad).**
  La vieja `min(30, len/esperado*30)` daba 30/30 a CUALQUIER conteo >= esperado:
  48 tracks para 6 jugadoras puntuaba 30/30 y la CALIDAD global salia
  91/EXCELENTE -> mentira peligrosa para el coach. Ahora es un ratio SIMETRICO
  `30 * min(n,esp)/max(n,esp)`: premia ACERTAR el numero de jugadoras y penaliza
  sobre-deteccion tanto como sub-deteccion. Efecto medido: WA5 (48/6) 91->65,
  test (8/6) 80->72, 174341 (21/12) 85->72, WA6 (82/12) 91->65. Todos bajan a
  "BUENO" — honesto: ninguno tiene tracking limpio por jugadora.

- **Esquema de track enriquecido para el coach** (`scouting_data.json`):
  `seconds_tracked` (el coach piensa en tiempo, no en "muestras"),
  `zone_profile_pct` (% de tiempo por zona de CADA jugadora — revela la tendencia
  posicional que el histograma global ocultaba; el "72% en A4" agregado era el
  track mas largo dominando, no el equipo) y `reliability` (alta/media/baja por
  nº de muestras, para que el coach sepa de cuales tracks fiarse).

- **`stitch_tracks` 5s/6m: validado, NO universal.** Barrido en metros sobre los
  4 clips (herramienta: el `raw_tracks.json` que se vuelca + un sweep offline,
  sin re-trackear). 5/6 es optimo donde el problema es fragmentacion (test) e
  inofensivo donde no (los demas hacen plateau). NO se cambio: no hay numero
  magico, el optimo depende del clip. Criterio anti-over-merge: el SPREAD
  espacial de cada track — si aflojar sube el spread maximo, se estan uniendo
  jugadoras distintas.

- **Modo de falla dominante = sobre-deteccion, no fragmentacion.** En 3 de 4
  clips (cancha completa o gym apretado) salen MUCHOS mas tracks que jugadoras
  (21/12, 48/6, 82/12): ambos equipos + arbitros + suplentes + publico cercano
  proyectado en cancha, y en 848x478 el ID-churn por baja resolucion. El stitch
  NO puede arreglarlo (no son fragmentos de la misma jugadora). Callejones sin
  salida verificados offline (para que nadie los reintente): filtrar tracks
  casi-inmoviles NO sirve (los extra se mueven) y subir el umbral de confianza NO
  generaliza (conf>=0.6 limpia algo en WA6 pero destruye test, que cae de 8 a 4).
  El arreglo real pediria footage de mayor resolucion, un detector de personas
  especifico de voleibol, o enmascarar zonas de publico — fuera del alcance de un
  parametro. Mientras tanto, el score honesto y `reliability` se lo advierten al
  coach.

### Tracking y métrica de calidad

Contexto: una corrida sobre `data/test.mp4` (media cancha 9×9, 108 s) producía
18 tracks "limpios" para ~6 jugadoras reales, balón ~0% y `estabilidad 1/10`.
La investigación reveló que parte del diagnóstico estaba *roto*, no solo el
tracking. Los cambios y su razón:

- **Métrica `estabilidad` reescrita** (`compute_quality_score`). La vieja era
  `media(duración_de_track) / total_de_muestras`, con 50% para sacar 10/10.
  Eso medía *presencia*, no *continuidad de identidad*, y marcaba ~1/10 hiciera
  lo que hiciera el tracker (jugadoras que entran/salen nunca promedian 50% del
  clip). Peor: **peleaba contra la sub-métrica `tracks`** — más fragmentos subían
  `tracks` (premia cantidad) y hundían `estabilidad` (baja la media). La nueva es
  un **blend 50/50**: `persistencia` (qué fracción del video duran las
  `expected_tracks` más largas, ignorando la cola de fragmentos) + `consolidación`
  (`1 - |#tracks - esperado| / esperado`, penaliza sobre-segmentar Y sub-detectar).
  El breakdown ahora expone ambos componentes. Si tocas esta métrica, recuerda:
  debe *responder* a la fragmentación, no leer constante.

- **`stitch_tracks` aflojado a `max_gap_s=5.0`, `max_jump_m=6.0`** (antes 1.5/2.5,
  pasó por 3.0/4.0). Voleibol tiene oclusiones largas (red, bloqueos) que parten
  una jugadora en varios IDs. Los valores se eligieron **midiendo, no a ojo**: un
  barrido sobre `raw_tracks.json` mostró que 5/6 baja de 14→8 tracks SIN aumentar
  el *spread espacial máximo* (4.08 m, idéntico a 3/4) — o sea fusiona fragmentos
  sin unir jugadoras distintas; a 8/8 ya colapsa a 4 (menos que las 6 reales =
  over-merge). Validado end-to-end: 8 tracks, `estabilidad 6/10`, CALIDAD 70→80.

- **`match_thresh` en `bytetrack_clara.yaml` subido 0.85 → 0.90.** OJO, la
  intuición engaña: en Ultralytics el costo de asociación es `1 - IoU` y un par
  se acepta si `costo <= match_thresh` (ver `trackers/utils/matching.py`), o sea
  **IoU >= 1 - match_thresh**. Por eso *subir* el umbral lo hace MÁS permisivo
  (0.90 acepta hasta IoU≥0.10) y reduce los cambios de ID. *Bajarlo* (p.ej. a 0.80)
  lo haría más estricto (IoU≥0.20) y EMPEORARÍA la fragmentación. No lo bajes
  "para ser más estricto" sin leer el código primero.

- **`track_buffer` subido 90 → 300** (`bytetrack_clara.yaml`). Desde que el tracker
  procesa el video completo (sin saltar frames), 90 frames eran solo ~3 s a 30 fps.
  300 (~10 s) recupera la tolerancia a oclusión que daba el modo viejo.

- **`raw_tracks.json` se vuelca al `--out`** (tracks crudos pre-cosido). Permite
  iterar los parámetros de stitch *offline* importando `stitch_tracks`, sin
  re-trackear (lo caro: ~15 min/corrida). Es la herramienta que se usó para el
  barrido de arriba.

- **Fix de etiqueta**: el campo `"raw_tracks"` del JSON reportaba el conteo
  *post-cosido* (porque `raw_tracks` se reasignaba antes de reportarlo). Ahora
  reporta el crudo real y se añade `"tracks_after_stitch"`.

- **Hallazgo de detección de balón** (no es cambio de código, es para quien corra
  esto): el modo `--ball-detector yolo` base tiene ~0% recall en voleibol y es el
  peor caso. **VballNet es obligatorio para el balón** (`--ball-detector vballnet
  --vballnet-model <.onnx>`): pasó de 0.9% a 12.9% on-court en `test.mp4`. La
  mayoría de los "tracks crudos" descartados (≈78%) son el equipo del fondo
  (inherente a media cancha) + público, no fragmentación.

### Migración del detector a YOLO11

- Detector de personas migrado de YOLOv8 a **YOLO11m** (`src/clara.py`,
  `person_model = YOLO("yolo11m.pt")`). Mejor recall en jugadoras
  chicas/lejanas/borrosas de footage de celular; menos huecos en los tracks.
- El balón en modo `--ball-detector yolo` base ahora sale de la clase COCO 32
  del mismo modelo YOLO11m.
- `clara_train_colab.ipynb`: el detector de balón custom ahora se entrena sobre
  base `yolo11m.pt` (antes `yolov8n.pt`).
- Calibración de cancha (`court_keypoints.py`) y docs actualizados a nomenclatura
  YOLO11. Los modelos entrenados existentes (`clara_balon_v1.pt`, court-pose)
  siguen cargando, pero para migrarlos de verdad hay que reentrenar sobre base v11.

## v0.7.0 — Identificación de jugadora por número de jersey

- NEW: módulo `src/jersey_id.py`. Convierte un track anónimo de ByteTrack en
  una jugadora con nombre. Lee el número del jersey con OCR (easyocr) y cruza
  contra un roster conocido. Run con `--identify --roster roster.json`.
- Diseño: **votación por track, no por frame**. El número solo es legible en
  una fracción de los frames; cada lectura de OCR se acumula como un voto
  ponderado por confianza sobre el track, y el número con más peso gana. Una
  mala lectura aislada queda enterrada por las buenas.
- **Conjunto cerrado**: solo cuentan votos por números presentes en el roster.
  Un OCR que lee un número inexistente se descarta — filtra casi todos los
  errores sin entrenar nada. Consecuencia útil: las rivales (fuera del roster)
  se quedan como tracks anónimos, que es lo deseable para scouting.
- Reusa la pose de `--pose rtmlib` cuando está disponible: hombros (5,6) y
  caderas (11,12) recortan la zona del número con precisión. Sin pose, cae a
  un recorte por proporciones del bbox.
- Preprocesamiento con CLAHE para rescatar números de bajo contraste (kit Casa
  hueso/rosa palo).
- Costo de OCR acotado por `--id-stride` (cada N muestras) y por un tope
  interno de observaciones por track (`max_obs_per_track`).
- JSON: cada `tracks[]` gana un campo `identity` con `{number, name,
  confidence, votes, weight}`; nuevo campo raíz `identified_players`.
- ADITIVO: con `--identify` apagado, CLARA se comporta idéntico a v0.6.2.
- Nuevo CLI: `--identify`, `--roster <path>` (default `roster.json`),
  `--id-stride N` (default 6). Nueva dependencia opcional: `easyocr`.
- Limitación conocida: una rival con un número que coincide con el roster se
  etiqueta mal. Mitigación: filtrar por `side` (A/B) en el reporte.

## v0.6.2 — Filtro de borde inferior (fg_at_edge)

- FIX: `classify_detection` ya no rechaza toda caja cuyo borde inferior toque el frame. El check `y2 >= frame_h - 5` descartaba CUALQUIER detección pegada al borde, asumiendo que era público cortado. En video de baja resolución (p.ej. 848x478) las jugadoras llenan buena parte del alto y muchas tocan el borde en algún frame — el filtro tiró 3699 detecciones válidas en un caso real, dejando 0 tracks y score 0/100. Ahora `fg_at_edge` solo rechaza si la caja toca el borde **y además** es grande (>35% del alto), patrón de un espectador/entrenador en primer plano. Jugadoras lejanas cuyos pies se cortan un poco se conservan; si quedaran fuera de cancha, `is_in_court()` las filtra tras la proyección. Los balones tocando el borde ya no se descartan (antes: `ball_fg_at_edge`).
- Nuevo parámetro `edge_reject_height_ratio` en `classify_detection` (default 0.35).

## v0.6.1 — Filtro de horizonte & VballNet stride

- FIX: `classify_detection` ya no rechaza por banda de píxeles en torno a `court_horizon_y`. La banda fija (±5%/+10% del alto) asumía cámara casi frontal; en ángulos elevados u oblicuos rechazaba jugadoras reales (un día tiró 1911 detecciones y dejó 1 sola jugadora en pie). El filtrado válido se hace post-proyección con `is_in_court()`, que ya estaba activo aguas abajo. `court_horizon_y` sigue aceptado en la calibración para retrocompatibilidad pero es ignorado; se conservan los checks de altura/ancho relativos y el guard de borde inferior.
- NEW: `--vballnet-stride N` (default 1). Alimenta el buffer de 9 frames de VballNet sólo cada N frames del video — reduce ~N× las inferencias y el footprint de RAM del session ONNX que reventaba en clips largos. Recall baja proporcionalmente.

## v0.6 — Multimodal scouting

### New features
- **VballNet ball detector**: TrackNetV4-based motion detection. Run with `--ball-detector vballnet --vballnet-model VballNet.onnx`. ~70% recall in any gym without training. Wrapper: `src/ball_vballnet.py`.
- **rtmlib pose estimation**: RTMPose 17-keypoint extraction per player. Run with `--pose rtmlib`. Adds `torso_lean_deg`, `stance_width_px`, `knee_flexion_deg` per track. Wrapper: `src/pose_rtmlib.py`.
- **Pose sample image**: when `--pose rtmlib`, CLARA renders `pose_sample.png` with skeletons overlaid on a representative frame.
- **JSON output**: now includes `ball_detector`, `pose_mode`, and per-track `pose_stats`.

### CLI changes
- New: `--ball-detector {yolo,vballnet}` (default `yolo`)
- New: `--vballnet-model <path>` (required if `--ball-detector vballnet`)
- New: `--pose {none,rtmlib}` (default `none`)

## v0.5.1 — Bug fixes

- FIX: Quality score rebalanced. Ball detection rate of 10% no longer gives full 20/20. Now requires 50%+ for max. Copa test: 14.4% balls → 78/100 (was 94/100).
- FIX: Stability score formula corrected. 50%+ presence for max.
- FIX: Foreground filter now applied to ball detections too.
- FIX: `out.mkdir(parents=True)` for nested paths.
- FIX: Ball loop uses `cap.set(POS_FRAMES)` to skip frames efficiently.
- FIX: `zone_for_court_pos` and `is_in_court` margin consistency.

## v0.5

- Half-court mode for partial court views
- Foreground filter for spectators
- Quality score 0-100 with breakdown
- Diagnostic image showing rejected detections

## v0.4

- HTML report generator
- Topdown with zones A1-A6/B1-B6 marked
- Custom ball model support via `--ball-model`

## v0.3

- Python pipeline: YOLOv8 + ByteTrack + homography + zone analytics

## v0.1-0.2

- HTML/JS prototype
