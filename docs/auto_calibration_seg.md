# Auto-calibración por segmentación — sin entrenar

> **Estado en CLARA:** `src/court_segmentation.py` ya está integrado. La
> "opción B" (cablearlo en `run()`) **ya viene de fábrica** vía el flag
> `--court-seg-model` de `clara.py` (fallback de `--court-model`, mismo
> `cal.json`). La "opción A" (standalone) sigue disponible para generar el
> JSON aparte. Validado con `src/test_seg.py` (no requiere modelos).

`court_segmentation.py` es un backend de auto-calibración para CLARA que **no
requiere entrenar nada**. Usa el modelo de **segmentación de cancha
pre-entrenado** de `volleyball_analytics` (descargable hoy), saca las 4 esquinas
de la máscara, y construye la homografía. Escupe el **mismo `cal.json`** que
`court_keypoints.py`, con la **misma convención de coordenadas**, así que enchufa
directo a `clara.py` y `zone_for_court_pos` sigue dando las zonas correctas.

Es el "lo mejor de todos los mundos": el modelo pre-entrenado de un repo + la
lógica de máscara→esquinas→homografía validada contra las funciones reales de
CLARA (round-trip con error 0.0, zonas consistentes con tu `zone_for_court_pos`).

## Qué necesitas

```bash
pip install ultralytics opencv-python numpy
```

Pesos de cancha (gratis, fuera de Roboflow): ZIP de `volleyball_analytics`
**https://drive.google.com/file/d/1__zkTmGwZo2z0EgbJvC14I_3kOpgQx3o/view** →
usa `weights/court/weights/best.pt`.

## Uso — opción A: standalone (genera el cal.json aparte)

Corres el backend una vez, te genera el `cal.json`, y se lo pasas a CLARA como
si fuera calibración manual:

```bash
# 1. Auto-calibrar (genera cal_auto.json + cal_check.jpg)
python src/court_segmentation.py partido.mp4 --model court_best.pt \
    --out cal_auto.json --qc cal_check.jpg

# 2. REVISA cal_check.jpg: el contorno verde debe pegar con la cancha real.

# 3. Correr CLARA con esa calibración
python src/clara.py partido.mp4 --calibration cal_auto.json --ball-model clara_balon.pt
```

## Uso — opción B: cableado en run() (ya integrado)

Ya no hay que tocar nada: pásale `--court-seg-model` a `clara.py`. Se usa como
**fallback** de `--court-model` (keypoints) o, si no das keypoints, como backend
principal. Si la segmentación falla, `clara.py` cae al `--calibration` manual.

```bash
# segmentación como backend principal (sin modelo de keypoints):
python src/clara.py partido.mp4 --court-seg-model court_best.pt \
    --calibration cal_manual.json --ball-model clara_balon.pt

# keypoints primero, segmentación de respaldo si keypoints falla:
python src/clara.py partido.mp4 --court-model kpts.pt \
    --court-seg-model court_best.pt --calibration cal_manual.json
```

El overlay de control de calidad se guarda en `out/cal_check.jpg` cuando se usa
la auto-calibración por segmentación dentro de `run()`.

## Convención y orientación

Asume **cámara con el fondo de la cancha ARRIBA en la imagen** (cercana abajo) —
que es tu caso desde la grada. Si alguna cámara tuviera la zona cercana arriba,
usa `--flip-vertical` (en el CLI standalone) o `flip_vertical=True` en
`auto_calibrate_seg`. El `cal_check.jpg` te lo dice de un vistazo: si las zonas
salen volteadas, voltea la bandera.

> Convención idéntica a `court_keypoints.py`:
> cercana (abajo en imagen) → `y=18`, lejana (arriba) → `y=0`, centro → `y=9`,
> `x`: 0 lateral izquierda → 9 lateral derecha. `project()` y
> `zone_for_court_pos()` quedan intactas.

## Caveats honestos (ya probados con fotos de Chihuahua)

1. **Segmentación→esquinas es frágil en gimnasio multiuso / ángulo oblicuo.**
   Cuando el borde de la cancha no se distingue limpio de las líneas de basket,
   las esquinas salen flojas o la auto-cal falla. Por eso el `cal_check.jpg` es
   obligatorio antes de confiar, y `clara.py` cae solo a manual si falla.

2. **Solo da 4 puntos** (esquinas), no los 6 de `court_keypoints` (que incluye
   la línea central). Suficiente para la homografía, pero menos robusto. Cuando
   puedas entrenar el modelo de 6 keypoints, ése será mejor.

3. **Trípode fijo = calibra una vez.** Corre la auto-cal sobre un clip con
   POCOS jugadores en cancha (calentamiento, entre puntos), verifica el overlay,
   guarda el JSON, y reúsalo para todas las grabaciones con ese encuadre. Cero
   clics, cero entrenamiento, y la mejor máscara posible.

## Validación

`python src/test_seg.py` corre sin YOLO: prueba máscara→esquinas→homografía
contra las funciones reales `project()` y `zone_for_court_pos()` copiadas de
`clara.py`. Debe imprimir **"TODO LISTO"** y generar `qc_test.jpg`.
