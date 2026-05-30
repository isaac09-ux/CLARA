"""
court_geometry.py
=================
Geometría oficial de cancha de voleibol (FIVB) + utilidades de zonas y render
top-down para CLARA.

SISTEMA DE COORDENADAS CANÓNICO  (todo en METROS)
-------------------------------------------------
    x: 0  (línea lateral izquierda)  ->  9  (línea lateral derecha)
    y: 0  (línea de fondo lado A)    ->  18 (línea de fondo lado B)

    - Red en             y = 9.0
    - Línea de ataque A  y = 6.0   (3 m de la red, lado A)
    - Línea de ataque B  y = 12.0  (3 m de la red, lado B)

Toda la data de scouting se guarda en METROS sobre este sistema (medidas
oficiales). El render top-down es una transformación aparte (metros -> pixeles),
así la data nunca depende de cómo se dibuje.

NOTA DE INTEGRACIÓN (CLARA):
Este módulo usa su PROPIA convención de coordenadas (lado A en y=0). El pipeline
principal `clara.py` / `court_keypoints.py` usa otra convención independiente
(cámara cercana en y=18, zonas como strings "A4"/"B1"). Las dos NO se mezclan
dentro de una misma calibración: cada subsistema es autocontenido. Aquí las
zonas son ENTEROS 1-6 (estándar FIVB).

CONVENCIÓN DE ZONAS (posiciones 1-6 de voleibol)
------------------------------------------------
Render estándar: lado A abajo, lado B arriba, x crece hacia la derecha.

    LADO B (arriba, mira hacia abajo)  --  izq/der ESPEJADAS
        FONDO:                 Z1(izq)  Z6(centro)  Z5(der)
        FRENTE (cerca de red): Z2(izq)  Z3(centro)  Z4(der)
    ----------------------------- RED -----------------------------
        FRENTE (cerca de red): Z4(izq)  Z3(centro)  Z2(der)
        FONDO:                 Z5(izq)  Z6(centro)  Z1(der)
    LADO A (abajo, mira hacia arriba)

`flip_lr=True` invierte izquierda/derecha si tu cámara está espejada respecto
a esta convención. Verifícalo SIEMPRE contra tu primer video.
"""

import numpy as np
import cv2

# --- Dimensiones oficiales FIVB (metros) ---
COURT_WIDTH = 9.0       # eje x
COURT_LENGTH = 18.0     # eje y
NET_Y = 9.0
ATTACK_LINE_A = 6.0     # 3 m de la red, lado A
ATTACK_LINE_B = 12.0    # 3 m de la red, lado B
THIRD = COURT_WIDTH / 3.0   # ancho de cada columna de zona (3 m)

# 4 esquinas en orden [A_izq, A_der, B_der, B_izq]  ==  [TL, TR, BR, BL]
# en el render estándar. Este ORDEN debe coincidir con el orden de los puntos
# imagen que se le pasan a la homografía.
COURT_CORNERS_M = np.array([
    [0.0,         0.0],          # A lateral izquierda - fondo
    [COURT_WIDTH, 0.0],          # A lateral derecha  - fondo
    [COURT_WIDTH, COURT_LENGTH], # B lateral derecha  - fondo
    [0.0,         COURT_LENGTH], # B lateral izquierda - fondo
], dtype=np.float32)

# Puntos de referencia con nombre (para calibración manual flexible).
# Útiles cuando una esquina está ocluida: puedes clicar la red o las líneas
# de ataque, que casi siempre están visibles.
LANDMARKS_M = {
    "A_corner_left":  (0.0,         0.0),
    "A_corner_right": (COURT_WIDTH, 0.0),
    "B_corner_right": (COURT_WIDTH, COURT_LENGTH),
    "B_corner_left":  (0.0,         COURT_LENGTH),
    "net_left":       (0.0,         NET_Y),
    "net_right":      (COURT_WIDTH, NET_Y),
    "attackA_left":   (0.0,         ATTACK_LINE_A),
    "attackA_right":  (COURT_WIDTH, ATTACK_LINE_A),
    "attackB_left":   (0.0,         ATTACK_LINE_B),
    "attackB_right":  (COURT_WIDTH, ATTACK_LINE_B),
}


def in_court(x, y, margin=0.0):
    """True si el punto (metros) cae dentro de la cancha (con margen opcional)."""
    return (-margin <= x <= COURT_WIDTH + margin) and \
           (-margin <= y <= COURT_LENGTH + margin)


def side_of(y):
    """Devuelve 'A' (y<9) o 'B' (y>=9)."""
    return 'A' if y < NET_Y else 'B'


def classify_zone(x, y, side, flip_lr=False):
    """
    Zona/posición de voleibol (1-6) para un punto en metros.

    side : 'A' (mitad y<9, mira hacia la red en +y)
           'B' (mitad y>9, mira hacia la red en -y)
    flip_lr : invierte izquierda/derecha del equipo (cámara espejada).

    Devuelve int 1..6, o 0 si el punto está fuera de rango lateral.
    """
    if x < 0 or x > COURT_WIDTH:
        return 0

    # columna en imagen: 0 = izquierda, 1 = centro, 2 = derecha
    if x < THIRD:
        col_img = 0
    elif x < 2 * THIRD:
        col_img = 1
    else:
        col_img = 2

    if side == 'A':
        is_front = y >= ATTACK_LINE_A   # cerca de la red (y alto)
        col = col_img                   # A: izquierda imagen = su izquierda
    else:  # 'B'
        is_front = y <= ATTACK_LINE_B   # cerca de la red (y bajo)
        col = 2 - col_img               # B: izq/der espejadas

    if flip_lr:
        col = 2 - col

    # col relativo al equipo: 0 = izquierda, 1 = centro, 2 = derecha
    if is_front:
        return {0: 4, 1: 3, 2: 2}[col]
    return {0: 5, 1: 6, 2: 1}[col]


# ---------------------------------------------------------------------------
# Render top-down
# ---------------------------------------------------------------------------
def m2px_render(x, y, scale, margin):
    """Metros -> pixel del canvas top-down (y=0 abajo, lado A abajo)."""
    px = int(round(margin + x * scale))
    py = int(round(margin + (COURT_LENGTH - y) * scale))
    return px, py


def make_topdown_canvas(scale=40, margin=40, bg=(34, 34, 40)):
    """Crea un lienzo en blanco para la cancha top-down. scale = pixeles/metro."""
    w = int(COURT_WIDTH * scale) + 2 * margin
    h = int(COURT_LENGTH * scale) + 2 * margin
    img = np.full((h, w, 3), bg, np.uint8)
    return img


def draw_court(img, scale=40, margin=40, line=(190, 190, 200), thick=2,
               draw_zone_grid=True):
    """Dibuja líneas oficiales (perímetro, red, líneas de ataque, centro de zonas)."""
    def P(x, y):
        return m2px_render(x, y, scale, margin)

    # perímetro
    cv2.rectangle(img, P(0, 0), P(COURT_WIDTH, COURT_LENGTH), line, thick)
    # líneas de ataque
    cv2.line(img, P(0, ATTACK_LINE_A), P(COURT_WIDTH, ATTACK_LINE_A), line, 1)
    cv2.line(img, P(0, ATTACK_LINE_B), P(COURT_WIDTH, ATTACK_LINE_B), line, 1)
    # red (resaltada)
    cv2.line(img, P(0, NET_Y), P(COURT_WIDTH, NET_Y), (90, 200, 255), thick + 1)

    if draw_zone_grid:
        grid = (70, 70, 82)
        for gx in (THIRD, 2 * THIRD):
            cv2.line(img, P(gx, 0), P(gx, COURT_LENGTH), grid, 1)
        # marcar números de zona (centros aproximados) por lado
        font = cv2.FONT_HERSHEY_SIMPLEX
        centers = {
            'A': {1: (7.5, 3.0), 6: (4.5, 3.0), 5: (1.5, 3.0),
                  2: (7.5, 7.5), 3: (4.5, 7.5), 4: (1.5, 7.5)},
            'B': {1: (1.5, 15.0), 6: (4.5, 15.0), 5: (7.5, 15.0),
                  2: (1.5, 10.5), 3: (4.5, 10.5), 4: (7.5, 10.5)},
        }
        for s in ('A', 'B'):
            for z, (zx, zy) in centers[s].items():
                px, py = P(zx, zy)
                cv2.putText(img, str(z), (px - 6, py + 5), font, 0.45,
                            (110, 110, 122), 1, cv2.LINE_AA)
    return img
