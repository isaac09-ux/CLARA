"""
test_seg.py — Valida court_segmentation.py SIN modelos (sin YOLO).
=================================================================

Prueba la cadena máscara -> esquinas -> homografía del backend de
auto-calibración por segmentación, y verifica que la homografía resultante
es consistente con las funciones REALES de clara.py:

    project()             (pixel -> metros)
    zone_for_court_pos()  (metros -> zona 1-6)

Esas dos funciones se COPIAN aquí verbatim (no se importa clara.py, que
arrastra ultralytics) para validar contra la lógica de producción real.

Corre:  python test_seg.py
Debe imprimir "TODO LISTO" y generar qc_test.jpg.
"""

import numpy as np
import cv2

from court_segmentation import (
    order_points, corners_from_mask, homography_from_corners,
    draw_qc_overlay, CORNER_METERS,
)

PPM = 40
COURT_W, COURT_H = 9, 18


# ============================================================
#  Funciones REALES de clara.py (copiadas verbatim para validar
#  sin importar clara.py / ultralytics).
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


# ============================================================
#  Escenario sintético: cámara con la cancha en perspectiva.
#  Orden [TL, TR, BR, BL] = [lejana-izq, lejana-der, cercana-der, cercana-izq].
# ============================================================
IMG_W, IMG_H = 1280, 720
img_corners = np.array([
    [400, 225],    # TL  lejana-izq  (arriba)
    [880, 230],    # TR  lejana-der  (arriba)
    [1100, 700],   # BR  cercana-der (abajo)
    [180, 690],    # BL  cercana-izq (abajo)
], dtype=np.float32)

ok = True

print("=" * 60)
print("1) HOMOGRAFÍA esquinas -> metros (round-trip vs project())")
print("=" * 60)
H = homography_from_corners(img_corners, ppm=PPM)
assert H is not None, "homography_from_corners devolvió None"

# project() devuelve metros*ppm; /ppm -> metros. Debe dar CORNER_METERS.
got = np.array([project(H, x, y) for x, y in img_corners]) / PPM
print("imagen->metros (debe dar las 4 esquinas):")
print(np.round(got, 3))
err = np.abs(got - CORNER_METERS).max()
print(f"error máx vs esquinas canónicas: {err:.4f} m")
ok &= err < 1e-3

print("\n" + "=" * 60)
print("2) MÁSCARA sintética -> corners_from_mask -> homografía")
print("=" * 60)
mask = np.zeros((IMG_H, IMG_W), np.uint8)
cv2.fillConvexPoly(mask, img_corners.astype(np.int32).reshape(-1, 1, 2), 255)
# oclusión: un jugador tapando una línea (hueco en la máscara)
cv2.circle(mask, (640, 470), 45, 0, -1)
recovered = corners_from_mask(mask)
print("esquinas recuperadas (TL,TR,BR,BL):")
print(np.round(recovered, 1))

H2 = homography_from_corners(recovered, ppm=PPM)
got2 = np.array([project(H2, x, y) for x, y in recovered]) / PPM
err2 = np.abs(np.sort(got2, axis=0) - np.sort(CORNER_METERS, axis=0)).max()
print(f"error máx esquinas auto vs canónicas: {err2:.4f} m")
ok &= err2 < 0.5

print("\n" + "=" * 60)
print("3) ZONAS 1-6: metros -> imagen (Hinv) -> project() -> zone_for_court_pos")
print("=" * 60)
Hinv = np.linalg.inv(np.array(H))
# centros conocidos en metros (convención cercana=18, lejana=0) y su zona
tests = [
    ((7.5, 1.5),  "A1"), ((4.5, 1.5),  "A6"), ((1.5, 1.5),  "A5"),
    ((7.5, 7.5),  "A2"), ((4.5, 7.5),  "A3"), ((1.5, 7.5),  "A4"),
    ((1.5, 16.5), "B1"), ((4.5, 16.5), "B6"), ((7.5, 16.5), "B5"),
    ((1.5, 10.5), "B2"), ((4.5, 10.5), "B3"), ((7.5, 10.5), "B4"),
]
zones_ok = True
for (xm, ym), exp in tests:
    # metros -> imagen px usando Hinv (court_px = metros*ppm)
    img_pt = cv2.perspectiveTransform(
        np.array([[[xm * PPM, ym * PPM]]], np.float32), Hinv)[0][0]
    # imagen -> metros usando la función real project()
    bx, by = project(H, img_pt[0], img_pt[1])
    z = zone_for_court_pos(bx / PPM, by / PPM, COURT_W, COURT_H)
    good = (z == exp)
    zones_ok &= good
    flag = "" if good else "   <-- MAL"
    print(f"  ({xm:>4},{ym:>5}) -> px {np.round(img_pt,0)} -> zona {z} "
          f"(esperado {exp}){flag}")
ok &= zones_ok
print("RESULTADO ZONAS:", "OK" if zones_ok else "ERROR")

print("\n" + "=" * 60)
print("4) flip_vertical invierte cercana<->lejana")
print("=" * 60)
Hf = homography_from_corners(img_corners, ppm=PPM, flip_vertical=True)
gotf = np.array([project(Hf, x, y) for x, y in img_corners]) / PPM
# con flip, la esquina que antes era (0,0) ahora debe ser (0,18)
print("imagen->metros con flip_vertical:")
print(np.round(gotf, 3))
flip_ok = abs(gotf[0][1] - 18) < 1e-3 and abs(got[0][1] - 0) < 1e-3
ok &= flip_ok
print("RESULTADO FLIP:", "OK" if flip_ok else "ERROR")

# ------------------------------------------------------------
# QC overlay sobre un frame sintético
# ------------------------------------------------------------
frame = np.full((IMG_H, IMG_W, 3), 30, np.uint8)
cv2.fillConvexPoly(frame, img_corners.astype(np.int32).reshape(-1, 1, 2),
                   (60, 90, 60))
qc = draw_qc_overlay(frame, H, ppm=PPM, court_size=(COURT_W, COURT_H))
cv2.imwrite("qc_test.jpg", qc)
print("\n[qc] overlay escrito en qc_test.jpg")

print("\nTODO LISTO" if ok else "\nREVISAR")
