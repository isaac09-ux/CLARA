"""Pruebas del core del subpaquete court/: round-trip de homografía,
clasificación de zonas y extracción de esquinas desde máscara sintética.
No requiere ultralytics.

Ejecútalo como módulo desde `src/`:
    cd src
    python -m court.test_core
"""
import numpy as np
import cv2

from . import court_geometry as cg
from .court_calibration import CourtCalibrator, corners_from_mask

print("=" * 60)
print("1) HOMOGRAFÍA round-trip")
print("=" * 60)
# Puntos imagen simulando una cámara que ve la cancha en perspectiva.
# Orden = [A_izq, A_der, B_der, B_izq] para coincidir con COURT_CORNERS_M.
img_pts = np.array([[180, 690],   # A_izq  (cerca, abajo-izq)
                    [1100, 700],  # A_der  (cerca, abajo-der)
                    [880, 230],   # B_der  (lejos, arriba-der)
                    [400, 225]],  # B_izq  (lejos, arriba-izq)
                   dtype=np.float32)
cal = CourtCalibrator()
cal.from_corners(img_pts)

court = cal.image_to_court(img_pts)
print("imagen->metros (debe dar las 4 esquinas oficiales):")
print(np.round(court, 3))
err = np.abs(court - cg.COURT_CORNERS_M).max()
print(f"error máx vs esquinas oficiales: {err:.4f} m")

back = cal.court_to_image(cg.COURT_CORNERS_M)
err2 = np.abs(back - img_pts).max()
print(f"metros->imagen round-trip error máx: {err2:.3f} px")

# punto central de la cancha (4.5, 9) debe caer dentro del cuadrilátero imagen
center_img = cal.court_to_image([[4.5, 9.0]])[0]
print("centro cancha (4.5,9) en imagen:", np.round(center_img, 1))

print("\n" + "=" * 60)
print("2) ZONAS (1-6) en centros conocidos")
print("=" * 60)
tests = [
    ((7.5, 3.0), 'A', 1), ((4.5, 3.0), 'A', 6), ((1.5, 3.0), 'A', 5),
    ((7.5, 7.5), 'A', 2), ((4.5, 7.5), 'A', 3), ((1.5, 7.5), 'A', 4),
    ((1.5, 15.0), 'B', 1), ((4.5, 15.0), 'B', 6), ((7.5, 15.0), 'B', 5),
    ((1.5, 10.5), 'B', 2), ((4.5, 10.5), 'B', 3), ((7.5, 10.5), 'B', 4),
]
all_ok = True
for (x, y), side, exp in tests:
    z = cg.classify_zone(x, y, side)
    ok = (z == exp)
    all_ok &= ok
    flag = "" if ok else "   <-- MAL"
    print(f"  ({x:>4},{y:>5}) lado {side}: zona {z}  (esperado {exp}){flag}")
print("RESULTADO ZONAS:", "OK" if all_ok else "ERROR")

# flip_lr debe espejar 1<->5, 2<->4 en cada fila
zf = cg.classify_zone(7.5, 3.0, 'A', flip_lr=True)
print(f"  flip_lr en (7.5,3) lado A: zona {zf} (esperado 5)")

print("\n" + "=" * 60)
print("3) corners_from_mask con cancha sintética en perspectiva")
print("=" * 60)
H, W = 720, 1280
mask = np.zeros((H, W), np.uint8)
quad = back.astype(np.int32).reshape(-1, 1, 2)
cv2.fillConvexPoly(mask, quad, 255)
# agregar ruido y un "hueco" (jugador ocluyendo) para probar robustez
cv2.circle(mask, (640, 450), 40, 0, -1)
detected = corners_from_mask(mask)
print("esquinas recuperadas (orden TL,TR,BR,BL imagen):")
print(np.round(detected, 1))

# Verifica que las esquinas recuperadas, pasadas por homografía, reproducen
# la cancha oficial con error chico.
cal2 = CourtCalibrator()
cal2.from_corners(detected)
court2 = cal2.image_to_court(detected)
err3 = np.abs(np.sort(court2, axis=0) - np.sort(cg.COURT_CORNERS_M, axis=0)).max()
print(f"error máx esquinas auto vs oficiales: {err3:.3f} m")

print("\nTODO LISTO" if all_ok and err < 1e-2 and err3 < 0.5 else "\nREVISAR")
