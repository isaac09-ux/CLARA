"""
test_motion_cal.py — Valida court_motion_calibration.py SIN modelos (sin YOLO).
==============================================================================

Prueba la cadena puntos-de-pie -> envolvente -> esquinas -> homografía del
backend de auto-calibración por MOVIMIENTO, y verifica que es consistente con
las funciones REALES de clara.py (project, zone_for_court_pos), igual que
test_seg.py. También valida la detección automática media (9×9) vs completa
(9×18) por aspecto métrico, y la orientación (flip_vertical).

Para que la detección de tamaño tenga sentido físico, los escenarios sintéticos
se renderizan con una CÁMARA pinhole real (cv2.projectPoints) sobre rectángulos
de 9×18 y 9×9 m. Las funciones collect_* del módulo hacen import perezoso de
YOLO/VballNet, así que importar el módulo sólo necesita numpy + opencv.

Corre:  python test_motion_cal.py    ->  debe imprimir "TODO LISTO".
"""

import numpy as np
import cv2

from court_motion_calibration import (
    build_homography, motion_envelope_mask, corners_from_mask,
    detect_court_size, detect_orientation, rectangle_aspect, order_points,
)

PPM = 40
IMG_W, IMG_H = 1280, 720


# Funciones REALES de clara.py (copiadas verbatim, como en test_seg.py).
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
#  Cámara pinhole sintética: detrás de la línea cercana, elevada.
# ============================================================
def cam_from(C, T, f=900):
    """Cámara pinhole mirando de C a T (up=+z). Devuelve (K, rvec, tvec)."""
    K = np.array([[f, 0, IMG_W / 2], [0, f, IMG_H / 2], [0, 0, 1]], np.float64)
    C = np.array(C, np.float64); T = np.array(T, np.float64)
    fwd = T - C; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])                       # world -> cámara
    return K, cv2.Rodrigues(R)[0], (-R @ C).reshape(3, 1)


def synth_camera(court_h):
    """Cámara CENTRADA sobre el eje de la cancha (caso fronto-paralelo en ancho)."""
    return cam_from([4.5, -court_h * 0.45, court_h * 0.42],
                    [4.5, court_h * 0.5, 0.0])


def to_img(pts3, cam):
    K, rvec, tvec = cam
    img, _ = cv2.projectPoints(np.asarray(pts3, np.float64).reshape(-1, 1, 3),
                               rvec, tvec, K, None)
    return img.reshape(-1, 2).astype(np.float32)


def court_quad(cw, ch, cam):
    """4 esquinas de cancha cw×ch proyectadas y ordenadas [TL,TR,BR,BL]."""
    world = [[0, 0, 0], [cw, 0, 0], [cw, ch, 0], [0, ch, 0]]
    return order_points(to_img(world, cam))


def scatter(cw, ch, cam, n, seed=0):
    """n puntos de pie (z=0) uniformes en la cancha, en espacio imagen."""
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0.3, cw - 0.3, n)
    ys = rng.uniform(0.3, ch - 0.3, n)
    world = np.column_stack([xs, ys, np.zeros(n)])
    return to_img(world, cam)


ok = True

print("=" * 60)
print("1) build_homography round-trip (full 9x18 y half 9x9)")
print("=" * 60)
ref_quad = np.array([[400, 225], [880, 230], [1100, 700], [180, 690]], np.float32)
for cw, ch in [(9, 18), (9, 9)]:
    H = build_homography(ref_quad, cw, ch, PPM)
    assert H is not None
    got = np.array([project(H, x, y) for x, y in ref_quad]) / PPM
    canon = np.array([[0, 0], [cw, 0], [cw, ch], [0, ch]], np.float32)
    err = np.abs(got - canon).max()
    print(f"  {cw}x{ch}: error máx round-trip = {err:.5f} m")
    ok &= err < 1e-3

print("\n" + "=" * 60)
print("2) MOVIMIENTO completo: pies -> envolvente -> esquinas -> zonas")
print("=" * 60)
cam_full = synth_camera(18)
pts_full = scatter(9, 18, cam_full, 1800, seed=1)
in_frame = ((pts_full[:, 0] >= 0) & (pts_full[:, 0] < IMG_W) &
            (pts_full[:, 1] >= 0) & (pts_full[:, 1] < IMG_H))
pts_full = pts_full[in_frame]
mask = motion_envelope_mask(pts_full, (IMG_H, IMG_W), percentile=10, sigma=16)
recovered = corners_from_mask(mask)
assert recovered is not None, "corners_from_mask devolvió None"
print("esquinas recuperadas (TL,TR,BR,BL):")
print(np.round(recovered, 1))
Hrec = build_homography(recovered, 9, 18, PPM)
assert Hrec is not None

# zonas: metros -> imagen (Hinv) -> project() real -> zone_for_court_pos
Hinv = np.linalg.inv(np.array(Hrec))
tests = [
    ((7.5, 1.5),  "A1"), ((4.5, 1.5),  "A6"), ((1.5, 1.5),  "A5"),
    ((7.5, 7.5),  "A2"), ((4.5, 7.5),  "A3"), ((1.5, 7.5),  "A4"),
    ((1.5, 16.5), "B1"), ((4.5, 16.5), "B6"), ((7.5, 16.5), "B5"),
    ((1.5, 10.5), "B2"), ((4.5, 10.5), "B3"), ((7.5, 10.5), "B4"),
]
zones_ok = True
for (xm, ym), exp in tests:
    ip = cv2.perspectiveTransform(
        np.array([[[xm * PPM, ym * PPM]]], np.float32), Hinv)[0][0]
    bx, by = project(Hrec, ip[0], ip[1])
    z = zone_for_court_pos(bx / PPM, by / PPM, 9, 18)
    good = (z == exp)
    zones_ok &= good
    print(f"  ({xm:>4},{ym:>5}) -> zona {z} (esperado {exp})"
          f"{'' if good else '   <-- MAL'}")
ok &= zones_ok
print("RESULTADO ZONAS:", "OK" if zones_ok else "ERROR")

print("\n" + "=" * 60)
print("3) Auto-detección de tamaño por aspecto métrico (cámara OBLICUA)")
print("=" * 60)
# Cámara OBLICUA (fuera del eje) -> ambos puntos de fuga finitos -> el aspecto
# métrico es exacto. Se pasan las esquinas en orden cíclico conocido (world)
# porque order_points es frágil en quads muy oblicuos: aquí validamos la
# matemática del aspecto/tamaño, no el etiquetado de esquinas.
def oblique_quad(cw, ch, C, T):
    return to_img([[0, 0, 0], [cw, 0, 0], [cw, ch, 0], [0, ch, 0]],
                  cam_from(C, T))

q_full = oblique_quad(9, 18, (-3, -3, 5), (4.5, 9, 0))
q_half = oblique_quad(9, 9, (-3, -2, 3.5), (4.5, 4.5, 0))
a_full, rel_full = rectangle_aspect(q_full, (IMG_H, IMG_W))
a_half, rel_half = rectangle_aspect(q_half, (IMG_H, IMG_W))
print(f"  aspecto full(9x18) = {a_full:.2f} reliable={rel_full} (esperado ~2, True)")
print(f"  aspecto half(9x9)  = {a_half:.2f} reliable={rel_half} (esperado ~1, True)")
cw1, ch1, half1 = detect_court_size(q_full, (IMG_H, IMG_W), "auto")
cw2, ch2, half2 = detect_court_size(q_half, (IMG_H, IMG_W), "auto")
print(f"  full -> {cw1}x{ch1} half={half1}")
print(f"  half -> {cw2}x{ch2} half={half2}")
# overrides explícitos
ovf = detect_court_size(q_half, (IMG_H, IMG_W), "full", verbose=False)
ovh = detect_court_size(q_full, (IMG_H, IMG_W), "half", verbose=False)
# cámara CENTRADA (eje de cancha) -> dir. ancho fronto-paralela -> no confiable,
# y detect_court_size debe caer a FULL avisando.
q_center = court_quad(9, 18, cam_full)        # cam_full es centrada (test 2)
_, rel_center = rectangle_aspect(q_center, (IMG_H, IMG_W))
cwc, chc, halfc = detect_court_size(q_center, (IMG_H, IMG_W), "auto")
print(f"  centrada reliable={rel_center} -> {cwc}x{chc} half={halfc} (esperado "
      f"False, 9x18)")
size_ok = (abs(a_full - 2) < 0.1 and rel_full and ch1 == 18.0 and half1 is False
           and abs(a_half - 1) < 0.1 and rel_half and ch2 == 9.0 and half2 is True
           and ovf == (9.0, 18.0, False) and ovh == (9.0, 9.0, True)
           and rel_center is False and chc == 18.0 and halfc is False)
ok &= size_ok
print("RESULTADO TAMAÑO:", "OK" if size_ok else "ERROR")

print("\n" + "=" * 60)
print("4) Orientación: jugadoras grandes ABAJO => sin flip; ARRIBA => flip")
print("=" * 60)
pts4 = np.array([[640, 600], [640, 620], [640, 120], [640, 140]], np.float32)
flip_n = detect_orientation(pts4, np.array([90, 95, 35, 40], np.float32),
                            (IMG_H, IMG_W))          # grandes abajo
flip_i = detect_orientation(pts4, np.array([35, 40, 90, 95], np.float32),
                            (IMG_H, IMG_W))          # grandes arriba
print(f"  grandes abajo -> flip={flip_n} (esperado False)")
print(f"  grandes arriba -> flip={flip_i} (esperado True)")
orient_ok = (flip_n is False and flip_i is True)
ok &= orient_ok
print("RESULTADO ORIENTACIÓN:", "OK" if orient_ok else "ERROR")

print("\nTODO LISTO" if ok else "\nREVISAR")
