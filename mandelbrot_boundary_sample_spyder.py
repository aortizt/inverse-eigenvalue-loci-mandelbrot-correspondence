#!/usr/bin/env python3
# mandelbrot_boundary_sample_spyder.py
# Spyder-friendly: edit parameters below, then Run (no CLI args needed).

import os, numpy as np
import matplotlib.pyplot as plt

# ==== USER PARAMETERS ====
xlim = (-2.1, 0.9)
ylim = (-1.5, 1.5)
res = 2000
max_iter = 500
level = 0.96      # fraction of max_iter for the isocontour
output_prefix = "outputs/mandel"  # folder will be created if missing
# =========================

def mandelbrot_dwell(x, y, max_iter=300):
    c = x + 1j*y
    z = 0+0j
    for n in range(max_iter):
        z = z*z + c
        if (z.real*z.real + z.imag*z.imag) > 4.0:
            return n
    return max_iter

def compute_grid(xlim, ylim, res, max_iter):
    xs = np.linspace(xlim[0], xlim[1], res)
    ys = np.linspace(ylim[0], ylim[1], res)
    Z = np.zeros((res, res), dtype=float)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            Z[j, i] = mandelbrot_dwell(x, y, max_iter=max_iter)
    return xs, ys, Z

def extract_contour(xs, ys, Z, max_iter, level_frac=0.96):
    target = level_frac * max_iter
    cs = plt.contour(xs, ys, Z, levels=[target])
    seglists = getattr(cs, 'allsegs', [])
    plt.clf()
    if not seglists or not seglists[0]:
        return None
    best = max(seglists[0], key=lambda arr: arr.shape[0])
    return best

# Run
os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
xs, ys, Z = compute_grid(xlim, ylim, res, max_iter)
contour = extract_contour(xs, ys, Z, max_iter, level_frac=level)
if contour is None or contour.shape[0] < 50:
    raise SystemExit("Contour failed. Try level in [0.94, 0.98], higher res or max_iter.")

csv_path = f"{output_prefix}_boundary.csv"
np.savetxt(csv_path, contour, delimiter=",", header="x,y", comments="")

plt.figure(figsize=(6,6))
plt.scatter(contour[:,0], contour[:,1], s=1)
plt.axis('equal'); plt.axis('off'); plt.tight_layout()
png_path = f"{output_prefix}_boundary.png"
plt.savefig(png_path, dpi=220)
plt.show()

with open(f"{output_prefix}_meta.txt", "w") as f:
    f.write(f"xlim={xlim}\nylim={ylim}\nres={res}\nmax_iter={max_iter}\nlevel={level}\n")

print("Wrote:", csv_path, "and", png_path)
