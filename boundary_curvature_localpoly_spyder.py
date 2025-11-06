#!/usr/bin/env python3
# boundary_curvature_localpoly_spyder.py
# Spyder-friendly: set input_csv to a path, or leave as None to pick a file via dialog.

import os, numpy as np
import matplotlib.pyplot as plt

# If None, a file dialog will open. Otherwise set to e.g. "outputs/mandel_boundary.csv"
# For Mandelbrot:
#input_csv   = "outputs/mandel_boundary.csv"
#output_prefix = "outputs/curv_localpoly/mandel"
#neighbors   = 7     # try 7–11 for ~1500 points
#closed      = True
#stride      = 1

# For Construct:
input_csv   = "outputs/construct_boundary.csv"   # the 1500-point file you just made
output_prefix = "outputs/curv_localpoly/construct"
neighbors   = 7
closed      = True
stride      = 1


def pick_file_dialog():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        path = filedialog.askopenfilename(title="Pick ordered boundary CSV",
                                          filetypes=[("CSV files","*.csv"), ("All files","*.*")])
        root.update(); root.destroy()
        return path
    except Exception:
        return None

def load_points(csv_path):
    try:
        arr = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=float)
        if ('x' in arr.dtype.names) and ('y' in arr.dtype.names):
            return np.c_[arr['x'], arr['y']]
    except Exception:
        pass
    pts = np.genfromtxt(csv_path, delimiter=",", dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(-1, 2)
    if pts.shape[1] != 2:
        raise ValueError("Could not load 2D points from CSV (expect two columns 'x,y').")
    return pts

def local_arclength_parameters(P, idxs):
    XY = P[idxs]
    mid = len(idxs)//2
    s = np.zeros(len(idxs), dtype=float)
    for k in range(mid+1, len(idxs)):
        s[k] = s[k-1] + np.linalg.norm(XY[k] - XY[k-1])
    for k in range(mid-1, -1, -1):
        s[k] = s[k+1] - np.linalg.norm(XY[k+1] - XY[k])
    return s, XY

def quadratic_design(s):
    return np.c_[np.ones_like(s), s, s**2]

def fit_quadratic(s, vals):
    A = quadratic_design(s)
    coefs, *_ = np.linalg.lstsq(A, vals, rcond=None)
    return coefs

def curvature_from_param_quadratic(ax, bx):
    x1 = ax[1]; x2 = 2.0*ax[2]
    y1 = bx[1]; y2 = 2.0*bx[2]
    cross = x1*y2 - y1*x2
    speed = (x1*x1 + y1*y1)**0.5 + 1e-16
    denom = speed**3
    kappa_signed = cross / denom
    kappa = abs(kappa_signed)
    return kappa, kappa_signed, speed, x1, y1, x2, y2

def index_window(i, m, N, closed=True):
    idxs = []
    for d in range(-m, m+1):
        j = i + d
        if closed: idxs.append(j % N)
        else: idxs.append(min(max(j, 0), N-1))
    return idxs

def compute_curvature_localpoly(P, neighbors=7, closed=True, stride=1):
    N = P.shape[0]; m = int(neighbors)
    if m < 2: raise ValueError("neighbors must be >= 2")
    kappa = np.zeros(N); kappa_s = np.zeros(N); speed = np.zeros(N)
    x1_all = np.zeros(N); y1_all = np.zeros(N); x2_all = np.zeros(N); y2_all = np.zeros(N)
    eval_idx = range(0, N, max(1, int(stride)))
    for i in eval_idx:
        idxs = index_window(i, m, N, closed=closed)
        s, XY = local_arclength_parameters(P, idxs)
        ax = fit_quadratic(s, XY[:,0]); bx = fit_quadratic(s, XY[:,1])
        k, ks, sp, x1, y1, x2, y2 = curvature_from_param_quadratic(ax, bx)
        kappa[i] = k; kappa_s[i] = ks; speed[i] = sp
        x1_all[i], y1_all[i], x2_all[i], y2_all[i] = x1, y1, x2, y2
    if stride > 1:
        known = np.array(list(eval_idx))
        for arr in (kappa, kappa_s, speed, x1_all, y1_all, x2_all, y2_all):
            missing = np.setdiff1d(np.arange(N), known)
            arr[missing] = np.interp(missing, known, arr[known])
    aux = dict(xprime=x1_all, yprime=y1_all, x2=x2_all, y2=y2_all)
    return kappa, kappa_s, speed, aux

# Run
if input_csv is None:
    input_csv = pick_file_dialog()
    if not input_csv:
        raise SystemExit("No file selected. Set input_csv to a path and re-run.")

P = load_points(input_csv)
if P.shape[0] < 2*neighbors + 1:
    raise SystemExit(f"Need at least {2*neighbors+1} points; got {P.shape[0]}.")

kappa, kappa_s, speed, aux = compute_curvature_localpoly(P, neighbors=neighbors, closed=closed, stride=stride)

os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
out_csv = f"{output_prefix}_curvature.csv"
idx = np.arange(P.shape[0])
data = np.c_[idx, P[:,0], P[:,1], kappa, kappa_s, speed, aux['xprime'], aux['yprime'], aux['x2'], aux['y2']]
np.savetxt(out_csv, data, delimiter=",", header="idx,x,y,curvature,kappa_signed,speed,xprime,yprime,x2,y2", comments="", fmt="%.10g")

plt.figure(figsize=(6,4))
plt.hist(kappa, bins=64)
plt.xlabel(r"Curvature $\kappa$"); plt.ylabel("Count"); plt.title("Local-Polynomial Curvature Histogram")
plt.tight_layout()
hist_png = f"{output_prefix}_curvature_hist.png"
plt.savefig(hist_png, dpi=200)

plt.figure(figsize=(5,5))
sc = plt.scatter(P[:,0], P[:,1], c=kappa, s=8)
plt.axis('equal'); plt.axis('off')
cbar = plt.colorbar(sc, fraction=0.046, pad=0.04); cbar.set_label(r"$\kappa$")
plt.title("Curvature Overlay (Local-Polynomial)"); plt.tight_layout()
overlay_png = f"{output_prefix}_curvature_overlay.png"
plt.savefig(overlay_png, dpi=220)
plt.show()

with open(f"{output_prefix}_summary.txt", "w") as f:
    import numpy as np
    f.write("Local-Polynomial Curvature Summary\n")
    f.write(f"n={len(kappa)}\nmean={np.mean(kappa):.10g}\nmedian={np.median(kappa):.10g}\nstd={np.std(kappa):.10g}\n")
    f.write(f"q05={np.quantile(kappa,0.05):.10g}\nq95={np.quantile(kappa,0.95):.10g}\nmax={np.max(kappa):.10g}\n")

print("Wrote:", out_csv, hist_png, overlay_png)
