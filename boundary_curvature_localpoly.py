#!/usr/bin/env python3
"""
boundary_curvature_localpoly.py

Local-polynomial curvature estimator for 2D boundaries (Construct & Mandelbrot).

Implements the curvature exactly as described in the paper’s prose:
- Take an ordered sampling of a closed boundary curve Γ ⊂ R^2.
- For each point i, build a local arclength coordinate s=0 at i using its ±m neighbors.
- Fit quadratic polynomials x(s) ≈ a0 + a1 s + a2 s^2 and y(s) ≈ b0 + b1 s + b2 s^2 by least squares.
- Evaluate curvature at s=0 via κ = |x'(0) y''(0) − y'(0) x''(0)| / (x'(0)^2 + y'(0)^2)^(3/2),
  where x'(0)=a1, x''(0)=2 a2, y'(0)=b1, y''(0)=2 b2.

Notes
-----
* INPUTS are assumed **ordered along the boundary** (counterclockwise or clockwise).
  If your CSVs are not ordered, use a separate ordering step first.
* The estimator is purely local and rotationally invariant (since we fit in 2D).
* Handles both open and closed curves; default assumes closed boundaries (wraps neighbors).

Outputs
-------
- CSV with columns: idx, x, y, curvature, kappa_signed (optional), speed, xprime, yprime, x2, y2
- PNG histogram of κ and optional color-overlay scatter of κ along the curve
- Summary TXT with basic stats

Usage
-----
python boundary_curvature_localpoly.py \
  --input_csv out_clean/mandel_boundary_sample.csv \
  --output_prefix outputs/mandel_curv_localpoly \
  --neighbors 7 --closed True

python boundary_curvature_localpoly.py \
  --input_csv out_clean/construct_boundary_ordered.csv \
  --output_prefix outputs/construct_curv_localpoly \
  --neighbors 7 --closed True
"""
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

def load_points(csv_path):
    """
    Load 2D points from CSV.
    Expected columns (flexible): either headerless 'x,y' or a header containing 'x' and 'y'.
    """
    try:
        arr = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=float)
        if ('x' in arr.dtype.names) and ('y' in arr.dtype.names):
            pts = np.c_[arr['x'], arr['y']]
            return pts
    except Exception:
        pass
    # Try headerless
    pts = np.genfromtxt(csv_path, delimiter=",", dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(-1, 2)
    if pts.shape[1] != 2:
        raise ValueError("Could not load 2D points from CSV (expect two columns 'x,y').")
    return pts

def local_arclength_parameters(P, idxs):
    """
    Given point set P (N x 2) and a list of ordered indices idxs around a central index,
    compute signed cumulative arclength s with s=0 at the central point (middle of idxs).
    Returns s (len M), and the corresponding XY subarray.
    """
    XY = P[idxs]
    # central index in window:
    mid = len(idxs)//2
    # forward arclength from center
    s = np.zeros(len(idxs), dtype=float)
    # accumulate forward (to the right)
    for k in range(mid+1, len(idxs)):
        s[k] = s[k-1] + np.linalg.norm(XY[k] - XY[k-1])
    # accumulate backward (to the left)
    for k in range(mid-1, -1, -1):
        s[k] = s[k+1] - np.linalg.norm(XY[k+1] - XY[k])
    return s, XY

def quadratic_design(s):
    """Return design matrix for quadratic fit: [1, s, s^2]."""
    return np.c_[np.ones_like(s), s, s**2]

def fit_quadratic(s, vals):
    """
    Fit vals ≈ c0 + c1 s + c2 s^2 by least squares.
    Returns (c0, c1, c2).
    """
    A = quadratic_design(s)
    # Solve via robust least squares
    coefs, *_ = np.linalg.lstsq(A, vals, rcond=None)
    return coefs  # (c0, c1, c2)

def curvature_from_param_quadratic(ax, bx):
    """
    Given parametric quadratic coefficients:
      x(s) = ax[0] + ax[1] s + ax[2] s^2
      y(s) = bx[0] + bx[1] s + bx[2] s^2
    Compute curvature at s=0 using:
      kappa = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
      where x' = ax[1], x'' = 2*ax[2], y' = bx[1], y'' = 2*bx[2]
    Also return the signed curvature (orientation via cross product sign).
    """
    x1 = ax[1]; x2 = 2.0*ax[2]
    y1 = bx[1]; y2 = 2.0*bx[2]
    cross = x1*y2 - y1*x2
    speed = np.sqrt(x1*x1 + y1*y1) + 1e-16
    denom = speed**3
    kappa_signed = cross / denom
    kappa = np.abs(kappa_signed)
    return kappa, kappa_signed, speed, x1, y1, x2, y2

def index_window(i, m, N, closed=True):
    """
    Build an index window of size (2*m+1) centered at i.
    If closed=True, wrap indices modulo N (closed curve).
    If closed=False, clamp at ends (open curve).
    """
    idxs = []
    for d in range(-m, m+1):
        j = i + d
        if closed:
            idxs.append(j % N)
        else:
            j = min(max(j, 0), N-1)
            idxs.append(j)
    return idxs

def compute_curvature_localpoly(P, neighbors=7, closed=True, stride=1):
    """
    Compute curvature at each point using local quadratic polynomial fits in arclength.
    Parameters
    ----------
    P : (N,2) array of ordered boundary points
    neighbors : use ±neighbors points (window size = 2*neighbors+1); neighbors>=2 recommended
    closed : treat curve as closed (wrap indexing)
    stride : evaluate every 'stride' points (for speed), others will be linearly interpolated

    Returns
    -------
    kappa : (N,) curvature (nonnegative)
    kappa_signed : (N,) signed curvature
    speed : (N,) |(x',y')|
    aux   : dict with derivatives for diagnostics
    """
    N = P.shape[0]
    m = int(neighbors)
    if m < 2:
        raise ValueError("neighbors must be >= 2 for a meaningful quadratic fit.")

    kappa = np.zeros(N, dtype=float)
    kappa_s = np.zeros(N, dtype=float)
    speed = np.zeros(N, dtype=float)
    x1_all = np.zeros(N, dtype=float)
    y1_all = np.zeros(N, dtype=float)
    x2_all = np.zeros(N, dtype=float)
    y2_all = np.zeros(N, dtype=float)

    eval_idx = range(0, N, max(1, int(stride)))

    for i in eval_idx:
        idxs = index_window(i, m, N, closed=closed)
        s, XY = local_arclength_parameters(P, idxs)
        ax = fit_quadratic(s, XY[:,0])
        bx = fit_quadratic(s, XY[:,1])
        k, ks, sp, x1, y1, x2, y2 = curvature_from_param_quadratic(ax, bx)
        kappa[i] = k
        kappa_s[i] = ks
        speed[i] = sp
        x1_all[i], y1_all[i], x2_all[i], y2_all[i] = x1, y1, x2, y2

    # If stride>1, fill missing via linear interpolation along index
    if stride > 1:
        known = np.array(list(eval_idx))
        for arr in (kappa, kappa_s, speed, x1_all, y1_all, x2_all, y2_all):
            missing = np.setdiff1d(np.arange(N), known)
            arr[missing] = np.interp(missing, known, arr[known])

    aux = dict(xprime=x1_all, yprime=y1_all, x2=x2_all, y2=y2_all)
    return kappa, kappa_s, speed, aux

def save_csv(prefix, P, kappa, kappa_s, speed, aux):
    os.makedirs(os.path.dirname(prefix), exist_ok=True)
    out_csv = f"{prefix}_curvature.csv"
    header = "idx,x,y,curvature,kappa_signed,speed,xprime,yprime,x2,y2"
    idx = np.arange(P.shape[0])
    out = np.c_[idx, P[:,0], P[:,1], kappa, kappa_s, speed, aux['xprime'], aux['yprime'], aux['x2'], aux['y2']]
    np.savetxt(out_csv, out, delimiter=",", header=header, comments="", fmt="%.10g")
    return out_csv

def plot_outputs(prefix, P, kappa):
    # Histogram
    plt.figure(figsize=(6,4))
    plt.hist(kappa, bins=64)
    plt.xlabel(r"Curvature $\kappa$")
    plt.ylabel("Count")
    plt.title("Local-Polynomial Curvature Histogram")
    plt.tight_layout()
    hist_png = f"{prefix}_curvature_hist.png"
    plt.savefig(hist_png, dpi=200)
    plt.close()

    # Color overlay
    plt.figure(figsize=(5,5))
    sc = plt.scatter(P[:,0], P[:,1], c=kappa, s=8)
    plt.axis('equal'); plt.axis('off')
    cbar = plt.colorbar(sc, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\kappa$")
    plt.title("Curvature Overlay (Local-Polynomial)")
    plt.tight_layout()
    overlay_png = f"{prefix}_curvature_overlay.png"
    plt.savefig(overlay_png, dpi=220)
    plt.close()
    return hist_png, overlay_png

def write_summary(prefix, kappa):
    out_txt = f"{prefix}_summary.txt"
    stats = dict(
        n = len(kappa),
        mean = float(np.mean(kappa)),
        median = float(np.median(kappa)),
        std = float(np.std(kappa)),
        q05 = float(np.quantile(kappa, 0.05)),
        q95 = float(np.quantile(kappa, 0.95)),
        max = float(np.max(kappa)),
    )
    lines = [f"{k}: {v:.10g}" for k,v in stats.items()]
    with open(out_txt, "w") as f:
        f.write("Local-Polynomial Curvature Summary\n")
        f.write("\n".join(lines) + "\n")
    return out_txt

def main():
    ap = argparse.ArgumentParser(description="Local-polynomial curvature on 2D boundary points.")
    ap.add_argument("--input_csv", required=True, help="CSV with ordered boundary points (columns: x,y or header with x,y).")
    ap.add_argument("--output_prefix", required=True, help="Prefix for outputs (CSV/PNG/TXT).")
    ap.add_argument("--neighbors", type=int, default=7, help="Use ±neighbors points for quadratic fits (window size=2*neighbors+1).")
    ap.add_argument("--closed", type=lambda s: s.lower() in ['true','1','yes','y'], default=True, help="Treat boundary as closed (wrap indices).")
    ap.add_argument("--stride", type=int, default=1, help="Evaluate every 'stride' points (others interpolated).")
    args = ap.parse_args()

    P = load_points(args.input_csv)
    if P.shape[0] < 2*args.neighbors + 1:
        print(f"ERROR: Need at least {2*args.neighbors+1} points; got {P.shape[0]}.", file=sys.stderr)
        sys.exit(2)

    kappa, kappa_s, speed, aux = compute_curvature_localpoly(
        P, neighbors=args.neighbors, closed=args.closed, stride=args.stride
    )

    out_csv = save_csv(args.output_prefix, P, kappa, kappa_s, speed, aux)
    hist_png, overlay_png = plot_outputs(args.output_prefix, P, kappa)
    out_txt = write_summary(args.output_prefix, kappa)

    print("Wrote:")
    print("  ", out_csv)
    print("  ", hist_png)
    print("  ", overlay_png)
    print("  ", out_txt)

if __name__ == "__main__":
    main()
