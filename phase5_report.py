#!/usr/bin/env python3
# phase5_report.py
# Integrative report: combines metrics from previous phases into CSV + plots.

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import directed_hausdorff
import os, csv

# --- File paths (edit if needed) ---
BASE = "/home/Merlin/Desktop/out_clean/"
OUTDIR = os.path.join(BASE, "phase5_report")
os.makedirs(OUTDIR, exist_ok=True)

PATH_CONSTRUCT = os.path.join(BASE, "construct_points.csv")
PATH_MANDEL = os.path.join(BASE, "mandel_boundary_sample.csv")
PATH_ALIGNED = os.path.join(BASE, "construct_aligned.csv")
PATH_MATCHES = os.path.join(BASE, "matches_indices.csv")
PATH_SPECTRAL = os.path.join("/mnt/data", "spectral_slope_results.txt")
PATH_FOURIER = os.path.join("/mnt/data", "First 10 Fourier modes.txt")

# --- Load data ---
def safe_load(path):
    if os.path.exists(path):
        try:
            return np.loadtxt(path, delimiter=",")
        except Exception:
            # fallback: try whitespace delimiter
            return np.loadtxt(path)
    else:
        print("Warning: file not found:", path)
        return None

C = safe_load(PATH_CONSTRUCT)
M = safe_load(PATH_MANDEL)
C_aligned = safe_load(PATH_ALIGNED)
matches = None
if os.path.exists(PATH_MATCHES):
    try:
        matches = np.loadtxt(PATH_MATCHES, delimiter=",", dtype=int)
    except Exception:
        try:
            matches = np.loadtxt(PATH_MATCHES, dtype=int)
        except Exception:
            matches = None
            print("Warning: could not read matches file.")

# --- Basic counts ---
nC = 0 if C is None else C.shape[0]
nM = 0 if M is None else M.shape[0]
nA = 0 if C_aligned is None else C_aligned.shape[0]

# --- Matching distances ---
dist_stats = {}
if C_aligned is not None and M is not None and matches is not None:
    L = min(len(matches), C_aligned.shape[0], M.shape[0])
    diffs = C_aligned[:L] - M[matches[:L]]
    dists = np.linalg.norm(diffs, axis=1)
    dist_stats['min'] = float(np.min(dists))
    dist_stats['median'] = float(np.median(dists))
    dist_stats['mean'] = float(np.mean(dists))
    dist_stats['max'] = float(np.max(dists))
    dist_stats['std'] = float(np.std(dists))
    # save histogram
    plt.figure(figsize=(6,4))
    plt.hist(dists, bins=40)
    plt.xlabel("Matching distance")
    plt.ylabel("Count")
    plt.title("Histogram of matching distances")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "matching_distance_hist.png"), dpi=200)
    plt.close()
else:
    print("Skipping matching distance stats (missing data)")

# --- Hausdorff distance ---
haus = None
if C_aligned is not None and M is not None:
    try:
        h1 = directed_hausdorff(C_aligned, M)[0]
        h2 = directed_hausdorff(M, C_aligned)[0]
        haus = max(h1, h2)
    except Exception as e:
        print("Hausdorff computation failed:", e)

# --- Curvature stats ---
def estimate_curvature(points):
    if points is None or len(points) < 3:
        return None
    dx = np.gradient(points[:,0])
    dy = np.gradient(points[:,1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    denom = (dx**2 + dy**2)**1.5
    denom[denom == 0] = np.nan
    curvature = np.abs(dx * ddy - dy * ddx) / denom
    return curvature

curv_stats = {}
if C_aligned is not None:
    curvC = estimate_curvature(C_aligned)
    if curvC is not None:
        curvC = curvC[np.isfinite(curvC)]
        if curvC.size > 0:
            curv_stats['construct_median'] = float(np.median(curvC))
            curv_stats['construct_mean'] = float(np.mean(curvC))
            curv_stats['construct_std'] = float(np.std(curvC))
            plt.figure(figsize=(6,3))
            plt.plot(curvC, linewidth=0.5)
            plt.yscale('log')
            plt.title('Construct curvature (log scale)')
            plt.xlabel('point index')
            plt.ylabel('curvature')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTDIR, 'curvature_construct.png'), dpi=200)
            plt.close()

if M is not None:
    curvM = estimate_curvature(M)
    if curvM is not None:
        curvM = curvM[np.isfinite(curvM)]
        if curvM.size > 0:
            curv_stats['mandel_median'] = float(np.median(curvM))
            curv_stats['mandel_mean'] = float(np.mean(curvM))
            curv_stats['mandel_std'] = float(np.std(curvM))
            plt.figure(figsize=(6,3))
            plt.plot(curvM, linewidth=0.5)
            plt.yscale('log')
            plt.title('Mandel curvature (log scale)')
            plt.xlabel('point index')
            plt.ylabel('curvature')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTDIR, 'curvature_mandel.png'), dpi=200)
            plt.close()

# --- Fractal dimension (box-counting) ---
def fractal_dimension(points, nscales=10):
    if points is None or len(points) == 0:
        return None, None
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    rng = maxs - mins
    scales = np.logspace(-2, 0, nscales)
    N = []
    for s in scales:
        step = rng * s
        step[step == 0] = 1e-12
        grid = np.floor((points - mins) / step).astype(int)
        N.append(len(np.unique(grid, axis=0)))
    coeffs = np.polyfit(np.log(1/np.array(scales)), np.log(N), 1)
    return float(coeffs[0]), (np.log(1/np.array(scales)), np.log(N))

fd_construct, logplotC = (None, None)
fd_mandel, logplotM = (None, None)
if C_aligned is not None:
    fd_construct, logplotC = fractal_dimension(C_aligned)
if M is not None:
    fd_mandel, logplotM = fractal_dimension(M)

if fd_construct is not None and fd_mandel is not None:
    plt.figure(figsize=(6,5))
    plt.plot(logplotC[0], logplotC[1], 'o-', label='Construct D~{:.3f}'.format(fd_construct))
    plt.plot(logplotM[0], logplotM[1], 'o-', label='Mandel D~{:.3f}'.format(fd_mandel))
    plt.xlabel('log(1/scale)')
    plt.ylabel('log(N boxes)')
    plt.legend()
    plt.title('Box-counting fractal dimension')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'boxcount_fd.png'), dpi=200)
    plt.close()

# --- Spectral slopes: attempt to read spectral_slope_results.txt if present ---
spectral_results = {}
if os.path.exists(PATH_SPECTRAL):
    try:
        with open(PATH_SPECTRAL, 'r') as f:
            spectral_results['raw'] = f.read()
    except Exception as e:
        spectral_results['raw'] = f"Failed to read: {e}"

# --- First 10 Fourier modes file (optional) ---
first10 = {}
if os.path.exists(PATH_FOURIER):
    try:
        with open(PATH_FOURIER, 'r') as f:
            first10['raw'] = f.read()
    except Exception as e:
        first10['raw'] = f"Failed to read: {e}"

# --- Aggregate summary CSV ---
summary = {
    'n_construct': nC,
    'n_mandel': nM,
    'n_aligned': nA,
    'hausdorff': haus,
    'match_min': dist_stats.get('min') if dist_stats else None,
    'match_median': dist_stats.get('median') if dist_stats else None,
    'match_mean': dist_stats.get('mean') if dist_stats else None,
    'match_max': dist_stats.get('max') if dist_stats else None,
    'curv_construct_mean': curv_stats.get('construct_mean') if curv_stats else None,
    'curv_construct_median': curv_stats.get('construct_median') if curv_stats else None,
    'curv_mandel_mean': curv_stats.get('mandel_mean') if curv_stats else None,
    'curv_mandel_median': curv_stats.get('mandel_median') if curv_stats else None,
    'fd_construct': fd_construct,
    'fd_mandel': fd_mandel,
    'spectral_raw': spectral_results.get('raw'),
    'first10_raw': first10.get('raw')
}

csv_path = os.path.join(OUTDIR, 'phase5_summary.csv')
with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for k, v in summary.items():
        writer.writerow([k, v])

print('Phase 5 report generated in', OUTDIR)
print('Summary CSV:', csv_path)
