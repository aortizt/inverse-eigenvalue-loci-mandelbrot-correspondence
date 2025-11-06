#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  1 20:28:18 2025

@author: dakini
"""

# -*- coding: utf-8 -*-
"""
Variograms for Construct vs Mandelbrot (recomputed from scratch).
- Generates Construct points (inverse eigenvalues of Lucas-type companions)
- Generates Mandelbrot boundary proxy by distance-estimator mask
- Builds potentials on a shared grid:
    * U_C: logarithmic potential of Construct point cloud
    * U_M: escape/Green-function style potential for Mandelbrot proxy
- Computes isotropic empirical semivariograms and a cross-semivariogram
- Saves plots and CSV with numeric results

Run in Spyder (or python) as-is.
"""

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from dataclasses import dataclass
import csv
from pathlib import Path

# ----------------------------
# Reproducibility & IO paths
# ----------------------------
np.random.seed(42)
OUTDIR = Path("./outputs_variograms")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Construct (Lucas companion) 
# ----------------------------
def lucas_companion(n: int) -> np.ndarray:
    """First row all ones; ones on subdiagonal; a_{2,1}=1."""
    A = np.zeros((n, n), dtype=float)
    A[0, :] = 1.0
    A[1:, :-1] = np.eye(n - 1)
    A[1, 0] = 1.0
    return A

def construct_points(n_list):
    """Inverse eigenvalues for all n in n_list."""
    pts = []
    for n in n_list:
        A = lucas_companion(n)
        vals = la.eigvals(A)
        vals = vals[np.abs(vals) > 1e-14]
        pts.extend(1.0 / vals)
    return np.asarray(pts, dtype=np.complex128)

# ----------------------------
# Mandelbrot boundary proxy
# ----------------------------
def mandelbrot_distance_estimator(c, max_iter=500, R=4.0, eps=1e-14):
    """
    Standard distance estimator. Returns (escaped_mask, dist, last_z, last_dz).
    """
    z = np.zeros_like(c, dtype=np.complex128)
    dz = np.ones_like(c, dtype=np.complex128)
    escaped = np.zeros(c.shape, dtype=bool)
    last_z = np.zeros_like(c, dtype=np.complex128)
    last_dz = np.ones_like(c, dtype=np.complex128)

    for _ in range(max_iter):
        dz = 2.0 * z * dz + 1.0
        z = z * z + c
        mask = (~escaped) & (np.abs(z) > R)
        escaped |= mask
        last_z[mask] = z[mask]
        last_dz[mask] = dz[mask]

    dist = np.zeros(c.shape, dtype=float)
    m = escaped
    if np.any(m):
        z_ = last_z[m]
        dz_ = last_dz[m]
        num = np.log(np.maximum(np.abs(z_), 1.0)) * np.abs(z_)
        den = np.maximum(np.abs(2.0 * z_ * dz_), eps)
        dist[m] = np.nan_to_num(num / den, nan=0.0, posinf=0.0, neginf=0.0)

    return escaped, dist, last_z, last_dz

def mandelbrot_boundary_points(xmin=-2.25, xmax=1.25, ymin=-1.75, ymax=1.75,
                               N=600, dist_thresh=0.002, max_iter=500):
    """
    Sample grid, compute distance estimator, and keep near-boundary points
    (escaped & small distance). Returns complex array of boundary samples.
    """
    xs = np.linspace(xmin, xmax, N)
    ys = np.linspace(ymin, ymax, N)
    XX, YY = np.meshgrid(xs, ys, indexing='xy')
    C = XX + 1j * YY

    escaped, dist, *_ = mandelbrot_distance_estimator(C, max_iter=max_iter)
    near = escaped & (dist <= dist_thresh)

    return C[near]  # complex boundary proxy

# ----------------------------
# Potentials on a shared grid
# ----------------------------
@dataclass
class Grid:
    x: np.ndarray
    y: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    Z: np.ndarray  # complex coordinates X + iY
    dx: float
    dy: float

def make_grid(xmin, xmax, ymin, ymax, Nx=256, Ny=256) -> Grid:
    xs = np.linspace(xmin, xmax, Nx)
    ys = np.linspace(ymin, ymax, Ny)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    Z = X + 1j * Y
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    return Grid(xs, ys, X, Y, Z, dx, dy)

def log_potential_from_points(grid: Grid, pts: np.ndarray, eps=1e-6, chunk=5000):
    """
    U_C(z) = (1/N) sum log(1 / |z - p_k| + eps).
    Chunked for memory safety.
    """
    H, W = grid.Z.shape
    U = np.zeros((H, W), dtype=float)
    N = len(pts)
    if N == 0:
        return U

    pts = pts.ravel()
    for start in range(0, N, chunk):
        P = pts[start:start+chunk][None, None, :]  # (1,1,K)
        diff = grid.Z[..., None] - P
        r = np.abs(diff) + eps
        U += np.log(1.0 / r).sum(axis=-1)
    U /= N
    return U

def mandelbrot_escape_potential(grid: Grid, max_iter=500, R=4.0):
    """
    Approx Green-function-like potential:
    For escaped points, G(c) ~ log|z_n| / 2^n (approx);
    Interior points set to 0. Smooth for numerical stability.
    """
    c = grid.Z
    z = np.zeros_like(c, dtype=np.complex128)
    g = np.zeros(c.shape, dtype=float)
    esc = np.zeros(c.shape, dtype=bool)

    for n in range(1, max_iter + 1):
        z = z * z + c
        mask = (~esc) & (np.abs(z) > R)
        if np.any(mask):
            # store potential at first escape
            g[mask] = np.log(np.abs(z[mask])) / (2.0 ** n)
            esc[mask] = True

    # small smoothing to avoid sharp discontinuities
    # (simple 3x3 average; pure numpy for portability)
    G = g.copy()
    G[1:-1,1:-1] = (
        g[1:-1,1:-1] + g[:-2,1:-1] + g[2:,1:-1] + g[1:-1,:-2] + g[1:-1,2:]
    ) / 5.0
    return G

# ----------------------------
# Semivariogram utilities
# ----------------------------
def sample_semivariogram(field: np.ndarray, grid: Grid,
                         r_bins: np.ndarray, max_pairs_per_bin=20000):
    """
    Empirical isotropic semivariogram on a regular grid.
    - Randomly subsamples pixel locations to control O(N^2) cost.
    - For each bin, uses up to max_pairs_per_bin pairs.
    Returns (r_centers, gamma).
    """
    H, W = field.shape
    # Build coordinate list and values
    yy, xx = np.mgrid[0:H, 0:W]
    coords = np.column_stack([grid.X.ravel(), grid.Y.ravel()])
    vals = field.ravel()

    # Subsample points to keep pair counts manageable
    # Target ~ M points → O(M^2) pairs; choose based on grid size
    M_target = min(15000, coords.shape[0])
    idx = np.random.choice(coords.shape[0], size=M_target, replace=False)
    C = coords[idx]        # (M,2)
    V = vals[idx]          # (M,)

    # Pairwise distances in chunks to avoid huge memory
    # We'll collect per-bin squared diffs until we hit cap.
    nbins = len(r_bins) - 1
    sums = np.zeros(nbins, dtype=float)
    counts = np.zeros(nbins, dtype=int)

    # Chunking loop
    chunk = 4000
    for a in range(0, M_target, chunk):
        a_end = min(a + chunk, M_target)
        Ca = C[a:a_end]           # (A,2)
        Va = V[a:a_end][:, None]  # (A,1)

        # full block vs all to form pairs without double count:
        # We'll compute against b >= a to avoid duplication, then
        # handle diagonals carefully by ignoring identical indices.
        for b in range(a, M_target, chunk):
            b_end = min(b + chunk, M_target)
            Cb = C[b:b_end]       # (B,2)
            Vb = V[b:b_end][None, :]  # (1,B)

            # distances and squared differences
            D = la.norm(Ca[:, None, :] - Cb[None, :, :], axis=2)  # (A,B)
            # mask out identical pairs if a == b
            if a == b:
                diag = np.eye(D.shape[0], D.shape[1], dtype=bool)
            else:
                diag = np.zeros_like(D, dtype=bool)

            dV2 = (Va - Vb) ** 2   # (A,B)

            # For each bin, collect up to cap
            for k in range(nbins):
                m = (D >= r_bins[k]) & (D < r_bins[k+1]) & (~diag)
                if not np.any(m):
                    continue
                # Cap pairs per bin
                room = max_pairs_per_bin - counts[k]
                if room <= 0:
                    continue
                where = np.where(m)
                if where[0].size > room:
                    sel = np.random.choice(where[0].size, size=room, replace=False)
                    part = dV2[where][sel]
                else:
                    part = dV2[where]
                sums[k] += part.sum()
                counts[k] += part.size

    gamma = np.zeros(nbins, dtype=float)
    nz = counts > 0
    gamma[nz] = 0.5 * (sums[nz] / counts[nz])
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
    return r_centers, gamma

def sample_cross_semivariogram(field1: np.ndarray, field2: np.ndarray, grid: Grid,
                               r_bins: np.ndarray, max_pairs_per_bin=20000):
    """
    Cross-semivariogram using fields on the SAME grid:
      gamma_12(r) = (1/2) E[(Z1(x) - Z2(y))^2 | ||x - y|| in bin r]
    Empirical: same sampling/aggregation as above, but with mixed fields.
    """
    assert field1.shape == field2.shape
    H, W = field1.shape
    yy, xx = np.mgrid[0:H, 0:W]
    coords = np.column_stack([grid.X.ravel(), grid.Y.ravel()])
    V1 = field1.ravel()
    V2 = field2.ravel()

    M_target = min(15000, coords.shape[0])
    idx1 = np.random.choice(coords.shape[0], size=M_target, replace=False)
    idx2 = np.random.choice(coords.shape[0], size=M_target, replace=False)

    C1 = coords[idx1]
    C2 = coords[idx2]
    W1 = V1[idx1][:, None]  # (M,1)
    W2 = V2[idx2][None, :]  # (1,M)

    nbins = len(r_bins) - 1
    sums = np.zeros(nbins, dtype=float)
    counts = np.zeros(nbins, dtype=int)

    chunk = 4000
    for a in range(0, M_target, chunk):
        a_end = min(a + chunk, M_target)
        Ca = C1[a:a_end]
        Wa = W1[a:a_end]

        for b in range(0, M_target, chunk):
            b_end = min(b + chunk, M_target)
            Cb = C2[b:b_end]
            Wb = W2[:, b:b_end]

            D = la.norm(Ca[:, None, :] - Cb[None, :, :], axis=2)  # (A,B)
            dV2 = (Wa - Wb) ** 2

            for k in range(nbins):
                m = (D >= r_bins[k]) & (D < r_bins[k+1])
                if not np.any(m):
                    continue
                room = max_pairs_per_bin - counts[k]
                if room <= 0:
                    continue
                where = np.where(m)
                if where[0].size > room:
                    sel = np.random.choice(where[0].size, size=room, replace=False)
                    part = dV2[where][sel]
                else:
                    part = dV2[where]
                sums[k] += part.sum()
                counts[k] += part.size

    gamma12 = np.zeros(nbins, dtype=float)
    nz = counts > 0
    gamma12[nz] = 0.5 * (sums[nz] / counts[nz])
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
    return r_centers, gamma12

# ----------------------------
# Main pipeline
# ----------------------------
if __name__ == "__main__":

    # 1) Generate Construct points
    n_list = [30, 60, 90, 120, 180, 240, 300]  # raise if desired
    Cpts = construct_points(n_list)
    print(f"Construct points: {Cpts.size}")

    # 2) Mandelbrot boundary proxy (distance-threshold sample)
    Mpts = mandelbrot_boundary_points(N=700, dist_thresh=0.0018, max_iter=600)
    print(f"Mandelbrot boundary samples: {Mpts.size}")

    # 3) Shared grid
    grid = make_grid(xmin=-2.25, xmax=1.25, ymin=-1.75, ymax=1.75, Nx=256, Ny=256)

    # 4) Potentials on grid
    print("Computing Construct log-potential on grid ...")
    Uc = log_potential_from_points(grid, Cpts, eps=1e-6, chunk=6000)

    print("Computing Mandelbrot escape potential on grid ...")
    Um = mandelbrot_escape_potential(grid, max_iter=600, R=4.0)

    # Optional: normalize fields (helpful for variogram comparability)
    Uc_norm = (Uc - np.nanmin(Uc)) / (np.nanmax(Uc) - np.nanmin(Uc) + 1e-12)
    Um_norm = (Um - np.nanmin(Um)) / (np.nanmax(Um) - np.nanmin(Um) + 1e-12)

    # 5) Variogram binning
    # Use up to ~half the min box size as max lag; tune as needed.
    rmax = 1.3
    nbins = 35
    r_bins = np.linspace(0.0, rmax, nbins + 1)

    print("Sampling semivariograms ... (this may take a few minutes for large grids)")
    rC, gC = sample_semivariogram(Uc_norm, grid, r_bins, max_pairs_per_bin=20000)
    rM, gM = sample_semivariogram(Um_norm, grid, r_bins, max_pairs_per_bin=20000)
    rX, gX = sample_cross_semivariogram(Uc_norm, Um_norm, grid, r_bins, max_pairs_per_bin=20000)

    # 6) Save CSV
    csv_path = OUTDIR / "variograms_construct_mandelbrot.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["r_center", "gamma_Construct", "gamma_Mandelbrot", "gamma_cross"])
        for i in range(len(rC)):
            w.writerow([rC[i], gC[i], gM[i], gX[i]])
    print(f"Saved: {csv_path.resolve()}")

    # 7) Plots
    plt.figure(figsize=(8,5.5))
    plt.plot(rC, gC, "o-", label="Construct semivariogram")
    plt.plot(rM, gM, "s-", label="Mandelbrot semivariogram")
    plt.plot(rX, gX, "^-", label="Cross semivariogram")
    plt.xlabel("lag distance r")
    plt.ylabel(r"$\hat{\gamma}(r)$")
    plt.title("Semivariograms: Construct vs Mandelbrot (potentials on shared grid)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    fig1 = OUTDIR / "variogram_construct_mandelbrot.png"
    plt.tight_layout()
    plt.savefig(fig1, dpi=200)
    print(f"Saved: {fig1.resolve()}")

    # Quick visuals of potentials (optional, useful for sanity check)
    plt.figure(figsize=(10,4.2))
    plt.subplot(1,2,1)
    plt.imshow(Uc_norm, extent=[grid.x[0], grid.x[-1], grid.y[0], grid.y[-1]],
               origin='lower', aspect='equal')
    plt.title("Construct log-potential (normalized)")
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.subplot(1,2,2)
    plt.imshow(Um_norm, extent=[grid.x[0], grid.x[-1], grid.y[0], grid.y[-1]],
               origin='lower', aspect='equal')
    plt.title("Mandelbrot escape potential (normalized)")
    plt.colorbar(fraction=0.046, pad=0.04)

    fig2 = OUTDIR / "potentials_construct_mandelbrot.png"
    plt.tight_layout()
    plt.savefig(fig2, dpi=200)
    print(f"Saved: {fig2.resolve()}")

    print("Done.")
