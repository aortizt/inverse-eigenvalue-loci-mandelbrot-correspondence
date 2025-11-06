#!/usr/bin/env python3
"""
Iterative variogram <-> potential coupling experiment
(Construct <-> Mandelbrot)

Files expected in current directory:
 - construct_points.csv     (Nx2)
 - mandel_boundary_sample.csv (Mx2)
 - matches_indices.csv      (N,)   (for each Construct point i, matches[i] is matched index in Mandel sample)

Outputs per iteration:
 - variogram CSVs (construct)
 - potentials images and Laplacian images
 - a summary CSV with key metrics per iteration

Author: assistant (adapted for your project)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import pdist, squareform
from math import isnan

# ----------------------------
# USER PARAMETERS (tweakable)
# ----------------------------
N_ITER = 4         # number of coupling iterations (3-5 recommended)
VARIO_BINS = 50
GRID_RES = 300     # grid resolution for potentials (use 200-400 depending on speed)
MAX_ITER_MB = 300  # iterations for escape potential
ESCAPE_RAD = 10.0
NUDGE_ALPHA = 0.25 # base learning rate for moving Construct points toward matched M (0..1)
SMOOTH_FACTOR = 1.0 # multiplier that maps variogram range -> gaussian sigma
VARIO_PERCENT = 0.90 # use 90% of max variogram to estimate 'range' a
WIN_LOCAL_CORR = 12  # half-window size for local correlation map (pixels)
OUTPUT_PREFIX = "iter"  # file prefix

# ----------------------------
# Utility / core functions
# ----------------------------
def load_data():
    C = np.loadtxt("construct_points.csv", delimiter=",")
    M = np.loadtxt("mandel_boundary_sample.csv", delimiter=",")
    matches = np.loadtxt("matches_indices.csv", dtype=int, delimiter=",", ndmin=1)
    # ensure shapes
    if C.ndim==1: C = C.reshape((-1,2))
    if M.ndim==1: M = M.reshape((-1,2))
    print(f"Loaded Construct: {C.shape}, Mandelbrot: {M.shape}, matches: {matches.shape}")
    return C, M, matches

def empirical_variogram_from_field_locs(locs, values=None, max_dist=None, nbins=50):
    """
    If values is None: compute variogram from coordinates (use squared distance as 'field diff')
    Else: compute semivariogram for scalar 'values' at locs (default use distances if values is None)
    """
    N = len(locs)
    D = pdist(locs)  # pairwise distances
    if max_dist is None:
        max_dist = 0.5 * D.max() if D.size>0 else 1.0
    bins = np.linspace(0, max_dist, nbins+1)
    centers = 0.5*(bins[:-1]+bins[1:])
    gamma = np.full(nbins, np.nan)
    counts = np.zeros(nbins, dtype=int)

    if values is None:
        # use squared Euclidean differences between points themselves (not recommended normally)
        sqdiffs = D**2
    else:
        # vector of differences for each pair
        V = values
        # compute pairwise differences using indexing trick
        idx = np.arange(N)
        I, J = np.triu_indices(N, k=1)
        diffs = (V[I] - V[J])**2
        # but we want distances as pairwise distances:
        sqdiffs = diffs
        D = np.linalg.norm(locs[I] - locs[J], axis=1)

    for k in range(nbins):
        mask = (D >= bins[k]) & (D < bins[k+1])
        if np.any(mask):
            gamma[k] = 0.5 * np.mean(sqdiffs[mask])
            counts[k] = np.sum(mask)
    return centers, gamma, counts

def variogram_estimate_range(lags, gamma, pct=0.9):
    # return lag where gamma reaches pct * max(non-nan gamma)
    finite = np.isfinite(gamma)
    if not np.any(finite):
        return None
    maxi = np.nanmax(gamma)
    threshold = pct * maxi
    # find first index where gamma >= threshold
    for lag,g in zip(lags, gamma):
        if np.isfinite(g) and g >= threshold:
            return lag
    # fallback
    return lags[-1]

def log_potential(points, grid_x, grid_y):
    X, Y = np.meshgrid(grid_x, grid_y)
    U = np.zeros_like(X, dtype=float)
    pts = np.asarray(points)
    for (cx,cy) in pts:
        dx = X - cx
        dy = Y - cy
        r = np.hypot(dx, dy)
        U += np.log(r + 1e-12)
    U = U / len(pts)
    return U

def escape_potential(grid_x, grid_y, max_iter=200, R=10.0):
    X, Y = np.meshgrid(grid_x, grid_y)
    nx, ny = X.shape
    U = np.zeros_like(X, dtype=float)
    # iterate rows for cache
    for iy in range(nx):
        for ix in range(ny):
            c = X[iy,ix] + 1j*Y[iy,ix]
            z = 0+0j
            val = 0.0
            for k in range(max_iter):
                z = z*z + c
                if abs(z) > R:
                    val = np.log(abs(z)) / (k+1)
                    break
            U[iy,ix] = val
    return U

def laplacian_fd(U, h):
    # central 5-point Laplacian using roll (periodic-like boundaries)
    lap = (-4*U + np.roll(U,1,axis=0) + np.roll(U,-1,axis=0)
           + np.roll(U,1,axis=1) + np.roll(U,-1,axis=1)) / (h*h)
    return lap

def local_correlation_map(U1, U2, win):
    ny, nx = U1.shape
    corr_map = np.full_like(U1, np.nan, dtype=float)
    for iy in range(win, ny - win):
        for ix in range(win, nx - win):
            a = U1[iy-win:iy+win, ix-win:ix+win].ravel()
            b = U2[iy-win:iy+win, ix-win:ix+win].ravel()
            mask = ~ (np.isnan(a) | np.isnan(b))
            if mask.sum() > 10:
                try:
                    corr_map[iy, ix] = pearsonr(a[mask], b[mask])[0]
                except Exception:
                    corr_map[iy, ix] = np.nan
    return corr_map

# ----------------------------
# Main iterative experiment
# ----------------------------
def run_iterative_pipeline(n_iter=N_ITER):
    # load
    C, M, matches = load_data()
    N = len(C)
    # sanity: matches should be length N
    if matches.shape[0] != N:
        print("Warning: matches length != number of Construct points")
    # choose grid extent from bounding box of union, with margin
    all_points = np.vstack((C, M))
    xmin, ymin = all_points.min(axis=0) - 0.5
    xmax, ymax = all_points.max(axis=0) + 0.5
    # create regular grid
    grid_x = np.linspace(xmin, xmax, GRID_RES)
    grid_y = np.linspace(ymin, ymax, GRID_RES)
    h = grid_x[1] - grid_x[0]

    # summary container
    rows = []
    C_current = C.copy()

    for it in range(1, n_iter+1):
        print(f"\n=== ITERATION {it} ===")

        # compute matching distances d_i using current C and static M + matches
        # matches assumed 1D: matches[i] = index in M
        matched_M = M[matches]   # shape (N,2)
        diffs = C_current - matched_M
        dists = np.linalg.norm(diffs, axis=1)

        # experimental variogram on matching distances (treat dists as scalar field on locs C_current)
        lags, gamma, counts = empirical_variogram_from_field_locs(C_current, values=dists, nbins=VARIO_BINS)
        # estimate range 'a' as lag where gamma reaches VARIO_PERCENT of max
        a_est = variogram_estimate_range(lags, gamma, pct=VARIO_PERCENT)
        print(f"Estimated variogram range (a) = {a_est:.4f}")

        # compute potentials
        print("Computing Construct log-potential...")
        U_C = log_potential(C_current, grid_x, grid_y)
        print("Computing Mandelbrot escape potential...")
        U_M = escape_potential(grid_x, grid_y, max_iter=MAX_ITER_MB, R=ESCAPE_RAD)

        # smooth Construct potential according to variogram-derived scale
        if (a_est is None) or (a_est <= 0):
            sigma_px = 1.0
        else:
            # map spatial range a_est (in same units as grid) -> gaussian sigma in pixels
            sigma_px = max(0.5, SMOOTH_FACTOR * (a_est / h) / 2.0)
        U_C_smooth = gaussian_filter(U_C, sigma=sigma_px)
        print(f"Smoothing sigma (pixels) = {sigma_px:.3f}")

        # Laplacians
        Lap_C = laplacian_fd(U_C_smooth, h)
        Lap_M = laplacian_fd(U_M, h)

        # global correlations between U_C_smooth and U_M
        flatC = U_C_smooth.ravel()
        flatM = U_M.ravel()
        mask = ~(np.isnan(flatC) | np.isnan(flatM))
        if mask.sum() > 10:
            corr_global, pval = pearsonr(flatC[mask], flatM[mask])
        else:
            corr_global, pval = np.nan, np.nan
        print(f"Global Pearson corr (potentials): r={corr_global:.4f}, p={pval:.2e}")

        # correlation between Laplacians (fields)
        flatLC = Lap_C.ravel()
        flatLM = Lap_M.ravel()
        mask2 = ~(np.isnan(flatLC) | np.isnan(flatLM))
        if mask2.sum() > 10:
            corr_lap = np.corrcoef(flatLC[mask2], flatLM[mask2])[0,1]
        else:
            corr_lap = np.nan
        print(f"Correlation (Laplacians): {corr_lap:.4f}")

        # local correlation map (coarse window to save time)
        print("Computing local correlation map (this may take a moment)...")
        local_corr = local_correlation_map(U_C_smooth, U_M, WIN_LOCAL_CORR)

        # Save iteration plots and data
        fig, axs = plt.subplots(2,3, figsize=(18,10))
        axs = axs.ravel()
        im0 = axs[0].imshow(U_C, extent=[grid_x[0], grid_x[-1], grid_y[0], grid_y[-1]],
                            origin='lower', cmap='viridis')
        axs[0].set_title(f"U_C (raw) iter{it}")
        plt.colorbar(im0, ax=axs[0])
        im1 = axs[1].imshow(U_M, extent=[grid_x[0], grid_x[-1], grid_y[0], grid_y[-1]],
                            origin='lower', cmap='inferno')
        axs[1].set_title(f"U_M iter{it}")
        plt.colorbar(im1, ax=axs[1])
        im2 = axs[2].imshow(U_C_smooth - U_M, extent=[grid_x[0], grid_x[-1], grid_y[0], grid_y[-1]],
                            origin='lower', cmap='coolwarm', vmin=-np.max(np.abs(U_C_smooth-U_M)), vmax=np.max(np.abs(U_C_smooth-U_M)))
        axs[2].set_title("U_C_smooth - U_M")
        plt.colorbar(im2, ax=axs[2])

        im3 = axs[3].imshow(Lap_C, extent=[grid_x[0], grid_x[-1], grid_y[0], grid_y[-1]],
                            origin='lower', cmap='bwr')
        axs[3].set_title("Lap U_C (smoothed)")
        plt.colorbar(im3, ax=axs[3])
        im4 = axs[4].imshow(Lap_M, extent=[grid_x[0], grid_x[-1], grid_y[0], grid_y[-1]],
                            origin='lower', cmap='bwr')
        axs[4].set_title("Lap U_M")
        plt.colorbar(im4, ax=axs[4])
        im5 = axs[5].imshow(local_corr, extent=[grid_x[0], grid_x[-1], grid_y[0], grid_y[-1]],
                            origin='lower', cmap='RdYlGn', vmin=-1, vmax=1)
        axs[5].set_title("Local corr (U_C_smooth, U_M)")
        plt.colorbar(im5, ax=axs[5])

        plt.suptitle(f"Iteration {it} -- corr_pot={corr_global:.4f}, corr_lap={corr_lap:.4f}, a_est={a_est:.4f}")
        plt.tight_layout(rect=[0,0,1,0.96])
        figfile = f"{OUTPUT_PREFIX}_{it}_potentials_and_lap.png"
        plt.savefig(figfile, dpi=180)
        plt.close(fig)
        print("Saved figure:", figfile)

        # Save variogram CSV
        var_csv = f"{OUTPUT_PREFIX}_{it}_variogram_construct.csv"
        np.savetxt(var_csv, np.c_[lags, gamma, counts], delimiter=",", header="lag,gamma,count", comments="")
        print("Saved variogram table:", var_csv)

        # Save local corr map
        np.save(f"{OUTPUT_PREFIX}_{it}_localcorr.npy", local_corr)

        # append summary row
        rows.append((it, a_est, sigma_px, corr_global, corr_lap, np.nanmean(dists), np.nanmedian(dists), np.nanmax(dists)))

        # --------------------------
        # Inverse-step: nudge Construct points towards matched M
        # --------------------------
        # compute normalized weights: closer matches move less? we'll use weight ~ (1 - d / (maxd + eps))
        maxd = np.nanmax(dists) if np.isfinite(np.nanmax(dists)) and np.nanmax(dists)>0 else 1.0
        weights = 1.0 - (dists / (maxd + 1e-12))    # in [0,1], larger for small dists
        # learning rate scaled by NUDGE_ALPHA and by variogram scale: larger a_est -> larger influence (coarser scale)
        if a_est is None or a_est<=0:
            scale = 1.0
        else:
            scale = min(2.0, max(0.1, a_est))  # bound scale
        lr = NUDGE_ALPHA * (scale / (scale + 1.0))
        # update: C_new = C_old + lr * weights[:,None] * (M_matched - C_old)
        displacement = lr * weights[:,None] * (matched_M - C_current)
        C_current = C_current + displacement

        # End iteration loop

    # save summary table
    import csv
    summary_file = f"{OUTPUT_PREFIX}_summary_metrics.csv"
    with open(summary_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iter","vario_range_a","sigma_px","corr_pot","corr_lap","d_mean","d_median","d_max"])
        for r in rows:
            writer.writerow(r)
    print("Saved summary:", summary_file)
    return

# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    run_iterative_pipeline(N_ITER)
