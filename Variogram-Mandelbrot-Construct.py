#!/usr/bin/env python3
"""
Robust variogram + cross-variogram script for Construct vs Mandelbrot.

- Accepts matches in several formats:
    * single column (length == #Construct): mandel_index for each Construct i
    * two columns (construct_index, mandel_index)
    * two columns (mandel_index) with implicit construct index as row
- Produces: variogram_construct.csv, variogram_mandel.csv, crossvariogram_CM.csv
- Produces: variogram_CM.png and crossvariogram_CM.png

Author: assistant (corrected)
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
import sys

# -------------------------
# Parameters (tweak if needed)
# -------------------------
NBINS = 60
MAX_DIST_FACTOR = 0.5  # default max lag = factor * max_pairwise_dist
VERBOSE = True

# -------------------------
# Robust loaders
# -------------------------
def load_points(fname):
    """
    Load 2-column CSV of points. Accepts headerless numeric CSV.
    """
    arr = np.genfromtxt(fname, delimiter=',')
    if arr.ndim == 1:
        if arr.size == 2:
            arr = arr.reshape((1,2))
        else:
            raise ValueError(f"{fname} appears 1D with length {arr.size}; expected Nx2")
    if arr.shape[1] < 2:
        raise ValueError(f"{fname} should have at least 2 columns (x,y). Got shape {arr.shape}")
    return arr[:,0:2].astype(float)

def load_matches(fname, nC, nM):
    """
    Robustly interpret matches file.
    Returns (construct_idx, mandel_idx) arrays (both 1D, same length).
    Accept formats:
      - single column length nC: mandel index per construct (construct_idx = arange(nC))
      - two columns (construct_idx, mandel_idx)
      - two columns but first column equals 0..n-1 => treat as (construct_idx, mandel_idx)
      - other sensible variants tried.
    """
    raw = np.genfromtxt(fname, delimiter=',')
    if raw.size == 0:
        raise ValueError("Matches file appears empty.")
    raw = np.atleast_1d(raw)
    # if 1D:
    if raw.ndim == 1:
        if raw.size == nC:
            construct_idx = np.arange(nC, dtype=int)
            mandel_idx = raw.astype(int)
            return construct_idx, mandel_idx
        elif raw.size == 2:
            # ambiguous: one pair only
            construct_idx = np.array([int(raw[0])], dtype=int)
            mandel_idx = np.array([int(raw[1])], dtype=int)
            return construct_idx, mandel_idx
        else:
            # if length matches nM, perhaps it's mandel indices with implicit mapping?
            if raw.size == nM:
                construct_idx = np.arange(raw.size, dtype=int)
                mandel_idx = raw.astype(int)
                return construct_idx, mandel_idx
            raise ValueError(f"Cannot interpret 1D matches array of length {raw.size}. Expected length {nC} or {nM}.")
    # if 2D:
    if raw.ndim == 2:
        if raw.shape[1] == 1:
            # single column but returned as (N,1)
            col = raw[:,0].astype(int)
            if col.size == nC:
                return np.arange(nC, dtype=int), col
            else:
                return np.arange(col.size, dtype=int), col
        # if >=2 columns:
        col0 = raw[:,0].astype(int)
        col1 = raw[:,1].astype(int)
        # case: first column is [0..nC-1]
        if np.array_equal(col0, np.arange(col0.size)):
            return col0, col1
        # case: first column entirely in mandel index range and second in construct range (maybe swapped)
        if np.all((col0 >= 0) & (col0 < nM)) and np.all((col1 >= 0) & (col1 < nC)):
            # swapped -> col1 is construct index
            return col1, col0
        # case: both columns in range; assume (construct, mandel)
        if np.all((col0 >= 0) & (col0 < nC)) and np.all((col1 >= 0) & (col1 < nM)):
            return col0, col1
        # fallback: assume second column is mandel
        if np.all((col1 >= 0) & (col1 < nM)):
            return np.arange(col1.size, dtype=int), col1
        raise ValueError("Cannot reliably interpret matches file columns. Please provide either single-column mandel indices (one-per-construct) or two columns (construct_idx, mandel_idx).")

# -------------------------
# Variogram utilities
# -------------------------
def empirical_variogram_field(locs, values, nbins=50, max_dist=None):
    """
    Empirical semivariogram of a scalar field `values` sampled at positions `locs`.
    locs: (N,2), values: (N,)
    returns: lag_centers (K,), gamma (K,), counts (K,)
    """
    N = locs.shape[0]
    if N < 2:
        return np.array([]), np.array([]), np.array([])
    D = pdist(locs)  # pairwise distances (length M = N*(N-1)/2)
    Vdiff = pdist(values.reshape(-1,1), metric='euclidean')  # abs differences
    sqdiff = Vdiff**2
    if max_dist is None:
        max_dist = MAX_DIST_FACTOR * D.max()
    bins = np.linspace(0.0, max_dist, nbins+1)
    centers = 0.5*(bins[:-1] + bins[1:])
    gamma = np.full(nbins, np.nan)
    counts = np.zeros(nbins, dtype=int)
    inds = np.digitize(D, bins) - 1
    for k in range(nbins):
        mask = (inds == k)
        if np.any(mask):
            gamma[k] = 0.5 * np.mean(sqdiff[mask])
            counts[k] = mask.sum()
    return centers, gamma, counts

def empirical_variogram_coords(locs, nbins=50, max_dist=None):
    """
    Experimental variogram computed directly from coordinates (not usually used;
    included only for compatibility with previous code).
    Uses pairwise squared distances as 'differences'.
    """
    D = pdist(locs)
    sq = D**2
    if max_dist is None:
        max_dist = MAX_DIST_FACTOR * D.max()
    bins = np.linspace(0.0, max_dist, nbins+1)
    centers = 0.5*(bins[:-1] + bins[1:])
    gamma = np.full(nbins, np.nan)
    counts = np.zeros(nbins, dtype=int)
    inds = np.digitize(D, bins) - 1
    for k in range(nbins):
        mask = (inds == k)
        if np.any(mask):
            gamma[k] = 0.5 * np.mean(sq[mask])
            counts[k] = mask.sum()
    return centers, gamma, counts

def cross_variogram_from_matches(C, M, construct_idx, mandel_idx, nbins=50, max_dist=None):
    """
    Cross-variogram defined from matched pairs.
    We compute the displacement vector d_i = C[ci] - M[mi], treat its magnitude as the 'lag'
    and store semivariance = 0.5 * |d_i|^2 in bins over |d_i|.
    This mirrors the earlier script's behaviour for matched-pair cross-plotting.
    """
    if len(construct_idx) == 0:
        return np.array([]), np.array([]), np.array([])
    diffs = C[construct_idx] - M[mandel_idx]
    mags = np.linalg.norm(diffs, axis=1)
    sq = np.sum(diffs**2, axis=1)
    if max_dist is None:
        max_dist = mags.max() if mags.size>0 else 1.0
    bins = np.linspace(0.0, max_dist, nbins+1)
    centers = 0.5*(bins[:-1] + bins[1:])
    gamma = np.full(nbins, np.nan)
    counts = np.zeros(nbins, dtype=int)
    inds = np.digitize(mags, bins) - 1
    for k in range(nbins):
        mask = (inds == k)
        if np.any(mask):
            gamma[k] = 0.5 * np.mean(sq[mask])
            counts[k] = mask.sum()
    return centers, gamma, counts

# -------------------------
# Main script
# -------------------------
def main():
    # load points
    try:
        C = load_points("construct_points.csv")
        M = load_points("mandel_boundary_sample.csv")
    except Exception as e:
        print("Error loading points:", e)
        sys.exit(1)

    # load matches robustly
    try:
        construct_idx, mandel_idx = load_matches("matches_indices.csv", nC=C.shape[0], nM=M.shape[0])
    except Exception as e:
        print("Error loading matches_indices.csv:", e)
        print("If your file contains headers or text, use a plain numeric CSV without header.")
        sys.exit(1)

    if VERBOSE:
        print(f"Loaded {C.shape[0]} Construct points, {M.shape[0]} Mandelbrot points")
        print(f"Interpreted matches arrays of length {construct_idx.size}")
        print("Sample (first 8):")
        for a,b in zip(construct_idx[:8], mandel_idx[:8]):
            print(f"  C[{a}]  <->  M[{b}]")

    # 1) variogram of Construct matching distances (we need a scalar field per Construct point)
    # compute distances between each Construct point and its matched Mandelbrot point
    if construct_idx.size != mandel_idx.size:
        print("Warning: construct_idx and mandel_idx differ in length.")

    # ensure indices are in valid range
    if np.any(mandel_idx < 0) or np.any(mandel_idx >= M.shape[0]):
        raise IndexError("Some mandel indices are out of bounds; check matches_indices.csv")
    if np.any(construct_idx < 0) or np.any(construct_idx >= C.shape[0]):
        raise IndexError("Some construct indices are out of bounds; check matches_indices.csv")

    matched_C = C[construct_idx]
    matched_M = M[mandel_idx]
    matching_distances = np.linalg.norm(matched_C - matched_M, axis=1)

    # experimental variogram of the scalar field matching_distances at Construct locations
    lags_d, gamma_d, counts_d = empirical_variogram_field(matched_C, matching_distances, nbins=NBINS)
    # also variogram directly for coords (if desired)
    lags_c, gamma_c, counts_c = empirical_variogram_coords(C, nbins=NBINS)

    # cross-variogram / matched-pair semivariance
    lags_x, gamma_x, counts_x = cross_variogram_from_matches(C, M, construct_idx, mandel_idx, nbins=NBINS)

    # Save CSVs
    np.savetxt("variogram_construct_field.csv", np.c_[lags_d, gamma_d, counts_d], delimiter=",", header="lag,gamma,count", comments="")
    np.savetxt("variogram_construct_coords.csv", np.c_[lags_c, gamma_c, counts_c], delimiter=",", header="lag,gamma,count", comments="")
    np.savetxt("crossvariogram_CM.csv", np.c_[lags_x, gamma_x, counts_x], delimiter=",", header="lag,gamma,count", comments="")
    if VERBOSE:
        print("Saved variogram_construct_field.csv, variogram_construct_coords.csv, crossvariogram_CM.csv")

    # Plots
    plt.figure(figsize=(8,6))
    if lags_d.size>0:
        plt.plot(lags_d, gamma_d, 'o-', label="Construct variogram (matching distances)")
    if lags_c.size>0:
        plt.plot(lags_c, gamma_c, 's-', label="Construct coords variogram")
    plt.xlabel("lag distance h")
    plt.ylabel("γ(h)")
    plt.title("Construct variograms")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("variogram_construct_combined.png", dpi=180)
    if VERBOSE: print("Saved variogram_construct_combined.png")

    plt.figure(figsize=(8,6))
    if lags_x.size>0:
        plt.plot(lags_x, gamma_x, 'd-', label="Cross-variogram (matched pairs)")
    plt.xlabel("lag distance h (matched-pair magnitude)")
    plt.ylabel("γ(h)")
    plt.title("Cross-variogram Construct–Mandelbrot")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("crossvariogram_CM.png", dpi=180)
    if VERBOSE: print("Saved crossvariogram_CM.png")

    plt.show()

if __name__ == "__main__":
    main()
