#!/usr/bin/env python3
"""
dynamical_embeddings_phase7.py

Build kernel/Markov embeddings (diffusion map style) for Construct and Mandelbrot point clouds.
Outputs:
 - eigenvalues_construct.csv, eigenvectors_construct.npy
 - eigenvalues_mandel.csv, eigenvectors_mandel.npy
 - spectra_compare.png, embedding_compare.png
 - spectral_distance.txt

Author: assistant
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
import os

# -------------------------
# Parameters
# -------------------------
construct_file = "construct_points.csv"
mandel_file = "mandel_boundary_sample.csv"

k_nn = 20           # number of neighbors for sparse kernel
n_eigs = 8          # number of eigenvectors/eigenvalues to compute (largest magnitude)
eps_scale = 0.5     # multiplier for kernel bandwidth (sigma) relative to median distance
normalize_markov = True  # row-normalize kernel to obtain Markov matrix
symmetric_normalization = True

# -------------------------
# Utilities
# -------------------------
def load_points(fname):
    pts = np.loadtxt(fname, delimiter=",")
    if pts.ndim == 1:
        pts = pts.reshape((-1,2))
    return pts[:, :2].astype(float)

def build_sparse_kernel(points, k=20, eps_scale_local=1.0):
    """
    Build a sparse Gaussian kernel matrix using k-NN for each point.
    Returns sparse csr_matrix K (n x n) (symmetric).
    """
    n = points.shape[0]
    tree = cKDTree(points)
    dists, idxs = tree.query(points, k=k+1)  # includes self at distance 0
    # remove self (first col)
    dists = dists[:,1:]
    idxs = idxs[:,1:]
    # choose bandwidth sigma as median of nonzero distances times eps_scale_local
    sigma = np.median(dists.ravel()) * eps_scale_local
    if sigma <= 0:
        sigma = 1.0
    rows = []
    cols = []
    data = []
    for i in range(n):
        for j_idx, d in zip(idxs[i], dists[i]):
            rows.append(i)
            cols.append(j_idx)
            data.append(np.exp(-(d**2)/(2*sigma**2)))
    # assemble sparse, then symmetrize
    K = csr_matrix((data, (rows, cols)), shape=(n,n))
    K = 0.5*(K + K.T)
    return K, sigma

def markov_from_kernel(K):
    """Row-normalize kernel to make Markov transition matrix P."""
    row_sum = np.array(K.sum(axis=1)).flatten()
    inv = np.reciprocal(row_sum, where=row_sum!=0)
    D_inv = csr_matrix((inv, (np.arange(len(inv)), np.arange(len(inv)))), shape=(len(inv), len(inv)))
    P = D_inv.dot(K)
    return P

def spectral_embedding(P, neigs=8):
    """
    Compute leading eigenpairs of P (largest magnitude). Use eigsh on symmetric case or on P if okay.
    If P is non-symmetric, compute eigenpairs of symmetrized version S = (P + P.T)/2 for stability (approx).
    Returns eigenvalues (sorted descending) and eigenvectors (columns).
    """
    # symmetrize for safety
    S = 0.5*(P + P.T)
    # convert to csr
    S = S.tocsr()
    # compute neigs largest eigenvalues
    try:
        vals, vecs = eigsh(S, k=min(neigs, S.shape[0]-2), which='LM')
    except Exception as e:
        # fallback: dense
        from numpy.linalg import eigh
        Sdense = S.toarray()
        vals_all, vecs_all = eigh(Sdense)
        vals = vals_all[::-1][:neigs]
        vecs = vecs_all[:, ::-1][:, :neigs]
    # sort descending
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    return vals, vecs

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    if not os.path.exists(construct_file) or not os.path.exists(mandel_file):
        print("Missing inputs. Place construct_points.csv and mandel_boundary_sample.csv in current folder.")
        raise SystemExit(1)

    C = load_points(construct_file)
    M = load_points(mandel_file)
    print("Loaded", C.shape, "Construct points,", M.shape, "Mandelbrot points")

    print("Building kernel for Construct...")
    Kc, sigma_c = build_sparse_kernel(C, k=k_nn, eps_scale_local=eps_scale)
    Pc = markov_from_kernel(Kc)
    print("Sigma (Construct) ~", sigma_c)

    print("Building kernel for Mandelbrot...")
    Km, sigma_m = build_sparse_kernel(M, k=k_nn, eps_scale_local=eps_scale)
    Pm = markov_from_kernel(Km)
    print("Sigma (Mandel) ~", sigma_m)

    # spectral embeddings
    print("Computing spectrum Construct...")
    vals_c, vecs_c = spectral_embedding(Pc, neigs=n_eigs)
    print("Computing spectrum Mandel...")
    vals_m, vecs_m = spectral_embedding(Pm, neigs=n_eigs)

    # save results
    np.savetxt("eigenvalues_construct.csv", np.column_stack((np.arange(1,len(vals_c)+1), vals_c)), delimiter=",", header="idx,lambda")
    np.save("eigenvectors_construct.npy", vecs_c)
    np.savetxt("eigenvalues_mandel.csv", np.column_stack((np.arange(1,len(vals_m)+1), vals_m)), delimiter=",", header="idx,lambda")
    np.save("eigenvectors_mandel.npy", vecs_m)
    print("Saved eigenvalues/eigenvectors files.")

    # spectral comparison plot: eigenvalue decay
    plt.figure(figsize=(6,4))
    plt.plot(np.arange(1,len(vals_c)+1), vals_c, 'o-', label="Construct")
    plt.plot(np.arange(1,len(vals_m)+1), vals_m, 's-', label="Mandelbrot")
    plt.xlabel("Mode index")
    plt.ylabel("Eigenvalue (symmetrized kernel)")
    plt.title("Spectrum (leading eigenvalues)")
    plt.legend()
    plt.grid(True)
    plt.savefig("spectra_compare.png", dpi=200)
    plt.show()

    # embedding: plot first 2 nontrivial eigenvectors (skip first trivial constant)
    def plot_embedding(vecs, pts, prefix):
        # choose components 1 and 2 if present (0 is largest)
        if vecs.shape[1] < 3:
            comps = (0,1)
        else:
            comps = (1,2)
        x = vecs[:, comps[0]]
        y = vecs[:, comps[1]]
        plt.figure(figsize=(6,6))
        plt.scatter(pts[:,0], pts[:,1], s=6, c=x, cmap='Spectral', alpha=0.8)
        plt.title(f"{prefix} embedding (colored by eigenvector {comps[0]})")
        plt.colorbar()
        plt.savefig(f"{prefix}_embedding_vec{comps[0]}.png", dpi=200)
        plt.show()

    plot_embedding(vecs_c, C, "construct")
    plot_embedding(vecs_m, M, "mandel")

    # compute simple spectral distance: L2 between vector of leading eigenvalues (pad to equal length)
    L = min(len(vals_c), len(vals_m))
    spec_dist = np.linalg.norm(vals_c[:L] - vals_m[:L])
    with open("spectral_distance.txt", "w") as f:
        f.write(f"spectral_distance_norm = {spec_dist}\n")
    print("Spectral distance (L2 on leading eigenvalues):", spec_dist)
    print("Done.")
