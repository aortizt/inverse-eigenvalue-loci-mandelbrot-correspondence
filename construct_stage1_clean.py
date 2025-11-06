#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
construct_stage1_clean.py
Minimal, safe Stage 1 pipeline:
 - builds The Construct (inverse eigenvalues of Lucas companion matrices)
 - computes a simple Mandelbrot distance-estimator sampling
 - computes a simple local orientation feature (PCA) for points
 - runs a Sinkhorn OT (if POT is available) or falls back to a simple nearest-neighbor pairing
 - aligns via Procrustes
This script is intentionally conservative to avoid syntax/encoding issues.
"""
import os
import sys
import math
import numpy as np
from scipy.linalg import eigvals
import matplotlib.pyplot as plt

# Optional imports
try:
    from sklearn.neighbors import NearestNeighbors
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    import ot
    OT_AVAILABLE = True
except Exception:
    OT_AVAILABLE = False

def construct_points(maxN=40):
    pts = []
    for n in range(2, maxN + 1):
        A = np.zeros((n, n), dtype=float)
        A[0, :] = 1.0
        if n >= 2:
            A[1, 0] = 1.0
        for j in range(1, n - 1):
            A[j + 1, j] = 1.0
        ev = eigvals(A)
        ev_nonzero = ev[np.abs(ev) > 1e-12]
        inv = 1.0 / ev_nonzero
        for v in inv:
            pts.append([float(v.real), float(v.imag)])
    return np.array(pts, dtype=float)

def mandelbrot_distance_estimator(c, max_iter=200, bailout=1e6):
    z = 0 + 0j
    dz = 0 + 0j
    for _ in range(int(max_iter)):
        dz = 2.0 * z * dz + 1.0
        z = z * z + c
        if abs(z) > bailout:
            return abs(z) * math.log(abs(z)) / max(abs(dz), 1e-16)
    return 0.0

def sample_mandelbrot_boundary(nx=120, ny=80, max_iter=200, threshold_low=1e-6, threshold_high=1e-1, nsamples=800):
    xs = np.linspace(-2.25, 1.25, nx)
    ys = np.linspace(-1.25, 1.25, ny)
    cand = []
    vals = []
    for y in ys:
        for x in xs:
            c = complex(x, y)
            d = mandelbrot_distance_estimator(c, max_iter=max_iter)
            if d > threshold_low and d < threshold_high:
                cand.append([x, y])
                vals.append(d)
    cand = np.array(cand, dtype=float)
    vals = np.array(vals, dtype=float)
    if cand.size == 0:
        return np.empty((0, 2), dtype=float)
    if len(cand) <= nsamples:
        return cand
    probs = vals / np.sum(vals)
    idx = np.random.choice(len(cand), size=nsamples, replace=False, p=probs)
    return cand[idx]

def compute_orientation_features(X, k=8):
    N = X.shape[0]
    if N == 0:
        return np.zeros((0,2), dtype=float)
    if SKLEARN_AVAILABLE:
        nbr = NearestNeighbors(n_neighbors=min(k, N)).fit(X)
        _, idxs = nbr.kneighbors(X)
    else:
        idxs = np.zeros((N, min(k, N)), dtype=int)
        for i in range(N):
            d = np.sum((X - X[i])**2, axis=1)
            order = np.argsort(d)
            take = order[1:1+min(k, N)]
            idxs[i,:len(take)] = take
    orientations = np.zeros((N,2), dtype=float)
    for i in range(N):
        neighbors = X[idxs[i]]
        M = neighbors - np.mean(neighbors, axis=0)
        if M.shape[0] >= 2:
            C = np.dot(M.T, M)
            vals, vecs = np.linalg.eigh(C)
            v = vecs[:, np.argmax(vals)]
            orientations[i,0] = float(v[0])
            orientations[i,1] = float(v[1])
        else:
            orientations[i,:] = np.array([1.0, 0.0])
    return orientations

def sinkhorn_transport(XA, XB, reg=1e-2):
    if not OT_AVAILABLE:
        return None, None
    a = np.ones(XA.shape[0]) / float(XA.shape[0])
    b = np.ones(XB.shape[0]) / float(XB.shape[0])
    M = ot.dist(XA, XB, metric='euclidean')
    G = ot.sinkhorn(a, b, M, reg)
    return G, M

def greedy_match(XA, XB):
    try:
        from sklearn.neighbors import KDTree
        tree = KDTree(XB)
        d, idx = tree.query(XA, k=1)
        return idx.flatten(), d.flatten()
    except Exception:
        idxs = []
        dists = []
        for a in XA:
            d = np.sum((XB - a)**2, axis=1)
            j = int(np.argmin(d))
            idxs.append(j)
            dists.append(math.sqrt(float(d[j])))
        return np.array(idxs, dtype=int), np.array(dists, dtype=float)

def procrustes_align(X, Y, matches):
    A = np.array(X, dtype=float)
    B = Y[np.array(matches, dtype=int)]
    muA = np.mean(A, axis=0)
    muB = np.mean(B, axis=0)
    A0 = A - muA
    B0 = B - muB
    U, s, Vt = np.linalg.svd(B0.T.dot(A0))
    R = U.dot(Vt)
    A_aligned = A0.dot(R.T) + muB
    return A_aligned, R

def run_pipeline(outdir="out_clean", maxN=40, nx=120, ny=80, boundary_samples=600, use_sinkhorn=True):
    np.random.seed(0)
    os.makedirs(outdir, exist_ok=True)

    print("Building Construct (inverse eigenvalues)...")
    C = construct_points(maxN=maxN)
    print("Construct points:", C.shape)

    print("Sampling Mandelbrot boundary via DE grid...")
    M = sample_mandelbrot_boundary(nx=nx, ny=ny, nsamples=boundary_samples)
    print("Sampled boundary points:", M.shape)

    print("Computing orientation features...")
    F_C = compute_orientation_features(C, k=8)
    F_M = compute_orientation_features(M, k=8)

    XA = np.hstack([F_C, C])
    XB = np.hstack([F_M, M])

    print("Matching sets...")
    if use_sinkhorn and OT_AVAILABLE:
        G, costM = sinkhorn_transport(XA, XB, reg=1e-2)
        matches = np.argmax(G, axis=1)
        print("Used Sinkhorn transport.")
    else:
        matches, dists = greedy_match(XA, XB)
        print("Used greedy nearest neighbor matching.")

    print("Aligning Construct to Mandelbrot sample...")
    C_aligned, R = procrustes_align(C, M, matches)

    np.savetxt(os.path.join(outdir, "construct_points.csv"), C, delimiter=",")
    np.savetxt(os.path.join(outdir, "mandel_boundary_sample.csv"), M, delimiter=",")
    np.savetxt(os.path.join(outdir, "construct_aligned.csv"), C_aligned, delimiter=",")
    np.savetxt(os.path.join(outdir, "matches_indices.csv"), matches, delimiter=",", fmt="%d")

    plt.figure(figsize=(8,6))
    if M.shape[0] > 0:
        plt.scatter(M[:,0], M[:,1], s=6, c='red', label='Mandel sample')
    if C.shape[0] > 0:
        plt.scatter(C[:,0], C[:,1], s=6, c='blue', alpha=0.6, label='Construct')
    if C_aligned.shape[0] > 0:
        plt.scatter(C_aligned[:,0], C_aligned[:,1], s=6, c='cyan', alpha=0.65, label='Construct aligned')
    plt.legend(); plt.axis('equal')
    plt.title("Construct vs Mandelbrot (aligned)")
    plt.savefig(os.path.join(outdir, "alignment.png"), dpi=200)
    plt.close()
    print("Outputs saved in:", outdir)
    return {"C": C, "M": M, "C_aligned": C_aligned, "matches": matches}

if __name__ == "__main__":
    run_pipeline()
