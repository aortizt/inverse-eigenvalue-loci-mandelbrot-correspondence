#!/usr/bin/env python3
"""
multifractal_phase6.py

Box-counting based multifractal analysis for two point clouds:
 - construct_points.csv
 - mandel_boundary_sample.csv

Outputs:
 - multifractal_construct.csv, multifractal_mandel.csv (q, D_q, tau, alpha, f_alpha)
 - plots: Dq_compare.png, falpha_compare.png

Author: assistant
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from math import isclose

# -------------------------
# Parameters (tweakable)
# -------------------------
construct_file = "construct_points.csv"
mandel_file = "mandel_boundary_sample.csv"

q_vals = np.concatenate((np.linspace(-5, -1, 5), np.linspace(-0.8, 0.8, 9), np.linspace(1, 5, 5)))
# remove exactly q=1 from q_vals because D1 requires special handling
q_vals = np.array([q for q in q_vals if not isclose(q, 1.0)])
scales = np.logspace(np.log10(0.002), np.log10(0.5), 12)  # box sizes (in same units as points); adjust if needed
min_count_boxes = 5  # minimum boxes occupied to consider scale valid

# -------------------------
# Utilities
# -------------------------
def load_points(fname):
    pts = np.loadtxt(fname, delimiter=",")
    if pts.ndim == 1:
        pts = pts.reshape((-1,2))
    return pts[:, :2].astype(float)

def box_partition_counts(points, eps):
    """
    Partition plane into grid of boxes of size eps, return counts per non-empty box and box indices.
    """
    xs = points[:,0]
    ys = points[:,1]
    xmin, ymin = xs.min(), ys.min()
    # shift so boxes start at grid aligned position for stability (use floor)
    ix = np.floor((xs - xmin)/eps).astype(int)
    iy = np.floor((ys - ymin)/eps).astype(int)
    # combine to single index
    keys = ix.astype(np.int64) * (10**9) + iy.astype(np.int64)  # unique pairing (works for moderate sizes)
    # group counts
    unique_keys, counts = np.unique(keys, return_counts=True)
    return counts

def partition_probabilities(points, eps):
    counts = box_partition_counts(points, eps)
    total = counts.sum()
    ps = counts / total
    return ps

def compute_Zq(ps, q):
    if q == 0:
        return ps.size  # number of occupied boxes
    else:
        return np.sum(ps**q)

# -------------------------
# Multifractal estimation function
# -------------------------
def multifractal_spectrum(points, q_values, scales_list):
    """
    For a given set of points, compute Z(q,eps) for eps in scales_list and q in q_values.
    Return arrays of tau(q), Dq(q) (via linear regression on log-log), and estimated alpha,f(alpha) via Legendre transform.
    """
    # compute Z matrix: shape (len(q), len(eps))
    Z = np.zeros((len(q_values), len(scales_list)))
    valid_mask = np.zeros(len(scales_list), dtype=bool)
    for j, eps in enumerate(scales_list):
        ps = partition_probabilities(points, eps)
        if len(ps) < min_count_boxes:
            Z[:, j] = np.nan
            continue
        valid_mask[j] = True
        for i, q in enumerate(q_values):
            Z[i, j] = compute_Zq(ps, q)

    # For each q, do linear fit log Z vs log eps (use only valid scales)
    log_eps = np.log(scales_list[valid_mask])
    tau = np.full(len(q_values), np.nan)
    Dq = np.full(len(q_values), np.nan)

    for i, q in enumerate(q_values):
        y = np.log(Z[i, valid_mask])
        if np.any(np.isfinite(y)):
            # linear regression
            A = np.vstack([log_eps, np.ones_like(log_eps)]).T
            m, c = np.linalg.lstsq(A, y, rcond=None)[0]
            # tau(q) = slope of log Z vs log eps (note sign: Z ~ eps^{tau})
            tau_i = m
            Dq_i = tau_i/(q-1) if not isclose(q,1.0) else np.nan
            tau[i] = tau_i
            Dq[i] = Dq_i

    # estimate alpha and f(alpha) via Legendre transform: alpha = d tau / dq, f = q alpha - tau
    # numerical differentiation
    dq = np.gradient(q_values)
    dtau_dq = np.gradient(tau, q_values, edge_order=2)
    alpha = dtau_dq
    f_alpha = q_values * alpha - tau

    # return structured array
    return {
        "q": q_values,
        "tau": tau,
        "Dq": Dq,
        "alpha": alpha,
        "f_alpha": f_alpha,
        "scales": scales_list,
        "Z": Z
    }

# -------------------------
# Main run
# -------------------------
if __name__ == "__main__":
    if not os.path.exists(construct_file) or not os.path.exists(mandel_file):
        print("Missing input CSVs. Please place construct_points.csv and mandel_boundary_sample.csv in current folder.")
        raise SystemExit(1)

    C = load_points(construct_file)
    M = load_points(mandel_file)
    print("Loaded", C.shape, "Construct points,", M.shape, "Mandelbrot points")

    resC = multifractal_spectrum(C, q_vals, scales)
    resM = multifractal_spectrum(M, q_vals, scales)

    # Save Dq and tau and alpha-f
    def save_results(res, prefix):
        out = np.column_stack((res["q"], res["tau"], res["Dq"], res["alpha"], res["f_alpha"]))
        header = "q,tau,Dq,alpha,f_alpha"
        np.savetxt(f"{prefix}_multifractal.csv", out, delimiter=",", header=header, comments="")
        print("Saved", f"{prefix}_multifractal.csv")

    save_results(resC, "construct")
    save_results(resM, "mandel")

    # Plot Dq comparison
    plt.figure(figsize=(8,5))
    plt.plot(resC["q"], resC["Dq"], "o-", label="Construct D(q)")
    plt.plot(resM["q"], resM["Dq"], "s-", label="Mandel D(q)")
    plt.xlabel("q")
    plt.ylabel("D(q)")
    plt.legend()
    plt.grid(True)
    plt.title("Generalized dimensions D(q)")
    plt.savefig("Dq_compare.png", dpi=200)
    plt.show()

    # Plot f(alpha)
    plt.figure(figsize=(8,5))
    plt.plot(resC["alpha"], resC["f_alpha"], "o-", label="Construct f(α)")
    plt.plot(resM["alpha"], resM["f_alpha"], "s-", label="Mandel f(α)")
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$f(\alpha)$")
    plt.legend()
    plt.grid(True)
    plt.title("Singularity spectrum")
    plt.savefig("falpha_compare.png", dpi=200)
    plt.show()

    print("Done.")
