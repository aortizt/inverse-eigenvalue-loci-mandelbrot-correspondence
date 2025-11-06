import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

# --------------------------
# Helper functions
# --------------------------

def pair_correlation(points, r_max, dr):
    """
    Compute pair correlation function g(r) for 2D points.
    """
    N = len(points)
    area = (np.max(points[:,0]) - np.min(points[:,0])) * (np.max(points[:,1]) - np.min(points[:,1]))
    rho = N / area  # density
    dists = distance_matrix(points, points)
    dists = dists[np.triu_indices(N, k=1)]  # upper triangle only
    
    r_vals = np.arange(0, r_max, dr)
    g_r = []
    
    for r in r_vals:
        shell = (dists >= r) & (dists < r+dr)
        count = np.sum(shell)
        norm = 2 * np.pi * r * dr * N * rho
        g_r.append(count / norm if norm > 0 else 0)
    
    return r_vals, np.array(g_r)

def ripley_K(points, r_max, dr):
    """
    Compute Ripley's K function for 2D points.
    """
    N = len(points)
    area = (np.max(points[:,0]) - np.min(points[:,0])) * (np.max(points[:,1]) - np.min(points[:,1]))
    rho = N / area
    dists = distance_matrix(points, points)
    dists = dists[np.triu_indices(N, k=1)]
    
    r_vals = np.arange(0, r_max, dr)
    K_r = []
    
    for r in r_vals:
        count = np.sum(dists < r)
        K_r.append((2 * count) / (N * rho))
    
    return r_vals, np.array(K_r)

# --------------------------
# Load data
# --------------------------

base_path = "/home/Merlin/Desktop/out_clean/"
C = np.loadtxt(base_path + "construct_aligned.csv", delimiter=",")
M = np.loadtxt(base_path + "mandel_boundary_sample.csv", delimiter=",")

# --------------------------
# Analysis parameters
# --------------------------
r_max = 1.5   # max distance to probe
dr = 0.05     # radial bin width

# --------------------------
# Compute statistics
# --------------------------
r_C, g_C = pair_correlation(C, r_max, dr)
r_M, g_M = pair_correlation(M, r_max, dr)

rC_K, KC = ripley_K(C, r_max, dr)
rM_K, KM = ripley_K(M, r_max, dr)

# --------------------------
# Plot results
# --------------------------

plt.figure(figsize=(12,5))

# Pair correlation
plt.subplot(1,2,1)
plt.plot(r_C, g_C, label="Construct (aligned)", color='cyan')
plt.plot(r_M, g_M, label="Mandel boundary", color='red')
plt.axhline(1.0, color='gray', linestyle='--')
plt.xlabel("r")
plt.ylabel("g(r)")
plt.title("Pair Correlation Function")
plt.legend()

# Ripley's K
plt.subplot(1,2,2)
plt.plot(rC_K, KC, label="Construct (aligned)", color='cyan')
plt.plot(rM_K, KM, label="Mandel boundary", color='red')
plt.xlabel("r")
plt.ylabel("K(r)")
plt.title("Ripley's K Function")
plt.legend()

plt.tight_layout()
plt.show()

