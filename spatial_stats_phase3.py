import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import directed_hausdorff

# --- Load data ---
C = np.loadtxt("/home/Merlin/Desktop/out_clean/construct_aligned.csv", delimiter=",")
M = np.loadtxt("/home/Merlin/Desktop/out_clean/mandel_boundary_sample.csv", delimiter=",")
matches = np.loadtxt("/home/Merlin/Desktop/out_clean/matches_indices.csv", delimiter=",", dtype=int)

# --- 1. Hausdorff distance ---
h1 = directed_hausdorff(C, M)[0]
h2 = directed_hausdorff(M, C)[0]
hausdorff_dist = max(h1, h2)

print("Hausdorff distance between Construct and Mandelbrot:", hausdorff_dist)

# --- 2. Curvature estimation ---
def estimate_curvature(points):
    # Assumes points ordered around boundary
    dx = np.gradient(points[:,0])
    dy = np.gradient(points[:,1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
    return curvature

curv_C = estimate_curvature(C)
curv_M = estimate_curvature(M)

plt.figure(figsize=(10,5))
plt.hist(curv_C, bins=100, alpha=0.5, label="Construct curvature")
plt.hist(curv_M, bins=100, alpha=0.5, label="Mandelbrot curvature")
plt.yscale("log")
plt.xlabel("Curvature")
plt.ylabel("Frequency (log scale)")
plt.title("Curvature distribution")
plt.legend()
plt.show()

# --- 3. Fractal dimension (box counting) ---
def fractal_dimension(points, scales=None):
    if scales is None:
        scales = np.logspace(-2, 0, 10, base=10.0)  # relative box sizes
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    rng = maxs - mins
    N = []
    for s in scales:
        step = rng * s
        # Map points into grid
        grid = np.floor((points - mins) / step).astype(int)
        # Count unique boxes
        N.append(len(np.unique(grid, axis=0)))
    coeffs = np.polyfit(np.log(1/scales), np.log(N), 1)
    return coeffs[0], (np.log(1/scales), np.log(N))

fd_C, logplot_C = fractal_dimension(C)
fd_M, logplot_M = fractal_dimension(M)

print("Fractal dimension (Construct):", fd_C)
print("Fractal dimension (Mandelbrot):", fd_M)

plt.figure(figsize=(6,5))
plt.plot(*logplot_C, "o-", label=f"Construct (D≈{fd_C:.2f})")
plt.plot(*logplot_M, "o-", label=f"Mandelbrot (D≈{fd_M:.2f})")
plt.xlabel("log(1/box size)")
plt.ylabel("log(N boxes)")
plt.title("Box-counting fractal dimension")
plt.legend()
plt.show()
