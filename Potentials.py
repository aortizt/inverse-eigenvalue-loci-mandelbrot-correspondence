import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# ------------------------------------------------------------
# Load Construct and Mandelbrot boundary points
# ------------------------------------------------------------
C = np.loadtxt("construct_points.csv", delimiter=",")   # Nx2
M = np.loadtxt("mandel_boundary_sample.csv", delimiter=",")   # Mx2

C = C[:, :2]  # ensure 2D
M = M[:, :2]

print("Loaded Construct:", C.shape, "Mandelbrot:", M.shape)

# ------------------------------------------------------------
# Logarithmic potential of Construct
# ------------------------------------------------------------
def log_potential(points, grid_x, grid_y):
    """Compute U(z) = (1/N) sum log|z - p| over grid."""
    U = np.zeros((len(grid_y), len(grid_x)))
    for p in points:
        dx = grid_x[None, :] - p[0]
        dy = grid_y[:, None] - p[1]
        dist = np.sqrt(dx**2 + dy**2)
        U += np.log(dist + 1e-12)  # avoid log(0)
    return U / len(points)

# ------------------------------------------------------------
# Escape potential of Mandelbrot
# ------------------------------------------------------------
def escape_potential(grid_x, grid_y, max_iter=200, R=10):
    """Compute escape-rate potential for Mandelbrot set."""
    U = np.zeros((len(grid_y), len(grid_x)))
    for iy, y in enumerate(grid_y):
        for ix, x in enumerate(grid_x):
            c = x + 1j*y
            z = 0 + 0j
            for k in range(max_iter):
                z = z*z + c
                if abs(z) > R:
                    break
            if abs(z) > 0:
                U[iy, ix] = np.log(abs(z)) / (2**k)
            else:
                U[iy, ix] = 0
    return U

# ------------------------------------------------------------
# Define grid
# ------------------------------------------------------------
grid_x = np.linspace(-2, 2, 400)
grid_y = np.linspace(-2, 2, 400)

# ------------------------------------------------------------
# Compute potentials
# ------------------------------------------------------------
U_C = log_potential(C, grid_x, grid_y)
U_M = escape_potential(grid_x, grid_y, max_iter=300)

# Difference map
U_diff = U_C - U_M

# ------------------------------------------------------------
# Global correlation coefficient
# ------------------------------------------------------------
U_C_flat = U_C.flatten()
U_M_flat = U_M.flatten()
mask = ~(np.isnan(U_C_flat) | np.isnan(U_M_flat))

corr, pval = pearsonr(U_C_flat[mask], U_M_flat[mask])
print(f"Global Pearson correlation: r = {corr:.4f}, p = {pval:.2e}")

# ------------------------------------------------------------
# Local correlation map (sliding window)
# ------------------------------------------------------------
def local_correlation(U1, U2, win=20):
    """
    Compute local Pearson correlation map in sliding windows.
    win: half window size in pixels.
    """
    ny, nx = U1.shape
    corr_map = np.full((ny, nx), np.nan)
    for iy in range(win, ny - win):
        for ix in range(win, nx - win):
            a = U1[iy - win:iy + win, ix - win:ix + win].flatten()
            b = U2[iy - win:iy + win, ix - win:ix + win].flatten()
            if np.all(np.isnan(a)) or np.all(np.isnan(b)):
                continue
            mask = ~(np.isnan(a) | np.isnan(b))
            if np.sum(mask) > 5:
                corr_map[iy, ix] = pearsonr(a[mask], b[mask])[0]
    return corr_map

U_corrmap = local_correlation(U_C, U_M, win=15)

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
fig, axs = plt.subplots(1, 4, figsize=(22, 5))

im0 = axs[0].imshow(U_C, extent=[grid_x[0], grid_x[-1], grid_y[0], grid_y[-1]],
                    origin="lower", cmap="viridis")
axs[0].set_title("Logarithmic Potential (Construct)")
plt.colorbar(im0, ax=axs[0])

im1 = axs[1].imshow(U_M, extent=[grid_x[0], grid_x[-1], grid_y[0], grid_y[-1]],
                    origin="lower", cmap="inferno")
axs[1].set_title("Escape Potential (Mandelbrot)")
plt.colorbar(im1, ax=axs[1])

im2 = axs[2].imshow(U_diff, extent=[grid_x[0], grid_x[-1], grid_y[0], grid_y[-1]],
                    origin="lower", cmap="coolwarm",
                    vmin=-np.max(abs(U_diff)), vmax=np.max(abs(U_diff)))
axs[2].set_title("Difference (Construct - Mandelbrot)")
plt.colorbar(im2, ax=axs[2])

im3 = axs[3].imshow(U_corrmap, extent=[grid_x[0], grid_x[-1], grid_y[0], grid_y[-1]],
                    origin="lower", cmap="RdYlGn", vmin=-1, vmax=1)
axs[3].set_title("Local Correlation Map")
plt.colorbar(im3, ax=axs[3])

plt.tight_layout()
plt.savefig("potential_comparison_with_corrmap.png", dpi=200)
plt.show()
