import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Load data
# ============================================================
C = np.loadtxt("construct_points.csv", delimiter=",")   # Construct points Nx2
M = np.loadtxt("mandel_boundary_sample.csv", delimiter=",")  # Mandelbrot sample (optional)

print(f"Loaded {C.shape[0]} Construct points, {M.shape[0]} Mandelbrot points")

# ============================================================
# Potential functions
# ============================================================

def construct_potential(Zx, Zy, C):
    """
    Logarithmic potential of Construct eigenvalue inverses.
    U_C(z) = -(1/N) * sum log |z - c_i|
    """
    U = np.zeros_like(Zx, dtype=float)
    for (cx, cy) in C:
        dz = np.sqrt((Zx - cx)**2 + (Zy - cy)**2)
        U -= np.log(dz + 1e-12) / len(C)   # +epsilon to avoid log(0)
    return U

def mandelbrot_potential(Zx, Zy, max_iter=200, R=2.0):
    """
    Escape-rate potential for Mandelbrot set.
    U_M(c) = (1/n_iter) * log|z_n|, with z_{n+1} = z_n^2 + c
    """
    nx, ny = Zx.shape
    U = np.zeros((nx, ny), dtype=float)
    for i in range(nx):
        for j in range(ny):
            c = Zx[i,j] + 1j*Zy[i,j]
            z = 0+0j
            for k in range(max_iter):
                z = z*z + c
                if abs(z) > R:
                    U[i,j] = np.log(abs(z)) / (k+1)
                    break
    return U

# ============================================================
# Numerical Laplacian (finite differences)
# ============================================================

def laplacian(U, h):
    """
    Compute Laplacian ΔU with central finite differences.
    h = grid spacing
    """
    lap = (
        -4*U
        + np.roll(U,1,axis=0) + np.roll(U,-1,axis=0)
        + np.roll(U,1,axis=1) + np.roll(U,-1,axis=1)
    ) / h**2
    return lap

# ============================================================
# Grid setup
# ============================================================
Ngrid = 200
x = np.linspace(-2, 2, Ngrid)
y = np.linspace(-2, 2, Ngrid)
X, Y = np.meshgrid(x, y)

print("Computing Construct potential...")
U_C = construct_potential(X, Y, C)

print("Computing Mandelbrot potential (escape rate)...")
U_M = mandelbrot_potential(X, Y, max_iter=200)

# ============================================================
# Laplacians
# ============================================================
h = x[1]-x[0]  # grid spacing
Lap_C = laplacian(U_C, h)
Lap_M = laplacian(U_M, h)

# ============================================================
# Visualization
# ============================================================
plt.figure(figsize=(14,10))

plt.subplot(2,2,1)
plt.imshow(U_C, extent=[-2,2,-2,2], origin='lower', cmap='viridis')
plt.colorbar(); plt.title("Construct Potential")

plt.subplot(2,2,2)
plt.imshow(U_M, extent=[-2,2,-2,2], origin='lower', cmap='viridis')
plt.colorbar(); plt.title("Mandelbrot Escape Potential")

plt.subplot(2,2,3)
plt.imshow(Lap_C, extent=[-2,2,-2,2], origin='lower', cmap='bwr')
plt.colorbar(); plt.title("Laplacian ΔU (Construct)")

plt.subplot(2,2,4)
plt.imshow(Lap_M, extent=[-2,2,-2,2], origin='lower', cmap='bwr')
plt.colorbar(); plt.title("Laplacian ΔU (Mandelbrot)")

plt.tight_layout()
plt.savefig("potential_and_laplacians.png", dpi=200)
plt.show()

# ============================================================
# Correlation between Laplacians
# ============================================================
mask = (U_M > 0) & (U_C != 0)  # avoid undefined areas
corr = np.corrcoef(Lap_C[mask].flatten(), Lap_M[mask].flatten())[0,1]
print("Correlation between Laplacians:", corr)
