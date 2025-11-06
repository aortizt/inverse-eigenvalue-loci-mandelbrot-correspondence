import numpy as np
import matplotlib.pyplot as plt

# --- Load data ---
C = np.loadtxt("/home/Merlin/Desktop/out_clean/construct_aligned.csv", delimiter=",")
M = np.loadtxt("/home/Merlin/Desktop/out_clean/mandel_boundary_sample.csv", delimiter=",")

# --- Curvature estimation function ---
def estimate_curvature(points):
    dx = np.gradient(points[:,0])
    dy = np.gradient(points[:,1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
    return curvature

curv_C = estimate_curvature(C)
curv_M = estimate_curvature(M)

# --- Normalize for color scale ---
curv_C_norm = np.log1p(curv_C)  # log-scale helps visualization
curv_M_norm = np.log1p(curv_M)

# --- Plot Construct ---
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.scatter(C[:,0], C[:,1], c=curv_C_norm, cmap="plasma", s=6)
plt.colorbar(label="log(1+curvature)")
plt.title("Construct curvature hotspots")
plt.axis("equal")

# --- Plot Mandelbrot ---
plt.subplot(1,2,2)
plt.scatter(M[:,0], M[:,1], c=curv_M_norm, cmap="plasma", s=6)
plt.colorbar(label="log(1+curvature)")
plt.title("Mandelbrot boundary curvature hotspots")
plt.axis("equal")

plt.suptitle("Curvature overlay: Construct vs Mandelbrot")
plt.tight_layout()
plt.show()
