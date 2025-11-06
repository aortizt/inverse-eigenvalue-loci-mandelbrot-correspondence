import numpy as np
import matplotlib.pyplot as plt

# Paths to CSV files
base_path = "/home/Merlin/Desktop/out_clean/"
C = np.loadtxt(base_path + "construct_points.csv", delimiter=",")
M = np.loadtxt(base_path + "mandel_boundary_sample.csv", delimiter=",")
C_aligned = np.loadtxt(base_path + "construct_aligned.csv", delimiter=",")
matches = np.loadtxt(base_path + "matches_indices.csv", delimiter=",", dtype=int)

# Plot
plt.figure(figsize=(9,6))

# Mandel boundary (reference)
plt.scatter(M[:,0], M[:,1], s=6, c='red', label='Mandel boundary')

# Construct aligned
plt.scatter(C_aligned[:,0], C_aligned[:,1], s=6, c='cyan', alpha=0.7, label='Construct aligned')

# Draw matching lines
for i, j in enumerate(matches):
    x_vals = [C_aligned[i,0], M[j,0]]
    y_vals = [C_aligned[i,1], M[j,1]]
    plt.plot(x_vals, y_vals, color='gray', linewidth=0.3, alpha=0.5)

plt.legend()
plt.axis('equal')
plt.title("Point-by-Point Matches: Construct ↔ Mandelbrot Boundary")
plt.show()
