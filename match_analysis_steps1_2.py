import numpy as np
import matplotlib.pyplot as plt

# Paths to CSV files
base_path = "/home/Merlin/Desktop/out_clean/"
C = np.loadtxt(base_path + "construct_points.csv", delimiter=",")              # shape (nA,2)
M = np.loadtxt(base_path + "mandel_boundary_sample.csv", delimiter=",")        # shape (nB,2)
C_aligned = np.loadtxt(base_path + "construct_aligned.csv", delimiter=",")     # shape (nA,2)
matches = np.loadtxt(base_path + "matches_indices.csv", delimiter=",", dtype=int)

# Step 1: Visual validation
plt.figure(figsize=(9,6))
plt.scatter(M[:,0], M[:,1], s=6, c='red', label='Mandel boundary')
plt.scatter(C[:,0], C[:,1], s=6, c='blue', alpha=0.6, label='Construct original')
plt.scatter(C_aligned[:,0], C_aligned[:,1], s=6, c='cyan', alpha=0.7, label='Construct aligned')
plt.legend()
plt.axis('equal')
plt.title('Initial matching visualization')
plt.show()

# Step 2: Matching quality metrics
distances = np.linalg.norm(C_aligned - M[matches], axis=1)
print("Matching distances stats:")
print("  Min:", np.min(distances))
print("  Median:", np.median(distances))
print("  Max:", np.max(distances))

plt.hist(distances, bins=50)
plt.xlabel("Distance between matched points")
plt.ylabel("Count")
plt.title("Matching Distance Distribution")
plt.show()
