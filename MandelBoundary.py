import numpy as np import matplotlib.pyplot as plt from scipy.linalg import eigvals from sklearn.neighbors import NearestNeighbors from sklearn.decomposition import PCA import ot from tqdm import tqdm import os

-----------------------------

1. Generate The Construct

-----------------------------

def construct_points(maxN=60): points = [] for n in range(2, maxN+1): # Build Lucas-type companion matrix ak = np.zeros((n,n)) ak[0,:] = 1 ak[1,0] = 1 for j in range(1,n-1): ak[j+1,j] = 1 vals = eigvals(ak) inv_vals = 1.0/vals for v in inv_vals: points.append([np.real(v), np.imag(v)]) return np.array(points)

-----------------------------

2. Mandelbrot Distance Estimator (DE)

-----------------------------

def mandelbrot_distance_estimator(c, max_iter=200, bailout=1e6): z = 0 dz = 0 for i in range(max_iter): dz = 2zdz + 1 z = z*z + c if abs(z) > bailout: return np.log(abs(z))*abs(z)/abs(dz) return 0.0

def sample_mandelbrot_boundary(nx=360, ny=270, max_iter=200, threshold=1e-3, nsamples=500): xs = np.linspace(-2,1,nx) ys = np.linspace(-1.5,1.5,ny) boundary_pts = [] for x in xs: for y in ys: c = x + 1j*y d = mandelbrot_distance_estimator(c, max_iter=max_iter) if d < threshold and d>0: boundary_pts.append([x,y]) boundary_pts = np.array(boundary_pts) # Subsample if len(boundary_pts)>nsamples: idx = np.random.choice(len(boundary_pts), nsamples, replace=False) boundary_pts = boundary_pts[idx] return boundary_pts

-----------------------------

3. Extract Local Features (PCA orientation)

-----------------------------

def compute_features(X, k=5): nbrs = NearestNeighbors(n_neighbors=k).fit(X) , idx = nbrs.kneighbors(X) features = [] for i in range(len(X)): neigh = X[idx[i]] pca = PCA(n_components=2).fit(neigh) orient = pca.components[0] features.append([orient[0], orient[1]]) return np.array(features)

-----------------------------

4. Sinkhorn OT + ICP refinement

-----------------------------

def match_sets(X, Y, reg=1e-1): nx, ny = len(X), len(Y) a, b = np.ones((nx,))/nx, np.ones((ny,))/ny M = ot.dist(X,Y) G = ot.sinkhorn(a, b, M, reg) return G

def procrustes_align(X, Y, G): # Weighted Procrustes using transport plan G X_mean = np.average(X, axis=0, weights=G.sum(1)) Y_mean = np.average(Y, axis=0, weights=G.sum(0)) Xc = X - X_mean Yc = Y - Y_mean C = Xc.T @ G @ Yc U,s,Vt = np.linalg.svd(C) R = U@Vt X_new = Xc@R return X_new+Y_mean, R

-----------------------------

5. Pipeline Runner

-----------------------------

def run_pipeline(maxN=60, nx=360, ny=270, outdir="output_stage1"): os.makedirs(outdir, exist_ok=True) print("Generating Construct...") X = construct_points(maxN) print("Sampling Mandelbrot boundary...") Y = sample_mandelbrot_boundary(nx, ny) print("Computing features...") FX, FY = compute_features(X), compute_features(Y) print("Matching sets with Sinkhorn...") G = match_sets(X,Y) print("Aligning via Procrustes...") X_aligned, R = procrustes_align(X,Y,G)

# Save CSV of matches
np.savetxt(os.path.join(outdir,"construct_points.csv"),X,delimiter=",")
np.savetxt(os.path.join(outdir,"mandelbrot_points.csv"),Y,delimiter=",")
np.savetxt(os.path.join(outdir,"construct_aligned.csv"),X_aligned,delimiter=",")
np.savetxt(os.path.join(outdir,"transport_plan.csv"),G,delimiter=",")

# Plot before and after
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.scatter(X[:,0],X[:,1],s=5,c='blue',label='Construct')
plt.scatter(Y[:,0],Y[:,1],s=5,c='red',label='Mandelbrot boundary')
plt.legend(); plt.title("Original")
plt.subplot(1,2,2)
plt.scatter(X_aligned[:,0],X_aligned[:,1],s=5,c='blue',label='Construct aligned')
plt.scatter(Y[:,0],Y[:,1],s=5,c='red',label='Mandelbrot boundary')
plt.legend(); plt.title("Aligned")
plt.tight_layout()
plt.savefig(os.path.join(outdir,"alignment.png"))
plt.close()

print(f"Results saved in {outdir}")
return X,Y,X_aligned,G

if name=="main": run_pipeline(maxN=60, nx=180, ny=120)

