import numpy as np from scipy.linalg import eigvals from sklearn.neighbors import NearestNeighbors from sklearn.decomposition import PCA import ot import matplotlib.pyplot as plt import os

-----------------------------

1. Generate The Construct

-----------------------------

def construct_points(maxN=60): points = [] for n in range(2, maxN + 1): ak = np.zeros((n, n), dtype=float) ak[0, :] = 1.0 if n >= 2: ak[1, 0] = 1.0 for j in range(1, n - 1): ak[j + 1, j] = 1.0 vals = eigvals(ak) nonzero = vals[np.abs(vals) > 1e-12] inv_vals = 1.0 / nonzero for v in inv_vals: points.append([float(np.real(v)), float(np.imag(v))]) return np.array(points)

-----------------------------

2. Mandelbrot Distance Estimator (DE)

-----------------------------

def mandelbrot_distance_estimator(c, max_iter=200, bailout=1e6): z = 0 + 0j dz = 0 + 0j for i in range(max_iter): dz = 2 * z * dz + 1 z = z * z + c if abs(z) > bailout: return np.log(abs(z)) * abs(z) / max(abs(dz), 1e-16) return 0.0

def sample_mandelbrot_boundary(nx=180, ny=120, max_iter=200, threshold=1e-3, nsamples=500): xs = np.linspace(-2.25, 1.25, nx) ys = np.linspace(-1.5, 1.5, ny) pts = [] for j, y in enumerate(ys): for i, x in enumerate(xs): c = x + 1j * y d = mandelbrot_distance_estimator(c, max_iter=max_iter) if d > 0 and d < threshold: pts.append([x, y]) pts = np.array(pts) if pts.shape[0] == 0: return np.empty((0, 2)) if pts.shape[0] > nsamples: idx = np.random.choice(pts.shape[0], nsamples, replace=False) pts = pts[idx] return pts

-----------------------------

3. Extract Local Features (small PCA orientation)

-----------------------------

def compute_features(X, k=8): if X.shape[0] == 0: return np.empty((0, 2)) nbrs = NearestNeighbors(n_neighbors=min(k, X.shape[0])).fit(X) , idxs = nbrs.kneighbors(X) orientations = [] for i in range(X.shape[0]): neigh = X[idxs[i]] try: pca = PCA(n_components=2).fit(neigh - np.mean(neigh, axis=0)) v = pca.components[0] orientations.append([v[0], v[1]]) except Exception: orientations.append([1.0, 0.0]) return np.array(orientations)

-----------------------------

4. Sinkhorn OT + Procrustes alignment

-----------------------------

def sinkhorn_transport(X, Y, reg=1e-2): a = np.ones(X.shape[0]) / float(X.shape[0]) b = np.ones(Y.shape[0]) / float(Y.shape[0]) M = ot.dist(X, Y, metric='euclidean') G = ot.sinkhorn(a, b, M, reg) return G, M

def procrustes_align(X, Y, matches): A = X B = Y[matches] muA = A.mean(axis=0) muB = B.mean(axis=0) A0 = A - muA B0 = B - muB U, s, Vt = np.linalg.svd(B0.T @ A0) R = U @ Vt A_aligned = (A0 @ R.T) + muB return A_aligned, R

-----------------------------

5. Runner (simple)

-----------------------------

def run_pipeline(outdir="output_stage1", maxN=60, nx=180, ny=120, boundary_samples=800, sinkhorn_reg=1e-2): os.makedirs(outdir, exist_ok=True) print("Generating Construct...") X = construct_points(maxN=maxN) print("Sampling Mandelbrot boundary...") Y = sample_mandelbrot_boundary(nx=nx, ny=ny, nsamples=boundary_samples) print("Computing features...") FX = compute_features(X, k=8) FY = compute_features(Y, k=8) XA = np.hstack([FX, X]) XB = np.hstack([FY, Y]) print("Running Sinkhorn...") G, M = sinkhorn_transport(XA, XB, reg=sinkhorn_reg) matches = np.argmax(G, axis=1) costs = M[np.arange(len(matches)), matches] print("Aligning via Procrustes...") X_aligned, R = procrustes_align(X, Y, matches) np.savetxt(os.path.join(outdir, "construct_points.csv"), X, delimiter=",") np.savetxt(os.path.join(outdir, "mandel_points.csv"), Y, delimiter=",") np.savetxt(os.path.join(outdir, "construct_aligned.csv"), X_aligned, delimiter=",") np.savetxt(os.path.join(outdir, "transport_plan_rows_argmax.csv"), matches, delimiter=",", fmt="%d") plt.figure(figsize=(9,6)) plt.scatter(Y[:,0], Y[:,1], s=6, c='red', label='Mandel boundary') plt.scatter(X[:,0], X[:,1], s=6, c='blue', alpha=0.6, label='Construct') plt.scatter(X_aligned[:,0], X_aligned[:,1], s=6, c='cyan', alpha=0.5, label='Construct aligned') plt.legend() plt.axis('equal') plt.title('Construct vs Mandelbrot boundary (blue=orig, cyan=aligned, red=target)') plt.savefig(os.path.join(outdir, "alignment.png"), dpi=200) plt.close() print("Done. Outputs in", outdir) return {"X": X, "Y": Y, "X_aligned": X_aligned, "matches": matches, "costs": costs}

if name == "main": run_pipeline()

