# tci_construct_mandelbrot.py
import numpy as np
from numpy.linalg import svd

def lucas_companion(n: int):
    A = np.zeros((n, n), dtype=float)
    A[0, :] = 1.0
    A[1:, :-1] = np.eye(n-1)
    return A

def construct_points(ns):
    pts = []
    for n in ns:
        A = lucas_companion(n)
        vals = np.linalg.eigvals(A)
        vals = vals[np.abs(vals) > 1e-10]
        inv = 1.0 / vals
        pts.extend(inv)
    return np.array(pts, dtype=complex)

def mandelbrot_distance_estimator(c, max_iter=200, R=4.0, eps=1e-12):
    z = np.zeros_like(c, dtype=np.complex128)
    dz = np.zeros_like(c, dtype=np.complex128) + 1.0
    escaped = np.zeros(c.shape, dtype=bool)
    last_z = np.zeros_like(c, dtype=np.complex128)
    for _ in range(max_iter):
        dz = 2*z*dz + 1
        z = z*z + c
        mask = (np.abs(z) > R) & (~escaped)
        escaped |= mask
        last_z[mask] = z[mask]
    dist = np.zeros(c.shape, dtype=float)
    mask = escaped
    z_ = last_z[mask]
    dz_ = dz[mask]
    with np.errstate(divide='ignore', invalid='ignore'):
        dist_val = np.log(np.abs(z_)) * np.abs(z_) / np.maximum(np.abs(2*z_*dz_), eps)
    dist[mask] = np.nan_to_num(dist_val, nan=0.0, posinf=0.0, neginf=0.0)
    return escaped, dist, last_z

def procrustes_align_no_scale(Xc: np.ndarray, Yc: np.ndarray):
    X = np.column_stack([Xc.real, Xc.imag])
    Y = np.column_stack([Yc.real, Yc.imag])
    X_mu = X.mean(axis=0)
    Y_mu = Y.mean(axis=0)
    X0 = X - X_mu
    Y0 = Y - Y_mu
    U, _, Vt = svd(Y0.T @ X0, full_matrices=False)
    R = U @ Vt
    X_aligned = (X0 @ R) + Y_mu
    Xc_aligned = X_aligned[:,0] + 1j*X_aligned[:,1]
    t = Y_mu - (X_mu @ R)
    return Xc_aligned, R, t

def cloud_to_probability(cloud: np.ndarray, bbox, bins: int, eps=1e-12):
    x_min, x_max, y_min, y_max = bbox
    H, _, _ = np.histogram2d(cloud.real, cloud.imag,
                             bins=(bins, bins),
                             range=[[x_min, x_max], [y_min, y_max]])
    H = H.astype(float)
    H_sum = H.sum()
    if H_sum < eps:
        P = np.full(H.shape, 1.0 / H.size)
    else:
        P = H / H_sum
    return P

def kl_divergence(P, X, eps=1e-12):
    P_ = np.clip(P, eps, None)
    X_ = np.clip(X, eps, None)
    return float(np.sum(P_ * (np.log(P_) - np.log(X_))))

def tci_flow(P, X0, alpha=0.2, T=40, eps=1e-12):
    X = X0.copy()
    kls = [kl_divergence(P, X, eps)]
    traj = [X.copy()]
    for _ in range(T):
        X = (1 - alpha) * X + alpha * P
        kls.append(kl_divergence(P, X, eps))
        traj.append(X.copy())
    return np.array(kls), traj
