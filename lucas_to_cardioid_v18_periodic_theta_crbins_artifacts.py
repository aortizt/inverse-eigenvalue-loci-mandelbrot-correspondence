
# lucas_to_cardioid_v13_normalized_theta.py
# v13: Stabilize θ-iteration by circle-normalizing the (u+iv) boundary each iteration,
#      and replace ill-conditioned affine alignment by an optimal boundary rotation after normalization.
#
# Requirements: numpy, scipy, shapely, alphashape, matplotlib (matplotlib not used for plotting in v13)

import numpy as np
import os, json, csv
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

from shapely.geometry import Point, Polygon, MultiPolygon
from scipy.spatial import Delaunay
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import alphashape



# ================================================================
# Parameters
# ================================================================

VERSION = 18
# ------------------------------------------------------------
# v14: enforce 2π-periodicity of boundary angle after each theta update
# This prevents slow unwrap drift from contaminating the Dirichlet angle data.
# ------------------------------------------------------------
PERIODIC_THETA_ENFORCE = True

N_MIN = 2
N_MAX = 100

# Lucas-domain alpha-shape
ALPHA_FIXED = 4.5

# Mesh spacing (smaller = finer mesh)
REFINEMENT_LEVELS = [
    dict(name="L0", h_L=0.08,  h_C=0.06,  boundary_h=0.04),
    dict(name="L1", h_L=0.05,  h_C=0.04,  boundary_h=0.025),
    dict(name="L2", h_L=0.035, h_C=0.03,  boundary_h=0.015),
    dict(name="L3", h_L=0.025, h_C=0.02,  boundary_h=0.010),
]

# UV triangulation robustness
UV_QHULL_OPTIONS = "QJ Qbb Qc"

# Interior reporting
DELTA_SWEEP_FACTORS = [2.0, 4.0, 6.0]

# Theta iteration
THETA_ITERS = 6
THETA_RELAX = 0.7
THETA_SMOOTH = 7        # odd integer for moving-average smoothing of theta on boundary
THETA_UNWRAP_ANCHOR = 0 # index in ordered boundary list used as unwrapping anchor

# Beltrami / K reporting robustness
EPS_FZ = 1e-10
MU_CAP = 0.9999

# Angle diagnostic robustness
EPS_NORM = 1e-14
# Cauchy–Riemann relative defect regularization
CR_REL_EPS = 1e-12



# ================================================================
# 1. Generate Lucas companion matrices and their inverse eigenvalues
# ================================================================

def generate_lucas_companion(n: int) -> np.ndarray:
    """Companion matrix for x^n - x^{n-1} - ... - x - 1."""
    C = np.zeros((n, n))
    C[0, :] = 1.0
    for i in range(1, n):
        C[i, i - 1] = 1.0
    return C


def compute_inverse_eigenvalues(n_min=N_MIN, n_max=N_MAX) -> np.ndarray:
    inv_eigs = []
    for n in range(n_min, n_max + 1):
        C = generate_lucas_companion(n)
        eigs = np.linalg.eigvals(C)
        nonzero = np.abs(eigs) > 1e-12
        inv = 1.0 / eigs[nonzero]
        inv_eigs.append(inv)
        print(f"[generate_lucas_companion] n={n}, eigenvalues={len(eigs)}, kept={np.count_nonzero(nonzero)}")
    inv_eigs = np.concatenate(inv_eigs)
    print(f"[generate_lucas_companion] total points: {len(inv_eigs)}")
    return inv_eigs




def _largest_polygon(geom):
    """Return the largest Polygon from a Polygon/MultiPolygon."""
    if isinstance(geom, Polygon):
        return geom
    if isinstance(geom, MultiPolygon):
        polys = list(geom.geoms)
        polys.sort(key=lambda p: p.area, reverse=True)
        return polys[0]
    raise TypeError(f"Unexpected alpha shape type: {type(geom)}")


def _resample_closed_polyline(xy: np.ndarray, n_out: int) -> np.ndarray:
    """
    Resample a closed polyline (N,2) to n_out points by arclength.
    Expects first point != last point; closure is implicit.
    """
    # Close it explicitly for arclength computation
    pts = np.vstack([xy, xy[0]])
    seg = pts[1:] - pts[:-1]
    d = np.sqrt((seg**2).sum(axis=1))
    s = np.concatenate([[0.0], np.cumsum(d)])
    L = s[-1]
    if L <= 0:
        raise ValueError("Degenerate boundary (zero length).")

    # Target arclengths, excluding the final L to avoid duplicating the start
    t = np.linspace(0.0, L, n_out + 1)[:-1]

    out = np.zeros((n_out, 2), dtype=float)
    j = 0
    for i, ti in enumerate(t):
        while j < len(d) - 1 and s[j + 1] < ti:
            j += 1
        # linear interpolation within segment j
        if d[j] == 0:
            out[i] = pts[j]
        else:
            u = (ti - s[j]) / d[j]
            out[i] = pts[j] * (1 - u) + pts[j + 1] * u
    return out


def export_lucas_boundary_npy(
    n_min: int,
    n_max: int,
    *,
    alpha: float = 3.5,
    n_boundary: int = 2000,
    out_path: str = "lucas_points.npy",
    center: complex | None = None,
    radial_clip: float | None = None,
):
    """
    Build Lucas inverse-eigenvalue cloud and export a single boundary curve as lucas_points.npy.

    Parameters
    ----------
    alpha : float
        Alpha-shape parameter. Larger -> closer to convex hull; smaller -> more concave.
        You may need to tune this per (n_min,n_max).
    n_boundary : int
        Number of boundary samples to save for v20.
    center : complex or None
        Optional: subtract a center before boundary extraction, then add back.
        Useful if you want to center around 0.
    radial_clip : float or None
        Optional: discard points with |z| > radial_clip to remove far outliers.
    """
    # 1) Cloud
    inv_eigs = compute_inverse_eigenvalues(n_min=n_min, n_max=n_max)  # complex array

    z = inv_eigs.copy()
    if center is not None:
        z = z - center

    if radial_clip is not None:
        z = z[np.abs(z) <= radial_clip]

    # Convert to (N,2)
    pts = np.column_stack([z.real, z.imag])

    # 2) Alpha shape boundary (concave hull)
    shape = alphashape.alphashape(pts, alpha)
    poly = _largest_polygon(shape)

    # Exterior coords are already ordered (usually CCW, but we enforce)
    xy = np.asarray(poly.exterior.coords[:-1], dtype=float)  # drop repeated last point

    # Enforce CCW orientation via signed area
    signed_area = 0.5 * np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - np.roll(xy[:, 0], -1) * xy[:, 1])
    if signed_area < 0:
        xy = xy[::-1]

    # 3) Resample uniformly by arclength
    xy_rs = _resample_closed_polyline(xy, n_boundary)

    if center is not None:
        xy_rs[:, 0] += center.real
        xy_rs[:, 1] += center.imag

    # 4) Save
    np.save(out_path, xy_rs)
    print(f"[export_lucas_boundary_npy] saved {out_path} with shape {xy_rs.shape}")
    return xy_rs


# ================================================================
# 2. Alpha-shape polygon, mesh generation
# ================================================================

def alpha_shape_polygon(points: np.ndarray, alpha: float) -> Polygon:
    """Alpha-shape polygon (largest component if multipolygon)."""
    pts2d = np.column_stack([points.real, points.imag])
    ashape = alphashape.alphashape(pts2d, alpha)
    if isinstance(ashape, MultiPolygon):
        polys = list(ashape.geoms)
        polys.sort(key=lambda p: p.area, reverse=True)
        return polys[0]
    if isinstance(ashape, Polygon):
        return ashape
    raise RuntimeError("alpha_shape returned unsupported geometry type")


def polygon_to_mesh(poly: Polygon,
                    h=0.05,
                    boundary_h=None,
                    boundary_layers=1,
                    layer_factor=2.0,
                    verbose=True,
                    seed=0):
    """
    Build a triangulation of a (possibly concave) polygon domain.

    - Explicit arclength boundary sampling
    - Optional boundary-layer refinement
    - Filter triangles by centroid-in-polygon (concave-safe)
    """
    rng = np.random.default_rng(seed)

    # 1) Boundary sampling by arclength
    if boundary_h is None:
        boundary_h = 0.5 * h

    ring = poly.exterior
    L = ring.length
    nB = max(16, int(np.ceil(L / boundary_h)))
    svals = np.linspace(0.0, L, nB, endpoint=False)
    B = np.array([(ring.interpolate(s).x, ring.interpolate(s).y) for s in svals])

    # 2) Interior coarse grid points
    minx, miny, maxx, maxy = poly.bounds
    xs = np.arange(minx, maxx + h, h)
    ys = np.arange(miny, maxy + h, h)
    grid = np.array([(x, y) for x in xs for y in ys], dtype=float)
    inside = np.array([poly.covers(Point(p)) for p in grid])
    I = grid[inside]

    # 3) Optional boundary-layer refinement
    BL = []
    if boundary_layers > 0:
        for k in range(1, boundary_layers + 1):
            r = (k / boundary_layers) * (h / layer_factor)
            jitter = rng.normal(scale=r, size=B.shape)
            candidates = B + jitter
            keep = []
            for p in candidates:
                if poly.covers(Point(p)):
                    keep.append(p)
            if keep:
                BL.append(np.array(keep, dtype=float))

    BL = np.vstack(BL) if BL else np.zeros((0, 2), dtype=float)

    # 4) Combine points + Delaunay
    P = np.vstack([B, I, BL])
    P = np.unique(np.round(P, 12), axis=0)

    if verbose:
        print(f"[polygon_to_mesh] boundary samples: {len(B)} (boundary_h={boundary_h})")
        print(f"[polygon_to_mesh] interior grid inside: {len(I)} (h={h})")
        if boundary_layers > 0:
            print(f"[polygon_to_mesh] boundary-layer points: {len(BL)} (layers={boundary_layers})")
        print(f"[polygon_to_mesh] total points (dedup): {len(P)}")

    if len(P) < 30:
        raise RuntimeError("Too few points; decrease h or boundary_h")

    tri = Delaunay(P, qhull_options=UV_QHULL_OPTIONS)
    T = tri.simplices

    # 5) Filter triangles (centroid inside)
    centroids = P[T].mean(axis=1)
    keepT = np.array([poly.contains(Point(c)) for c in centroids])
    T = T[keepT]

    # 6) Drop near-degenerate triangles
    eps_area = 1e-14
    P0 = P[T[:, 0]]
    P1 = P[T[:, 1]]
    P2 = P[T[:, 2]]
    dbl_area = np.abs((P1[:,0]-P0[:,0])*(P2[:,1]-P0[:,1]) - (P1[:,1]-P0[:,1])*(P2[:,0]-P0[:,0]))
    keepA = dbl_area > (2.0 * eps_area)
    T = T[keepA]

    if verbose:
        print(f"[polygon_to_mesh] triangles raw: {len(tri.simplices)}")
        print(f"[polygon_to_mesh] triangles kept (centroid-in-poly): {len(T)}")
        print(f"[polygon_to_mesh] triangles kept (area>{eps_area:g}): {len(T)}")

    return P, T


# ================================================================
# 3. FEM (P1) helpers
# ================================================================

def _p1_local_grads(p0, p1, p2):
    """Return gradients of barycentric basis (λ0,λ1,λ2) and area."""
    B = np.array([[p1[0] - p0[0], p2[0] - p0[0]],
                  [p1[1] - p0[1], p2[1] - p0[1]]], dtype=float)
    detB = np.linalg.det(B)
    area = 0.5 * abs(detB)
    if area < 1e-14:
        raise RuntimeError("Degenerate triangle")
    invBT = np.linalg.inv(B).T
    g1 = invBT @ np.array([1.0, 0.0])
    g2 = invBT @ np.array([0.0, 1.0])
    g0 = -g1 - g2
    grads = np.vstack([g0, g1, g2])  # (3,2)
    return grads, area


def assemble_stiffness_p1(points: np.ndarray, triangles: np.ndarray) -> csr_matrix:
    n = len(points)
    K = lil_matrix((n, n), dtype=float)
    for t in triangles:
        i0, i1, i2 = t
        p0, p1, p2 = points[i0], points[i1], points[i2]
        try:
            grads, area = _p1_local_grads(p0, p1, p2)
        except RuntimeError:
            continue
        ke = area * (grads @ grads.T)  # (3,3)
        idx = [i0, i1, i2]
        for a in range(3):
            for b in range(3):
                K[idx[a], idx[b]] += ke[a, b]
    return K.tocsr()


def boundary_dofs(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Return vertex indices on boundary (triangle edges used by only one triangle)."""
    edge_count = {}
    for tri in triangles:
        edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
        for a, b in edges:
            e = (a, b) if a < b else (b, a)
            edge_count[e] = edge_count.get(e, 0) + 1
    bnd = set()
    for (a, b), c in edge_count.items():
        if c == 1:
            bnd.add(a)
            bnd.add(b)
    return np.array(sorted(bnd), dtype=int)


def solve_laplace_dirichlet_arclength(points, triangles, poly: Polygon, g_theta, *, pin=None):
    """Solve Laplace with Dirichlet boundary data given as a function of arclength θ."""
    K = assemble_stiffness_p1(points, triangles)
    bnd = boundary_dofs(points, triangles)
    L = poly.exterior.length

    g = np.zeros(len(points), dtype=float)
    for i in bnd:
        px, py = points[i]
        s = poly.exterior.project(Point(float(px), float(py)))
        theta = -np.pi + 2.0 * np.pi * (s / L)
        g[i] = float(g_theta(theta))

    A = K.tolil()
    rhs = np.zeros(len(points), dtype=float)
    free = np.ones(len(points), dtype=bool)
    free[bnd] = False

    for i in np.where(free)[0]:
        row = A.rows[i]
        data = A.data[i]
        for jj, j in enumerate(row):
            if not free[j]:
                rhs[i] -= data[jj] * g[j]

    for j in bnd:
        A.rows[j] = [j]
        A.data[j] = [1.0]
        rhs[j] = g[j]
    A = A.tocsr()

    if pin is not None:
        A = A.tolil()
        A.rows[pin] = [pin]
        A.data[pin] = [1.0]
        rhs[pin] = float(rhs[pin])
        A = A.tocsr()

    u = spsolve(A, rhs)
    return u


def solve_harmonic_conjugate(points, triangles, u, *, pin=0):
    """Compute v so that ∇v ≈ J∇u in weak form."""
    n = len(points)
    K = assemble_stiffness_p1(points, triangles).tolil()
    rhs = np.zeros(n, dtype=float)

    for tri in triangles:
        i0, i1, i2 = tri
        p0, p1, p2 = points[i0], points[i1], points[i2]
        try:
            grads, area = _p1_local_grads(p0, p1, p2)
        except RuntimeError:
            continue

        u_loc = np.array([u[i0], u[i1], u[i2]], dtype=float)
        grad_u = (u_loc[:, None] * grads).sum(axis=0)  # (2,)
        Ju = np.array([-grad_u[1], grad_u[0]])
        for a, idx in enumerate([i0, i1, i2]):
            rhs[idx] += area * float(Ju @ grads[a])

    K.rows[pin] = [pin]
    K.data[pin] = [1.0]
    rhs[pin] = 0.0
    v = spsolve(K.tocsr(), rhs)
    return v


# ================================================================
# 4. Cardioid polygon
# ================================================================

def cardioid_polygon(n=401):
    t = np.linspace(-np.pi, np.pi, n, endpoint=True)
    z = 0.5 * np.exp(1j * t) - 0.25 * np.exp(2j * t)
    pts = np.column_stack([z.real, z.imag])
    return Polygon(pts)


# ================================================================
# 5. UV→physical inversion via Delaunay+barycentric
# ================================================================

def invert_uv_to_z(uv_query: np.ndarray,
                   uv_nodes: np.ndarray,
                   z_nodes: np.ndarray,
                   *,
                   qhull_options: str = UV_QHULL_OPTIONS):
    """
    Map uv_query -> z by triangulating uv_nodes and barycentric interpolation of z_nodes.
    """
    tri = Delaunay(uv_nodes, qhull_options=qhull_options)

    simp = tri.find_simplex(uv_query)
    ok = simp >= 0
    z_out = np.full(len(uv_query), np.nan + 1j * np.nan, dtype=complex)
    if not np.any(ok):
        return z_out, ok, simp

    X = uv_query[ok]
    s = simp[ok]

    T = tri.transform[s, :2, :]
    r = X - tri.transform[s, 2, :]
    bary12 = np.einsum('ijk,ik->ij', T, r)
    b1 = bary12[:, 0]
    b2 = bary12[:, 1]
    b0 = 1.0 - b1 - b2

    verts = tri.simplices[s]
    z0 = z_nodes[verts[:, 0]]
    z1 = z_nodes[verts[:, 1]]
    z2 = z_nodes[verts[:, 2]]
    z_out[ok] = b0 * z0 + b1 * z1 + b2 * z2
    return z_out, ok, simp


# ================================================================
# 6. Diagnostics: Beltrami K, angle distortion
# ================================================================

def beltrami_K_on_triangles(points, triangles, phi: np.ndarray, valid_vertex: np.ndarray):
    mus = []
    Ks = []
    used = 0
    for tri in triangles:
        if not (valid_vertex[tri[0]] and valid_vertex[tri[1]] and valid_vertex[tri[2]]):
            continue
        i0, i1, i2 = tri
        p0, p1, p2 = points[i0], points[i1], points[i2]
        try:
            grads, area = _p1_local_grads(p0, p1, p2)
        except RuntimeError:
            continue
        if area < 1e-14:
            continue

        f_loc = np.array([phi[i0], phi[i1], phi[i2]], dtype=complex)
        fx = np.sum(f_loc * grads[:, 0])
        fy = np.sum(f_loc * grads[:, 1])
        fz = 0.5 * (fx - 1j * fy)
        fzb = 0.5 * (fx + 1j * fy)
        if abs(fz) < EPS_FZ:
            continue

        mu = fzb / fz
        a = abs(mu)
        if not np.isfinite(a) or a >= MU_CAP:
            continue

        K = (1 + a) / (1 - a)
        mus.append(mu)
        Ks.append(K)
        used += 1

    return np.array(mus, dtype=complex), np.array(Ks, dtype=float), used


def beltrami_K_full(points, triangles, phi: np.ndarray, valid_vertex: np.ndarray):
    """Compute quasiconformal dilatation K per triangle, aligned with input.

    Returns
    -------
    Ks_full : (nT,) array
        K value per triangle; NaN where undefined (invalid vertices, degenerate tri,
        or |mu|>=1).
    used : (nT,) bool array
        True where K was computed.
    """
    Ks_full = np.full(len(triangles), np.nan, dtype=float)
    used = np.zeros(len(triangles), dtype=bool)


    # Ensure complex representations
    if points.ndim == 2 and points.shape[1] == 2:
        Z = points[:, 0] + 1j * points[:, 1]
    else:
        Z = points.astype(complex, copy=False)
    if isinstance(phi, np.ndarray) and phi.ndim == 2 and phi.shape[1] == 2:
        W = phi[:, 0] + 1j * phi[:, 1]
    else:
        W = np.asarray(phi, dtype=complex)
    for ti, tri in enumerate(triangles):
        a, b, c = tri
        if not (valid_vertex[a] and valid_vertex[b] and valid_vertex[c]):
            continue

        z1, z2, z3 = Z[a], Z[b], Z[c]
        w1, w2, w3 = W[a], W[b], W[c]

        A = np.array([[z2.real - z1.real, z2.imag - z1.imag],
                      [z3.real - z1.real, z3.imag - z1.imag]], dtype=float)
        detA = float(np.linalg.det(A))
        if abs(detA) < 1e-14:
            continue

        rhs_re = np.array([w2.real - w1.real, w3.real - w1.real], dtype=float)
        rhs_im = np.array([w2.imag - w1.imag, w3.imag - w1.imag], dtype=float)
        grad_re = np.linalg.solve(A, rhs_re)
        grad_im = np.linalg.solve(A, rhs_im)

        u_x, u_y = float(grad_re[0]), float(grad_re[1])
        v_x, v_y = float(grad_im[0]), float(grad_im[1])

        f_z    = 0.5 * ((u_x + v_y) + 1j * (v_x - u_y))
        f_zbar = 0.5 * ((u_x - v_y) + 1j * (v_x + u_y))
        if abs(f_z) < 1e-15:
            continue

        mu = f_zbar / f_z
        amu = abs(mu)
        if not np.isfinite(amu) or amu >= 0.999999:
            continue

        Ks_full[ti] = (1.0 + amu) / (1.0 - amu)
        used[ti] = True

    return Ks_full, used


def angle_distortion_on_triangles(points, triangles, phi, valid_vertex):
    """
    Angle distortion per triangle using Jacobian action on two directions.
    Returns array of angle errors (radians).
    """
    angle_errors = []

    for tri in triangles:
        if not (valid_vertex[tri[0]] and valid_vertex[tri[1]] and valid_vertex[tri[2]]):
            continue

        i0, i1, i2 = tri
        p0, p1, p2 = points[i0], points[i1], points[i2]

        try:
            grads, area = _p1_local_grads(p0, p1, p2)
        except RuntimeError:
            continue

        if area < 1e-14:
            continue

        d1 = p1 - p0
        d2 = p2 - p0

        f_loc = np.array([phi[i0], phi[i1], phi[i2]], dtype=complex)
        fx = np.sum(f_loc * grads[:, 0])
        fy = np.sum(f_loc * grads[:, 1])
        J = np.array([[fx.real, fy.real],
                      [fx.imag, fy.imag]])

        D1 = J @ d1
        D2 = J @ d2

        def angle(u, v):
            nu = np.linalg.norm(u)
            nv = np.linalg.norm(v)
            if nu < EPS_NORM or nv < EPS_NORM:
                return np.nan
            c = np.dot(u, v) / (nu * nv)
            return np.arccos(np.clip(c, -1.0, 1.0))

        theta_before = angle(d1, d2)
        theta_after  = angle(D1, D2)

        if np.isfinite(theta_before) and np.isfinite(theta_after):
            angle_errors.append(abs(theta_after - theta_before))

    return np.array(angle_errors, dtype=float)


# ================================================================
# 7. Boundary ordering + normalization utilities
# ================================================================

def boundary_order_by_arclength(points, triangles, poly: Polygon):
    bnd = boundary_dofs(points, triangles)
    L = poly.exterior.length
    s_b = np.array([poly.exterior.project(Point(points[i,0], points[i,1])) for i in bnd])
    order = np.argsort(s_b)
    return bnd[order], s_b[order], L


def moving_average_periodic(x, w):
    """Simple periodic moving average with window w (odd)."""
    if w <= 1:
        return x
    w = int(w)
    if w % 2 == 0:
        w += 1
    m = len(x)
    pad = w // 2
    x_ext = np.concatenate([x[-pad:], x, x[:pad]])
    kernel = np.ones(w) / w
    y_ext = np.convolve(x_ext, kernel, mode="valid")
    return y_ext[:m]


def unwrap_theta(theta, anchor_index=0):
    """Unwrap theta array (periodic sequence) anchored at a chosen index."""
    theta = np.asarray(theta, float)
    th0 = theta[anchor_index]
    # shift so anchor is near 0 to avoid large drift
    theta_shift = theta - th0
    theta_un = np.unwrap(theta_shift)
    return theta_un + th0


def circle_normalize_boundary(wb):
    """
    Robust-ish circle normalization for boundary complex values.
    Returns (center, radius, wb_norm). Uses mean center and median radius.
    """
    c = np.mean(wb)
    r = np.median(np.abs(wb - c))
    if not np.isfinite(r) or r < 1e-12:
        r = np.mean(np.abs(wb - c)) + 1e-12
    wb_norm = (wb - c) / r
    return c, r, wb_norm


def optimal_rotation(w_src, w_tgt):
    """
    Find rotation e^{iα} minimizing || e^{iα} w_src - w_tgt ||_2 over boundary samples.
    """
    num = np.sum(w_tgt * np.conj(w_src))
    if abs(num) < 1e-14:
        return 1.0 + 0.0j
    return num / abs(num)


# ================================================================
# 8. Theta-iteration with normalization (per domain)
# ================================================================

def solve_uv_with_theta_iteration(points, triangles, poly: Polygon, tag: str):
    """
    Iterate:
      - solve u with boundary u=cos(theta)
      - solve v as harmonic conjugate of u
      - compute boundary w=u+iv, normalize to near unit circle
      - update theta := arg(w_norm) (smoothed + relaxed)
    Return final (u,v) AND normalization parameters (center, radius) for final chart.
    """
    bnd_ord, s_b, L = boundary_order_by_arclength(points, triangles, poly)

    # initial theta from arclength
    theta = -np.pi + 2.0 * np.pi * (s_b / L)
    period_mis_last = np.nan

    # normalized arclength parameter on boundary (0..1)
    t_param = (s_b / L).copy()

    c_last = 0.0 + 0.0j
    r_last = 1.0

    print(f"[{tag}] theta-iteration (n={THETA_ITERS}, relax={THETA_RELAX}, smooth={THETA_SMOOTH})")
    for k in range(1, THETA_ITERS + 1):
        # Dirichlet boundary: u = cos(theta)
        theta_map = dict(zip(bnd_ord, theta))
        u = solve_laplace_dirichlet_arclength(points, triangles, poly, lambda th: np.cos(th))
        v = solve_harmonic_conjugate(points, triangles, u, pin=0)

        wb = (u[bnd_ord] + 1j * v[bnd_ord])

        # circle-normalize boundary and compute new theta from arg
        c_last, r_last, wb_norm = circle_normalize_boundary(wb)
        theta_new = np.angle(wb_norm)

        # smooth + unwrap (to prevent 2π jumps along boundary ordering)
        theta_new = moving_average_periodic(theta_new, THETA_SMOOTH)
        theta_new = unwrap_theta(theta_new, anchor_index=THETA_UNWRAP_ANCHOR)


        if PERIODIC_THETA_ENFORCE:
            # Anchor and distribute the 2π mismatch linearly along arclength parameter t∈[0,1]
            theta_new = theta_new - theta_new[0]
            period_mis = (theta_new[-1] - theta_new[0]) - 2.0*np.pi
            period_mis_last = period_mis
            theta_new = theta_new - period_mis * t_param
        # relax
        theta = (1.0 - THETA_RELAX) * theta + THETA_RELAX * theta_new

        # report drift (how far theta_new deviates from arclength theta baseline, up to mean)
        drift = float(np.median(np.abs(theta_new - theta)))
        print(f"    [theta-iter] k={k}/{THETA_ITERS}  median(|θ_new-θ|)≈{drift:.6f} rad")

    # final solve with final theta
    u = solve_laplace_dirichlet_arclength(points, triangles, poly, lambda th: np.cos(th))
    v = solve_harmonic_conjugate(points, triangles, u, pin=0)

    # final normalization (apply to all nodes, affine in uv so preserves harmonicity)
    wb = (u[bnd_ord] + 1j * v[bnd_ord])
    c_last, r_last, _ = circle_normalize_boundary(wb)
    w = (u + 1j * v - c_last) / r_last
    return w.real, w.imag, c_last, r_last, period_mis_last


# ================================================================
# 9. Interior selection by distance to boundary
# ================================================================

def tri_centroids(points, triangles):
    return points[triangles].mean(axis=1)


def select_interior_triangles(points, triangles, poly: Polygon, delta: float):
    cent = tri_centroids(points, triangles)
    # shapely distance expects Point; this is a bit expensive but ok at these sizes.
    dist = np.array([poly.exterior.distance(Point(float(c[0]), float(c[1]))) for c in cent])
    return dist >= delta


# ================================================================
# 10. Main experiment (v15: periodic theta + CR metrics + boundary-distance bins)
# ================================================================


# ================================================================
# 10A. Triangle gradient + CR defect metrics
# ================================================================

def triangle_gradients(points, triangles, values):
    """Piecewise-linear gradient of a nodal scalar field on each triangle.

    Returns grad array of shape (nT, 2) with [df/dx, df/dy].
    """
    P = points
    T = triangles
    x1,y1 = P[T[:,0],0], P[T[:,0],1]
    x2,y2 = P[T[:,1],0], P[T[:,1],1]
    x3,y3 = P[T[:,2],0], P[T[:,2],1]
    f1,f2,f3 = values[T[:,0]], values[T[:,1]], values[T[:,2]]

    # Twice signed area
    det = (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)
    # Avoid divide-by-zero; caller may mask degenerate triangles already
    det_safe = np.where(np.abs(det) < 1e-30, np.sign(det)*1e-30 + 1e-30, det)

    dfdx = (f1*(y2-y3) + f2*(y3-y1) + f3*(y1-y2)) / det_safe
    dfdy = (f1*(x3-x2) + f2*(x1-x3) + f3*(x2-x1)) / det_safe
    return np.column_stack([dfdx, dfdy])


def cr_defect_metrics(points, triangles, u, v, eps=1e-12):
    """Compute relative Cauchy–Riemann defect per triangle for (u,v)."""
    gu = triangle_gradients(points, triangles, u)
    gv = triangle_gradients(points, triangles, v)
    ux, uy = gu[:,0], gu[:,1]
    vx, vy = gv[:,0], gv[:,1]

    # CR residuals for u_x = v_y and u_y = -v_x
    r1 = ux - vy
    r2 = uy + vx
    abs_def = np.sqrt(r1*r1 + r2*r2)
    scale = np.sqrt(ux*ux + uy*uy) + np.sqrt(vx*vx + vy*vy) + eps
    rel_def = abs_def / scale
    return abs_def, rel_def


def boundary_distance(points, triangles, poly: Polygon):
    cent = tri_centroids(points, triangles)
    return np.array([poly.exterior.distance(Point(float(c[0]), float(c[1]))) for c in cent])


def binned_median(x, y, bin_edges):
    out = []
    for a,b in zip(bin_edges[:-1], bin_edges[1:]):
        m = (x >= a) & (x < b)
        if np.any(m):
            out.append((float(a), float(b), float(np.median(y[m])), int(np.sum(m))))
        else:
            out.append((float(a), float(b), float('nan'), 0))
    return out

def run_experiment(h_L, h_C, boundary_h, tag):
    inv_eigs = compute_inverse_eigenvalues(N_MIN, N_MAX)
    poly_L = alpha_shape_polygon(inv_eigs, ALPHA_FIXED)

    # --- Meshes
    P_L, T_L = polygon_to_mesh(poly_L, h=h_L, boundary_h=boundary_h,
                              boundary_layers=1, layer_factor=2.0, verbose=True, seed=0)

    poly_C = cardioid_polygon(n=401)
    P_C, T_C = polygon_to_mesh(poly_C, h=h_C, boundary_h=boundary_h,
                              boundary_layers=1, layer_factor=2.0, verbose=True, seed=0)

    # --- θ-iteration with normalization (each domain separately)
    u_L, v_L, cL, rL, perL = solve_uv_with_theta_iteration(P_L, T_L, poly_L, tag=f"{tag}-Lucas")
    u_C, v_C, cC, rC, perC = solve_uv_with_theta_iteration(P_C, T_C, poly_C, tag=f"{tag}-Cardioid")

    # --- Boundary rotation alignment AFTER normalization (prevents insane affine fits)
    bL, _, _ = boundary_order_by_arclength(P_L, T_L, poly_L)
    bC, _, _ = boundary_order_by_arclength(P_C, T_C, poly_C)

    wL_b = u_L[bL] + 1j * v_L[bL]
    wC_b = u_C[bC] + 1j * v_C[bC]

    # resample to same length by index mapping
    m = min(len(wL_b), len(wC_b))
    wL_s = wL_b[:m]
    wC_s = wC_b[:m]
    rot = optimal_rotation(wL_s, wC_s)

    # apply rotation to all Lucas uv
    wL = (u_L + 1j * v_L) * rot
    uv_L = np.column_stack([wL.real, wL.imag])


    # --- CR defect diagnostics on the (u,v) chart itself (before inversion)
    # Use the rotated Lucas chart for consistency.
    uLr, vLr = uv_L[:,0], uv_L[:,1]
    abs_cr_L_all, rel_cr_L_all = cr_defect_metrics(P_L, T_L, uLr, vLr, eps=CR_REL_EPS)
    abs_cr_C_all, rel_cr_C_all = cr_defect_metrics(P_C, T_C, u_C, v_C, eps=CR_REL_EPS)

    # --- Build uv->z inverter on cardioid
    uvC = np.column_stack([u_C, v_C])
    zC  = P_C[:, 0] + 1j * P_C[:, 1]

    uvC_round = np.round(uvC, 12)
    _, idx = np.unique(uvC_round, axis=0, return_index=True)
    uv_C_nodes = uvC[idx]
    z_C_nodes  = zC[idx]

    # --- Invert Lucas uv to cardioid z
    phi_nodes, ok_nodes, _ = invert_uv_to_z(uv_L, uv_C_nodes, z_C_nodes)
    valid = ok_nodes & np.isfinite(phi_nodes.real) & np.isfinite(phi_nodes.imag)

    # --- QC on ALL triangles
    mus_all, Ks_all, used_all = beltrami_K_on_triangles(P_L, T_L, phi_nodes, valid)
    ang_all = angle_distortion_on_triangles(P_L, T_L, phi_nodes, valid)
    mu_L2_all = float(np.sqrt(np.mean(np.abs(mus_all)**2))) if len(mus_all) else np.nan
    K_med_all = float(np.median(Ks_all)) if len(Ks_all) else np.nan
    ang_med_all = float(np.median(ang_all)) if len(ang_all) else np.nan


    # --- Summaries for CR defects (all triangles)
    cr_summary = dict(
        lucas=dict(
            abs_med=float(np.median(abs_cr_L_all)),
            abs_p90=float(np.quantile(abs_cr_L_all, 0.9)),
            rel_med=float(np.median(rel_cr_L_all)),
            rel_p90=float(np.quantile(rel_cr_L_all, 0.9)),
            tris=int(len(abs_cr_L_all)),
        ),
        cardioid=dict(
            abs_med=float(np.median(abs_cr_C_all)),
            abs_p90=float(np.quantile(abs_cr_C_all, 0.9)),
            rel_med=float(np.median(rel_cr_C_all)),
            rel_p90=float(np.quantile(rel_cr_C_all, 0.9)),
            tris=int(len(abs_cr_C_all)),
        )
    )

    # --- Where is QC distortion coming from? Bin by boundary-distance on Lucas mesh.
    d_all = boundary_distance(P_L, T_L, poly_L)
    # Use interior delta = 2*h_L as a standard slice (can compare across refinements)
    d0 = 2.0 * h_L
    mask_ref = d_all >= d0
    if np.any(mask_ref) and len(Ks_all):
        # Need Ks aligned with *all* Lucas triangles for distance-binning diagnostics.
        # beltrami_K_full returns an array of length len(T_L), with NaN for invalid/degenerate.
        Ks_full, _tri_used = beltrami_K_full(P_L, T_L, phi_nodes, valid)
        x = d_all[mask_ref]
        y = Ks_full[mask_ref]
        good = np.isfinite(y)
        if np.any(good):
            q = np.quantile(x[good], [0,0.25,0.5,0.75,1.0])
            bins = binned_median(x[good], y[good], q)
        else:
            bins = []
    else:
        bins = []

    # --- Interior sweeps
    sweep = []
    for fac in DELTA_SWEEP_FACTORS:
        delta = fac * h_L
        interior_mask = select_interior_triangles(P_L, T_L, poly_L, delta)
        T_int = T_L[interior_mask]
        mus, Ks, used = beltrami_K_on_triangles(P_L, T_int, phi_nodes, valid)
        ang = angle_distortion_on_triangles(P_L, T_int, phi_nodes, valid)

        sweep.append(dict(
            delta_factor=float(fac),
            delta=float(delta),
            used_tris=int(used),
            mu_L2=float(np.sqrt(np.mean(np.abs(mus)**2))) if len(mus) else np.nan,
            K_median=float(np.median(Ks)) if len(Ks) else np.nan,
            angle_median=float(np.median(ang)) if len(ang) else np.nan,
        ))

    return dict(
        tag=tag,
        h_L=h_L, h_C=h_C, boundary_h=boundary_h,
        valid_frac=float(np.mean(valid)),
        rot=rot,
        period_mismatch=dict(lucas=float(perL), cardioid=float(perC)),
        all=dict(
            used_tris=int(used_all),
            mu_L2=mu_L2_all,
            K_median=K_med_all,
            angle_median=ang_med_all
        ),
        cr=cr_summary,
        K_bins_d2h=bins,
        sweep=sweep
    )



def _to_jsonable(x):
    """Convert numpy/scalar/complex containers to JSON-serializable objects."""
    import numpy as _np
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, complex):
        return {"re": float(x.real), "im": float(x.imag)}
    if isinstance(x, _np.generic):
        # numpy scalar
        if _np.iscomplexobj(x):
            return {"re": float(_np.real(x)), "im": float(_np.imag(x))}
        return float(x)
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, _np.ndarray):
        return _to_jsonable(x.tolist())
    return str(x)

def save_results(results, out_dir="out_v18"):
    """Write JSON/CSV summaries + a few quick diagnostic plots."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- JSON (full)
    with (out / "results.json").open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(results), f, indent=2)

    # --- CSV (compact)
    fieldnames = [
        "tag","h_L","h_C","boundary_h","valid_frac",
        "K_med_all","mu_L2_all","angle_med_all",
        "crL_rel_med","crL_rel_p90","crC_rel_med","crC_rel_p90",
        "perL","perC"
    ]
    with (out / "results_compact.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            crL = (r.get("cr") or {}).get("lucas", {})
            crC = (r.get("cr") or {}).get("cardioid", {})
            per = r.get("period_mismatch", {})
            w.writerow({
                "tag": r.get("tag"),
                "h_L": r.get("h_L"),
                "h_C": r.get("h_C"),
                "boundary_h": r.get("boundary_h"),
                "valid_frac": r.get("valid_frac"),
                "K_med_all": (r.get("all") or {}).get("K_median"),
                "mu_L2_all": (r.get("all") or {}).get("mu_L2"),
                "angle_med_all": (r.get("all") or {}).get("angle_median"),
                "crL_rel_med": crL.get("rel_med"),
                "crL_rel_p90": crL.get("rel_p90"),
                "crC_rel_med": crC.get("rel_med"),
                "crC_rel_p90": crC.get("rel_p90"),
                "perL": per.get("lucas"),
                "perC": per.get("cardioid"),
            })

    # --- Plot: K_med vs distance-bin for each refinement (delta>=2*h_L bins)
    for r in results:
        bins = r.get("K_bins_d2h") or []
        if not bins:
            continue
        mids = [(a+b)/2.0 for a,b,_,_ in bins]
        Kmed = [k for _,_,k,_ in bins]
        counts = [n for *_,n in bins]
        plt.figure()
        plt.plot(mids, Kmed, marker="o")
        plt.xlabel("boundary-distance bin midpoint (d)")
        plt.ylabel("median K in bin")
        plt.title(f"K vs distance bins ({r.get('tag')})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out / f"K_bins_{r.get('tag')}.png", dpi=180)
        plt.close()

        plt.figure()
        plt.plot(mids, counts, marker="o")
        plt.xlabel("boundary-distance bin midpoint (d)")
        plt.ylabel("triangles per bin")
        plt.title(f"bin counts ({r.get('tag')})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out / f"bin_counts_{r.get('tag')}.png", dpi=180)
        plt.close()

    return str(out)


def main():
    # --- NEW: export ordered boundary for v20 (or reuse later)
    if not os.path.exists("lucas_points.npy"):
     export_lucas_boundary_npy(
        n_min=N_MIN,
        n_max=N_MAX,
        alpha=ALPHA_FIXED,      # or 3.5 if you want; you already have ALPHA_FIXED=4.5
        n_boundary=2000,
        out_path="lucas_points.npy",
    )
    print(f"[main] Lucas inverse-eigenvalue points: {len(compute_inverse_eigenvalues(N_MIN, N_MAX))}")
    print(f"[main] Delta sweep factors: {DELTA_SWEEP_FACTORS} (delta = factor * h_L)")
    print(f"[main] Theta iters: {THETA_ITERS}, relax={THETA_RELAX}, smooth={THETA_SMOOTH}\n")

    results = []
    for lvl in REFINEMENT_LEVELS:
        print("\n" + "="*60)
        print(f"[main] Refinement {lvl}")
        print("="*60)

        res = run_experiment(
            h_L=lvl["h_L"],
            h_C=lvl["h_C"],
            boundary_h=lvl["boundary_h"],
            tag=lvl["name"]
        )

        print(f"valid={res['valid_frac']:.3f}  rot≈{res['rot']}")
        a = res["all"]
        print(f"  ALL: used={a['used_tris']:<6d}  mu_L2={a['mu_L2']:.3e}  K_med={a['K_median']:.3f}  angle_med={a['angle_median']:.3e}")

        if 'cr' in res:
            crL=res['cr']['lucas']; crC=res['cr']['cardioid']
            print(f"  CR (Lucas chart):   rel_med={crL['rel_med']:.3e}  rel_p90={crL['rel_p90']:.3e}  abs_med={crL['abs_med']:.3e}")
            print(f"  CR (Cardioid chart): rel_med={crC['rel_med']:.3e}  rel_p90={crC['rel_p90']:.3e}  abs_med={crC['abs_med']:.3e}")
        if res.get('K_bins_d2h'):
            print("  K_med binned by boundary distance (delta>=2*h_L; quartile bins):")
            for a,b,km,n in res['K_bins_d2h']:
                print(f"    d∈[{a:.4g},{b:.4g}):  K_med={km:.3f}  (n={n})")
        print("  Interior sweep (delta_factor, used, K_med, mu_L2, angle_med):")
        for sw in res["sweep"]:
            print(f"    {sw['delta_factor']:<4.1f} used={sw['used_tris']:<6d}  K_med={sw['K_median']:.3f}  mu_L2={sw['mu_L2']:.3e}  angle_med={sw['angle_median']:.3e}")

        results.append(res)

    
    # --- Save artifacts (JSON/CSV/plots)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"lucas_cardioid_v18_{timestamp}"
    saved_path = save_results(results, out_dir=out_dir)
    print(f"[main] wrote artifacts to: {saved_path}")
    print("\n=== Summary (dicts) ===")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
