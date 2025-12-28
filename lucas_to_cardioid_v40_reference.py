#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lucas_to_cardioid_v40_reference.py

v40_reference: FIRST stable, referential Lucas->disk->cardioid uniformization via boundary-integral potential theory.

Why "reference" (historical path in one paragraph):
- v3x versions tried to recover the Riemann map f by path integrating Φ'(z) and then using Re Φ(z) as a Green function.
  In practice that mixes analytic continuation error with the modulus and caused exponential amplification inside the domain.
- v39 fixed the key issue: the modulus is defined from the *real* Green representation

      g(z,a) = -log|z-a| + ∫_{∂Ω} σ(ζ) log|z-ζ| ds(ζ) + C ,

  where (σ,C) are fitted by enforcing g≈0 on boundary-in points and ∫σ ds = 0.
  Then |f(z)| = exp(-g(z,a)), while the phase comes from Im Φ_raw via path integration of Φ'(z).
  This cleanly separates (i) a stable potential-theoretic modulus from (ii) a path-integrated harmonic conjugate.

What this script produces (for sharing + paper figures):
- A single-row CSV with the core numerical diagnostics (boundary modulus, boundary residuals, g statistics, interior radii stats, inverse-check stats).
- A radii histogram CSV for |w_raw|.
- Color-correspondence plots showing the boundary map in both directions:
    Lucas boundary (parameter t) -> disk boundary (same t),
    disk boundary -> cardioid boundary (exact polynomial map).
- Standard scatter plots for interior samples in Lucas / disk / cardioid.

Notes:
- Even though the internal version label is "v40", this is the first implementation that is mathematically correct *and* numerically stable
  enough to be treated as a reference baseline for ICM2026 exposition.

"""

import os
import math
import datetime
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from shapely.geometry import Point, Polygon
import alphashape

try:
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except Exception:
    HAVE_PLT = False


# -----------------------------
# Knobs
# -----------------------------
ALPHA_FIXED = 4.5
N_BDY = 2000

PATH_GAUSS_N = 16
EPS_POLE = 1e-6

DZ_EPS = 1e-14
EXP_CLIP = 60.0
RIDGE_LAMBDA = 1e-8

TARGET_R_CLAMP = 0.995

ENABLE_JITTER = True
INWARD_EPS = 1e-3

# Interior sampling
INTERIOR_N = 20000
INTERIOR_SEED = 0
INTERIOR_MAX_TRIES = 2_000_000

# Chunk size for evaluating g(z) to avoid huge (M x N) matrices
G_CHUNK = 600

DO_EXACT_INVERSE_CHECK = True

OUTDIR = None


# -----------------------------
# Geometry helpers
# -----------------------------
def polygon_from_points_alpha(points_xy, alpha=ALPHA_FIXED):
    shp = alphashape.alphashape(points_xy, alpha)
    if shp.geom_type == "Polygon":
        return shp
    if shp.geom_type == "MultiPolygon":
        polys = list(shp.geoms)
        polys.sort(key=lambda p: p.area, reverse=True)
        return polys[0]
    raise ValueError(f"alphashape returned geometry type {shp.geom_type}")


def sample_polygon_boundary(poly: Polygon, n=N_BDY):
    if not poly.exterior.is_ccw:
        poly = Polygon(list(poly.exterior.coords)[::-1])

    coords = np.array(poly.exterior.coords, dtype=float)
    if np.allclose(coords[0], coords[-1]):
        coords = coords[:-1]

    seg = np.roll(coords, -1, axis=0) - coords
    seglen = np.sqrt((seg**2).sum(axis=1))
    s = np.concatenate([[0.0], np.cumsum(seglen)])
    L = s[-1]
    if L <= 0:
        raise ValueError("Degenerate polygon boundary length.")

    su = np.linspace(0, L, n, endpoint=False)
    idx = np.searchsorted(s, su, side="right") - 1
    idx = np.clip(idx, 0, len(seglen) - 1)
    t = (su - s[idx]) / np.maximum(seglen[idx], 1e-15)
    pts = coords[idx] + seg[idx] * t[:, None]

    z = pts[:, 0] + 1j * pts[:, 1]
    ds = np.full(n, L / n, dtype=float)
    return z, ds


def ensure_interior_point(poly: Polygon, z0: complex):
    c = poly.centroid
    cc = c.x + 1j * c.y
    z = z0
    if poly.contains(Point(z.real, z.imag)):
        return z
    for _ in range(60):
        z = 0.5 * z + 0.5 * cc
        if poly.contains(Point(z.real, z.imag)):
            return z
    return cc


def slightly_inside(z: np.ndarray, a: complex, eps: float = INWARD_EPS) -> np.ndarray:
    z = np.asarray(z, dtype=np.complex128)
    return (1.0 - eps) * z + eps * a


def clamp_to_disk(w: complex, rmax: float = TARGET_R_CLAMP) -> complex:
    r = abs(w)
    if not np.isfinite(r):
        return complex(np.nan, np.nan)
    if r <= rmax:
        return w
    return w * (rmax / r)


def sample_interior_points(poly: Polygon, n: int, seed: int = 0, max_tries: int = 2_000_000):
    rng = np.random.default_rng(seed)
    minx, miny, maxx, maxy = poly.bounds
    out = np.empty(n, dtype=np.complex128)
    k = 0
    tries = 0
    while k < n and tries < max_tries:
        tries += 1
        x = rng.uniform(minx, maxx)
        y = rng.uniform(miny, maxy)
        if poly.contains(Point(float(x), float(y))):
            out[k] = x + 1j * y
            k += 1
    return out[:k], tries


# -----------------------------
# Gauss–Legendre on [0,1]
# -----------------------------
def gauss_legendre_01(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w


def safe_exp_minus_real(g: np.ndarray) -> np.ndarray:
    """Return exp(-g) with clipping to avoid overflow/underflow."""
    g = np.asarray(g, dtype=float)
    a = -g
    a = np.clip(a, -EXP_CLIP, EXP_CLIP)
    return np.exp(a)


# -----------------------------
# Riemann map object (Lucas only)
# -----------------------------
@dataclass
class RiemannMapDisk_GreenModulus:
    bdy_z: np.ndarray
    ds: np.ndarray
    sigma: np.ndarray
    a: complex
    C: float

    # shift applied to g(z) so median g(bdy-in)=0
    g_shift: float = 0.0

    _gx: np.ndarray = None
    _gw: np.ndarray = None

    def __post_init__(self):
        self._gx, self._gw = gauss_legendre_01(PATH_GAUSS_N)

    def dPhi(self, z: np.ndarray) -> np.ndarray:
        """Evaluate Φ'(z) = -1/(z-a) + ∫ σ(ζ)/(z-ζ) ds(ζ) (Nyström)."""
        z = np.asarray(z, dtype=np.complex128).ravel()
        DZ0 = z - self.a
        DZ0 = np.where(np.abs(DZ0) < DZ_EPS, DZ_EPS + 0j, DZ0)

        DZ = z[:, None] - self.bdy_z[None, :]
        DZ = np.where(np.abs(DZ) < DZ_EPS, DZ_EPS + 0j, DZ)

        integ = (self.sigma[None, :] * self.ds[None, :]) / DZ
        return -1.0 / DZ0 + integ.sum(axis=1)

    def Phi_raw(self, z: np.ndarray) -> np.ndarray:
        """Complex Φ from path integrating Φ'(z). (Used only for Im part / phase.)"""
        z = np.asarray(z, dtype=np.complex128).ravel()
        out = np.empty_like(z)

        for k, zk in enumerate(z):
            if zk == self.a:
                out[k] = np.inf + 0j
                continue

            direction = (zk - self.a) / abs(zk - self.a)
            z0 = self.a + EPS_POLE * direction

            seg = zk - z0
            xi = z0 + self._gx * seg
            dphi = self.dPhi(xi)
            integral = (dphi * seg) @ self._gw

            # Anchor only the REAL part here (imag anchor = 0 at z0).
            # But we will NOT use this real part for modulus.
            real_sl = float(np.sum(self.sigma * self.ds * np.log(np.abs(z0 - self.bdy_z) + 1e-300)))
            phi0 = (-math.log(EPS_POLE) + real_sl + self.C) + 0j

            out[k] = phi0 + integral

        return out

    def g_real(self, z: np.ndarray) -> np.ndarray:
        """
        Real Green representation:
            g(z) = -log|z-a| + ∫ σ log|z-ζ| ds + C + g_shift
        Evaluated in chunks to avoid huge temporary arrays.
        """
        z = np.asarray(z, dtype=np.complex128).ravel()
        out = np.empty(z.shape[0], dtype=float)

        sigw = (self.sigma * self.ds).astype(float)  # (N,)
        for i0 in range(0, z.shape[0], G_CHUNK):
            i1 = min(i0 + G_CHUNK, z.shape[0])
            zz = z[i0:i1]  # (m,)
            logabs = np.log(np.abs(zz[:, None] - self.bdy_z[None, :]) + 1e-300)
            sl = logabs @ sigw
            out[i0:i1] = (-np.log(np.abs(zz - self.a) + 1e-300) + sl + self.C + self.g_shift)

        return out

    def Phi(self, z: np.ndarray) -> np.ndarray:
        """Composite Φ with Re part from g_real and Im part from Phi_raw."""
        z = np.asarray(z, dtype=np.complex128).ravel()
        phi_raw = self.Phi_raw(z)
        g = self.g_real(z)
        return g + 1j * phi_raw.imag

    def f(self, z: np.ndarray) -> np.ndarray:
        """Riemann map: f(z) = exp(-g(z)) * exp(-i ImPhi(z))."""
        z = np.asarray(z, dtype=np.complex128).ravel()
        phi_raw = self.Phi_raw(z)
        g = self.g_real(z)
        amp = safe_exp_minus_real(g)
        return amp * np.exp(-1j * phi_raw.imag)


# -----------------------------
# Fit sigma and C
# -----------------------------
def fit_riemann_map_to_disk(poly: Polygon, n_bdy=N_BDY, a=None, verbose=True):
    z, ds = sample_polygon_boundary(poly, n=n_bdy)

    if a is None:
        c = poly.centroid
        a = c.x + 1j * c.y
    a = ensure_interior_point(poly, a)

    N = len(z)

    Zi = z[:, None]
    Zj = z[None, :]
    absD = np.abs(Zi - Zj)

    # log|x-ζ| kernel with a crude diagonal surrogate
    K = np.log(absD + 1e-300)
    diag = np.diag_indices_from(K)
    K[diag] = np.log(np.maximum(ds, 1e-300) / 2.0) - 1.0

    Kds = K * ds[None, :]

    # Least squares: (Kds) sigma + C = log|z-a|
    A = np.zeros((N, N + 1), dtype=float)
    A[:, :N] = Kds
    A[:, N] = 1.0
    b = np.log(np.abs(z - a) + 1e-300).astype(float)

    # Constraint: ∫σ ds = 0
    A_con = np.zeros((1, N + 1), dtype=float)
    A_con[0, :N] = ds
    b_con = np.array([0.0], dtype=float)

    A0 = np.vstack([A, A_con])
    b0 = np.concatenate([b, b_con])

    lam = float(RIDGE_LAMBDA)
    if lam > 0:
        A_reg = np.zeros((N, N + 1), dtype=float)
        A_reg[:, :N] = math.sqrt(lam) * np.eye(N)
        b_reg = np.zeros(N, dtype=float)
        A_aug = np.vstack([A0, A_reg])
        b_aug = np.concatenate([b0, b_reg])
    else:
        A_aug, b_aug = A0, b0

    x, *_ = np.linalg.lstsq(A_aug, b_aug, rcond=None)
    sigma = x[:N]
    C = float(x[N])

    # Robust C recomputation
    C = float(np.median(np.log(np.abs(z - a) + 1e-300) - (Kds @ sigma)))

    rm = RiemannMapDisk_GreenModulus(bdy_z=z, ds=ds, sigma=sigma, a=a, C=C, g_shift=0.0)

    # Calibrate g_shift so median g on boundary-in is 0
    z_in = slightly_inside(z, a, INWARD_EPS)
    g0 = float(np.median(rm.g_real(z_in)))
    rm.g_shift = -g0

    if verbose:
        fb = rm.f(z_in)
        mod = np.abs(fb)
        print(
            f"[riemann v40_reference] a={a.real:+.6f}{a.imag:+.6f}i  "
            f"|f(bdy-in)| median={np.median(mod):.9f}  p90={np.quantile(mod,0.90):.9f}"
        )
        print(f"[riemann v40_reference] |f(bdy-in)| min/max = {mod.min():.9f} / {mod.max():.9f}")

        # boundary residual check
        r = (Kds @ sigma) + C - np.log(np.abs(z - a) + 1e-300)
        print(
            f"[riemann v40_reference] bdy-resid: median={np.median(r):+.3e}  "
            f"p90={np.quantile(np.abs(r),0.90):.3e}  maxabs={np.max(np.abs(r)):.3e}"
        )
        # g check
        g_in = rm.g_real(z_in)
        print(
            f"[riemann v40_reference] g(bdy-in) after shift: min={g_in.min():+.3e}  "
            f"median={np.median(g_in):+.3e}  max={g_in.max():+.3e}"
        )
        print(f"[riemann v40_reference] g_shift={rm.g_shift:+.6e}  (median g_raw(bdy-in)={g0:+.6e})")

    return rm


# -----------------------------
# Exact disk<->cardioid maps
# -----------------------------
def disk_to_cardioid(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.complex128)
    return 0.5 * w - 0.25 * w * w


def cardioid_to_disk(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.complex128)
    return 1.0 - np.sqrt(1.0 - 4.0 * z)


def cardioid_polygon(num=4000) -> Polygon:
    """Return a shapely Polygon approximating the cardioid boundary."""
    t = np.linspace(0, 2*np.pi, num, endpoint=False)
    z = 0.5*np.exp(1j*t) - 0.25*np.exp(2j*t)
    coords = np.column_stack([z.real, z.imag])
    return Polygon(coords)


# -----------------------------
# Export helpers (CSV + plots)
# -----------------------------
def write_summary_csv(path, rows):
    """Write a one-row (or few-row) CSV with stable column order."""
    keys = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_hist_csv(path, values, bins=80, range_=None):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    hist, edges = np.histogram(values, bins=bins, range=range_, density=False)
    centers = 0.5 * (edges[:-1] + edges[1:])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bin_left", "bin_right", "bin_center", "count"])
        for i in range(len(hist)):
            w.writerow([float(edges[i]), float(edges[i+1]), float(centers[i]), int(hist[i])])


def plot_boundary_correspondence(z_bdy, w_bdy, outpath, title):
    """Plot boundary correspondence using a parameter-color along the boundary."""
    if not HAVE_PLT:
        return
    z_bdy = np.asarray(z_bdy, dtype=np.complex128).ravel()
    w_bdy = np.asarray(w_bdy, dtype=np.complex128).ravel()
    t = np.linspace(0.0, 1.0, len(z_bdy), endpoint=False)

    fig = plt.figure(figsize=(10, 4.5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.scatter(z_bdy.real, z_bdy.imag, c=t, s=6, cmap="hsv")
    ax1.set_title("Lucas boundary (t-colored)")
    ax1.set_aspect("equal", "box")

    ax2.scatter(w_bdy.real, w_bdy.imag, c=t, s=6, cmap="hsv")
    th = np.linspace(0, 2*np.pi, 800, endpoint=False)
    circ = np.cos(th) + 1j*np.sin(th)
    ax2.plot(circ.real, circ.imag, "-", linewidth=1)
    ax2.set_title("Mapped boundary in disk (same t)")
    ax2.set_aspect("equal", "box")

    fig.suptitle(title)
    fig.colorbar(ax2.collections[0], ax=[ax1, ax2], fraction=0.046, pad=0.04, label="boundary parameter t")
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_disk_to_cardioid_correspondence(outpath, num=1200):
    if not HAVE_PLT:
        return
    t = np.linspace(0.0, 1.0, num, endpoint=False)
    th = 2*np.pi*t
    w = np.exp(1j*th)
    z = disk_to_cardioid(w)

    fig = plt.figure(figsize=(10, 4.5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.scatter(w.real, w.imag, c=t, s=6, cmap="hsv")
    ax1.set_title("Unit circle (t-colored)")
    ax1.set_aspect("equal", "box")

    ax2.scatter(z.real, z.imag, c=t, s=6, cmap="hsv")
    polyC = cardioid_polygon(num=5000)
    b = np.array(polyC.exterior.coords)
    ax2.plot(b[:, 0], b[:, 1], "-", linewidth=1)
    ax2.set_title("Cardioid boundary (exact map, same t)")
    ax2.set_aspect("equal", "box")

    fig.suptitle("Exact disk → cardioid boundary correspondence")
    fig.colorbar(ax2.collections[0], ax=[ax1, ax2], fraction=0.046, pad=0.04, label="boundary parameter t")
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------
def main():
    global OUTDIR
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTDIR = f"lucas_cardioid_v40_reference_{stamp}"
    os.makedirs(OUTDIR, exist_ok=True)
    outdir = Path(OUTDIR)

    print("=" * 72)
    print("[v40_reference] Lucas->disk (Green modulus) + disk->cardioid (exact) with interior sampling")
    print("=" * 72)
    print(f"[v40_reference] Output dir: {OUTDIR}")
    print(f"[v40_reference] N_BDY={N_BDY} PATH_GAUSS_N={PATH_GAUSS_N}  ridge={RIDGE_LAMBDA:g}")
    print(f"[v40_reference] INTERIOR_N={INTERIOR_N}  ENABLE_JITTER={ENABLE_JITTER}  DO_EXACT_INVERSE_CHECK={DO_EXACT_INVERSE_CHECK}")

    if not os.path.exists("lucas_points.npy"):
        raise FileNotFoundError("v40_reference expects lucas_points.npy in the working directory (N,2) array.")

    pts = np.load("lucas_points.npy")
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("lucas_points.npy must be an (N,2) array of x,y.")

    pts = np.ascontiguousarray(pts, dtype=float)
    pts = np.unique(pts, axis=0)
    if ENABLE_JITTER:
        rng = np.random.default_rng(0)
        pts = pts + 1e-12 * rng.standard_normal(pts.shape)

    polyL = polygon_from_points_alpha(pts, alpha=ALPHA_FIXED)
    print(f"[v40_reference] Lucas polygon area={polyL.area:.6f}  bounds={polyL.bounds}")

    print("[v40_reference] Fit Lucas domain -> disk")
    rmL = fit_riemann_map_to_disk(polyL, n_bdy=N_BDY, verbose=True)

    # interior sampling
    z_int, tries = sample_interior_points(polyL, INTERIOR_N, seed=INTERIOR_SEED, max_tries=INTERIOR_MAX_TRIES)
    if len(z_int) < INTERIOR_N:
        print(f"[v40_reference] WARNING: only sampled {len(z_int)} interior points after {tries} tries (cap={INTERIOR_MAX_TRIES}).")
    else:
        print(f"[v40_reference] sampled {len(z_int)} interior points after {tries} tries.")

    # requested diagnostic
    RePhi_int = rmL.Phi(z_int).real
    print("[v40_reference] RePhi(z_int): min/median/max =",
          float(np.min(RePhi_int)), float(np.median(RePhi_int)), float(np.max(RePhi_int)))

    # map to disk (unclamped)
    w_raw = rmL.f(z_int)
    rad_raw = np.abs(w_raw)
    finite = np.isfinite(rad_raw)
    rad_raw_f = rad_raw[finite]
    print(
        f"[v40_reference] disk radii |w_raw| (finite={finite.mean():.3f}): "
        f"median={np.median(rad_raw_f):.6f}  p90={np.quantile(rad_raw_f,0.90):.6f}  max={np.max(rad_raw_f):.6f}"
    )

    # clamp (for safety, but ideally should be unnecessary now)
    w = np.array([clamp_to_disk(wi, TARGET_R_CLAMP) for wi in w_raw], dtype=np.complex128)
    rad = np.abs(w)
    print(
        f"[v40_reference] disk radii |w| (clamped rmax={TARGET_R_CLAMP}): "
        f"median={np.median(rad):.6f}  p90={np.quantile(rad,0.90):.6f}  max={np.max(rad):.6f}"
    )

    mapped = disk_to_cardioid(w)

    if DO_EXACT_INVERSE_CHECK:
        w_back = cardioid_to_disk(mapped)
        err = np.abs(w_back - w)
        print(
            f"[v40_reference] exact inverse check: median|w_back-w|={np.median(err):.3e}  "
            f"p90={np.quantile(err,0.90):.3e}  max={np.max(err):.3e}"
        )
    else:
        err = np.array([])

    if HAVE_PLT:
        # Lucas interior
        fig = plt.figure()
        plt.title("Lucas interior samples")
        plt.plot(z_int.real, z_int.imag, ".", markersize=1, alpha=0.25)
        plt.axis("equal")
        fig.savefig(outdir / "lucas_interior_samples_v40_reference.png", dpi=220)
        plt.close(fig)

        # disk image
        fig = plt.figure()
        plt.title("Lucas -> disk (w_raw)")
        plt.plot(w_raw.real, w_raw.imag, ".", markersize=1, alpha=0.25)
        th = np.linspace(0, 2*np.pi, 800, endpoint=False)
        circ = np.cos(th) + 1j*np.sin(th)
        plt.plot(circ.real, circ.imag, "-", linewidth=1)
        plt.axis("equal")
        fig.savefig(outdir / "lucas_to_disk_v40_reference.png", dpi=220)
        plt.close(fig)

        # cardioid image
        polyC = cardioid_polygon(num=6000)
        b = np.array(polyC.exterior.coords)
        fig = plt.figure()
        plt.title("Lucas -> cardioid via exact disk map")
        plt.plot(mapped.real, mapped.imag, ".", markersize=1, alpha=0.25, label="mapped")
        plt.plot(b[:, 0], b[:, 1], "-", linewidth=1, label="cardioid boundary")
        plt.axis("equal")
        plt.legend()
        fig.savefig(outdir / "lucas_to_cardioid_exact_v40_reference.png", dpi=220)
        plt.close(fig)

    # ------------------------------------------------------------
    # Exports for sharing (CSV diagnostics + correspondence plots)
    # ------------------------------------------------------------
    z_bdy = rmL.bdy_z
    z_bdy_in = slightly_inside(z_bdy, rmL.a, INWARD_EPS)

    w_bdy_in = rmL.f(z_bdy_in)
    mod_bdy = np.abs(w_bdy_in)

    Zi = z_bdy[:, None]
    Zj = z_bdy[None, :]
    absD = np.abs(Zi - Zj)
    K = np.log(absD + 1e-300)
    diag = np.diag_indices_from(K)
    K[diag] = np.log(np.maximum(rmL.ds, 1e-300) / 2.0) - 1.0
    Kds = K * rmL.ds[None, :]
    resid = (Kds @ rmL.sigma) + rmL.C - np.log(np.abs(z_bdy - rmL.a) + 1e-300)

    g_bdy_in = rmL.g_real(z_bdy_in)

    summary_row = dict(
        version="v40_reference",
        timestamp=stamp,
        N_BDY=int(N_BDY),
        PATH_GAUSS_N=int(PATH_GAUSS_N),
        RIDGE_LAMBDA=float(RIDGE_LAMBDA),
        INWARD_EPS=float(INWARD_EPS),
        INTERIOR_N=int(len(z_int)),
        a_real=float(rmL.a.real),
        a_imag=float(rmL.a.imag),
        g_shift=float(rmL.g_shift),
        bdy_mod_median=float(np.median(mod_bdy)),
        bdy_mod_p90=float(np.quantile(mod_bdy, 0.90)),
        bdy_mod_min=float(np.min(mod_bdy)),
        bdy_mod_max=float(np.max(mod_bdy)),
        bdy_resid_median=float(np.median(resid)),
        bdy_resid_p90_abs=float(np.quantile(np.abs(resid), 0.90)),
        bdy_resid_max_abs=float(np.max(np.abs(resid))),
        g_bdy_in_min=float(np.min(g_bdy_in)),
        g_bdy_in_median=float(np.median(g_bdy_in)),
        g_bdy_in_max=float(np.max(g_bdy_in)),
        RePhi_int_min=float(np.min(RePhi_int)),
        RePhi_int_median=float(np.median(RePhi_int)),
        RePhi_int_max=float(np.max(RePhi_int)),
        rad_raw_median=float(np.median(rad_raw_f)),
        rad_raw_p90=float(np.quantile(rad_raw_f, 0.90)),
        rad_raw_max=float(np.max(rad_raw_f)),
        rad_clamped_median=float(np.median(rad)),
        rad_clamped_p90=float(np.quantile(rad, 0.90)),
        rad_clamped_max=float(np.max(rad)),
    )
    if len(err) > 0:
        summary_row.update(
            inverse_err_median=float(np.median(err)),
            inverse_err_p90=float(np.quantile(err, 0.90)),
            inverse_err_max=float(np.max(err)),
        )

    write_summary_csv(outdir / "diagnostics_v40_reference.csv", [summary_row])
    write_hist_csv(outdir / "radii_hist_w_raw_v40_reference.csv", rad_raw_f, bins=80, range_=(0.0, 1.05))

    plot_boundary_correspondence(
        z_bdy=z_bdy,
        w_bdy=w_bdy_in / (np.abs(w_bdy_in) + 1e-300),
        outpath=outdir / "bdy_correspondence_lucas_to_disk_v40_reference.png",
        title="Boundary parameter correspondence: Lucas → disk (Riemann map)",
    )
    plot_disk_to_cardioid_correspondence(
        outpath=outdir / "bdy_correspondence_disk_to_cardioid_v40_reference.png",
        num=1400,
    )

    np.savez(
        outdir / "lucas_to_cardioid_map_v40_reference.npz",
        lucas_interior=z_int,
        disk_points_raw=w_raw,
        disk_points=w,
        cardioid_points=mapped,
        rmL_a=rmL.a,
        rmL_sigma=rmL.sigma,
        rmL_C=rmL.C,
        rmL_g_shift=rmL.g_shift,
        rmL_bdy=rmL.bdy_z,
        rmL_ds=rmL.ds,
        inverse_err=err,
    )

    print("[v40_reference] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
