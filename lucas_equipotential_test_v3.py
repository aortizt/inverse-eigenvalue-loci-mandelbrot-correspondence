#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lucas_equipotential_test_v3.py

Spyder-friendly script to:
  • Generate inverse eigenvalues for Lucas companion matrices (and optional families)
  • Compute Mandelbrot parameter potential g_M(c) and Phi(c)
  • Plot empirical density of g_M (hist + KDE)
  • Compare to reference laws: uniform in g, exponential in g, log-uniform in |Phi|
  • Produce per-n and cumulative “convergence” curves (escaped fraction, median/std of g outside)
  • (Optional) compare multiple companion-matrix families via KDE overlays

Outputs are written into ./equipotential_v3_out/

Run: press F5 in Spyder.
"""
from __future__ import annotations

import os, math, csv
import numpy as np
import matplotlib.pyplot as plt

# Optional KDE (scipy). If scipy isn't installed, we fall back to a smoothed histogram.
try:
    from scipy.stats import gaussian_kde
    HAVE_SCIPY = True
except Exception:
    gaussian_kde = None
    HAVE_SCIPY = False

# ============================================================
# 0) SPYDER KNOBS — edit these
# ============================================================

OUT_DIR = "equipotential_v3_out"

N_MIN = 2
N_MAX = 200

MAX_ITER = 20000
ESCAPE_RADIUS = 2.0

EIG_TOL = 1e-12

HIST_BINS = 120
KDE_GRID_N = 800

RUN_FAMILY_COMPARISON = True

# If you want to test your stored boundary/curve points:
LUCAS_POINTS_NPY = ""  # e.g. "lucas_points.npy"  (leave "" to skip)

# ============================================================
# 1) Companion matrices / families
# ============================================================

def generate_lucas_companion(n: int) -> np.ndarray:
    """Companion matrix for x^n - x^{n-1} - ... - x - 1."""
    C = np.zeros((n, n), dtype=float)
    C[0, :] = 1.0
    for i in range(1, n):
        C[i, i - 1] = 1.0
    return C

def generate_companion_from_toprow(n: int, top: np.ndarray) -> np.ndarray:
    """Generalized companion with given first row, and ones on subdiagonal."""
    top = np.asarray(top, dtype=float).reshape(-1)
    assert top.shape[0] == n
    C = np.zeros((n, n), dtype=float)
    C[0, :] = top
    for i in range(1, n):
        C[i, i - 1] = 1.0
    return C

def family_toprow(name: str, n: int) -> np.ndarray:
    """Define a few toy families. Add more here."""
    if name == "lucas_all_ones":
        return np.ones(n)
    if name == "pell_like_all_twos":
        return 2.0 * np.ones(n)
    if name == "sparser_gap_1_0_1_then_ones":
        top = np.ones(n)
        if n >= 2:
            top[1] = 0.0
        return top
    if name == "padovan_like_0_1_then_ones":
        top = np.ones(n)
        top[0] = 0.0
        return top
    raise ValueError(f"Unknown family '{name}'")

def compute_inverse_eigenvalues(n_min: int, n_max: int, tol: float = 1e-12) -> np.ndarray:
    inv_eigs = []
    for n in range(n_min, n_max + 1):
        C = generate_lucas_companion(n)
        eigs = np.linalg.eigvals(C)
        nonzero = np.abs(eigs) > tol
        inv_eigs.append(1.0 / eigs[nonzero])
        if n % 10 == 0 or n in (n_min, n_max):
            print(f"[lucas] n={n}, eigenvalues={len(eigs)}, kept={np.count_nonzero(nonzero)}")
    inv_eigs = np.concatenate(inv_eigs).astype(np.complex128)
    print(f"[lucas] total points: {len(inv_eigs)}")
    return inv_eigs

def compute_inverse_eigenvalues_family(family: str, n_min: int, n_max: int, tol: float = 1e-12) -> np.ndarray:
    inv_eigs = []
    for n in range(n_min, n_max + 1):
        top = family_toprow(family, n)
        C = generate_companion_from_toprow(n, top)
        eigs = np.linalg.eigvals(C)
        nonzero = np.abs(eigs) > tol
        inv_eigs.append(1.0 / eigs[nonzero])
        if n % 10 == 0 or n in (n_min, n_max):
            print(f"[{family}] n={n}, eigenvalues={len(eigs)}, kept={np.count_nonzero(nonzero)}")
    inv_eigs = np.concatenate(inv_eigs).astype(np.complex128)
    print(f"[{family}] total points: {len(inv_eigs)}")
    return inv_eigs

# ============================================================
# 2) Mandelbrot parameter potential g_M(c) + Phi(c)
# ============================================================

def mandelbrot_parameter_potential(c: complex, max_iter: int = 4000, escape_radius: float = 2.0):
    """
    Approximate parameter-plane Green function:
        g_M(c) = lim_{n→∞} 2^{-n} log|f_c^n(0)|,  f_c(z)=z^2+c.

    Returns:
      g   : approx potential (0 if no escape detected)
      k   : escape iteration (or max_iter)
      phi : approx Phi(c) = exp(log(z_k)/2^k) when escaped (else nan)

    Numerical guard:
      Use log_phi = log(z) * 2^{-k} (via np.exp2(-k)) to avoid 2^k overflow.
    """
    z = 0.0 + 0.0j
    R2 = escape_radius * escape_radius

    for k in range(1, max_iter + 1):
        z = z*z + c
        if (z.real*z.real + z.imag*z.imag) > R2:
            log_z = np.log(z)               # principal branch
            log_phi = log_z * np.exp2(-k)   # 2^{-k}
            g = float(log_phi.real)         # log|Phi(c)|
            phi = np.exp(log_phi)
            if (not np.isfinite(g)) or g < 0:
                g = 0.0
            return g, k, phi

    return 0.0, max_iter, np.nan + 1j*np.nan

def batch_potential(C: np.ndarray, max_iter: int = 4000, escape_radius: float = 2.0):
    g = np.empty(len(C), dtype=float)
    it = np.empty(len(C), dtype=int)
    phi = np.empty(len(C), dtype=np.complex128)

    for idx, c in enumerate(C):
        g[idx], it[idx], phi[idx] = mandelbrot_parameter_potential(c, max_iter=max_iter, escape_radius=escape_radius)
        if (idx + 1) % 5000 == 0:
            print(f"[potential] processed {idx+1}/{len(C)}")
    return g, it, phi

# ============================================================
# 3) Stats + CSV helpers
# ============================================================

def summarize_g(g: np.ndarray, label: str = "") -> dict:
    outside = g > 0
    out = {
        "count": int(len(g)),
        "escaped": int(outside.sum()),
        "escaped_frac": float(outside.mean()),
        "g_median": float(np.median(g[outside])) if outside.any() else float("nan"),
        "g_mean": float(np.mean(g[outside])) if outside.any() else float("nan"),
        "g_std": float(np.std(g[outside])) if outside.any() else float("nan"),
        "g_p10": float(np.quantile(g[outside], 0.10)) if outside.any() else float("nan"),
        "g_p90": float(np.quantile(g[outside], 0.90)) if outside.any() else float("nan"),
    }
    print(f"{label}escaped: {out['escaped']}/{out['count']}  ({out['escaped_frac']*100:.2f}%)")
    if outside.any():
        print(f"{label}g median={out['g_median']:.6g}  mean={out['g_mean']:.6g}  std={out['g_std']:.6g}")
        print(f"{label}g p10/p90={out['g_p10']:.6g}/{out['g_p90']:.6g}")
    return out

def write_csv(path: str, rows: list[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"[csv] wrote {path}")

# ============================================================
# 4) KDE + reference law comparisons
# ============================================================

def kde_or_smooth_hist(samples: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if len(samples) < 5:
        return np.zeros_like(grid)
    if HAVE_SCIPY:
        return gaussian_kde(samples)(grid)
    # fallback: smoothed histogram
    hist, edges = np.histogram(samples, bins=min(HIST_BINS, max(10, len(samples)//50)), density=True)
    centers = 0.5*(edges[:-1] + edges[1:])
    dens = np.interp(grid, centers, hist, left=0.0, right=0.0)
    w = max(3, len(grid)//200)
    return np.convolve(dens, np.ones(w)/w, mode="same")

def compare_reference_laws(g_out: np.ndarray, out_prefix: str):
    g = np.asarray(g_out, dtype=float)
    g = g[np.isfinite(g)]
    g = g[g > 0]
    if len(g) < 30:
        print("[compare] not enough outside points.")
        return

    gmin, gmax = float(np.min(g)), float(np.max(g))
    mean = float(np.mean(g))
    rate = 1.0 / max(mean, 1e-15)

    grid = np.linspace(0.0, gmax, KDE_GRID_N)
    g_sorted = np.sort(g)
    ecdf = np.searchsorted(g_sorted, grid, side="right") / len(g_sorted)

    # CDFs
    cdf_unif_0 = np.clip(grid / (gmax + 1e-15), 0.0, 1.0)  # Uniform on [0,gmax]
    cdf_exp = 1.0 - np.exp(-rate * np.maximum(grid, 0.0))
    cdf_unif_gmin = np.clip((grid - gmin) / ((gmax - gmin) + 1e-15), 0.0, 1.0)  # Uniform on [gmin,gmax]
    # log-uniform in |Phi| on [Rmin,Rmax] is exactly uniform in g on [gmin,gmax]

    ks_unif_0 = float(np.max(np.abs(ecdf - cdf_unif_0)))
    ks_exp = float(np.max(np.abs(ecdf - cdf_exp)))
    ks_logunif = float(np.max(np.abs(ecdf - cdf_unif_gmin)))

    ll_unif_0 = float(len(g) * (-math.log(gmax + 1e-15)))
    ll_exp = float(len(g) * math.log(rate + 1e-15) - rate * np.sum(g))
    ll_logunif = float(len(g) * (-math.log((gmax - gmin) + 1e-15)))

    print("[compare] Reference laws (lower KS better; higher LL better):")
    print(f"  uniform g on [0,gmax]:              KS={ks_unif_0:.4g}  LL={ll_unif_0:.4g}")
    print(f"  exponential g (rate=1/mean):        KS={ks_exp:.4g}  LL={ll_exp:.4g}")
    print(f"  log-uniform |Phi| (uniform g[gmin,gmax]): KS={ks_logunif:.4g}  LL={ll_logunif:.4g}")

    os.makedirs(OUT_DIR, exist_ok=True)

    # Plot in g-space: hist + KDE + model PDFs
    plt.figure()
    plt.hist(g, bins=HIST_BINS, density=True, alpha=0.6, label="empirical hist (outside)")
    kde = kde_or_smooth_hist(g, grid)
    plt.plot(grid, kde, linewidth=2.0, label="KDE")

    pdf_unif_0 = np.where((grid >= 0) & (grid <= gmax), 1.0/(gmax + 1e-15), 0.0)
    pdf_exp = rate * np.exp(-rate * np.maximum(grid, 0.0))
    pdf_logunif = np.where((grid >= gmin) & (grid <= gmax), 1.0/((gmax - gmin) + 1e-15), 0.0)

    plt.plot(grid, pdf_unif_0, linewidth=1.5, label="uniform g on [0,gmax]")
    plt.plot(grid, pdf_exp, linewidth=1.5, label="exponential g")
    plt.plot(grid, pdf_logunif, linewidth=1.5, label="log-uniform |Phi| (uniform g on [gmin,gmax])")

    plt.xlabel("g_M(c)")
    plt.ylabel("density")
    plt.title("Empirical density of g_M(c) (outside) + reference laws")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{out_prefix}_g_density_compare.png"), dpi=200)
    plt.close()

    # Plot in |Phi|-space: hist + KDE + log-uniform pdf
    R = np.exp(g)
    Rmin, Rmax = float(np.min(R)), float(np.max(R))
    Rgrid = np.linspace(1.0, Rmax, KDE_GRID_N)

    plt.figure()
    plt.hist(R, bins=HIST_BINS, density=True, alpha=0.6, label="empirical hist of |Phi|")
    kdeR = kde_or_smooth_hist(R, Rgrid)
    plt.plot(Rgrid, kdeR, linewidth=2.0, label="KDE(|Phi|)")
    norm = math.log((Rmax + 1e-15) / (Rmin + 1e-15))
    pdf_logunif_R = np.where((Rgrid >= Rmin) & (Rgrid <= Rmax), 1.0/(Rgrid * (norm + 1e-15)), 0.0)
    plt.plot(Rgrid, pdf_logunif_R, linewidth=1.5, label="log-uniform |Phi| model")
    plt.xlabel("|Phi(c)|")
    plt.ylabel("density")
    plt.title("Empirical density of |Phi(c)| (outside)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{out_prefix}_Phi_density_logunif.png"), dpi=200)
    plt.close()

# ============================================================
# 5) Per-n and cumulative stats (for Spyder)
# ============================================================

def per_n_stats(n_min: int, n_max: int, family: str | None = None) -> list[dict]:
    rows = []
    for n in range(n_min, n_max + 1):
        if family is None:
            eigs = np.linalg.eigvals(generate_lucas_companion(n))
        else:
            top = family_toprow(family, n)
            eigs = np.linalg.eigvals(generate_companion_from_toprow(n, top))

        nonzero = np.abs(eigs) > EIG_TOL
        inv = (1.0 / eigs[nonzero]).astype(np.complex128)
        g, it, phi = batch_potential(inv, max_iter=MAX_ITER, escape_radius=ESCAPE_RADIUS)
        s = summarize_g(g, label=f"[per-n n={n}] ")
        rows.append({"n": n, **s})
    return rows

def cumulative_stats(n_min: int, n_max: int, family: str | None = None) -> list[dict]:
    rows = []
    acc = []
    for N in range(n_min, n_max + 1):
        if family is None:
            eigs = np.linalg.eigvals(generate_lucas_companion(N))
        else:
            top = family_toprow(family, N)
            eigs = np.linalg.eigvals(generate_companion_from_toprow(N, top))

        nonzero = np.abs(eigs) > EIG_TOL
        acc.append((1.0 / eigs[nonzero]).astype(np.complex128))

        C = np.concatenate(acc)
        g, it, phi = batch_potential(C, max_iter=MAX_ITER, escape_radius=ESCAPE_RADIUS)
        s = summarize_g(g, label=f"[cum N={N}] ")
        rows.append({"N": N, **s})
    return rows

def plot_convergence(rows: list[dict], xkey: str, out_prefix: str, title_prefix: str):
    x = np.array([r[xkey] for r in rows], dtype=float)
    escaped = np.array([r["escaped_frac"] for r in rows], dtype=float)
    gmed = np.array([r["g_median"] for r in rows], dtype=float)
    gstd = np.array([r["g_std"] for r in rows], dtype=float)

    plt.figure()
    plt.plot(x, escaped, marker="o")
    plt.xlabel(xkey); plt.ylabel("escaped fraction")
    plt.title(f"{title_prefix}: escaped fraction")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{out_prefix}_escaped_frac.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.plot(x, gmed, marker="o")
    plt.xlabel(xkey); plt.ylabel("median g_M (outside)")
    plt.title(f"{title_prefix}: median g_M (outside)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{out_prefix}_g_median.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.plot(x, gstd, marker="o")
    plt.xlabel(xkey); plt.ylabel("std(g_M) (outside)")
    plt.title(f"{title_prefix}: std of g_M (outside)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{out_prefix}_g_std.png"), dpi=200)
    plt.close()

# ============================================================
# 6) Main
# ============================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # A) Aggregate Lucas cloud
    C_inv = compute_inverse_eigenvalues(N_MIN, N_MAX, tol=EIG_TOL)
    g, it, phi = batch_potential(C_inv, max_iter=MAX_ITER, escape_radius=ESCAPE_RADIUS)
    print("\n=== Aggregate Lucas cloud ===")
    summarize_g(g, label="  ")

    np.save(os.path.join(OUT_DIR, "C_lucas.npy"), C_inv)
    np.save(os.path.join(OUT_DIR, "g_lucas.npy"), g)
    np.save(os.path.join(OUT_DIR, "it_lucas.npy"), it)
    np.save(os.path.join(OUT_DIR, "phi_lucas.npy"), phi)

    compare_reference_laws(g[g > 0], out_prefix="lucas")

    # B) Per-n and cumulative (Lucas)
    print("\n=== Per-n stats (Lucas) ===")
    per_rows = per_n_stats(N_MIN, N_MAX, family=None)
    write_csv(os.path.join(OUT_DIR, "per_n_stats.csv"), per_rows)
    plot_convergence(per_rows, xkey="n", out_prefix="per_n", title_prefix="Per-n convergence")

    print("\n=== Cumulative stats (Lucas) ===")
    cum_rows = cumulative_stats(N_MIN, N_MAX, family=None)
    write_csv(os.path.join(OUT_DIR, "cumulative_stats.csv"), cum_rows)
    plot_convergence(cum_rows, xkey="N", out_prefix="cumulative", title_prefix="Cumulative convergence")

    # C) Optional: your stored curve
    if LUCAS_POINTS_NPY:
        try:
            P = np.load(LUCAS_POINTS_NPY)
            if P.ndim == 2 and P.shape[1] == 2:
                C_curve = (P[:, 0] + 1j * P[:, 1]).astype(np.complex128)
            else:
                C_curve = np.asarray(P).astype(np.complex128).ravel()

            g_curve, it_curve, phi_curve = batch_potential(C_curve, max_iter=MAX_ITER, escape_radius=ESCAPE_RADIUS)
            print("\n=== lucas_points.npy curve ===")
            summarize_g(g_curve, label="  ")
            np.save(os.path.join(OUT_DIR, "g_curve.npy"), g_curve)
            compare_reference_laws(g_curve[g_curve > 0], out_prefix="lucas_curve")
        except FileNotFoundError:
            print(f"[warn] can't find {LUCAS_POINTS_NPY}; skipped.")

    # D) Family comparison
    if RUN_FAMILY_COMPARISON:
        families = [
            "lucas_all_ones",
            "pell_like_all_twos",
            "sparser_gap_1_0_1_then_ones",
            "padovan_like_0_1_then_ones",
        ]
        summary_rows = []
        kde_grid = None
        curves = []

        for fam in families:
            print(f"\n=== Family: {fam} ===")
            C_fam = compute_inverse_eigenvalues_family(fam, N_MIN, N_MAX, tol=EIG_TOL)
            g_fam, it_fam, phi_fam = batch_potential(C_fam, max_iter=MAX_ITER, escape_radius=ESCAPE_RADIUS)
            s = summarize_g(g_fam, label="  ")
            s["family"] = fam
            summary_rows.append(s)

            compare_reference_laws(g_fam[g_fam > 0], out_prefix=f"family_{fam}")

            g_out = g_fam[g_fam > 0]
            if len(g_out) > 50:
                gmax = float(np.max(g_out))
                if kde_grid is None or gmax > kde_grid[-1]:
                    kde_grid = np.linspace(0.0, gmax, KDE_GRID_N)
                curves.append((fam, kde_or_smooth_hist(g_out, kde_grid)))

        write_csv(os.path.join(OUT_DIR, "family_summary.csv"), summary_rows)

        if kde_grid is not None and curves:
            plt.figure()
            for fam, dens in curves:
                plt.plot(kde_grid, dens, label=fam)
            plt.xlabel("g_M(c)"); plt.ylabel("density (KDE)")
            plt.title("KDE overlays of g_M(c) for different families (outside)")
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, "family_kde_overlay.png"), dpi=200)
            plt.close()

    print("\nDone. Outputs in:", OUT_DIR)

if __name__ == "__main__":
    main()
