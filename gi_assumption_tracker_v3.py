#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gi_assumption_tracker_v3.py

Cleaned and robust "assumption tracker" for Appendix A.

Features:
- Rebuild clouds at each resolution (densification with bins).
- OT matching + Procrustes alignment.
- Compare matched target cloud (Mmatch) against aligned construct (Caligned).
- Mollified 2D histograms (optional Gaussian blur in bin units).
- GI-flow modes:
    * adaptive: run until KL(P_M||X_t) <= kl_threshold (or max_steps)
    * fixed-T: run exactly T steps (--T-fixed > 0)
- Outputs CSV + JSON with diagnostics:
    delta_n = KL(P_M||X_T), kl_PM_PC = KL(P_M||P_C), TV, overlap, outside mass,
    Pinsker bounds and compound=(1-alpha)^(-T)*sqrt(delta_n).

Example:
  python3 gi_assumption_tracker_v3.py \
    --module ./tci_construct_mandelbrot_v002_fixed.py \
    --sigma-bins 2.0 \
    --T-fixed 25 \
    --bins-start 64 --bins-max 512 \
    --out-prefix v3_T25_sigma2
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import time
from dataclasses import dataclass, asdict
from typing import List, Tuple

import numpy as np

try:
    from scipy.ndimage import gaussian_filter
except Exception:
    gaussian_filter = None


@dataclass
class Row:
    bins: int
    mesh_proxy: float

    construct_max_n: int
    construct_step: int
    n_construct_pts: int
    mandelbrot_grid: int
    mandelbrot_samples: int
    n_mandel_pts: int

    alpha: float
    sigma_bins: float
    mode: str
    T_n: int

    kl_initial: float
    delta_n: float
    kl_PM_PC: float
    pinsker_tv_bound_XT_PM: float
    tv_XT_PM: float
    tv_PC_PM: float
    overlap_mass_PC_PM: float

    mass_outside_domain_C: float
    mass_outside_domain_M: float

    tv_bound_PC_PM: float
    compound: float
    compound_with_pinsker: float

    stop_reason: str
    runtime_sec: float


def load_module(module_path: str, module_name: str = "tci_fixed_import"):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module at {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tv_distance(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.sum(np.abs(p - q)))


def overlap_mass(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(np.minimum(p, q)))


def fraction_outside_domain(cloud: np.ndarray, domain: Tuple[float, float, float, float]) -> float:
    xmin, xmax, ymin, ymax = domain
    x = cloud.real
    y = cloud.imag
    inside = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
    return float(1.0 - np.mean(inside))


def mollified_histogram(mod, cloud: np.ndarray, bins: int, sigma_bins: float) -> np.ndarray:
    H, _, _ = np.histogram2d(
        cloud.real, cloud.imag,
        bins=(bins, bins),
        range=[[mod.domain[0], mod.domain[1]], [mod.domain[2], mod.domain[3]]],
    )
    eps = float(getattr(mod, "eps", 1e-12))
    H = np.maximum(H, eps)

    if sigma_bins and sigma_bins > 0:
        if gaussian_filter is None:
            raise RuntimeError("scipy.ndimage.gaussian_filter not available; set --sigma-bins 0")
        H = gaussian_filter(H, sigma=float(sigma_bins), mode="nearest")
        H = np.maximum(H, eps)

    H = H / H.sum()
    return H


def gi_flow_fixed_T(KL_fn, P_target, X0, alpha: float, T: int):
    X = X0.copy()
    kl0 = float(KL_fn(P_target, X))
    for _ in range(int(T)):
        X = (1.0 - alpha) * X + alpha * P_target
    klT = float(KL_fn(P_target, X))
    return X, int(T), kl0, klT


def gi_flow_to_threshold(KL_fn, P_target, X0, alpha: float, kl_threshold: float, max_steps: int, min_steps: int = 1):
    X = X0.copy()
    kl0 = float(KL_fn(P_target, X))
    kl = kl0
    T = 0
    for t in range(1, int(max_steps) + 1):
        X = (1.0 - alpha) * X + alpha * P_target
        kl = float(KL_fn(P_target, X))
        T = t
        if t >= int(min_steps) and kl <= float(kl_threshold):
            break
    return X, int(T), kl0, kl


def parse_construct_ns(construct_max_n: int, step: int) -> List[int]:
    return list(range(int(step), int(construct_max_n) + 1, int(step)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True, help="Path to your tci_construct_mandelbrot_v002_fixed.py file.")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--domain", type=str, default="-2.2:1.2:-1.6:1.6", help="xmin:xmax:ymin:ymax")
    ap.add_argument("--alpha", type=float, default=0.1)

    ap.add_argument("--bins-start", type=int, default=64)
    ap.add_argument("--bins-max", type=int, default=1024)

    ap.add_argument("--construct-step", type=int, default=20)
    ap.add_argument("--construct-max-start", type=int, default=300)
    ap.add_argument("--construct-max-growth", type=float, default=1.35)

    ap.add_argument("--mandelbrot-grid-start", type=int, default=600)
    ap.add_argument("--mandelbrot-grid-growth", type=float, default=1.15)
    ap.add_argument("--mandelbrot-samples-start", type=int, default=25000)
    ap.add_argument("--mandelbrot-samples-growth", type=float, default=1.35)
    ap.add_argument("--mandelbrot-samples-max", type=int, default=150000)

    ap.add_argument("--sigma-bins", type=float, default=1.0,
                    help="Gaussian blur sigma measured in *bins*. Use 0 for raw histogram.")

    ap.add_argument("--T-fixed", type=int, default=-1,
                    help="If >0, run exactly T GI steps (no adaptive stopping).")
    ap.add_argument("--kl-threshold", type=float, default=1e-6)
    ap.add_argument("--max-steps", type=int, default=800)
    ap.add_argument("--min-steps", type=int, default=5)

    ap.add_argument("--compound-threshold", type=float, default=1e-3)
    ap.add_argument("--tv-threshold", type=float, default=0.05)

    ap.add_argument("--out-prefix", type=str, default="gi_assumptions_v3")
    args = ap.parse_args()

    np.random.seed(int(args.seed))

    domain = tuple(float(x) for x in args.domain.split(":"))
    mod = load_module(args.module)
    mod.domain = domain

    rows: List[Row] = []
    bins = int(args.bins_start)

    construct_max_n = int(args.construct_max_start)
    mandel_grid = int(args.mandelbrot_grid_start)
    mandel_samples = int(args.mandelbrot_samples_start)

    global_stop_reason = ""

    while bins <= int(args.bins_max):
        t_bin = time.time()

        mod.mandelbrot_grid = int(mandel_grid)
        mod.mandelbrot_samples = int(mandel_samples)
        ns = parse_construct_ns(construct_max_n, int(args.construct_step))

        C = mod.construct_points(ns)
        M = mod.sample_mandelbrot_boundary()

        Mmatch, Csub = mod.entropic_ot_alignment(C, M)
        Caligned = mod.procrustes_align_no_scale(Csub, Mmatch)
        M_aligned = Mmatch

        outside_C = fraction_outside_domain(Caligned, domain)
        outside_M = fraction_outside_domain(M_aligned, domain)

        P_M = mollified_histogram(mod, M_aligned, bins=bins, sigma_bins=float(args.sigma_bins))
        P_C = mollified_histogram(mod, Caligned, bins=bins, sigma_bins=float(args.sigma_bins))
        kl_PM_PC = float(mod.KL(P_M, P_C))

        if int(args.T_fixed) > 0:
            mode = f"fixedT={int(args.T_fixed)}"
            X_T, Tn, kl0, delta = gi_flow_fixed_T(mod.KL, P_M, P_C, float(args.alpha), int(args.T_fixed))
            stop_reason = "fixed_T"
            assert Tn == int(args.T_fixed), (args.T_fixed, Tn)
        else:
            mode = "adaptive"
            X_T, Tn, kl0, delta = gi_flow_to_threshold(
                mod.KL, P_M, P_C, float(args.alpha), float(args.kl_threshold),
                int(args.max_steps), int(args.min_steps)
            )
            stop_reason = "kl_threshold_met" if delta <= float(args.kl_threshold) else "max_steps_reached"

        tv_XT_PM = tv_distance(X_T, P_M)
        tv_PC_PM = tv_distance(P_C, P_M)
        ov = overlap_mass(P_C, P_M)

        pinsker = math.sqrt(0.5 * float(delta))
        factor = (1.0 - float(args.alpha)) ** (-int(Tn)) if int(Tn) > 0 else float("inf")
        tv_bound_PC_PM = factor * pinsker
        compound = factor * math.sqrt(float(delta))
        compound_p = factor * pinsker

        rows.append(Row(
            bins=bins,
            mesh_proxy=1.0 / float(bins),

            construct_max_n=int(construct_max_n),
            construct_step=int(args.construct_step),
            n_construct_pts=int(Caligned.size),

            mandelbrot_grid=int(mandel_grid),
            mandelbrot_samples=int(mandel_samples),
            n_mandel_pts=int(M_aligned.size),

            alpha=float(args.alpha),
            sigma_bins=float(args.sigma_bins),
            mode=mode,
            T_n=int(Tn),

            kl_initial=float(kl0),
            delta_n=float(delta),
            kl_PM_PC=float(kl_PM_PC),
            pinsker_tv_bound_XT_PM=float(pinsker),
            tv_XT_PM=float(tv_XT_PM),
            tv_PC_PM=float(tv_PC_PM),
            overlap_mass_PC_PM=float(ov),

            mass_outside_domain_C=float(outside_C),
            mass_outside_domain_M=float(outside_M),

            tv_bound_PC_PM=float(tv_bound_PC_PM),
            compound=float(compound),
            compound_with_pinsker=float(compound_p),

            stop_reason=stop_reason,
            runtime_sec=float(time.time() - t_bin),
        ))

        print(f"[{mode} bins={bins}] "
              f"δ_n={delta:.3e}  Tn={Tn}  "
              f"TV(PC,PM)={tv_PC_PM:.3e}  overlap={ov:.3e}  "
              f"KL(PM||PC)={kl_PM_PC:.3e}  "
              f"outside(C)={outside_C:.3e} outside(M)={outside_M:.3e}  "
              f"compound={compound:.3e}")

        if (float(delta) <= float(args.kl_threshold)) and (float(compound) <= float(args.compound_threshold)) and (float(tv_PC_PM) <= float(args.tv_threshold)):
            global_stop_reason = "global_stop: kl<=threshold AND compound<=threshold AND TV(P_C,P_M)<=tv_threshold"
            break

        bins *= 2
        construct_max_n = int(round((construct_max_n * float(args.construct_max_growth)) / int(args.construct_step))) * int(args.construct_step)
        mandel_grid = int(round(mandel_grid * float(args.mandelbrot_grid_growth)))
        mandel_samples = min(int(args.mandelbrot_samples_max), int(round(mandel_samples * float(args.mandelbrot_samples_growth))))

    csv_path = f"{args.out_prefix}.csv"
    json_path = f"{args.out_prefix}.json"

    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
            w.writeheader()
            for r in rows:
                w.writerow(asdict(r))
    else:
        open(csv_path, "w", encoding="utf-8").close()

    meta = {
        "module": args.module,
        "seed": int(args.seed),
        "domain": domain,
        "alpha": float(args.alpha),
        "sigma_bins": float(args.sigma_bins),
        "bins_start": int(args.bins_start),
        "bins_max": int(args.bins_max),
        "T_fixed": int(args.T_fixed),
        "kl_threshold": float(args.kl_threshold),
        "max_steps": int(args.max_steps),
        "min_steps": int(args.min_steps),
        "compound_threshold": float(args.compound_threshold),
        "tv_threshold": float(args.tv_threshold),
        "construct_step": int(args.construct_step),
        "construct_max_start": int(args.construct_max_start),
        "construct_max_growth": float(args.construct_max_growth),
        "mandelbrot_grid_start": int(args.mandelbrot_grid_start),
        "mandelbrot_grid_growth": float(args.mandelbrot_grid_growth),
        "mandelbrot_samples_start": int(args.mandelbrot_samples_start),
        "mandelbrot_samples_growth": float(args.mandelbrot_samples_growth),
        "mandelbrot_samples_max": int(args.mandelbrot_samples_max),
        "global_stop_reason": global_stop_reason,
        "rows": [asdict(r) for r in rows],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nWrote:\n  {csv_path}\n  {json_path}")
    if global_stop_reason:
        print(f"Stopped early: {global_stop_reason}")


if __name__ == "__main__":
    main()
