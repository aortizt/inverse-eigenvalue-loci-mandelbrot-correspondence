#!/usr/bin/env python3
"""
mandelbrot_boundary_sample.py

Generate a sampled boundary polyline for the Mandelbrot set by:
1) computing dwell (escape-time) values on a grid;
2) extracting an isocontour near the escape threshold using Matplotlib's contouring;
3) returning the longest contour as an ordered boundary sample.

Outputs:
- <prefix>_boundary.csv           (columns: x,y)
- <prefix>_boundary.png           (overlay of boundary sample)
- <prefix>_meta.txt               (parameters used)

Example:
python mandelbrot_boundary_sample.py \
  --xlim -2.1 0.9 --ylim -1.5 1.5 --res 2000 --max_iter 500 \
  --level 0.95 --output_prefix outputs/mandel
"""
import argparse, os, numpy as np, matplotlib.pyplot as plt

def mandelbrot_dwell(x, y, max_iter=300):
    """Escape-time dwell count for point c = x + i y."""
    c = x + 1j*y
    z = 0+0j
    for n in range(max_iter):
        z = z*z + c
        if (z.real*z.real + z.imag*z.imag) > 4.0:
            return n
    return max_iter

def compute_grid(xlim, ylim, res, max_iter):
    xs = np.linspace(xlim[0], xlim[1], res)
    ys = np.linspace(ylim[0], ylim[1], res)
    Z = np.zeros((res, res), dtype=float)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            Z[j, i] = mandelbrot_dwell(x, y, max_iter=max_iter)
    return xs, ys, Z

def extract_contour(xs, ys, Z, max_iter, level_frac=0.96):
    """
    Extract the isocontour for dwell ≈ level_frac * max_iter.
    Returns the longest contour path (as Nx2 array).
    """
    target = level_frac * max_iter
    cs = plt.contour(xs, ys, Z, levels=[target])
    paths = cs.collections[0].get_paths()
    plt.clf()
    if not paths:
        return None
    # choose the longest path
    best = max(paths, key=lambda p: p.vertices.shape[0])
    return best.vertices  # Nx2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlim", nargs=2, type=float, default=[-2.1, 0.9])
    ap.add_argument("--ylim", nargs=2, type=float, default=[-1.5, 1.5])
    ap.add_argument("--res", type=int, default=1500)
    ap.add_argument("--max_iter", type=int, default=400)
    ap.add_argument("--level", type=float, default=0.96, help="Fraction of max_iter for isocontour")
    ap.add_argument("--output_prefix", required=True)
    args = ap.parse_args()

    xs, ys, Z = compute_grid(args.xlim, args.ylim, args.res, args.max_iter)
    contour = extract_contour(xs, ys, Z, args.max_iter, level_frac=args.level)
    if contour is None or contour.shape[0] < 50:
        raise SystemExit("Failed to extract a usable contour. Try different --level or higher --res.")

    # Save CSV
    out_csv = f"{args.output_prefix}_boundary.csv"
    os.makedirs(os.path.dirname(args.output_prefix), exist_ok=True)
    np.savetxt(out_csv, contour, delimiter=",", header="x,y", comments="")
    # Plot overlay
    plt.figure(figsize=(6,6))
    plt.scatter(contour[:,0], contour[:,1], s=1)
    plt.axis('equal'); plt.axis('off')
    plt.tight_layout()
    out_png = f"{args.output_prefix}_boundary.png"
    plt.savefig(out_png, dpi=220)
    plt.close()
    # Meta
    out_meta = f"{args.output_prefix}_meta.txt"
    with open(out_meta, "w") as f:
        f.write(f"xlim={args.xlim}\nylim={args.ylim}\nres={args.res}\nmax_iter={args.max_iter}\nlevel={args.level}\n")
    print("Wrote:")
    print(" ", out_csv)
    print(" ", out_png)
    print(" ", out_meta)

if __name__ == "__main__":
    main()
