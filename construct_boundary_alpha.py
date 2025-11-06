#!/usr/bin/env python3
"""
construct_boundary_alpha.py

Extract a boundary polyline from an unordered 2D point cloud using an alpha-shape.

Algorithm:
- Compute Delaunay triangulation
- Keep triangles whose circumradius R satisfies R < 1/alpha  (alpha > 0)
- Build edge set from kept triangles; boundary edges are those used by exactly one kept triangle
- Trace boundary edges into an ordered polyline

Inputs:
- CSV with x,y columns (header 'x,y' or headerless 2 columns)

Outputs:
- <prefix>_boundary.csv      (ordered boundary points x,y)
- <prefix>_edges.csv         (boundary edges i,j indices in the ordered polyline)
- <prefix>_boundary.png      (overlay)
- <prefix>_meta.txt

Example:
python construct_boundary_alpha.py \
  --input_csv out_clean/construct_points.csv \
  --alpha 30.0 \
  --output_prefix outputs/construct
"""
import argparse, os, numpy as np
from scipy.spatial import Delaunay

def load_points(csv_path):
    try:
        arr = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=float)
        if ('x' in arr.dtype.names) and ('y' in arr.dtype.names):
            return np.c_[arr['x'], arr['y']]
    except Exception:
        pass
    pts = np.genfromtxt(csv_path, delimiter=",", dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(-1,2)
    if pts.shape[1] != 2:
        raise ValueError("Expect 2 columns (x,y)")
    return pts

def circumradius(p, q, r):
    # radius of circumcircle of triangle pqr
    a = np.linalg.norm(q - r)
    b = np.linalg.norm(p - r)
    c = np.linalg.norm(p - q)
    s = (a + b + c)/2.0
    A = max(s*(s-a)*(s-b)*(s-c), 0.0)
    if A == 0.0:
        return np.inf
    area = np.sqrt(A)
    return (a*b*c) / (4.0*area + 1e-16)

def alpha_shape_edges(P, alpha):
    tri = Delaunay(P)
    keep = []
    inv_alpha = 1.0/alpha
    for t in tri.simplices:
        R = circumradius(P[t[0]], P[t[1]], P[t[2]])
        if R < inv_alpha:
            keep.append(t)
    keep = np.asarray(keep, dtype=int)
    if keep.size == 0:
        return []
    # count edges
    from collections import defaultdict
    edge_count = defaultdict(int)
    def add_edge(i,j):
        if i<j:
            edge_count[(i,j)] += 1
        else:
            edge_count[(j,i)] += 1
    for t in keep:
        add_edge(t[0], t[1])
        add_edge(t[1], t[2])
        add_edge(t[2], t[0])
    # boundary edges occur once
    boundary_edges = [e for e,c in edge_count.items() if c==1]
    return boundary_edges

def order_boundary(P, edges):
    # build adjacency
    from collections import defaultdict, deque
    adj = defaultdict(list)
    for i,j in edges:
        adj[i].append(j); adj[j].append(i)
    # find a start (degree==1 preferred), else any
    start = None
    for k,v in adj.items():
        if len(v)==1:
            start = k; break
    if start is None:
        # closed loop; pick an arbitrary start
        start = edges[0][0]
    # trace
    ordered = []
    visited = set()
    curr = start
    prev = None
    while True:
        ordered.append(curr)
        visited.add(curr)
        nbrs = adj[curr]
        nxt = None
        for n in nbrs:
            if n != prev:
                nxt = n
                break
        if nxt is None:
            break
        prev, curr = curr, nxt
        if curr==start:
            ordered.append(curr)
            break
        if len(ordered) > len(P) + 5:
            break
    return ordered

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--alpha", type=float, default=25.0, help="Larger alpha => tighter boundary")
    ap.add_argument("--output_prefix", required=True)
    args = ap.parse_args()

    P = load_points(args.input_csv)
    edges = alpha_shape_edges(P, alpha=args.alpha)
    if len(edges)==0:
        raise SystemExit("Alpha-shape produced no boundary edges. Try smaller alpha (tighter) or larger (looser).")

    ordered_idx = order_boundary(P, edges)
    B = P[ordered_idx, :]

    # save outputs
    os.makedirs(os.path.dirname(args.output_prefix), exist_ok=True)
    b_csv = f"{args.output_prefix}_boundary.csv"
    np.savetxt(b_csv, B, delimiter=",", header="x,y", comments="")
    e_csv = f"{args.output_prefix}_edges.csv"
    np.savetxt(e_csv, np.asarray(edges, dtype=int), fmt="%d", delimiter=",", header="i,j", comments="")

    # plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,6))
    plt.scatter(P[:,0], P[:,1], s=2, alpha=0.25)
    plt.plot(B[:,0], B[:,1], lw=1.0)
    plt.axis('equal'); plt.axis('off')
    plt.tight_layout()
    out_png = f"{args.output_prefix}_boundary.png"
    plt.savefig(out_png, dpi=220); plt.close()

    # meta
    meta = f"{args.output_prefix}_meta.txt"
    with open(meta, "w") as f:
        f.write(f"alpha={args.alpha}\nN={len(P)}\nordered_points={len(B)}\n")

    print("Wrote:")
    print(" ", b_csv)
    print(" ", e_csv)
    print(" ", out_png)
    print(" ", meta)

if __name__ == "__main__":
    main()
