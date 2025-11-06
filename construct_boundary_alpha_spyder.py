#!/usr/bin/env python3
# construct_boundary_alpha_spyder.py
# Spyder-friendly: either set input_csv to a path, or leave as None to pick a file via a dialog.

import os, numpy as np
import matplotlib.pyplot as plt

# Try to use a simple file dialog if input_csv is None
# ==== USER PARAMETERS (set these and press Run) ====
input_csv = "/home/dakini/Downloads/CM-TCI/construct_points.csv"
alpha = 65.0
output_prefix = "outputs/construct"
# ===================================================


def pick_file_dialog():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        path = filedialog.askopenfilename(title="Pick Construct points CSV",
                                          filetypes=[("CSV files","*.csv"), ("All files","*.*")])
        root.update()
        root.destroy()
        return path
    except Exception:
        return None

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

from scipy.spatial import Delaunay

def circumradius(p, q, r):
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
    from collections import defaultdict
    edge_count = defaultdict(int)
    def add_edge(i,j):
        if i<j: edge_count[(i,j)] += 1
        else: edge_count[(j,i)] += 1
    for t in keep:
        add_edge(t[0], t[1]); add_edge(t[1], t[2]); add_edge(t[2], t[0])
    return [e for e,c in edge_count.items() if c==1]

def order_boundary(P, edges):
    from collections import defaultdict
    adj = defaultdict(list)
    for i,j in edges:
        adj[i].append(j); adj[j].append(i)
    start = None
    for k,v in adj.items():
        if len(v)==1:
            start = k; break
    if start is None:
        start = edges[0][0]
    ordered = []
    visited = set()
    curr = start; prev = None
    while True:
        ordered.append(curr); visited.add(curr)
        nbrs = adj[curr]; nxt = None
        for n in nbrs:
            if n != prev:
                nxt = n; break
        if nxt is None: break
        prev, curr = curr, nxt
        if curr==start:
            ordered.append(curr); break
        if len(ordered) > len(P) + 5:
            break
    return ordered

# Run
# Run
import os
P = load_points(input_csv)
edges = alpha_shape_edges(P, alpha=alpha)
if len(edges)==0:
    raise SystemExit("Alpha-shape produced no boundary edges. Try adjusting alpha.")

ordered_idx = order_boundary(P, edges)
B = P[ordered_idx, :]

os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
b_csv = f"{output_prefix}_boundary.csv"
np.savetxt(b_csv, B, delimiter=",", header="x,y", comments="")
e_csv = f"{output_prefix}_edges.csv"
np.savetxt(e_csv, np.asarray(edges, dtype=int), fmt="%d", delimiter=",", header="i,j", comments="")

plt.figure(figsize=(6,6))
plt.scatter(P[:,0], P[:,1], s=2, alpha=0.25)
plt.plot(B[:,0], B[:,1], lw=1.0)
plt.axis('equal'); plt.axis('off'); plt.tight_layout()
png_path = f"{output_prefix}_boundary.png"
plt.savefig(png_path, dpi=220)
plt.show()

with open(f"{output_prefix}_meta.txt", "w") as f:
    f.write(f"alpha={alpha}\nN={len(P)}\nordered_points={len(B)}\n")

print("Wrote:", b_csv, "and", png_path)
