#!/usr/bin/env python3
# construct_boundary_alpha_spyder_v2.py
# Robust boundary extraction: finds all boundary components, selects the longest closed loop;
# if no closed loop exists, falls back to the longest open chain.

import os, numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from scipy.spatial import Delaunay

# ==== USER PARAMETERS (edit & Run) ====
input_csv   = "/home/dakini/Downloads/CM-TCI/construct_points.csv"
alpha       = 65.0         # keep your working value
output_prefix = "outputs/construct"
min_points  = 200          # script will warn if the boundary is shorter
# =====================================

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
    edge_count = {}
    def add_edge(i,j):
        e = (i,j) if i<j else (j,i)
        edge_count[e] = edge_count.get(e, 0) + 1
    for t in keep:
        add_edge(t[0], t[1]); add_edge(t[1], t[2]); add_edge(t[2], t[0])
    boundary_edges = [e for e,c in edge_count.items() if c==1]
    return boundary_edges

def connected_components(edges):
    adj = defaultdict(list)
    nodes = set()
    for i,j in edges:
        adj[i].append(j); adj[j].append(i)
        nodes.add(i); nodes.add(j)
    visited = set()
    comps = []
    for v in nodes:
        if v in visited: continue
        q = deque([v]); visited.add(v)
        comp_nodes = set([v])
        comp_edges = []
        while q:
            u = q.popleft()
            for w in adj[u]:
                e = (u,w) if u<w else (w,u)
                comp_edges.append(e)
                if w not in visited:
                    visited.add(w); q.append(w); comp_nodes.add(w)
        comp_edges = list(set(comp_edges))
        comps.append((comp_nodes, comp_edges))
    return comps, adj

def trace_loop_or_chain(adj, comp_nodes):
    endpoints = [v for v in comp_nodes if len(adj[v]) != 2]
    if len(endpoints) == 0 and len(comp_nodes) > 2:
        start = next(iter(comp_nodes))
        ordered = [start]
        prev = None; curr = start
        for _ in range(len(comp_nodes)+5):
            nbrs = adj[curr]
            nxt = nbrs[0] if nbrs[0] != prev else (nbrs[1] if len(nbrs)>1 else None)
            if nxt is None: break
            ordered.append(nxt)
            prev, curr = curr, nxt
            if curr == start:
                break
        return ordered, True
    else:
        starts = [v for v in endpoints if len(adj[v]) == 1] or endpoints or list(comp_nodes)
        best = []
        for s in starts:
            seen = set([s]); path = [s]
            prev = None; curr = s
            for _ in range(len(comp_nodes)+5):
                nbrs = [x for x in adj[curr] if x != prev]
                if not nbrs: break
                nxt = nbrs[0]
                if nxt in seen: break
                path.append(nxt); seen.add(nxt)
                prev, curr = curr, nxt
            if len(path) > len(best): best = path
        return best, False

# ---------- RUN ----------
os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
P = load_points(input_csv)
edges = alpha_shape_edges(P, alpha=alpha)
if len(edges)==0:
    raise SystemExit("Alpha-shape produced no boundary edges. Try adjusting alpha.")

comps, adj_global = connected_components(edges)

closed_candidates = []
open_candidates = []
for comp_nodes, comp_edges in comps:
    local_adj = defaultdict(list)
    for i,j in comp_edges:
        local_adj[i].append(j); local_adj[j].append(i)
    ordered_idx, is_closed = trace_loop_or_chain(local_adj, comp_nodes)
    if len(ordered_idx) < 5:
        continue
    if is_closed:
        closed_candidates.append(ordered_idx)
    else:
        open_candidates.append(ordered_idx)

if closed_candidates:
    ordered_idx = max(closed_candidates, key=len)
    was_closed = True
elif open_candidates:
    ordered_idx = max(open_candidates, key=len)
    was_closed = False
else:
    raise SystemExit("No usable boundary component found. Adjust alpha.")

B = P[ordered_idx, :]

# --- DENSIFY: resample boundary to a target number of points along arclength ---
target_n = 1500   # 800–3000 is fine; pick what you like

# Drop duplicates to avoid zero-length segments
_, uniq_idx = np.unique(B, axis=0, return_index=True)
B = B[np.sort(uniq_idx)]

# If the loop is closed, ensure last point repeats the first; if not, treat as open
closed_loop = np.allclose(B[0], B[-1])
if not closed_loop:
    # Optional: if you know it *should* be closed, force close
    B = np.vstack([B, B[0]])

# Build arclength
seg = np.linalg.norm(np.diff(B, axis=0), axis=1)
s = np.concatenate([[0.0], np.cumsum(seg)])

# Guard against degenerate tiny boundaries
if s[-1] < 1e-12:
    raise SystemExit("Boundary arclength too small after cleaning; adjust alpha or input.")

# New uniform arclength grid and linear resample (x(s), y(s))
s_new = np.linspace(0.0, s[-1], target_n)
Bx = np.interp(s_new, s, B[:, 0])
By = np.interp(s_new, s, B[:, 1])
B = np.c_[Bx, By]


if B.shape[0] < min_points:
    print(f"WARNING: boundary has only {B.shape[0]} points (< {min_points}). "
          f"Consider increasing point density or adjusting alpha.")

b_csv = f"{output_prefix}_boundary.csv"
np.savetxt(b_csv, B, delimiter=",", header="x,y", comments="")
e_csv = f"{output_prefix}_edges.csv"
ord_edges = np.c_[np.arange(len(ordered_idx)-1), np.arange(1,len(ordered_idx))]
np.savetxt(e_csv, ord_edges.astype(int), fmt="%d", delimiter=",", header="i,j", comments="")

plt.figure(figsize=(6,6))
plt.scatter(P[:,0], P[:,1], s=2, alpha=0.25)
plt.plot(B[:,0], B[:,1], lw=1.0)
plt.title(f"Construct boundary (alpha={alpha}, {'closed' if was_closed else 'open'})")
plt.axis('equal'); plt.axis('off'); plt.tight_layout()
png_path = f"{output_prefix}_boundary.png"
plt.savefig(png_path, dpi=220); plt.show()

with open(f"{output_prefix}_meta.txt", "w") as f:
    f.write(f"alpha={alpha}\nN={len(P)}\nordered_points={len(B)}\nclosed={was_closed}\n")

print("Wrote:", b_csv, "and", png_path, "points:", B.shape[0], "closed:", was_closed)
