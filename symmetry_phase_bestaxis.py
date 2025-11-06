#!/usr/bin/env python3
# symmetry_phase_bestaxis.py
# Symmetry analysis for Construct and Mandelbrot samples.
# - tests specific symmetries (x-reflection, y-reflection, rotation by pi)
# - searches for best-fit reflection axis (through centroid) by scanning angles
# - quantifies preservation fraction under each symmetry (within tolerance)
# - cross-checks whether matched correspondences are consistent with symmetries
# - outputs CSV, figures, and a readable summary
#
# Usage: edit BASE if your files are elsewhere (default assumes /home/Merlin/Desktop/out_clean/).
#
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.optimize import minimize_scalar
import os, csv, math

# ----------------------
# Parameters / paths
# ----------------------
BASE = "/home/Merlin/Desktop/out_clean/"
OUTDIR = os.path.join(BASE, "symmetry_report_bestaxis")
os.makedirs(OUTDIR, exist_ok=True)

PATH_CONSTRUCT = os.path.join(BASE, "construct_points.csv")
PATH_MANDEL = os.path.join(BASE, "mandel_boundary_sample.csv")
PATH_ALIGNED = os.path.join(BASE, "construct_aligned.csv")
PATH_MATCHES = os.path.join(BASE, "matches_indices.csv")

# tolerance for considering two points matched (Euclidean)
TOL = 0.05  # adjust (median matching ~0.2 suggests this is strict)

# ----------------------
# Utilities
# ----------------------
def load_csv(path):
    try:
        return np.loadtxt(path, delimiter=",")
    except Exception:
        return np.loadtxt(path)

def nearest_mapping(A, B):
    """
    For each point in A, find nearest index in B and distance.
    Returns arrays (idx_in_B, dist).
    """
    tree = cKDTree(B)
    d, idx = tree.query(A, k=1)
    return idx, d

def reflect_across_line(points, angle, origin=None):
    """
    Reflect points across a line through 'origin' with direction given by 'angle' (radians).
    Steps: translate origin->0, rotate so line aligns with x-axis, reflect y->-y, rotate back, translate back.
    points: (N,2)
    angle: radians
    origin: (2,) or None (if None, use centroid)
    """
    if origin is None:
        origin = points.mean(axis=0)
    # translate
    P = points - origin
    ca = math.cos(-angle)
    sa = math.sin(-angle)
    # rotation matrix for -angle
    Rn = np.array([[ca, -sa],[sa, ca]])
    P_rot = P.dot(Rn.T)
    # reflect across x-axis: (x,y) -> (x, -y)
    P_ref = P_rot.copy()
    P_ref[:,1] = -P_ref[:,1]
    # rotate back (angle)
    ca2 = math.cos(angle)
    sa2 = math.sin(angle)
    R = np.array([[ca2, -sa2],[sa2, ca2]])
    P_back = P_ref.dot(R.T)
    P_final = P_back + origin
    return P_final

def apply_symmetry_op(points, op, angle=None):
    if op == 'identity':
        return points.copy()
    if op == 'reflect_x':
        P = points.copy(); P[:,1] = -P[:,1]; return P
    if op == 'reflect_y':
        P = points.copy(); P[:,0] = -P[:,0]; return P
    if op == 'rot_pi':
        return -points.copy()
    if op == 'reflect_angle':
        if angle is None:
            raise ValueError("angle must be provided for reflect_angle")
        centroid = points.mean(axis=0)
        return reflect_across_line(points, angle, origin=centroid)
    raise ValueError("Unknown op")

# ----------------------
# Load data
# ----------------------
C = load_csv(PATH_CONSTRUCT)            # original Construct points (2D)
M = load_csv(PATH_MANDEL)               # Mandel sample (2D)
C_aligned = load_csv(PATH_ALIGNED)      # aligned Construct (2D)
matches = None
if os.path.exists(PATH_MATCHES):
    try:
        matches = np.loadtxt(PATH_MATCHES, delimiter=",", dtype=int)
    except Exception:
        matches = np.loadtxt(PATH_MATCHES, dtype=int)

# Trim to compatible lengths
if matches is not None and C_aligned is not None:
    L = min(len(matches), C_aligned.shape[0])
    matches = matches[:L]
    C_aligned = C_aligned[:L]

# ----------------------
# Test specific ops (identity, reflect_x, reflect_y, rot_pi)
# ----------------------
ops = ['identity', 'reflect_x', 'reflect_y', 'rot_pi']
results = []

for op in ops:
    C_op = apply_symmetry_op(C_aligned, op)
    idxC, dC = nearest_mapping(C_op, C_aligned)
    preserved_C = np.sum(dC <= TOL)
    frac_C = preserved_C / len(C_aligned) if len(C_aligned)>0 else 0.0

    M_op = apply_symmetry_op(M, op)
    idxM, dM = nearest_mapping(M_op, M)
    preserved_M = np.sum(dM <= TOL)
    frac_M = preserved_M / len(M) if len(M)>0 else 0.0

    # cross-preserved: compare C_op vs M_op at matched indices
    if matches is not None:
        M_op_for_matches = M_op[matches]
        d_cross = np.linalg.norm(C_op - M_op_for_matches, axis=1)
        cross_pres = np.sum(d_cross <= TOL)
        cross_frac = cross_pres / len(d_cross) if len(d_cross)>0 else 0.0
    else:
        cross_pres = None; cross_frac = None

    results.append({'op': op,
                    'preserved_construct_count': int(preserved_C),
                    'preserved_construct_frac': float(frac_C),
                    'preserved_mandel_count': int(preserved_M),
                    'preserved_mandel_frac': float(frac_M),
                    'cross_preserved_count': int(cross_pres) if cross_pres is not None else None,
                    'cross_preserved_frac': float(cross_frac) if cross_frac is not None else None,
                    'mean_distC': float(np.mean(dC)),
                    'mean_distM': float(np.mean(dM))})

# ----------------------
# Search best-fit reflection axis through centroid
# ----------------------
centroid_C = C_aligned.mean(axis=0)
centroid_M = M.mean(axis=0)

def score_angle(angle):
    # apply reflection to Construct aligned and compute fraction preserved within TOL (self-symmetry)
    C_ref = reflect_across_line(C_aligned, angle, origin=centroid_C)
    _, dC = nearest_mapping(C_ref, C_aligned)
    fracC = np.sum(dC <= TOL) / len(dC)

    # do same for Mandel
    M_ref = reflect_across_line(M, angle, origin=centroid_M)
    _, dM = nearest_mapping(M_ref, M)
    fracM = np.sum(dM <= TOL) / len(dM)

    # combined score (we maximize): give equal weight
    return -(fracC + fracM)  # negative for minimizer

# coarse scan then refine
angles = np.linspace(0, np.pi, 361)  # 0..180 degrees (mirror repeats every pi)
scores = []
for a in angles:
    s = score_angle(a)
    scores.append(s)
scores = np.array(scores)
best_idx = np.argmin(scores)
best_angle_coarse = angles[best_idx]

# refine via scalar minimization near best
res = minimize_scalar(lambda x: score_angle(x), bounds=max(0, best_angle_coarse-math.pi/36), args=(), method='bounded', options={'xatol':1e-4}, bounds=(max(0,best_angle_coarse-math.pi/36), min(math.pi, best_angle_coarse+math.pi/36)))
best_angle = res.x if res.success else best_angle_coarse

# compute final preservation fractions at best angle
C_ref_best = reflect_across_line(C_aligned, best_angle, origin=centroid_C)
_, dC_best = nearest_mapping(C_ref_best, C_aligned)
presC_best = np.sum(dC_best <= TOL) / len(dC_best)

M_ref_best = reflect_across_line(M, best_angle, origin=centroid_M)
_, dM_best = nearest_mapping(M_ref_best, M)
presM_best = np.sum(dM_best <= TOL) / len(dM_best)

# cross-preserved at best angle
if matches is not None:
    M_ref_for_matches = reflect_across_line(M, best_angle, origin=centroid_M)[matches]
    d_cross_best = np.linalg.norm(C_ref_best - M_ref_for_matches, axis=1)
    cross_pres_best = np.sum(d_cross_best <= TOL) / len(d_cross_best)
else:
    cross_pres_best = None

results.append({'op': 'reflect_best_angle',
                'angle_deg': float(np.degrees(best_angle)),
                'preserved_construct_count': int(np.sum(dC_best <= TOL)),
                'preserved_construct_frac': float(presC_best),
                'preserved_mandel_count': int(np.sum(dM_best <= TOL)),
                'preserved_mandel_frac': float(presM_best),
                'cross_preserved_count': None,
                'cross_preserved_frac': float(cross_pres_best) if cross_pres_best is not None else None,
                'mean_distC': float(np.mean(dC_best)),
                'mean_distM': float(np.mean(dM_best))})

# ----------------------
# Save CSV report
# ----------------------
csv_path = os.path.join(OUTDIR, "symmetry_report_bestaxis.csv")
with open(csv_path, 'w', newline='') as cf:
    writer = csv.writer(cf)
    header = ['op', 'angle_deg', 'preserved_construct_count', 'preserved_construct_frac', 'preserved_mandel_count', 'preserved_mandel_frac', 'cross_preserved_count', 'cross_preserved_frac', 'mean_distC', 'mean_distM']
    writer.writerow(header)
    for r in results:
        row = [r.get(h) if h in r else None for h in header]
        writer.writerow(row)

# ----------------------
# Plots and visualizations
# ----------------------
# plot fraction vs angle (coarse scan)
plt.figure(figsize=(8,4))
plt.plot(np.degrees(angles), 1 + (-scores))  # convert negative score back to sum of fracs
plt.xlabel('Angle (degrees)')
plt.ylabel('sum_preserved_fraction (Construct + Mandel)')
plt.title('Coarse scan of reflection preservation vs angle')
plt.axvline(np.degrees(best_angle), color='red', linestyle='--', label=f'best angle {np.degrees(best_angle):.2f} deg')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'preservation_vs_angle.png'), dpi=200)
plt.close()

# plot original and reflected for best angle for Mandel and Construct
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.scatter(M[:,0], M[:,1], s=6, c='red', label='Mandel orig')
M_ref_plot = reflect_across_line(M, best_angle, origin=centroid_M)
plt.scatter(M_ref_plot[:,0], M_ref_plot[:,1], s=6, c='orange', alpha=0.5, label='Mandel reflected')
plt.title('Mandel original vs reflection (best angle)'); plt.axis('equal'); plt.legend()

plt.subplot(1,2,2)
plt.scatter(C_aligned[:,0], C_aligned[:,1], s=6, c='cyan', label='Construct aligned')
C_ref_plot = reflect_across_line(C_aligned, best_angle, origin=centroid_C)
plt.scatter(C_ref_plot[:,0], C_ref_plot[:,1], s=6, c='blue', alpha=0.5, label='Construct reflected')
plt.title('Construct original vs reflection (best angle)'); plt.axis('equal'); plt.legend()

plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, 'orig_vs_reflected_bestangle.png'), dpi=200); plt.close()

# overlay matches colored by whether reflection around best angle preserves correspondence
if matches is not None:
    M_ref_all = reflect_across_line(M, best_angle, origin=centroid_M)
    C_ref_all = reflect_across_line(C_aligned, best_angle, origin=centroid_C)
    d_cross = np.linalg.norm(C_ref_all - M_ref_all[matches], axis=1)
    preserved_mask = d_cross <= TOL

    plt.figure(figsize=(8,6))
    plt.scatter(M[:,0], M[:,1], s=6, c='red', label='Mandel')
    plt.scatter(C_aligned[:,0], C_aligned[:,1], s=6, c='cyan', alpha=0.7, label='Construct aligned')
    for i in range(len(matches)):
        j = matches[i]
        x_vals = [C_aligned[i,0], M[j,0]]
        y_vals = [C_aligned[i,1], M[j,1]]
        if preserved_mask[i]:
            plt.plot(x_vals, y_vals, color='green', linewidth=0.4, alpha=0.7)
        else:
            plt.plot(x_vals, y_vals, color='gray', linewidth=0.2, alpha=0.3)
    plt.title('Matches (green = preserved under best-angle reflection)')
    plt.axis('equal')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'matches_bestangle_preservation.png'), dpi=200)
    plt.close()

# ----------------------
# Save human-readable summary
# ----------------------
txt_path = os.path.join(OUTDIR, 'symmetry_bestaxis_readme.txt')
with open(txt_path, 'w') as tf:
    tf.write('Symmetry analysis (best-axis) report\n')
    tf.write(f'TOL = {TOL}\\n')
    tf.write('\\nOperations tested:\\n')
    for r in results:
        tf.write(str(r) + '\\n')
    tf.write('\\nBest angle (degrees) = {:.4f}\\n'.format(np.degrees(best_angle)))
    tf.write('Preserved fractions at best angle: Construct = {:.4f}, Mandel = {:.4f}\\n'.format(presC_best, presM_best))
    if cross_pres_best is not None:
        tf.write('Cross-preserved fraction (matches consistent) at best angle = {:.4f}\\n'.format(cross_pres_best))
    else:
        tf.write('Cross-preserved fraction: matches file not available.\\n')

print('Symmetry (best-axis) analysis complete. Results saved to', OUTDIR)
