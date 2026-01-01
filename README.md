---

# Lucas Loci – Mandelbrot (CM–TCI) Analysis Codebase  
*(formerly: Construct–Mandelbrot)*

This repository contains the full, reproducible computational pipeline used in the paper  

**“From Homotopies to Statistics: Quantitative Correspondences between  
Inverse Eigenvalue Loci of Generalized Lucas Sequences and the Mandelbrot Set”**,

including boundary generation and stabilization, curvature analysis (via independent estimators),  
variograms and spatial statistics, potential/Laplacian fields, spectral and multifractal analyses,  
dynamical embeddings, symmetry diagnostics, and information-theoretic (TCI / GI-flow) experiments.

> **Terminology note.**  
> Earlier versions of this project referred to the inverse eigenvalue point cloud as the  
> **“Construct”**. Throughout this repository and the associated paper, the preferred term is  
> **Lucas Loci**. The change is terminological and reflects conceptual clarification rather than  
> a change in methodology.

> **TL;DR for Spyder users** — Run the numbered steps in **Quick Start (Spyder)** below.  
> **CLI equivalents** are provided immediately after that.

---
## Repository Layout
```text
CM-TCI/
├─ 0_data/
│  ├─ construct_points.csv
│  └─ ...
├─ outputs/
│  ├─ mandel_boundary.csv / .png / .txt
│  ├─ construct_boundary.csv / .png / .txt
│  ├─ curv_localpoly/
│  │  ├─ mandel_*
│  │  └─ construct_*
│  └─ ...
├─ scripts/
│  ├─ mandelbrot_boundary_sample_spyder.py
│  ├─ construct_boundary_alpha_spyder_v2.py
│  ├─ boundary_curvature_localpoly_spyder.py
│  ├─ mandelbrot_boundary_sample.py
│  ├─ construct_boundary_alpha.py
│  ├─ boundary_curvature_localpoly.py
│  ├─ construct_stage1_clean.py
│  ├─ MandelBoundary.py
│  ├─ mandel2.py
│  ├─ symmetry_phase_bestaxis.py
│  ├─ match_visual_pairs.py
│  ├─ match_analysis_steps1_2.py
│  ├─ spatial_stats_phase2.py
│  ├─ spatial_stats_phase3.py
│  ├─ spatial_stats_phase3b.py
│  ├─ spatial_stats_phase4.py
│  ├─ phase4b_spectral_bootstrap.py
│  ├─ phase5_report.py
│  ├─ spectral_decay_exponent.py
│  ├─ variograms_construct_mandelbrot.py
│  ├─ variograms_construct_mandelbrotv2.py
│  ├─ Iterative_Variogram_Laplacian.py
│  ├─ Variogram-Mandelbrot-Construct.py
│  ├─ Potentials.py
│  ├─ Potentials-C_M.py
│  ├─ Laplacian_C-M.py
│  ├─ multifractal_phase6.py
│  ├─ dynamical_embeddings_phase7.py
│  ├─ tci_construct_mandelbrot.py
│  ├─ tci_construct_mandelbrot-v002.py
│  ├─ tci_construct_mandelbrot_v002_fixed.py
│  ├─ lucas_to_cardioid_v18_periodic_theta_crbins_artifacts.py
│  ├─ lucas_to_cardioid_v40_reference.py
│  └─ lucas_equipotential_test_v3.py
└─ README.md
```
````

**Note on input paths and legacy filenames:**  

Some scripts expect CSV inputs such as `construct_points.csv` and/or
`mandel_boundary_sample.csv` to be visible from the current working directory.

- Files named `construct_points.csv` correspond to the inverse eigenvalue
  point cloud of Lucas-type recurrences (now referred to as **Lucas Loci**).
- The filename is retained for backward compatibility with earlier scripts.

If you run a script from another directory, you may either:

1. Place the required CSV files next to the script, **or**
2. Edit the line `input_csv = ...` inside the script, **or**
3. Use command-line flags such as  
   `--input_csv path/to/data.csv` (where supported) to make paths explicit.
---

## Quick Start (Spyder)

### **Step 1 — Mandelbrot boundary**
`scripts/mandelbrot_boundary_sample_spyder.py`  

Edit parameters at the top of the script if necessary, then **Run**.

**Outputs:**  
- `outputs/mandel_boundary.csv`  
- `outputs/mandel_boundary.png`  
- `outputs/mandel_boundary.txt` (summary statistics)

---

### **Step 2 — Lucas Loci boundary (alpha-shape)**  
*(legacy name: “Construct”)*

Use the robust version:  
`scripts/construct_boundary_alpha_spyder_v2.py`

Set:
```python
input_csv     = "0_data/construct_points.csv"   # inverse eigenvalue cloud (Lucas Loci)
alpha         = 65.0
output_prefix = "outputs/construct"
# v2 extracts the main closed loop and can densify to ~1500 points

Run.

**Outputs:**

* `outputs/construct_boundary.csv` — ordered boundary points (Lucas Loci)
* `outputs/construct_boundary.png` — boundary visualization
* `outputs/construct_boundary.txt` — summary and diagnostics

*Note:*  
This step produces a boundary suitable for curvature, variogram, and spatial
statistics workflows.  
The boundary used in the paper’s reference uniformization is generated separately
by `lucas_to_cardioid_v18_periodic_theta_crbins_artifacts.py`.

### **Step 3 — Curvature (local polynomial)**

`scripts/boundary_curvature_localpoly_spyder.py`

Run **once for each boundary**:

```python
# Mandelbrot
input_csv     = "outputs/mandel_boundary.csv"
output_prefix = "outputs/curv_localpoly/mandel"
neighbors     = 7
closed        = True

# Lucas Loci (legacy name: Construct)
input_csv     = "outputs/construct_boundary.csv"
output_prefix = "outputs/curv_localpoly/construct"
neighbors     = 7
closed        = True

---

## CLI Equivalents

The following commands reproduce the **Quick Start (Spyder)** steps using the
command line interface (CLI).  
Not all scripts in this repository are fully CLI-parameterized; the examples
below correspond to the core boundary-generation steps.

### Mandelbrot boundary (CLI equivalent of Step 1)

```bash
python scripts/mandelbrot_boundary_sample.py \
  --xlim -2.1 0.9 --ylim -1.5 1.5 --res 2000 --max_iter 500 \
  --level 0.96 \
  --output_prefix outputs/mandel

```

```bash
python scripts/construct_boundary_alpha.py \
  --input_csv 0_data/construct_points.csv \
  --alpha 65.0 \
  --output_prefix outputs/construct
```

```bash
python scripts/boundary_curvature_localpoly.py \
  --input_csv outputs/mandel_boundary.csv \
  --output_prefix outputs/curv_localpoly/mandel \
  --neighbors 7 --closed True

python scripts/boundary_curvature_localpoly.py \
  --input_csv outputs/construct_boundary.csv \
  --output_prefix outputs/curv_localpoly/construct \
  --neighbors 7 --closed True
```

---

## Important CSV Path Notes

* Files named `construct_points.csv` correspond to the inverse eigenvalue
  point cloud of Lucas-type recurrences (now referred to as **Lucas Loci**).
  The filename is retained for compatibility with earlier scripts.
* `mandel_boundary_sample.csv` (when used) contains sampled Mandelbrot boundary data.
* If you work inside Spyder, the simplest approach is to keep input CSV files
  inside **0_data/** and adjust, for example:  
  `input_csv = "0_data/construct_points.csv"`.
* Some scripts can be trivially extended with a CLI flag such as
  `--input_csv path/to/data.csv` to make paths explicit.

---

## Script Catalog (What Each One Does)

### Reference pipeline (used in the paper)

These scripts define the stabilized, reproducible computational path used
for the main results reported in the paper.

* `lucas_to_cardioid_v18_periodic_theta_crbins_artifacts.py`  
  Final diagnostic and boundary-extraction pipeline for inverse eigenvalue
  loci of Lucas-type recurrences (Lucas Loci).  
  Generates `lucas_points.npy`.

* `lucas_to_cardioid_v40_reference.py`  
  Reference harmonic / conformal uniformization of the Lucas Loci boundary
  against the Mandelbrot cardioid.  
  Consumes `lucas_points.npy` and computes quasiconformal diagnostics.

* `lucas_equipotential_test_v3.py`  
  Green-function and equipotential statistics comparing inverse Lucas spectra
  with Mandelbrot equipotentials.

---

### Boundary generation & preprocessing (auxiliary)

* `mandelbrot_boundary_sample_spyder.py`, `mandelbrot_boundary_sample.py` —  
  Mandelbrot boundary via escape-time grid and isocontours.

* `construct_boundary_alpha_spyder_v2.py`,  
  `construct_boundary_alpha_spyder.py`, `construct_boundary_alpha.py` —  
  Alpha-shape boundary extraction from point clouds  
  (legacy name: *Construct*; now Lucas Loci).

* `construct_stage1_clean.py` —  
  Optional cleaning, filtering, and ordering utilities for the Lucas Loci
  point cloud.

* `MandelBoundary.py`, `mandel2.py` —  
  Supplementary Mandelbrot boundary generators.

---

### Curvature

* `boundary_curvature_localpoly_spyder.py`, `boundary_curvature_localpoly.py` —  
  Local-polynomial curvature estimation (quadratic least squares in a sliding
  window).

  *Produces curvature CSV files, histograms, boundary overlays, and summaries.*

---

### Variograms / Spatial statistics

* `variograms_construct_mandelbrot.py`, `variograms_construct_mandelbrotv2.py`
* `Iterative_Variogram_Laplacian.py`
* `Variogram-Mandelbrot-Construct.py`
* `spatial_stats_phase2.py`, `spatial_stats_phase3.py`,
  `spatial_stats_phase3b.py`, `spatial_stats_phase4.py`
* `phase4b_spectral_bootstrap.py`
* `phase5_report.py`

---

### Potentials / Laplacians

* `Potentials.py`, `Potentials-C_M.py`, `Laplacian_C-M.py`

---

### Spectral / Multifractal

* `spectral_decay_exponent.py`
* `multifractal_phase6.py`

---

### Embeddings / Symmetry / Matching

* `dynamical_embeddings_phase7.py`
* `symmetry_phase_bestaxis.py`
* `match_visual_pairs.py`, `match_analysis_steps1_2.py`

---

### Information-theoretic convergence (TCI)

* `tci_construct_mandelbrot.py`
* `tci_construct_mandelbrot-v002.py`
* `tci_construct_mandelbrot_v002_fixed.py`

---

## Environment

The codebase is pure Python and does not rely on conda-specific features.
It has been run successfully in multiple setups, including system-wide
Python installations and Spyder installed outside conda.

One convenient option is to use a conda environment, for example:

```bash
conda create -n cm-tci python=3.11 numpy scipy matplotlib scikit-learn
conda activate cm-tci

Alternatively, the required packages can be installed using pip
in an existing Python environment:

pip install numpy scipy matplotlib scikit-learn shapely alphashape

Spyder may be installed either inside or outside the environment used
to run the scripts, as long as it points to a Python interpreter with
the required packages available.

---

## Notes

* All scripts are pure Python; no external binaries are required.
* The code has been run successfully using standard Python virtual
  environments (`venv`) as well as other Python setups.
* For local-polynomial curvature estimation, a neighborhood size
  `k = 7`–`11` is typically safe for boundaries with approximately
  1 000–20 000 points.
* The curvature analysis reported in the final paper uses `k = 7`.
* The v2 alpha-boundary script is recommended, as it robustly extracts
  the main closed component and produces evenly densified boundary points.

---

If you use this code in academic work, please cite the associated paper.

For reproducibility questions or issues, please open a GitHub issue and include:
the script name, parameter values, and the output directory or prefix used.

```

---

```



---

## Numerical Assumption Verification (Appendix A Support)

The current version of the repository includes an explicit **assumption–tracking
pipeline** used to verify the numerical regime underlying Appendix~A of the paper.
These files do **not** introduce new results; they document, in a reproducible way,
that the hypotheses required by the geometric–information correspondence are
satisfied by the computational experiments.

### Assumption tracker script

* **`gi_assumption_tracker_v3.py`**

  A cleaned and robust numerical driver that:
  - rebuilds geometric point clouds at increasing resolutions,
  - performs entropic OT matching and Procrustes alignment,
  - constructs (optionally mollified) 2D histograms,
  - runs the GI–flow in either fixed-$T$ or adaptive stopping mode,
  - records KL contraction, Pinsker bounds, total variation distance,
    histogram overlap, and window leakage.

  This script is the one referenced in **Appendix A** of the paper and
  supersedes earlier experimental variants.

### Representative numerical outputs

The following CSV/JSON files are included as *reference artifacts*:

* **`v3_T25_sigma3_dense.csv`**  
* **`v3_T25_sigma3_dense.json`**  

  Fixed-$T$ GI–flow experiments ($T=25$, $\sigma=3$ bins), used to populate
  Table~A.1 (numerical diagnostics supporting Appendix~A).

* **`v3_adaptive.csv`**  
* **`v3_adaptive.json`**  

  Adaptive GI–flow experiments with stopping based on a KL threshold,
  illustrating resolution-dependent contraction behavior.

These files are provided to ensure **full numerical reproducibility** of the
assumption-verification step. They are not required to run the main analysis
pipeline unless Appendix~A diagnostics are being regenerated.

### Reproducing the table in Appendix A

To regenerate the numerical diagnostics table reported in Appendix~A, run:

```bash
python gi_assumption_tracker_v3.py   --module scripts/tci_construct_mandelbrot_v002_fixed.py   --sigma-bins 3.0   --T-fixed 25   --bins-start 64 --bins-max 512   --out-prefix v3_T25_sigma3_dense
```

The resulting CSV can be converted directly into the LaTeX table included
in the paper.

---

