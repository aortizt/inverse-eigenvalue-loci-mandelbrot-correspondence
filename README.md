

#**UPDATED README.md** (clean and current)

*(Version with copy-files removed and CSV-path note added)*

```markdown
# Construct–Mandelbrot (CM–TCI) Analysis Codebase

This repository contains the full, reproducible pipeline used in the paper  
**“From Homotopies to Statistics: Quantitative Correspondences between Lucas Eigenvalue Constructs and the Mandelbrot Set”**,  
including boundary generation, curvature analysis (two independent estimators),  
variograms, potential/Laplacian fields, spectral/multifractal analyses,  
diffusion/dynamical embeddings, symmetry diagnostics, and TCI/informational convergence experiments.

> **TL;DR for Spyder users** — Run the numbered steps in **Quick Start (Spyder)** below.  
> **CLI equivalents** are provided immediately after that.

---

## Repository Layout

```

CM-TCI/
├─ 0_data/                                 # (Optional) raw inputs and caches
│   ├─ construct_points.csv                # Construct point cloud (x,y); required for Step 2
│   └─ ...
├─ outputs/                                # All generated results land here by default
│   ├─ mandel_boundary.csv / .png / .txt
│   ├─ construct_boundary.csv / .png / .txt
│   ├─ curv_localpoly/mandel_*             # curvature outputs
│   ├─ curv_localpoly/construct_*          # curvature outputs
│   └─ ...
├─ scripts/
│   ├─ mandelbrot_boundary_sample_spyder.py
│   ├─ construct_boundary_alpha_spyder_v2.py
│   ├─ boundary_curvature_localpoly_spyder.py
│   ├─ mandelbrot_boundary_sample.py
│   ├─ construct_boundary_alpha.py
│   ├─ boundary_curvature_localpoly.py
│   ├─ construct_stage1_clean.py
│   ├─ MandelBoundary.py
│   ├─ mandel2.py
│   ├─ symmetry_phase_bestaxis.py
│   ├─ match_visual_pairs.py
│   ├─ match_analysis_steps1_2.py
│   ├─ spatial_stats_phase2.py
│   ├─ spatial_stats_phase3.py
│   ├─ spatial_stats_phase3b.py
│   ├─ spatial_stats_phase4.py
│   ├─ phase4b_spectral_bootstrap.py
│   ├─ phase5_report.py
│   ├─ spectral_decay_exponent.py
│   ├─ variograms_construct_mandelbrot.py
│   ├─ variograms_construct_mandelbrotv2.py
│   ├─ Iterative_Variogram_Laplacian.py
│   ├─ Variogram-Mandelbrot-Construct.py
│   ├─ Potentials.py
│   ├─ Potentials-C_M.py
│   ├─ Laplacian_C-M.py
│   ├─ multifractal_phase6.py
│   ├─ dynamical_embeddings_phase7.py
│   ├─ tci_construct_mandelbrot.py
│   ├─ tci_construct_mandelbrot-v002.py
│   ├─ tci_construct_mandelbrot_v002_fixed.py
└─ README.md

````

**Note:**  
Some scripts expect `construct_points.csv` and/or `mandel_boundary_sample.csv` to be in the working folder.  
If you run from another directory, either:

1. Place the CSVs next to the script, **or**
2. Edit the `input_csv = ...` line, **or**
3. Add CLI flags such as `--input_csv mypath/file.csv` to make paths explicit.

---

## Quick Start (Spyder)

### **Step 1 — Mandelbrot boundary**
`scripts/mandelbrot_boundary_sample_spyder.py`  
Edit parameters at top if necessary and **Run**.

**Outputs:**  
- `outputs/mandel_boundary.csv`  
- `outputs/mandel_boundary.png`  
- `outputs/mandel_boundary.txt` (summary)

---

### **Step 2 — Construct boundary (alpha-shape)**  
Use the robust version:  
`scripts/construct_boundary_alpha_spyder_v2.py`

Set:
```python
input_csv     = "0_data/construct_points.csv"   # your Construct cloud
alpha         = 65.0
output_prefix = "outputs/construct"
# v2 automatically extracts the main closed loop and can densify to ~1500 points
````

Run.

**Outputs:**

* `outputs/construct_boundary.csv`
* `outputs/construct_boundary.png`
* `outputs/construct_boundary.txt`

---

### **Step 3 — Curvature (local polynomial)**

`scripts/boundary_curvature_localpoly_spyder.py`

Run **once for each boundary**:

```python
# Mandelbrot
input_csv     = "outputs/mandel_boundary.csv"
output_prefix = "outputs/curv_localpoly/mandel"
neighbors     = 7
closed        = True

# Construct
input_csv     = "outputs/construct_boundary.csv"
output_prefix = "outputs/curv_localpoly/construct"
neighbors     = 7
closed        = True
```

**Outputs:**

* `_curvature.csv`
* `_curvature_hist.png`
* `_curvature_overlay.png`
* `_summary.txt`

---

## CLI Equivalents

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

* `construct_points.csv` and `mandel_boundary_sample.csv` must be visible to the script.
* If you work inside Spyder, the simplest approach is to keep them inside **0_data/** and adjust `input_csv = "0_data/construct_points.csv"`.
* Any of the scripts can be trivially extended with a CLI flag such as `--input_csv myfolder/data.csv` for flexibility.

---

## Script Catalog (What Each One Does)

**Boundary generation & preprocessing**

* `mandelbrot_boundary_sample_spyder.py`, `mandelbrot_boundary_sample.py` — Mandelbrot boundary via escape-time grid + isocontours.
* `construct_boundary_alpha_spyder_v2.py`, `construct_boundary_alpha_spyder.py`, `construct_boundary_alpha.py` — Construct boundary from point cloud via alpha-shapes.
* `construct_stage1_clean.py` — optional Construct cloud cleaning and ordering utilities.
* `MandelBoundary.py`, `mandel2.py` — supplementary Mandelbrot boundary generators.

**Curvature**

* `boundary_curvature_localpoly_spyder.py`, `boundary_curvature_localpoly.py` — local-polynomial curvature (quadratic LS in sliding window).

  * Produces curvature CSV, histogram, PDF overlay, summary text.

**Variograms / Spatial statistics**

* `variograms_construct_mandelbrot.py`, `variograms_construct_mandelbrotv2.py`
* `Iterative_Variogram_Laplacian.py`
* `Variogram-Mandelbrot-Construct.py`
* `spatial_stats_phase2.py`, `spatial_stats_phase3.py`, `spatial_stats_phase3b.py`, `spatial_stats_phase4.py`
* `phase4b_spectral_bootstrap.py`
* `phase5_report.py`

**Potentials / Laplacians**

* `Potentials.py`, `Potentials-C_M.py`, `Laplacian_C-M.py`

**Spectral / Multifractal**

* `spectral_decay_exponent.py`
* `multifractal_phase6.py`

**Embeddings / Symmetry / Matching**

* `dynamical_embeddings_phase7.py`
* `symmetry_phase_bestaxis.py`
* `match_visual_pairs.py`, `match_analysis_steps1_2.py`

**TCI**

* `tci_construct_mandelbrot.py`
* `tci_construct_mandelbrot-v002.py`
* `tci_construct_mandelbrot_v002_fixed.py`

---

## Environment

```bash
conda create -n cm-tci python=3.11 numpy scipy matplotlib scikit-learn
conda activate cm-tci
```

---

## Notes

* All scripts are pure Python; no external binaries are needed.
* Curvature neighbors: `7`–`11` is safe for boundaries from ~1k to ~20k points.
* The curvature analysis used in the final paper used `k=7`.
* The v2 alpha-boundary script is recommended because it robustly extracts the main closed component and densifies points evenly.

---

If you use this code in academic work, please cite the associated paper.
For reproducibility questions, open an issue with the script name, parameters, and output prefix used.

```

---

```
