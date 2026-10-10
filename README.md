# Spatial Inference for Tractometry (SPECTRA)

**Spatial Inference for Tractometry (SPECTRA)** is a Python framework for white matter tractometry analysis. It extends along-tract (1D) profiling from Bundle Analytics (BUAN) to a 2D parameterization that captures both along-tract and radial variation across bundle cross-sections, defined directly on atlas streamlines. For statistical inference, SPECTRA implements linear mixed modeling via `pymer4` and a hierarchical FDR procedure (hFDR) that improves statistical power and data efficiency compared to the standard BH FDR procedure. 
<img src="docs/assets/figures/spectra_overview.jpg" alt="My figure" width="600">


## Installation

SPECTRA requires Python ≥ 3.11.
```bash
git clone https://github.com/wendyfyx/SPECTRA.git
cd SPECTRA
```

If you only need to compute tract profiles or generate visualizations with (`spectra_profile`, `spectra_viz`), install with [uv](https://docs.astral.sh/uv/):
```bash
uv sync
```

The statistical analysis pipeline (`spectra_lmm_fdr`) requires [`pymer4`](https://github.com/ejolly/pymer4), which depends on R. Because of this R dependency, we recommend installing `pymer4` in a independent conda environment
```bash
# use conda solver
conda create -n spectra_env -c ejolly -c conda-forge python=3.11 pymer4=0.8.2
conda activate spectra_env
# OR use mamba solver (faster)
mamba create -n spectra_env -c ejolly -c conda-forge python=3.11 pymer4=0.8.2
mamba activate spectra_env
```
Then installing SPECTRA with
```bash
uv pip install . 
# OR editable install
uv pip install -e .
```

Verify install
```bash
spectra_lmm_fdr -h
```

> **Note:** `pymer4` is not listed as a package dependency and must be installed separately. SPECTRA will import successfully without it, but calls to the `spectra_lmm_fdr` CLI or the `spectra.tractometry.MultiBundleLM` module will fail at runtime if `pymer4` is not available.

## CLI Reference

### `spectra_profile`

Compute 1D (BUAN weighted mean) and 2D (SPECTRA grid) microstructure profiles for a single subject's white matter bundle. Outputs one `.npz` file per scalar map containing both profile types.

```bash
cd sample_data/
mkdir -p outputs/

spectra_profile \
    -rec AF_L__rec.trk \
    -org AF_L__org.trk \
    -ref AF_L__atlas.trk \
    -maps fa.nii.gz md.nii.gz \
    -profiles outputs/AF_L_fa outputs/AF_L_md \
    -s 5 -r 5 \
    -robust linear
```

**Output:** One `.npz` per scalar map with keys `ns`, `nr`, `s_len`, `r_len`, `profile_1d` (shape `(ns,)`), `profile_2d` (shape `(ns, nr)`).

---

### `spectra_lmm_fdr`

Two-stage pipeline for LMM-based tractometry with FDR or hierarchical FDR (hFDR) correction across all bundles. Designed for HPC: `single` fits one bundle per job; `aggregate` loads all saved results and applies correction.

> Requires `pymer4`. See [Installation](#installation).

#### Stage 1 — `single`

Fit the model at each grid cell for one bundle. Run as an HPC job array (one job per bundle).

With `--group_col` (e.g. site or protocol), a linear mixed model with a random intercept for that column is fitted. Without it, ordinary least squares is used.

```bash
spectra_lmm_fdr single \
    --metadata path/to/data/subjects.csv \
    --root_dir path/to/data/profiles \
    --output_dir path/to/results/hfdr \
    --bundle AF_L \
    --group_col site \
    --stats_mode hfdr_resel
```

#### Stage 2 — `aggregate`

Load all saved per-bundle results and apply FDR or hFDR correction. Multiple `--alpha_cell` values sweep Stage 2 thresholding without re-fitting.

```bash
# hFDR with alpha_cell sweep
spectra_lmm_fdr aggregate \
    --metadata path/to/data/subjects.csv \
    --root_dir path/to/data/profiles \
    --output_dir path/to/results/hfdr \
    --bundle_list path/to/data/bundles.txt \
    --stats_mode hfdr_resel \
    --alpha_block 0.05 \
    --alpha_cell 0.05 0.15 0.25 \
    --combination_method simes

# Standard BH-FDR
spectra_lmm_fdr aggregate \
    --metadata path/to/data/subjects.csv \
    --root_dir path/to/data/profiles \
    --output_dir path/to/results/fdr \
    --bundle_list path/to/data/bundles.txt \
    --stats_mode fdr \
    --alpha 0.05 --adaptive
```

**Output files:**
- FDR: `{mode}__stats_all.csv`, `{mode}__stats_summary.csv`, optionally `{mode}__fdr_diagnostics.npz`
- hFDR: `{mode}__stats_blocks.csv` (once), `{mode}__stats_all_ac{NN}.csv`, `{mode}__stats_summary_ac{NN}.csv` per `alpha_cell`

---

### `spectra_export_stats`

Export one grid per bundle and statistic from a `stats_all` CSV written by `spectra_lmm_fdr aggregate`. Each file holds an `(ns, nr)` array (`nr = 1` for 1D profiles) with `NaN` for cells that were not tested, and can be passed directly to `spectra_viz -c`.

```bash
spectra_export_stats \
    -i path/to/results/hfdr/hfdr_resel__stats_all_ac15.csv \
    -o path/to/results/hfdr/grids/ \
    -export_stats d_cohen p_hfdr \
    -use_log_p -sig_only
```

- `-use_log_p` writes −log10(p) for columns whose name starts with `p_`.
- `-sig_only` sets non-significant cells to `NaN`.
- `-fmt txt` writes text files instead of `.npy`; `-bundle_list` limits the export to the listed bundles.

**Output files:** `{out_prefix}{bundle}__{stat}.{npy|txt}`

---

### `spectra_viz`

Render 3D bundle visualizations using [FURY](https://fury.gl/). Supports scalar-map coloring, 1D/2D profile arrays, segment-based alternating colors, glass brain overlays, ROI masks, and multiple camera views.

```bash
cd sample_data/
mkdir -p outputs/

# Basic render with glass brain and preset camera with interactive mode
spectra_viz \
    -i AF_L__rec.trk \
    -o outputs/AF_L.png \
    -glass default_glass \
    -cam Sagittal_L \
    -show

# Color by FA volume
spectra_viz \
    -i AF_L__org.trk \
    -o outputs/AF_L_fa.png \
    -c fa.nii.gz \
    -cmap turbo \
    -vmin 0.2 \
    -vmax 0.8 \
    -ctitle FA \
    -cbar outputs/colorbar_fa.png \
    -show

# Grid visualization — color streamlines by 2D (s, r) grid cell
spectra_viz \
    -i AF_L__rec.trk \
    -o outputs/AF_L_grid2d.png \
    -ref AF_L__atlas.trk \
    -c alternate_2d \
    -s 5 -r 5 \
    -cam Sagittal_L \
    -show

# Statistics — color the atlas bundle by a grid from spectra_export_stats
spectra_viz \
    -i AF_L__atlas.trk \
    -o outputs/AF_L_d_cohen.png \
    -c path/to/results/hfdr/grids/AF_L__d_cohen.npy \
    -s 5 -r 5 -robust linear \
    -cmap coolwarm -vmin -0.8 -vmax 0.8 \
    -cam Sagittal_L
```