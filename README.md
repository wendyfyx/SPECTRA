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

All examples below run on the files in `sample_data/`, which come from the publicly available DIPY-processed PPMI data derivatives [[1]](#citation):
- `sample_data/profile/`: the left corticospinal tract (CST_L) of one control subject, with its FA and MD maps and the atlas bundle [[2]](#citation).
- `sample_data/stats/`: FA profiles of 3 bundles (AF_L, CST_L, CC_ForcepsMajor) for all 64 subjects (32 controls, 32 patients), computed with `spectra_profile`, and a metadata table.

### `spectra_profile`

Compute 1D (BUAN weighted mean) and 2D (SPECTRA grid) microstructure profiles for a single subject's white matter bundle. Outputs one `.npz` file per scalar map containing both profile types.

```bash
cd sample_data/profile/
mkdir -p outputs/

spectra_profile \
    -rec CST_L__rec.trk \
    -org CST_L__org.trk \
    -ref CST_L__atlas.trk \
    -maps fa.nii.gz md.nii.gz \
    -profiles outputs/CST_L_fa outputs/CST_L_md \
    -s 5 -r 5 \
    -robust linear

# Verify outputs: list the keys and print the profile shapes
python -c "import numpy as np; p = np.load('outputs/CST_L_fa.npz'); print(list(p.keys())); print('1D:', p['profile_1d'].shape, '2D:', p['profile_2d'].shape)"
```

**Output:** One `.npz` per scalar map with keys `ns`, `nr`, `s_len`, `r_len`, `profile_1d` (shape `(ns,)`), `profile_2d` (shape `(ns, nr)`).

---

### `spectra_lmm_fdr`

Two-stage pipeline for LMM-based tractometry with FDR or hierarchical FDR (hFDR) correction across all bundles. Designed for HPC: `single` fits one bundle per job; `aggregate` loads all saved results and applies correction.

> Requires `pymer4`. See [Installation](#installation).

The sample data (64 subjects, 3 bundles) shows no significant patient vs. control difference, so the commands below run end to end but report no significant cells. `age`, `sex` and `site` in `metadata.csv` are simulated.

#### Stage 1 — `single`

Fit the model at each grid cell for one bundle. Run as an HPC job array (one job per bundle). With `--group_col` (e.g. site or protocol), a linear mixed model with a random intercept for that column is fitted. Without it, ordinary least squares is used.

```bash
cd sample_data/stats/

for bundle in $(cat bundle_list.txt); do
    spectra_lmm_fdr single \
        --metadata metadata.csv \
        --root_dir . \
        --output_dir outputs \
        --metric fa --mode 2d \
        --target_col dx --target_ref control --target_case patient \
        --covar_cols age sex --cont_cols age \
        --group_col site \
        --bundle ${bundle} \
        --stats_mode hfdr_resel
done
```

#### Stage 2 — `aggregate`

Load all saved per-bundle results and apply FDR or hFDR correction. Multiple `--alpha_cell` values sweep Stage 2 thresholding without re-fitting.

```bash
cd sample_data/stats/

# hFDR with alpha_cell sweep
spectra_lmm_fdr aggregate \
    --metadata metadata.csv \
    --root_dir . \
    --output_dir outputs \
    --metric fa --mode 2d \
    --target_col dx --target_ref control --target_case patient \
    --covar_cols age sex --cont_cols age \
    --group_col site \
    --bundle_list bundle_list.txt \
    --stats_mode hfdr_resel \
    --alpha_block 0.05 \
    --alpha_cell 0.05 0.15 \
    --combination_method simes

# Standard BH-FDR
spectra_lmm_fdr aggregate \
    --metadata metadata.csv \
    --root_dir . \
    --output_dir outputs \
    --metric fa --mode 2d \
    --target_col dx --target_ref control --target_case patient \
    --covar_cols age sex --cont_cols age \
    --group_col site \
    --bundle_list bundle_list.txt \
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
cd sample_data/stats/

spectra_export_stats \
    -i outputs/hfdr_resel__stats_all_ac15.csv \
    -o outputs/grids/ \
    -export_stats d_cohen p_obs \
    -use_log_p
```

- `-use_log_p` writes −log10(p) for columns whose name starts with `p_`.
- `-sig_only` sets non-significant cells to `NaN` (not used above, since the sample data has no significant cells).
- `-fmt txt` writes text files instead of `.npy`; `-bundle_list` limits the export to the listed bundles.

**Output files:** `{out_prefix}{bundle}__{stat}.{npy|txt}`

---

### `spectra_viz`

Render 3D bundle visualizations using [FURY](https://fury.gl/). Supports scalar-map coloring, 1D/2D profile arrays, segment-based alternating colors, glass brain overlays, ROI masks, and multiple camera views.

```bash
cd sample_data/profile/
mkdir -p outputs/

# Basic render with glass brain and preset camera with interactive mode
spectra_viz \
    -i CST_L__rec.trk \
    -o outputs/CST_L.png \
    -glass default_glass \
    -cam Sagittal_L \
    -show

# Color by FA volume
spectra_viz \
    -i CST_L__org.trk \
    -o outputs/CST_L_fa.png \
    -c fa.nii.gz \
    -cmap turbo \
    -vmin 0.2 \
    -vmax 0.8 \
    -ctitle FA \
    -cbar outputs/colorbar_fa.png \
    -cam Sagittal_L \
    -show

# Grid visualization — color streamlines by 2D (s, r) grid cell
spectra_viz \
    -i CST_L__rec.trk \
    -o outputs/CST_L_grid2d.png \
    -ref CST_L__atlas.trk \
    -c alternate_2d \
    -s 5 -r 5 \
    -cam Sagittal_L \
    -show

# Statistics — color the atlas bundle by a grid from spectra_export_stats
spectra_viz \
    -i CST_L__atlas.trk \
    -o outputs/CST_L_d_cohen.png \
    -c ../stats/outputs/grids/CST_L__d_cohen.npy \
    -s 5 -r 5 -robust linear \
    -cmap coolwarm -vmin -0.8 -vmax 0.8 \
    -cam Sagittal_L
```

## Citation

1. Chandio, Bramsh Qamar (2020). DIPY Processed Parkinson's Progression Markers Initiative (PPMI) Data Derivatives. The NIH Figshare Archive. Dataset. https://doi.org/10.35092/yhjc.12033390.v1
2. Garyfallidis, Eleftherios (2020). Atlas of 30 Human Brain Bundles in MNI space. figshare. Dataset. https://doi.org/10.6084/m9.figshare.12089652.v5
