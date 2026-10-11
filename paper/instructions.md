# Study scripts for the SPECTRA paper

These are the scripts used for the analyses in the SPECTRA paper, with the file paths replaced by placeholders (`/path/to/...`). They document how each result was produced and are not meant to run on the sample data. For a runnable walk-through on the sample data, see the [README](../README.md) and the notebooks in [`tutorials/`](../tutorials/).

The cluster scripts are SGE job arrays (`qsub <script>`); each has the `#$` header we used and reads its task from `${SGE_TASK_ID}`. The other scripts are run locally. Before running, set the paths at the top of each script.

## Overview

| Script | Run as | What it does | Paper Figures |
|---|---|---|---|
| `01_profile.sh` | SGE job array, one subject per task | 1D and 2D profiles of FA, MD, RD and AxD for every bundle | - |
| `02a_synth_eval.sh`, `02a_synth_eval.py` | SGE job array, one run per task | Synthetic effect injection and detection scoring | - |
| `02b_synth_aggregate.py` | locally | Collects the scores of all runs into three tables | - |
| `02c_synth_plot.py` | locally | Sensitivity and FDP figures, and the sample-size (Gompertz) figure | Figs. 5, 6, 7 |
| `03a_stats_single.sh` | SGE job array, one mode × metric × bundle per task | Fits the model at every cell of one bundle | - |
| `03b_stats_aggregate.sh` | SGE job array, one mode × metric per task | BH-FDR and hFDR-RESEL across all bundles | - |
| `03c_stats_maps.sh` | locally | Exports the significant Cohen's d grids and renders them on the atlas bundles | Fig. 8a |
| `03d_plot_cohen_dist.py` | locally | Distribution of Cohen's d over significant cells, 1D vs 2D | Fig. 8b |

## 1. Profiles

`01_profile.sh` runs `spectra_profile` with 5 mm cells along and across the tract, at least 20 streamlines per bundle, and the linear robust centroid. Run once per cohort (ADNI, HABS-HD). A profile value of 0 marks a cell with no data for that subject.

## 2. Synthetic experiments
> Methods 2.5, Results 3.1

1. **Runs** (`02a_synth_eval.sh`). One task per setting and repetition: peak effect size d ∈ {0.2, 0.5, 0.8}, n ∈ {50, 100, 200, 400, 600, 800, 1000, 1200} per group, B ∈ {5, 10, 15} affected bundles out of 20, diffuse or localized effects, 100 repetitions; 14,400 tasks. Each task calls `02a_synth_eval.py`, which samples 2n controls, injects the effect into one half, fits the models once, and scores BH-FDR, adaptive BH-FDR, hFDR-RESEL and hFDR-BUNDLE against the effect maps. `MODE=2d` gives the main-text results and `MODE=1d` the supplementary ones.
2. **Aggregation** (`02b_synth_aggregate.py`). Run once per α₂.
3. **Figures** (`02c_synth_plot.py`). With the 2D results: block-level sensitivity and FDP (Fig. 5), cell-level sensitivity and FDP (Fig. 6), and the sample size needed for 70% sensitivity from Gompertz fits (Fig. 7). With the 1D results: the figures of Supplementary Section 3.

The injection code is in [`src/spectra/simulation/`](../src/spectra/simulation/), and [`tutorials/spectra_tutorial_synthetic_injection.ipynb`](../tutorials/spectra_tutorial_synthetic_injection.ipynb) walks through one run on the sample data.

## 3. Empirical analysis
> Methods 2.3-2.4, Results 3.2

1. **Model fitting** (`03a_stats_single.sh`). `metric ~ dx + age + sex + (1 | protocol)` at every cell, for 2 modes (1D, 2D) × 4 metrics × 63 bundles. Run once with `target_case=Dementia` and once with `target_case=MCI`.
2. **Correction** (`03b_stats_aggregate.sh`). BH-FDR at α = 0.05, and hFDR-RESEL with Stage 1 α₁ = 0.05 (Simes) and Stage 2 α₂ = 0.05, 0.15 and 0.25. The detection rates in Table 3 come from the `*__stats_summary*.csv` files.
3. **Maps** (`03c_stats_maps.sh`). `spectra_export_stats -sig_only` writes one Cohen's d grid per bundle (hFDR-RESEL, α₂ = 0.15), and `spectra_viz` colors the atlas bundles with them (Fig. 8a).
4. **Effect-size distributions** (`03d_plot_cohen_dist.py`). Fig. 8b.

The split-cohort analysis (Supplementary Section 4) runs steps 1–4 separately on the ADNI and HABS-HD subjects, with each cohort's metadata table.
