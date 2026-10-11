'''
Step 2a: synthetic effect evaluation (one repetition of one setting).

Called by 02a_synth_eval.sh, once per alpha_cell.

  1. Sample 2n control subjects and split them into two groups of n.
  2. Build effect maps (diffuse or localized) and pick the affected bundles.
  3. Inject the effect into the first group (Cohen's d scale, observed cells only).
  4. Fit the models once, then apply hFDR-RESEL, hFDR-BUNDLE, BH-FDR and
     adaptive BH-FDR. The fits are cached, so later calls with another
     alpha_cell reuse them.
  5. Score detections against the effect maps, per cell and per Stage 1 block.

Output (in the output folder, <setting> encodes the run):
  SYNTH_EVAL__FDR__<setting>.csv          cell-level scores, BH-FDR variants
  SYNTH_EVAL__HFDR__<setting>.csv         cell-level scores, hFDR variants
  SYNTH_EVAL_BLOCK__HFDR__<setting>.csv   block-level scores, hFDR Stage 1
  HFDR_{CELLS,SUMMARY,BLOCKS}__<setting>.csv, FDR_SUMMARY__<setting>.csv
  effect_maps_<setting>.npz               injected effect maps and realized d

USAGE:
    python 02a_synth_eval.py -o /path/to/output -n 100 -d 0.5 -b 5 -loc diffuse -ac 0.15

    # Reuse the cached fits for another alpha_cell (hFDR only)
    python 02a_synth_eval.py -o /path/to/output -n 100 -d 0.5 -b 5 -loc diffuse -ac 0.25 --methods hfdr
'''
import logging
import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from spectra.tractometry.BundleProfileLoader import BundleProfileLoader
from spectra.tractometry.MultiBundleLM import MultiBundleLM
from spectra.tractometry.results_data import MultiBundleHFDRResult
from spectra.simulation.evaluation import mask_metrics, evaluate_synthetic_results
from spectra.utils.utils import get_rng, setup_logging, file_exists
from spectra.utils.io import load_list, save_csv

## -------------------------------------------------------------------------
## GLOBAL VARIABLES
## -------------------------------------------------------------------------

S_LEN = 5
R_LEN = 5
ROOT_DIR = f"/path/to/profiles/s{S_LEN}_r{R_LEN}"
BUNDLES = load_list("/path/to/metadata/bundle_list_synth_n20.txt")
DF_META = pd.read_csv("/path/to/metadata/metadata.csv")

SUBJ_COL  = 'subject'
TARGET_COL = 'age'
GROUP_COL  = 'protocol'
COVAR_COLS = ['sex']
CONT_COLS  = ['age']

# Use control subjects as the pool for injecting synthetic effects
SUBJ_POOL = DF_META.loc[DF_META['dx'] == 'CN', SUBJ_COL].values

# Effect-map configurations keyed by --locality argument
LOCALITY_CONFIGS = {
    'diffuse': dict(
        n_blobs=1,
        center_range=(0.3, 0.7),
        extent_range=(0.3, 0.5),
    ),
    'localized': dict(
        n_blobs=4,
        center_range=(0.2, 0.8),
        extent_range=(0.05, 0.15),
    ),
}

## -------------------------------------------------------------------------
## ARGUMENT PARSING
## -------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run FDR and/or hierarchical FDR synthetic effect experiments"
    )

    # Seed arguments (final seed = base_seed + run_id)
    parser.add_argument("--base_seed", "-seed", type=int, default=2026,
        help="Base random seed")
    parser.add_argument("--run_id", "-id", type=int, default=0,
        help="ID of this repetition, added to base_seed")

    # Data / output
    parser.add_argument("--output_folder", "-o", type=str, required=True,
        help="Output directory")
    parser.add_argument("--metric", "-metric", type=str, default='fa',
        help="Diffusion metric (e.g. fa, md)")
    parser.add_argument("--mode", "-mode", type=str, default='1d',
        choices=['1d', '2d'], help="Bundle profile mode")
    parser.add_argument("--force", '-f', action="store_true",
        help="Force overwrite existing output files")

    # Experiment settings
    parser.add_argument("--n_per_group", "-n", type=int, default=100,
        help="Subjects per group")
    parser.add_argument("--max_effect_size", "-d", type=float, default=0.5,
        help="Maximum Cohen's d to inject")
    parser.add_argument("--n_affected_bundles", "-b", type=int, default=5,
        help="Number of bundles to inject effects into")
    parser.add_argument("--locality", "-loc", type=str, default='diffuse',
        choices=list(LOCALITY_CONFIGS.keys()),
        help="Effect spatial pattern: diffuse / localized")
    parser.add_argument("--combination", "-comb", type=str, default='simes',
        choices=['simes', 'stouffer'],
        help="Combination method for block statistics: simes / stouffer")

    # Method selection
    parser.add_argument("--methods", type=str, default='both',
        choices=['fdr', 'hfdr', 'both'],
        help="Which methods to run")

    # FDR / alpha settings
    parser.add_argument("--alpha", "-alpha", type=float, default=0.05,
        help="Alpha level for FDR and hFDR Stage 1")
    parser.add_argument("--alpha_cell", "-ac", type=float, default=0.25,
        help="Alpha level for hFDR Stage 2 cell-level FDR")

    return parser.parse_args()


## -------------------------------------------------------------------------
## HELPER FUNCTIONS
## -------------------------------------------------------------------------

def get_loader(mode: str, metric: str) -> BundleProfileLoader:
    return BundleProfileLoader(
        metadata_df=DF_META,
        bundles=BUNDLES,
        metric=metric,
        root_dir=ROOT_DIR,
        mode=mode,
        subj_col=SUBJ_COL,
        target_col=TARGET_COL,
        group_col=GROUP_COL,
        covar_cols=COVAR_COLS,
        cont_cols=CONT_COLS,
        disable_tqdm=True,
    )


def _run_setting(args) -> str:
    """Filename stem shared by all outputs of one run."""
    return (
        f"{args.metric}_{args.mode}"
        f"_n{args.n_per_group}_d{args.max_effect_size}"
        f"_b{args.n_affected_bundles}_loc{args.locality}"
        f"_run{args.run_id}"
    )


def _fdr_setting(args, adaptive: bool) -> str:
    """Unique filename stem for one FDR variant."""
    return (
        f"FDR__{args.metric}_{args.mode}_s{S_LEN}_r{R_LEN}"
        f"_n{args.n_per_group}_d{args.max_effect_size}"
        f"_b{args.n_affected_bundles}_loc{args.locality}"
        f"_run{args.run_id}"
        + ("_adaptive" if adaptive else "")
    )


def _hfdr_setting(args, resel_blocks: bool) -> str:
    """Unique filename stem for one hFDR variant."""
    return (
        f"HFDR__{args.metric}_{args.mode}_s{S_LEN}_r{R_LEN}"
        f"_n{args.n_per_group}_d{args.max_effect_size}"
        f"_b{args.n_affected_bundles}_loc{args.locality}"
        f"_ac{args.alpha_cell}"
        f"_run{args.run_id}"
        + ("_resel" if resel_blocks else "_bundle")
    )


def evaluate_block_results(
    result: MultiBundleHFDRResult,
    effect_maps: Dict[str, np.ndarray],
    d_maps: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """
    Evaluate Stage 1 detection at the block level.

    A block is a true positive if ANY of its cells has a non-zero entry in
    ``effect_maps``.  ``d_maps`` are used only for reporting max_d.

    Returns a DataFrame with one row per block plus a GLOBAL summary row
    containing classification metrics and sqrt(c*h) diagnostics.

    sqrt(c*h) diagnostic
    --------------------
    For each block:
      c = number of cells in the block
      h = fraction of cells with effect_map > 0
      sqrt_ch = sqrt(c * h)
    Values > 1 indicate the regime where cluster testing gains power over
    cell-level testing (Benjamini & Heller 2007, Theorem 1).
    """
    rows = []

    for blk in result.block_results:
        bundle    = blk.bundle_name
        effect_map = effect_maps.get(bundle)
        d_map      = d_maps.get(bundle)
        n_cells    = blk.n_cells

        if n_cells > 0 and blk.cell_indices:
            s_arr = np.array([c[0] for c in blk.cell_indices])
            r_arr = np.array([c[1] for c in blk.cell_indices])

            # Ground truth
            if effect_map is not None:
                cell_effect = (effect_map[s_arr] if effect_map.ndim == 1
                               else effect_map[s_arr, r_arr])
                true_has_effect = bool(np.any(cell_effect > 0))
                h = float(np.mean(cell_effect > 0))
            else:
                true_has_effect = False
                h = 0.0

            # Realized Cohen's d for reporting only
            if d_map is not None:
                cell_d = (d_map[s_arr] if d_map.ndim == 1
                          else d_map[s_arr, r_arr])
                max_d = float(np.nanmax(cell_d))
            else:
                max_d = 0.0
        else:
            true_has_effect = False
            h = 0.0
            max_d = 0.0

        sqrt_ch = float(np.sqrt(n_cells * h)) if n_cells > 0 else 0.0

        rows.append({
            'bundle':           bundle,
            'block_id':         blk.block_id,
            'n_cells':          n_cells,
            'true_has_effect':  true_has_effect,
            'pred_significant': blk.significant,
            'max_d':            max_d,
            'h':                h,
            'sqrt_ch':          sqrt_ch,
            'z':                blk.z,
            'p':                blk.p,
            'p_adj':            blk.p_adj,
        })

    # Global classification metrics
    true_labels = np.array([r['true_has_effect']  for r in rows])
    pred_labels = np.array([r['pred_significant'] for r in rows])
    block_metrics = mask_metrics(true_labels, pred_labels)

    # sqrt(c*h) summary across all blocks
    sqrt_ch_vals = np.array([r['sqrt_ch'] for r in rows])
    mean_c = float(np.mean([r['n_cells'] for r in rows]))

    global_row = {
        'bundle':           'GLOBAL',
        'block_id':         np.nan,
        'n_cells':          sum(r['n_cells'] for r in rows),
        'true_has_effect':  int(true_labels.sum()),
        'pred_significant': int(pred_labels.sum()),
        'max_d':            np.nan,
        'h':                np.nan,
        'sqrt_ch':          np.nan,
        'z':                np.nan,
        'p':                np.nan,
        'p_adj':            np.nan,
        # Classification metrics
        'TP':        block_metrics['TP'],
        'FP':        block_metrics['FP'],
        'FN':        block_metrics['FN'],
        'TN':        block_metrics['TN'],
        'precision': block_metrics['precision'],
        'recall':    block_metrics['recall'],
        'f1':        block_metrics['f1'],
        'accuracy':  block_metrics['accuracy'],
        # sqrt(c*h) diagnostics
        'sqrt_ch_mean':         float(np.mean(sqrt_ch_vals)),
        'sqrt_ch_median':       float(np.median(sqrt_ch_vals)),
        'sqrt_ch_frac_above1':  float(np.mean(sqrt_ch_vals > 1)),
        'mean_block_size':      mean_c,
        # hFDR diagnostics
        'n_blocks':         result.n_blocks,
        'n_blocks_sig':     result.n_blocks_sig,
        'block_threshold':  result.block_threshold,
        'use_resel_blocks': result.use_resel_blocks,
    }
    rows.append(global_row)

    return pd.DataFrame(rows)


def save_hfdr_outputs(result, effect_maps, d_maps, out, setting):
    """Score one hFDR result and save its tables."""
    df_eval_cells  = evaluate_synthetic_results(result, effect_maps, d_maps)
    df_eval_blocks = evaluate_block_results(result, effect_maps, d_maps)

    result.get_stats_all(f"{out}/HFDR_CELLS__{setting}.csv")
    result.get_stats_summary(f"{out}/HFDR_SUMMARY__{setting}.csv")
    result.get_stats_blocks( f"{out}/HFDR_BLOCKS__{setting}.csv")
    save_csv(f"{out}/SYNTH_EVAL__{setting}.csv",       df_eval_cells,  force=True)
    save_csv(f"{out}/SYNTH_EVAL_BLOCK__{setting}.csv", df_eval_blocks, force=True)


## -------------------------------------------------------------------------
## MAIN
## -------------------------------------------------------------------------

def run(args):
    out = args.output_folder
    Path(out).mkdir(parents=True, exist_ok=True)

    run_fdr  = args.methods in ('fdr',  'both')
    run_hfdr = args.methods in ('hfdr', 'both')

    # Skip entirely if all requested outputs already exist
    fdr_sentinel  = f"{out}/SYNTH_EVAL__{_fdr_setting(args, False)}.csv"
    hfdr_sentinel = f"{out}/SYNTH_EVAL__{_hfdr_setting(args, True)}.csv"

    fdr_done  = (not run_fdr)  or (file_exists(fdr_sentinel)  and not args.force)
    hfdr_done = (not run_hfdr) or (file_exists(hfdr_sentinel) and not args.force)

    if fdr_done and hfdr_done:
        logging.info(f"All requested outputs exist, skipping run {args.run_id}")
        return

    # Shared setup: loader, subjects, effect maps, effect injection
    rng = get_rng(args.base_seed + args.run_id)
    logging.info(f"run_id={args.run_id}, methods={args.methods}, locality={args.locality}")

    loader = get_loader(args.mode, args.metric)
    subjs_pool = np.intersect1d(loader.subjects, SUBJ_POOL)
    logging.info(
        f"Sampling {args.n_per_group} subjects per group from "
        f"{len(subjs_pool)}/{len(SUBJ_POOL)} available controls"
    )
    subjs_selected = rng.choice(subjs_pool, size=2 * args.n_per_group, replace=False)
    loader.set_subjects(subjs_selected)

    # Build effect maps using the chosen locality configuration
    effect_maps_all = loader.create_effect_maps(
        s_len=S_LEN,
        r_len=R_LEN,
        max_effect_range=(args.max_effect_size, args.max_effect_size),
        min_effect=0.1,
        decay='cosine',
        geometry='ellipse',
        rng=rng,
        **LOCALITY_CONFIGS[args.locality],
    )

    # Randomly select affected bundles
    n_affected = min(args.n_affected_bundles, len(BUNDLES))
    affected_bundles = list(rng.choice(BUNDLES, size=n_affected, replace=False))
    logging.info(
        f"Injecting effects into {n_affected}/{len(BUNDLES)} bundles: {affected_bundles}"
    )
    effect_maps_filtered = {b: effect_maps_all[b] for b in affected_bundles}

    # Inject effects into first half of selected subjects ("case" group)
    subjs_effect = subjs_selected[:args.n_per_group]
    d_maps = loader.inject_synthetic_effect(
        effect_maps=effect_maps_filtered,
        effect_size=1.0,   # effect_maps already encode Cohen's d scale
        effect_subjects=subjs_effect,
        effect_col_name='synthetic_group',
    )

    # Fill zero maps for unaffected bundles (needed for evaluation)
    for b in BUNDLES:
        if b not in effect_maps_filtered:
            ns, nr, mask = loader.get_grid_info(b)
            shape = (ns, nr) if nr > 1 else (ns,)
            effect_maps_all[b] = np.zeros(shape)
            d_maps[b]          = np.zeros(shape)

    effect_maps_path = f"{out}/effect_maps_{_run_setting(args)}.npz"
    if not file_exists(effect_maps_path) or args.force:
        np.savez_compressed(
            effect_maps_path,
            **{f'effect_{b}': effect_maps_all[b] for b in BUNDLES},
            **{f'd_{b}': d_maps[b] for b in BUNDLES},
            bundles=np.array(BUNDLES),
        )
        logging.info(f"Saved effect maps: {effect_maps_path}")

    # Run-specific cache for the fitted models. The fits are saved once and
    # reused by all methods (hFDR-RESEL -> hFDR-BUNDLE -> BH-FDR), and each
    # concurrent job writes to its own subdirectory.
    lmm_tmp_dir = f"{out}/.lmm_cache/{_run_setting(args)}"
    lmm_synth = MultiBundleLM(
        dataloader=loader,
        s_len=S_LEN,
        r_len=R_LEN,
        group_col=GROUP_COL,
        target_col='synthetic_group',
        target_ref_group=0,
        target_case_group=1,
        covar_cols=['age', 'sex'],
        disable_tqdm=True,
        output_dir=lmm_tmp_dir,
    )
    saved_obs_dir = Path(lmm_tmp_dir) / 'observed'

    # ------------------------------------------------------------------
    # hFDR branch
    # ------------------------------------------------------------------
    if not hfdr_done:
        logging.info("Running hFDR branch")
        cache_exists = saved_obs_dir.exists() and any(saved_obs_dir.glob('*.npz'))

        # --- hFDR-RESEL: fits full + reduced models, or loads the cached fits ---
        setting_resel = _hfdr_setting(args, resel_blocks=True)
        sentinel_resel = f"{out}/SYNTH_EVAL__{setting_resel}.csv"

        if file_exists(sentinel_resel) and not args.force:
            if cache_exists:
                logging.info(f"hFDR-RESEL output exists, skipping: {sentinel_resel}")
            else:
                logging.warning(
                    "hFDR-RESEL output exists but no cached fits found "
                    f"in {saved_obs_dir}. hFDR-BUNDLE will be skipped. "
                    "Re-run with --force to regenerate both."
                )
        else:
            hfdr_result_resel = lmm_synth.fit_hierarchical_fdr(
                alpha_block=args.alpha,
                alpha_cell=args.alpha_cell,
                use_resel_blocks=True,
                combination_method=args.combination,
                save=not cache_exists or args.force,
                from_saved=cache_exists and not args.force
            )
            save_hfdr_outputs(hfdr_result_resel, effect_maps_all, d_maps, out, setting_resel)
            logging.info(f"Saved hFDR-RESEL results: {setting_resel}")

            # Update cache check in case this run fitted the models
            cache_exists = saved_obs_dir.exists() and any(saved_obs_dir.glob('*.npz'))

        # --- hFDR-BUNDLE: loads the cached fits, no refitting ---
        setting_bundle = _hfdr_setting(args, resel_blocks=False)
        sentinel_bundle = f"{out}/SYNTH_EVAL__{setting_bundle}.csv"
        if file_exists(sentinel_bundle) and not args.force:
            logging.info(f"hFDR-BUNDLE output exists, skipping: {sentinel_bundle}")
        elif cache_exists:
            hfdr_result_bundle = lmm_synth.fit_hierarchical_fdr(
                alpha_block=args.alpha,
                alpha_cell=args.alpha_cell,
                use_resel_blocks=False,
                combination_method=args.combination,
                from_saved=True,
            )
            save_hfdr_outputs(hfdr_result_bundle, effect_maps_all, d_maps, out, setting_bundle)
            logging.info(f"Saved hFDR-BUNDLE results: {setting_bundle}")

        logging.info("hFDR branch complete")

    # ------------------------------------------------------------------
    # FDR branch: loads the same cached fits, applies BH only
    # ------------------------------------------------------------------
    if not fdr_done:
        logging.info("Running FDR branch")

        # If hFDR was skipped (--methods fdr) there is no cache, so fit now
        cache_exists = saved_obs_dir.exists() and any(saved_obs_dir.glob('*.npz'))
        result = lmm_synth.fit_observed_fdr(
            weight_by=None,
            alpha=args.alpha,
            adaptive=False,
            from_saved=cache_exists,
        )

        for adaptive_mode in [True, False]:
            result.apply_global_fdr(
                weight_by=None,
                alpha=args.alpha,
                adaptive=adaptive_mode,
            )

            df_eval = evaluate_synthetic_results(result, effect_maps_all, d_maps)
            setting = _fdr_setting(args, adaptive_mode)

            result.get_stats_summary(f"{out}/FDR_SUMMARY__{setting}.csv")
            if adaptive_mode:
                result.save_fdr_diagnostics(f"{out}/FDR_DIAG__{setting}.npz")
            save_csv(f"{out}/SYNTH_EVAL__{setting}.csv", df_eval, force=True)

        logging.info("FDR branch complete")


if __name__ == '__main__':
    setup_logging('INFO')
    args = parse_args()
    print(args)
    run(args)
