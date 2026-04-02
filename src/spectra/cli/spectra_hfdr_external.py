#!/usr/bin/env python
"""
DESCRIPTION
    Run SPECTRA hierarchical FDR (hFDR) from externally fitted per-cell statistics.

INPUT FILES
    1. Statistics CSV  (required)
        One combined CSV for all grids. Required columns:

            grid_name, s_idx, r_idx, t_stat, p_value

        Cells absent from the CSV are treated as masked (not tested).
        Grid dimensions ns/nr are inferred as max(s_idx)+1 and max(r_idx)+1
        per grid_name unless --ns and --nr are supplied.

    2. Residuals CSV  (one per grid, optional — needed for RESEL-block mode)
        Long-format CSV with one row per (cell, subject). Required columns:

            s_idx, r_idx, subject_id, residual

        Files must be named  {grid_name}_residuals.csv  inside --residuals_dir.
        The residuals should come from the *reduced* model (covariates only,
        without the target variable), the same residuals used to estimate
        the Matern spatial correlation length scales.

        Not required when --no_resel_blocks is set (bundle-mode hFDR).

USAGE
    # hFDR RESEL — estimate length scales from residuals
    spectra_hfdr_external \\
        --stats_csv      ./results/stats.csv \\
        --residuals_dir  ./results/residuals/ \\
        --output_dir     ./hfdr_output/ \\
        --s_len 5.0 --r_len 5.0 \\
        --alpha_block 0.05 \\
        --alpha_cell  0.05 0.15 0.25 \\
        --combination_method simes

    # hFDR bundle-mode — no residuals needed
    spectra_hfdr_external \\
        --stats_csv   ./results/stats.csv \\
        --output_dir  ./hfdr_output/ \\
        --s_len 5.0 --r_len 5.0 \\
        --no_resel_blocks

OUTPUT FILES
    Written to --output_dir:

        hfdr_external__stats_blocks.csv          — Stage 1 block-level results (once)
        hfdr_external__stats_all_ac{NN}.csv      — cell-level results per alpha_cell
        hfdr_external__stats_summary_ac{NN}.csv  — per-grid summary per alpha_cell

    where {NN} is the alpha_cell percentage (e.g. ac05 for 0.05).
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from spectra.tractometry.MultiBundleLM import _run_hfdr_core
from spectra.tractometry.results_data import BundleResult, MultiBundleHFDRResult
from spectra.detection.resel_stationary import estimate_length_scales
from spectra.utils.utils import setup_logging, make_directory
from spectra.utils.io import load_list


def load_stats_csv(
    path: str,
    grid_name_list: Optional[List[str]] = None,
    ns_override: Optional[int] = None,
    nr_override: Optional[int] = None,
) -> Dict[str, BundleResult]:
    """
    Load per-cell LMM statistics from a combined CSV and build BundleResult objects.

    Parameters
    ----------
    path : str
        Path to stats CSV with columns: grid_name, s_idx, r_idx, t_stat, p_value.
    grid_name_list : list of str, optional
        Restrict to these grid_names. Processes all grids found if None.
    ns_override : int, optional
        Force grid dimension ns for all grids. Inferred as max(s_idx)+1 if None.
    nr_override : int, optional
        Force grid dimension nr for all grids. Inferred as max(r_idx)+1 if None.

    Returns
    -------
    Dict[str, BundleResult]
        Keyed by grid_name. Value is imported as BundleResult object for hFDR.
    """
    df = pd.read_csv(path)

    # Validate required columns
    required = {'grid_name', 's_idx', 'r_idx', 't_stat', 'p_value'}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Stats CSV is missing required columns: {sorted(missing_cols)}. "
            f"Found columns: {list(df.columns)}"
        )

    # Filter to requested grids
    if grid_name_list is not None:
        df = df[df['grid_name'].isin(grid_name_list)]
        if df.empty:
            raise ValueError(
                f"No rows remain after filtering to grid_name_list. "
                f"Check that grid_name values in the CSV match --grid_name_list."
            )

    grids: Dict[str, BundleResult] = {}

    for grid_name, gdf in df.groupby('grid_name'):
        grid_name = str(grid_name)

        s_vals = gdf['s_idx'].values.astype(int)
        r_vals = gdf['r_idx'].values.astype(int)

        ns = ns_override if ns_override is not None else int(s_vals.max()) + 1
        nr = nr_override if nr_override is not None else int(r_vals.max()) + 1

        # Build mask: only cells present in the CSV are valid
        mask = np.zeros((ns, nr), dtype=bool)
        mask[s_vals, r_vals] = True

        # Build stat arrays — NaN outside mask
        def _fill(col: str, default=np.nan) -> np.ndarray:
            arr = np.full((ns, nr), default)
            if col in gdf.columns:
                arr[s_vals, r_vals] = gdf[col].values.astype(float)
            return arr

        t_obs    = _fill('t_stat')
        p_obs    = _fill('p_value')
        beta_obs = _fill('beta')
        se_obs   = _fill('se')
        dof_obs  = _fill('dof')
        d_cohen  = _fill('d_cohen') if 'd_cohen' in gdf.columns else None

        # Sanity check: warn about clearly invalid p-values
        p_valid = p_obs[mask]
        if np.any(p_valid <= 0) or np.any(p_valid > 1):
            n_bad = int(((p_valid <= 0) | (p_valid > 1)).sum())
            logging.warning(
                f"{grid_name}: {n_bad} cells have p-values outside (0, 1]. "
                f"These will produce NaN z-scores and be skipped in Stage 1."
            )

        # Dummy fields not used in hFDR computation
        rpv            = np.zeros((ns, nr))
        resel_density  = 0.0
        n_subj_by_bin  = np.zeros((ns, nr))

        grids[grid_name] = BundleResult(
            name=grid_name,
            ns=ns,
            nr=nr,
            mask=mask,
            rpv=rpv,
            resel_density=resel_density,
            t_obs=np.where(mask, t_obs, np.nan),
            p_obs=np.where(mask, p_obs, np.nan),
            beta_obs=np.where(mask, beta_obs, np.nan),
            se_obs=np.where(mask, se_obs, np.nan),
            dof_obs=np.where(mask, dof_obs, np.nan),
            ci_lower_obs=np.full((ns, nr), np.nan),
            ci_upper_obs=np.full((ns, nr), np.nan),
            r2_partial_obs=np.full((ns, nr), np.nan),
            n_subj=0,
            n_subj_by_bin=n_subj_by_bin,
            d_cohen_obs=np.where(mask, d_cohen, np.nan) if d_cohen is not None else None,
        )
        logging.info(
            f"Loaded {grid_name}: ({ns} x {nr}) grid, "
            f"{int(mask.sum())} valid cells"
        )

    if not grids:
        raise ValueError(f"No valid grids loaded from {path}")

    logging.info(f"Loaded {len(grids)} grids from {path}")
    return grids


def load_residuals_csv(path: str, ns: int, nr: int) -> np.ndarray:
    """
    Load per-cell reduced-model residuals from a long-format CSV.

    Parameters
    ----------
    path : str
        CSV with required columns: s_idx, r_idx, subject_id, residual.
    ns : int
        Number of along-tract segments (s dimension of grid).
    nr : int
        Number of radial bins (r dimension of grid).

    Returns
    -------
    np.ndarray, shape (ns, nr, n_subj)
        Residuals array. Cells absent from the CSV are NaN.
        Subjects are ordered by sorted subject_id.

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    df = pd.read_csv(path)

    required = {'s_idx', 'r_idx', 'subject_id', 'residual'}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Residuals CSV '{path}' is missing columns: {sorted(missing_cols)}. "
            f"Found: {list(df.columns)}"
        )

    subjects = sorted(df['subject_id'].unique())
    n_subj = len(subjects)
    subj_idx = {s: i for i, s in enumerate(subjects)}

    resid = np.full((ns, nr, n_subj), np.nan)

    for _, row in df.iterrows():
        s = int(row['s_idx'])
        r = int(row['r_idx'])
        si = subj_idx[row['subject_id']]
        if 0 <= s < ns and 0 <= r < nr:
            resid[s, r, si] = float(row['residual'])
        else:
            logging.debug(
                f"Residual at s={s}, r={r} is out of grid bounds "
                f"({ns} x {nr}), skipping."
            )

    logging.info(
        f"Loaded residuals from '{Path(path).name}': "
        f"{n_subj} subjects, ({ns} x {nr}) grid"
    )
    return resid


def _build_grid_data(
    grids: Dict[str, BundleResult],
    residuals_dir: Optional[str],
    s_len: float,
    r_len: float,
    use_resel_blocks: bool,
) -> Dict[str, dict]:
    """
    Convert loaded BundleResult objects into the dict format, estimating Matern
    length scales from residuals where available.

    Parameters
    ----------
    grids : Dict[str, BundleResult]
    residuals_dir : str or None
        Directory containing {grid_name}_residuals.csv files.
        If None, length scales are not estimated (bundle-mode fallback).
    s_len, r_len : float
        Grid spacing in mm.
    use_resel_blocks : bool
        When True, attempt to load residuals and estimate length scales.

    Returns
    -------
    Dict[str, dict]  — keyed by grid_name, values match _run_hfdr_core schema.
    """
    bundle_data: Dict[str, dict] = {}

    for grid_name, grid in grids.items():
        valid = grid.mask & ~np.isnan(grid.p_obs)
        if not valid.any():
            logging.warning(f"Skipping {grid_name} — no valid cells after masking")
            continue

        valid_indices = list(zip(*np.where(valid)))
        t_obs = grid.t_obs[valid]
        p_obs = grid.p_obs[valid]

        # Guard against p-values that would produce inf z-scores
        p_safe = np.clip(p_obs, 1e-15, 1 - 1e-15)
        z_cells = np.sign(t_obs) * stats.norm.ppf(1 - p_safe / 2)
        z_cells = np.clip(z_cells, -10, 10)

        # Estimate length scales from residuals if requested
        l_s: Optional[float] = None
        l_r: Optional[float] = None

        if use_resel_blocks and residuals_dir is not None:
            resid_path = Path(residuals_dir) / f"{grid_name}_residuals.csv"
            if resid_path.is_file():
                try:
                    resid_arr = load_residuals_csv(
                        str(resid_path), ns=grid.ns, nr=grid.nr
                    )
                    r_len_for_est = r_len if grid.nr > 1 else None
                    l_s, l_r = estimate_length_scales(
                        resid_arr, s_len, r_len_for_est
                    )
                    logging.info(
                        f"{grid_name}: estimated length scales "
                        f"l_s={l_s:.2f}mm"
                        + (f", l_r={l_r:.2f}mm" if l_r is not None else "")
                    )
                except Exception as e:
                    logging.warning(
                        f"{grid_name}: length scale estimation failed "
                        f"({e}). Falling back to single-bundle block."
                    )
            else:
                logging.warning(
                    f"{grid_name}: residuals file not found at "
                    f"'{resid_path}'. Falling back to single-bundle block."
                )
        elif use_resel_blocks and residuals_dir is None:
            logging.warning(
                f"{grid_name}: --residuals_dir not provided. "
                f"Falling back to single-bundle block."
            )

        # Attach estimated length scales to grid so they appear in output
        grid.l_s = l_s
        grid.l_r = l_r

        bundle_data[grid_name] = {
            'grid':          grid,
            'valid_mask':    valid,
            'cell_z':        z_cells,
            'cell_p':        p_obs,
            'valid_indices': valid_indices,
            'l_s':           l_s,
            'l_r':           l_r,
        }

    return bundle_data


def run_hfdr_from_csv(
    stats_path: str,
    s_len: float,
    r_len: float,
    residuals_dir: Optional[str] = None,
    alpha_block: float = 0.05,
    alpha_cell: Union[float, List[float]] = 0.25,
    use_resel_blocks: bool = True,
    combination_method: str = 'simes',
    grid_name_list: Optional[List[str]] = None,
    ns_override: Optional[int] = None,
    nr_override: Optional[int] = None,
) -> MultiBundleHFDRResult:
    """
    Run SPECTRA hierarchical FDR from externally fitted per-cell statistics.
 
    Parameters
    ----------
    stats_path : str
        Combined CSV with columns: grid_name, s_idx, r_idx, t_stat, p_value.
        Optional: beta, se, dof, d_cohen.
    s_len : float
        Grid spacing in s direction (mm). Must match the profiling step.
    r_len : float
        Grid spacing in r direction (mm). Must match the profiling step.
    residuals_dir : str, optional
        Directory containing per-grid residual CSV files named
        {grid_name}_residuals.csv with columns: s_idx, r_idx, subject_id, residual.
        Required for RESEL-block mode unless use_resel_blocks=False.
    alpha_block : float
        Stage 1 BH-FDR threshold across spatial blocks. Default 0.05.
    alpha_cell : float
        Stage 2 BH-FDR threshold(s) within rejected blocks. 
    use_resel_blocks : bool
        If True (default), partition each grid into RESEL-sized spatial blocks
        using Matern length scales estimated from residuals.
        If False, treat each entire grid as one Stage 1 block (bundle mode).
    combination_method : str
        Stage 1 p-value combination method: 'simes' (default) or 'stouffer'.
    grid_name_list : list of str, optional
        Restrict to these grid_names. All grids in the CSV are used if None.
    ns_override : int, optional
        Force ns for all grids (overrides inference from max s_idx).
    nr_override : int, optional
        Force nr for all grids (overrides inference from max r_idx).
 
    Returns
    -------
    MultiBundleHFDRResult
        Result object with Stage 2 applied at alpha_cell[0] (or the scalar
        alpha_cell). Call result.reapply_stage2(ac) to re-threshold at a
        different alpha_cell without re-fitting.
    """
    # Load stats
    grids = load_stats_csv(
        stats_path,
        grid_name_list=grid_name_list,
        ns_override=ns_override,
        nr_override=nr_override,
    )
 
    # Build bundle_data dict (estimates length scales if residuals available)
    bundle_data = _build_grid_data(
        grids=grids,
        residuals_dir=residuals_dir,
        s_len=s_len,
        r_len=r_len,
        use_resel_blocks=use_resel_blocks,
    )
 
    if not bundle_data:
        raise RuntimeError("No valid grids to process after loading stats CSV.")
 
    mode_str = "RESEL-block" if use_resel_blocks else "bundle"
    logging.info(
        f"Running hFDR on {len(bundle_data)} grids "
        f"(mode={mode_str}, combination={combination_method}, "
        f"alpha_block={alpha_block}, alpha_cell={alpha_cell})"
    )
 
    result = _run_hfdr_core(
        bundle_data=bundle_data,
        s_len=s_len,
        r_len=r_len,
        alpha_block=alpha_block,
        alpha_cell=alpha_cell,
        use_resel_blocks=use_resel_blocks,
        combination_method=combination_method,
    )
 
    return result


def _save_hfdr_results(
    result: MultiBundleHFDRResult,
    output_dir: Path,
    alpha_cell: float,
    save_blocks: bool,
) -> None:
    """Write stats_blocks (once), stats_all, and stats_summary for one alpha_cell."""
    ac_sfx = f"_ac{str(int(alpha_cell * 100)).zfill(2)}"
    stem = "hfdr_external"

    if save_blocks:
        result.get_stats_blocks(str(output_dir / f"{stem}__stats_blocks.csv"))
        logging.info(f"Saved Stage 1 blocks: {stem}__stats_blocks.csv")

    result.get_stats_all(str(output_dir / f"{stem}__stats_all{ac_sfx}.csv"))
    result.get_stats_summary(str(output_dir / f"{stem}__stats_summary{ac_sfx}.csv"))
    logging.info(
        f"Saved hFDR results (alpha_cell={alpha_cell}): "
        f"{stem}__stats_*{ac_sfx}.csv"
    )


def main():
    setup_logging("INFO")

    parser = argparse.ArgumentParser(
        # description=("Run SPECTRA hierarchical FDR from externally fitted statistics "),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )

    # Required
    parser.add_argument(
        '--stats_csv', '-stats', type=str, required=True,
        help=(
            "Combined CSV with per-cell statistics. "
            "Required columns: grid_name, s_idx, r_idx, t_stat, p_value. "
            "Optional: beta, se, dof, d_cohen."
        ),
    )
    parser.add_argument(
        '--output_dir', '-o', type=str, required=True,
        help="Directory for output CSV files.",
    )
    parser.add_argument(
        '--s_len', '-s', type=float, required=True,
        help="Grid spacing in s direction (mm). Must match profiling step.",
    )
    parser.add_argument(
        '--r_len', '-r', type=float, required=True,
        help="Grid spacing in r direction (mm). Must match profiling step.",
    )

    # Optional inputs
    parser.add_argument(
        '--residuals_dir', '-resid', type=str, default=None,
        help=(
            "Directory containing per-grid residual CSV files named "
            "{grid_name}_residuals.csv. "
            "Required columns: s_idx, r_idx, subject_id, residual. "
            "Needed for RESEL-block mode. Omit with --no_resel_blocks."
        ),
    )
    parser.add_argument(
        '--grid_name_list', '-grid_name_list', type=str, default=None,
        help=(
            "Path to text file with grid_names to process (one per line). "
            "Processes all grids in stats_csv if omitted."
        ),
    )
    parser.add_argument(
        '--ns', type=int, default=None,
        help=(
            "Override ns (number of along-tract segments) for all grids. "
            "Inferred as max(s_idx)+1 per grid if omitted."
        ),
    )
    parser.add_argument(
        '--nr', type=int, default=None,
        help=(
            "Override nr (number of radial bins) for all grids. "
            "Inferred as max(r_idx)+1 per grid if omitted."
        ),
    )

    # hFDR parameters
    parser.add_argument(
        '--alpha_block', type=float, default=0.05,
        help="Stage 1 BH-FDR threshold across spatial blocks. Default: 0.05.",
    )
    parser.add_argument(
        '--alpha_cell', type=float, nargs='+', default=[0.05, 0.15, 0.25],
        help=(
            "Stage 2 BH-FDR threshold(s) within rejected blocks. "
            "Multiple values sweep Stage 2 without re-fitting. Default: 0.05 0.15 0.25."
        ),
    )
    parser.add_argument(
        '--combination_method', type=str, default='simes',
        choices=['simes', 'stouffer'],
        help="Stage 1 p-value combination method within blocks. Default: simes.",
    )
    parser.add_argument(
        '--no_resel_blocks', action='store_true',
        help=(
            "Use bundle-mode hFDR: treat each entire grid as one Stage 1 block. "
            "No residuals needed. Equivalent to hfdr_bundle in spectra_lmm_fdr.py."
        ),
    )

    args = parser.parse_args()

    make_directory(args.output_dir)
    output_dir = Path(args.output_dir)

    use_resel_blocks = not args.no_resel_blocks

    # Validate: warn if RESEL-block mode but no residuals provided
    if use_resel_blocks and args.residuals_dir is None:
        logging.warning(
            "--residuals_dir not provided. Length scales cannot be estimated; "
            "all grids will fall back to single-bundle blocks. "
            "Use --no_resel_blocks to suppress this warning."
        )

    # Load grid_name_list if provided
    grid_name_list: Optional[List[str]] = None
    if args.grid_name_list is not None:
        grid_name_list = load_list(args.grid_name_list)
        logging.info(f"Loaded {len(grid_name_list)} grids from {args.grid_name_list}")

    alpha_cell_values: List[float] = args.alpha_cell

    # Run hFDR — returns result with Stage 2 applied at the first alpha_cell value
    result = run_hfdr_from_csv(
        stats_path=args.stats_csv,
        s_len=args.s_len,
        r_len=args.r_len,
        residuals_dir=args.residuals_dir,
        alpha_block=args.alpha_block,
        alpha_cell=alpha_cell_values[0],
        use_resel_blocks=use_resel_blocks,
        combination_method=args.combination_method,
        grid_name_list=grid_name_list,
        ns_override=args.ns,
        nr_override=args.nr,
    )
 
    # Save outputs. stats_blocks is written once (Stage 1 does not change).
    # For each alpha_cell: reapply Stage 2, then save immediately before the
    # next reapply mutates the grids. The first value is already applied.
    for i, ac in enumerate(alpha_cell_values):
        if i > 0:
            logging.info(f"Reapplying Stage 2 at alpha_cell={ac}")
            result.reapply_stage2(ac)
        _save_hfdr_results(result, output_dir, ac, save_blocks=(i == 0))
 
    logging.info(
        f"Done. Saved all hFDR results to {args.output_dir} "
        f"at {len(alpha_cell_values)} alpha_cell level(s)."
    )


if __name__ == '__main__':
    main()