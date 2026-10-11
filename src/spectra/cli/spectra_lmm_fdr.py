#!/usr/bin/env python
"""
DESCRIPTION
    Two-stage workflow for FDR- or hFDR-corrected multi-bundle LMM analysis.

    1. Run 'single' for each bundle (parallelizable via HPC job arrays).
       Use --stats_mode to determine what is fitted:
         fdr          - full model only (fast)
         hfdr_resel   - full + reduced model with RESEL-based block partitioning)
         hfdr_bundle  - full + reduced model with bundle-based block partitioning)

    2. Run 'aggregate' to load saved per-bundle results and apply correction.
       --stats_mode must match what was used in 'single'.
       For hFDR modes, pass multiple --alpha_cell values to sweep Stage 2
       thresholding without re-fitting any models.

USAGE
    # hFDR RESEL — Stage 1 (per bundle)
    spectra_lmm_fdr single \\
        --metadata /data/subjects.csv \\
        --root_dir /data/profiles \\
        --output_dir ./results/hfdr \\
        --bundle_list bundles.txt \\
        --bundle AF_R \\
        --stats_mode hfdr_resel

    # hFDR RESEL — Stage 2 with alpha_cell sweep
    spectra_lmm_fdr aggregate \\
        --metadata /data/subjects.csv \\
        --root_dir /data/profiles \\
        --output_dir ./results/hfdr \\
        --bundle_list bundles.txt \\
        --stats_mode hfdr_resel \\
        --alpha_block 0.05 \\
        --alpha_cell 0.05 0.15 0.25 \\
        --combination_method simes

    # FDR — Stage 2 (aggregate)
    python spectra_lmm_fdr.py aggregate \\
        --metadata /data/subjects.csv \\
        --root_dir /data/profiles \\
        --output_dir ./results/fdr \\
        --bundle_list bundles.txt \\
        --stats_mode fdr \\
        --alpha 0.05 \\
        --adaptive


NOTES - HFDR vs. FDR
    We recommend always running Stage 1 (single) with --stats_mode hfdr_resel,
    which fits the reduced mode in addition to the full model, so that all FDR
    and hFDR modes can be applied in Stage 2. If Stage 1 was run with FDR or 
    hFDR-Bundle mode, then running Stage 2 with `hfdr_resel` mode will default 
    to hFDR-Bundle.

NOTES - TARGET VARIABLE CONFIGURATION
    The --target_col, --target_ref, and --target_case arguments together
    determine how the outcome variable is modelled.

    For continuous target (e.g. age, cognitive score), omit --target_ref and 
    --target_case. The column enters the LMM as a numeric predictor; the 
    output coefficient is the slope. Cohen's d is not computed.
        --target_col mmse

    For Categorical target with string labels (e.g. diagnosis groups), pass both
    --target_ref and --target_case. The reference group is set as the LMM baseline; 
    the coefficient is the case minus reference contrast.
        --target_col dx --target_ref CN --target_case AD

    For Categorical target with integer labels (e.g. 0/1 or 1/2/3 codes), same as above. 
    Integer-valued strings are automatically cast to int, so the values in your CSV must 
    be stored as integers, not floats. A column containing 0.0/1.0 will not be detected 
    as categorical and will be modelled as continuous.
            --target_col dx --target_ref 0 --target_case 1

    WARNING: the continuous/categorical distinction is driven entirely by
    whether --target_ref and --target_case are supplied, not by column dtype.
    A binary 0/1 column passed without --target_ref will be treated as a linear predictor. 
    Always specify --target_ref and --target_case for binary outcomes. Multi-class is 
    currently not support as only one contrast is tested.
"""

import argparse
import logging
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

from spectra.tractometry.BundleProfileLoader import BundleProfileLoader
from spectra.tractometry.MultiBundleLM import MultiBundleLM
from spectra.tractometry.results_data import MultiBundleResult, MultiBundleHFDRResult
from spectra.utils.io import load_list
from spectra.utils.utils import make_directory, setup_logging


def add_dataloader_args(parser: argparse.ArgumentParser) -> None:
    """Add BundleProfileLoader arguments to parser."""
    # Required inputs
    parser.add_argument('--metadata', '-meta', type=str, required=True,
                        help='Path to metadata CSV file')
    parser.add_argument('--root_dir', '-root', type=str, required=True,
                        help='Root directory containing metric/bundle folders')

    # Data configuration
    parser.add_argument('--metric', '-metric', type=str, default='fa',
                        help='Metric name')
    parser.add_argument('--mode', '-mode', type=str, default='1d',
                        choices=['1d', '2d'],
                        help='Profile mode')
    parser.add_argument('--file_ext', '-ext', type=str, default='npz',
                        help='File extension for profile files')
    parser.add_argument('--min_prop', '-min_prop', type=float, default=0.3,
                        help='Minimum proportion of subjects that must have non-zero data '
                        'at a grid cell for it to be included in the spatial mask.')
    parser.add_argument('--edge_mask', '-edge_mask', type=int, nargs='+', default=None,
                        help='Edge cells to mask: one int (s only) or two ints (s, r)')

    # Metadata column names
    parser.add_argument('--subj_col', '-subj', type=str, default='subject',
                        help='Subject ID column in metadata')
    parser.add_argument('--target_col', '-target', type=str, default='dx',
                        help='Target variable column')
    parser.add_argument('--target_ref', '-target_ref', type=str, default=None,
                        help='Reference group for categorical target')
    parser.add_argument('--target_case', '-target_case', type=str, default=None,
                        help='Case group for categorical target')
    parser.add_argument('--group_col', '-group', type=str, default=None,
                        help='Column for random intercept (e.g. site). If omitted, '
                        'an ordinary linear regression is fitted instead of an LMM.')
    parser.add_argument('--covar_cols', '-covar', type=str, nargs='*',
                        default=['age', 'sex'],
                        help='Covariate columns')
    parser.add_argument('--cont_cols', '-cont', type=str, nargs='*',
                        default=['age'],
                        help='Continuous columns to standardize')

    # Flags
    parser.add_argument('--require_common', '-common', action='store_true',
                        help='Use only subjects with data for all bundles')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Disable progress bars')


def add_lmm_args(parser: argparse.ArgumentParser) -> None:
    """Add MultiBundleLM and stats-mode arguments to parser."""
    parser.add_argument('--output_dir', '-o', type=str, required=True,
                        help='Directory for results')
    parser.add_argument('--s_len', '-s', type=float, default=5.0,
                        help='Grid spacing in s direction (mm)')
    parser.add_argument('--r_len', '-r', type=float, default=5.0,
                        help='Grid spacing in r direction (mm)')
    parser.add_argument('--min_subj', '-min_subj', type=int, default=20,
                        help='Minimum number of subjects with valid data required to fit '
                        'the LMM at a grid cell. Safety guard against degenerate fits '
                        'in cells that passed min_prop but have too few observations ')
    parser.add_argument('--stats_mode', '-stats_mode', type=str, default='hfdr_resel',
                        choices=['fdr', 'hfdr_resel', 'hfdr_bundle'],
                        help=(
                            'Statistical correction mode. '
                            'fdr: standard BH-FDR across all cells. '
                            'hfdr_resel: two-stage hierarchical FDR with RESEL blocks '
                            'hfdr_bundle: two-stage hierarchical FDR with bundle blocks '
                        ))


def add_aggregate_args(parser: argparse.ArgumentParser) -> None:
    """Add aggregate-subcommand-specific arguments (FDR and hFDR)."""

    # Required by both FDR and hFDR
    parser.add_argument('--bundle_list', '-bundle_list', type=str, required=True,
                    help='Path to text file with bundle names (required for aggregate)')

    # FDR-specific
    parser.add_argument('--alpha', '-alpha', type=float, default=0.05,
                        help='[fdr] BH-FDR significance threshold')
    parser.add_argument('--adaptive', '-adaptive', action='store_true',
                        help='[fdr] Use adaptive FDR with Storey pi0 estimation')

    # hFDR-specific
    parser.add_argument('--alpha_block', '-alpha_block', type=float, default=0.05,
                        help='[hfdr] Stage 1 BH-FDR threshold across spatial blocks')
    parser.add_argument('--alpha_cell', '-alpha_cell', type=float, nargs='+',
                        default=[0.05, 0.15, 0.25],
                        help=(
                            '[hfdr] Stage 2 BH-FDR threshold(s) within rejected blocks. '
                            'Accepts multiple values to sweep Stage 2 without re-fitting. '
                            'Recommend values between 0.05 and 0.25. '
                        ))
    parser.add_argument('--combination_method', '-combination', type=str,
                        default='simes', choices=['simes', 'stouffer'],
                        help='[hfdr] Stage 1 p-value combination method within blocks')


def create_dataloader(args, bundles=None) -> BundleProfileLoader:
    """Create BundleProfileLoader from parsed arguments."""

    logging.info(f"Loading metadata from {args.metadata}")
    metadata_df = pd.read_csv(args.metadata, dtype={args.subj_col: str})

    if bundles is None:
        bundles = load_list(args.bundle_list)
        logging.info(f"Loaded {len(bundles)} bundles from {args.bundle_list}")

    edge_mask = None
    if args.edge_mask is not None:
        edge_mask = args.edge_mask[0] if len(args.edge_mask) == 1 else tuple(args.edge_mask)
        
    return BundleProfileLoader(
        metadata_df=metadata_df,
        bundles=bundles,
        metric=args.metric,
        root_dir=args.root_dir,
        mode=args.mode,
        file_ext=args.file_ext,
        min_prop=args.min_prop,
        subj_col=args.subj_col,
        target_col=args.target_col,
        target_ref_group=None if args.target_ref is None \
                            else (int(args.target_ref) if args.target_ref.isdecimal() \
                            else args.target_ref),
        target_case_group=None if args.target_case is None \
                            else (int(args.target_case) if args.target_case.isdecimal() \
                            else args.target_case),
        group_col=args.group_col,
        covar_cols=args.covar_cols or [],
        cont_cols=args.cont_cols or [],
        require_common=args.require_common,
        disable_tqdm=args.quiet,
        edge_mask=edge_mask,
    )


def create_lmm(args, loader: BundleProfileLoader) -> MultiBundleLM:
    """Create MultiBundleLM from parsed arguments and dataloader."""
    return MultiBundleLM(
        dataloader=loader,
        s_len=args.s_len,
        r_len=args.r_len,
        group_col=args.group_col,
        min_subj=args.min_subj,
        disable_tqdm=args.quiet,
        output_dir=args.output_dir,
    )


def save_results(
    result: Union[MultiBundleResult, MultiBundleHFDRResult],
    args,
    alpha_cell: Optional[float] = None,
    save_blocks: bool = True,
) -> None:
    """
    Save all output files for a given result object.

    For FDR results: writes stats_all, stats_summary, and (if adaptive)
    fdr_diagnostics. alpha_cell and save_blocks are ignored.

    For hFDR results: writes stats_all and stats_summary tagged with the
    alpha_cell suffix. If save_blocks=True, also writes stats_blocks (this
    should be True only for the first alpha_cell since blocks are identical
    across all alpha_cell sweeps).

    Parameters
    ----------
    result : MultiBundleResult or MultiBundleHFDRResult
    args : argparse.Namespace
    alpha_cell : float, optional
        Current alpha_cell level (hFDR only). Used to construct file suffixes.
    save_blocks : bool
        If True, write stats_blocks.csv (hFDR only, written once per run).
    """

    out = Path(args.output_dir)
    mode = args.stats_mode

    # FDR
    if isinstance(result, MultiBundleResult):
        adaptive_sfx = "_adaptive" if args.adaptive else ""
        stem = f"{mode}{adaptive_sfx}"

        result.get_stats_all(str(out / f"{stem}__stats_all.csv"))
        result.get_stats_summary(str(out / f"{stem}__stats_summary.csv"))
        if args.adaptive:
            result.save_fdr_diagnostics(str(out / f"{stem}__fdr_diagnostics.npz"))
        logging.info(f"Saved FDR results: {out / stem}__stats_*.csv")

    # HFDR
    elif isinstance(result, MultiBundleHFDRResult):
        ac_sfx = f"_ac{str(int(alpha_cell*100)).zfill(2)}" if alpha_cell is not None else ""
        stem = mode

        if save_blocks:
            result.get_stats_blocks(str(out / f"{stem}__stats_blocks.csv"))
            logging.info(f"Saved Stage 1 blocks: {out / f'{stem}__stats_blocks.csv'}")

        result.get_stats_all(str(out / f"{stem}__stats_all{ac_sfx}.csv"))
        result.get_stats_summary(str(out / f"{stem}__stats_summary{ac_sfx}.csv"))
        logging.info(
            f"Saved hFDR results (alpha_cell={alpha_cell}): "
            f"{out / f'{stem}__stats_*{ac_sfx}.csv'}"
        )


# ---------------------------------------------------------------------------
# Subcommand helpers
# ---------------------------------------------------------------------------

def run_single(args) -> None:
    """Fit observed model for a single bundle and save to output_dir/observed/."""
    make_directory(args.output_dir)

    fit_reduced = args.stats_mode in ('hfdr_resel', 'hfdr_bundle')
    if fit_reduced:
        logging.info(
            f"stats_mode={args.stats_mode}: fitting full + reduced model for '{args.bundle}' "
        )
    else:
        logging.info(
            f"stats_mode=fdr: fitting full model for '{args.bundle}'"
        )

    loader = create_dataloader(args, bundles=[args.bundle])
    lmm = create_lmm(args, loader)

    grid = lmm._fit_observed_single_grid(args.bundle, fit_reduced=fit_reduced, save=True)
    if grid is None:
        logging.warning(f"Skipping {args.bundle}: no valid cells or insufficient subjects")
    else:
        logging.info(f"Saved observed results: {args.output_dir}/observed/{args.bundle}.npz")

    logging.info(
        f"Saved observed results: {args.output_dir}/observed/{args.bundle}.npz"
    )


def run_aggregate(args) -> None:
    """Load saved per-bundle results and apply FDR or hFDR correction."""

    if args.bundle_list is None:
        raise ValueError("--bundle_list is required for aggregate mode")

    loader = create_dataloader(args)
    lmm = create_lmm(args, loader)

    # adaptive mode only applies for FDR
    if args.stats_mode != 'fdr':
        if args.adaptive:
            logging.warning(
                f"--adaptive has no effect with --stats_mode {args.stats_mode}"
            )

    # FDR path
    if args.stats_mode == 'fdr':
        logging.info(
            f"Aggregating FDR (alpha={args.alpha}, adaptive={args.adaptive})"
        )
        result = lmm.fit_observed_fdr(
            weight_by=None,
            alpha=args.alpha,
            adaptive=args.adaptive,
            from_saved=True,
        )
        save_results(result, args)
        return

    # hFDR paths 
    # validate l_s before proceeding
    saved_grids = lmm.load_observed_results()
    missing_ls = [b for b, g in saved_grids.items() if g.l_s is None]
    if missing_ls:
        raise RuntimeError(
            f"{len(missing_ls)} bundle(s) are missing Matern length scales (l_s=None). "
            f"This means they were fitted with --stats_mode fdr (no reduced model). "
            f"Re-run 'single' with --stats_mode {args.stats_mode} for these bundles:\n"
            + "\n".join(f"  {b}" for b in missing_ls)
        )

    use_resel_blocks = (args.stats_mode == 'hfdr_resel')
    alpha_cell_values: List[float] = args.alpha_cell

    logging.info(
        f"Aggregating hFDR (mode={args.stats_mode}, alpha_block={args.alpha_block}, "
        f"alpha_cell={alpha_cell_values}, combination={args.combination_method})"
    )

    # Fit Stage 1 and compute all conditional p-values using the first alpha_cell.
    # Stage 2 will be re-applied for all values in the sweep loop below.
    result = lmm.fit_hierarchical_fdr(
        alpha_block=args.alpha_block,
        alpha_cell=alpha_cell_values[0],
        use_resel_blocks=use_resel_blocks,
        combination_method=args.combination_method,
        from_saved=True,
    )

    # Save Stage 1 blocks once — identical across all alpha_cell values
    save_results(result, args, alpha_cell=alpha_cell_values[0], save_blocks=True)

    # Sweep remaining alpha_cell values via reapply_stage2 (no LMM re-fitting)
    for ac in alpha_cell_values[1:]:
        logging.info(f"Reapplying Stage 2 at alpha_cell={ac}")
        result.reapply_stage2(ac)
        save_results(result, args, alpha_cell=ac, save_blocks=False)

    logging.info(
        f"Saved all hFDR results to {args.output_dir} "
        f"at {len(alpha_cell_values)} alpha_cell level(s)"
    )


def main():
    setup_logging("INFO")

    # Parent parser — shared by both subcommands
    parent_parser = argparse.ArgumentParser(add_help=False)
    add_dataloader_args(parent_parser)
    add_lmm_args(parent_parser)

    # Top-level parser
    parser = argparse.ArgumentParser(
        description='Run FDR- or hFDR-corrected multi-bundle LMM analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    # single subcommand
    single_parser = subparsers.add_parser(
        'single',
        parents=[parent_parser],
        help='Fit observed model for a single bundle (run as HPC job array)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    single_parser.add_argument('--bundle', '-bundle', type=str, required=True,
                               help='Bundle name to process')

    # aggregate subcommand
    agg_parser = subparsers.add_parser(
        'aggregate',
        parents=[parent_parser],
        help='Load saved per-bundle results and apply FDR or hFDR correction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    add_aggregate_args(agg_parser)

    args = parser.parse_args()

    if args.command == 'single':
        run_single(args)
    elif args.command == 'aggregate':
        run_aggregate(args)