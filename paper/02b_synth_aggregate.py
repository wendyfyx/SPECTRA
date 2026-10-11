"""
Step 2b: aggregate the synthetic-experiment results across all runs.

Run locally (not a qsub script), after all 02a_synth_eval.sh tasks finish.
Run once per alpha_cell.

Reads the output of 02a_synth_eval.py and writes three CSV files into the
same folder, one per output type:

  synth_eval_fdr.csv               -- SYNTH_EVAL__FDR__*.csv   (cell-level, FDR variants)
  synth_eval_hfdr_ac{AC}.csv       -- SYNTH_EVAL__HFDR__*.csv  (cell-level, hFDR variants)
  synth_eval_hfdr_block_ac{AC}.csv -- SYNTH_EVAL_BLOCK__HFDR__*.csv (block-level, hFDR Stage 1)

Only the GLOBAL summary row is retained from each file.

USAGE:
    python 02b_synth_aggregate.py --base_dir /path/to/synthetic_eval/fa_2d -ac 0.05
    python 02b_synth_aggregate.py --base_dir /path/to/synthetic_eval/fa_2d -ac 0.15
"""

import argparse
import glob
import re
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np


EFFECT_SIZES       = [0.2, 0.5, 0.8]
SAMPLE_SIZES       = [50, 100, 200, 400, 600, 800, 1000, 1200]
N_AFFECTED_BUNDLES = [5, 10, 15]
LOCALITIES         = ['diffuse', 'localized']


def _parse_filename(fname: str) -> dict:
    """
    Extract metadata encoded in the filename stem.

    Parses (all optional / None when absent):
      run_id, ac, is_weighted, is_adaptive, use_resel_block, method
    """
    meta = {}

    m = re.search(r'_run(\d+)', fname)
    meta['run_id'] = int(m.group(1)) if m else None

    m = re.search(r'_ac([\d.]+)', fname)
    meta['ac'] = float(m.group(1)) if m else None

    meta['is_weighted'] = 'weighted' in fname
    meta['is_adaptive'] = 'adaptive' in fname

    if 'resel' in fname:
        meta['use_resel_block'] = 'resel'
    elif 'bundle' in fname:
        meta['use_resel_block'] = 'bundle'
    else:
        meta['use_resel_block'] = None

    # Determine method from filename prefix
    if 'SYNTH_EVAL_BLOCK__HFDR__' in fname:
        meta['method'] = 'hfdr_block'
    elif 'SYNTH_EVAL__HFDR__' in fname:
        meta['method'] = 'hfdr'
    elif 'SYNTH_EVAL__FDR__' in fname:
        meta['method'] = 'fdr'
    else:
        meta['method'] = None

    return meta


def load_global_row(fname: str) -> Optional[pd.DataFrame]:
    """
    Load a single SYNTH_EVAL CSV and return only the GLOBAL row,
    with metadata columns attached.  Returns None if file is empty
    or has no GLOBAL row.
    """
    try:
        df = pd.read_csv(fname)
    except Exception as e:
        print(f"  WARNING: could not read {fname}: {e}")
        return None

    df = df.loc[df['bundle'] == 'GLOBAL']
    if df.empty:
        print(f"  WARNING: no GLOBAL row in {fname}")
        return None

    meta = _parse_filename(fname)
    for col, val in meta.items():
        df[col] = val

    return df


def _glob_and_load(pattern: str, label: str) -> pd.DataFrame:
    """Glob files matching pattern, load GLOBAL rows, concatenate."""
    files = sorted(glob.glob(pattern))
    if not files:
        return pd.DataFrame()
    rows = [load_global_row(f) for f in files]
    rows = [r for r in rows if r is not None]
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    print(f"  {label}: {len(files)} files → {len(df)} rows")
    return df


def load_condition(
    base_dir: str,
    effect_size: float,
    sample_size: int,
    n_affected: int,
    locality: str,
    alpha_cell: float = 0.25
) -> dict:
    """
    Load all three output types for one (effect, sample, bundles, locality) cell.

    Returns dict with keys 'fdr', 'hfdr', 'hfdr_block', each a DataFrame
    (possibly empty).
    """
    cond_dir = (
        f"{base_dir}/effect_{effect_size}"
        f"/sample_{sample_size}"
        f"/bundles_{n_affected}"
        f"/{locality}"
    )
    label = f"d={effect_size} n={sample_size} b={n_affected} loc={locality}"

    fdr_df        = _glob_and_load(f"{cond_dir}/SYNTH_EVAL__FDR__*.csv",                        f"{label} [FDR]")
    hfdr_df       = _glob_and_load(f"{cond_dir}/SYNTH_EVAL__HFDR__*_ac{alpha_cell}*.csv",       f"{label} [hFDR]")
    hfdr_block_df = _glob_and_load(f"{cond_dir}/SYNTH_EVAL_BLOCK__HFDR__*_ac{alpha_cell}*.csv", f"{label} [hFDR-block]")

    # Attach grid coordinates to every loaded frame
    for df in (fdr_df, hfdr_df, hfdr_block_df):
        if not df.empty:
            df['effect_size']       = effect_size
            df['sample_size']       = sample_size
            df['n_bundles_affected'] = n_affected
            df['locality']          = locality

    return {'fdr': fdr_df, 'hfdr': hfdr_df, 'hfdr_block': hfdr_block_df}


def load_all(
    base_dir: str,
    effect_sizes=EFFECT_SIZES,
    sample_sizes=SAMPLE_SIZES,
    n_affected_list=N_AFFECTED_BUNDLES,
    localities=LOCALITIES,
    alpha_cell=0.25
) -> dict:
    """
    Iterate over the full parameter grid and concatenate results.

    Returns dict with keys 'fdr', 'hfdr', 'hfdr_block'.
    """
    buckets = {'fdr': [], 'hfdr': [], 'hfdr_block': []}

    for d in effect_sizes:
        for n in sample_sizes:
            for b in n_affected_list:
                for loc in localities:
                    cond = load_condition(base_dir, d, n, b, loc, alpha_cell=alpha_cell)
                    for key in buckets:
                        if not cond[key].empty:
                            buckets[key].append(cond[key])

    return {
        key: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        for key, frames in buckets.items()
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate 02a_synth_eval.py results across all runs"
    )
    parser.add_argument('--base_dir', type=str, required=True,
        help="Output folder of 02a_synth_eval.sh for one metric and mode "
             "(dir_out_base, e.g. /path/to/synthetic_eval/fa_2d)")
    parser.add_argument('--effect_sizes', nargs='+', type=float,
        default=EFFECT_SIZES, metavar='D')
    parser.add_argument('--sample_sizes', nargs='+', type=int,
        default=SAMPLE_SIZES, metavar='N')
    parser.add_argument('--n_affected', nargs='+', type=int,
        default=N_AFFECTED_BUNDLES, metavar='B')
    parser.add_argument('--localities', nargs='+', type=str,
        default=LOCALITIES, metavar='LOC')
    parser.add_argument('--alpha_cell', '-ac', type=float, default=0.25,
        help="Alpha cell level used in hFDR runs (default: 0.25)")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    base_dir = args.base_dir
    out_path = Path(base_dir)

    print(f"Base directory : {base_dir}")
    print(f"Effect sizes   : {args.effect_sizes}")
    print(f"Sample sizes   : {args.sample_sizes}")
    print(f"N affected     : {args.n_affected}")
    print(f"Localities     : {args.localities}")
    print(f"Alpha cell     : {args.alpha_cell}")
    print()

    results = load_all(
        base_dir=base_dir,
        effect_sizes=args.effect_sizes,
        sample_sizes=args.sample_sizes,
        n_affected_list=args.n_affected,
        localities=args.localities,
        alpha_cell=args.alpha_cell
    )

    outputs = {
        f'synth_eval_fdr.csv':                              results['fdr'],
        f'synth_eval_hfdr_ac{args.alpha_cell}.csv':         results['hfdr'],
        f'synth_eval_hfdr_block_ac{args.alpha_cell}.csv':   results['hfdr_block'],
    }

    for fname, df in outputs.items():
        fpath = out_path / fname
        if df.empty:
            print(f"No data for {fname}, skipping")
        else:
            df.to_csv(fpath, index=False)
            print(f"Wrote {len(df):,} rows → {fpath}")