#!/usr/bin/env python
"""
DESCRIPTION
    Export per-bundle statistic grids from a stats_all CSV written by
    `spectra_lmm_fdr aggregate`. Each output file holds one (ns, nr) grid
    (nr = 1 for 1D profiles), with NaN for cells that were not tested.
    The files can be passed directly to `spectra_viz -c`.

    Output files are named {out_prefix}{bundle}__{stat}.{npy|txt}

USAGE
    spectra_export_stats \\
        -i results/hfdr_resel__stats_all_ac15.csv \\
        -o results/grids/ \\
        -export_stats d_cohen p_hfdr \\
        -use_log_p -sig_only
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd

from spectra.tractometry.results_data import get_sig_map, get_stat_map
from spectra.utils.io import load_list
from spectra.utils.utils import setup_logging


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--in_stats_csv', '-i', type=str, required=True,
                        help='stats_all CSV from spectra_lmm_fdr aggregate (FDR or hFDR).')
    parser.add_argument('--out_prefix', '-o', type=str, required=True,
                        help='Output prefix. End it with "/" to write into a folder.')
    parser.add_argument('--export_stats', '-export_stats', nargs='+', type=str,
                        default=['d_cohen'],
                        help='Columns to export, e.g. d_cohen t_obs p_hfdr (default: d_cohen).')
    parser.add_argument('--use_log_p', '-use_log_p', action='store_true',
                        help='Write -log10(p) for columns whose name starts with "p_".')
    parser.add_argument('--sig_only', '-sig_only', action='store_true',
                        help='Set non-significant cells to NaN (sig_hfdr, or sig_fdr for FDR results).')
    parser.add_argument('--fmt', '-fmt', type=str, default='npy', choices=['npy', 'txt'],
                        help='Output format (default: npy).')
    parser.add_argument('--bundle_list', '-bundle_list', type=str, default=None,
                        help='Text file with one bundle per line (default: all bundles in the CSV).')
    args = parser.parse_args()

    setup_logging()

    df = pd.read_csv(args.in_stats_csv)
    is_hfdr = 'sig_hfdr' in df.columns

    if args.bundle_list:
        bundles = load_list(args.bundle_list, is_str=True)
    else:
        bundles = list(df['bundle'].unique())

    os.makedirs(os.path.dirname(args.out_prefix) or '.', exist_ok=True)

    for bundle in bundles:
        if bundle not in df['bundle'].values:
            logging.warning(f"Bundle '{bundle}' not in {args.in_stats_csv}; skipped.")
            continue

        sig_mask = None
        if args.sig_only:
            # NaN (untested) cells count as not significant
            sig_mask = get_sig_map(df, bundle, is_hfdr=is_hfdr)['sig_grid_cell'] == 1

        for stat in args.export_stats:
            grid = get_stat_map(df, bundle, stat_col=stat, sig_mask=sig_mask)
            if args.use_log_p and stat.startswith('p_'):
                grid = -np.log10(grid)

            out_fname = f"{args.out_prefix}{bundle}__{stat}.{args.fmt}"
            if args.fmt == 'npy':
                np.save(out_fname, grid)
            else:
                np.savetxt(out_fname, grid)

    logging.info(f"Exported {len(args.export_stats)} statistic(s) for {len(bundles)} bundle(s) "
                 f"to {args.out_prefix}")


if __name__ == '__main__':
    main()
