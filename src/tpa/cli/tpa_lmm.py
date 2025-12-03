#!/ifs/loni/faculty/thompson/four_d/wfeng/miniconda3/envs/pymer4_v2/bin/python
"""
DESCRIPTION:
    Run linear mixed model from profile
USAGE:
    TODO
    
"""

import os
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from tpa.tractometry.lmm import aggregate_profiles, run_lmm_pymer4
from tpa.utils.utils import setup_logging, file_exists
from tpa.utils.io import load_list, save_csv

def run(args):
    # Aggregate profile
    subj_list = load_list(args.subj_list_fpath) if args.subj_list_fpath is not None else None
    group_avg, mask, df_profile = aggregate_profiles(args.metric, 
                                                     args.bundle,
                                                     args.root_dir,
                                                     subj_list=subj_list, 
                                                     mode=args.mode, 
                                                     min_prop=args.min_prop,
                                                     disable_tqdm=(not args.tqdm))
    # Read metadata and merge with subject profiles
    df_meta = pd.read_csv(args.metadata_fpath)
    if args.subject_col == 'subject':
        df = pd.merge(df_meta, df_profile, on='subject')
    else:
        df = pd.merge(df_meta, df_profile, left_on=args.subject_col, 
                      right_on='subject').drop(columns=[args.subject_col])
    
    # Run linear mixed model
    df_lmm, df_resid = run_lmm_pymer4(df, 
            target_col=args.target_col, 
            target_ref_group=args.target_ref_group,
            site_col=args.random_col,
            covar_cols=args.fixed_cols,
            cont_cols=args.cont_cols,
            disable_tqdm=(not args.tqdm))

    # Save outputs (group average, mask, df_lmm and df_resid)
    ns, nr = group_avg.shape
    np.savez(
        f"{args.output_prefix}_group_avg.npz",
        ns=ns,
        nr=nr,
        group_avg=group_avg.astype(np.float32),
        mask=mask.astype(np.int32),
    )
    save_csv(f"{args.output_prefix}_lmm.csv", df_lmm, force=args.force)
    save_csv(f"{args.output_prefix}_resid.csv", df_resid, force=args.force)


def main():
    setup_logging()
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_dir', '-i', type=str, required=True,
                    help='Root directory of bundle profiles, containing [metric]/[bundle]/[subj].txt')
    parser.add_argument('--metric', '-m', type=str, required=True,
                        help='Diffusion metric name')
    parser.add_argument('--bundle', '-b', type=str, required=True,
                        help='Bundle name')
    parser.add_argument('--metadata_fpath', '-metadata', type=str, required=True,
                        help='Path to metadata CSV file.')
    parser.add_argument('--subj_list_fpath', '-subj_list', type=str, default=None,
                        help='Path to subject IDs to include (default: None, use all available subjects).')
    parser.add_argument('--output_prefix', '-o', type=str, required=True,
                        help='Prefix for output files.')
    parser.add_argument('--mode', '-mode', type=str, default='2d', choices=['1d', '2d'],
                        help='Processing mode: 1d or 2d (default: 2d).')
    parser.add_argument('--min_prop', '-prop', type=float, default=0.3,
                        help='Minimum proportion of subjects each grid cell must have (default: 0.3)')
    
    # Stat args
    parser.add_argument('--subject_col', '-subj', type=str, required=True,
                        help='Subject column name in the metadata')
    parser.add_argument('--target_col', '-target', type=str, required=True,
                        help='Target column of interest in the metadata')
    parser.add_argument('--target_ref_group', '-target_ref', type=str, default=None,
                        help='Reference group for the target variable if --target_col is categorical,\
                            set to None if continuous')
    parser.add_argument('--random_col', '-random', type=str, default=None,
                        help='Column name for the random effect variable (e.g. site) if any')
    parser.add_argument('--fixed_cols', '-fixed', type=str, nargs='*', default=[],
                        help='List of column name(s) of the fixed effects variable(s)')
    parser.add_argument('--cont_cols', '-cont', type=str, nargs='*', default=[],
                        help='List of column names(s) of the continuous variable(s)')

    # General args
    parser.add_argument( "--force", "-f", action="store_true", 
                        help="Force saving files", )
    parser.add_argument( "--tqdm", "-tqdm", action="store_true", 
                        help="Show progress bar", )
    args = parser.parse_args()
    run(args)