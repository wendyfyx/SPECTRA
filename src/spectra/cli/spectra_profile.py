"""
DESCRIPTION:
    Create microstructure profile using 1D (BUAN weighted mean) and 2D (spectra)
    Both 1D and 2D profiles are saved in the same .npz file
USAGE:
    spectra_profile -rec sub01_AF_L_mni.trk \\
                -org sub01_AF_L.trk -ref atlas_AF_L.trk \\
                -maps sub01_fa.nii.gz sub01_md.nii.gz \\
                -profiles sub01_AF_L_fa sub01_AF_L_md \\
                -s 5 \\
                -r 5 \\
                -robust linear
"""

import argparse
import logging
from pathlib import Path

import numpy as np
from dipy.io.image import load_nifti

from spectra.utils.utils import setup_logging
from spectra.utils.streamline import read_streamlines
from spectra.tractometry.parameterization import get_grid_from_atlas, parameterize_bundle, \
                                             prepare_bundle_pairs, buan_profile, grid_profile

def _use_robust_centroid(robust_method):
    if robust_method in ('spline', 'linear'):
        return True, robust_method
    logging.info(
        f"robust_method='{robust_method}' is not 'spline' or 'linear'; "
        "falling back to standard QuickBundles centroid."
    )
    return False, robust_method

def run(args):
    assert len(args.scalar_map_fnames) == len(args.profile_fnames), \
    "The number of scalar map paths must match the number of output paths"

    # Load bundles
    rec_bundle, _, _, _ = read_streamlines(args.rec_bundle_fname)   # subject bundle in the atlas space
    if len(rec_bundle) < args.min_lines:
        logging.warning(f"Not enough streamlines! {len(rec_bundle)} lines loaded, {args.min_lines} required")
        return
    org_bundle, _, _, _ = read_streamlines(args.org_bundle_fname)   # subject bundle in the subject space
    ref_bundle, _, _, _ = read_streamlines(args.ref_bundle_fname)   # atlas bundle

    # Ensure both subject space and MNI space bundles have the same number of points
    org_bundle, rec_bundle = prepare_bundle_pairs(org_bundle, rec_bundle, use_even_points=True)

    # Create grid on atlas bundle
    use_robust, robust_method = _use_robust_centroid(args.robust_method)
    _, _, centroid, u_vecs, r_edges, _, _ = get_grid_from_atlas(
        ref_bundle,
        s_len=args.s_len,
        r_len=args.r_len,
        use_robust_centroid=use_robust,
        robust_method=robust_method,
    )
    ns = len(centroid)
    nr = len(r_edges) - 1
    
    # Parameterize input bundle
    s_index, r_index, s_dist, valid_mask, counts = parameterize_bundle(rec_bundle, 
                                                                      centroid, 
                                                                      u_vecs, 
                                                                      r_edges)
    seg_counts = counts.sum(axis=1)

    # Create profiles
    for i, path in enumerate(args.scalar_map_fnames):

        scalar_map, scalar_affine = load_nifti(path)
        out_path = Path(args.profile_fnames[i]).with_suffix(".npz")

        # 1D profile (BUAN weighted mean)
        profile_1d = buan_profile(org_bundle, s_index, s_dist, ns, scalar_map, scalar_affine, seg_counts=seg_counts)
        # 2D profile (grid)
        profile_2d = grid_profile(org_bundle, s_index, r_index, valid_mask, ns, nr, scalar_map, scalar_affine)

        # Save both profiles to .npz
        np.savez(
            out_path,
            ns=ns,
            nr=nr,
            s_len=args.s_len,
            r_len=args.r_len,
            profile_1d=profile_1d.astype(np.float32),
            profile_2d=profile_2d.astype(np.float32),
        )
        logging.info(f"Created profile to <{out_path}>")

def main():
    setup_logging()
    parser = argparse.ArgumentParser()

    parser.add_argument('--rec_bundle_fname', '-rec', type=str, required=True, 
                        help="Path to bundle in the MNI space")
    parser.add_argument('--org_bundle_fname', '-org', type=str, required=True, 
                        help="Path to bundle in the native space")
    parser.add_argument('--ref_bundle_fname', '-ref', type=str, required=True, 
                        help="Path to reference or atlas bundle")
    parser.add_argument('--scalar_map_fnames', '-maps', nargs="+", type=str, required=True, 
                        help="Path(s) to scalar maps to project onto tracts")
    parser.add_argument('--profile_fnames', '-profiles', nargs="+", type=str, required=True, 
                        help="Path(s) to output profile txt files corresponding to each scalar map")
    parser.add_argument('--s_len', '-s', type=int, required=False, default=5, 
                        help="Length of along-tract segments (mm)")
    parser.add_argument('--r_len', '-r', type=int, required=False, default=5, 
                        help="Length of radial bins (mm)")
    parser.add_argument('--min_lines', '-min', type=int, required=False, default=20, 
                        help="Minimum number of lines the bundle must have to output a profile")
    parser.add_argument('--robust_method', '-robust', type=str, default='linear',
                        choices=['spline', 'linear', 'none'],
                        help='Extrapolation method for robust centroid')
    parser.add_argument("--force", "-f", action="store_true", 
                        help="Force saving files")
    args = parser.parse_args()
    run(args)