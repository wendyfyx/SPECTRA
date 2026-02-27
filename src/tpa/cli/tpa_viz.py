"""
DESCRIPTION:
    Bundle Visualization with 1D/2D profile color support
"""
import sys
import logging
import pickle
import argparse
import warnings
from importlib.resources import files

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from nibabel.streamlines.array_sequence import ArraySequence

from dipy.io.image import load_nifti
from dipy.io.streamline import load_trk
from dipy.io.surface import load_gifti
from dipy.stats.analysis import assignment_map
from dipy.tracking.streamline import transform_streamlines
from fury import actor, window
from fury.utils import map_coordinates_3d_4d

from tpa.tractometry.parameterization import (
    get_grid_from_atlas, get_centroid, parameterize_bundle
)
from tpa.viz.viz_fury import get_cam_settings_fury
from tpa.viz.colors import color_from_values, alternating_colors
from tpa.utils.io import load_list
from tpa.utils.utils import setup_logging, get_wm_mni_path

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def autocrop_png(fname, border=20):
    img = plt.imread(fname)
    if img.shape[-1] == 4:
        rgb = img[:, :, :3]
    else:
        rgb = img
    mask = np.any(rgb < 1.0, axis=-1)
    if not np.any(mask):
        print("Image is all white, skipping crop.")
        return
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    y0 = max(y0 - border, 0)
    x0 = max(x0 - border, 0)
    y1 = min(y1 + border, img.shape[0] - 1)
    x1 = min(x1 + border, img.shape[1] - 1)
    plt.imsave(fname, img[y0:y1 + 1, x0:x1 + 1])


def get_default_glass():
    from scipy.ndimage import gaussian_filter
    img, affine = load_nifti(get_wm_mni_path())
    img[img < 0.25] = 0
    img = gaussian_filter(img, 0.3)
    return img, affine


def get_default_pial():
    from nilearn.datasets import load_fsaverage
    fsaverage_meshes = load_fsaverage()
    pial_l = fsaverage_meshes['pial'].parts['left']
    pial_r = fsaverage_meshes['pial'].parts['right']
    faces_r_shifted = pial_r.faces + len(pial_l.coordinates)
    vertices = np.vstack((pial_l.coordinates, pial_r.coordinates))
    faces = np.vstack((pial_l.faces, faces_r_shifted))
    return vertices, faces


# ---------------------------------------------------------------------------
# Parameterization
# ---------------------------------------------------------------------------

def _use_robust_centroid(robust_method):
    if robust_method in ('spline', 'linear'):
        return True, robust_method
    logging.info(
        f"robust_method='{robust_method}' is not 'spline' or 'linear'; "
        "falling back to standard QuickBundles centroid."
    )
    return False, robust_method


def _compute_parameterization(bundle, args, ref_bundle=None):
    """
    Compute (s_index, r_index, valid_mask, centroid) for a bundle.

    Parameters
    ----------
    bundle : ArraySequence
        Subject bundle to parameterize.
    args : argparse.Namespace
    ref_bundle : ArraySequence or None
        If provided, build the grid from this reference/atlas bundle and
        parameterize ``bundle`` onto it (subject-bundle path).
        If None, self-parameterize ``bundle`` (atlas/stats path).

    Returns
    -------
    s_index : np.ndarray or None
    r_index : np.ndarray or None
    valid_mask : np.ndarray or None
        Point-level boolean mask (only when ref_bundle is provided).
    centroid : np.ndarray, shape (ns, 3)
        Centroid used for parameterization; reused by centroid_only plotting
        so the geometry exactly matches the profiling step.
    """
    use_robust, robust_method = _use_robust_centroid(args.robust_method)
    source = ref_bundle if ref_bundle is not None else bundle

    s_index, r_index, centroid, u_vecs, r_edges = get_grid_from_atlas(
        source,
        s_len=args.s_len,
        r_len=args.r_len,
        use_robust_centroid=use_robust,
        robust_method=robust_method,
    )

    # Parameterize subject bundle onto the reference grid
    if ref_bundle is not None:
        s_index, r_index, _, valid_mask, _ = parameterize_bundle(
            bundle, centroid, u_vecs, r_edges
        )
        return s_index, r_index, valid_mask, centroid
    else:
        return s_index, r_index, None, centroid


# ---------------------------------------------------------------------------
# Color resolution
# ---------------------------------------------------------------------------

def _is_2d_color_mode(bundle_color_arg, values):
    """
    Return True when the coloring is 2D (grid-level or alternate_2d).

    Used to decide whether to forward valid_mask to the color functions.
    """
    if bundle_color_arg is None:
        return False
    spec = bundle_color_arg[0].lower()
    if 'alternate_2d' in spec:
        return True
    if values is not None and np.asarray(values).squeeze().ndim == 2:
        return True
    return False


def _resolve_bundle_colors(bundle, bundle_color_arg, s_index, r_index,
                           valid_mask, args, bundle_idx):
    """
    Resolve bundle color specification to a per-point color array.

    Parameters
    ----------
    bundle : ArraySequence
    bundle_color_arg : list of str or None
    s_index, r_index : np.ndarray or None
    valid_mask : np.ndarray or None
        Point-level mask (from parameterize_bundle) or grid-level mask
        (from --color_mask_fname).  Only forwarded for 2D color modes.
    args : argparse.Namespace
    bundle_idx : int
        Index of this bundle in the loop (for per-bundle color files).

    Returns
    -------
    colors : array-like or None
    """
    if bundle_color_arg is None:
        return None

    spec = bundle_color_arg[0]
    ends = spec.endswith
    values = None
    colors = None

    # --- Load values ---
    if ends('.npy') or ends('.txt'):
        # One color file per bundle
        values = load_list(bundle_color_arg[bundle_idx])

    elif ends('nii.gz'):
        # Single shared volume — log a warning if multiple were supplied
        if len(bundle_color_arg) > 1:
            logging.warning(
                "Volume coloring only supports one file; using bundle_color[0] "
                "for all bundles."
            )
        img_data, affine = load_nifti(bundle_color_arg[0])
        bundle_native = transform_streamlines(bundle, np.linalg.inv(affine))
        bundle_native = ArraySequence(bundle_native).get_data()
        values = map_coordinates_3d_4d(img_data, bundle_native).T

    elif 'alternate_2d' in spec.lower():
        # 2D checkerboard — apply valid_mask
        colors = alternating_colors(
            s_index=s_index,
            r_index=r_index,
            pattern=[['red', 'blue'], ['black', 'white']],
            valid_mask=valid_mask,
        )

    elif 'alternate_1d' in spec.lower():
        # 1D checkerboard — no valid_mask (not meaningful at segment level)
        colors = alternating_colors(
            s_index=s_index,
            r_index=None,
            pattern=['red', 'blue', 'orange'],
            valid_mask=None,
        )

    else:
        # Named matplotlib color — one per bundle
        colors = np.array(mpl.colors.to_rgba(bundle_color_arg[bundle_idx]))

    # Convert values to colors
    if values is not None:
        # Only pass valid_mask for 2D grid values
        mask_for_colors = valid_mask if _is_2d_color_mode(bundle_color_arg, values) else None
        colors = color_from_values(
            values,
            s_index=s_index,
            r_index=r_index,
            valid_mask=mask_for_colors,
            cmap=args.cmap,
            vmin=args.vmin,
            vmax=args.vmax,
            vcenter=getattr(args, 'vcenter', None),
            plot_cmap_label=args.cmap_title,
            plot_horizontal=True,
            show_cmap=False,
            save_cmap_path=args.out_cbar_fname,
        )

    return colors

def _resolve_centroid_colors(centroid, bundle_color_arg, s_index, r_index, args):
    """
    Resolve colors for the centroid streamline.

    The centroid has ns points; s_index and r_index are passed in from the
    already-computed parameterization so this function doesn't recompute them.

    Parameters
    ----------
    centroid : np.ndarray, shape (ns, 3)
    bundle_color_arg : list of str or None
    s_index : np.ndarray
        s-coordinates from the parameterization step (used for value-mapped modes).
    r_index : np.ndarray or None
        r-coordinates from the parameterization step (used for value-mapped modes).
    args : argparse.Namespace

    Returns
    -------
    colors : array-like or None
    """
    ns = len(centroid)
    c_s = np.arange(ns)
    c_r = np.zeros(ns, dtype=int)

    if bundle_color_arg is not None:
        spec = bundle_color_arg[0].lower()
        if 'alternate_2d' in spec:
            logging.warning(
                "alternate_2d requested in centroid mode — r is always 0, "
                "coloring will behave like alternate_1d."
            )
        # Reuse existing resolution logic with centroid indices
        return _resolve_bundle_colors(
            ArraySequence([centroid]),
            bundle_color_arg,
            s_index=c_s,
            r_index=c_r,
            valid_mask=None,
            args=args,
            bundle_idx=0,
        )
    return None

# ---------------------------------------------------------------------------
# Actor helpers
# ---------------------------------------------------------------------------

def _build_bundle_actor(bundle_plot, colors, args, linewidth=None, opacity=None):
    """Build and return a fury actor for the bundle."""
    lw = linewidth if linewidth is not None else args.linewidth
    op = opacity if opacity is not None else args.opacity
    if args.as_points:
        return actor.point(
            bundle_plot.get_data(),
            colors=window.colors.green,
            point_radius=0.2,
        )
    else:
        return actor.line(
            bundle_plot,
            fake_tube=args.fake_tube,
            linewidth=lw,
            colors=colors,
            opacity=op,
        )

def _add_glass_brain(scene, surf_path):
    """Add a glass-brain surface (NIFTI or GIFTI) to the scene."""
    ends = surf_path.endswith

    if 'default_glass' in surf_path or ends('.nii.gz') or ends('.nii'):
        if 'default_glass' in surf_path:
            mask, affine = get_default_glass()
            logging.info("Using default glass brain...")
        else:
            mask, affine = load_nifti(surf_path)
        scene.add(
            actor.contour_from_roi(
                mask,
                affine=affine,
                color=np.array([0, 0, 0]),
                opacity=0.04,
            )
        )

    if 'default_pial' in surf_path or ends('.gii.gz') or ends('.gii'):
        if 'default_pial' in surf_path:
            vertices, faces = get_default_pial()
            logging.info('Using default pial surface...')
        else:
            surface = load_gifti(surf_path)
            vertices, faces = surface
            if not (len(vertices) and len(faces)):
                warnings.warn(
                    f"{surf_path} does not have any surface geometry."
                )
                return
        surf_colors = np.zeros((vertices.shape[0], 3))
        surf_actor = actor.surface(vertices, faces=faces, colors=surf_colors)
        surf_actor.GetProperty().SetOpacity(0.05)
        scene.add(surf_actor)

# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def run(args):

    if args.centroid_only and args.centroid_with_bundle:
        raise ValueError(
            "--centroid_only and --centroid_with_bundle are mutually exclusive."
        )

    scene = window.Scene()
    scene.SetBackground(1, 1, 1)

    # Camera setup
    if args.in_camera_fname is not None:
        with open(args.in_camera_fname, 'rb') as f:
            cam_settings = pickle.load(f)
        scene.set_camera(
            position=cam_settings['pos'],
            focal_point=cam_settings['foc'],
            view_up=cam_settings['vup'],
        )
    else:
        cam = get_cam_settings_fury(args.cam_view)
        logging.info(f'Camera is set to {args.cam_view}.')
        scene.set_camera(**cam)

    # Load reference bundle once (shared across all input bundles)
    ref_bundle = None
    if args.ref_bundle_fname is not None:
        ref_bundle, _ = load_trk(
            args.ref_bundle_fname, 'same', bbox_valid_check=False
        ).streamlines, None
        ref_bundle = load_trk(
            args.ref_bundle_fname, 'same', bbox_valid_check=False
        ).streamlines
        logging.info(
            f'Loaded reference bundle ({len(ref_bundle)} streamlines) '
            f'from {args.ref_bundle_fname}.'
        )

    # Load grid-level color mask if provided (stats/atlas use case)
    color_mask = None
    if args.color_mask_fname is not None:
        color_mask = np.load(args.color_mask_fname)
        logging.info(
            f'Loaded color mask shape={color_mask.shape} '
            f'from {args.color_mask_fname}.'
        )

    # Plot bundles
    for i, bundle_path in enumerate(args.in_bundle_fnames):
        bundle = load_trk(
            bundle_path, 'same', bbox_valid_check=False
        ).streamlines
        logging.info(
            f'Loaded {len(bundle)} streamlines from {bundle_path}.'
        )

        # --- Parameterization ---
        s_index, r_index, valid_mask, centroid = None, None, None, None

        if args.centroid_only or args.centroid_with_bundle or args.bundle_color is not None:
            s_index, r_index, valid_mask, centroid = _compute_parameterization(
                bundle, args, ref_bundle=ref_bundle
            )
            # Grid-level mask overrides point-level mask when provided
            if color_mask is not None:
                valid_mask = color_mask

        # --- Add actor(s) ---
        if args.centroid_only:
            # Colored centroid only — geometry matches the profiling step exactly
            centroid_colors = _resolve_centroid_colors(
                centroid, args.bundle_color, s_index, r_index, args
            )
            scene.add(_build_bundle_actor(
                ArraySequence([centroid]), centroid_colors, args,
                linewidth=35, opacity=1.0,
            ))

        elif args.centroid_with_bundle:
            # Background: full bundle in light gray at fixed low opacity
            gray = np.array(mpl.colors.to_rgb('lightgray'))
            scene.add(_build_bundle_actor(
                bundle, gray, args,
                linewidth=args.linewidth, opacity=0.1,
            ))
            # Foreground: colored centroid added after for correct render order
            centroid_colors = _resolve_centroid_colors(
                centroid, args.bundle_color, s_index, r_index, args
            )
            scene.add(_build_bundle_actor(
                ArraySequence([centroid]), centroid_colors, args,
                linewidth=30, opacity=1.0,
            ))

        else:
            # Standard bundle rendering
            colors = _resolve_bundle_colors(
                bundle, args.bundle_color,
                s_index, r_index, valid_mask,
                args, bundle_idx=i,
            )
            scene.add(_build_bundle_actor(bundle, colors, args))

    # Glass brain
    if args.glass_brain_fnames is not None:
        for surf_path in args.glass_brain_fnames:
            _add_glass_brain(scene, surf_path)

    # Masks / ROIs
    if args.mask_path is not None:
        for j, mask_path in enumerate(args.mask_path):
            if args.mask_color is not None:
                mask_color = np.array(mpl.colors.to_rgba(args.mask_color[j]))
            else:
                mask_color = np.array([0, 0, 0])
            if mask_path.endswith('.nii.gz') or mask_path.endswith('.nii'):
                MASK, AFFINE = load_nifti(mask_path)
                logging.info(f'Loaded mask {mask_path}')
                scene.add(
                    actor.contour_from_roi(
                        MASK,
                        affine=AFFINE,
                        color=mask_color,
                        opacity=args.mask_opacity,
                    )
                )

    # Interactive window
    if args.show_bundle:
        window.show(scene, size=(1200, 1200))

    # Save camera
    if args.out_camera_fname is not None:
        cam_settings = {}
        cam_settings['pos'], cam_settings['foc'], cam_settings['vup'] = (
            scene.get_camera()
        )
        with open(args.out_camera_fname, 'wb') as f:
            logging.info(f'Saving camera settings to {args.out_camera_fname}')
            pickle.dump(cam_settings, f)

    # Save snapshot
    if args.out_image_fname is not None:
        window.record(scene=scene, size=(1600, 1600),
                      out_path=args.out_image_fname)
        if args.auto_crop:
            autocrop_png(args.out_image_fname)
        logging.info(f'Saved bundle snapshot to {args.out_image_fname}')


def main():
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument('--in_bundle_fnames', '-i', nargs='+', type=str, required=False,
                        help='Input bundle file(s), i.e. bundle.trk')
    parser.add_argument('--out_image_fname', '-o', type=str, default=None,
                        help='Output filepath to save bundle snapshot, i.e. bundle.png')

    # Camera
    parser.add_argument('--cam_view', '-cam', type=str, default='Axial',
                        help='Preset cam view: Axial (default), Saggital_L/R, Coronal_A/P')
    parser.add_argument('--in_camera_fname', '-si', type=str, default=None,
                        help='Input camera setting file, i.e ./cam.pkl')
    parser.add_argument('--out_camera_fname', '-so', type=str, default=None,
                        help='Output camera setting file, i.e ./cam.pkl')

    # Bundle color
    parser.add_argument('--bundle_color', '-c', nargs='*', type=str, default=None,
                        help='Bundle color(s): mpl color string, .npy/.txt array, '
                             '.nii.gz volume, alternate_1d, or alternate_2d')
    parser.add_argument('--ref_bundle_fname', '-ref', type=str, default=None,
                        help='Reference/atlas bundle for parameterization. When provided, '
                             'subject bundles are parameterized onto this grid (enables '
                             'valid_mask from parameterize_bundle for 2D color modes).')
    parser.add_argument('--color_mask_fname', '-cmask', type=str, default=None,
                        help='.npy file containing a grid-level boolean mask of shape '
                             '(ns,) or (ns, nr). Cells that are False are colored gray. '
                             'Intended for stats/atlas visualization where some grid cells '
                             'have insufficient subjects.')
    parser.add_argument('--s_len', '-s', type=int, required=False, default=5,
                        help='Length of along-tract segments, must match bundle color file')
    parser.add_argument('--r_len', '-r', type=int, required=False, default=5,
                        help='Length of radial bins, must match bundle color file')
    parser.add_argument('--robust_method', '-robust', type=str, default='linear',
                        choices=['spline', 'linear', 'none'],
                        help='Extrapolation method for robust centroid')
    parser.add_argument('--cmap', '-cmap', type=str, default='YlOrBr_r',
                        help='Colormap name from matplotlib or cmasher, i.e. Blues')
    parser.add_argument('--cmap_title', '-ctitle', type=str, default='values',
                        help='Colormap title')
    parser.add_argument('--out_cbar_fname', '-cbar', type=str, default=None,
                        help='Output filepath to save the colorbar')
    parser.add_argument('--vmin', '-vmin', type=float, default=None,
                        help='Minimum value of colormap')
    parser.add_argument('--vmax', '-vmax', type=float, default=None,
                        help='Maximum value of colormap')
    parser.add_argument('--vcenter', '-vcenter', type=float, default=None,
                        help='Center value of colormap')

    # Glass brain
    parser.add_argument('--glass_brain_fnames', '-glass', nargs='+', type=str, default=None,
                        help='Glass brain: file paths, default_glass, or default_pial')

    # Mask/ROI
    parser.add_argument('--mask_path', '-mask', nargs='+', type=str, required=False,
                        help='Input mask file(s), i.e. roi.nii.gz')
    parser.add_argument('--mask_color', '-mask_color', nargs='*', type=str, default=None,
                        help='Color names for each mask')
    parser.add_argument('--mask_opacity', '-mask_opacity', type=float, default=0.3,
                        help='Opacity for mask(s)')

    # Plot options
    parser.add_argument('--centroid_only', '-centroid', action='store_true',
                        help='Only plot bundle centroid')
    parser.add_argument('--centroid_with_bundle', '-centroid_bg', action='store_true',
                        help='Plot colored centroid over the full bundle in light gray '
                             '(bundle rendered at fixed opacity=0.2)')
    parser.add_argument('--auto_crop', '-crop', action='store_true',
                        help='Automatically crop image when saving')
    parser.add_argument('--show_bundle', '-show', action='store_true',
                        help='Show the interactive window')
    parser.add_argument('--as_points', '-points', action='store_true',
                        help='Plot the bundle using points instead of streamtubes')
    parser.add_argument('--linewidth', '-linewidth', type=float, default=4,
                        help='Thickness of streamlines')
    parser.add_argument('--opacity', '-a', type=float, default=1,
                        help='Opacity for bundles')
    parser.add_argument('--fake_tube', '-fake_tube', action='store_true', default=True,
                        help='Show streamline as fake tube')

    args = parser.parse_args()
    run(args)