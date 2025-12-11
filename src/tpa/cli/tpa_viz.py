"""
DESCRIPTION:
    Bundle Visualization with 1D/2D profile color support
USAGE:
    TODO
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

from tpa.tractometry.parameterization import get_grid_from_atlas, get_centroid
from tpa.viz.viz_fury import get_cam_settings_fury
from tpa.viz.colors import color_from_values, alternating_colors
from tpa.utils.io import load_list
from tpa.utils.utils import setup_logging, get_wm_mni_path

warnings.filterwarnings("ignore", category=UserWarning)

def autocrop_png(fname, border=20):
    # Load image
    img = plt.imread(fname)
    
    # If PNG has alpha channel, ignore it for cropping
    if img.shape[-1] == 4:
        rgb = img[:, :, :3]
    else:
        rgb = img

    # Assume white border is [1,1,1] in RGB (matplotlib loads PNG normalized 0-1)
    mask = np.any(rgb < 1.0, axis=-1)  # True where pixels are not white
    
    if not np.any(mask):
        print("Image is all white, skipping crop.")
        return

    # Find bounding box of non-white pixels
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)

    # Apply border, but keep within image dimensions
    y0 = max(y0 - border, 0)
    x0 = max(x0 - border, 0)
    y1 = min(y1 + border, img.shape[0]-1)
    x1 = min(x1 + border, img.shape[1]-1)

    cropped_img = img[y0:y1+1, x0:x1+1]
    plt.imsave(fname, cropped_img)

def _get_sr_index_self(bundle, s_len=3, r_len=3):
    s_index, r_index, _, _, _ = get_grid_from_atlas(bundle, s_len=s_len, r_len=r_len)
    return s_index, r_index


def get_default_glass():
    from scipy.ndimage import gaussian_filter
    img, affine = load_nifti(get_wm_mni_path())
    img[img<0.25]=0
    img=gaussian_filter(img, 0.3)
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

def run(args):

    scene = window.Scene()
    scene.SetBackground(1, 1, 1)

    # Set camera given camera setting file
    if args.in_camera_fname is not None:
        with open(args.in_camera_fname, "rb") as f:
            cam_settings = pickle.load(f)
        scene.set_camera(
            position=cam_settings["pos"],
            focal_point=cam_settings["foc"],
            view_up=cam_settings["vup"],
        )
    else:
        cam=get_cam_settings_fury(args.cam_view)
        logging.info(f'Camera is set to {args.cam_view}.')
        scene.set_camera(**cam)

    # Plot bundles
    for i, bundle_path in enumerate(args.in_bundle_fnames):
        bundle = load_trk(
            bundle_path, "same", bbox_valid_check=False
        ).streamlines
        logging.info(f'Loaded {len(bundle)} streamlines from {bundle_path}.')

        # Make segment colors if values are supplied
        colors = None
        values = None
        if args.bundle_color is not None:

            ends = args.bundle_color[0].endswith

            # Compute grids
            s_index, r_index = None, None
            if not args.centroid_only:
                s_index, r_index =  _get_sr_index_self(bundle, s_len=args.s_len, r_len=args.r_len)

             # Get 1d or 2d color from values
            if ends('.npy') or ends('.txt'):
                values = load_list(args.bundle_color[i])

             # Per-point color from input volume
            elif ends('nii.gz'):
                img_data, affine = load_nifti(args.bundle_color[0]) # only support one volume
                bundle_native = transform_streamlines(bundle, np.linalg.inv(affine))
                bundle_native = ArraySequence(bundle_native).get_data()
                values = map_coordinates_3d_4d(img_data, bundle_native).T
            
            # Grid color for 2d
            elif 'alternate_2d' in args.bundle_color[0].lower():
                colors = alternating_colors(s_index=s_index, r_index=r_index, 
                                            pattern=[['red','blue'],['black','white']])
            elif 'alternate_1d' in args.bundle_color[0].lower():
                colors = alternating_colors(s_index=s_index, r_index=None, 
                                            pattern=['blue','white'])

            # Per bundle color by named colors
            else:
                colors = np.array(mpl.colors.to_rgba(args.bundle_color[i]))

            if values is not None: # Convert along-tract or volume values to color
                colors = color_from_values(values, s_index=s_index, r_index=r_index,
                                     cmap=args.cmap, vmin=args.vmin, vmax=args.vmax,
                                     plot_cmap_label=args.cmap_title, plot_horizontal=True,
                                     show_cmap=False, save_cmap_path=args.out_cbar_fname)

        # Plot bundle
        if args.centroid_only:
            bundle_plot = ArraySequence([get_centroid(bundle, n_points=len(colors), thresh=100)])
        else:
            bundle_plot = bundle

        if args.as_points: # Plot bundle as points 
            scene.add(actor.point(
                bundle_plot.get_data(), colors=window.colors.green, point_radius=0.2
            ))
        else: # Plot bundle as streamlines 
            scene.add(actor.line(
                bundle_plot, 
                fake_tube=args.fake_tube, 
                linewidth=args.linewidth if not args.centroid_only else 30, 
                colors=colors, 
                opacity=args.opacity if not args.centroid_only else 1
            ))

    # Plot glass brain
    if args.glass_brain_fnames is not None:
        for i, surf_path in enumerate(args.glass_brain_fnames):

            ends = surf_path.endswith

            # NIFTI image
            if 'default_glass' in surf_path or ends(".nii.gz") or ends(".nii"):
                if 'default_glass' in surf_path:
                    mask, affine = get_default_glass()
                    logging.info(f"Using default glass brain...")
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

            # GIFTI image
            if 'default_pial' in surf_path or ends(".gii.gz") or ends(".gii"):
                if 'default_pial' in surf_path:
                    vertices, faces = get_default_pial()
                    logging.info('Using default pial surface...')
                else:
                    surface = load_gifti(surf_path)
                    vertices, faces = surface
                    if len(vertices) and len(faces):
                        vertices, faces = surface
                    else:
                        warnings.warn(
                            "{} does not have any surface geometry.".format(
                                args.glass_brain_fnames
                            )
                        )

                colors = np.zeros((vertices.shape[0], 3))

                surf_actor = actor.surface(
                    vertices, faces=faces, colors=colors
                )
                surf_actor.GetProperty().SetOpacity(0.05)

                scene.add(surf_actor)

    if args.mask_path is not None:
        for i, mask_path in enumerate(args.mask_path):
            if args.mask_color is not None:
                mask_color = np.array(mpl.colors.to_rgba(args.mask_color[i]))
            else:
                mask_color = np.array([0, 0, 0])
            if mask_path.endswith(".nii.gz") or mask_path.endswith(".nii"):
                MASK, AFFINE = load_nifti(mask_path)
                logging.info(f"Loaded mask {mask_path}")
                scene.add(
                    actor.contour_from_roi(
                        MASK,
                        affine=AFFINE,
                        color=mask_color,
                        opacity=args.mask_opacity,
                    )
                )

    # Show bundle window
    if args.show_bundle:
        window.show(scene, size=(1200, 1200))

    # Save camera settings
    if args.out_camera_fname is not None:
        cam_settings = {}
        cam_settings["pos"], cam_settings["foc"], cam_settings["vup"] = (
            scene.get_camera()
        )
        with open(args.out_camera_fname, "wb") as f:
            logging.info(f"Saving camera settings to {args.out_camera_fname}")
            pickle.dump(cam_settings, f)

    # Save bundle snapshot
    if args.out_image_fname is not None:
        window.record(scene=scene, size=(1600, 1600), out_path=args.out_image_fname)
        if args.auto_crop:
            autocrop_png(args.out_image_fname)
        logging.info(f"Saved bundle snapshot to {args.out_image_fname}")


def main():
    setup_logging()    

    parser = argparse.ArgumentParser()
    parser.add_argument( "--in_bundle_fnames", "-i", nargs="+", type=str, required=False, 
                        help="Input bundle file(s), i.e. bundle.trk", )
    parser.add_argument( "--out_image_fname", "-o", type=str, default=None, 
                        help="Output filepath to save bundle snapshot, i.e. bundle.png", )
    
    # Cam settings args
    parser.add_argument( "--cam_view", "-cam", type=str, default="Axial", 
                        help="Select a preset cam view, from Axial (default), Saggital_L/R, Coronal_A/P", )
    parser.add_argument( "--in_camera_fname", "-si", type=str, default=None, 
                        help="Input camera setting file, i.e ./cam.pkl", )
    parser.add_argument( "--out_camera_fname", "-so", type=str, default=None, 
                        help="Output camera setting file, i.e ./cam.pkl", )
    
    # Bundle color args
    parser.add_argument( "--bundle_color", "-c", nargs="*", type=str, default=None, 
                        help="Bundle color(s), can be mpl colors (string), along-tract colors (.npy, .txt), \
                             or volume color (.nii.gz, only one file is supported")
    parser.add_argument('--s_len', '-s', type=int, required=False, default=3, 
                        help="Length of along-tract segments, must match with bundle color file")
    parser.add_argument('--r_len', '-r', type=int, required=False, default=3, 
                        help="Length of radial bins, must match with bundle color file")
    parser.add_argument( "--cmap", "-cmap", type=str, default="YlOrBr_r", 
                        help="Colormap name from matplotlib or cmasher i.e. Blues", )
    parser.add_argument( "--cmap_title", "-ctitle", type=str, default="values", 
                        help="Colormap title", )
    parser.add_argument( "--out_cbar_fname", "-cbar", type=str, default=None, 
                        help="Output filepath to save the colorbar", )
    parser.add_argument( "--vmin", "-vmin", type=float, default=None, 
                        help="Minimum value of colormap", )
    parser.add_argument( "--vmax", "-vmax", type=float, default=None, 
                        help="Maximum value of colormap", )
    parser.add_argument( "--vcenter", "-vcenter", type=float, default=None, 
                        help="Center value of colormap", )
    
    # Glass brain args
    parser.add_argument( "--glass_brain_fnames", "-glass", nargs="+", type=str, default=None, 
                        help="Glass brain to plot, specify file paths, or default_glass, or default pial", )
    
    # Mask/ROI args
    parser.add_argument( "--mask_path", "-mask", nargs="+", type=str, required=False, 
                        help="Input mask file(s), i.e. roi.nii.gz", )
    parser.add_argument( "--mask_color", "-mask_color", nargs="*", type=str, default=None, 
                        help="List of color names (available in mpl) to use for each mask", )
    parser.add_argument( "--mask_opacity", "-mask_opacity", type=float, default=0.3, 
                        help="Opacity for mask(s)", )

    # Plot option args
    parser.add_argument( "--centroid_only", "-centroid", action="store_true", 
                        help="Only plot bundle centroid", )
    parser.add_argument( "--auto_crop", "-crop", action="store_true", 
                        help="Automatically crop image when saving", )
    parser.add_argument( "--show_bundle", "-show", action="store_true", 
                        help="Show the interactive window", )
    parser.add_argument( "--as_points", "-points", action="store_true", 
                        help="Plots the bundle using points instead of streamtubes.", )
    parser.add_argument( "--linewidth", "-linewidth", type=float, default=4, 
                        help="Thickness of streamlines", )
    parser.add_argument( "--opacity", "-a", type=float, default=1, 
                        help="Opacity for bundles", )
    parser.add_argument( "--fake_tube", "-fake_tube", action="store_true", default=True, 
                        help="Show streamline as fake tube", )
    args = parser.parse_args()
    run(args)
