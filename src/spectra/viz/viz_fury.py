import logging
import pickle
import warnings

import numpy as np
from dipy.io.image import load_nifti
from dipy.io.surface import load_gifti
from fury import actor, window

from spectra.utils.utils import get_wm_mni_path


def get_cam_settings_fury(cam_name='Axial'):
    EYE_SCALE=2
    CAM_SETTINGS = {
        'Axial': {'view_up': (0,1,0), 'focal_point': (0,0,0), 'position': (0,0,EYE_SCALE) },
        'Sagittal_L': {'view_up': (0,0,1), 'focal_point': (0,0,0),'position': (-1*EYE_SCALE, 0, 0) },
        'Sagittal_R': {'view_up': (0,0,1), 'focal_point': (0,0,0),'position': (EYE_SCALE, 0, 0) },
        'Coronal_A': {'view_up': (0,0,1), 'focal_point': (0,0,0), 'position': (0, EYE_SCALE, 0)},
        'Coronal_P': {'view_up': (0,0,1), 'focal_point': (0,0,0), 'position': (0, -1*EYE_SCALE, 0)}
        }
    return CAM_SETTINGS[cam_name]


# ---------------------------------------------------------------------------
# Default brain geometry loaders
# ---------------------------------------------------------------------------

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
# Actor helpers
# ---------------------------------------------------------------------------

def build_bundle_actor(bundle_plot, colors, args, linewidth=None, opacity=None):
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
            fake_tube=args.no_tube,
            linewidth=lw,
            colors=colors,
            opacity=op,
        )


def add_glass_brain(scene, surf_path):
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
# Per-frame rendering
# ---------------------------------------------------------------------------

def render_frame(scene, cam, cam_type, out_path, args, autocrop_fn=None):
    """
    Set the camera on ``scene`` and save a snapshot to ``out_path``.

    Parameters
    ----------
    scene : fury.window.Scene
    cam : str
        Either a preset name (e.g. 'Axial') when cam_type='preset',
        or a path to a .pkl file when cam_type='pkl'.
    cam_type : {'preset', 'pkl'}
    out_path : str or None
    args : argparse.Namespace
    autocrop_fn : callable or None
        Optional function to call for auto-cropping the saved PNG.
    """
    if cam_type == 'pkl':
        with open(cam, 'rb') as f:
            cam_settings = pickle.load(f)
        scene.set_camera(
            position=cam_settings['pos'],
            focal_point=cam_settings['foc'],
            view_up=cam_settings['vup'],
        )
    else:
        cam_settings = get_cam_settings_fury(cam)
        logging.info(f'Camera set to {cam}.')
        scene.set_camera(**cam_settings)

    if args.show_bundle:
        window.show(scene, size=(1200, 1200))

    if out_path is not None:
        window.record(scene=scene, size=(1600, 1600), out_path=out_path)
        if args.auto_crop and autocrop_fn is not None:
            autocrop_fn(out_path)
        logging.info(f'Saved snapshot to {out_path}')