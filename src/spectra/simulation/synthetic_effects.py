import logging
from typing import List, Dict, Optional, Tuple, Union

import numpy as np

from spectra.utils.utils import get_rng


def compute_std_map(raw_data: np.ndarray) -> np.ndarray:
    """
    Compute standard deviation map across subjects. Zeros mark missing
    values and are excluded.

    Parameters
    ----------
    raw_data : np.ndarray
        Shape (n_subj, ns) for 1D or (n_subj, ns, nr) for 2D

    Returns
    -------
    std_map : np.ndarray
        Shape (ns,) for 1D or (ns, nr) for 2D
    """
    with np.errstate(invalid='ignore'):
        std_map = np.nanstd(np.where(raw_data != 0, raw_data, np.nan), axis=0, ddof=1)
    return np.nan_to_num(std_map, nan=1.0)


def _apply_decay(dist: np.ndarray, decay: str) -> np.ndarray:
    """
    Apply decay function to normalized distance field.

    Parameters
    ----------
    dist : np.ndarray
        Normalized distances (0 = center, 1 = boundary)
    decay : str
        'cosine', 'gaussian', or 'flat'

    Returns
    -------
    factor : np.ndarray
        Decay factor in [0, 1]. Same shape as dist.
    """
    if decay == 'cosine':
        return np.where(dist <= 1.0, 0.5 * (1 + np.cos(np.pi * dist)), 0.0)
    elif decay == 'gaussian':
        return np.exp(-0.5 * dist ** 2)
    elif decay == 'flat':
        return np.where(dist <= 1.0, 1.0, 0.0)
    else:
        raise ValueError(f"Unknown decay type: {decay}. Use 'cosine', 'gaussian', or 'flat'.")


def create_effect_map(
    ns: int,
    nr: int = 1,
    center: Optional[Union[float, Tuple[float, float]]] = None,
    extent: Union[float, Tuple[float, float]] = 0.5,
    max_effect: float = 0.8,
    min_effect: float = 0.0,
    decay: str = 'cosine',
    geometry: str = 'ellipse',
    s_len: float = 3.0,
    r_len: float = 3.0,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Create a single effect region on a 2D (or 1D) grid.

    The effect decays from max_effect at the center to min_effect at the
    boundary (normalized distance = 1), then drops to 0 outside. Setting
    min_effect > 0 produces a discontinuous step at the boundary; set to
    0 for smooth decay to zero.

    For gaussian decay, there is no hard boundary — the effect extends
    smoothly toward zero with no min_effect floor.

    Parameters
    ----------
    ns, nr : int
        Grid dimensions. nr=1 for 1D.
    center : float or (float, float), optional
        Center as fraction of grid extent [0, 1].
        Scalar for 1D/shared, tuple (s_frac, r_frac) for 2D.
        Default: 0.5 (center of grid).
    extent : float or (float, float)
        Fractional half-extent of effect region in each direction [0, 1].
        tuple (s_extent, r_extent) for anisotropic.
    max_effect : float
        Effect magnitude at center.
    min_effect : float
        Effect magnitude at boundary edge (normalized dist = 1).
        Ignored for gaussian decay. Default 0.0.
    decay : str
        'cosine': smooth taper with hard cutoff at boundary.
        'gaussian': smooth exponential decay, no hard boundary.
        'flat': uniform within boundary, zero outside.
    geometry : str
        'ellipse' or 'rect'
    s_len, r_len : float
        Grid spacing in mm.
    mask : np.ndarray, optional
        Boolean mask of valid grid cells. Shape (ns,) or (ns, nr).

    Returns
    -------
    effect_map : np.ndarray
        Shape (ns,) if nr==1, else (ns, nr). Values in [0, max_effect].
    """
    is_1d = (nr == 1)

    # Parse center
    if center is None:
        center_s_frac, center_r_frac = 0.5, 0.5
    elif isinstance(center, (int, float)):
        center_s_frac = float(center)
        center_r_frac = 0.5
    else:
        center_s_frac, center_r_frac = center

    # Parse extent -> sigma in mm
    if isinstance(extent, (int, float)):
        extent_s, extent_r = float(extent), float(extent)
    else:
        extent_s, extent_r = extent

    sigma_s = extent_s * ns * s_len
    sigma_r = extent_r * nr * r_len

    # Build coordinate grids and compute normalized distance
    s_coords = np.arange(ns) * s_len
    center_s_mm = center_s_frac * ns * s_len

    if is_1d:
        ds = np.abs(s_coords - center_s_mm) / sigma_s if sigma_s > 0 else np.zeros(ns)
        dist = ds
        factor = _apply_decay(dist, decay)

        if decay == 'gaussian':
            effect_map = max_effect * factor
        else:
            effect_map = np.where(
                dist <= 1.0,
                min_effect + (max_effect - min_effect) * factor,
                0.0,
            )

        if mask is not None:
            mask_1d = mask.flatten() if mask.ndim > 1 else mask
            effect_map[~mask_1d] = 0.0
    else:
        r_coords = np.arange(nr) * r_len
        S, R = np.meshgrid(s_coords, r_coords, indexing='ij')

        center_r_mm = center_r_frac * nr * r_len
        ds = np.abs(S - center_s_mm) / sigma_s if sigma_s > 0 else np.zeros_like(S)
        dr = np.abs(R - center_r_mm) / sigma_r if sigma_r > 0 else np.zeros_like(R)

        if geometry == 'rect':
            dist = np.maximum(ds, dr)
        elif geometry == 'ellipse':
            dist = np.sqrt(ds ** 2 + dr ** 2)
        else:
            raise ValueError(f"Unknown geometry: {geometry}. Use 'ellipse' or 'rect'.")

        factor = _apply_decay(dist, decay)

        if decay == 'gaussian':
            effect_map = max_effect * factor
        else:
            effect_map = np.where(
                dist <= 1.0,
                min_effect + (max_effect - min_effect) * factor,
                0.0,
            )

        if mask is not None:
            effect_map[~mask] = 0.0

    return effect_map


def create_effect_map_multi(
    ns: int,
    nr: int = 1,
    n_blobs: int = 3,
    center_range: Tuple[float, float] = (0.15, 0.85),
    extent_range: Tuple[float, float] = (0.1, 0.4),
    max_effect_range: Tuple[float, float] = (0.3, 0.8),
    min_effect: float = 0.0,
    decay: str = 'cosine',
    geometry: str = 'ellipse',
    s_len: float = 3.0,
    r_len: float = 3.0,
    mask: Optional[np.ndarray] = None,
    rng: Optional[Union[np.random.Generator, int]] = None,
) -> np.ndarray:
    """
    Composite multiple random effect regions via elementwise max.

    Each blob's center, extent, and max_effect are sampled uniformly from
    the given ranges. All blobs share the same min_effect, decay, and
    geometry.

    Parameters
    ----------
    ns, nr : int
        Grid dimensions. nr=1 for 1D.
    n_blobs : int
        Number of blobs to generate.
    center_range : (float, float)
        Uniform range for each center coordinate (fractional, [0, 1]).
    extent_range : (float, float)
        Uniform range for fractional half-extent per dimension.
    max_effect_range : (float, float)
        Uniform range for max_effect per blob.
    min_effect : float
        Shared min_effect for all blobs.
    decay : str
        Shared decay type for all blobs ('cosine', 'gaussian', 'flat').
    geometry : str
        Shared geometry for all blobs ('ellipse' or 'rect').
    s_len, r_len : float
        Grid spacing in mm.
    mask : np.ndarray, optional
        Boolean mask. Shape (ns,) or (ns, nr).
    rng : np.random.Generator or int, optional
        Random generator or seed.

    Returns
    -------
    effect_map : np.ndarray
        Shape (ns,) if nr==1, else (ns, nr).
    """
    is_1d = (nr == 1)
    rng = get_rng(rng)

    effect_map = np.zeros(ns) if is_1d else np.zeros((ns, nr))

    for _ in range(n_blobs):
        if is_1d:
            center = rng.uniform(*center_range)
            extent = rng.uniform(*extent_range)
        else:
            center = (rng.uniform(*center_range), rng.uniform(*center_range))
            extent = (rng.uniform(*extent_range), rng.uniform(*extent_range))

        blob_map = create_effect_map(
            ns, nr,
            center=center,
            extent=extent,
            max_effect=rng.uniform(*max_effect_range),
            min_effect=min_effect,
            decay=decay,
            geometry=geometry,
            s_len=s_len,
            r_len=r_len,
            mask=mask,
        )
        effect_map = np.maximum(effect_map, blob_map)

    return effect_map


def inject_effect_to_array(
    raw_data: np.ndarray,
    effect_map: np.ndarray,
    effect_size: float,
    subject_mask: np.ndarray,
    std_map: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Inject synthetic effects into a raw data array. Zeros mark missing
    values and are left unchanged.

    Parameters
    ----------
    raw_data : np.ndarray
        Shape (n_subj, ns) for 1D or (n_subj, ns, nr) for 2D
    effect_map : np.ndarray
        Shape (ns,) for 1D or (ns, nr) for 2D
    effect_size : float
        Effect magnitude (Cohen's d if std_map provided)
    subject_mask : np.ndarray (n_subj,)
        Boolean array indicating which subjects receive the effect
    std_map : np.ndarray, optional
        If provided, effect_size is scaled by local std.

    Returns
    -------
    modified_data : np.ndarray
        Same shape as raw_data, with effects injected
    """
    modified_data = raw_data.copy()

    if std_map is not None:
        effect_to_add = effect_map * effect_size * std_map
    else:
        effect_to_add = effect_map * effect_size

    observed = modified_data[subject_mask] != 0
    modified_data[subject_mask] += effect_to_add * observed

    return modified_data