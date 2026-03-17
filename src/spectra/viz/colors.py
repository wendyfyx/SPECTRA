import logging
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch


OKABE_ITO = [
    '#000000',  # black
    '#E69F00',  # orange
    '#56B4E9',  # sky blue
    '#009E73',  # bluish green
    '#F0E442',  # yellow
    '#0072B2',  # blue
    '#D55E00',  # vermillion
    '#CC79A7',  # reddish purple
]

def get_palette(name='okabe', n_start=0, n_end=None):
    if name=='okabe':
        n_end = len(OKABE_ITO) if n_end is None else n_end
        return OKABE_ITO[n_start:n_end]
    else: 
        cmap = plt.get_cmap(name)
        n_end = cmap.N if n_end is None else n_end
        return [cmap(i) for i in range(cmap.N)][n_start:n_end]


def make_colormap(cmap_name, vmin, vmax, vcenter=None, 
                  plot_cmap_label=None, plot_horizontal=True,
                  show_cmap=False, save_cmap_path=None):
    '''Make colormap given name and value range'''
    if vcenter is not None and vmin < vcenter:
        norm = mpl.colors.TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=vcenter)
    else:
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)

    cmap = plt.get_cmap(cmap_name)
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    if plot_cmap_label is not None:
        figsize = (8, 1) if plot_horizontal else (0.5, 10)
        orientation = 'horizontal' if plot_horizontal else "vertical"
        fig, ax = plt.subplots(figsize=figsize)
        fig.subplots_adjust(bottom=0.5)
        cb = fig.colorbar(cmap, cax=ax, orientation=orientation)
        cb.set_label(label=plot_cmap_label, size=20)
        ax.tick_params(axis='x' if plot_horizontal else 'x', labelsize=18)
        if show_cmap:
            fig.show()
        if save_cmap_path:
            fig.savefig(save_cmap_path, bbox_inches='tight')
    return cmap

def save_legend_fig(
    labels: List[str],
    colors: List[str],
    save_path: Optional[str] = None,
    orientation: str = 'vertical',
    edgecolor: str = 'none',
    fontsize: int = 18,
    show: bool = False,
) -> plt.Figure:
    """
    Save a standalone legend figure with colored patch entries.

    Parameters
    ----------
    labels : list of str
        Legend entry labels, one per patch.
    colors : list of str
        Patch face colors, same length as labels.
    save_path : str, optional
        If provided, saves the figure to this path.
    orientation : {'vertical', 'horizontal'}
        Layout direction of legend entries. 'vertical' stacks entries in a
        single column; 'horizontal' places them in a single row.
    edgecolor : str
        Edge color for each patch. Default 'none' (no border).
    fontsize : int
        Font size for legend labels.
    show : bool
        If True, call plt.show() after creating the figure.

    Returns
    -------
    fig : plt.Figure
    """
    if len(labels) != len(colors):
        raise ValueError(
            f"labels and colors must have the same length "
            f"(got {len(labels)} and {len(colors)})"
        )

    handles = [
        Patch(facecolor=c, edgecolor=edgecolor, label=lbl)
        for c, lbl in zip(colors, labels)
    ]

    ncols = len(labels) if orientation == 'horizontal' else 1
    # Size figure to content: wider for horizontal, taller for vertical
    if orientation == 'horizontal':
        figsize = (2.5 * len(labels), 1.0)
    else:
        figsize = (2.5, 0.55 * len(labels))

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    ax.legend(
        handles=handles,
        loc='center',
        ncols=ncols,
        fontsize=fontsize,
        frameon=True,
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        logging.info(f"Saved legend figure: {save_path}")
    if show:
        plt.show()

    return fig

    
def alternating_colors(s_index=None, 
                       r_index=None, 
                       pattern=get_palette(), 
                       valid_mask=None):
    """
    Generate RGB colors for points based on optional s_index and r_index,
    with invalid points marked as gray.
    
    Parameters
    ----------
    s_index : array-like, shape (N,), optional
        Axial segment indices for each point.
    r_index : array-like, shape (N,), optional
        Radial bin indices for each point.
    pattern : list of colors, or list of list of colors
        - If list of colors (e.g., ['red','green','blue']), used as 1D repeating
          pattern along s or r — any length supported.
        - If list of list of colors (e.g., [[a,b,c],[d,e,f]]), used as 2D pattern
          along s and r — outer list cycles along s, inner along r.
    valid_mask : array-like, shape (N,), optional
        1 for valid points, 0 for invalid points (marked gray).
        
    Returns
    -------
    colors : list of tuple
        List of RGB tuples for each point.
    """
    
    if s_index is not None:
        s_index = np.asarray(s_index)
        N = len(s_index)
    elif r_index is not None:
        r_index = np.asarray(r_index)
        N = len(r_index)
    else:
        return [tuple(mpl.colors.to_rgb('gray'))]
    
    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask)
        if len(valid_mask) != N:
            raise ValueError("valid_mask must have the same length as s_index or r_index")
    else:
        valid_mask = np.ones(N, dtype=bool)
    
    colors = np.zeros((N, 3))
    gray = np.array(mpl.colors.to_rgb('gray'))
    
    is_2d = (
        isinstance(pattern, (list, tuple)) and
        all(isinstance(p, (list, tuple)) for p in pattern)
    )

    if is_2d and s_index is not None and r_index is not None:
        # 2D pattern: outer list cycles along s, inner list cycles along r
        r_index = np.asarray(r_index)
        for i in range(N):
            if valid_mask[i]:
                row = pattern[s_index[i] % len(pattern)]
                c = row[r_index[i] % len(row)]
                colors[i] = mpl.colors.to_rgb(c)
            else:
                colors[i] = gray

    else:
        # 1D pattern: flat list of any length, cycles along s or r
        idx_array = s_index if s_index is not None else r_index
        n_colors = len(pattern)
        for i in range(N):
            if valid_mask[i]:
                colors[i] = mpl.colors.to_rgb(pattern[idx_array[i] % n_colors])
            else:
                colors[i] = gray

    return [tuple(c) for c in colors]


def color_from_values(values, s_index, r_index=None, valid_mask=None, 
                cmap='magma', vmin=None, vmax=None, vcenter=None, **kwargs):
    """
    Map 2D grid values (ns, nr) to RGB colors for individual points.

    Parameters
    ----------
    grid_values : array, shape (N,), (ns,) or (ns,nr)
        Scalar values defined on a 1D, 2D grid or per point.
    s_index : array-like, shape (N,), optional
        s-axis grid coordinates for each point.
    r_index : array-like, shape (N,), optional
        r-axis grid coordinates for each point.
    valid_mask : array-like, shape (N,), optional
        A mask where 1 means valid and 0 means invalid. Invalid points become gray.
    cmap : str or Colormap
        Name of a matplotlib colormap or a Colormap instance.

    Returns
    -------
    colors : list of tuple
        List of (r, g, b) tuples for each point.
    """

    values = np.asarray(values).squeeze()

    # Determine number of points N depending on mode
    if s_index is None and r_index is None:
        # per-point mode
        N = values.shape[0]
        s_index = r_index = None
    else:
        # index-based mode
        s_index = np.asarray(s_index) if s_index is not None else None
        r_index = np.asarray(r_index) if r_index is not None else None
        N = len(s_index) if s_index is not None else len(r_index)

    # default all-valid mask
    if valid_mask is None:
        valid_mask = np.ones(N, dtype=bool)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool)

    # Mode 1: Per-point values (N,)
    if values.ndim == 1 and (s_index is None and r_index is None):
        point_values = values

    # Mode 2: 1D grid (ns,)
    elif values.ndim == 1:
        ns = values.shape[0]
        if s_index is None:
            raise ValueError("1D grid requires s_index.")
        if np.max(s_index) + 1 != ns:
            raise ValueError(f"s_index expects ns={ns}, got max {np.max(s_index)}")
        point_values = values[s_index]

    # Mode 3: 2D grid (ns,nr)
    elif values.ndim == 2:
        if s_index is None or r_index is None:
            raise ValueError("2D grid requires both s_index and r_index.")
        ns, nr = values.shape
        if np.max(s_index) + 1 != ns:
            raise ValueError(f"s_index expects ns={ns}, got max {np.max(s_index)}")
        if np.max(r_index) + 1 != nr:
            raise ValueError(f"r_index expects nr={nr}, got max {np.max(r_index)}")
        point_values = values[s_index, r_index]

    else:
        raise ValueError("grid_values must be (N,), (ns,) or (ns,nr).")

    # Map values to colors
    values_valid = point_values[np.isfinite(point_values)]
    if len(values_valid) == 0: # if all NaN, return grey
        return [(0.5, 0.5, 0.5)] * N
    if vmin is None:
        vmin = np.percentile(values_valid, 5)
    if vmax is None:
        vmax = np.percentile(values_valid, 95)
    logging.debug(f"Plotting values with using range [{vmin:.4f},{vmax:.4f}].")

    cmap = make_colormap(cmap, vmin, vmax, vcenter=vcenter, **kwargs)
    colors = cmap.to_rgba(point_values)[:, :3]

    # Assign gray to invalid points and NaN points
    nan_mask = ~np.isfinite(point_values)
    colors[~valid_mask | nan_mask] = np.array([0.5, 0.5, 0.5])

    return [tuple(c) for c in colors]