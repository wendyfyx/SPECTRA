import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def make_colormap(cmap_name, vmin, vmax, vcenter=None, 
                  plot_cmap_label=None, plot_horizontal=True):
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
        fig.show()
    return cmap


def orient2rgba(orient_vectors, alpha=1.0):
    if orient_vectors.ndim == 1:
        r = np.linalg.norm(orient_vectors)
        orient = np.abs(np.divide(orient_vectors, r, where=r != 0))
    elif orient_vectors.ndim == 2:
        orientn = np.sqrt(orient_vectors[:, 0] ** 2 + orient_vectors[:, 1] ** 2 + orient_vectors[:, 2] ** 2)
        orientn.shape = orientn.shape + (1,)
        orient = np.abs(np.divide(orient_vectors, orientn, where=orientn != 0))
    else:
        raise IOError("Wrong vector dimension, It should be an array" " with a shape (N, 3)")

    ## append alpha channel to the orient
    if orient_vectors.ndim == 1:
        rgba = np.concatenate([orient, [alpha]])
    elif orient_vectors.ndim == 2:
        alpha_channel = np.full((orient.shape[0], 1), alpha)
        rgba = np.concatenate([orient, alpha_channel], axis=1)
    return rgba

def directional_color(bundle):
    '''Plot WM bundles with color based on the direction of the streamline (RGB)'''
    streamline_colors = []
    for streamline in bundle:
        diff = np.diff(streamline, axis=0)
        diff = np.concatenate([diff[:1], diff], axis=0)
        colors_tract = orient2rgba(np.asarray(diff))
        streamline_colors.append(colors_tract)
    return streamline_colors

def alternating_colors(s_index=None, 
                       r_index=None, 
                       pattern=[['red','blue'],['blue','red']], 
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
    pattern : str, list of 2 colors, or list of list of 2 colors
        - If list of 2 colors (e.g., ['red','green']), used as 1D checkerboard along s or r.
        - If list of list of 2 colors (e.g., [[a,b],[c,d]]), used as 2D checkerboard along s and r.
    valid_mask : array-like, shape (N,), optional
        1 for valid points, 0 for invalid points (marked gray).
        
    Returns
    -------
    colors : list of tuple
        List of RGB tuples for each point.
    """
    
    # Determine number of points
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
    
    colors = np.zeros((N,3))
    gray = np.array(mpl.colors.to_rgb('gray'))
    
    # 2D checkerboard
    if s_index is not None and r_index is not None and isinstance(pattern, (list, tuple)) and all(isinstance(p, (list, tuple)) for p in pattern):
        r_index = np.asarray(r_index)
        for i in range(N):
            if valid_mask[i]:
                row = pattern[s_index[i] % len(pattern)]
                c = row[r_index[i] % len(row)]
                colors[i] = mpl.colors.to_rgb(c)
            else:
                colors[i] = gray
                
    # 1D checkerboard
    elif (s_index is not None or r_index is not None) and isinstance(pattern, (list, tuple)) and len(pattern) == 2:
        idx_array = s_index if s_index is not None else r_index
        for i in range(N):
            colors[i] = mpl.colors.to_rgb(pattern[idx_array[i] % 2]) if valid_mask[i] else gray

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

    values = np.asarray(values)

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

    # ------- Mode 1: Per-point values (N,) -------
    if values.ndim == 1 and (s_index is None and r_index is None):
        point_values = values

    # ------- Mode 2: 1D grid (ns,) -------
    elif values.ndim == 1:
        ns = values.shape[0]
        if s_index is None:
            raise ValueError("1D grid requires s_index.")
        if np.max(s_index) + 1 != ns:
            raise ValueError(f"s_index expects ns={ns}, got max {np.max(s_index)}")
        point_values = values[s_index]

    # ------- Mode 3: 2D grid (ns,nr) -------
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

    # Assign gray to invalid points
    colors[~valid_mask] = np.array([0.5, 0.5, 0.5])

    return [tuple(c) for c in colors]