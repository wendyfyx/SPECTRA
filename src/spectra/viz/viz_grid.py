"""
Provides grid-building utilities and individual plot functions for:
- RESEL block checkerboard patterns
- Block-level and cell-level significance maps
- Stat maps (Cohen's d, beta, p-values, etc.)
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import seaborn as sns

from spectra.utils.io import save_fig, save_npy
from spectra.viz.colors import make_colormap

def get_checkerboard(
    block_grid: np.ndarray,
    block_w: Optional[int] = None,
    block_h: Optional[int] = None,
) -> np.ndarray:
    """
    Derive a 2-color checkerboard pattern from a block ID grid.

    Parameters
    ----------
    block_grid : np.ndarray (ns, nr)
        Block IDs; NaN outside the valid mask.
    block_w : int, optional
        Override inferred block width (columns per block).
    block_h : int, optional
        Override inferred block height (rows per block).

    Returns
    -------
    checker : np.ndarray (ns, nr), float
        Values 0 or 1 in alternating pattern; NaN outside mask.
    """
    grid = np.asarray(block_grid)
    valid = np.isfinite(grid)

    if not valid.any():
        return np.full_like(grid, np.nan)

    # Infer block width from column run-lengths
    if block_w is None:
        widths = []
        for row in grid:
            row_valid = row[np.isfinite(row)]
            if len(row_valid) < 2:
                continue
            changes = np.where(np.diff(row_valid) != 0)[0]
            run_lengths = np.diff(np.r_[-1, changes, len(row_valid) - 1])
            widths.extend(run_lengths.tolist())
        block_w = int(np.median(widths)) if widths else 1

    # Infer block height from row run-lengths
    if block_h is None:
        heights = []
        for col in grid.T:
            col_valid = col[np.isfinite(col)]
            if len(col_valid) < 2:
                continue
            changes = np.where(np.diff(col_valid) != 0)[0]
            run_lengths = np.diff(np.r_[-1, changes, len(col_valid) - 1])
            heights.extend(run_lengths.tolist())
        block_h = int(np.median(heights)) if heights else 1

    ns, nr = grid.shape
    block_row = np.arange(ns)[:, None] // block_h
    block_col = np.arange(nr)[None, :] // block_w
    checker = ((block_row + block_col) % 2).astype(float)
    checker[~valid] = np.nan
    return checker


def _plot_binary_grid(
    grid: np.ndarray,
    ax: plt.Axes,
    colors: Tuple[str, str],
    legend: bool,
    legend_labels: Tuple[str, str] = ('Not significant', 'Significant'),
) -> None:
    """Internal: render a 0/1/NaN binary grid in 1D or 2D."""
    c0 = mcolors.to_rgba(colors[0])
    c1 = mcolors.to_rgba(colors[1])

    ns, nr = grid.shape

    if nr == 1:
        # 1D: fill_between bands along s
        s = np.arange(ns)
        vals = grid[:, 0]
        valid = np.isfinite(vals)
        # non-sig band
        ax.fill_between(s, 0, 1,
                        where=(valid & (vals < 0.5)),
                        color=c0, step='mid', linewidth=0)
        # sig band
        ax.fill_between(s, 0, 1,
                        where=(valid & (vals >= 0.5)),
                        color=c1, step='mid', linewidth=0)
        ax.set_xlim(0, ns - 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('s position')
        ax.set_yticks([])
    else:
        # 2D: pcolormesh with two-color ListedColormap; NaN → transparent
        cmap_bin = mcolors.ListedColormap([c0, c1])
        masked = np.ma.masked_invalid(grid)

        ax.pcolormesh(masked.T, cmap=cmap_bin, vmin=0, vmax=1)
        ax.grid(False)
        ax.set_xlabel('s')
        ax.set_ylabel('r')
        ax.set_aspect('equal')

    if legend:
        handles = [
            Patch(facecolor=c0, edgecolor='gray', label=legend_labels[0]),
            Patch(facecolor=c1, edgecolor='gray', label=legend_labels[1]),
        ]
        ax.legend(
            handles=handles, 
            loc='upper center', 
            bbox_to_anchor=(0.5, 1.25),
            ncols=2, 
            fontsize=12, 
            frameon=True)


def _plot_stat_grid(
    stat_grid: np.ndarray,
    ax: plt.Axes,
    cmap: str,
    vmin: Optional[float],
    vmax: Optional[float],
    vcenter: Optional[float],
    muted_color: Optional[str] = 'lightgray',
    cmap_label: Optional[str] = None,
) -> None:
    """Internal: render a stat grid in 1D or 2D with muted non-sig background."""
    finite_vals = stat_grid[np.isfinite(stat_grid)]
    if vmin is None:
        vmin = float(np.nanpercentile(finite_vals, 5)) if len(finite_vals) else 0.0
    if vmax is None:
        vmax = float(np.nanpercentile(finite_vals, 95)) if len(finite_vals) else 1.0

    if vcenter is not None and vmin < vcenter < vmax:
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)

    cmap_obj = plt.get_cmap(cmap)

    ns, nr = stat_grid.shape

    if nr == 1:
        # 1D: background fill for NaN (non-sig or invalid) then line
        s = np.arange(ns)
        vals = stat_grid[:, 0]
        valid_sig = np.isfinite(vals)

        # muted background for non-sig/invalid cells
        ax.fill_between(s, vmin, vmax,
                        where=~valid_sig,
                        color=muted_color, step='mid', alpha=0.4, linewidth=0)
        # plot line only over sig cells
        line_vals = np.where(valid_sig, vals, np.nan)
        ax.plot(s, line_vals, color=cmap_obj(0.6), linewidth=2)
        ax.scatter(s[valid_sig], vals[valid_sig],
                   c=vals[valid_sig], cmap=cmap, norm=norm,
                   s=20, zorder=3)
        ax.set_xlim(0, ns - 1)
        ax.set_ylim(vmin, vmax)
        ax.set_xlabel('s position')
        ax.set_ylabel(cmap)
    else:
        # 2D: gray background, then sig cells via pcolormesh
        masked = np.ma.masked_invalid(stat_grid)
        ax.pcolormesh(masked.T, cmap=cmap, norm=norm)
        ax.grid(False)
        ax.set_xlabel('s')
        ax.set_ylabel('r')
        ax.set_aspect('equal')

    if cmap_label is not None:
        make_colormap(cmap, vmin, vmax, vcenter=vcenter, 
                      plot_horizontal=True, 
                      plot_cmap_label=cmap_label,
                      show_cmap=True)


def plot_checkerboard(
    block_grid: np.ndarray,
    ax: Optional[plt.Axes] = None,
    colors: Tuple[str, str] = ('dimgrey', 'silver'),
    title: Optional[str] = None,
    show: bool = False,
    fig_output: Optional[str] = None,
    export_output: Optional[str] = None,
    block_w: Optional[int] = None,
    block_h: Optional[int] = None,
    disable_ax: bool = False
) -> Tuple[plt.Axes, np.ndarray]:
    """
    Plot RESEL block checkerboard pattern.

    Parameters
    ----------
    block_grid : np.ndarray (ns, nr)
        Block ID grid from get_sig_map()['block_grid'].
    ax : plt.Axes, optional
        Axes to plot into. Creates a new figure if None.
    colors : tuple of 2 str
        Colors for alternating blocks (color0, color1).
    title : str, optional
    show : bool
        Call plt.show() after plotting.
    fig_output : str, optional
        Path to save figure (PNG/PDF).
    export_output : str, optional
        Path to save checkerboard array as .npy.
    block_w, block_h : int, optional
        Override inferred block dimensions for get_checkerboard().

    Returns
    -------
    ax : plt.Axes
    checker : np.ndarray (ns, nr)
        The checkerboard array (0/1/NaN).
    """
    checker = get_checkerboard(block_grid, block_w=block_w, block_h=block_h)

    ns, nr = block_grid.shape
    created_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3) if nr == 1 else (6, 4))

    _plot_binary_grid(
        checker, ax,
        colors=colors,
        legend=False,
        legend_labels=('Block A', 'Block B'),
    )
    if title:
        ax.set_title(title)

    if created_fig:
        fig = ax.get_figure()
        fig.tight_layout()
        if disable_ax:
            ax.axis('off')
        if fig_output:
            save_fig(fig, fig_output)
        if show:
            plt.show()

    if export_output:
        save_npy(checker, export_output)

    return ax, checker


def plot_significance(
    sig_grid: np.ndarray,
    ax: Optional[plt.Axes] = None,
    colors: Tuple[str, str] = ('#B3CDE0', '#005B96'),
    title: Optional[str] = None,
    legend: bool = False,
    show: bool = False,
    fig_output: Optional[str] = None,
    export_output: Optional[str] = None,
    disable_ax: bool = False
) -> Tuple[plt.Axes, np.ndarray]:
    """
    Plot Stage 1 block-level significance map.

    Parameters
    ----------
    sig_grid : np.ndarray (ns, nr)
        Block or cell significance grid from get_sig_map()
    ax : plt.Axes, optional
    colors : tuple of 2 str
        (non-significant color, significant color).
    title : str, optional
    legend : bool
        Show legend.
    show : bool
    fig_output : str, optional
    export_output : str, optional

    Returns
    -------
    ax : plt.Axes
    sig_grid : np.ndarray (ns, nr)
    """
    ns, nr = sig_grid.shape
    created_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3) if nr == 1 else (6, 4))

    _plot_binary_grid(sig_grid, ax, colors=colors,
                      legend=legend,
                      legend_labels=('Not significant (block)', 'Significant (block)'))
    if title:
        ax.set_title(title)

    if created_fig:
        fig = ax.get_figure()
        fig.tight_layout()
        if disable_ax:
            ax.axis('off')
        if fig_output:
            save_fig(fig, fig_output)
        if show:
            plt.show()

    if export_output:
        save_npy(sig_grid, export_output)

    return ax, sig_grid


def plot_stats_map(
    stat_grid: np.ndarray,
    ax: Optional[plt.Axes] = None,
    cmap: str = 'rocket',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    vcenter: Optional[float] = None,
    title: Optional[str] = None,
    show: bool = False,
    fig_output: Optional[str] = None,
    export_output: Optional[str] = None,
    cmap_label: Optional[str] = None,
    disable_ax: bool = False
) -> Tuple[plt.Axes, np.ndarray]:
    """
    Plot a stat map (Cohen's d, beta, p-values, etc.) without a colorbar.
    Call make_colormap() separately to save a standalone colorbar.

    Parameters
    ----------
    stat_grid : np.ndarray (ns, nr)
        Output of get_stat_map(). NaN for invalid/non-sig cells.
    ax : plt.Axes, optional
    cmap : str
        Matplotlib colormap name.
    vmin, vmax : float, optional
        Color range. Defaults to 5th/95th percentile of finite values.
    vcenter : float, optional
        Diverging colormap center (e.g. 0 for beta maps).
    title : str, optional
    show : bool
    fig_output : str, optional
    export_output : str, optional

    Returns
    -------
    ax : plt.Axes
    stat_grid : np.ndarray (ns, nr)
        The input stat_grid (unchanged).
    """
    ns, nr = stat_grid.shape
    created_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3) if nr == 1 else (6, 4))

    _plot_stat_grid(stat_grid, ax,
                    cmap=cmap, vmin=vmin, vmax=vmax,
                    vcenter=vcenter,
                    cmap_label=cmap_label)
    if title:
        ax.set_title(title)

    if created_fig:
        fig = ax.get_figure()
        fig.tight_layout()
        if disable_ax:
            ax.axis('off')
        if fig_output:
            save_fig(fig, fig_output)
        if show:
            plt.show()

    if export_output:
        save_npy(stat_grid, export_output)

    return ax, stat_grid