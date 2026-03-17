"""
Hierarchical FDR for spatial signals.
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from scipy import stats, integrate
from typing import List, Tuple, Optional


@dataclass
class BlockInfo:
    """
    A spatial block (cluster) within a bundle for hierarchical FDR Stage 1.
    """
    bundle_name: str
    block_id: int
    cell_indices: List[Tuple[int, int]]
    z: float = np.nan
    p: float = np.nan
    p_adj: float = np.nan
    significant: bool = False
    combination_method: str = 'stouffer'

    # Spatial metadata (kept for logging/diagnostics)
    l_s: Optional[float] = None
    l_r: Optional[float] = None
    s_len: Optional[float] = None
    r_len: Optional[float] = None

    # Kernel for hFDR only when using Stouffer
    K: Optional[np.ndarray] = field(default=None, repr=False)
    row_sums: Optional[np.ndarray] = field(default=None, repr=False)
    total_sum: Optional[float] = None

    @property
    def n_cells(self) -> int:
        return len(self.cell_indices)

    def set_kernel(self, K: np.ndarray) -> None:
        """
        Store the Matern correlation matrix and precompute sufficient statistics.

        Parameters
        ----------
        K : np.ndarray, shape (n_cells, n_cells)
            Full correlation matrix from build_masked_kernel_2d restricted to
            this block's cells.
        """
        self.K = K
        self.row_sums = K.sum(axis=1)       # (n_cells,): Σ_j K_ij for each i
        self.total_sum = float(K.sum())     # scalar: Σ_ij K_ij

    def get_rho(self, local_idx: int) -> float:
        """
        Correlation between cell ``local_idx`` and the block Stouffer statistic Z_S.

        Parameters
        ----------
        local_idx : int
            Zero-based index into cell_indices for this block.
        """
        if self.row_sums is None or self.total_sum is None or self.total_sum <= 0:
            n = self.n_cells
            return 1.0 / np.sqrt(n) if n > 0 else np.nan
        return float(self.row_sums[local_idx] / np.sqrt(self.total_sum))

    @property
    def rho_avg(self) -> float:
        """Average pairwise correlation — for diagnostics/logging only."""
        if self.total_sum is None or self.n_cells <= 1:
            return 0.0
        n = self.n_cells
        off_diag = self.total_sum - n   # subtract diagonal (all 1s)
        return float(np.clip(off_diag / (n * (n - 1)), 0.0, 1.0))

    @property
    def rho(self) -> float:
        """
        Average per-cell rho — scalar summary for backward compat.
        Prefer get_rho(local_idx) in Stage 2 for per-cell accuracy.
        """
        if self.row_sums is None or self.total_sum is None or self.total_sum <= 0:
            n = self.n_cells
            return 1.0 / np.sqrt(n) if n > 0 else np.nan
        return float(self.row_sums.mean() / np.sqrt(self.total_sum))

def compute_conditional_pvalue(
    z_cell: float,
    u1: float,
    rho: float,
) -> float:
    """
    Conditional p-value for a cell within a rejected block (two-sided test).

    Uses the conservative upper-bound formula (mu_i -> 0, pi0 -> 1), which is
    valid regardless of the alternative distribution assumption.

    Parameters
    ----------
    z_cell : float
        Cell z-score (signed).
    u1 : float
        Two-sided Stage 1 p-value threshold (max raw p among rejected blocks).
    rho : float
        Correlation between this cell's z-score and the block Stouffer Z_S.
        Use BlockInfo.get_rho(local_idx) — NOT the scalar BlockInfo.rho.

    Returns
    -------
    float
        Conditional p-value in [0, 1].

    References
    ----------
    Benjamini Y, Heller R (2007). False Discovery Rates for Spatial Signals. JASA.
    """
    if np.isnan(z_cell) or np.isnan(u1) or np.isnan(rho):
        return np.nan

    z_abs = np.abs(z_cell)
    t = stats.norm.ppf(1 - u1 / 2)   # two-sided Stage 1 cutoff

    if np.abs(rho) < 1e-10:
        return float(2 * stats.norm.sf(z_abs))

    sqrt_1_rho2 = np.sqrt(max(1 - rho ** 2, 0.0))

    def integrand(u):
        cond_mean = rho * u
        upper = stats.norm.sf((z_abs - cond_mean) / sqrt_1_rho2)
        lower = stats.norm.cdf((-z_abs - cond_mean) / sqrt_1_rho2)
        return (upper + lower) * stats.norm.pdf(u)

    num_pos, _ = integrate.quad(integrand, t, np.inf)
    num_neg, _ = integrate.quad(integrand, -np.inf, -t)
    denom = 2 * stats.norm.sf(t)

    if denom < 1e-15:
        return float(2 * stats.norm.sf(z_abs))

    return float(np.clip((num_pos + num_neg) / denom, 0.0, 1.0))

def stouffer_combine(
    z_values: np.ndarray,
    weights: np.ndarray = None,
    K: np.ndarray = None,
) -> Tuple[float, float]:
    """
    Combine z-scores using the generalized Stouffer method.

    Parameters
    ----------
    z_values : array-like
        Cell z-scores.
    weights : array-like, optional
        Cell weights. Default: uniform.
    K : np.ndarray, shape (n, n), optional
        Matern correlation matrix from BlockInfo.K.
        If None, independence is assumed.

    Returns
    -------
    z_combined : float
    p_combined : float (two-sided)
    """
    z_values = np.asarray(z_values, dtype=float)
    n = len(z_values)

    if n == 0:
        return np.nan, np.nan
    if n == 1:
        return float(z_values[0]), float(2 * stats.norm.sf(np.abs(z_values[0])))

    w = np.ones(n) / n if weights is None else np.asarray(weights, dtype=float)
    w = w / w.sum()

    numerator = float(np.dot(w, z_values))

    if K is not None:
        var_numerator = float(w @ K @ w)   # exact: w^T K w
    else:
        var_numerator = float(np.dot(w, w))  # independence: Σ w_i^2

    var_numerator = max(var_numerator, 1e-12)
    z_combined = numerator / np.sqrt(var_numerator)
    p_combined = float(2 * stats.norm.sf(np.abs(z_combined)))

    return float(z_combined), p_combined

def simes_combine(
    p_values: np.ndarray,
    z_values: np.ndarray,
) -> Tuple[float, float]:
    """
    Combine p-values using Simes' test (valid under PRDS dependence).
    For a block with c cells, Simes' combined p-value is:
        p_Simes = min_k( c * p_(k) / k )
    where p_(1) <= ... <= p_(c) are the sorted p-values.

    Parameters
    ----------
    p_values : array-like
        Raw (uncorrected) two-sided p-values for each cell in the block.
    z_values : array-like
        Signed z-scores for each cell. Used only to determine the sign of
        the returned z_combined (sign of the cell with the smallest p-value),
        so that blk.z has a meaningful direction for diagnostics.

    Returns
    -------
    z_combined : float
        Normal quantile back-computed from p_Simes, signed by the direction
        of the most significant cell. Used as blk.z for diagnostics and as
        mu_i in Stage 2 (exact=False only; exact=True is disabled for Simes).
    p_combined : float
        Simes combined p-value (two-sided).
    """
    p_values = np.asarray(p_values, dtype=float)
    z_values = np.asarray(z_values, dtype=float)
    n = len(p_values)

    if n == 0:
        return np.nan, np.nan
    if n == 1:
        return float(z_values[0]), float(p_values[0])

    # Sort p-values and compute Simes statistic
    sort_idx = np.argsort(p_values)
    p_sorted = p_values[sort_idx]
    ranks = np.arange(1, n + 1)
    simes_vals = n * p_sorted / ranks
    p_simes = float(np.min(simes_vals))
    p_simes = float(np.clip(p_simes, 0.0, 1.0))

    # Back-compute z from p_simes, signed by the most significant cell
    z_magnitude = float(stats.norm.ppf(1 - p_simes / 2))
    most_sig_idx = sort_idx[0]   # index of smallest p-value
    sign = float(np.sign(z_values[most_sig_idx])) or 1.0
    z_combined = sign * z_magnitude

    return z_combined, p_simes

def partition_bundle_into_blocks(
    ns: int,
    nr: int,
    mask: np.ndarray,
    l_s: float,
    s_len: float,
    l_r: Optional[float] = None,
    r_len: Optional[float] = None,
    min_cells: int = 3,
) -> List[BlockInfo]:
    """
    Partition a bundle grid into RESEL-sized spatial blocks for hFDR Stage 1.
    Blocks are non-overlapping rectangles whose side lengths match the Matern
    correlation length scales (one block ~ one RESEL).

    Parameters
    ----------
    ns, nr : int
        Grid dimensions.
    mask : np.ndarray (ns, nr)
        Boolean valid-cell mask.
    l_s : float
        Matern length scale along s (mm).
    s_len : float
        Grid spacing along s (mm).
    l_r, r_len : float or None
        Length scale and spacing along r. None for 1D bundles.
    min_cells : int
        Minimum valid cells per block; smaller blocks merged into neighbours.

    Returns
    -------
    list of BlockInfo  (block_id zero-based, bundle_name filled by caller)
    """
    is_1d = (nr == 1 or l_r is None or r_len is None)

    # block_size_s = max(2, round(l_s / s_len))
    # block_size_r = 1 if is_1d else max(2, round(l_r / r_len))
    block_size_s = max(2, round(2.355 * l_s / s_len))
    block_size_r = 1 if is_1d else max(2, round(2.355 * l_r / r_len))

    raw_blocks: List[List[Tuple[int, int]]] = []
    for s0 in range(0, ns, block_size_s):
        s1 = min(s0 + block_size_s, ns)
        r_iter = [0] if is_1d else range(0, nr, block_size_r)
        for r0 in r_iter:
            r1 = min(r0 + block_size_r, nr)
            cells = [(s, r) for s in range(s0, s1) for r in range(r0, r1) if mask[s, r]]
            if cells:
                raw_blocks.append(cells)

    if not raw_blocks:
        return []

    # Merge small boundary blocks
    blocks = raw_blocks
    changed = True
    while changed and len(blocks) > 1:
        changed = False
        sizes = [len(b) for b in blocks]
        idx = int(np.argmin(sizes))
        if sizes[idx] >= min_cells:
            break
        candidates = [i for i in [idx - 1, idx + 1] if 0 <= i < len(blocks)]
        target = min(candidates, key=lambda i: sizes[i])
        lo, hi = sorted([idx, target])
        blocks = blocks[:lo] + [blocks[lo] + blocks[hi]] + blocks[hi + 1:]
        changed = True

    result = []
    for bid, cells in enumerate(blocks):
        result.append(BlockInfo(
            bundle_name="",
            block_id=bid,
            cell_indices=cells,
            l_s=l_s,
            l_r=l_r if not is_1d else None,
            s_len=s_len,
            r_len=r_len if not is_1d else None,
        ))

    logging.debug(
        f"partition_bundle_into_blocks: ns={ns}, nr={nr}, "
        f"block_size=({block_size_s},{block_size_r}), "
        f"n_blocks={len(result)}, "
        f"cells=[{min(len(b) for b in blocks)},{max(len(b) for b in blocks)}]"
    )
    return result