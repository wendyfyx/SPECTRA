import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field

from spectra.detection.fdr import adaptive_weighted_fdr
from spectra.utils.utils import safe_float


def compute_pooled_cohen_d(
    case_values: np.ndarray,
    control_values: np.ndarray,
) -> float:
    """
    Compute Cohen's d using pooled standard deviation.
    
    Parameters
    ----------
    case_values : np.ndarray
        Values for the case/treatment group
    control_values : np.ndarray
        Values for the control/reference group
    
    Returns
    -------
    float
        Cohen's d effect size, or np.nan if insufficient data
    """
    case_values = np.asarray(case_values)
    control_values = np.asarray(control_values)
    
    # Remove NaN values
    case_valid = case_values[~np.isnan(case_values)]
    control_valid = control_values[~np.isnan(control_values)]
    
    n_case = len(case_valid)
    n_control = len(control_valid)
    
    if n_case < 2 or n_control < 2:
        return np.nan
    
    mean_diff = np.mean(case_valid) - np.mean(control_valid)
    var_case = np.var(case_valid, ddof=1)
    var_control = np.var(control_valid, ddof=1)
    
    pooled_sd = np.sqrt(
        ((n_case - 1) * var_case + (n_control - 1) * var_control) / 
        (n_case + n_control - 2)
    )
    
    if pooled_sd > 0:
        return mean_diff / pooled_sd
    return np.nan

def _grids_to_stats_df(grids: Dict[str, 'BundleResult'], correction_name: str = 'fdr') -> pd.DataFrame:
    """
    Get voxel-level results as DataFrame from a dict of grids.
    
    Parameters
    ----------
    grids : Dict[str, BundleResult]
        Dictionary mapping bundle names to BundleResult objects
    correction_name : str
        Name for corrected columns ('fdr' or 'fwe')
    
    Returns
    -------
    pd.DataFrame
        Voxel-level statistics
    """
    rows = []
    for bundle, grid in grids.items():
        for s_idx in range(grid.ns):
            for r_idx in range(grid.nr):
                if not grid.mask[s_idx, r_idx]:
                    continue
                row = {
                    'bundle': bundle,
                    's_idx': s_idx,
                    'r_idx': r_idx,
                    'ns': grid.ns,
                    'nr': grid.nr,
                    't_obs': grid.t_obs[s_idx, r_idx],
                    'p_obs': grid.p_obs[s_idx, r_idx],
                    'beta_obs': grid.beta_obs[s_idx, r_idx],
                    'se': grid.se_obs[s_idx, r_idx],
                    'dof': grid.dof_obs[s_idx, r_idx],
                    'ci_lower': grid.ci_lower_obs[s_idx, r_idx],
                    'ci_upper': grid.ci_upper_obs[s_idx, r_idx],
                    'r2_partial': grid.r2_partial_obs[s_idx, r_idx],
                    'd_cohen': grid.d_cohen_obs[s_idx, r_idx] if grid.d_cohen_obs is not None else np.nan,
                    f'p_{correction_name}': grid.p_corr[s_idx, r_idx] if grid.p_corr is not None else np.nan,
                    f'sig_{correction_name}': grid.sig_mask[s_idx, r_idx] if grid.sig_mask is not None else False,
                }
                rows.append(row)
    return pd.DataFrame(rows).sort_values(by=['bundle', 's_idx', 'r_idx'])

def _grids_to_summary_df(grids: Dict[str, 'BundleResult']) -> pd.DataFrame:
    """
    Get per-bundle summary statistics from a dict of grids.
    
    Parameters
    ----------
    grids : Dict[str, BundleResult]
        Dictionary mapping bundle names to BundleResult objects
    
    Returns
    -------
    pd.DataFrame
        Per-bundle summary with a GLOBAL row appended
    """
    rows = []
    all_sig = {'r2': [], 'beta': [], 'd_cohen': []}
    has_d_cohen = any(grid.d_cohen_obs is not None for grid in grids.values())

    def _compute_stats(arr: np.ndarray) -> Tuple[float, float, float]:
        """Compute mean, min, max from array, returning (nan, nan, nan) if empty."""
        valid = arr[~np.isnan(arr)] if arr is not None and len(arr) > 0 else []
        if len(valid) == 0:
            return np.nan, np.nan, np.nan
        return float(np.mean(valid)), float(np.min(valid)), float(np.max(valid))
    
    for bundle, grid in grids.items():
        n_total = int(grid.mask.sum())
        sig_mask = grid.sig_mask if grid.sig_mask is not None else np.zeros_like(grid.mask, dtype=bool)
        n_sig = int(sig_mask.sum())
        
        # Extract significant values
        r2_sig = grid.r2_partial_obs[sig_mask] if n_sig > 0 else np.array([])
        beta_sig = grid.beta_obs[sig_mask] if n_sig > 0 else np.array([])
        d_sig = grid.d_cohen_obs[sig_mask] if n_sig > 0 and grid.d_cohen_obs is not None else np.array([])
        
        # Accumulate for global stats
        all_sig['r2'].extend(r2_sig[~np.isnan(r2_sig)].tolist())
        all_sig['beta'].extend(beta_sig[~np.isnan(beta_sig)].tolist())
        if grid.d_cohen_obs is not None:
            all_sig['d_cohen'].extend(d_sig[~np.isnan(d_sig)].tolist())
        
        r2_mean, r2_min, r2_max = _compute_stats(r2_sig)
        beta_mean, beta_min, beta_max = _compute_stats(beta_sig)
        d_mean, d_min, d_max = _compute_stats(d_sig)
        
        row = {
            'bundle': bundle,
            'ns': grid.ns,
            'nr': grid.nr,
            'n_sig': n_sig,
            'n_total': n_total,
            'prop_sig': n_sig / n_total if n_total > 0 else 0.0,
            'r2_partial_mean': r2_mean, 'r2_partial_min': r2_min, 'r2_partial_max': r2_max,
            'beta_mean': beta_mean, 'beta_min': beta_min, 'beta_max': beta_max,
        }
        if has_d_cohen:
            row.update({'d_cohen_mean': d_mean, 'd_cohen_min': d_min, 'd_cohen_max': d_max})
        rows.append(row)
    
    # Global row
    total_n_sig = sum(r['n_sig'] for r in rows)
    total_n_total = sum(r['n_total'] for r in rows)
    r2_mean, r2_min, r2_max = _compute_stats(np.array(all_sig['r2']))
    beta_mean, beta_min, beta_max = _compute_stats(np.array(all_sig['beta']))
    
    global_row = {
        'bundle': 'GLOBAL',
        'ns': None,
        'nr': None,
        'n_sig': total_n_sig,
        'n_total': total_n_total,
        'prop_sig': total_n_sig / total_n_total if total_n_total > 0 else 0.0,
        'r2_partial_mean': r2_mean, 'r2_partial_min': r2_min, 'r2_partial_max': r2_max,
        'beta_mean': beta_mean, 'beta_min': beta_min, 'beta_max': beta_max,
    }
    if has_d_cohen:
        d_mean, d_min, d_max = _compute_stats(np.array(all_sig['d_cohen']))
        global_row.update({'d_cohen_mean': d_mean, 'd_cohen_min': d_min, 'd_cohen_max': d_max})
    
    rows.append(global_row)
    return pd.DataFrame(rows).sort_values(by='bundle')
    

def get_sig_map(df, bundle: str, is_hfdr: bool = True) -> dict:
    """
    Build spatial significance grids from get_stats_all() output.
    Invalid (masked) cells are set to np.nan

    Parameters
    ----------
    df : pd.DataFrame
        Output of MultiBundleHFDRResult.get_stats_all() or
        MultiBundleResult.get_stats_all().
    bundle : str
        Bundle name to extract.
    is_hfdr : bool
        If True (default), expects hFDR columns: block_id, block_sig, sig_hfdr.
        If False, expects FDR column: sig_fdr.

    Returns
    -------
    dict with keys:
        ns, nr : int — grid dimensions
        is_hfdr : bool
        For is_hfdr=True:
            block_grid       : np.ndarray (ns, nr), float — block IDs; NaN outside mask
            sig_grid_block   : np.ndarray (ns, nr), float — 0/1/NaN
            sig_grid_cell    : np.ndarray (ns, nr), float — 0/1/NaN
        For is_hfdr=False:
            sig_grid_cell    : np.ndarray (ns, nr), float — 0/1/NaN
    """
    bdf = df[df['bundle'] == bundle]
    if bdf.empty:
        raise ValueError(f"No data for bundle '{bundle}' in DataFrame")

    ns = int(bdf['ns'].iloc[0])
    nr = int(bdf['nr'].iloc[0])

    result = {'ns': ns, 'nr': nr, 'is_hfdr': is_hfdr}

    if is_hfdr:
        block_grid     = np.full((ns, nr), np.nan)
        sig_grid_block = np.full((ns, nr), np.nan)
        sig_grid_cell  = np.full((ns, nr), np.nan)

        for _, row in bdf.iterrows():
            s, r = int(row['s_idx']), int(row['r_idx'])
            block_grid[s, r]     = float(row['block_id'])
            sig_grid_block[s, r] = float(row['block_sig'])
            sig_grid_cell[s, r]  = float(row['sig_hfdr'])

        result['block_grid']     = block_grid
        result['sig_grid_block'] = sig_grid_block
        result['sig_grid_cell']  = sig_grid_cell
    else:
        sig_grid_cell = np.full((ns, nr), np.nan)
        for _, row in bdf.iterrows():
            s, r = int(row['s_idx']), int(row['r_idx'])
            sig_grid_cell[s, r] = float(row['sig_fdr'])
        result['sig_grid_cell'] = sig_grid_cell

    return result


def get_stat_map(
    df,
    bundle: str,
    stat_col: str = 'd_cohen',
    sig_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Build a spatial stat grid from get_stats_all() output.

    Parameters
    ----------
    df : pd.DataFrame
        Output of get_stats_all() (FDR or hFDR).
    bundle : str
        Bundle name to extract.
    stat_col : str
        Column to use as values, e.g. 'd_cohen', 'beta_obs', 'p_obs'.
    sig_mask : np.ndarray (ns, nr), optional
        Boolean or 0/1 float array. When provided, cells where sig_mask is
        False/0 are set to np.nan. Invalid (masked) cells are always np.nan
        regardless. If None, returns values for all valid cells.

    Returns
    -------
    stat_grid : np.ndarray (ns, nr)
        NaN for invalid cells; NaN for non-sig cells if sig_mask provided;
        full stat values otherwise.
    """
    bdf = df[df['bundle'] == bundle]
    if bdf.empty:
        raise ValueError(f"No data for bundle '{bundle}' in DataFrame")
    if stat_col not in bdf.columns:
        raise ValueError(f"Column '{stat_col}' not found. Available: {list(bdf.columns)}")

    ns = int(bdf['ns'].iloc[0])
    nr = int(bdf['nr'].iloc[0])
    stat_grid = np.full((ns, nr), np.nan)

    for _, row in bdf.iterrows():
        s, r = int(row['s_idx']), int(row['r_idx'])
        stat_grid[s, r] = float(row[stat_col])

    if sig_mask is not None:
        sig_bool = np.asarray(sig_mask, dtype=bool)
        stat_grid[~sig_bool] = np.nan

    return stat_grid


@dataclass
class BundleResult:
    """Observed results for a single bundle."""
    name: str
    ns: int
    nr: int
    mask: np.ndarray                # (ns, nr)
    rpv: np.ndarray                 # (ns, nr)
    resel_density: float
    t_obs: np.ndarray               # (ns, nr) t-statistics
    p_obs: np.ndarray               # (ns, nr) uncorrected p-values
    beta_obs: np.ndarray            # (ns, nr) coefficients
    se_obs: np.ndarray              # (ns, nr) standard error
    dof_obs: np.ndarray             # (ns, nr) degrees of freedom
    ci_lower_obs: np.ndarray        # (ns, nr) lower 95% CI
    ci_upper_obs: np.ndarray        # (ns, nr) upper 95% CI
    r2_partial_obs: np.ndarray      # (ns, nr) partial R^2 from t-statistic
    n_subj: int                     # number of valid subject for the  bundle
    n_subj_by_bin: np.ndarray       # (ns, nr) number of valid subjects for the bundle bin
    d_cohen_obs: np.ndarray = None  # (ns, nr) Cohen's d (None if continuous target)
    p_corr: np.ndarray = None       # (ns, nr) corrected p-values
    sig_mask: np.ndarray = None     # (ns, nr) mask of significant results
    l_s: Optional[float] = None     # Matern length scale along s (mm); None if not estimated
    l_r: Optional[float] = None     # Matern length scale along r (mm); None for 1D or not estimated
    
    def save(self, path: str) -> None:
        """Save to .npz file."""
        np.savez_compressed(
            path,
            name=self.name,
            ns=self.ns,
            nr=self.nr,
            mask=self.mask,
            rpv=self.rpv,
            resel_density=self.resel_density,
            t_obs=self.t_obs,
            p_obs=self.p_obs,
            beta_obs=self.beta_obs,
            se_obs=self.se_obs,
            dof_obs=self.dof_obs,
            ci_lower_obs=self.ci_lower_obs,
            ci_upper_obs=self.ci_upper_obs,
            r2_partial_obs=self.r2_partial_obs,
            n_subj=self.n_subj,
            n_subj_by_bin=self.n_subj_by_bin,
            d_cohen_obs=self.d_cohen_obs if self.d_cohen_obs is not None else np.array([]),
            p_corr=self.p_corr,
            sig_mask=self.sig_mask,
            l_s=self.l_s if self.l_s is not None else np.nan,
            l_r=self.l_r if self.l_r is not None else np.nan,
        )
    
    @classmethod
    def load(cls, path: str) -> 'BundleResult':
        """Load from .npz file."""
        data = np.load(path, allow_pickle=True)
        
        # Handle optional arrays that may be None
        p_corr = data['p_corr']
        if p_corr.ndim == 0:
            p_corr = None
        sig_mask = data['sig_mask']
        if sig_mask.ndim == 0:
            sig_mask = None
        d_cohen_obs = data['d_cohen_obs']
        if d_cohen_obs.size == 0:
            d_cohen_obs = None

        l_s = safe_float(data['l_s']) if 'l_s' in data else None
        l_r = safe_float(data['l_r']) if 'l_r' in data else None
            
        return cls(
            name=str(data['name']),
            ns=int(data['ns']),
            nr=int(data['nr']),
            mask=data['mask'],
            rpv=data['rpv'],
            resel_density=float(data['resel_density']),
            t_obs=data['t_obs'],
            p_obs=data['p_obs'],
            beta_obs=data['beta_obs'],
            se_obs=data['se_obs'],
            dof_obs=data['dof_obs'],
            ci_lower_obs=data['ci_lower_obs'],
            ci_upper_obs=data['ci_upper_obs'],
            r2_partial_obs=data['r2_partial_obs'],
            n_subj=int(data['n_subj']),
            n_subj_by_bin=data['n_subj_by_bin'],
            d_cohen_obs=d_cohen_obs,
            p_corr=p_corr,
            sig_mask=sig_mask,
            l_s=l_s,
            l_r=l_r,
        )
    
    @classmethod
    def from_arrays(
        cls,
        name: str,
        mask: np.ndarray,
        rpv: np.ndarray,
        resel_density: float,
        results: Dict[str, np.ndarray],
        n_subj: int,
        n_subj_by_bin: np.ndarray,
    ) -> 'BundleResult':
        """
        Create BundleResult from fitting results, applying mask to all arrays.
        
        Parameters
        ----------
        name : str
            Bundle name
        mask : np.ndarray
            Boolean mask of valid voxels, shape (ns, nr)
        rpv : np.ndarray
            RESEL per voxel, shape (ns, nr)
        resel_density : float
            Global RESEL density
        results : dict
            Dict with keys: 't', 'p', 'beta', 'se', 'dof', 'ci_lower', 
            'ci_upper', 'r2_partial', and optionally 'd_cohen'
        n_subj : int
            Number of subjects
        n_subj_by_bin : np.ndarray
            Subjects per bin, shape (ns, nr)
        
        Returns
        -------
        BundleResult
        """
        ns, nr = mask.shape
        
        def _masked(arr):
            """Apply mask: NaN where mask is False."""
            if arr is None:
                return None
            return np.where(mask, arr, np.nan)
        
        return cls(
            name=name,
            ns=ns,
            nr=nr,
            mask=mask,
            rpv=rpv,
            resel_density=resel_density,
            t_obs=_masked(results['t']),
            p_obs=_masked(results['p']),
            beta_obs=_masked(results['beta']),
            se_obs=_masked(results['se']),
            dof_obs=_masked(results['dof']),
            ci_lower_obs=_masked(results['ci_lower']),
            ci_upper_obs=_masked(results['ci_upper']),
            r2_partial_obs=_masked(results['r2_partial']),
            n_subj=n_subj,
            n_subj_by_bin=n_subj_by_bin,
            d_cohen_obs=_masked(results.get('d_cohen')),
        )


@dataclass
class BundleResultFull:
    """Full bundle data during fitting (freed after processing)."""
    name: str
    ns: int
    nr: int
    mask: np.ndarray              # (ns, nr)
    rpv: np.ndarray               # (ns, nr)
    resel_density: float
    t_obs: np.ndarray             # (ns, nr)
    p_obs: np.ndarray             # (ns, nr)
    beta_obs: np.ndarray          # (ns, nr)
    se_obs: np.ndarray            # (ns, nr)
    dof_obs: np.ndarray           # (ns, nr)
    ci_lower_obs: np.ndarray      # (ns, nr)
    ci_upper_obs: np.ndarray      # (ns, nr)
    r2_partial_obs: np.ndarray    # (ns, nr)
    n_subj: int
    n_subj_by_bin: np.ndarray     # (ns, nr)
    d_cohen_obs: np.ndarray       # (ns, nr) or None
    
    # Freedman-Lane storage (large arrays)
    fitted_reduced: np.ndarray    # (ns, nr, n_subj)
    resid_reduced: np.ndarray     # (ns, nr, n_subj)
    
    # For LMM fitting
    df_long: pd.DataFrame
    bin_to_idx: Dict[str, Tuple[int, int]]
    
    # Bundle-specific subjects
    bundle_subjs: np.ndarray
    
    # Optional smoother diagnostics (when residual smoothing is applied)
    smoother_report: Optional[Dict[str, Any]] = None
    
    # Precomputed smoothed reduced residuals for efficient permutation (when smooth=True)
    # Shape: (ns, nr, n_subj) - permutation just reorders the subject axis
    smoothed_resid_reduced: Optional[np.ndarray] = None

    # Matern length scales estimated from reduced-model residuals (set when
    # fit_reduced=True).  Used by fit_hierarchical_fdr for block partitioning.
    l_s: Optional[float] = None   # length scale along s (mm)
    l_r: Optional[float] = None   # length scale along r (mm); None for 1D
    
    def to_summary(self) -> BundleResult:
        """Convert to lightweight summary."""
        return BundleResult(
            name=self.name,
            ns=self.ns,
            nr=self.nr,
            mask=self.mask,
            rpv=self.rpv,
            resel_density=self.resel_density,
            t_obs=self.t_obs,
            p_obs=self.p_obs,
            beta_obs=self.beta_obs,
            se_obs=self.se_obs,
            dof_obs=self.dof_obs,
            ci_lower_obs=self.ci_lower_obs,
            ci_upper_obs=self.ci_upper_obs,
            r2_partial_obs=self.r2_partial_obs,
            n_subj=self.n_subj,
            n_subj_by_bin=self.n_subj_by_bin,
            d_cohen_obs=self.d_cohen_obs,
            l_s=self.l_s,
            l_r=self.l_r,
        )


@dataclass
class MultiBundleResult:
    """Aggregated observed results across multiple bundles with FDR correction."""
    grids: Dict[str, BundleResult]
    alpha: float = None                 # alpha threshold
    weight_by: str = None               # weighting scheme (if RESEL or RPV is used)
    n_total_tests: int = None           # total number of tests
    n_significant: int = None           # number of significant tests
    p_raw: np.ndarray = None            # aggregated uncorrected unweighted p-values
    p_weights: np.ndarray = None        # p-values weights
    p_weighted: np.ndarray = None       # aggregated uncorrected weighted p-values
    p_corr: np.ndarray = None           # corrected p-values (q-values)
    adaptive: bool = False              # whether adaptive FDR was used
    pi0_estimate: float = None          # (adaptive mode) pi0 estimate from Storey
    effective_alpha: float = None       # (adaptive mode) effect alpha threshold weighted by pi0
    lambda_seq: np.ndarray = None       # (adaptive mode) lambda values used for Storey
    pi0_at_lambda: np.ndarray = None    # (adaptive mode) pi0 estimate at each lambda
    
    @property
    def bundles(self) -> List[str]:
        return list(self.grids.keys())
    
    @property
    def n_bundles(self) -> int:
        return len(self.grids)
    
    def apply_global_fdr(
        self, 
        weight_by: Optional[str] = None, 
        alpha: float = 0.05,
        adaptive: bool = False,
        lambda_seq: np.ndarray = None,
    ) -> None:
        """
        Apply global FDR correction across all grids.
        
        Parameters
        ----------
        weight_by : str, optional
            Weight by 'resel' or 'rpv'
        alpha : float
            FDR significance threshold
        adaptive : bool
            If True, use Storey's adaptive FDR with pi0 estimation
        lambda_seq : np.ndarray, optional
            Lambda sequence for pi0 estimation.
        """
        self.alpha = alpha
        self.weight_by = weight_by
        self.adaptive = adaptive
        
        p_values = []
        w_values = []
        locations = []
        
        for grid_name, grid in self.grids.items():
            for s_idx in range(grid.ns):
                for r_idx in range(grid.nr):
                    p = grid.p_obs[s_idx, r_idx]
                    if not np.isnan(p) and grid.mask[s_idx, r_idx]:
                        p_values.append(p)
                        if weight_by is None:
                            w_values.append(1.0)
                        elif weight_by.lower() == 'resel':
                            w_values.append(grid.resel_density)
                        elif weight_by.lower() == 'rpv':
                            w_values.append(grid.rpv[s_idx, r_idx])
                        else:
                            w_values.append(1.0)
                        locations.append((grid_name, s_idx, r_idx))
        
        # Initialize output arrays
        for grid in self.grids.values():
            grid.p_corr = np.full((grid.ns, grid.nr), np.nan)
            grid.sig_mask = np.zeros((grid.ns, grid.nr), dtype=bool)
        
        if len(p_values) == 0:
            self.n_total_tests = 0
            self.n_significant = 0
            self.pi0_estimate = 1.0
            self.effective_alpha = alpha
            return
        
        p_values = np.array(p_values)
        w_values = np.array(w_values)

        results = adaptive_weighted_fdr(
            p_values=p_values,
            weights=w_values,
            alpha=alpha,
            adaptive=adaptive,
            lambda_seq=lambda_seq,
        )

        p_corr_flat = results["p_corr"]
        self.p_raw = p_values
        self.p_weights = w_values
        self.p_weighted = results["p_weighted"]
        self.p_corr = p_corr_flat
        self.pi0_estimate = results["pi0"]
        self.effective_alpha = results["effective_alpha"]
        self.lambda_seq = results["lambda_seq"]
        self.pi0_at_lambda = results["pi0_at_lambda"]
        
        for i, (grid_name, s_idx, r_idx) in enumerate(locations):
            grid = self.grids[grid_name]
            grid.p_corr[s_idx, r_idx] = p_corr_flat[i]
            grid.sig_mask[s_idx, r_idx] = p_corr_flat[i] < alpha
        
        self.n_total_tests = len(p_values)
        self.n_significant = sum(g.sig_mask.sum() for g in self.grids.values())
        logging.info(f"{self.n_significant}/{self.n_total_tests} cells significant")
    
    def get_stats_all(self, save_path: Optional[str] = None) -> pd.DataFrame:
        """Get cell-level results as DataFrame."""
        df = _grids_to_stats_df(self.grids, correction_name='fdr')
        if save_path:
            df.to_csv(save_path, index=False)
        return df
    
    def get_stats_summary(self, save_path: Optional[str] = None) -> pd.DataFrame:
        """Get per-bundle summary statistics with adaptive FDR info."""
        df = _grids_to_summary_df(self.grids)
        if save_path:
            df.to_csv(save_path, index=False)
        return df

    def save_fdr_diagnostics(self, save_path: str) -> None:
        """Save adaptive FDR diagnostics to npz."""
        np.savez_compressed(
            save_path,
            alpha=self.alpha,
            p_raw=self.p_raw,
            p_weighted=self.p_weighted,
            p_weights=self.p_weights,
            lambda_seq=self.lambda_seq,
            pi0_at_lambda=self.pi0_at_lambda,
            pi0_estimate=self.pi0_estimate,
            effective_alpha=self.effective_alpha,
        )
    
    @classmethod
    def from_files(cls, paths: List[str]) -> 'MultiBundleResult':
        """Load from multiple BundleResult .npz files."""
        grids = {}
        for path in paths:
            grid = BundleResult.load(path)
            grids[grid.name] = grid
        return cls(grids=grids)


@dataclass
class MultiBundleHFDRResult:
    """
    Results from hierarchical FDR (Benjamini-Heller two-stage procedure).

    Stage 1 tests spatial clusters (either one per bundle, or RESEL-based blocks
    within each bundle) using BH-FDR on Stouffer-aggregated z-scores.
    Stage 2 tests cells within significant clusters using conditional p-values.
    """
    # block level results 
    block_results: List[Any]            # List[BlockInfo]
    block_threshold: float              # u1: max rejected raw p-value at Stage 1

    # Cell level results 
    grids: Dict[str, 'BundleResult']

    # Counts
    n_blocks: int                       # Total # blocks
    n_blocks_sig: int                   # Total # blocks significant in Stage 1
    n_cells_tested: int                 # Total # cells in Stage 2
    n_cells_sig: int                    # Total # cells significant in Stage 2

    # Parameters
    alpha_block: float
    alpha_cell: float
    use_resel_blocks: bool              # True -> RESEL mode, False -> bundle mode

    # Raw conditional p-values from Stage 2, before thresholding. Key is (bundle, s, r).
    # Cells not in any rejected block have no entry.
    p_cond_by_cell: Dict[Tuple[str, int, int], float] = field(default_factory=dict)

    @property
    def significant_blocks(self) -> List[Any]:
        """BlockInfo objects rejected at Stage 1."""
        return [b for b in self.block_results if b.significant]

    @property
    def significant_bundles(self) -> List[str]:
        """
        Names of bundles that contain at least one significant block.
        """
        return list({b.bundle_name for b in self.block_results if b.significant})

    @property
    def bundles(self) -> List[str]:
        """All bundle names (in grid order)."""
        return list(self.grids.keys())

    @property
    def n_total_tests(self) -> int:
        """Total cells tested in Stage 2 (for compatibility)."""
        return self.n_cells_tested

    @property
    def n_significant(self) -> int:
        """Total significant cells (for compatibility)."""
        return self.n_cells_sig

    def _build_cell_block_lookup(self) -> Dict[Tuple[str, int, int], Tuple[int, bool]]:
        """
        Build a lookup from (bundle, s, r) -> (block_id, block_significant).
        In bundle mode every cell maps to block_id=0.
        """
        lookup: Dict[Tuple[str, int, int], Tuple[int, bool]] = {}
        for blk in self.block_results:
            for s, r in blk.cell_indices:
                lookup[(blk.bundle_name, s, r)] = (blk.block_id, blk.significant)
        return lookup
    
    def _apply_stage2_fdr(self, alpha_cell: float) -> None:
        """
        Apply BH-FDR within each rejected Stage 1 block at the given alpha_cell.

        Resets p_corr and sig_mask on all grids, then for each significant block
        collects pre-computed conditional p-values from p_cond_by_cell, runs
        BH-FDR, and writes results back to the grids. Also updates alpha_cell,
        n_cells_sig, and n_cells_tested on self.

        Called internally by fit_hierarchical_fdr and publicly by reapply_stage2.
        """
        # Reset all grids
        for grid in self.grids.values():
            grid.p_corr = np.full((grid.ns, grid.nr), np.nan)
            grid.sig_mask = np.zeros((grid.ns, grid.nr), dtype=bool)

        self.alpha_cell = alpha_cell
        n_cells_tested = 0
        n_cells_sig = 0

        for blk in self.block_results:
            if not blk.significant:
                continue

            grid = self.grids.get(blk.bundle_name)
            if grid is None:
                continue

            # Collect pre-computed conditional p-values for cells in this block
            p_cond_list = []
            block_coords = []
            for coord in blk.cell_indices:
                key = (blk.bundle_name, coord[0], coord[1])
                if key in self.p_cond_by_cell:
                    p_cond_list.append(self.p_cond_by_cell[key])
                    block_coords.append(coord)

            if not p_cond_list:
                continue

            p_cond_array = np.array(p_cond_list)
            n_cells_tested += len(p_cond_array)

            cell_fdr = adaptive_weighted_fdr(
                p_values=p_cond_array,
                weights=None,
                alpha=alpha_cell,
                adaptive=False,
            )

            for idx, (s, r) in enumerate(block_coords):
                grid.p_corr[s, r] = cell_fdr['p_corr'][idx]
                grid.sig_mask[s, r] = cell_fdr['p_corr'][idx] < alpha_cell

            n_cells_sig += int(sum(cell_fdr['p_corr'] < alpha_cell))

        self.n_cells_tested = n_cells_tested
        self.n_cells_sig = n_cells_sig
        logging.info(
            f"Stage 2 (alpha_cell={alpha_cell}): "
            f"{n_cells_sig}/{n_cells_tested} cells significant"
        )

    def reapply_stage2(self, alpha_cell: float) -> None:
        """
        Re-apply Stage 2 BH-FDR at a new alpha_cell without re-fitting LMMs.
        Stage 1 block results and conditional p-values are unchanged. Only the
        thresholding within rejected blocks is re-computed. 

        Parameters
        ----------
        alpha_cell : float
            New Stage 2 significance threshold (e.g. 0.05, 0.10, 0.25, 0.50).
        """
        # p_cond_by_cell is empty when Stage 1 rejected no blocks; Stage 2 then
        # has nothing to test and no cell is significant at any alpha_cell.
        self._apply_stage2_fdr(alpha_cell)

    def get_stats_blocks(self, save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Block-level summary table (one row per Stage 1 hypothesis).

        Columns:
        - bundle, block_id:  block identity
        - n_cells:           number of cells in block
        - z, p, p_adj:       Stage 1 combined statistic, raw and BH-corrected p-values
        - significant :      Stage 1 rejection at alpha_block
        """
        rows = []
        for blk in self.block_results:
            rows.append({
                'bundle':      blk.bundle_name,
                'block_id':    blk.block_id,
                'n_cells':     blk.n_cells,
                'z':           blk.z,
                'p':           blk.p,
                'p_adj':       blk.p_adj,
                'significant': blk.significant,
            })
        df = pd.DataFrame(rows).sort_values(['bundle','block_id']).reset_index(drop=True)
        if save_path:
            df.to_csv(save_path, index=False)
        return df

    def get_stats_all(self, save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Cell-level results for ALL bundles, including block assignment.

        One row per valid masked cell across all bundles. Columns:
        - bundle, s_idx, r_idx, ns, nr: meta info
        - t_obs, p_obs, beta_obs, se, dof, ci_lower, ci_upper,
          r2_partial, d_cohen:  per-cell LMM statistics
        - p_cond:               raw Stage 2 conditional p-value
                                (NaN for cells in non-rejected Stage 1 blocks)
        - p_hfdr, sig_hfdr:     Stage 2 BH-corrected p-value and significance
                                (NaN/False for cells in non-rejected Stage 1 blocks)
        - block_id:             Stage 1 block index within bundle (0 in bundle mode)
        - block_sig:            whether the cell's Stage 1 block was rejected
        """
        cell_block = self._build_cell_block_lookup()
        df = _grids_to_stats_df(self.grids, correction_name='hfdr')

        block_ids = []
        block_sigs = []
        p_cond_vals = []
        for _, row in df.iterrows():
            key = (row['bundle'], int(row['s_idx']), int(row['r_idx']))
            bid, bsig = cell_block.get(key, (np.nan, False))
            block_ids.append(bid)
            block_sigs.append(bsig)
            p_cond_vals.append(self.p_cond_by_cell.get(key, np.nan))

        sig_hfdr_col = df.pop('sig_hfdr')
        p_hfdr_col = df.pop('p_hfdr')
        df['p_cond'] = p_cond_vals
        df['p_hfdr'] = p_hfdr_col
        df['sig_hfdr'] = sig_hfdr_col
        df['block_id'] = block_ids
        df['block_sig'] = block_sigs
        df = df.sort_values(['bundle', 's_idx', 'r_idx'])

        if save_path:
            df.to_csv(save_path, index=False)
        return df

    def get_stats_summary(self, save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Per-bundle summary statistics merged from cell-level and block-level results.

        One row per bundle plus a GLOBAL row. Columns:

        From cell-level aggregation (significant cells only):
        - bundle, ns, nr n_sig, n_total, prop_sig
        - r2_partial_{mean/min/max}, beta_{mean/min/max}
        - d_cohen_{mean/min/max}:   only when target is binary

        From block-level aggregation:
        - n_blocks, n_blocks_sig:   Stage 1 block counts
        - min_p_block:              smallest raw Stage 1 p-value in bundle
        - sig_stage1:               True if any block was rejected
        - l_s, l_r:                 Matern length scales (NaN if not estimated)

        The GLOBAL row aggregates cell-level columns across all bundles; block
        columns are NaN for GLOBAL (block structure is per-bundle only).
        """
        from collections import defaultdict

        # cell-level part (includes GLOBAL row)
        df_cells = _grids_to_summary_df(self.grids)

        # block-level part (one row per bundle, no GLOBAL)
        agg: Dict[str, dict] = defaultdict(lambda: {
            'n_blocks': 0, 'n_blocks_sig': 0, 'min_p_block': 1.0,
        })
        for blk in self.block_results:
            b = blk.bundle_name
            agg[b]['n_blocks'] += 1
            agg[b]['n_blocks_sig'] += int(blk.significant)
            agg[b]['min_p_block'] = min(agg[b]['min_p_block'], blk.p)
        for b, grid in self.grids.items():
            agg[b]['l_s'] = grid.l_s
            agg[b]['l_r'] = grid.l_r

        block_rows = []
        for b, d in agg.items():
            block_rows.append({
                'bundle':        b,
                'n_blocks':      d['n_blocks'],
                'n_blocks_sig':  d['n_blocks_sig'],
                'min_p_block':   d['min_p_block'],
                'sig_stage1':    d['n_blocks_sig'] > 0,
                'l_s':           d.get('l_s'),
                'l_r':           d.get('l_r'),
            })
        df_blocks = pd.DataFrame(block_rows)

        # Left-merge: GLOBAL row gets NaN for all block columns (correct)
        df = df_cells.merge(df_blocks, on='bundle', how='left')
        df = df.sort_values('bundle').reset_index(drop=True)

        if save_path:
            df.to_csv(save_path, index=False)
        return df