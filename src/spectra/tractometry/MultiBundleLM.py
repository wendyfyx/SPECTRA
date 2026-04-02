"""
Multi-grid LMM analysis with FDR and hierarchical FDR inference.
"""

import re
import logging
from typing import List, Dict, Optional, Tuple, Union
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

from spectra.math.kernel_fit import build_kernel_matrix
from spectra.tractometry.BundleProfileLoader import BundleProfileLoader
from spectra.tractometry.results_data import BundleResult, BundleResultFull, \
                                         MultiBundleResult, \
                                         MultiBundleHFDRResult, \
                                         compute_pooled_cohen_d
from spectra.detection.resel_stationary import resel_density_from_resid, \
                                           estimate_length_scales, build_masked_kernel_2d
from spectra.detection.resel_nonstationary import rpv_from_resid
from spectra.detection.hierarchical_fdr import (
    compute_conditional_pvalue, stouffer_combine, simes_combine,
    partition_bundle_into_blocks, BlockInfo,
)
from spectra.utils.utils import get_rng, suppress_output

try:
    from pymer4.models import Lmer
except ImportError:
    Lmer = None
    logging.warning("pymer4 not available, LMM fitting will fail")


def _run_hfdr_core(
    bundle_data: Dict[str, dict],
    s_len: float,
    r_len: float,
    alpha_block: float,
    alpha_cell: float,
    use_resel_blocks: bool,
    combination_method: str,
) -> 'MultiBundleHFDRResult':
    """
    Core hierarchical FDR computation (Benjamini & Heller 2007).
 
    Parameters
    ----------
    bundle_data : dict
        Mapping from grid/bundle name to a dict with keys:
            grid          : BundleResult
            valid_mask    : np.ndarray (ns, nr) bool
            cell_z        : np.ndarray (n_valid,) — signed z-scores per valid cell
            cell_p        : np.ndarray (n_valid,) — raw p-values per valid cell
            valid_indices : List[Tuple[int, int]] — (s, r) coords of valid cells
            l_s           : Optional[float] — Matern length scale along s (mm)
            l_r           : Optional[float] — Matern length scale along r (mm)
    s_len : float
        Grid spacing in s direction (mm).
    r_len : float
        Grid spacing in r direction (mm).
    alpha_block : float
        Stage 1 BH-FDR threshold across spatial blocks.
    alpha_cell : float
        Stage 2 BH-FDR threshold within rejected blocks.
    use_resel_blocks : bool
        If True, partition each bundle into RESEL-sized blocks using l_s/l_r.
        Falls back to a single-bundle block when l_s is None.
    combination_method : str
        'simes' or 'stouffer'.
 
    Returns
    -------
    MultiBundleHFDRResult
    """
    from spectra.detection.fdr import adaptive_weighted_fdr
 
    if not bundle_data:
        raise RuntimeError("bundle_data is empty — no grids to process")
 
    # Build Stage 1 blocks
    all_blocks: List[BlockInfo] = []
 
    for bundle, bdata in bundle_data.items():
        grid = bdata['grid']
        valid_indices = bdata['valid_indices']
        cell_z = bdata['cell_z']
        idx_map = {coord: i for i, coord in enumerate(valid_indices)}
 
        if use_resel_blocks and bdata['l_s'] is None:
            logging.warning(
                f"{bundle}: use_resel_blocks=True but l_s is None. "
                f"Falling back to single-bundle block."
            )
 
        if use_resel_blocks and bdata['l_s'] is not None:
            blocks = partition_bundle_into_blocks(
                ns=grid.ns,
                nr=grid.nr,
                mask=bdata['valid_mask'],
                l_s=bdata['l_s'],
                s_len=s_len,
                l_r=bdata['l_r'],
                r_len=r_len if grid.nr > 1 else None,
            )
            if not blocks:
                blocks = [BlockInfo(
                    bundle_name=bundle,
                    block_id=0,
                    cell_indices=valid_indices,
                    l_s=bdata['l_s'],
                    l_r=bdata['l_r'],
                    s_len=s_len,
                    r_len=r_len if grid.nr > 1 else None,
                )]
                logging.warning(
                    f"{bundle}: block partition returned empty list, "
                    f"falling back to single-bundle block"
                )
        else:
            blocks = [BlockInfo(
                bundle_name=bundle,
                block_id=0,
                cell_indices=valid_indices,
            )]
 
        # Build bundle-level 1D kernel matrices for Stouffer (skipped for Simes)
        l_s_val = bdata['l_s']
        l_r_val = bdata['l_r']
        K_s_bundle = None
        K_r_bundle = None
        if combination_method == 'stouffer' and l_s_val is not None:
            K_s_bundle = build_kernel_matrix(grid.ns, s_len, l_s_val, nu=0.5)
            if l_r_val is not None and grid.nr > 1:
                K_r_bundle = build_kernel_matrix(grid.nr, r_len, l_r_val, nu=0.5)
 
        for blk in blocks:
            blk.bundle_name = bundle
            blk.combination_method = combination_method
            block_z_vals = np.array([
                cell_z[idx_map[coord]]
                for coord in blk.cell_indices
                if coord in idx_map
            ])
            block_p_vals = np.array([
                bdata['cell_p'][idx_map[coord]]
                for coord in blk.cell_indices
                if coord in idx_map
            ])
            if len(block_z_vals) == 0:
                blk.z = np.nan
                blk.p = 1.0
                continue
 
            if combination_method == 'simes':
                blk.z, blk.p = simes_combine(block_p_vals, block_z_vals)
            elif K_s_bundle is not None:
                block_mask = np.zeros((grid.ns, grid.nr), dtype=bool)
                for s, r in blk.cell_indices:
                    block_mask[s, r] = True
                K_block, _ = build_masked_kernel_2d(block_mask, K_s_bundle, K_r_bundle)
                blk.set_kernel(K_block)
                blk.z, blk.p = stouffer_combine(block_z_vals, K=blk.K)
            else:
                blk.z, blk.p = stouffer_combine(block_z_vals)
 
        all_blocks.extend(blocks)
 
    # Stage 1 BH-FDR across all blocks
    p_blocks = np.array([blk.p for blk in all_blocks])
    fdr_result = adaptive_weighted_fdr(
        p_values=p_blocks,
        weights=None,
        alpha=alpha_block,
        adaptive=False,
    )
 
    for i, blk in enumerate(all_blocks):
        blk.p_adj = float(fdr_result['p_corr'][i])
        blk.significant = blk.p_adj < alpha_block
 
    rejected_mask = fdr_result['p_corr'] < alpha_block
    if rejected_mask.any():
        u1 = float(p_blocks[rejected_mask].max())
    else:
        u1 = alpha_block
 
    n_blocks_sig = int(rejected_mask.sum())
    logging.info(
        f"Stage 1: {n_blocks_sig}/{len(all_blocks)} blocks rejected "
        f"at u1={u1:.4f} (pi0={fdr_result['pi0']:.3f})"
    )
 
    # Stage 2 conditional p-values for cells in rejected blocks 
    grids: Dict[str, BundleResult] = {}
    for bundle, bdata in bundle_data.items():
        grid = bdata['grid']
        grid.p_corr = np.full((grid.ns, grid.nr), np.nan)
        grid.sig_mask = np.zeros((grid.ns, grid.nr), dtype=bool)
        grids[bundle] = grid
 
    p_cond_by_cell: Dict[Tuple[str, int, int], float] = {}
 
    for blk in all_blocks:
        if not blk.significant:
            continue
 
        bdata = bundle_data[blk.bundle_name]
        valid_indices = bdata['valid_indices']
        cell_z = bdata['cell_z']
        idx_map = {coord: i for i, coord in enumerate(valid_indices)}
 
        for local_idx, coord in enumerate(blk.cell_indices):
            ci = idx_map.get(coord)
            if ci is None:
                continue
            z_cell = cell_z[ci]
            rho = blk.get_rho(local_idx)
            p_cond = compute_conditional_pvalue(z_cell, u1, rho)
            key = (blk.bundle_name, coord[0], coord[1])
            p_cond_by_cell[key] = p_cond
 
    # Compile result and apply Stage 2 BH-FDR
    result = MultiBundleHFDRResult(
        block_results=all_blocks,
        block_threshold=u1,
        grids=grids,
        n_blocks=len(all_blocks),
        n_blocks_sig=n_blocks_sig,
        n_cells_tested=0,
        n_cells_sig=0,
        alpha_block=alpha_block,
        alpha_cell=alpha_cell,
        use_resel_blocks=use_resel_blocks,
        p_cond_by_cell=p_cond_by_cell,
    )
 
    logging.info(f"Stage 2: applying BH-FDR at alpha_cell={alpha_cell}")
    result._apply_stage2_fdr(alpha_cell)
 
    return result


class MultiBundleLM:
    """
    Multi-grid LMM analysis with FDR and hierarchical FDR inference.

    Parameters
    ----------
    dataloader : BundleProfileLoader
        Data loader instance
    s_len : float
        Grid spacing in s direction (mm)
    r_len : float
        Grid spacing in r direction (mm)
    group_col : str
        Column for within-group permutation (e.g., 'site')
    min_subj : int
        Minimum subjects per bin for LMM fitting
    disable_tqdm : bool
        Disable progress bars
    output_dir : str, optional
        Directory for saving intermediate results. If set, creates
        the {output_dir}/observed/ subdirectory.
    target_col, target_ref_group, target_case_group, covar_cols, cont_cols : optional
        Override corresponding dataloader attributes.
        Note these are mainly used for evaluating synthetic effects.
    """

    def __init__(
        self,
        dataloader: BundleProfileLoader,
        s_len: float = 3.0,
        r_len: float = 3.0,
        group_col: Optional[str] = 'site',
        min_subj: int = 20,
        disable_tqdm: bool = False,
        output_dir: Optional[str] = None,
        target_col: Optional[str] = None,           # override parameter
        target_ref_group: Optional[str] = None,     # override parameter
        target_case_group: Optional[str] = None,    # override parameter
        covar_cols: Optional[List[str]] = None,     # override parameter
        cont_cols: Optional[List[str]] = None,      # override parameter
    ):
        self.dataloader = dataloader
        self.s_len = s_len
        self.r_len = r_len
        self.group_col = group_col
        self.min_subj = min_subj
        self.disable_tqdm = disable_tqdm

        # Output directory
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            (self.output_dir / 'observed').mkdir(parents=True, exist_ok=True)

        self.bundles = dataloader.bundles
        self.subj_col = dataloader.subj_col
        self.require_common = dataloader.require_common

        # Override parameters (ONLY use for synthetic effects)
        self.target_col = target_col if target_col is not None else dataloader.target_col
        self.target_ref_group = target_ref_group if target_ref_group is not None else dataloader.target_ref_group
        self.target_case_group = target_case_group if target_case_group is not None else dataloader.target_case_group
        self.covar_cols = covar_cols if covar_cols is not None else dataloader.covar_cols
        self.cont_cols = cont_cols if cont_cols is not None else dataloader.cont_cols

        # Determine if target is binary (for Cohen's d calculation)
        self.target_is_binary = (
            self.target_col not in (self.cont_cols or []) and
            self.target_ref_group is not None
        )

        # Per-bundle subject tracking
        self.subjs_by_bundle: Dict[str, np.ndarray] = {}
        self.n_subj_by_bundle: Dict[str, int] = {}
        self.groups_by_bundle: Dict[str, np.ndarray] = {}

        for bundle in self.bundles:
            bundle_subjs = dataloader.get_subjects_for_bundle(bundle)
            self.subjs_by_bundle[bundle] = bundle_subjs
            self.n_subj_by_bundle[bundle] = len(bundle_subjs)
            self.groups_by_bundle[bundle] = dataloader.get_groups_for_bundle(bundle)

        self.subjs = dataloader.subjects
        self.n_subj = len(self.subjs)

        # Validate minimum subjects
        self._skip_bundles = {
            bundle for bundle, n in self.n_subj_by_bundle.items()
            if n < min_subj
        }
        if self._skip_bundles:
            logging.warning(
                f"{len(self._skip_bundles)} bundle(s) have insufficient subjects "
                f"(min_subj={min_subj}) and will be skipped during fitting: "
                f"{sorted(self._skip_bundles)}"
            )

        # Target variable name for LMM output
        if self.target_ref_group is not None:# and not isinstance(self.target_ref_group, int):
            self.target_var_name = f"{self.target_col}{self.target_case_group}"
        else:
            self.target_var_name = self.target_col

        # Group labels for permutation when using common subjects else, groups_by_bundle is used
        if self.require_common:
            meta = dataloader.get_metadata()
            meta_filtered = meta[meta[self.subj_col].isin(self.subjs)].copy()
            meta_filtered = meta_filtered.set_index(self.subj_col).loc[self.subjs].reset_index()
            if self.group_col and self.group_col in meta_filtered.columns:
                self.groups = meta_filtered[self.group_col].values
            else:
                self.groups = np.zeros(self.n_subj, dtype=int)
        else:
            self.groups = None

        self._build_formulas()
        self.grids: Dict[str, BundleResult] = {}

        logging.info(f"MultiBundleLM initialized: {len(self.bundles)} bundles, "
                    f"{self.n_subj} subjects, target={self.target_var_name}, "
                    f"target_is_binary={self.target_is_binary}")
        if self.target_ref_group is not None:
            meta = dataloader.get_metadata()
            group_counts = Counter(meta.loc[meta[self.subj_col].isin(self.subjs), self.target_col])
            logging.info(f"---Groups for {self.target_col}: {dict(group_counts)}")
        logging.info(f"---Formula: {self.formula}")

    # -------------------------------------------------------------------------
    # Path helpers
    # -------------------------------------------------------------------------

    def _observed_path(self, bundle: str) -> Optional[Path]:
        if self.output_dir is None:
            return None
        return self.output_dir / 'observed' / f'{bundle}.npz'

    # -------------------------------------------------------------------------
    # Saved-result helpers
    # -------------------------------------------------------------------------

    def load_observed_results(
        self,
        paths: Optional[List[str]] = None,
    ) -> Dict[str, 'BundleResult']:
        """
        Load per-bundle observed results from disk.

        Parameters
        ----------
        paths : list of str, optional
            Explicit list of .npz file paths to load.  If None, globs all
            ``*.npz`` files under ``output_dir/observed/``.

        Returns
        -------
        Dict[str, BundleResult]
            Mapping of bundle name → BundleResult.  Also stored in ``self.grids``.
        """
        if paths is None:
            if self.output_dir is None:
                raise ValueError(
                    "output_dir must be set when paths is None"
                )
            paths = [str(p) for p in (self.output_dir / 'observed').glob('*.npz')]

        grids: Dict[str, BundleResult] = {}
        for path in paths:
            grid = BundleResult.load(path)
            grids[grid.name] = grid

        missing = [b for b in self.bundles if b not in grids]
        if missing:
            logging.warning(
                f"load_observed_results: {len(missing)} bundle(s) not found "
                f"in saved files: {missing}"
            )

        self.grids = grids
        logging.info(f"Loaded observed results for {len(grids)} bundles")
        return grids

    def _extract_hfdr_inputs(
        self,
        grid_full_or_summary: Union['BundleResultFull', 'BundleResult'],
    ) -> Tuple['BundleResult', Optional[float], Optional[float]]:
        """
        Unpack either a BundleResultFull or BundleResult into the three
        values needed by fit_hierarchical_fdr.

        Returns
        -------
        (grid_summary, l_s, l_r)
        """
        if isinstance(grid_full_or_summary, BundleResultFull):
            return (
                grid_full_or_summary.to_summary(),
                grid_full_or_summary.l_s,
                grid_full_or_summary.l_r,
            )
        else:
            return (
                grid_full_or_summary,
                grid_full_or_summary.l_s,
                grid_full_or_summary.l_r,
            )

    # -------------------------------------------------------------------------
    # Formula building
    # -------------------------------------------------------------------------

    def _build_formulas(self):
        """Build LMM formulas for full and reduced models."""
        fixed_effects = [self.target_col] + self.covar_cols
        self.formula = f"metric ~ {' + '.join(fixed_effects)}"
        if self.group_col:
            self.formula += f" + (1|{self.group_col})"

        reduced_effects = self.covar_cols if self.covar_cols else ['1']
        self.formula_reduced = f"metric ~ {' + '.join(reduced_effects)}"
        if self.group_col:
            self.formula_reduced += f" + (1|{self.group_col})"

    # -------------------------------------------------------------------------
    # LMM fitting helpers
    # -------------------------------------------------------------------------

    def _get_bin_idx(self, bin_name: str) -> Tuple[int, int]:
        match = re.match(r'bin_s(\d+)_r(\d+)', bin_name)
        if not match:
            logging.warning(f"Could not parse bin name: {bin_name}")
            return None, None
        return int(match.group(1)), int(match.group(2))

    def _fit_lmm(self, df: pd.DataFrame, formula: str):
        if Lmer is None:
            raise ImportError("pymer4 is required for LMM fitting")
        try:
            with suppress_output():
                model = Lmer(formula, data=df)
                model.fit(summarize=False)
                return model
        except Exception as e:
            logging.debug(f"LMM fit failed: {e}")
            return None

    def _extract_lmm_stats(self, model) -> Optional[Dict]:
        """
        Extract statistics from a fitted pymer4 model.

        Returns
        -------
        dict with keys: t, p, beta, se, dof, ci_lower, ci_upper, r2_partial, d_cohen
        or None if target variable not in model
        """
        if self.target_var_name not in model.coefs.index:
            return None

        t_stat = model.coefs.loc[self.target_var_name, 'T-stat']
        dof = model.coefs.loc[self.target_var_name, 'DF']
        beta = model.coefs.loc[self.target_var_name, 'Estimate']
        se = model.coefs.loc[self.target_var_name, 'SE']
        ci_lower = model.coefs.loc[self.target_var_name, '2.5_ci']
        ci_upper = model.coefs.loc[self.target_var_name, '97.5_ci']
        p_val = model.coefs.loc[self.target_var_name, 'P-val']
        r2_partial = t_stat**2 / (t_stat**2 + dof)

        # Compute Cohen's d for binary targets using shared utility
        d_cohen = None
        if self.target_is_binary:
            y = model.data['metric'].values
            group = model.data[self.target_col].values
            try:
                group_coerced = group.astype(type(self.target_case_group))
            except (ValueError, TypeError):
                group_coerced = group
            case_values = y[group_coerced == self.target_case_group]
            control_values = y[group_coerced == self.target_ref_group]
            d_cohen = compute_pooled_cohen_d(case_values, control_values)

        return {
            't': t_stat,
            'p': p_val,
            'beta': beta,
            'se': se,
            'dof': dof,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'r2_partial': r2_partial,
            'd_cohen': d_cohen,
        }

    # -------------------------------------------------------------------------
    # Data preparation helpers
    # -------------------------------------------------------------------------

    def _prepare_bundle_data(
        self,
        bundle: str
    ) -> Tuple[pd.DataFrame, Dict[str, Tuple[int, int]], np.ndarray, int, int, np.ndarray]:
        """
        Load and prepare bundle data for LMM fitting.

        Returns
        -------
        df_long : pd.DataFrame
            Long-format dataframe with standardized metric
        bin_to_idx : dict
            Mapping from bin names to (s, r) indices
        mask : np.ndarray (ns, nr)
            Valid voxel mask
        ns, nr : int
            Grid dimensions
        bundle_subjs : np.ndarray
            Subject IDs for this bundle
        """
        bundle_subjs = self.subjs_by_bundle[bundle]
        group_avg, mask, df = self.dataloader.get_bundle_data(bundle, bundle_subjs)
        ns, nr = mask.shape if mask.ndim == 2 else (mask.shape[0], 1)

        # Standardize continuous columns
        if self.cont_cols:
            for c in self.cont_cols:
                if c in df.columns:
                    df[c] = StandardScaler().fit_transform(df[[c]])

        # Convert to long format
        df_long = df.melt(
            id_vars=[c for c in df.columns if not c.startswith('bin_')],
            var_name='bin',
            value_name='metric'
        )
        df_long['metric'] = StandardScaler().fit_transform(df_long[['metric']])

        bin_cols = [c for c in df.columns if c.startswith('bin_')]
        bin_to_idx = {col: self._get_bin_idx(col) for col in bin_cols}

        return df_long, bin_to_idx, mask, ns, nr, bundle_subjs

    # -------------------------------------------------------------------------
    # Core LMM grid fitting
    # -------------------------------------------------------------------------

    def _fit_lmm_grid(
        self,
        df_long: pd.DataFrame,
        bin_to_idx: Dict[str, Tuple[int, int]],
        mask: np.ndarray,
        ns: int,
        nr: int,
        bundle_subjs: np.ndarray,
        formula: str,
        extract_fitted: bool = False,
        desc: str = "Fitting",
    ) -> Dict[str, np.ndarray]:
        """
        Fit LMM at each grid cell.

        Parameters
        ----------
        df_long : pd.DataFrame
            Long-format data
        bin_to_idx : dict
            Bin name to (s, r) mapping
        mask : np.ndarray
            Valid voxel mask
        ns, nr : int
            Grid dimensions
        bundle_subjs : np.ndarray
            Subject IDs
        formula : str
            LMM formula
        extract_fitted : bool
            If True, also extract fitted values
        desc : str
            Progress bar description

        Returns
        -------
        dict with arrays:
            t, p, beta, se, dof, ci_lower, ci_upper, r2_partial, d_cohen,
            residuals, fitted (if extract_fitted), n_subj_by_bin, fit_mask
        """
        n_subj_bundle = len(bundle_subjs)

        # Initialize output arrays
        t_obs = np.full((ns, nr), np.nan)
        p_obs = np.full((ns, nr), np.nan)
        beta_obs = np.full((ns, nr), np.nan)
        se_obs = np.full((ns, nr), np.nan)
        dof_obs = np.full((ns, nr), np.nan)
        ci_lower_obs = np.full((ns, nr), np.nan)
        ci_upper_obs = np.full((ns, nr), np.nan)
        r2_partial_obs = np.full((ns, nr), np.nan)
        d_cohen_obs = np.full((ns, nr), np.nan) if self.target_is_binary else None
        n_subj_by_bin = np.full((ns, nr), np.nan)

        residuals = np.full((ns, nr, n_subj_bundle), np.nan)
        fitted = np.full((ns, nr, n_subj_bundle), np.nan) if extract_fitted else None

        for bin_name in tqdm(df_long['bin'].unique(), desc=desc, disable=self.disable_tqdm):
            s_idx, r_idx = bin_to_idx.get(bin_name, (None, None))
            if s_idx is None or not mask[s_idx, r_idx]:
                continue

            df_bin = df_long[
                (df_long['bin'] == bin_name) &
                (df_long['metric'] != 0) &
                (df_long['metric'].notna())
            ].copy()
            subj_order = df_bin[self.subj_col].values

            n_subj_by_bin[s_idx, r_idx] = len(df_bin)
            if len(df_bin) < self.min_subj:
                continue

            model = self._fit_lmm(df_bin, formula)
            if model is None:
                continue

            # Extract fitted and residuals - map from model order to bundle_subjs order
            model_fits = np.array(model.fits)
            model_resid = model.residuals
            for i, subj in enumerate(bundle_subjs):
                subj_idx = np.where(subj_order == subj)[0]
                if len(subj_idx) > 0:
                    residuals[s_idx, r_idx, i] = model_resid[subj_idx[0]]
                    if extract_fitted:
                        fitted[s_idx, r_idx, i] = model_fits[subj_idx[0]]

            # Extract statistics
            stats_dict = self._extract_lmm_stats(model)
            if stats_dict is None:
                continue

            t_obs[s_idx, r_idx] = stats_dict['t']
            p_obs[s_idx, r_idx] = stats_dict['p']
            beta_obs[s_idx, r_idx] = stats_dict['beta']
            se_obs[s_idx, r_idx] = stats_dict['se']
            dof_obs[s_idx, r_idx] = stats_dict['dof']
            ci_lower_obs[s_idx, r_idx] = stats_dict['ci_lower']
            ci_upper_obs[s_idx, r_idx] = stats_dict['ci_upper']
            r2_partial_obs[s_idx, r_idx] = stats_dict['r2_partial']
            if self.target_is_binary:
                d_cohen_obs[s_idx, r_idx] = stats_dict['d_cohen']

        # Create fit success mask
        fit_mask = ~np.isnan(p_obs) & ~np.isnan(t_obs)

        result = {
            't': t_obs,
            'p': p_obs,
            'beta': beta_obs,
            'se': se_obs,
            'dof': dof_obs,
            'ci_lower': ci_lower_obs,
            'ci_upper': ci_upper_obs,
            'r2_partial': r2_partial_obs,
            'd_cohen': d_cohen_obs,
            'residuals': residuals,
            'n_subj_by_bin': n_subj_by_bin,
            'fit_mask': fit_mask,
        }
        if extract_fitted:
            result['fitted'] = fitted

        return result

    # -------------------------------------------------------------------------
    # Core fitting
    # -------------------------------------------------------------------------

    def _fit_observed_single_grid(
        self,
        bundle: str,
        fit_reduced: bool = False,
        save: bool = False,
    ) -> Optional[BundleResultFull]:
        """
        Fit observed models for one bundle

        Parameters
        ----------
        bundle : str
            Bundle name
        fit_reduced : bool
            Whether to fit reduced model (needed for hFDR length-scale estimation)
        save : bool
            If True and output_dir is set, save BundleResult to file

        Returns
        -------
        BundleResultFull or None if fitting failed
        """

        if bundle in self._skip_bundles:
            logging.warning(f"Skipping {bundle}: insufficient subjects")
            return None
            
        # 1. Prepare data
        df_long, bin_to_idx, mask, ns, nr, bundle_subjs = self._prepare_bundle_data(bundle)
        n_subj_bundle = len(bundle_subjs)

        if df_long[self.target_col].dropna().nunique() < 2:
            logging.warning(f"No contrast for {bundle} (N={n_subj_bundle}), skipping")
            return None

        # 2. Fit full model
        logging.info(f"Fitting FDR for {bundle} with {n_subj_bundle} subjects ({ns} x {nr})")
        first_pass = self._fit_lmm_grid(
            df_long, bin_to_idx, mask, ns, nr, bundle_subjs,
            formula=self.formula,
            extract_fitted=False,
            desc=f"Fitting {bundle} (full)",
        )

        if np.isnan(first_pass['residuals']).all():
            logging.warning(f"All models failed for {bundle} (N={n_subj_bundle})")
            return None

        # Update mask based on fit success
        mask = mask & first_pass['fit_mask']
        if mask.sum() == 0:
            logging.warning(f"No valid cells after LMM fitting for {bundle}")
            return None

        # 3. Fit reduced model
        fitted_reduced = np.full((ns, nr, n_subj_bundle), np.nan)
        resid_reduced = np.full((ns, nr, n_subj_bundle), np.nan)

        if fit_reduced:
            reduced_pass = self._fit_lmm_grid(
                df_long, bin_to_idx, mask, ns, nr, bundle_subjs,
                formula=self.formula_reduced,
                extract_fitted=True,
                desc=f"Fitting {bundle} (reduced)",
            )
            fitted_reduced = reduced_pass.get('fitted', fitted_reduced)
            resid_reduced = reduced_pass['residuals']

        # Estimate Matern length scales from reduced-model residuals.
        # Used by fit_hierarchical_fdr for RESEL-based block partitioning.
        l_s_est: Optional[float] = None
        l_r_est: Optional[float] = None
        if fit_reduced and not np.isnan(resid_reduced).all():
            r_len_for_est = self.r_len if nr > 1 else None
            try:
                l_s_est, l_r_est = estimate_length_scales(
                    resid_reduced, self.s_len, r_len_for_est
                )
                logging.info(f"Length scale for {bundle}: ls={l_s_est}, lr={l_r_est}")
            except Exception as e:
                logging.warning(f"Length scale estimation failed for {bundle}: {e}")

        # 4. Compute RPV/RESEL from full-model residuals
        rpv = rpv_from_resid(first_pass['residuals'], self.s_len, self.r_len, mask)
        resel_density = resel_density_from_resid(first_pass['residuals'], self.s_len, self.r_len)

        # Filter mask to exclude cells with invalid RPV
        valid_rpv_mask = np.isfinite(rpv) & (rpv > 0)
        mask = mask & valid_rpv_mask
        if mask.sum() == 0:
            logging.warning(f"No valid voxels after RPV validation for {bundle}")
            return None

        # 5. Build result
        result = BundleResultFull(
            name=bundle,
            ns=ns,
            nr=nr,
            mask=mask,
            rpv=rpv,
            resel_density=resel_density,
            t_obs=np.where(mask, first_pass['t'], np.nan),
            p_obs=np.where(mask, first_pass['p'], np.nan),
            beta_obs=np.where(mask, first_pass['beta'], np.nan),
            se_obs=np.where(mask, first_pass['se'], np.nan),
            dof_obs=np.where(mask, first_pass['dof'], np.nan),
            ci_lower_obs=np.where(mask, first_pass['ci_lower'], np.nan),
            ci_upper_obs=np.where(mask, first_pass['ci_upper'], np.nan),
            r2_partial_obs=np.where(mask, first_pass['r2_partial'], np.nan),
            n_subj=n_subj_bundle,
            n_subj_by_bin=first_pass['n_subj_by_bin'],
            d_cohen_obs=np.where(mask, first_pass['d_cohen'], np.nan) if first_pass['d_cohen'] is not None else None,
            fitted_reduced=fitted_reduced,
            resid_reduced=resid_reduced,
            df_long=df_long,
            bin_to_idx=bin_to_idx,
            bundle_subjs=bundle_subjs,
            smoother_report=None,
            smoothed_resid_reduced=None,
            l_s=l_s_est,
            l_r=l_r_est,
        )

        if save and self.output_dir:
            result.to_summary().save(str(self._observed_path(bundle)))
            logging.info(f"Saved observed results: {self._observed_path(bundle)}")

        return result

    # -------------------------------------------------------------------------
    # Main fitting methods
    # -------------------------------------------------------------------------

    def fit_observed_fdr(
        self,
        weight_by: Optional[str] = None,
        alpha: float = 0.05,
        adaptive: bool = False,
        lambda_seq: np.ndarray = None,
        save: bool = False,
        from_saved: bool = False,
    ) -> MultiBundleResult:
        """
        Fit observed models with FDR correction.

        Parameters
        ----------
        weight_by : str, optional
            Weight by 'resel' or 'rpv'
        alpha : float
            FDR significance threshold
        adaptive : bool
            If True, use Storey's adaptive FDR with pi0 estimation
        lambda_seq : np.ndarray, optional
            Lambda sequence for pi0 estimation
        save : bool
            If True, save per-bundle BundleResult to output_dir/observed/.
            Has no effect when from_saved=True (nothing is fitted).
        from_saved : bool
            If True, load pre-computed results from output_dir/observed/
            instead of fitting.  Requires output_dir to be set.

        Returns
        -------
        MultiBundleResult
        """
        if from_saved:
            grids = self.load_observed_results()
            result = MultiBundleResult(grids=grids)
        else:
            for bundle in self.bundles:
                grid = self._fit_observed_single_grid(
                    bundle,
                    fit_reduced=False,
                    save=save,
                )
                if grid is None:
                    logging.warning(f"Skipping {bundle} - fitting failed")
                    continue
                self.grids[bundle] = grid.to_summary()
            result = MultiBundleResult(grids=self.grids)

        result.apply_global_fdr(
            weight_by=weight_by,
            alpha=alpha,
            adaptive=adaptive,
            lambda_seq=lambda_seq,
        )

        if adaptive:
            logging.info(f"Adaptive FDR: pi0={result.pi0_estimate:.3f}, "
                        f"effective_alpha={result.effective_alpha:.4f}")

        self.grids = result.grids
        return result

    def fit_hierarchical_fdr(
        self,
        alpha_block: float = 0.05,
        alpha_cell: float = 0.15,
        use_resel_blocks: bool = True,
        combination_method: str = 'simes',
        save: bool = False,
        from_saved: bool = False,
    ) -> MultiBundleHFDRResult:
        """
        Hierarchical FDR (Benjamini & Heller 2007).
 
        Stage 1:
        Each bundle is divided into spatial blocks whose size matches the Matern
        correlation length scales estimated from the reduced-model residuals.
        When ``use_resel_blocks=False`` the whole bundle is treated as a single
        block. Within each block, individual cell z-scores are aggregated with
        the Stouffer method or Simes' test. BH-FDR is applied across all block-level
        p-values at ``alpha_block``.
 
        Stage 2:
        For each rejected block, conditional p-values are computed for every
        constituent cell (conservative upper-bound formula) and BH-FDR is
        applied within each rejected block at ``alpha_cell``.
 
        Parameters
        ----------
        alpha_block : float
            FDR level for Stage 1 (block-level), default 0.05.
        alpha_cell : float
            FDR level for Stage 2 (cell-level within rejected blocks),
            default 0.15.
        use_resel_blocks : bool
            If True (default), subdivide each bundle into RESEL-sized spatial
            blocks for Stage 1.
            If False, treat each bundle as a single Stage 1 block.
        combination_method : str
            Stage 1 combination method. Either 'stouffer' or 'simes' (default).
        save : bool
            If True and ``output_dir`` is set, save per-bundle observed
            results to disk.  Has no effect when ``from_saved=True``.
        from_saved : bool
            If True, load pre-computed per-bundle results from
            ``output_dir/observed/`` instead of fitting LMMs.  ``save`` is
            ignored in this mode.  Requires ``output_dir`` to be set.
 
        Returns
        -------
        MultiBundleHFDRResult
        """
        if combination_method not in ('stouffer', 'simes'):
            raise ValueError(
                f"combination_method must be 'stouffer' or 'simes', got '{combination_method}'"
            )
 
        if from_saved and save:
            logging.warning(
                "fit_hierarchical_fdr: from_saved=True, so save=True has no effect"
            )
 
        mode_str = "RESEL-block" if use_resel_blocks else "bundle"
        logging.info(
            f"Hierarchical FDR Stage 1: {len(self.bundles)} bundles, "
            f"mode={mode_str}, combination={combination_method}, alpha_block={alpha_block}"
        )
 
        # Pre-load saved grids once if from_saved=True
        if from_saved:
            saved_grids = self.load_observed_results()
 
        # Fit all bundles and build per-bundle cell z-score maps
        bundle_data = {}
 
        for bundle in self.bundles:
            if from_saved:
                if bundle not in saved_grids:
                    logging.warning(f"Skipping {bundle} - not found in saved results")
                    continue
                grid, l_s, l_r = self._extract_hfdr_inputs(saved_grids[bundle])
            else:
                grid_full = self._fit_observed_single_grid(
                    bundle,
                    fit_reduced=True,
                    save=save
                )
                if grid_full is None:
                    logging.warning(f"Skipping {bundle} - fitting failed")
                    continue
                grid, l_s, l_r = self._extract_hfdr_inputs(grid_full)
 
            valid = grid.mask & ~np.isnan(grid.p_obs)
            if not valid.any():
                logging.warning(f"Skipping {bundle} - no valid cells")
                continue
            valid_indices = list(zip(*np.where(valid)))
 
            t_obs = grid.t_obs[valid]
            p_obs = grid.p_obs[valid]
            z_cells = np.sign(t_obs) * stats.norm.ppf(1 - p_obs / 2)
            z_cells = np.clip(z_cells, -10, 10)
 
            bundle_data[bundle] = {
                'grid':          grid,
                'valid_mask':    valid,
                'cell_z':        z_cells,
                'cell_p':        p_obs,
                'valid_indices': valid_indices,
                'l_s':           l_s,
                'l_r':           l_r,
            }
 
        if not bundle_data:
            raise RuntimeError("All bundles failed fitting")
 
        result = _run_hfdr_core(
            bundle_data=bundle_data,
            s_len=self.s_len,
            r_len=self.r_len,
            alpha_block=alpha_block,
            alpha_cell=alpha_cell,
            use_resel_blocks=use_resel_blocks,
            combination_method=combination_method,
        )
 
        self.grids = result.grids
        return result