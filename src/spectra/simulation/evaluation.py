"""
Evaluation metrics for synthetic effect experiments.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Union

# Type hint for result objects (avoid circular import)
try:
    from spectra.tractometry.results_data import MultiBundleResult, MultiBundleHFDRResult
except ImportError:
    MultiBundleResult = Any
    MultiBundleHFDRResult = Any


def mask_metrics(sig_true: np.ndarray, sig_pred: np.ndarray) -> Dict[str, Any]:
    """
    Compute classification metrics for binary significance masks.
    
    Parameters
    ----------
    sig_true : np.ndarray
        Ground truth binary mask
    sig_pred : np.ndarray
        Predicted binary mask
    
    Returns
    -------
    dict
        TP, FP, FN, TN, N, precision, recall, accuracy, f1,
        prop_true_effects, prop_pred_effects
    """
    sig_true = np.asarray(sig_true).astype(bool)
    sig_pred = np.asarray(sig_pred).astype(bool)

    TP = int(np.sum(sig_true & sig_pred))
    FP = int(np.sum(~sig_true & sig_pred))
    FN = int(np.sum(sig_true & ~sig_pred))
    TN = int(np.sum(~sig_true & ~sig_pred))
    N = TP + FP + FN + TN

    precision = TP / (TP + FP) if (TP + FP) > 0 else np.nan
    recall = TP / (TP + FN) if (TP + FN) > 0 else np.nan
    accuracy = (TP + TN) / N if N > 0 else np.nan
    f1 = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else np.nan
    prop_true_effects = (TP + FN) / N if N > 0 else np.nan
    prop_pred_effects = (TP + FP) / N if N > 0 else np.nan

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "N": N,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "f1": f1,
        "prop_true_effects": prop_true_effects,
        "prop_pred_effects": prop_pred_effects,
    }


def effect_size_metrics(
    d_true: np.ndarray,
    d_pred: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute effect size recovery metrics on signal regions.
    
    Parameters
    ----------
    d_true : np.ndarray
        Ground truth Cohen's d values
    d_pred : np.ndarray
        Predicted/estimated Cohen's d values
    mask : np.ndarray, optional
        Boolean mask indicating where to compute metrics.
        If None, uses where d_true != 0 and not NaN.
        This is useful because Cohen's d can be non-zero even where
        no effect was injected (due to sampling variability).
    
    Returns
    -------
    dict
        rmse, bias, pearson_r (computed only within mask)
    """
    d_true = np.asarray(d_true).flatten()
    d_pred = np.asarray(d_pred).flatten()
    
    # Default mask: where true effect exists
    if mask is None:
        mask = (d_true != 0) & ~np.isnan(d_true)
    else:
        mask = np.asarray(mask).astype(bool).flatten()
    
    if not np.any(mask):
        return {
            "rmse": np.nan,
            "bias": np.nan,
            "pearson_r": np.nan,
        }

    dt = d_true[mask]
    dp = d_pred[mask]
    
    # Remove NaN predictions
    valid = ~np.isnan(dp)
    if not np.any(valid):
        return {
            "rmse": np.nan,
            "bias": np.nan,
            "pearson_r": np.nan,
        }
    
    dt = dt[valid]
    dp = dp[valid]

    error = dp - dt

    rmse = float(np.sqrt(np.mean(error**2)))
    bias = float(np.mean(error))

    # Correlation (guard against constant vectors)
    if np.std(dt) > 0 and np.std(dp) > 0:
        pearson_r = float(np.corrcoef(dt, dp)[0, 1])
    else:
        pearson_r = np.nan

    return {
        "rmse": rmse,
        "bias": bias,
        "pearson_r": pearson_r,
    }


def evaluate_synthetic_results(
    result: Union[MultiBundleResult, MultiBundleHFDRResult],
    raw_effect_maps: Dict[str, np.ndarray],
    d_synth_maps: Dict[str, np.ndarray],
) -> 'pd.DataFrame':
    """
    Evaluate cell-level results against the true signals.
    
    Works for both FDR (MultiBundleResult) and hFDR (MultiBundleHFDRResult).
    Handles both 1D and 2D grids correctly.
    
    Parameters
    ----------
    result : MultiBundleResult or MultiBundleHFDRResult
        Analysis results containing grids with sig_mask and d_cohen_obs
    raw_effect_maps : Dict[str, np.ndarray]
        Ground truth effect location maps {bundle: (ns, nr)} with nonzero 
        where effect exists
    d_synth_maps : Dict[str, np.ndarray]
        Ground truth Cohen's d maps {bundle: (ns, nr)} with true effect sizes
    
    Returns
    -------
    pd.DataFrame
        Evaluation metrics per bundle and global, with columns:
        - bundle, n_valid_voxels
        - TP, FP, FN, TN, N, precision, recall, accuracy, f1
        - prop_true_effects, prop_pred_effects
        - rmse, bias, pearson_r
    """
    import pandas as pd
    
    rows = []
    all_mask_true = []
    all_mask_pred = []
    all_d_true = []
    all_d_pred = []
    all_effect_mask = []  # for effect size metrics

    for bundle, grid in result.grids.items():
        if bundle not in d_synth_maps:
            continue

        # Get predicted significance mask - shape (ns, nr)
        mask_pred = grid.sig_mask
        if mask_pred is None:
            mask_pred = np.zeros((grid.ns, grid.nr), dtype=bool)

        # Get true effect mask - ensure shape matches grid (ns, nr)
        effect_map = raw_effect_maps[bundle]
        if effect_map.ndim == 1:
            effect_map = effect_map.reshape(-1, 1)
        mask_true = (effect_map != 0)

        # Get predicted and true Cohen's d - shape (ns, nr)
        d_pred = grid.d_cohen_obs
        if d_pred is None:
            d_pred = np.full((grid.ns, grid.nr), np.nan)

        d_true = d_synth_maps[bundle]
        if d_true.ndim == 1:
            d_true = d_true.reshape(-1, 1)

        # Verify shapes match
        assert mask_pred.shape == mask_true.shape == (grid.ns, grid.nr), \
            f"Shape mismatch for {bundle}: pred={mask_pred.shape}, true={mask_true.shape}, grid=({grid.ns}, {grid.nr})"

        # Bundle metrics (use grid.mask to restrict to valid voxels)
        valid = grid.mask
        
        # Classification metrics
        metrics = mask_metrics(mask_true[valid], mask_pred[valid])
        
        # Effect size metrics - use mask_true (where effect was injected) 
        # to avoid including sampling noise from non-effect regions
        effect_region = mask_true[valid]
        metrics.update(effect_size_metrics(
            d_true[valid], 
            d_pred[valid], 
            mask=effect_region
        ))
        
        metrics['bundle'] = bundle
        metrics['n_valid_voxels'] = int(valid.sum())
        rows.append(metrics)

        # Accumulate valid voxels for global metrics
        all_mask_true.append(mask_true[valid].flatten())
        all_mask_pred.append(mask_pred[valid].flatten())
        all_d_true.append(d_true[valid].flatten())
        all_d_pred.append(d_pred[valid].flatten())
        all_effect_mask.append(effect_region.flatten())

    # Global metrics
    if all_mask_true:
        global_mask_true = np.concatenate(all_mask_true)
        global_mask_pred = np.concatenate(all_mask_pred)
        global_d_true = np.concatenate(all_d_true)
        global_d_pred = np.concatenate(all_d_pred)
        global_effect_mask = np.concatenate(all_effect_mask)

        global_metrics = mask_metrics(global_mask_true, global_mask_pred)
        global_metrics.update(effect_size_metrics(
            global_d_true, 
            global_d_pred,
            mask=global_effect_mask
        ))
        global_metrics['bundle'] = 'GLOBAL'
        global_metrics['n_valid_voxels'] = len(global_mask_true)
        rows.append(global_metrics)

    return pd.DataFrame(rows)


def bundle_detection_metrics(
    affected_bundles: List[str],
    significant_bundles: List[str],
    all_bundles: List[str],
) -> Dict[str, Any]:
    """
    Compute bundle-level detection metrics for hFDR Stage 1.
    
    Parameters
    ----------
    affected_bundles : List[str]
        Bundles that received synthetic effects (ground truth)
    significant_bundles : List[str]
        Bundles detected as significant by Stage 1
    all_bundles : List[str]
        All bundles in the analysis
    
    Returns
    -------
    dict
        Bundle-level TP, FP, FN, TN, precision, recall, f1, accuracy,
        n_affected, n_significant, n_total
    """
    affected_set = set(affected_bundles)
    significant_set = set(significant_bundles)
    
    true_labels = np.array([b in affected_set for b in all_bundles])
    pred_labels = np.array([b in significant_set for b in all_bundles])
    
    metrics = mask_metrics(true_labels, pred_labels)
    
    # Add bundle-specific counts
    metrics['n_affected'] = len(affected_bundles)
    metrics['n_significant'] = len(significant_bundles)
    metrics['n_total'] = len(all_bundles)
    
    return metrics