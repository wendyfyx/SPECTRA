import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import UnivariateSpline

def fdr_bh_threshold(pvals, alpha=0.05):
    """
    Compute the FDR threshold using Benjamini–Hochberg procedure.

    Args
    ----------
    pvals : list or np.ndarray
        List/array of p-values.
    alpha : float, optional
        Desired FDR control level (default=0.05).

    Returns
    -------
    float
        The FDR threshold. If no p-values pass, returns 0.
    """
    pvals = np.asarray(pvals)
    n = len(pvals)
    sorted_pvals = np.sort(pvals)
    ranks = np.arange(1, n+1)

    # BH critical values
    bh_criteria = ranks * alpha / n

    # Find the largest p-value that meets the BH condition
    below = sorted_pvals <= bh_criteria
    if not np.any(below):
        return 0.0
    else:
        threshold = sorted_pvals[below].max()
        return threshold


def estimate_pi0_storey(
    p_values: np.ndarray,
    lambda_seq: np.ndarray = None,
) -> float:
    """
    Estimate the proportion of true null hypotheses (pi0) using Storey's method.
    
    Parameters
    ----------
    p_values : np.ndarray
        Array of p-values
    lambda_seq : np.ndarray, optional
        Sequence of lambda values for estimation.
    
    Returns
    -------
    float
        Estimated pi0, constrained to (0, 1]
    
    References
    ----------
    Storey JD, Tibshirani R (2003). Statistical significance for genomewide 
    studies. PNAS 100(16):9440-9445.
    """
    p_values = np.asarray(p_values)
    p_values = p_values[~np.isnan(p_values)]
    m = len(p_values)
    
    if m == 0:
        return 1.0
    
    if lambda_seq is None:
        lambda_seq = np.arange(0.05, 0.95, 0.05)
    
    # Estimate pi0 at each lambda
    pi0_estimates = np.array([(p_values > lam).sum() / (m * (1 - lam)) 
                           for lam in lambda_seq])
    
    # Use natural cubic spline smoother and extrapolate to lambda=1
    spline = UnivariateSpline(lambda_seq, pi0_estimates, k=3)
    pi0 = float(spline(1.0))
    
    pi0 = max(np.clip(pi0, 1e-3, 1.0), min(pi0_estimates))

    return float(pi0), pi0_estimates, lambda_seq


def adaptive_weighted_fdr(
    p_values: np.ndarray,
    weights: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    adaptive: bool = False,
    lambda_seq: Optional[np.ndarray] = None,
    use_safeguard: bool = True,
) -> Dict[str, Any]:
    """
    Apply adaptive weighted FDR on a series of p-values.
    
    Parameters
    ----------
    p_values : (m,) array
        Raw p-values.
    weights : (m,) array, optional
        Nonnegative weights. Will be normalized to sum to m.
    alpha : float
        Target FDR level.
    adaptive : bool
        If True, use Storey's pi0 estimation.
    lambda_seq : np.ndarray, optional
        Lambda sequence for pi0 estimation.
    use_safeguard : bool
        Only applicable when adaptive is True, adjusted p-values are capped 
        from below by their corresponding raw p-values, ensuring no hypothesis 
        is rejected whose raw p-value exceeds alpha.

    Returns
    -------
    results : dict
        Contains:
        - p_corr : adjusted p-values
        - p_weighted : weighted p-values
        - pi0 : pi0 estimate
        - effective_alpha : alpha / pi0
        - lambda_seq, pi0_at_lambda (if adaptive)
        - threshold : largest raw p-value rejected at alpha (0 if none)
        - rejected : boolean mask of rejections at alpha
    """
    m = len(p_values)
    p_values = np.asarray(p_values)

    # Handle empty input
    if m == 0:
        return {
            "p_corr": np.array([]),
            "p_weighted": np.array([]),
            "pi0": 1.0,
            "effective_alpha": alpha,
            "lambda_seq": None,
            "pi0_at_lambda": None,
            "threshold": 0.0,
            "rejected": np.array([], dtype=bool),
        }

    # Estimate pi0 from RAW p-values
    if adaptive:
        pi0, pi0_at_lambda, lambda_seq = estimate_pi0_storey(p_values, lambda_seq)
        if pi0 < 0.3:
            logging.warning(
                f"Storey pi0 estimate is very low ({pi0:.3f}), indicating widespread non-null effects. "
                f"Set use_safeguard=True to cap corrected p-values at their raw values"
            )
    else:
        pi0 = 1.0
        pi0_at_lambda = None

    # Weighted p-values
    if weights is None:
        weights = np.ones(m)
    else:
        weights = np.asarray(weights)
        if np.any(weights <= 0):
            raise ValueError("All weights must be strictly positive.")
        weights = weights * (m / np.sum(weights))  # normalize to m tests

    p_weighted = p_values / weights
    p_weighted = np.minimum(p_weighted, 1.0)

    # BH on weighted p-values
    order = np.argsort(p_weighted)
    sorted_p = p_weighted[order]
    ranks = np.arange(1, m + 1)
    adj_sorted = np.minimum(1.0, sorted_p * m * pi0 / ranks)

    # Enforce monotonicity
    for i in range(m - 2, -1, -1):
        adj_sorted[i] = min(adj_sorted[i], adj_sorted[i + 1])

    p_corr = np.empty(m)
    p_corr[order] = adj_sorted

    if adaptive and use_safeguard:
        p_corr = np.maximum(p_corr, p_values)

    # Compute rejection threshold and mask
    rejected = p_corr < alpha
    if rejected.any():
        threshold = float(p_values[rejected].max())
    else:
        threshold = 0.0

    return {
        "p_corr": p_corr,
        "p_weighted": p_weighted,
        "pi0": pi0,
        "effective_alpha": alpha / pi0 if pi0 > 0 else alpha,
        "lambda_seq": lambda_seq,
        "pi0_at_lambda": pi0_at_lambda,
        "threshold": threshold,
        "rejected": rejected,
    }