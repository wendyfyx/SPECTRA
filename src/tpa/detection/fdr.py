import logging
import numpy as np

import numpy as np

def weighted_bh(pvals, weights=None, alpha=0.05):
    """
    Weighted Benjamini–Hochberg FDR procedure.

    Parameters
    ----------
    pvals: array shape (M,)
    weights: positive array shape (M,) normalized so sum(weights)=M.
             If None, standard BH is applied.
             
    Returns
    -------
    rejected: boolean mask of shape (M,)
    qvals: BH-corrected p-values (shape M)
    threshold: BH cutoff p-value
    """
    pvals = np.asarray(pvals)
    M = len(pvals)

    # Compute weighted pvals
    if weights is None:
        weights = np.ones(M)
    else:
        weights = np.asarray(weights)
        assert weights.shape == pvals.shape
        assert np.all(weights > 0)
        weights = weights * (M / np.sum(weights)) # normalize to # tests
    pw = pvals / weights 

    # BH procedure on weighted p
    order = np.argsort(pw)
    pw_sorted = pw[order]
    crit = alpha * np.arange(1, M+1) / M

    # Find largest k such that pw_sorted[k] <= crit[k]
    below = pw_sorted <= crit
    if not np.any(below):
        return np.zeros(M, dtype=bool), np.ones(M), 0.0
    k = np.max(np.nonzero(below))
    threshold = pw_sorted[k]
    rejected = pw <= threshold

    # Compute q-values
    qvals = np.empty(M)
    qvals[order] = np.minimum.accumulate((pw_sorted * M) / np.arange(1, M+1)[~np.isnan(pw_sorted)])

    return rejected, qvals, threshold, weights