import logging
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

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

def weighted_fdr_bh(pvals, weights=None, alpha=0.05):
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
        # assert np.all(weights >= 0)
        weights = weights * (M / np.sum(weights)) # normalize to # tests
    pw = pvals / weights

    threshold = fdr_bh_threshold(pw, alpha)
    rejected, qvals, _, _ = multipletests(pw, alpha=alpha, method='fdr_bh')
    return rejected, qvals, threshold, weights


def compute_lambda_gc(z_scores):
    """
    Compute genomic control inflation factor.
    """
    z_scores = np.asarray(z_scores).ravel()
    z_scores = z_scores[np.isfinite(z_scores)]
    
    # chi2_median = stats.chi2.ppf(0.5, df=1)  # 0.4549
    observed_median = np.median(z_scores**2)
    lambda_gc = observed_median / 0.455
    
    return lambda_gc


def fdr_multi(
        data_dict,
        alpha=0.05,
        weight_by="resel_density"
    ):
    """
    data_dict: dict[name] = {
        'pvals': 2D array (ns,nr),
        'resel_density': float,
        'rpv': 2D array (ns,nr),
    }

    Returns:
        threshold: float
        results: dict[name] containing:
            - "qvals": q-value 2D array (same shape)
            - "rejected": boolean mask 2D
            - "prop_rejected": float, proportion of valid grid cells rejected
        global_prop_rejections: float, proportion of all valid grid cells rejected
    """

    flat_p = []
    flat_w = []
    flat_t = []
    data_index = []  # mapping back to each bundle/grid cell
    total_valid = 0  # total valid grid cells

    for bname, bdata in data_dict.items():
        p = bdata["pvals"]
        t = bdata["tvals"]
        mask = ~np.isnan(p)
        pb = p[mask]
        flat_p.append(pb)
        flat_t.append(t[mask])

        # assign bundle weight
        if weight_by=='resel_density':
            wb = np.full(pb.shape, bdata[weight_by], dtype=float)
        elif weight_by=='rpv':
            wb = np.nan_to_num(bdata[weight_by][mask])
        else:
            wb = np.ones(pb.shape)
        flat_w.append(wb)

        # track valid index and the number of valid cells
        data_index.append((bname, mask))
        total_valid += pb.size

    flat_p = np.concatenate(flat_p)
    flat_t = np.concatenate(flat_t)
    flat_w = np.concatenate(flat_w)

    # lambda_gc = compute_lambda_gc(flat_t)
    # if lambda_gc > 1.1:
    #     z_corrected = flat_t / np.sqrt(lambda_gc)
    #     flat_p = 2 * stats.norm.sf(np.abs(z_corrected))  # two-sided

    # weighted BH
    rejected, qvals, threshold, weights = weighted_fdr_bh(flat_p, weights=flat_w, alpha=alpha)
    print(len(rejected), sum(rejected))

    # get bundle results
    results = {}
    idx = 0
    for bname, mask in data_index:
        p = data_dict[bname]["pvals"]
        n_grid = mask.sum()

        rej_b = np.zeros_like(p, dtype=bool)
        qv_b = np.full_like(p, np.nan, dtype=float)
        wts_b = np.full_like(p, np.nan, dtype=float)

        slice_rej = rejected[idx : idx + n_grid]
        slice_qv = qvals[idx : idx + n_grid]
        slice_wts = weights[idx : idx + n_grid]
        idx += n_grid

        rej_b[mask] = slice_rej
        qv_b[mask] = slice_qv
        wts_b[mask] = slice_wts

        prop_rej = slice_rej.sum() / n_grid if n_grid > 0 else 0.0

        results[bname] = {
            "qvals": qv_b,
            "rejected": rej_b,
            "prop_rejected": prop_rej,
            "weights" : wts_b
        }

    global_prop_rejections = rejected.sum() / total_valid if total_valid > 0 else 0.0

    return threshold, results, global_prop_rejections, weights