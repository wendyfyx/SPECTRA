import logging
from typing import Optional, Tuple

import numpy as np
import numpy.ma as ma
from scipy.ndimage import gaussian_filter

from spectra.math.kernel_fit import fit_matern

def empirical_correlation(resid, axis=0, max_lag=20):
    '''
    Compute empirical correlation rho(m)=C(m)/C(0) from residuals

    Parameters
    ----------
    resid : np.ndarray, shape (ns, nr, n_subj)
        Residuals (can contain NA)
    max_lag : int
        Maximum lag to compute (in grid steps)
    axis : int, 0 or 1
        0 for s-direction, 1 for r-direction
        
    Returns
    -------
    lags : array
        Lag values [0, 1, 2, ..., max_lag]
    rho : array
        correlation rho(m) for each lag to up max_lag
    n_pairs : array
        Number of valid pairs at each lag
    C0 : float
        Total variance C(0) at lag 0
    '''
    ns, nr, n_subj = resid.shape

    # mask and center
    resid = ma.masked_invalid(resid)
    resid_centered = resid - resid.mean(axis=2, keepdims=True)

    # define max lag
    if axis == 0:
        max_lag = min(ns - 1, max_lag)
    elif axis == 1:
        max_lag = min(nr - 1, max_lag)
    else:
        return
    
    lags = np.arange(max_lag + 1)
    C = np.zeros(max_lag + 1)
    n_pairs = np.zeros(max_lag + 1, dtype=int)

    for m in lags:
        if m == 0: # C(0)
            C[m] = ma.filled((resid_centered**2).mean(), np.nan)
            n_pairs[m] = (~resid_centered.mask).sum()
        else: # C(m)
            if axis == 0:
                slice1 = resid_centered[:-m, :, :] # resid[s, :, :]
                slice2 = resid_centered[m:, :, :] # resid[s+m, :, :]
            elif axis == 1:
                slice1 = resid_centered[:, :-m, :] # resid[:, r, :]
                slice2 = resid_centered[:, m:, :] # resid[:, r+m, :]
            else:
                return
            cov = slice1 * slice2
            C[m] = ma.filled(cov.mean(), np.nan)
            n_pairs[m] = (~cov.mask).sum()
    
    C0 = C[0]
    rho = C/C0
    return lags, rho, n_pairs, C0

def build_masked_kernel_2d(mask, K_s, K_r=None):
    """
    Build a single kernel matrix for valid (masked) points in a 2D grid,
    using separable Matern parameters.
    
    Parameters
    ----------
    mask : ndarray (n_s, n_r)
        Boolean mask (True = valid)
    K_s : ndarray (n_s, n_s)
        1D kernel along s
    K_r : ndarray (n_r, n_r) or None
        1D kernel along r (None for 1D case)
    
    Returns
    -------
    K_valid : ndarray (n_valid, n_valid)
        Kernel for valid points only
    valid_indices : list of tuples
        List of (s, r) coordinates for valid points
    """
    valid_indices = list(zip(*np.where(mask)))
    n_valid = len(valid_indices)
    
    K_valid = np.zeros((n_valid, n_valid))
    
    for i, (s_i, r_i) in enumerate(valid_indices):
        for j, (s_j, r_j) in enumerate(valid_indices):
            if K_r is not None:
                K_valid[i, j] = K_s[s_i, s_j] * K_r[r_i, r_j]
            else:
                K_valid[i, j] = K_s[s_i, s_j]
    
    return K_valid, valid_indices


def fit_and_build_smoother(residuals, mask, s_len, r_len=None, nu=0.5, rho_thresh=0.2):
    """
    Wrapper that fits Matern kernels and constructs the smoother,
    handling both 1D and 2D cases.
    
    Parameters
    ----------
    residuals : ndarray (n_s, n_r, n_subj)
        Residuals from LMM
    mask : ndarray (n_s, n_r)
        Valid voxel mask
    s_len : float
        Grid spacing in s-direction (mm)
    r_len : float or None
        Grid spacing in r-direction (mm), None for 1D
    nu : float
        Matern smoothness parameter (default 0.5)
    rho_thresh : float
        Correlation threshold for lag selection (default 0.2)
    
    Returns
    -------
    S_valid : ndarray (n_valid, n_valid)
        Smoother matrix for valid points
    valid_indices : list of tuples
        (s, r) coordinates of valid points
    fit_report : dict
        Diagnostic info (length scales, R2, etc.)
    """
    from spectra.detection.kernel_fit import build_kernel_matrix, build_smoother_matrix
    
    n_s, n_r, n_subj = residuals.shape
    is_1d = (n_r == 1 or r_len is None)
    
    # Fit s-direction
    lags_s, rho_s, n_pairs_s, C0 = empirical_correlation(residuals, axis=0, max_lag=n_s-1)
    report_s = fit_matern(
        lags_s, rho_s, n_pairs_s, s_len, 
        nu=nu, rho_thresh=rho_thresh,
        total_variance=C0, return_kernel=True, n_loc=n_s
    )
    l_s = report_s['matern_l']
    r2_s = report_s['matern_r2']
    K_s = report_s['K']
    
    fit_report = {
        'l_s': l_s,
        'r2_s': r2_s,
        'C0': C0,
    }
    
    if is_1d:
        # 1D case
        sigma2 = report_s.get('sigma2', (1 - r2_s) * C0)
        K_r = None
        fit_report['sigma2'] = sigma2
        fit_report['noise_fraction'] = report_s.get('noise_fraction', 1 - r2_s)
    else:
        # 2D case - fit r-direction
        lags_r, rho_r, n_pairs_r, _ = empirical_correlation(residuals, axis=1, max_lag=n_r-1)
        report_r = fit_matern(
            lags_r, rho_r, n_pairs_r, r_len,
            nu=nu, rho_thresh=rho_thresh,
            total_variance=C0, return_kernel=True, n_loc=n_r
        )
        l_r = report_r['matern_l']
        r2_r = report_r['matern_r2']
        K_r = report_r['K']
        
        # Global sigma2 from average R2
        avg_r2 = (r2_s + r2_r) / 2
        noise_fraction = 1 - np.clip(avg_r2, 0, 1)
        sigma2 = noise_fraction * C0
                
        fit_report.update({
            'l_r': l_r,
            'r2_r': r2_r,
            'sigma2': sigma2,
            'noise_fraction': noise_fraction,
        })
    
    # Build masked kernel
    K_valid, valid_indices = build_masked_kernel_2d(mask, K_s, K_r)
    
    # Build smoother
    S_valid = build_smoother_matrix(K_valid, sigma2)
    
    # Compute effective degrees of freedom
    smoother_df = np.trace(S_valid)
    fit_report['smoother_df'] = smoother_df
    fit_report['n_valid'] = len(valid_indices)
    
    logging.info(f"Matern smoothing: l_s={l_s:.2f}mm, " + 
                 (f"l_r={fit_report.get('l_r', 0):.2f}mm, " if not is_1d else "") +
                 f"sigma2={sigma2:.4f}, R2_s={r2_s:.3f}" +
                 (f", R2_r={fit_report.get('r2_r', 0):.3f}" if not is_1d else "") +
                 f", smoother_df={smoother_df:.1f}/{len(valid_indices)}")
    
    return S_valid, valid_indices, fit_report


def smooth_residuals(residuals, S_valid, valid_indices, mask):
    """
    Apply Matern smoothing to residuals using pre-computed smoother matrix.
    
    Parameters
    ----------
    residuals : ndarray (n_s, n_r, n_subj)
        Residuals to smooth
    S_valid : ndarray (n_valid, n_valid)
        Smoother matrix for valid points
    valid_indices : list of tuples
        (s, r) coordinates of valid points
    mask : ndarray (n_s, n_r)
        Boolean mask of valid voxels
    
    Returns
    -------
    smoothed : ndarray (n_s, n_r, n_subj)
        Smoothed residuals (NaN where mask is False)
    """
    n_s, n_r, n_subj = residuals.shape
    smoothed = np.full_like(residuals, np.nan)
    
    n_valid = len(valid_indices)
    
    for subj in range(n_subj):
        # Extract valid residuals into a flat vector
        resid_flat = np.array([residuals[s, r, subj] for s, r in valid_indices])
        
        # Handle NaN in input
        nan_mask = np.isnan(resid_flat)
        if nan_mask.any():
            resid_flat[nan_mask] = 0
        
        # Apply smoother and place back into grid
        smoothed_flat = S_valid @ resid_flat
        for i, (s, r) in enumerate(valid_indices):
            smoothed[s, r, subj] = smoothed_flat[i]
    
    return smoothed

def resel_density(l_s, s_len, l_r=None, r_len=None):
    '''Compute RESEL density given length scale and grid spacing'''
    fwhm_s = 2.355 * l_s
    if l_r is not None and r_len is not None:
        fwhm_r = 2.355 * l_r
        return (s_len * r_len) / (fwhm_s * fwhm_r)
    else:
        return s_len / fwhm_s

def resel_density_from_resid(resid, s_len, r_len=None, nu=0.5, rho_thresh=0.1):
    '''
    Compute RESEL density given 1D or 2D residuals by fitting 
    RBF or Matern kernel

    Parameters
    ----------
    resid : np.ndarray, shape (ns, nr, n_subj)
        Residuals (can contain NA)
    s_len : float
        Along-tract segment size (mm)
    r_len : float (optional)
        Radial bin size (mm)
        
    Returns
    -------
    resel_density : float
        RESEL density value computed from length scale
    report_s : dict
        Kernel fit report along s
    report_r : dict (optional)
        Kernel fit report along r
    '''
    ns, nr, _ = resid.shape
    kernel = 'rbf' if nu is None else 'matern'

    lags_s, rho_s, n_pairs_s, _ = empirical_correlation(resid, axis=0, max_lag=ns-1)
    report_s = fit_matern(lags_s, rho_s, n_pairs_s, s_len, rho_thresh=rho_thresh, nu=nu)
    l_s, lag_s, r2_s = report_s[f'{kernel}_l'], report_s['max_lag'], report_s[f'{kernel}_r2']
    logging.debug(f"Fit {kernel.upper()} (nu={nu}, max_lag={lag_s}) along s: l={l_s:.3f}, r2={r2_s:.3f}")
    
    if nr>1 and r_len is not None:
        lags_r, rho_r, n_pairs_r, _ = empirical_correlation(resid, axis=1, max_lag=nr-1)
        report_r = fit_matern(lags_r, rho_r, n_pairs_r, r_len, rho_thresh=rho_thresh, nu=nu)
        l_r, lag_r, r2_r = report_r[f'{kernel}_l'], report_r['max_lag'], report_r[f'{kernel}_r2']
        logging.debug(f"Fit {kernel.upper()} (nu={nu}, max_lag={lag_r}) along r: l={l_r:.3f}, r2={r2_r:.3f}")
        return resel_density(l_s, s_len, l_r, r_len)
    else:
        return resel_density(l_s, s_len)

def estimate_length_scales(
    residuals: np.ndarray,
    s_len: float,
    r_len: Optional[float] = None,
    nu: float = 0.5,
    rho_thresh: float = 0.2,
) -> Tuple[float, Optional[float]]:
    """
    Estimate Matern length scales from residuals without building the smoother.

    A lightweight alternative to fit_and_build_smoother when only the length
    scales are needed (e.g. for RESEL-based block partitioning in hFDR).
    Should be called on reduced-model residuals to keep the estimate
    independent of the target-variable effect.

    Parameters
    ----------
    residuals : np.ndarray, shape (ns, nr, n_subj)
        Residuals from the reduced LMM (covariates only, no target variable).
    s_len : float
        Grid spacing along s (mm).
    r_len : float or None
        Grid spacing along r (mm). Pass None for 1D bundles.
    nu : float
        Matern smoothness parameter (default 0.5).
    rho_thresh : float
        Correlation threshold for automatic lag truncation (default 0.2).

    Returns
    -------
    l_s : float
        Fitted length scale along s (mm).
    l_r : float or None
        Fitted length scale along r (mm), or None for 1D bundles.
    """
    ns, nr, _ = residuals.shape
    is_1d = (nr == 1 or r_len is None)

    # s-direction
    lags_s, rho_s, n_pairs_s, _ = empirical_correlation(
        residuals, axis=0, max_lag=ns - 1
    )
    report_s = fit_matern(
        lags_s, rho_s, n_pairs_s, s_len,
        nu=nu, rho_thresh=rho_thresh,
    )
    l_s = float(report_s['matern_l'])

    # r-direction (2D only)
    l_r = None
    if not is_1d:
        lags_r, rho_r, n_pairs_r, _ = empirical_correlation(
            residuals, axis=1, max_lag=nr - 1
        )
        report_r = fit_matern(
            lags_r, rho_r, n_pairs_r, r_len,
            nu=nu, rho_thresh=rho_thresh,
        )
        l_r = float(report_r['matern_l'])

    logging.debug(
        f"estimate_length_scales: l_s={l_s:.3f}mm" +
        (f", l_r={l_r:.3f}mm" if l_r is not None else "") +
        f" (nu={nu})"
    )
    return l_s, l_r