import logging
import numpy as np
import numpy.ma as ma
from scipy.ndimage import gaussian_filter

from tpa.detection.kernel_fit import fit_matern

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
            C[m] = (resid_centered**2).mean()
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
            C[m] = cov.mean()
            n_pairs[m] = (~cov.mask).sum()
    
    rho = C/C[0]
    return lags, rho, n_pairs


def resel_density(l_s, s_len, l_r=None, r_len=None):
    '''Compute RESEL density given length scale and grid spacing'''
    fwhm_s = 2.355 * l_s
    if l_r is not None and r_len is not None:
        fwhm_r = 2.355 * l_r
        return (s_len * r_len) / (fwhm_s * fwhm_r)
    else:
        return s_len / fwhm_s


def presmooth_residuals(residuals, mask, fwhm_mm, s_len, r_len=None):
    """
    Pre-smooth residuals with Gaussian filter to stabilize RPV estimation.
    Smooths each subject's residuals independently within the mask.
    
    Parameters
    ----------
    residuals : ndarray, shape (n_s, n_r, n_subj)
        Residuals to smooth
    mask : ndarray, shape (n_s, n_r)
        Valid voxel mask
    fwhm_mm : float
        Full-width at half-maximum of Gaussian kernel in mm
    voxel_size : tuple of float
        Physical dimensions of each grid cell (s_len, r_len) in mm
    
    Returns
    -------
    smoothed_residuals : ndarray, shape (n_s, n_r, n_subj)
        Smoothed residuals
    """
    n_s, n_r, n_subj = residuals.shape
    mask = mask.astype(bool)
    
    # Convert FWHM to sigma: sigma = FWHM / sqrt(8*ln(2))
    sigma_s = (fwhm_mm / s_len) / np.sqrt(8 * np.log(2))
    if n_r > 1 and r_len is not None:
        sigma_r = (fwhm_mm / r_len) / np.sqrt(8 * np.log(2))
        sigma = (sigma_s, sigma_r)
        logging.debug(f"Smoothing with FWHM={fwhm_mm}mm, sigma_s={sigma_s:.3f}, sigma_r={sigma_r:.3f}")
    else:
        sigma = sigma_s
        logging.debug(f"Smoothing with FWHM={fwhm_mm}mm, sigma_s={sigma_s:.3f}")

    smoothed = np.full_like(residuals, np.nan)
    for subj in range(n_subj):
        # Get subject residuals and mask
        resid_slice = residuals[:, :, subj].copy()
        resid_slice[~mask] = 0
        
        # Smooth the residuals and mask
        smoothed_slice = gaussian_filter(resid_slice, sigma=sigma, 
                                        mode='constant', cval=0)                            
        smoothed_mask = gaussian_filter(mask.astype(float), sigma=sigma, 
                                       mode='constant', cval=0)
        
        # Normalize to preserve intensity
        # valid = smoothed_mask > 1e-3
        # smoothed_slice[valid] /= smoothed_mask[valid]
        smoothed_slice[~mask] = np.nan
        smoothed[:, :, subj] = smoothed_slice
    
    return smoothed


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

    lags_s, rho_s, n_pairs_s = empirical_correlation(resid, axis=0, max_lag=ns-1)
    report_s = fit_matern(lags_s, rho_s, n_pairs_s, s_len, rho_thresh=rho_thresh, nu=nu)
    l_s, lag_s, r2_s = report_s[f'{kernel}_l'], report_s['max_lag'], report_s[f'{kernel}_r2']
    logging.debug(f"Fit {kernel.upper()} (nu={nu}, max_lag={lag_s}) along s: l={l_s:.3f}, r2={r2_s:.3f}")
    
    if nr>1 and r_len is not None:
        lags_r, rho_r, n_pairs_r = empirical_correlation(resid, axis=1, max_lag=nr-1)
        report_r = fit_matern(lags_r, rho_r, n_pairs_r, r_len, rho_thresh=rho_thresh, nu=nu)
        l_r, lag_r, r2_r = report_r[f'{kernel}_l'], report_r['max_lag'], report_r[f'{kernel}_r2']
        logging.debug(f"Fit {kernel.upper()} (nu={nu}, max_lag={lag_r}) along r: l={l_r:.3f}, r2={r2_r:.3f}")
        return resel_density(l_s, s_len, l_r, r_len), report_s, report_r
    else:
        return resel_density(l_s, s_len), report_s