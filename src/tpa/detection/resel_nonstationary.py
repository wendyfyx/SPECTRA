import numpy as np
import numpy.ma as ma
from scipy.ndimage import binary_dilation, gaussian_filter

def rpv_from_resid(resid, s_len, r_len=None, mask=None):
    """
    Calculate local RPV for each grid point using Hayasaka et al. 2004 method.
    Zero-padding is used to avoid border effects, and gradients are calculated
    with central differences. Supports both 1D and 2D.
    
    Parameters
    ----------
    resid : ndarray, shape (n_s, n_r, n_subj)
        Residuals from linear mixed model at each grid point
    mask : ndarray, shape (n_s, n_r), optional
        Boolean mask where True indicates valid brain voxels
        If None, will be computed from non-NaN values
    s_len : float
        Along-tract segment size (mm)
    r_len : float (optional)
        Radial bin size (mm)
    
    Returns
    -------
    rpv : ndarray, shape (n_s, n_r)
        Local RPV at each grid point (in voxel units)
    """
    n_s, n_r, n_subj = resid.shape
    is_1d = (n_r == 1)

    # Create mask from non-NaN values if not provided
    if mask is None:
        mask = ~np.isnan(resid).any(axis=2)
    resid_ma = ma.masked_invalid(resid)
    
    # Calculate normalized residuals u(v) = e(v) / sqrt(e(v)^T e(v))
    residual_norm = ma.sqrt(ma.sum(resid_ma**2, axis=2))
    residual_norm = ma.where(residual_norm == 0, 1, residual_norm)
    u = resid_ma / residual_norm[:, :, np.newaxis]
    
    # Extend and pad if edges have any valid values to prevent erosion
    pad_top = 1 if mask[0, :].any() else 0
    pad_bottom = 1 if mask[-1, :].any() else 0
    if is_1d:
        pad_left = 0
        pad_right = 0
    else:
        pad_left = 1 if mask[:, 0].any() else 0
        pad_right = 1 if mask[:, -1].any() else 0
    
    new_n_s = n_s + pad_top + pad_bottom
    new_n_r = n_r + pad_left + pad_right
    u_padded = np.zeros((new_n_s, new_n_r, n_subj))
    mask_padded = np.zeros((new_n_s, new_n_r), dtype=bool)
    u_padded[pad_top:pad_top+n_s, pad_left:pad_left+n_r, :] = u.filled(0)
    mask_padded[pad_top:pad_top+n_s, pad_left:pad_left+n_r] = mask
    
    # Dilate mask to include padding region
    padding_region = binary_dilation(mask_padded) & ~mask_padded
    u_padded[padding_region] = 0
    
    # Calculate gradients using central differences
    # Account for voxel size: gradient = Δu / Δx
    du_ds = np.zeros_like(u_padded)
    du_ds[1:-1, :, :] = (u_padded[2:, :, :] - u_padded[:-2, :, :]) / (2 * s_len)
    if not is_1d:
        du_dr = np.zeros_like(u_padded)
        du_dr[:, 1:-1, :] = (u_padded[:, 2:, :] - u_padded[:, :-2, :]) / (2 * r_len)
    
    # Calculate RPV
    rpv = np.full((n_s, n_r), np.nan)    
    for s in range(n_s):
        for r in range(n_r):
            if not mask[s, r]:
                continue
            s_pad = s + pad_top
            r_pad = r + pad_left
            grad_s = du_ds[s_pad, r_pad, :]
            if is_1d:
                grad_matrix = grad_s[np.newaxis, :]  # shape (1, n_subj)
                K_matrix = grad_matrix @ grad_matrix.T / n_subj
                K_value = K_matrix[0, 0]
                if K_value > 0:
                    rpv[s, r] = np.sqrt(4 * np.log(2)) * np.sqrt(K_value)
            else:
                grad_r = du_dr[s_pad, r_pad, :]
                grad_matrix = np.vstack([grad_s, grad_r])  # shape (2, n_subj)
                K_matrix = grad_matrix @ grad_matrix.T / n_subj
                det_K = np.linalg.det(K_matrix)
                if det_K > 0:
                    rpv[s, r] = (4 * np.log(2)) * np.sqrt(det_K)
    return rpv
