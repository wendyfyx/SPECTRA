import logging
import numpy as np
from scipy.optimize import minimize, curve_fit
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score

def matern_correlation(d, l, nu=0.5):
    '''Closed form Matern correlation given distance and length scale'''
    if nu == 0.5:
        return np.exp(-d / l)
    if nu == 1.5:
        r = np.sqrt(3.0) * d / l
        return (1.0 + r) * np.exp(-r)
    if nu == 2.5:
        r = np.sqrt(5.0) * d / l
        return (1.0 + r + (r**2)/3.0) * np.exp(-r)

def rbf_correlation(d, l):
    '''RBF kernel given distance and length scale'''
    return np.exp(-d**2 / (2 * l**2))

def _rbf_fit(lags, rho, delta, w):
    '''Fit RBF kernel using log-linear transform'''
    x = (lags * delta) ** 2 
    y = -2 * np.log(rho)
    a = np.sum(w * x * y) / np.sum(w * x * x)
    l = 1.0 / np.sqrt(a)
    y_pred = a * x
    rho_pred = np.exp(-0.5 * y_pred)
    return l, rho_pred, x, y, y_pred

def regression_metrics(y, y_pred, prefix=''):
    '''Evaluate regression fit'''
    scores = {}
    scores[f'{prefix}r2'] = r2_score(y, y_pred)
    scores[f'{prefix}ev'] = explained_variance_score(y, y_pred)
    scores[f'{prefix}rmse'] = np.sqrt(mean_squared_error(y, y_pred))
    return scores

def fit_matern(lags, rho, n_pairs, delta, nu=0.5, 
               rho_thresh=0.2, max_lag=12, min_pairs=20):
    """
    Fit Matérn kernel to empirical correlation.

    Parameters
    ----------
    lags : array
        Lag values (in grid steps)
    rho : array
        Correlation values rho(m)
    n_pairs : array
        Number of pairs at each lag, used as weights
    delta : float
        Grid spacing (mm), s_len or r_len
    nu : float or None
        Smoothness parameter. If None, will fit nu as well.
        Common choices: 0.5, 1.5, 2.5, or None to estimate.
    rho_thresh : float
        Select max_lag when rho first drops below threshold
    max_lag : int
        Max number of lag to fit kernel for
    min_pairs : int
        The minimum number of pairs required for lag to be considered
    
    Returns
    -------
    params : dict
        Fitted parameters {'ell': ..., 'nu': ...}
    fit_info : dict
        Fit diagnostics
    """

    # Dynamically select max_lag based on rho_thresh to avoid fitting on noisy tails
    indices = np.where(rho<rho_thresh)[0]
    n_lag = indices[0] if len(indices)>0 else np.argmin(rho)
    n_lag = np.clip(n_lag, 3, max_lag)
    lags = lags[:n_lag]
    rho = rho[:n_lag]
    n_pairs = n_pairs[:n_lag]

    # exclude lag=0 or if rho<=0
    valid = (lags >= 1) & (rho > 0) & (n_pairs > min_pairs)
    lags_valid = lags[valid]
    rho_valid = rho[valid]
    weights = n_pairs[valid]
    distances = lags_valid * delta

    # Fit RBF
    l0, rbf_rho_pred, _, _, _ = _rbf_fit(lags_valid, rho_valid, delta, weights)
    report = {
        'max_lag': n_lag,
        'n_pairs': weights,
        'lags_fit': lags_valid,
        'rho_fit': rho_valid,
        'rbf_l': l0,
        'rbf_rho_pred': rbf_rho_pred
    }
    report.update(regression_metrics(rho_valid, rbf_rho_pred, prefix='rbf_'))
    
    # Fit Matern
    if nu is not None:
        # Define objective function
        def objective(params_vec):
            l_opt = params_vec[0]        
            rho_pred = matern_correlation(distances, l_opt, nu)
            residuals = rho_valid - rho_pred
            wss = np.sum(weights * residuals**2)
            return wss
        
        # Optimize, using l0 from RBF to initialize
        bounds = [(l0 * 0.1, l0 * 10)]
        result = minimize(objective, [l0], method='L-BFGS-B', bounds=bounds)
        if not result.success:
            logging.warn(f"Warning: Optimization did not converge: {result.message}")
        l = result.x[0]

        # Evaluate results        
        rho_pred = matern_correlation(distances, l, nu)
        report.update({
            'matern_l': l,
            'matern_rho_pred': rho_pred,
            'matern_opt_success': result.success,
        })
        report.update(regression_metrics(rho_valid, rho_pred, prefix='matern_'))
    
    return report