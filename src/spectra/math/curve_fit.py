"""
curve_fit.py
------------
Sigmoid / Gompertz curve fitting for recall-vs-sample-size (or any monotone
response curve). Mirrors the functional style of kernel_fit.py.

Usage
-----
    report = fit_curve(x, y, model='gompertz')            # floor 0, ceiling 1 fixed
    report = fit_curve(x, y, model='gompertz', L=None, U=None)   # floor and ceiling fitted
    n_80   = predict_x(0.8, report)
    y_hat  = predict_y(300, report)
"""

import logging
import warnings

import numpy as np
from scipy.optimize import curve_fit, brentq
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ---------------------------------------------------------------------------
# model registry
# ---------------------------------------------------------------------------

def gompertz(x, L, U, k, x0):
    """Gompertz: asymmetric S-curve with floor L and ceiling U."""
    return L + (U - L) * np.exp(-np.exp(-k * (x - x0)))


def logistic(x, L, U, k, x0):
    """Logistic: symmetric S-curve with floor L and ceiling U."""
    return L + (U - L) / (1.0 + np.exp(-k * (x - x0)))

_MODELS = {'gompertz': gompertz, 'logistic': logistic}


# ---------------------------------------------------------------------------
# public functions
# ---------------------------------------------------------------------------

def fit_curve(x, y, model='gompertz', L=0.0, U=1.0):
    """
    Fit a logistic or Gompertz curve to (x, y) data.

    Parameters
    ----------
    x : array-like
        Predictor values (e.g. sample sizes). Should be positive.
    y : array-like
        Response values (e.g. recall). Should be in [0, 1].
    model : str
        One of 'gompertz' (default), 'logistic'.
    L, U : float or None
        Floor and ceiling. A number fixes the value (default 0 and 1);
        None fits it within [0, 1]. k and x0 are always fitted.

    Returns
    -------
    report : dict
        Keys:
        - 'model'       : str, model name
        - 'fn'          : callable, model function fn(x, L, U, k, x0)
        - 'popt'        : np.ndarray, [L, U, k, x0] (fixed and fitted values)
        - 'params'      : dict, {param_name: value}
        - 'free'        : list of str, names of the fitted parameters
        - 'r2'          : float, R²
        - 'rmse'        : float, root mean squared error (recall units)
        - 'mae'         : float, mean absolute error (recall units)
        - 'converged'   : bool
        - 'n_points'    : int
    """
    if model not in _MODELS:
        raise ValueError(f"Unknown model '{model}'. Choose from {sorted(_MODELS)}.")

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) != len(y):
        raise ValueError(f"x and y must have the same length, got {len(x)} and {len(y)}")

    fn = _MODELS[model]

    # Each parameter: (starting value, lower bound, upper bound)
    spec = {
        'L':  (0.0, 0.0, 1.0),
        'U':  (1.0, 0.0, 1.0),
        'k':  (0.01, 1e-6, np.inf),
        'x0': (float(np.median(x)), -np.inf, np.inf),
    }

    # Fixed parameters keep the value passed in; the rest are fitted
    fixed = {name: float(v) for name, v in (('L', L), ('U', U)) if v is not None}
    free  = [name for name in spec if name not in fixed]

    def fn_free(x, *theta):
        p = {**fixed, **dict(zip(free, theta))}
        return fn(x, p['L'], p['U'], p['k'], p['x0'])

    p0 = [spec[name][0] for name in free]

    converged = True
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            theta, _ = curve_fit(
                fn_free, x, y,
                p0=p0,
                bounds=([spec[name][1] for name in free],
                        [spec[name][2] for name in free]),
                method='trf',
                maxfev=10_000,
            )
    except Exception as e:
        logging.warning(f"fit_curve: curve_fit failed for model '{model}': {e}")
        theta = np.array(p0)
        converged = False

    params = {**fixed, **dict(zip(free, theta))}
    popt = np.array([params[name] for name in spec])
    y_pred = fn(x, *popt)

    report = {
        'model':    model,
        'fn':       fn,
        'popt':     popt,
        'params':   {name: float(params[name]) for name in spec},
        'free':     free,
        'r2':       float(r2_score(y, y_pred)),
        'rmse':     float(np.sqrt(mean_squared_error(y, y_pred))),
        'mae':      float(mean_absolute_error(y, y_pred)),
        'converged': converged,
        'n_points': len(x),
    }
    return report


def predict_y(x, report):
    """
    Predict response at a given x using a fitted report.

    Parameters
    ----------
    x : float
        Predictor value (e.g. sample size).
    report : dict
        Output of fit_curve().

    Returns
    -------
    float
        Predicted response value.
    """
    return float(report['fn'](np.array([x]), *report['popt'])[0])


def predict_x(y_target, report, x_search=(1.0, 1e6)):
    """
    Predict the x value required to achieve a target response.

    Uses analytic inversion where available, falling back to Brent's
    numerical root-finding. Returns np.inf with a warning if the
    fitted curve never reaches y_target.

    Parameters
    ----------
    y_target : float
        Target response value (e.g. 0.8 for 80% recall).
    report : dict
        Output of fit_curve().
    x_search : tuple of (float, float)
        Search interval for fallback numerical root-finding.

    Returns
    -------
    float
        Predicted x value, or np.inf if curve never reaches y_target.
    """
    params = report['params']
    model  = report['model']
    fn     = report['fn']
    popt   = report['popt']

    L = params.get('L', 0.0)
    U = params.get('U', 1.0)

    if y_target >= U:
        logging.warning(
            f"predict_x: target y={y_target:.3f} >= ceiling U={U:.4f}. "
            f"Curve never reaches this value. Returning inf."
        )
        return np.inf
    if y_target <= L:
        logging.warning(
            f"predict_x: target y={y_target:.3f} <= floor L={L:.4f}. "
            f"Already exceeded at any x. Returning 0."
        )
        return 0.0

    k  = params['k']
    x0 = params['x0']
    ratio = (y_target - L) / (U - L)   # in (0, 1)

    # Analytic inversion
    try:
        if model == 'gompertz':
            # ratio = exp(-exp(-k*(x-x0)))  =>  x = x0 - log(-log(ratio)) / k
            x = x0 - np.log(-np.log(ratio)) / k
        elif model == 'logistic':
            # ratio = 1/(1+exp(-k*(x-x0)))  =>  x = x0 + log(ratio/(1-ratio)) / k
            x = x0 + np.log(ratio / (1.0 - ratio)) / k
        else:
            raise NotImplementedError
        return float(x)
    except Exception:
        pass

    # Fallback: numerical root-finding
    lo, hi = x_search
    try:
        logging.info(f"Fallback to root-finding")
        return float(brentq(lambda x: fn(x, *popt) - y_target, lo, hi))
    except ValueError:
        logging.warning(
            f"predict_x: root-finding failed for y_target={y_target} "
            f"in [{lo}, {hi}]. Returning inf."
        )
        return np.inf