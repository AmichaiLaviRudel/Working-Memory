from Analysis.GNG_bpod_analysis.licking_and_outcome import preprocess_stimuli_outcomes, compute_lick_rate
from Analysis.GNG_bpod_analysis.metric import *
# Removed wildcard import to avoid circular dependency
from Analysis.GNG_bpod_analysis.GNG_bpod_general import (
    get_plotly_config,
    get_sessions_for_animal,
    get_global_early_response_filter,
)
import Analysis.GNG_bpod_analysis.colors as colors
import plotly.graph_objects as go
import numpy as np
import streamlit as st
import pandas as pd
# altair import removed - using Plotly instead
from scipy.optimize import curve_fit
from scipy import stats

# -------------------------------------------------------------------
# WEIBULL PSYCHOMETRIC FUNCTION
# -------------------------------------------------------------------
def weibull_cdf(x, alpha, beta, gamma=0.0, lam=0.0):
    """
    Weibull cumulative distribution function for psychometric fitting.
    
    Parameters
    ----------
    x : array-like
        Stimulus values (must be > 0 for valid Weibull).
    alpha : float
        Threshold parameter (stimulus level at ~63% performance).
    beta : float
        Slope parameter (steepness of the curve).
    gamma : float
        Guess rate / lower asymptote (typically 0 for Go/No-Go).
    lam : float
        Lapse rate (upper asymptote = 1 - lam).
    
    Returns
    -------
    y : array
        Performance values in [gamma, 1-lam].
    """
    # Clip x to avoid numerical issues with negative values
    x_safe = np.maximum(x, 1e-10)
    return gamma + (1 - gamma - lam) * (1 - np.exp(-(x_safe / alpha) ** beta))


def _compute_r_squared(y_actual, y_predicted):
    """Compute coefficient of determination (R²)."""
    ss_res = np.sum((y_actual - y_predicted) ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1.0 - (ss_res / ss_tot)


def _detect_outliers(residuals, threshold_std: float = 2.0):
    """
    Detect outliers based on residuals using standard deviation threshold.
    
    Parameters
    ----------
    residuals : array
        Residuals from the fit (y_actual - y_predicted).
    threshold_std : float
        Number of standard deviations to use as outlier threshold.
    
    Returns
    -------
    outlier_mask : array of bool
        True for outlier points.
    """
    if len(residuals) < 4:  # Need enough points for meaningful std
        return np.zeros(len(residuals), dtype=bool)
    
    std_res = np.std(residuals)
    if std_res == 0:
        return np.zeros(len(residuals), dtype=bool)
    
    # Points with |residual| > threshold_std * std are outliers
    return np.abs(residuals) > threshold_std * std_res


def _core_weibull_fit(x, y_norm, fix_gamma: bool = True):
    """
    Core Weibull fitting logic (internal helper).
    
    Returns
    -------
    params : dict or None
        {'alpha', 'beta', 'gamma', 'lambda'} if successful, None if failed.
    error_msg : str or None
        Error message if failed, None if successful.
    """
    # Smart initial guesses based on data
    alpha_init = np.median(x)
    try:
        mid_idx = np.argmin(np.abs(y_norm - 0.5))
        alpha_init = x[mid_idx]
    except Exception:
        pass
    
    beta_init = 2.0
    x_min, x_max = x.min(), x.max()
    
    if fix_gamma:
        def weibull_3p(x, alpha, beta, lam):
            return weibull_cdf(x, alpha, beta, gamma=0.0, lam=lam)
        
        p0 = [alpha_init, beta_init, 0.05]
        bounds = ([x_min * 0.5, 0.1, 0.0], [x_max * 2.0, 10.0, 0.2])
        
        try:
            popt, _ = curve_fit(weibull_3p, x, y_norm, p0=p0, bounds=bounds,
                               maxfev=10000, method='trf')
            return {'alpha': popt[0], 'beta': popt[1], 'gamma': 0.0, 'lambda': popt[2]}, None
        except (RuntimeError, ValueError) as e:
            return None, str(e)
    else:
        def weibull_4p(x, alpha, beta, gamma, lam):
            return weibull_cdf(x, alpha, beta, gamma=gamma, lam=lam)
        
        p0 = [alpha_init, beta_init, 0.0, 0.05]
        bounds = ([x_min * 0.5, 0.1, 0.0, 0.0], [x_max * 2.0, 10.0, 0.5, 0.2])
        
        try:
            popt, _ = curve_fit(weibull_4p, x, y_norm, p0=p0, bounds=bounds,
                               maxfev=10000, method='trf')
            return {'alpha': popt[0], 'beta': popt[1], 'gamma': popt[2], 'lambda': popt[3]}, None
        except (RuntimeError, ValueError) as e:
            return None, str(e)


def weibull_fit(x, y, *, x_boundary: float = 1.0, fix_gamma: bool = True, 
                remove_outliers: bool = True, outlier_threshold: float = 2.0,
                log_x: bool = True):
    """
    Fit Weibull psychometric function with robust parameter estimation.
    
    Supports log-transformed x values for frequency data (log_x=True).
    If the initial fit has R² < 0.7 and remove_outliers=True, attempts to
    identify and remove noisy data points based on residuals, then refits.
    
    Parameters
    ----------
    x : array
        Stimulus values (positive, e.g., frequencies in kHz).
    y : array
        Response rates (e.g., lick rate in %).
    x_boundary : float
        The x value at which to compute the slope (in original units).
    fix_gamma : bool
        If True, fix gamma=0 (appropriate for Go/No-Go tasks).
    remove_outliers : bool
        If True and initial fit is poor, try removing outliers and refit.
    outlier_threshold : float
        Number of standard deviations for outlier detection (default 2.0).
    log_x : bool
        If True (default), fit on log2-transformed x values. This is appropriate
        for frequency data which is typically spaced in octaves.
    
    Returns
    -------
    model_boundaries : np.array([threshold])
        The fitted threshold (alpha parameter) in original x units.
    slopes_mid : np.array([slope_at_threshold])
        Slope at the threshold point.
    slopes_at_model_boundaries : np.array([slope_at_x_boundary])
        Slope at the specified boundary.
    x_fit : array
        X values for plotting the fitted curve (in original units).
    y_fit : array
        Y values for plotting the fitted curve.
    fit_info : dict
        Fit quality metrics: r_squared, converged, message, outliers_removed.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # Filter out invalid values
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x, y = x[mask], y[mask]
    
    if len(x) < 3:
        return _weibull_fit_failure("Insufficient data points (need >= 3)")
    
    # Store original linear x values
    x_orig_linear = x.copy()
    y_orig = y.copy()
    
    # Transform to log scale if requested
    if log_x:
        x = np.log2(x)
        x_boundary_fit = np.log2(x_boundary) if x_boundary > 0 else 0.0
    else:
        x_boundary_fit = x_boundary
    
    x_orig = x.copy()
    
    # Normalize y to 0-1 range
    y_min, y_max = y.min(), y.max()
    if y_max > y_min:
        y_norm = (y - y_min) / (y_max - y_min)
    else:
        y_norm = y.copy()
    
    # For Weibull on log scale, we need to shift x to be positive
    # Since log2(freq) can be negative for freq < 1, shift to start from 0
    x_shift = x.min()
    x_shifted = x - x_shift + 0.1  # Shift so all values > 0
    x_boundary_shifted = x_boundary_fit - x_shift + 0.1
    
    # Initial fit on shifted data
    params, error = _core_weibull_fit(x_shifted, y_norm, fix_gamma)
    if params is None:
        return _weibull_fit_failure(f"Fitting failed: {error}")
    
    alpha, beta, gamma, lam = params['alpha'], params['beta'], params['gamma'], params['lambda']
    
    # Compute initial R²
    y_pred_norm = weibull_cdf(x_shifted, alpha, beta, gamma, lam)
    r_squared = _compute_r_squared(y_norm, y_pred_norm)
    
    outliers_removed = 0
    
    # If fit is poor and we have enough data, try removing outliers
    if remove_outliers and r_squared < 0.7 and len(x) >= 5:
        residuals = y_norm - y_pred_norm
        outlier_mask = _detect_outliers(residuals, outlier_threshold)
        n_outliers = np.sum(outlier_mask)
        
        if n_outliers > 0 and (len(x) - n_outliers) >= 3:
            x_clean = x_shifted[~outlier_mask]
            y_clean = y[~outlier_mask]
            
            y_clean_min, y_clean_max = y_clean.min(), y_clean.max()
            if y_clean_max > y_clean_min:
                y_clean_norm = (y_clean - y_clean_min) / (y_clean_max - y_clean_min)
            else:
                y_clean_norm = y_clean.copy()
            
            params_clean, _ = _core_weibull_fit(x_clean, y_clean_norm, fix_gamma)
            
            if params_clean is not None:
                alpha_c, beta_c = params_clean['alpha'], params_clean['beta']
                gamma_c, lam_c = params_clean['gamma'], params_clean['lambda']
                
                y_pred_clean = weibull_cdf(x_clean, alpha_c, beta_c, gamma_c, lam_c)
                r_squared_clean = _compute_r_squared(y_clean_norm, y_pred_clean)
                
                if r_squared_clean > r_squared:
                    alpha, beta, gamma, lam = alpha_c, beta_c, gamma_c, lam_c
                    r_squared = r_squared_clean
                    y_min, y_max = y_clean_min, y_clean_max
                    x_shifted = x_clean
                    outliers_removed = n_outliers
    
    # Generate fitted curve on shifted log scale
    x_fit_shifted = np.linspace(x_orig.min() - x_shift + 0.1, x_orig.max() - x_shift + 0.1, 200)
    y_fit_norm = weibull_cdf(x_fit_shifted, alpha, beta, gamma, lam)
    
    # Transform x_fit back to original linear scale
    x_fit_log = x_fit_shifted + x_shift - 0.1  # Back to log scale
    if log_x:
        x_fit_linear = 2 ** x_fit_log  # Back to linear frequency
    else:
        x_fit_linear = x_fit_log
    
    # Transform alpha (threshold) back to original scale
    alpha_log = alpha + x_shift - 0.1  # Back to log scale
    if log_x:
        alpha_linear = 2 ** alpha_log  # Back to linear frequency
    else:
        alpha_linear = alpha_log
    
    # Compute slope at threshold (on shifted scale)
    def weibull_slope(x_val_shifted):
        if x_val_shifted <= 0 or alpha <= 0:
            return 0.0
        ratio = x_val_shifted / alpha
        return (1 - gamma - lam) * (beta / alpha) * (ratio ** (beta - 1)) * np.exp(-(ratio ** beta))
    
    slope_at_threshold = weibull_slope(alpha)
    slope_at_boundary = weibull_slope(x_boundary_shifted)
    
    # Scale slopes back to original y scale
    if y_max > y_min:
        slope_at_threshold *= (y_max - y_min)
        slope_at_boundary *= (y_max - y_min)
        y_fit = y_fit_norm * (y_max - y_min) + y_min
    else:
        y_fit = y_fit_norm
    
    # Build fit quality message
    if r_squared >= 0.7:
        message = "Good fit (Weibull)"
    elif r_squared >= 0.5:
        message = "Moderate fit (Weibull) - interpret with caution"
    else:
        message = "Poor fit (Weibull) - data may not follow psychometric pattern"
    
    if outliers_removed > 0:
        message += f" ({outliers_removed} outlier{'s' if outliers_removed > 1 else ''} removed)"
    
    fit_info = {
        'r_squared': r_squared,
        'converged': True,
        'message': message,
        'model': 'weibull',
        'params': {'alpha': alpha_linear, 'alpha_log': alpha_log, 'beta': beta, 'gamma': gamma, 'lambda': lam},
        'outliers_removed': outliers_removed
    }
    
    return (
        np.array([alpha_linear]),  # Threshold in original units
        np.array([slope_at_threshold]),
        np.array([slope_at_boundary]),
        x_fit_linear,  # x values in original units for plotting
        y_fit,
        fit_info
    )


def _weibull_fit_failure(message: str):
    """Return NaN values when Weibull fitting fails."""
    fit_info = {
        'r_squared': 0.0,
        'converged': False,
        'message': message,
        'model': 'weibull',
        'params': None,
        'outliers_removed': 0
    }
    return (
        np.array([np.nan]),
        np.array([np.nan]),
        np.array([np.nan]),
        np.array([np.nan]),
        np.array([np.nan]),
        fit_info
    )


# -------------------------------------------------------------------
# LOGISTIC SIGMOID FITTING
# -------------------------------------------------------------------
def _sigmoid_4p(x, L, x0, k, b):
    """
    4-parameter logistic sigmoid: y = b + L / (1 + exp(-k*(x - x0)))
    
    Parameters:
        L: amplitude (can be negative for decreasing function)
        x0: midpoint (threshold)
        k: steepness (slope parameter)
        b: baseline (lower asymptote)
    """
    return b + L / (1.0 + np.exp(-k * (x - x0)))


def _estimate_sigmoid_direction(x, y):
    """
    Estimate if the sigmoid should be increasing or decreasing.
    Returns 1 for increasing, -1 for decreasing.
    """
    # Sort by x and check if y generally increases or decreases
    sort_idx = np.argsort(x)
    y_sorted = y[sort_idx]
    
    # Compare first quarter to last quarter
    n = len(y_sorted)
    q1 = np.mean(y_sorted[:max(1, n//4)])
    q4 = np.mean(y_sorted[-(max(1, n//4)):])
    
    return 1 if q4 > q1 else -1


def sigmoid_fit(x, y, *, x_boundary: float = 1.0, remove_outliers: bool = True, 
                outlier_threshold: float = 2.0, log_x: bool = True):
    """
    Robust 4-parameter logistic sigmoid fit with outlier removal.
    
    Handles both increasing and decreasing psychometric functions automatically.
    Fits on log-transformed x values by default (appropriate for frequency data).
    
    Parameters
    ----------
    x : array
        Stimulus values (e.g., frequencies in kHz).
    y : array
        Response rates (e.g., lick rate in %).
    x_boundary : float
        The x value at which to compute the slope (in original units).
    remove_outliers : bool
        If True and fit is poor, try removing outliers and refit.
    outlier_threshold : float
        Number of standard deviations for outlier detection.
    log_x : bool
        If True (default), fit on log2-transformed x values. This is appropriate
        for frequency data which is typically spaced in octaves.
    
    Returns
    -------
    model_boundaries : np.array([x0])
        The fitted threshold (midpoint) in original x units.
    slopes_mid : np.array([slope_at_midpoint])
        Slope at the midpoint.
    slopes_at_model_boundaries : np.array([slope_at_x_boundary])
        Slope at the specified boundary.
    x_fit : array
        X values for plotting (in original units).
    y_fit : array
        Y values for plotting.
    fit_info : dict
        Fit quality: r_squared, converged, message, outliers_removed.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # Filter invalid values (and x > 0 if using log transform)
    if log_x:
        mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
    else:
        mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    
    if len(x) < 3:
        return _sigmoid_fit_failure("Insufficient data points (need >= 3)")
    
    # Store original x values for output
    x_orig_linear = x.copy()
    y_orig = y.copy()
    
    # Transform to log scale if requested (fits better for octave-spaced data)
    if log_x:
        x = np.log2(x)
        x_boundary_fit = np.log2(x_boundary) if x_boundary > 0 else 0.0
    else:
        x_boundary_fit = x_boundary
    
    x_orig = x.copy()  # Store for outlier removal refit
    
    # Estimate direction and set up initial parameters
    direction = _estimate_sigmoid_direction(x, y)
    y_min, y_max = y.min(), y.max()
    y_range = y_max - y_min if y_max > y_min else 1.0
    
    # Initial guesses (now on log scale if log_x=True)
    L_init = direction * y_range  # Positive for increasing, negative for decreasing
    x0_init = np.median(x)
    k_init = direction * 4.0 / (x.max() - x.min() + 1e-6)  # Reasonable steepness
    b_init = y_min if direction > 0 else y_max
    
    # Parameter bounds (on transformed scale)
    x_min, x_max = x.min(), x.max()
    x_span = x_max - x_min if x_max > x_min else 1.0
    
    # Allow both positive and negative L for flexibility
    bounds = (
        [-y_range * 2, x_min - x_span * 0.5, -50.0, y_min - y_range],  # Lower bounds
        [y_range * 2, x_max + x_span * 0.5, 50.0, y_max + y_range]     # Upper bounds
    )
    
    p0 = [L_init, x0_init, k_init, b_init]
    
    # Try fitting
    try:
        popt, pcov = curve_fit(_sigmoid_4p, x, y, p0=p0, bounds=bounds,
                               maxfev=10000, method='trf')
        L, x0, k, b = popt
        converged = True
    except (RuntimeError, ValueError) as e:
        # Try with different initial guesses
        try:
            p0_alt = [L_init, x0_init, -k_init, b_init]  # Flip slope direction
            popt, pcov = curve_fit(_sigmoid_4p, x, y, p0=p0_alt, bounds=bounds,
                                   maxfev=10000, method='trf')
            L, x0, k, b = popt
            converged = True
        except (RuntimeError, ValueError):
            return _sigmoid_fit_failure(f"Fitting failed: {str(e)}")
    
    # Compute R²
    y_pred = _sigmoid_4p(x, L, x0, k, b)
    r_squared = _compute_r_squared(y, y_pred)
    
    outliers_removed = 0
    
    # If fit is poor, try removing outliers
    if remove_outliers and r_squared < 0.7 and len(x) >= 5:
        residuals = y - y_pred
        outlier_mask = _detect_outliers(residuals, outlier_threshold)
        n_outliers = np.sum(outlier_mask)
        
        if n_outliers > 0 and (len(x) - n_outliers) >= 3:
            x_clean = x[~outlier_mask]
            y_clean = y[~outlier_mask]
            
            # Refit without outliers
            try:
                popt_clean, _ = curve_fit(_sigmoid_4p, x_clean, y_clean, p0=popt,
                                          bounds=bounds, maxfev=10000, method='trf')
                y_pred_clean = _sigmoid_4p(x_clean, *popt_clean)
                r_squared_clean = _compute_r_squared(y_clean, y_pred_clean)
                
                if r_squared_clean > r_squared:
                    L, x0, k, b = popt_clean
                    r_squared = r_squared_clean
                    x, y = x_clean, y_clean
                    outliers_removed = n_outliers
            except (RuntimeError, ValueError):
                pass  # Keep original fit
    
    # Generate smooth fitted curve
    # x_fit_log is on log scale (for sigmoid evaluation), x_fit_linear is for plotting
    x_fit_log = np.linspace(x_orig.min(), x_orig.max(), 200)
    y_fit = _sigmoid_4p(x_fit_log, L, x0, k, b)
    
    # Transform x_fit back to linear scale for plotting
    if log_x:
        x_fit_linear = 2 ** x_fit_log  # Convert from log2 back to linear
    else:
        x_fit_linear = x_fit_log
    
    # Compute slopes (on log scale, which is where we fitted)
    # Derivative of sigmoid: dy/dx = L * k * exp(-k*(x-x0)) / (1 + exp(-k*(x-x0)))^2
    def sigmoid_slope(x_val_log):
        exp_term = np.exp(-k * (x_val_log - x0))
        return L * k * exp_term / ((1 + exp_term) ** 2)
    
    slope_at_midpoint = sigmoid_slope(x0)  # Maximum slope magnitude
    slope_at_boundary = sigmoid_slope(x_boundary_fit)
    
    # Transform x0 back to linear scale for output
    if log_x:
        x0_linear = 2 ** x0  # Threshold in original frequency units
    else:
        x0_linear = x0
    
    # Build fit quality message
    if r_squared >= 0.7:
        message = "Good fit"
    elif r_squared >= 0.5:
        message = "Moderate fit - interpret with caution"
    else:
        message = "Poor fit - data may not follow psychometric pattern"
    
    if outliers_removed > 0:
        message += f" ({outliers_removed} outlier{'s' if outliers_removed > 1 else ''} removed)"
    
    fit_info = {
        'r_squared': r_squared,
        'converged': converged,
        'message': message,
        'model': 'sigmoid',
        'params': {'L': L, 'x0_log': x0, 'x0': x0_linear, 'k': k, 'b': b},
        'outliers_removed': outliers_removed
    }
    
    return (
        np.array([x0_linear]),  # Threshold in original units (kHz)
        np.array([slope_at_midpoint]),
        np.array([slope_at_boundary]),
        x_fit_linear,  # x values in original units for plotting
        y_fit,
        fit_info
    )


def _sigmoid_fit_failure(message: str):
    """Return NaN values when sigmoid fitting fails."""
    fit_info = {
        'r_squared': 0.0,
        'converged': False,
        'message': message,
        'model': 'sigmoid',
        'params': None,
        'outliers_removed': 0
    }
    return (
        np.array([np.nan]),
        np.array([np.nan]),
        np.array([np.nan]),
        np.array([np.nan]),
        np.array([np.nan]),
        fit_info
    )

# -------------------------------------------------------------------
# MAIN FRONT-END FUNCTION
# -------------------------------------------------------------------
def psychometric_fitting(unique_stims, data_points, *, N_Boundaries=1, log2_x=True, b_fixed=None, 
                         lapse_fixed=0.0, return_fit_info=False, fit_method='auto',
                         low_boundary=None, high_boundary=None):
    """
    Universal psychometric fitter - tries both Sigmoid and Weibull, picks best.

    Parameters
    ----------
    unique_stims, data_points: 1-D arrays
        Stimulus axis and lick probability.
    N_Boundaries: 1 or 2
        • 1 → monotone single function
        • 2 → rise–fall model (two boundaries, fits two functions)
    log2_x : bool
        Log-transform x before fitting (recommended for frequency data).
    b_fixed : None | (tuple)
        If N_Boundaries==2 you can pin the two boundaries (not currently used).
    lapse_fixed : float
        Symmetric lapse (not currently used).
    return_fit_info : bool
        If True, returns fit_info dict as 6th element.
    fit_method : str
        'auto' (default): Try both sigmoid and Weibull, return best R²
        'sigmoid': Use only 4-parameter logistic sigmoid
        'weibull': Use only Weibull CDF
    low_boundary, high_boundary : float, optional
        For N_Boundaries==2, boundary values in stimulus units (e.g. kHz).
        If None, read from st.session_state (Streamlit only). Use explicit values for offline calls.

    Returns
    -------
    model_boundaries, slopes_mid, slopes_at_model_boundaries, x_fit, y_fit [, fit_info]
        Arrays sized according to N_Boundaries.
        fit_info (optional): dict with r_squared, converged, message, model.
    """
    # ── clean data ─────────────────────────────────────────────
    x = np.asarray(unique_stims, float)
    y = np.asarray(data_points, float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x, y = x[mask], y[mask]
    
    if len(x) < 3:
        fit_info = {'r_squared': 0.0, 'converged': False, 'message': "Insufficient data points", 'outliers_removed': 0}
        if return_fit_info:
            return np.array([np.nan]), np.array([np.nan]), np.array([np.nan]), np.array([np.nan]), np.array([np.nan]), fit_info
        raise ValueError(f"Insufficient data for fitting: need at least 3 data points, got {len(x)}")

    # ── fit the data ─────────────────────────────────────────────
    if N_Boundaries == 1:
        x_boundary = np.mean(x)
        
        # Try fitting with selected method(s)
        results = []
        
        if fit_method in ('auto', 'sigmoid'):
            try:
                result_sig = sigmoid_fit(x, y, x_boundary=x_boundary, log_x=log2_x)
                results.append(('sigmoid', result_sig))
            except Exception:
                pass
        
        if fit_method in ('auto', 'weibull'):
            try:
                result_weib = weibull_fit(x, y, x_boundary=x_boundary, log_x=log2_x)
                results.append(('weibull', result_weib))
            except Exception:
                pass
        
        if not results:
            fit_info = {'r_squared': 0.0, 'converged': False, 'message': "All fitting methods failed", 'outliers_removed': 0}
            if return_fit_info:
                return np.array([np.nan]), np.array([np.nan]), np.array([np.nan]), np.array([np.nan]), np.array([np.nan]), fit_info
            raise ValueError("All fitting methods failed")
        
        # Pick best result by R²
        best_name, best_result = max(results, key=lambda r: r[1][5].get('r_squared', 0))
        model_boundaries, slopes_mid, slopes_at_model_boundaries, x_fit, y_fit, fit_info = best_result
        
        if return_fit_info:
            return model_boundaries, slopes_mid, slopes_at_model_boundaries, x_fit, y_fit, fit_info
        return model_boundaries, slopes_mid, slopes_at_model_boundaries, x_fit, y_fit
        
    elif N_Boundaries == 2:
        # Two separate fits: one for low boundary, one for high boundary
        if low_boundary is not None and high_boundary is not None:
            low_bound, high_bound = low_boundary, high_boundary
        else:
            try:
                low_bound = st.session_state.low_boundary
                high_bound = st.session_state.high_boundary
            except Exception:
                low_bound, high_bound = 0.983, 1.525  # Defaults when not in Streamlit
        
        # Split data for low boundary fit (x <= high_bound)
        # This captures the transition from Go (high lick) to No-Go (low lick)
        mask_low = x <= high_bound
        x_low, y_low = x[mask_low], y[mask_low]
        
        # Split data for high boundary fit (x >= low_bound)
        # This captures the transition from No-Go (low lick) to Go (high lick)
        mask_high = x >= low_bound
        x_high, y_high = x[mask_high], y[mask_high]
        
        # Fallback to all data if insufficient points
        if len(x_low) < 3:
            x_low, y_low = x, y
        if len(x_high) < 3:
            x_high, y_high = x, y
        
        # Helper to fit with auto-selection
        def fit_boundary(x_data, y_data, boundary):
            results = []
            if fit_method in ('auto', 'sigmoid'):
                try:
                    results.append(sigmoid_fit(x_data, y_data, x_boundary=boundary, log_x=log2_x))
                except Exception:
                    pass
            if fit_method in ('auto', 'weibull'):
                try:
                    results.append(weibull_fit(x_data, y_data, x_boundary=boundary, log_x=log2_x))
                except Exception:
                    pass
            if not results:
                return _sigmoid_fit_failure("All fitting methods failed")
            # Pick best by R²
            return max(results, key=lambda r: r[5].get('r_squared', 0))
        
        # Fit each boundary with auto-selection
        result_low = fit_boundary(x_low, y_low, low_bound)
        result_high = fit_boundary(x_high, y_high, high_bound)
        
        # Extract results (both fit functions return 6 elements)
        mb_low, sm_low, sab_low, xf_low, yf_low, fi_low = result_low
        mb_high, sm_high, sab_high, xf_high, yf_high, fi_high = result_high
        
        # Safely extract scalar values
        def safe_val(arr):
            return arr[0] if isinstance(arr, np.ndarray) and len(arr) > 0 else np.nan
        
        model_boundaries = np.array([safe_val(mb_low), safe_val(mb_high)])
        slopes_mid = np.array([safe_val(sm_low), safe_val(sm_high)])
        slopes_at_model_boundaries = np.array([safe_val(sab_low), safe_val(sab_high)])
        x_fit = [xf_low, xf_high]
        y_fit = [yf_low, yf_high]
        
        # Combined fit info: use lower R² as overall quality
        combined_r2 = min(fi_low.get('r_squared', 0), fi_high.get('r_squared', 0))
        converged = fi_low.get('converged', False) and fi_high.get('converged', False)
        total_outliers = fi_low.get('outliers_removed', 0) + fi_high.get('outliers_removed', 0)
        model_low = fi_low.get('model', 'unknown')
        model_high = fi_high.get('model', 'unknown')
        
        if combined_r2 >= 0.7:
            message = f"Good fit ({model_low}/{model_high})"
        elif combined_r2 >= 0.5:
            message = f"Moderate fit ({model_low}/{model_high}) - interpret with caution"
        else:
            message = "Poor fit - data may not follow psychometric pattern"
        
        if total_outliers > 0:
            message += f" ({total_outliers} outlier{'s' if total_outliers > 1 else ''} removed)"
        
        fit_info = {
            'r_squared': combined_r2,
            'converged': converged,
            'message': message,
            'model': f"{model_low}/{model_high}",
            'low_boundary': fi_low,
            'high_boundary': fi_high,
            'outliers_removed': total_outliers
        }
        
        if return_fit_info:
            return model_boundaries, slopes_mid, slopes_at_model_boundaries, x_fit, y_fit, fit_info
        return model_boundaries, slopes_mid, slopes_at_model_boundaries, x_fit, y_fit
    else:
        raise ValueError("N_Boundaries must be 1 or 2")


def psychometric_curve(selected_data, index, plot=True):
    """
    Processes psychometric data, fits a sigmoid curve, and plots the psychometric curve.
    """
    try:
        # Decide whether to filter Early Response trials (shared with multi-session calls)
        filter_early = get_global_early_response_filter()

        # Extract and preprocess data
        stimuli, outcomes = preprocess_stimuli_outcomes(selected_data, index)

        # Optionally filter out 'Early Response' trials before computing lick rates
        if filter_early:
            try:
                early_mask = np.array(
                    ['Early Response' not in str(o) for o in outcomes],
                    dtype=bool,
                )
                if len(early_mask) == len(stimuli):
                    stimuli = stimuli[early_mask]
                    outcomes = outcomes[early_mask]
            except Exception:
                # If anything goes wrong, fall back to unfiltered data
                pass

        unique_stimuli, lick_rates, catch_stimuli, catch_lick_rates = compute_lick_rate(stimuli, outcomes)
        unique_stimuli = np.concatenate((unique_stimuli, catch_stimuli))
        lick_rates = np.concatenate((lick_rates, catch_lick_rates))
        n_b = selected_data.loc[index, 'N_Boundaries']
        session_type =  selected_data.at[index, "Notes"]
        if "TA" in session_type or "Discrimination" in session_type:
            st.info(f"this is {session_type} session")
            return None, None, None, None, None
        # Check if we have enough data points for fitting
        if len(unique_stimuli) < 3:
            return None, None, None, None, None
            
        # Fit psychometric curve with Weibull function, get fit quality info
        model_boundaries, slopes_mid, slopes_at_model_boundaries, x_fit, y_fit, fit_info = psychometric_fitting(
            unique_stimuli, lick_rates,
            N_Boundaries=n_b,
            log2_x=False,
            return_fit_info=True
        )
        
        # Check if fitting failed (all NaN values)
        if (isinstance(model_boundaries, np.ndarray) and np.all(np.isnan(model_boundaries))) or \
           (isinstance(slopes_mid, np.ndarray) and np.all(np.isnan(slopes_mid))):
            st.warning(f"Psychometric curve fitting failed: {fit_info.get('message', 'Unknown error')}")
            return None, None, None, None, None
        
        # Display fit quality info
        r_squared = fit_info.get('r_squared', 0.0)
        outliers_removed = fit_info.get('outliers_removed', 0)
        
        if outliers_removed > 0:
            st.info(f"Removed {outliers_removed} noisy data point{'s' if outliers_removed > 1 else ''} to improve fit.")
        
        if r_squared < 0.7:
            st.warning(f"Poor fit (R² = {r_squared:.2f}). Data may not follow a psychometric pattern.")
        
        # Plot the psychometric curve
        if plot:
            y_range_lim = [0, 110]
            if n_b == 2:
                # Check if we have enough elements for double sigmoid
                if isinstance(model_boundaries, np.ndarray) and isinstance(slopes_mid, np.ndarray) and \
                   len(model_boundaries) >= 2 and len(slopes_mid) >= 2:
                    # Safely extract values, handling NaN
                    low_x0 = model_boundaries[0] if not np.isnan(model_boundaries[0]) else np.nan
                    low_slope = slopes_mid[0] if not np.isnan(slopes_mid[0]) else np.nan
                    high_x0 = model_boundaries[1] if not np.isnan(model_boundaries[1]) else np.nan
                    high_slope = slopes_mid[1] if not np.isnan(slopes_mid[1]) else np.nan
                    st.latex(
                        r"\text{Sigmoid fit (R}^2 = " + f"{r_squared:.2f}" + r"):\ \\"
                        r"\text{Low:}\ x_0 = " + (f"{low_x0:.4g}" if not np.isnan(low_x0) else r"\mathrm{N/A}") +
                        r",\quad \text{Slope} = " + (f"{low_slope:.4g}" if not np.isnan(low_slope) else r"\mathrm{N/A}") + r"\\"
                        r"\text{High:}\ x_0 = " + (f"{high_x0:.4g}" if not np.isnan(high_x0) else r"\mathrm{N/A}") +
                        r",\quad \text{Slope} = " + (f"{high_slope:.4g}" if not np.isnan(high_slope) else r"\mathrm{N/A}")
                    )
                    fig = go.Figure()
                    # Data points
                    fig.add_trace(go.Scatter(x=unique_stimuli, y=lick_rates, mode='markers', name='Data Points', marker=dict(color='black')))
                    # Overlay both fitted sigmoids
                    if isinstance(x_fit, (list, tuple)) and len(x_fit) >= 2 and \
                       isinstance(y_fit, (list, tuple)) and len(y_fit) >= 2:
                        x_fit_A, x_fit_B = x_fit[0], x_fit[1]
                        y_fit_A, y_fit_B = y_fit[0], y_fit[1]
                    else:
                        # Fallback: use single sigmoid for both
                        x_fit_A, x_fit_B = x_fit, x_fit
                        y_fit_A, y_fit_B = y_fit, y_fit
                    fig.add_trace(go.Scatter(x=x_fit_A, y=y_fit_A, mode='lines', name='Sigmoid Low', line=dict(color=colors.COLOR_LOW_BD)))
                    fig.add_trace(go.Scatter(x=x_fit_B, y=y_fit_B, mode='lines', name='Sigmoid High', line=dict(color=colors.COLOR_HIGH_BD)))
                    # Boundaries
                    fig.add_trace(go.Scatter(x=[st.session_state.low_boundary, st.session_state.low_boundary], y=y_range_lim, mode='lines', name='Low Boundary', line=dict(dash='dash', color=colors.COLOR_GRAY)))
                    fig.add_trace(go.Scatter(x=[st.session_state.high_boundary, st.session_state.high_boundary], y=y_range_lim, mode='lines', name='High Boundary', line=dict(dash='dash', color=colors.COLOR_GRAY)))
                    fig.update_layout(title='Psychometric Curve (Double Sigmoid)', xaxis_title='Frequency [kHz] (log)', xaxis_type='log', yaxis_title='Lick Rate (%)', yaxis_range=y_range_lim, showlegend=False)
                    colors.apply_standard_font_sizes(fig)
                    st.plotly_chart(fig, use_container_width=True, config=get_plotly_config())
                else:
                    # Fallback to single sigmoid display for double boundary case
                    st.warning("Double sigmoid fitting not fully implemented. Showing single sigmoid fit.")
                    # Safely extract values
                    x0_val = model_boundaries[0] if isinstance(model_boundaries, np.ndarray) and len(model_boundaries) > 0 and not np.isnan(model_boundaries[0]) else np.nan
                    slope_val = slopes_mid[0] if isinstance(slopes_mid, np.ndarray) and len(slopes_mid) > 0 and not np.isnan(slopes_mid[0]) else np.nan
                    st.latex(
                        r"\text{Sigmoid fit (R}^2 = " + f"{r_squared:.2f}" + r"):\ \\"
                        r"x_0 = " + (f"{x0_val:.4g}" if not np.isnan(x0_val) else r"\mathrm{N/A}") +
                        r",\quad \text{Slope} = " + (f"{slope_val:.4g}" if not np.isnan(slope_val) else r"\mathrm{N/A}")
                    )
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=unique_stimuli, y=lick_rates, mode='markers', name='Data Points'))
                    fig.add_trace(go.Scatter(x=x_fit, y=y_fit, mode='lines', name='Fitted Curve'))
                    # Only plot the relevant boundary line
                    boundary = st.session_state.low_boundary
                    fig.add_trace(go.Scatter(x=[boundary, boundary], y=y_range_lim, mode='lines', name='Boundary', line=dict(dash='dash', color=colors.COLOR_GRAY)))
                    # Only use log axis if all x > 0
                    x_fit_arr = np.array(x_fit) if isinstance(x_fit, (list, tuple)) else x_fit
                    if np.all(unique_stimuli > 0) and np.all(x_fit_arr > 0):
                        fig.update_layout(title="Psychometric Curve", xaxis_title="Frequency [kHz] (log)", yaxis_title="Lick Rate (%)", xaxis_type='log', yaxis_range=y_range_lim, showlegend=False)
                    else:
                        fig.update_layout(title="Psychometric Curve", xaxis_title="Frequency [kHz] (log)", yaxis_title="Lick Rate (%)", yaxis_range=y_range_lim, showlegend=False)
                    colors.apply_standard_font_sizes(fig)
                    st.plotly_chart(fig, use_container_width=True, config=get_plotly_config('psychometric_curve', width=600*n_b))
            else:
                # Safely extract values for single boundary case
                x0_val = model_boundaries[0] if isinstance(model_boundaries, (list, tuple, np.ndarray)) and len(model_boundaries) > 0 and not np.isnan(model_boundaries[0]) else np.nan
                slope_val = slopes_mid[0] if isinstance(slopes_mid, (list, tuple, np.ndarray)) and len(slopes_mid) > 0 and not np.isnan(slopes_mid[0]) else np.nan
                st.latex(
                    r"\text{Sigmoid fit (R}^2 = " + f"{r_squared:.2f}" + r"):\ \\" +
                    r"x_0 = " + (f"{x0_val:.4g}" if not np.isnan(x0_val) else r"\mathrm{N/A}") +
                    r",\quad \text{Slope} = " + (f"{slope_val:.4g}" if not np.isnan(slope_val) else r"\mathrm{N/A}")
                )
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=unique_stimuli, y=lick_rates, mode='markers', name='Data Points'))
                fig.add_trace(go.Scatter(x=x_fit, y=y_fit, mode='lines', name='Fitted Curve'))
                # Only plot the relevant boundary line
                boundary = st.session_state.low_boundary
                fig.add_trace(go.Scatter(x=[boundary, boundary], y=y_range_lim, mode='lines', name='Boundary', line=dict(dash='dash', color=colors.COLOR_GRAY)))
                # Only use log axis if all x > 0
                x_fit_arr = np.array(x_fit) if isinstance(x_fit, (list, tuple)) else x_fit
                if np.all(unique_stimuli > 0) and np.all(x_fit_arr > 0):
                    fig.update_layout(title="Psychometric Curve", xaxis_title="Frequency [kHz] (log)", yaxis_title="Lick Rate (%)", xaxis_type='log', yaxis_range=y_range_lim, showlegend=False)
                else:
                    fig.update_layout(title="Psychometric Curve", xaxis_title="Frequency [kHz] (log)", yaxis_title="Lick Rate (%)", yaxis_range=y_range_lim, showlegend=False)
                colors.apply_standard_font_sizes(fig)
                st.plotly_chart(fig, use_container_width=True, config=get_plotly_config(width=600))
        return model_boundaries, slopes_mid, slopes_at_model_boundaries, x_fit, y_fit
    except ValueError as e:
        if "Insufficient data" in str(e):
            st.warning(f"Cannot fit psychometric curve: {e}")
        else:
            st.error(f"Data validation error in psychometric_curve: {e}")
        return None, None, None, None, None
    except NotImplementedError as e:
        st.error(str(e))
        return None, None, None, None, None
    except Exception as e:
        st.error(f"Error in psychometric_curve: {e}")
        return None, None, None, None, None

# -------------------------------------------------------------------
# MULTIPLE SESSIONS
# -------------------------------------------------------------------
def psychometric_curve_multiple_sessions(selected_data, animal_name = "None", plot=False):
    """
    Plots the progression of the slope at the boundary across multiple sessions for a selected animal.
    """
    # Reset index to ensure positional indexing works correctly
    selected_data = selected_data.reset_index(drop=True)
    
    if animal_name == "None":
        animal_name = st.selectbox("Choose an Animal", selected_data["MouseName"].unique(), key="slope_animal_select")
    session_indices, _ = get_sessions_for_animal(selected_data, animal_name)
    low_slopes, high_slopes, tones, n_bounds = [], [], [], []
    valid_session_indices = []  # Track which sessions are actually processed
    
    for idx in session_indices:
        session_type = selected_data.at[idx, "Notes"]
        if "TA" in session_type or "Discrimination" in session_type:
            continue
            
        valid_session_indices.append(idx)  # Only add sessions that pass the filter
        N_Boundaries = selected_data.at[idx, "N_Boundaries"]
        boundaries, slopes_mid, slopes_at_bnds, x_fit, y_fit = psychometric_curve(selected_data, idx, plot = False)
        # Robustly ensure slopes_mid is always a numpy array of length 2 for safe indexing
        if slopes_mid is None:
            slopes_mid = np.array([np.nan, np.nan])
        elif isinstance(slopes_mid, float):
            slopes_mid = np.array([slopes_mid, np.nan])
        elif isinstance(slopes_mid, (list, np.ndarray)):
            slopes_mid = np.array(slopes_mid, dtype=float)
            if slopes_mid.size == 1:
                slopes_mid = np.array([slopes_mid[0], np.nan])
            elif slopes_mid.size == 0:
                slopes_mid = np.array([np.nan, np.nan])
            elif slopes_mid.size > 2:
                slopes_mid = np.concatenate([slopes_mid[:2], np.full(slopes_mid.size-2, np.nan)])[:2]
        else:
            slopes_mid = np.array([np.nan, np.nan])
        
        # Safe indexing with bounds checking
        low = slopes_mid[0] if len(slopes_mid) > 0 else np.nan
        high = slopes_mid[1] if len(slopes_mid) > 1 else np.nan

        low_slopes.append(low)
        high_slopes.append(high)
        tones.append(selected_data.at[idx, "Tones_per_class"])
        n_bounds.append(selected_data.at[idx, "N_Boundaries"])

    # Check if we have any valid sessions
    if len(valid_session_indices) == 0:
        st.warning(f"No valid sessions found for {animal_name} (all sessions are TA or Discrimination)")
        return np.array([])
    
    # ── tidy dataframe for plotting ────────────────────────────────────────────────
    df = (
        pd.DataFrame(
            dict(
                Session = np.arange(1, len(valid_session_indices) + 1),
                Low = np.abs(low_slopes),
                High = high_slopes,
                tones_per_class = tones,
                Boundaries = n_bounds,
            )
        )
        .melt(
            id_vars = ["Session", "tones_per_class", "Boundaries"],
            value_vars = ["Low", "High"],
            var_name = "Boundary",
            value_name = "Slope",
        )
     )
    # Transform y to log(abs(slope)) for plotting (avoid log(0))
    df["Slope_log"] = np.log(np.abs(df["Slope"]) + 1e-10)
    # ── plotting with Plotly ────────────────────────────────────────────────────────
    if plot:
        fig = go.Figure()
        
        # Color mapping for boundaries
        boundary_colors = {
            "Low": colors.COLOR_LOW_BD,
            "High": colors.COLOR_HIGH_BD
        }
        # Symbol mapping for N_Boundaries (1=circle-open, 2=circle)
        boundary_symbols = {1: "circle-open", 2: "circle"}
        
        for boundary_type in df["Boundary"].unique():
            df_boundary = df[df["Boundary"] == boundary_type]
            color = boundary_colors.get(boundary_type, colors.COLOR_ACCENT)
            
            # Add line trace (y = log(|slope|))
            fig.add_trace(go.Scatter(
                x=df_boundary["Session"],
                y=df_boundary["Slope_log"],
                mode='lines+markers',
                name=boundary_type,
                line=dict(color=color, width=2),
                marker=dict(
                    size=10,
                    color=[color if nb == 2 else 'white' for nb in df_boundary["Boundaries"]],
                    symbol=[boundary_symbols.get(nb, "circle") for nb in df_boundary["Boundaries"]],
                    line=dict(color=color, width=2)
                ),
                hovertemplate=(
                    "Session: %{x}<br>"
                    "log(|slope|): %{y:.3f}<br>"
                    "|slope|: %{customdata[2]:.4f}<br>"
                    "Tones/class: %{customdata[0]}<br>"
                    "Boundaries: %{customdata[1]}<extra></extra>"
                ),
                customdata=np.column_stack([df_boundary["tones_per_class"].values, df_boundary["Boundaries"].values, df_boundary["Slope"].values])
            ))
        
        fig.update_layout(
            title=None,
            xaxis_title="Session index",
            yaxis_title="log(|slope|)",
            yaxis=dict(type="linear"),
            height=350,
            margin=dict(l=40, r=10, t=10, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            hovermode="x unified"
        )
        colors.apply_standard_font_sizes(fig)
        
        st.markdown(f"**Slope progression – {animal_name}**")
        st.plotly_chart(fig, use_container_width=True, config=get_plotly_config('slope_progression'))
    # ── numeric return (n_sessions × 2) ─────────────────────────────────────────
    return np.column_stack([low_slopes, high_slopes])

def correlation_log_slope_vs_dprime_multi_sessions(project_data, animal_name=None):
    """
    Correlation log(|slope|) vs d' across sessions.
    - Single animal (one MouseName or animal_name given): scatter of sessions with linear fit; report r, p.
    - Multiple animals: box plot of r (one r per animal). Requires precomputed Psychometric_slope_* columns.
    """
    slope_cols = ["Psychometric_slope_low", "Psychometric_slope_high"]
    has_slopes = any(c in project_data.columns for c in slope_cols)
    if not has_slopes or "d_prime" not in project_data.columns:
        st.info("Need columns: d_prime and at least one of Psychometric_slope_low / Psychometric_slope_high. Run 'Compute Metrics' on the global dataset.")
        return
    if animal_name is not None:
        project_data = project_data[project_data["MouseName"] == animal_name].copy()
    animals = project_data["MouseName"].dropna().unique() if "MouseName" in project_data.columns else []
    n_animals = len(animals)

    def _session_slope(row):
        low = row.get("Psychometric_slope_low")
        high = row.get("Psychometric_slope_high")
        low = float(low) if pd.notna(low) and np.isfinite(low) else None
        high = float(high) if pd.notna(high) and np.isfinite(high) else None
        if low is not None and high is not None:
            return (low + high) / 2.0
        return low if low is not None else high

    try:
        has_r2 = "Psychometric_r_squared" in project_data.columns
        filter_low_r2 = False
        r2_threshold = 0.7
        if has_r2:
            filter_col_a, filter_col_b = st.columns(2)
            with filter_col_a:
                filter_low_r2 = st.checkbox(
                    "Filter out low R² sessions",
                    value=False,
                    key="corr_filter_low_r2_sessions",
                    help="Exclude sessions below the psychometric fit R² threshold",
                )
            with filter_col_b:
                r2_threshold = st.number_input(
                    "R² minimum",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.7,
                    step=0.05,
                    key="corr_r2_threshold_sessions",
                    disabled=not filter_low_r2,
                )
        if has_r2 and filter_low_r2:
            project_data = project_data.copy()
            project_data["Psychometric_r_squared"] = pd.to_numeric(project_data["Psychometric_r_squared"], errors="coerce")
            project_data = project_data[project_data["Psychometric_r_squared"] >= r2_threshold]

        project_data = project_data.copy()
        project_data["d_prime"] = pd.to_numeric(project_data["d_prime"], errors="coerce")
        if "Psychometric_slope_low" in project_data.columns:
            project_data["Psychometric_slope_low"] = pd.to_numeric(project_data["Psychometric_slope_low"], errors="coerce")
        if "Psychometric_slope_high" in project_data.columns:
            project_data["Psychometric_slope_high"] = pd.to_numeric(project_data["Psychometric_slope_high"], errors="coerce")
        project_data["_slope"] = project_data.apply(_session_slope, axis=1)
        project_data = project_data.dropna(subset=["d_prime", "_slope"])
        project_data["_log_abs_slope"] = np.log(np.abs(project_data["_slope"]) + 1e-10)

        if n_animals <= 1:
            # Single animal: scatter + regression line
            if len(project_data) < 2:
                st.info("Need at least 2 sessions with valid d' and slope for this animal.")
                return
            x = project_data["d_prime"].values
            y = project_data["_log_abs_slope"].values
            slope_lr, intercept_lr, r_val, p_val, _ = stats.linregress(x, y)
            x_line = np.linspace(x.min(), x.max(), 50)
            y_line = slope_lr * x_line + intercept_lr
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Sessions", marker=dict(size=10, color=colors.COLOR_ORANGE)))
            fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines", name="Linear fit", line=dict(dash="dash", color=colors.COLOR_GRAY)))
            fig.update_layout(
                title="log(|slope|) vs d'",
                xaxis_title="d'",
                yaxis_title="log(|slope|)",
                height=400,
                showlegend=True,
            )
            colors.apply_standard_font_sizes(fig)
            st.plotly_chart(fig, use_container_width=True, config=get_plotly_config("correlation_slope_dprime_sessions"))
            st.caption(f"Linear regression: r = {r_val:.3f}, p = {p_val:.4f}, n = {len(project_data)} sessions.")
        else:
            # Multiple animals: box plot of r
            st.caption("Linear regression of log(|slope|) on d' for each animal; box plot shows distribution of r across animals.")
            r_values = []
            for animal in animals:
                df_animal = project_data[project_data["MouseName"] == animal].copy()
                df_animal = df_animal.dropna(subset=["d_prime", "_slope"])
                if len(df_animal) >= 2:
                    r_val = stats.linregress(df_animal["d_prime"].values, df_animal["_log_abs_slope"].values)[2]
                    r_values.append(r_val)
            if r_values:
                fig_r = go.Figure()
                fig_r.add_trace(go.Box(y=r_values, name="r (log|slope| vs d')", boxmean="sd"))
                fig_r.update_layout(title="Distribution of correlation r across animals", yaxis_title="r", height=400, showlegend=False)
                colors.apply_standard_font_sizes(fig_r)
                st.plotly_chart(fig_r, use_container_width=True, config=get_plotly_config("correlation_r_boxplot"))
                st.caption(f"n = {len(r_values)} animals with ≥2 sessions; median r = {np.median(r_values):.3f}.")
            else:
                st.info("No animal had ≥2 sessions with valid d' and slope.")
    except Exception as e:
        st.warning(f"Something went wrong with correlation plot :|\n\n{e}")


# -------------------------------------------------------------------
# MULTIPLE ANIMALS
# -------------------------------------------------------------------
def multi_animal_psychometric_slope_progression(selected_data, N_Boundaries=1):
    df = selected_data.copy()  
    if N_Boundaries is not None:
        df = (
            selected_data
            .loc[selected_data["N_Boundaries"] == N_Boundaries]
            .reset_index(drop=True)
        )
    # ─── parse stimuli strings to arrays ───────────────────────────────
    def parse_stimuli(s):
        try:
            return np.fromstring(s.strip("[]"), sep=" ")
        except:
            return np.array([])
    
    # Handle both possible column names for stimuli data
    if "Unique_Stimuli_Values" in df.columns:
        df["Parsed_Stimuli"] = df["Unique_Stimuli_Values"].apply(parse_stimuli)
    elif "Stimuli" in df.columns:
        df["Parsed_Stimuli"] = df["Stimuli"].apply(parse_stimuli)
    else:
        st.error("Neither 'Unique_Stimuli_Values' nor 'Stimuli' column found in data")
        return


    # ─── compute slopes for each subject & session ─────────────────────
    records = []
    for subj in df["MouseName"].unique():
        slopes = psychometric_curve_multiple_sessions(df, animal_name=subj, plot=False)
        # Handle case where no valid sessions are found
        if slopes.size == 0:
            continue
        for sess_idx, (low, high) in enumerate(slopes, start=1):
            records.append({"Mouse": subj, "Session": sess_idx, "Boundary": "Low",  "Slope": np.abs(low)})
            records.append({"Mouse": subj, "Session": sess_idx, "Boundary": "High", "Slope": high})
    
    # Check if we have any records
    if not records:
        st.warning("No valid sessions found for any animal")
        return
        
    long_df = pd.DataFrame(records)
    # Transform to log(|slope|) for plotting (avoid log(0))
    long_df["Slope_log"] = np.log(np.abs(long_df["Slope"]) + 1e-10)

    # ─── compute session‐wise average per boundary ────────────────────
    avg_df = (
        long_df
        .groupby(["Session", "Boundary"], as_index=False)["Slope_log"]
        .mean()
        .assign(Mouse="Average")
    )

    import plotly.graph_objects as go
    fig = go.Figure()
    # Build color map
    try:
        color_map = st.session_state.get('mouse_color_map', {})
        if not color_map:
            from Analysis.GNG_bpod_analysis.colors import get_subject_color_map
            color_map = get_subject_color_map(df["MouseName"])  # type: ignore
    except Exception:
        color_map = {}

    # Plot all animals (colored per mouse); y = log(|slope|)
    for subj in df["MouseName"].unique():
        for boundary, color in zip(["Low", "High"], [colors.COLOR_LOW_BD, colors.COLOR_HIGH_BD]):
            subj_data = long_df[(long_df["Mouse"] == subj) & (long_df["Boundary"] == boundary)]
            if not subj_data.empty:
                fig.add_trace(go.Scatter(
                    x=subj_data["Session"],
                    y=subj_data["Slope_log"],
                    mode='lines',
                    line=dict(color=color_map.get(str(subj), colors.COLOR_SUBTLE), width=2, dash='solid'),
                    name=f'{subj} {boundary}',
                    showlegend=False,
                    opacity=0.8
                ))
    # Plot average lines (bold)
    for boundary, color in zip(["Low", "High"], [colors.COLOR_LOW_BD, colors.COLOR_HIGH_BD]):
        avg_data = avg_df[avg_df["Boundary"] == boundary]
        if not avg_data.empty:
            fig.add_trace(go.Scatter(
                x=avg_data["Session"],
                y=avg_data["Slope_log"],
                mode='lines',
                line=dict(color=color, width=4),
                name=f'Average {boundary}',
                marker=dict(symbol='circle')
            ))
    # Reference at log(1)=0 (i.e. |slope|=1)
    fig.add_trace(go.Scatter(
        x=[long_df["Session"].min(), long_df["Session"].max()],
        y=[0, 0],
        mode='lines',
        name="Learning Threshold (|slope|=1)",
        line=dict(color=colors.COLOR_GRAY, dash='dash'),
        showlegend=True
    ))
    fig.update_layout(
        xaxis_title="Session Index",
        yaxis_title="log(|slope|)",
        yaxis=dict(type="linear"),
        title="Psychometric Slope Progression - all animals",
        legend=dict(title="Legend"),
        height=400,
        width=700
    )
    fig.update_yaxes(autorange=True)
    colors.apply_standard_font_sizes(fig)
    st.plotly_chart(fig, use_container_width=True, config=get_plotly_config('psychometric_curve_multiple_sessions'))

    return long_df, avg_df




def correlation_log_slope_vs_dprime_multi_animal(project_data):
    """
    Correlation log(|slope|) vs d': pool all sessions from all animals, apply filters, then single linear regression.
    Shows scatter (d' vs log|slope|) with regression line and reports r, p, n sessions.
    """
    slope_cols = ["Psychometric_slope_low", "Psychometric_slope_high"]
    has_slopes = any(c in project_data.columns for c in slope_cols)
    if not has_slopes or "d_prime" not in project_data.columns:
        st.info("Need columns: d_prime and at least one of Psychometric_slope_low / Psychometric_slope_high. Run 'Compute Metrics' on the global dataset.")
        return
    if "MouseName" not in project_data.columns:
        return
    try:
        st.subheader("Correlation log(|slope|) vs d' (all sessions)")
        st.caption("All sessions from all animals pooled; linear regression of log(|slope|) on d'.")
        has_r2 = "Psychometric_r_squared" in project_data.columns
        filter_low_r2 = True
        r2_threshold = 0.8
        if has_r2:
            filter_col_a, filter_col_b = st.columns(2)
            with filter_col_a:
                filter_low_r2 = st.checkbox(
                    "Filter out low R² sessions",
                    value=True,
                    key="corr_filter_low_r2_multi",
                    help="Exclude sessions below the psychometric fit R² threshold",
                )
            with filter_col_b:
                r2_threshold = st.number_input(
                    "R² minimum",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.8,
                    step=0.05,
                    key="corr_r2_threshold_multi",
                    disabled=not filter_low_r2,
                )

        def _session_slope(row):
            low = row.get("Psychometric_slope_low")
            high = row.get("Psychometric_slope_high")
            low = float(low) if pd.notna(low) and np.isfinite(low) else None
            high = float(high) if pd.notna(high) and np.isfinite(high) else None
            if low is not None and high is not None:
                return (low + high) / 2.0
            return low if low is not None else high

        df = project_data.copy()
        df["d_prime"] = pd.to_numeric(df["d_prime"], errors="coerce")
        if "Psychometric_slope_low" in df.columns:
            df["Psychometric_slope_low"] = pd.to_numeric(df["Psychometric_slope_low"], errors="coerce")
        if "Psychometric_slope_high" in df.columns:
            df["Psychometric_slope_high"] = pd.to_numeric(df["Psychometric_slope_high"], errors="coerce")
        if has_r2 and filter_low_r2:
            df["Psychometric_r_squared"] = pd.to_numeric(df["Psychometric_r_squared"], errors="coerce")
            df = df[df["Psychometric_r_squared"] >= r2_threshold]
        df["_slope"] = df.apply(_session_slope, axis=1)
        df = df.dropna(subset=["d_prime", "_slope"])
        df["_log_abs_slope"] = np.log(np.abs(df["_slope"]) + 1e-10)

        if len(df) < 2:
            st.info("Need at least 2 sessions with valid d' and slope after filters.")
            return

        x = df["d_prime"].values
        y = df["_log_abs_slope"].values
        slope_lr, intercept_lr, r_val, p_val, _ = stats.linregress(x, y)
        x_line = np.linspace(x.min(), x.max(), 50)
        y_line = slope_lr * x_line + intercept_lr

        fig = go.Figure()
        # Scatter: one point per session; color by animal if few animals
        animals = df["MouseName"].dropna().unique()
        if len(animals) <= 12:
            for i, animal in enumerate(animals):
                sub = df[df["MouseName"] == animal]
                fig.add_trace(go.Scatter(
                    x=sub["d_prime"],
                    y=sub["_log_abs_slope"],
                    mode="markers",
                    name=str(animal),
                    marker=dict(size=8),
                ))
        else:
            fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Sessions", marker=dict(size=8, color=colors.COLOR_GRAY)))
        fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines", name="Linear fit", line=dict(width=4, dash="dash", color=colors.COLOR_ACCENT)))
        fig.update_layout(
            title="log(|slope|) vs d' (all sessions)",
            xaxis_title="d'",
            yaxis_title="log(|slope|)",
            height=400,
            showlegend=len(animals) <= 12,
        )
        colors.apply_standard_font_sizes(fig)
        st.plotly_chart(fig, use_container_width=True, config=get_plotly_config("correlation_slope_dprime_multi"))
        n_animals = df["MouseName"].nunique()
        st.caption(f"Linear regression: r = {r_val:.3f}, p = {p_val}, n = {len(df)} sessions, {n_animals} animals.")
    except Exception as e:
        st.warning(f"Something went wrong with correlation plot :|\n\n{e}")


# Default task boundaries (kHz) when not in Streamlit
_DEFAULT_LOW_BOUNDARY = 0.983
_DEFAULT_HIGH_BOUNDARY = 1.525


def _get_boundaries():
    """Return (low_boundary, high_boundary) from session state or defaults."""
    try:
        low = getattr(st.session_state, "low_boundary", _DEFAULT_LOW_BOUNDARY)
        high = getattr(st.session_state, "high_boundary", _DEFAULT_HIGH_BOUNDARY)
        return float(low), float(high)
    except Exception:
        return _DEFAULT_LOW_BOUNDARY, _DEFAULT_HIGH_BOUNDARY


def distance_between_x0_and_boundary(project_data):
    """
    For each session: compute distance between fitted x0 and the true task boundary;
    for 2B, also report whether both x0s still fall between the two boundaries (bool).

    Adds columns to a copy of project_data:
    - 1B: Distance_x0 (abs difference to low boundary), In_between_boundaries (True).
    - 2B: Distance_x0_low, Distance_x0_high; In_between_boundaries_low (x0_low in [low_bound, high_bound]),
      In_between_boundaries_high (x0_high in [low_bound, high_bound]), In_between_boundaries (both True).

    Returns the DataFrame with new columns.
    """
    low_bound, high_bound = _get_boundaries()
    df = project_data.copy()
    df["Distance_x0"] = np.nan
    df["Distance_x0_low"] = np.nan
    df["Distance_x0_high"] = np.nan
    df["In_between_boundaries_low"] = False
    df["In_between_boundaries_high"] = False
    df["In_between_boundaries"] = False

    for idx in df.index:
        n_b = df.at[idx, "N_Boundaries"]
        try:
            n_b = int(n_b) if pd.notna(n_b) and np.isfinite(n_b) else 1
        except (TypeError, ValueError):
            n_b = 1
        if n_b == 1:
            x0 = df.at[idx, "Psychometric_x0"]
            if pd.notna(x0) and np.isfinite(x0):
                df.at[idx, "Distance_x0"] = np.abs(float(x0) - low_bound)
            df.at[idx, "In_between_boundaries_low"] = True
            df.at[idx, "In_between_boundaries_high"] = True
            df.at[idx, "In_between_boundaries"] = True
        else:
            x0_low = df.at[idx, "Psychometric_x0_low"]
            x0_high = df.at[idx, "Psychometric_x0_high"]
            if pd.notna(x0_low) and np.isfinite(x0_low):
                df.at[idx, "Distance_x0_low"] = np.abs(float(x0_low) - low_bound)
                df.at[idx, "In_between_boundaries_low"] = (low_bound <= float(x0_low) <= high_bound)
            if pd.notna(x0_high) and np.isfinite(x0_high):
                df.at[idx, "Distance_x0_high"] = np.abs(float(x0_high) - high_bound)
                df.at[idx, "In_between_boundaries_high"] = (low_bound <= float(x0_high) <= high_bound)
            df.at[idx, "In_between_boundaries"] = (
                df.at[idx, "In_between_boundaries_low"] and df.at[idx, "In_between_boundaries_high"]
            )
    return df


def _hit_rate_by_stimulus_region(project_data, idx, low_bound: float, high_bound: float):
    """
    For one session row: hit rate on trials with stimulus below low_bound, and hit rate on trials with stimulus above high_bound.
    Returns (hit_rate_below_low, hit_rate_above_high); each is in [0, 1] or np.nan if no trials.
    """
    import ast
    try:
        if "Stimuli" not in project_data.columns or "Outcomes" not in project_data.columns:
            return np.nan, np.nan
        row = project_data.loc[idx]
        stim_raw = row.get("Stimuli")
        out_raw = row.get("Outcomes")
        if pd.isna(stim_raw) or pd.isna(out_raw):
            return np.nan, np.nan
        if isinstance(stim_raw, str):
            stimuli = np.array([float(x) for x in stim_raw.strip("[]\n").split()])
        else:
            stimuli = np.asarray(stim_raw, dtype=float)
        outcomes = np.array(ast.literal_eval(out_raw)) if isinstance(out_raw, str) else np.asarray(out_raw)
        if len(stimuli) != len(outcomes) or len(stimuli) == 0:
            return np.nan, np.nan
        out_str = np.array([str(o).strip() for o in outcomes])
        hit = (out_str == "Hit")
        miss = (out_str == "Miss")
        below = stimuli < low_bound
        above = stimuli > high_bound
        n_below_go = np.sum(below)
        n_above_go = np.sum(above)
        hits_below = np.sum(hit & below)
        hits_above = np.sum(hit & above)
        hr_below = (hits_below / n_below_go) if n_below_go > 0 else np.nan
        hr_above = (hits_above / n_above_go) if n_above_go > 0 else np.nan
        return float(hr_below), float(hr_above)
    except Exception:
        return np.nan, np.nan


def compare_slope_and_distance_by_boundary(project_data, plot: bool = False):
    """
    Long-format comparison: log(|slope|) and distance by boundary (2B: Low vs High),
    with Hit_Rate per region (below low boundary / above high boundary) and overall Hit_Rate.

    When plot=True, renders boxplots (log(|slope|), Distance, Hit_Rate by Boundary) in Streamlit.

    Returns DataFrame with columns: Boundary, Slope (log(|slope|)), Distance, Hit_Rate, Hit_Rate_below_low, Hit_Rate_above_high, ...
    """
    df = distance_between_x0_and_boundary(project_data)
    low_bound, high_bound = _get_boundaries()

    hit_col = "Hit_Rate"
    if hit_col not in df.columns:
        df[hit_col] = np.nan

    rows = []
    for idx in df.index:
        n_b = df.at[idx, "N_Boundaries"]
        try:
            n_b = int(n_b) if pd.notna(n_b) and np.isfinite(n_b) else 1
        except (TypeError, ValueError):
            n_b = 1
        hit_rate = df.at[idx, hit_col]
        hit_rate = float(hit_rate) if pd.notna(hit_rate) and np.isfinite(hit_rate) else np.nan
        hit_rate_below_low, hit_rate_above_high = np.nan, np.nan
        if n_b == 2 and "Stimuli" in df.columns and "Outcomes" in df.columns:
            hit_rate_below_low, hit_rate_above_high = _hit_rate_by_stimulus_region(project_data, idx, low_bound, high_bound)
        def _log_abs_slope(s):
            if s is None or pd.isna(s) or not np.isfinite(s) or float(s) == 0:
                return np.nan
            return float(np.log(np.abs(float(s))))

        base = {"Hit_Rate": hit_rate, "Hit_Rate_below_low": hit_rate_below_low, "Hit_Rate_above_high": hit_rate_above_high, "index": idx}
        for c in ["MouseName", "SessionDate", "N_Boundaries", "In_between_boundaries", "Psychometric_r_squared"]:
            if c in df.columns:
                base[c] = df.at[idx, c]
        if n_b == 1:
            s_low = df.at[idx, "Psychometric_slope_low"] if "Psychometric_slope_low" in df.columns else np.nan
            rows.append({
                **base,
                "Boundary": "Single",
                "Slope": _log_abs_slope(s_low),
                "Distance": df.at[idx, "Distance_x0"],
            })
        else:
            s_low = df.at[idx, "Psychometric_slope_low"] if "Psychometric_slope_low" in df.columns else np.nan
            s_high = df.at[idx, "Psychometric_slope_high"] if "Psychometric_slope_high" in df.columns else np.nan
            rows.append({
                **base,
                "Boundary": "Low",
                "Slope": _log_abs_slope(s_low),
                "Distance": df.at[idx, "Distance_x0_low"],
            })
            rows.append({
                **base,
                "Boundary": "High",
                "Slope": _log_abs_slope(s_high),
                "Distance": df.at[idx, "Distance_x0_high"],
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    cols = ["Boundary", "Slope", "Distance", "Hit_Rate", "Hit_Rate_below_low", "Hit_Rate_above_high"]
    order_cols = [c for c in cols if c in out.columns]
    rest = [c for c in out.columns if c not in order_cols]
    out = out[order_cols + rest]

    if plot and "Boundary" in out.columns:
        from plotly.subplots import make_subplots

        boundary_vals = np.asarray(out["Boundary"]).ravel()
        boundaries = pd.unique(pd.Series(boundary_vals).dropna())
        if len(boundaries) == 0:
            return out

        # Filters: only successful sessions (d' & Hit Rate), then Min R²
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        with filter_col1:
            only_successful = st.checkbox(
                "Only successful sessions",
                value=False,
                help="Restrict to sessions with d' and Hit Rate above thresholds.",
                key="compare_slope_distance_only_successful",
            )
        dprime_threshold = 1.0
        hit_rate_threshold = 0.8
        if "d_prime" in df.columns and "Hit_Rate" in df.columns:
            with filter_col2:
                dprime_threshold = st.number_input("Min d'", 0.0, 5.0, 1.0, 0.1, key="compare_slope_distance_dprime")
            with filter_col3:
                hit_rate_threshold = st.number_input("Min Hit Rate", 0.0, 1.0, 0.8, 0.05, key="compare_slope_distance_hr")
        else:
            only_successful = False

        if only_successful and "d_prime" in df.columns and "Hit_Rate" in df.columns:
            successful_idx = set(
                df.index[
                    (pd.to_numeric(df["d_prime"], errors="coerce") > dprime_threshold)
                    & (pd.to_numeric(df["Hit_Rate"], errors="coerce") > hit_rate_threshold)
                ]
            )
            out_filtered = out[out["index"].isin(successful_idx)]
            st.caption(f"Using {len(successful_idx)} successful sessions (d' > {dprime_threshold}, Hit Rate > {hit_rate_threshold*100:.0f}%).")
        else:
            out_filtered = out

        r2_col = "Psychometric_r_squared"
        r2_threshold = 0.7
        if r2_col in out_filtered.columns:
            r2_threshold = st.number_input("Min R² (goodness of fit)", 0.0, 1.0, 0.7, 0.05, key="compare_slope_distance_r2_min")
            out_plot = out_filtered[pd.to_numeric(out_filtered[r2_col], errors="coerce") >= r2_threshold]
        else:
            out_plot = out_filtered
        if out_plot.empty:
            st.info("No rows left after filters.")
            return out
        n_rows = len(out_plot.index)
        bcol = np.asarray(out_plot["Boundary"])
        if bcol.size > n_rows:
            boundary_vals_1d = np.asarray(bcol).reshape(n_rows, -1)[:, 0]
        else:
            boundary_vals_1d = np.asarray(bcol).ravel()
        boundary_mask_series = pd.Series(boundary_vals_1d, index=out_plot.index, dtype=object)
        plot_boundaries = [b for b in boundaries if b in ("Low", "High")]
        if not plot_boundaries:
            st.info("No Low or High boundary rows to plot.")
            return out
        fig = make_subplots(rows=1, cols=3, subplot_titles=("log(|slope|)", "Distance", "Hit rate"), horizontal_spacing=0.08)
        for col, (metric_key, y_label) in enumerate(
            [("Slope", "log(|slope|)"), ("Distance", "Distance"), ("Hit_Rate", "Hit rate")], start=1
        ):
            for b in plot_boundaries:
                sub = out_plot.loc[boundary_mask_series == b]
                color = colors.COLOR_LOW_BD if b == "Low" else colors.COLOR_HIGH_BD
                y_vals = sub[metric_key].dropna()
                if len(y_vals) > 0:
                    fig.add_trace(
                        go.Box(y=y_vals, x=[str(b)] * len(y_vals), name=b, marker_color=color, showlegend=(col == 1)),
                        row=1, col=col,
                    )
            fig.update_yaxes(title_text=y_label, row=1, col=col)
        fig.update_layout(height=400, margin=dict(l=40, r=20, t=50, b=40), boxmode="group")
        colors.apply_standard_font_sizes(fig)
        st.plotly_chart(fig, use_container_width=True, config=get_plotly_config("compare_slope_distance_boundary"))

        # Statistical analysis: Low vs High for each metric (Mann-Whitney, Bonferroni)
        with st.expander("Statistical analysis (Low vs High)", expanded=False):
            from statsmodels.stats.multitest import multipletests

            stat_rows = []
            p_raw_list = []
            for metric_key, metric_label in [("Slope", "log(|slope|)"), ("Distance", "Distance"), ("Hit_Rate", "Hit rate")]:
                low_vals = out_plot.loc[boundary_mask_series == "Low", metric_key].dropna().values
                high_vals = out_plot.loc[boundary_mask_series == "High", metric_key].dropna().values
                if len(low_vals) >= 3 and len(high_vals) >= 3:
                    stat, p = stats.mannwhitneyu(low_vals, high_vals, alternative="two-sided")
                    p_raw_list.append(p)
                    r = 1 - (2 * stat) / (len(low_vals) * len(high_vals))
                    stat_rows.append({
                        "Metric": metric_label,
                        "n (Low)": len(low_vals),
                        "n (High)": len(high_vals),
                        "Median (Low)": round(float(np.median(low_vals)), 4),
                        "Median (High)": round(float(np.median(high_vals)), 4),
                        "U": round(float(stat), 1),
                        "p_raw": p,
                        "r": round(r, 3),
                    })
            if p_raw_list:
                _, p_adj, _, _ = multipletests(p_raw_list, method="bonferroni")
                for i, row in enumerate(stat_rows):
                    row["p-adj (Bonferroni)"] = f"{p_adj[i]:.4f}"
                    p = p_adj[i]
                    row["Sig."] = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                st.dataframe(pd.DataFrame(stat_rows), use_container_width=True, hide_index=True)
                st.caption("Mann-Whitney U, Low vs High. p-adj: Bonferroni across 3 metrics. *** p<0.001, ** p<0.01, * p<0.05, ns = not significant.")
            else:
                st.caption("Need ≥3 samples in both Low and High per metric to run tests.")

    return out

