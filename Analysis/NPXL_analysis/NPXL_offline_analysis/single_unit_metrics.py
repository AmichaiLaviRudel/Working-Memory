"""
Single unit metrics computation functions.

This module contains all the core computation functions for single unit analysis,
including selectivity metrics, PSTH metrics, GLM fitting, and statistical tests.
"""
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd
import streamlit as st
import hashlib


def compute_stimulus_selectivity(event_windows_data, stimuli_outcome_df, unit_idx, window=(-0.2, 0.5)):
    """
    Compute stimulus selectivity for a single unit.
    Returns frequency tuning curve, SEM, and best frequency.
    """
    if 'stimulus' not in stimuli_outcome_df.columns:
        return None, None, None, None
    
    unique_stimuli = np.unique(stimuli_outcome_df['stimulus'])
    tuning_curve = []
    tuning_sem = []
    
    # Extract data from event_windows_data
    event_windows_matrix, time_axis, valid_event_indices, event_stimuli_outcome_df, metadata = event_windows_data
    
    for stim in unique_stimuli:
        # Get trials with this stimulus
        stim_mask = (stimuli_outcome_df['stimulus'] == stim).values
        if np.sum(stim_mask) == 0:
            tuning_curve.append(0)
            tuning_sem.append(0)
            continue

        # Get the unit's data for trials with this stimulus
        stim_trials = np.where(stim_mask)[0]
        if len(stim_trials) == 0:
            tuning_curve.append(0)
            tuning_sem.append(0)
            continue
            
        # Find the time indices corresponding to the window
        start_time, end_time = window
        start_idx = np.argmin(np.abs(time_axis - start_time))
        end_idx = np.argmin(np.abs(time_axis - end_time))
        
        # Get the unit's data for the specified window and stimulus trials
        unit_data = event_windows_matrix[unit_idx, start_idx:end_idx, stim_trials]  # [time × trials]
        
        # Average across time bins for each trial
        trial_rates = np.mean(unit_data, axis=0)  # Average across time for each trial
        
        if len(trial_rates) > 0:
            avg_rate = np.mean(trial_rates)
            sem = np.std(trial_rates) / np.sqrt(len(trial_rates))
        else:
            avg_rate = 0
            sem = 0
        
        tuning_curve.append(avg_rate)
        tuning_sem.append(sem)
    
    # Find best frequency (stimulus that elicits highest response)
    if len(tuning_curve) > 0 and np.any(np.array(tuning_curve) > 0):
        best_stim_idx = np.argmax(tuning_curve)
        best_stimulus = unique_stimuli[best_stim_idx]
    else:
        best_stimulus = unique_stimuli[0] if len(unique_stimuli) > 0 else None
    
    return unique_stimuli, tuning_curve, tuning_sem, best_stimulus


def compute_go_nogo_coding(event_windows_data, stimuli_outcome_df, unit_idx, window=(-0.1, 0.5)):
    """
    Compute d' and ROC AUC for Go vs NoGo discrimination.
    """
    # Extract data from event_windows_data
    event_windows_matrix, time_axis, valid_event_indices, event_stimuli_outcome_df, metadata = event_windows_data
    
    # Separate Go and NoGo trials
    go_mask = np.isin(stimuli_outcome_df['outcome'], ['Hit', 'Miss'])
    nogo_mask = np.isin(stimuli_outcome_df['outcome'], ['False Alarm', 'CR'])
    
    if np.sum(go_mask) == 0 or np.sum(nogo_mask) == 0:
        return None, None, None
    
    # Find the time indices corresponding to the window
    start_time, end_time = window
    start_idx = np.argmin(np.abs(time_axis - start_time))
    end_idx = np.argmin(np.abs(time_axis - end_time))
    
    # Get the unit's data for the specified window
    unit_data = event_windows_matrix[unit_idx, start_idx:end_idx, :]  # [time × trials]
    
    # Average across time bins for each trial
    trial_rates = np.mean(unit_data, axis=0)  # Average across time for each trial
    
    # Get rates for Go and NoGo trials
    go_rates = trial_rates[go_mask]
    nogo_rates = trial_rates[nogo_mask]
    
    # Compute d'
    go_mean, go_std = np.mean(go_rates), np.std(go_rates)
    nogo_mean, nogo_std = np.mean(nogo_rates), np.std(nogo_rates)
    
    # Pooled standard deviation
    pooled_std = np.sqrt((go_std**2 + nogo_std**2) / 2)
    d_prime = (go_mean - nogo_mean) / pooled_std if pooled_std > 0 else 0
    
    # Compute ROC AUC
    try:
        # Create labels: 1 for Go, 0 for NoGo
        labels = np.concatenate([np.ones(len(go_rates)), np.zeros(len(nogo_rates))])
        scores = np.concatenate([go_rates, nogo_rates])
        roc_auc = roc_auc_score(labels, scores)
    except:
        roc_auc = 0.5
    
    return d_prime, roc_auc, (go_rates, nogo_rates)


def compute_outcome_modulation(event_windows_data, stimuli_outcome_df, unit_idx, window=(-0.1, 0.5)):
    """
    Compare responses between rewarded (Hit) and non-rewarded (Miss/FA) trials.
    """
    # Extract data from event_windows_data
    event_windows_matrix, time_axis, valid_event_indices, event_stimuli_outcome_df, metadata = event_windows_data
    
    # Separate rewarded and non-rewarded trials
    rewarded_mask = stimuli_outcome_df['outcome'] == 'Hit'
    non_rewarded_mask = np.isin(stimuli_outcome_df['outcome'], ['Miss', 'False Alarm'])
    
    if np.sum(rewarded_mask) == 0 or np.sum(non_rewarded_mask) == 0:
        return None, None, None
    
    # Find the time indices corresponding to the window
    start_time, end_time = window
    start_idx = np.argmin(np.abs(time_axis - start_time))
    end_idx = np.argmin(np.abs(time_axis - end_time))
    
    # Get the unit's data for the specified window
    unit_data = event_windows_matrix[unit_idx, start_idx:end_idx, :]  # [time × trials]
    
    # Average across time bins for each trial
    trial_rates = np.mean(unit_data, axis=0)  # Average across time for each trial
    
    # Get rates for rewarded and non-rewarded trials
    rewarded_rates = trial_rates[rewarded_mask]
    non_rewarded_rates = trial_rates[non_rewarded_mask]
    
    # Statistical test
    try:
        stat, p_value = stats.mannwhitneyu(rewarded_rates, non_rewarded_rates, alternative='two-sided')
    except:
        stat, p_value = 0, 1
    
    return p_value, (rewarded_rates, non_rewarded_rates), (np.mean(rewarded_rates), np.mean(non_rewarded_rates))


def compute_choice_probability(event_windows_data, stimuli_outcome_df, unit_idx, window=(-0.1, 0.5)):
    """
    Calculate choice probability (CP) - trial-by-trial correlation between spike counts and Go/NoGo choice.
    """
    # Extract data from event_windows_data
    event_windows_matrix, time_axis, valid_event_indices, event_stimuli_outcome_df, metadata = event_windows_data
    
    # Get Go trials only (where choice is relevant)
    go_mask = np.isin(stimuli_outcome_df['outcome'], ['Hit', 'Miss'])
    
    if np.sum(go_mask) == 0:
        return None, None
    
    # Find the time indices corresponding to the window
    start_time, end_time = window
    start_idx = np.argmin(np.abs(time_axis - start_time))
    end_idx = np.argmin(np.abs(time_axis - end_time))
    
    # Get the unit's data for the specified window
    unit_data = event_windows_matrix[unit_idx, start_idx:end_idx, :]  # [time × trials]
    
    # Average across time bins for each trial
    trial_rates = np.mean(unit_data, axis=0)  # Average across time for each trial
    
    # Get firing rates for Go trials only
    firing_rates = trial_rates[go_mask]
    
    # Create choice labels: 1 for Hit (correct Go), 0 for Miss (incorrect Go)
    go_data = stimuli_outcome_df.loc[go_mask]
    choices = (go_data['outcome'] == 'Hit').astype(int).values
    
    # Compute choice probability using ROC
    try:
        cp = roc_auc_score(choices, firing_rates)
        # Convert to correlation-like measure (-1 to 1)
        cp_corr = 2 * (cp - 0.5)
    except:
        cp = 0.5
        cp_corr = 0
    
    return cp, cp_corr


def compute_d_prime(event_windows_data, stimuli_outcome_df, unit_idx, condition1, condition2, window=(-0.1, 0.5)):
    """
    Compute d' between two conditions.
    """
    # Extract data from event_windows_data
    event_windows_matrix, time_axis, valid_event_indices, event_stimuli_outcome_df, metadata = event_windows_data
    
    # Get trials for each condition
    mask1 = stimuli_outcome_df['outcome'] == condition1
    mask2 = stimuli_outcome_df['outcome'] == condition2
    
    if np.sum(mask1) == 0 or np.sum(mask2) == 0:
        return None
    
    # Find the time indices corresponding to the window
    start_time, end_time = window
    start_idx = np.argmin(np.abs(time_axis - start_time))
    end_idx = np.argmin(np.abs(time_axis - end_time))
    
    # Get the unit's data for the specified window
    unit_data = event_windows_matrix[unit_idx, start_idx:end_idx, :]  # [time × trials]
    
    # Average across time bins for each trial
    trial_rates = np.mean(unit_data, axis=0)  # Average across time for each trial
    
    # Get rates for each condition
    rates1 = trial_rates[mask1]
    rates2 = trial_rates[mask2]
    
    # Compute d'
    mean1, std1 = np.mean(rates1), np.std(rates1)
    mean2, std2 = np.mean(rates2), np.std(rates2)
    
    pooled_std = np.sqrt((std1**2 + std2**2) / 2)
    d_prime = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
    
    return d_prime


def compute_peri_event_rate(spike_matrix, event_times, unit_idx, window=(-0.1, 0.5), bin_size=0.01):
    """
    Compute average firing rate around event times for a single unit.
    """
    if len(event_times) == 0:
        return 0
    
    n_bins = spike_matrix.shape[1]
    total_rate = 0
    valid_events = 0
    
    for t in event_times:
        event_bin = int(t / bin_size)
        start_bin = event_bin + int(window[0] / bin_size)
        end_bin = event_bin + int(window[1] / bin_size)
        
        if start_bin < 0 or end_bin > n_bins:
            continue
            
        segment = spike_matrix[unit_idx, start_bin:end_bin]
        if len(segment) > 0:
            total_rate += np.mean(segment)
            valid_events += 1
    
    return total_rate / valid_events if valid_events > 0 else 0


def compute_peri_event_rate_from_event_windows(event_windows_data, unit_idx, window=(-0.1, 0.5), bin_size=0.01):
    """
    Compute average firing rate around event times for a single unit using event windows data.
    """
    # Extract data from event_windows_data
    event_windows_matrix, time_axis, valid_event_indices, event_stimuli_outcome_df, metadata = event_windows_data
    
    # Find the time indices corresponding to the window
    start_time, end_time = window
    start_idx = np.argmin(np.abs(time_axis - start_time))
    end_idx = np.argmin(np.abs(time_axis - end_time))
    
    # Get the unit's data for the specified window
    unit_data = event_windows_matrix[unit_idx, start_idx:end_idx, :]  # [time × trials]
    # Average across time bins and trials
    if unit_data.size > 0:
        return np.mean(unit_data)
    else:
        return 0


def fit_glm_single_unit(event_windows_data, stimuli_outcome_df, unit_idx, window=(-0.1, 0.5)):
    """
    Fit Generalized Linear Model (Poisson regression) to single neuron spike trains.
    """
    # Extract data from event_windows_data
    event_windows_matrix, time_axis, valid_event_indices, event_stimuli_outcome_df, metadata = event_windows_data
    
    # Find the time indices corresponding to the window
    start_time, end_time = window
    start_idx = np.argmin(np.abs(time_axis - start_time))
    end_idx = np.argmin(np.abs(time_axis - end_time))
    
    # Get the unit's data for the specified window
    unit_data = event_windows_matrix[unit_idx, start_idx:end_idx, :]  # [time × trials]
    
    # Average across time bins for each trial
    trial_rates = np.mean(unit_data, axis=0)  # Average across time for each trial
    
    # Create design matrix
    design_matrix = []
    spike_counts = []
    
    for idx, row in stimuli_outcome_df.iterrows():
        # Get firing rate for this trial
        trial_idx = idx  # Assuming the DataFrame index corresponds to trial order
        if trial_idx < len(trial_rates):
            rate = trial_rates[trial_idx]
        else:
            rate = 0
        spike_counts.append(rate)
        
        # Create feature vector
        features = []
        
        # Stimulus identity (one-hot encoding)
        if 'stimulus' in row:
            features.append(row['stimulus'])
        
        # Trial type (Go=1, NoGo=0)
        trial_type = 1 if row['outcome'] in ['Hit', 'Miss'] else 0
        features.append(trial_type)
        
        # Outcome (Hit=1, others=0)
        outcome = 1 if row['outcome'] == 'Hit' else 0
        features.append(outcome)
        
        design_matrix.append(features)
    
    if len(design_matrix) == 0:
        return None, None
    
    # Convert to numpy arrays
    X = np.array(design_matrix)
    y = np.array(spike_counts)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit Poisson regression
    try:
        model = PoissonRegressor(alpha=0.1, max_iter=1000)
        model.fit(X_scaled, y)
        
        # Get coefficients
        coefficients = model.coef_
        intercept = model.intercept_
        
        # Compute R-squared
        y_pred = model.predict(X_scaled)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return coefficients, (intercept, r_squared), y_pred, y
    except:
        return None, None


def calculate_psth_metrics(unit_data, time_axis, baseline_window=(-0.5, 0)):
    """
    Calculate comprehensive PSTH metrics for a single unit.
    
    Args:
        unit_data: 2D array [time × events] for the unit
        time_axis: 1D array of time points
        baseline_window: tuple of (start, end) time for baseline calculation
        
    Returns:
        dict: Dictionary containing all PSTH metrics
    """
    # Calculate PSTH
    psth_mean = np.mean(unit_data, axis=1)  # Average across events
    psth_sem = np.std(unit_data, axis=1) / np.sqrt(unit_data.shape[1])  # SEM across events
    
    # Find baseline period
    baseline_mask = (time_axis >= baseline_window[0]) & (time_axis < baseline_window[1])
    if np.sum(baseline_mask) == 0:
        baseline_mask = time_axis < 0  # Fallback to pre-stimulus period
    
    baseline_mean = np.mean(psth_mean[baseline_mask])
    baseline_std = np.std(psth_mean[baseline_mask])
    
    # Find response period (post-stimulus)
    response_mask = time_axis >= 0
    response_data = psth_mean[response_mask]
    response_times = time_axis[response_mask]
    
    if len(response_data) == 0:
        return {
            'onset_latency': np.nan,
            'peak_latency': np.nan,
            'response_magnitude': np.nan,
            'fwhm': np.nan,
            'rise_time': np.nan,
            'decay_time': np.nan,
            'suppression_metrics': np.nan,
            'trial_variability': np.nan,
            'signal_to_noise': np.nan,
            'baseline_rate': baseline_mean,
            'peak_rate': np.nan
        }
    
    # 1. Determine response type (excitation vs suppression)
    max_response = np.max(response_data)
    min_response = np.min(response_data)
    max_deviation_from_baseline = max_response - baseline_mean
    min_deviation_from_baseline = baseline_mean - min_response
    
    if max_deviation_from_baseline >= min_deviation_from_baseline:
        # Excitatory response (or no clear preference)
        peak_idx = np.argmax(response_data)
        peak_latency = response_times[peak_idx]
        peak_rate = response_data[peak_idx]
        response_magnitude = peak_rate - baseline_mean
        response_type = "excitation"
    else:
        # Suppressive response
        peak_idx = np.argmin(response_data)
        peak_latency = response_times[peak_idx]
        peak_rate = response_data[peak_idx]
        response_magnitude = baseline_mean - peak_rate  # Positive value for suppression
        response_type = "suppression"
    
    # 2. Onset latency (first time point where response deviates significantly from baseline)
    if response_type == "excitation":
        threshold = baseline_mean + 2 * baseline_std
        onset_indices = np.where(response_data > threshold)[0]
    else:  # suppression
        threshold = baseline_mean - 2 * baseline_std
        onset_indices = np.where(response_data < threshold)[0]
    
    onset_latency = response_times[onset_indices[0]] if len(onset_indices) > 0 else np.nan
    
    # 3. Full-width at half-maximum (FWHM)
    if response_type == "excitation":
        half_max = baseline_mean + (peak_rate - baseline_mean) / 2
        above_half_max = response_data >= half_max
    else:  # suppression
        half_max = baseline_mean - (baseline_mean - peak_rate) / 2
        above_half_max = response_data <= half_max
    
    if np.sum(above_half_max) > 0:
        first_half_max = np.where(above_half_max)[0][0]
        last_half_max = np.where(above_half_max)[0][-1]
        fwhm = response_times[last_half_max] - response_times[first_half_max]
    else:
        fwhm = np.nan
    
    # 4. Rise time (time from onset to peak)
    if not np.isnan(onset_latency):
        rise_time = peak_latency - onset_latency
    else:
        rise_time = np.nan
    
    # 5. Decay time (time from peak to return to baseline)
    if response_type == "excitation":
        decay_threshold = baseline_mean + baseline_std
        decay_indices = np.where((response_data <= decay_threshold) & (response_times > peak_latency))[0]
    else:  # suppression
        decay_threshold = baseline_mean - baseline_std
        decay_indices = np.where((response_data >= decay_threshold) & (response_times > peak_latency))[0]
    
    decay_time = response_times[decay_indices[0]] - peak_latency if len(decay_indices) > 0 else np.nan
    
    # 6. Suppression metrics (general response characteristics)
    if response_type == "suppression":
        # For suppressive responses, calculate suppression characteristics
        suppression_mask = response_data < baseline_mean
        suppression_magnitude = baseline_mean - np.min(response_data)
        suppression_duration = np.sum(suppression_mask) * (time_axis[1] - time_axis[0]) if len(time_axis) > 1 else 0
        suppression_metrics = {
            'magnitude': suppression_magnitude,
            'duration': suppression_duration,
            'fraction_suppressed': np.sum(suppression_mask) / len(response_data)
        }
    else:
        # For excitatory responses, calculate any suppression periods
        suppression_mask = response_data < baseline_mean
        if np.sum(suppression_mask) > 0:
            suppression_magnitude = baseline_mean - np.min(response_data[suppression_mask])
            suppression_duration = np.sum(suppression_mask) * (time_axis[1] - time_axis[0]) if len(time_axis) > 1 else 0
            suppression_metrics = {
                'magnitude': suppression_magnitude,
                'duration': suppression_duration,
                'fraction_suppressed': np.sum(suppression_mask) / len(response_data)
            }
        else:
            suppression_metrics = {
                'magnitude': 0,
                'duration': 0,
                'fraction_suppressed': 0
            }
    
    # 7. Trial-to-trial variability (coefficient of variation)
    trial_variability = np.std(unit_data, axis=1) / (np.mean(unit_data, axis=1) + 1e-10)  # Add small constant to avoid division by zero
    mean_variability = np.mean(trial_variability[response_mask])
    
    # 8. Signal-to-noise ratio
    signal = response_magnitude
    noise = baseline_std
    signal_to_noise = signal / (noise + 1e-10)  # Add small constant to avoid division by zero
    
    return {
        'onset_latency': onset_latency,
        'peak_latency': peak_latency,
        'response_magnitude': response_magnitude,
        'response_type': response_type,
        'fwhm': fwhm,
        'rise_time': rise_time,
        'decay_time': decay_time,
        'suppression_metrics': suppression_metrics,
        'trial_variability': mean_variability,
        'signal_to_noise': signal_to_noise,
        'baseline_rate': baseline_mean,
        'peak_rate': peak_rate
    }


def _create_matrix_hash(matrix, additional_params=None):
    """
    Create a hash for caching based on matrix content and parameters.
    """
    # Use a subset of the matrix for hashing to improve performance
    sample_data = matrix[::max(1, matrix.shape[0]//100), ::max(1, matrix.shape[1]//100), ::max(1, matrix.shape[2]//100)]
    hash_input = f"{sample_data.shape}_{np.mean(sample_data):.6f}_{np.std(sample_data):.6f}"
    
    if additional_params:
        hash_input += f"_{additional_params}"
    
    return hashlib.md5(hash_input.encode()).hexdigest()


@st.cache_data(show_spinner="Computing p-values...")
def compute_psth_pvalues_from_event_windows_cached(matrix_hash, event_windows_matrix, event_times, bin_size=0.005, window=(-0.1, 0.2)):
    """
    Cached version of p-values computation.
    
    Args:
        matrix_hash: Hash of the matrix for cache invalidation
        event_windows_matrix: 3D array [units × time × events]
        event_times: 1D array of event times
        bin_size: float, bin size in seconds
        window: tuple, time window around event (start, end) in seconds
        
    Returns:
        numpy array of p-values for each unit
    """
    
    n_units = event_windows_matrix.shape[0]
    n_time_bins = event_windows_matrix.shape[1]
    n_events = event_windows_matrix.shape[2]
    sec2bin = 1/bin_size
    # Create time axis for the window
    peri_event_window = np.linspace(window[0]*sec2bin, window[1]*sec2bin, n_time_bins)
    pvals = []
    
    for unit_idx in range(n_units):
        # Get the unit's data: [time × events]
        unit_data = event_windows_matrix[unit_idx, :, :]
        
        # Calculate PSTH by averaging across events
        psth_mean = np.mean(unit_data, axis=1)  # Shape: [time]
        
        # Find the index corresponding to time 0
        zero_idx = np.argmin(np.abs(peri_event_window))
        
        # Split into pre and post event periods
        pre = psth_mean[:zero_idx]
        post = psth_mean[zero_idx:]
        
        # Perform Mann-Whitney U test
        try:
            stat, p = stats.mannwhitneyu(pre, post, alternative='two-sided')
        except:
            p = 1.0  # Default p-value if test fails
        
        pvals.append(p)
    
    return np.array(pvals)


@st.cache_data(show_spinner="Computing unit metrics...")
def compute_all_unit_metrics_cached(event_windows_data, stimuli_outcome_df, available_units_tuple, window):
    """
    Compute all metrics for available units and cache the results.
    """
    available_units = np.array(available_units_tuple)
    
    metrics = {
        'cp_values': {},
        'outcome_p_values': {},
        'dprime_values': {}
    }
    
    for unit_idx in available_units:
        try:
            cp_val, _ = compute_choice_probability(event_windows_data, stimuli_outcome_df, unit_idx, window)
            metrics['cp_values'][unit_idx] = float(cp_val) if cp_val is not None else np.nan
        except Exception:
            metrics['cp_values'][unit_idx] = np.nan
        
        try:
            p_val, *_ = compute_outcome_modulation(event_windows_data, stimuli_outcome_df, unit_idx, window)
            metrics['outcome_p_values'][unit_idx] = float(p_val) if p_val is not None else np.nan
        except Exception:
            metrics['outcome_p_values'][unit_idx] = np.nan
        
        try:
            d_val, _, _ = compute_go_nogo_coding(event_windows_data, stimuli_outcome_df, unit_idx, window)
            metrics['dprime_values'][unit_idx] = float(d_val) if d_val is not None else np.nan
        except Exception:
            metrics['dprime_values'][unit_idx] = np.nan
    
    return metrics


def compute_psth_pvalues_from_event_windows(event_windows_matrix, event_times, bin_size=0.005, window=(-1, 2)):
    """
    Compute p-values using event windows data for more accurate statistical analysis.
    
    Args:
        event_windows_matrix: 3D array [units × time × events]
        event_times: 1D array of event times
        bin_size: float, bin size in seconds
        window: tuple, time window around event (start, end) in seconds
        
    Returns:
        numpy array of p-values for each unit
    """
    # Create hash for caching
    matrix_hash = _create_matrix_hash(event_windows_matrix, f"{bin_size}_{window[0]}_{window[1]}")
    
    return compute_psth_pvalues_from_event_windows_cached(
        matrix_hash, event_windows_matrix, event_times, bin_size, window
    )

