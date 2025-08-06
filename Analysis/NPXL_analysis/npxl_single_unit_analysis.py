import plotly.graph_objects as go
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
import streamlit as st

# Add color imports
from Analysis.GNG_bpod_analysis.colors import COLOR_GO, COLOR_GRAY, COLOR_NOGO, COLOR_HIT, COLOR_FA, COLOR_CR, COLOR_MISS, COLOR_BLUE, COLOR_BLUE_TRANSPARENT, COLOR_ACCENT, COLOR_ACCENT_TRANSPARENT

def save_pvalues_to_folder(pvals, selected_folder, window=(-1, 2), bin_size=0.01):
    """
    Save p-values to the analysis output folder.
    
    Args:
        pvals: numpy array of p-values for each unit
        selected_folder: path to the analysis output folder
        window: time window used for p-value calculation
        bin_size: bin size used for p-value calculation
    """
    try:
        # Create a DataFrame with unit indices and p-values
        pvals_df = pd.DataFrame({
            'unit_index': range(len(pvals)),
            'p_value': pvals,
            'significant': pvals < 0.05
        })
        
        # Add metadata about the analysis
        metadata = {
            'window_start': window[0],
            'window_end': window[1],
            'bin_size': bin_size,
            'total_units': len(pvals),
            'significant_units': np.sum(pvals < 0.05),
            'significance_threshold': 0.05
        }
        
        # Save p-values DataFrame
        pvals_file = os.path.join(selected_folder, "pvalues_analysis.csv")
        pvals_df.to_csv(pvals_file, index=False)
        
        # Save metadata
        metadata_file = os.path.join(selected_folder, "pvalues_metadata.txt")
        with open(metadata_file, 'w') as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
        
        return True
    except Exception as e:
        print(f"Error saving p-values: {e}")
        return False

def save_all_psth_metrics(event_windows_data, selected_folder, display_window, pvals=None, baseline_window=(-0.5, 0)):
    """
    Save PSTH metrics for all units to the analysis output folder.
    
    Args:
        event_windows_data: tuple containing (event_windows_matrix, time_axis, valid_event_indices, event_stimuli_outcome_df, metadata)
        selected_folder: path to the analysis output folder
        display_window: tuple of (start, end) time for the display window
        pvals: numpy array of p-values for each unit (optional)
        baseline_window: tuple of (start, end) time for baseline calculation
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Extract data from event_windows_data
        event_windows_matrix, time_axis, valid_event_indices, event_stimuli_outcome_df, metadata = event_windows_data
        
        # Create single_unit subfolder
        single_unit_folder = os.path.join(selected_folder, "single_unit")
        os.makedirs(single_unit_folder, exist_ok=True)
        
        # Initialize list to store all metrics
        all_metrics = []
        
        # Calculate metrics for each unit
        num_units = event_windows_matrix.shape[0]
        for unit_idx in range(num_units):
            # Get unit data
            unit_data = event_windows_matrix[unit_idx, :, :]  # [time × events]
            
            # Calculate PSTH metrics
            metrics = calculate_psth_metrics(unit_data, time_axis, baseline_window)
            
            # Add unit index to metrics
            metrics['unit_index'] = unit_idx
            
            # Add p-value if available
            if pvals is not None and unit_idx < len(pvals):
                metrics['p_value'] = pvals[unit_idx]
                metrics['significant'] = pvals[unit_idx] < 0.05
            else:
                metrics['p_value'] = np.nan
                metrics['significant'] = False
            
            # Flatten suppression metrics for CSV storage
            if isinstance(metrics['suppression_metrics'], dict):
                metrics['suppression_magnitude'] = metrics['suppression_metrics']['magnitude']
                metrics['suppression_duration'] = metrics['suppression_metrics']['duration']
                metrics['fraction_suppressed'] = metrics['suppression_metrics']['fraction_suppressed']
                del metrics['suppression_metrics']  # Remove the nested dict
            
            all_metrics.append(metrics)
        
        # Convert to DataFrame
        metrics_df = pd.DataFrame(all_metrics)
        
        # Reorder columns to put unit_index first
        cols = ['unit_index'] + [col for col in metrics_df.columns if col != 'unit_index']
        metrics_df = metrics_df[cols]
        
        # Save to CSV
        metrics_file = os.path.join(single_unit_folder, "psth_metrics.csv")
        metrics_df.to_csv(metrics_file, index=False)
        
        # Save metadata
        metadata_file = os.path.join(single_unit_folder, "psth_metrics_metadata.txt")
        with open(metadata_file, 'w') as f:
            f.write(f"Analysis Date: {pd.Timestamp.now()}\n")
            f.write(f"Total Units: {num_units}\n")
            f.write(f"Display Window: {display_window[0]} to {display_window[1]} seconds\n")
            f.write(f"Baseline Window: {baseline_window[0]} to {baseline_window[1]} seconds\n")
            f.write(f"Time Axis Range: {time_axis[0]:.3f} to {time_axis[-1]:.3f} seconds\n")
            f.write(f"Number of Events: {event_windows_matrix.shape[2]}\n")
            f.write(f"Time Bins: {len(time_axis)}\n")
            if pvals is not None:
                f.write(f"P-values Available: Yes\n")
                f.write(f"Significant Units (p < 0.05): {np.sum(pvals < 0.05)}\n")
            else:
                f.write(f"P-values Available: No\n")
            f.write("\nMetrics Description and Mathematical Formulas:\n")
            f.write("- response_type: Type of response ('excitation' or 'suppression')\n")
            f.write("  Formula: excitation if max_deviation >= min_deviation, else suppression\n")
            f.write("- onset_latency: Time from event onset to first significant response (seconds)\n")
            f.write("  Formula: min(t | rate(t) > baseline_rate + 2*std) for excitation\n")
            f.write("  Formula: min(t | rate(t) < baseline_rate - 2*std) for suppression\n")
            f.write("- peak_latency: Time from event onset to peak response (seconds)\n")
            f.write("  Formula: argmax(rate(t)) for excitation, argmin(rate(t)) for suppression\n")
            f.write("- response_magnitude: Peak response magnitude relative to baseline (spikes/s)\n")
            f.write("  Formula: max(rate(t)) - baseline_rate for excitation\n")
            f.write("  Formula: baseline_rate - min(rate(t)) for suppression\n")
            f.write("- fwhm: Full-width at half-maximum of the response (seconds)\n")
            f.write("  Formula: t2 - t1 where rate(t1) = rate(t2) = baseline + (peak - baseline)/2 for excitation\n")
            f.write("  Formula: t2 - t1 where rate(t1) = rate(t2) = baseline - (baseline - peak)/2 for suppression\n")
            f.write("- rise_time: Time from onset to peak (seconds)\n")
            f.write("  Formula: peak_latency - onset_latency\n")
            f.write("- decay_time: Time from peak to return to baseline (seconds)\n")
            f.write("  Formula: t_return - peak_latency where rate(t_return) ≈ baseline_rate + std for excitation\n")
            f.write("  Formula: t_return - peak_latency where rate(t_return) ≈ baseline_rate - std for suppression\n")
            f.write("- trial_variability: Coefficient of variation across trials\n")
            f.write("  Formula: std(trial_rates) / mean(trial_rates)\n")
            f.write("- signal_to_noise: Response magnitude divided by baseline standard deviation\n")
            f.write("  Formula: response_magnitude / std(baseline_rate)\n")
            f.write("- baseline_rate: Average firing rate during baseline period (spikes/s)\n")
            f.write("  Formula: mean(rate(t)) where t is in baseline window\n")
            f.write("- peak_rate: Peak firing rate during response period (spikes/s)\n")
            f.write("  Formula: max(rate(t)) for excitation, min(rate(t)) for suppression\n")
            f.write("- suppression_magnitude: Magnitude of suppression below baseline (spikes/s)\n")
            f.write("  Formula: baseline_rate - min(rate(t)) where rate(t) < baseline_rate\n")
            f.write("- suppression_duration: Duration of suppression period (seconds)\n")
            f.write("  Formula: sum(dt) where rate(t) < baseline_rate\n")
            f.write("- fraction_suppressed: Fraction of response period that was suppressed\n")
            f.write("  Formula: suppression_duration / total_response_duration\n")
            f.write("- p_value: Statistical significance of response modulation\n")
            f.write("  Formula: Mann-Whitney U test between baseline and response periods\n")
            f.write("- significant: Boolean indicating p < 0.05\n")
            f.write("  Formula: p_value < 0.05\n")
        
        return True
        
    except Exception as e:
        print(f"Error saving PSTH metrics: {e}")
        return False

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

# def compute_peri_event_rate(spike_matrix, event_times, unit_idx, window=(-0.1, 0.5), bin_size=0.01):
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

def plot_advanced_unit_analysis(event_windows_data, stimuli_outcome_df, unit_idx, bin_size=0.01):
    """
    Create comprehensive plots for advanced single unit analysis.
    """
    # Create subplots
    fig = go.Figure()
    
    # 1. Stimulus selectivity
    stimuli, tuning_curve, tuning_sem, best_stim = compute_stimulus_selectivity(event_windows_data, stimuli_outcome_df, unit_idx)
    if stimuli is not None:
        fig.add_trace(go.Scatter(
            x=stimuli, y=tuning_curve, mode='lines+markers',
            name=f'Stimulus Tuning (Best: {best_stim:.2f})',
            line=dict(color='blue')
        ))
    
    # 2. Go/NoGo coding
    d_prime, roc_auc, (go_rates, nogo_rates) = compute_go_nogo_coding(event_windows_data, stimuli_outcome_df, unit_idx)
    
    # 3. Outcome modulation
    outcome_p, (rewarded_rates, non_rewarded_rates), (rewarded_mean, non_rewarded_mean) = compute_outcome_modulation(
        event_windows_data, stimuli_outcome_df, unit_idx
    )
    
    # 4. Choice probability
    cp, cp_corr = compute_choice_probability(event_windows_data, stimuli_outcome_df, unit_idx)
    
    # Update layout
    fig.update_layout(
        title=f'Advanced Analysis - Unit {unit_idx}<br>' +
              f'd\' = {d_prime:.3f}, ROC AUC = {roc_auc:.3f}, CP = {cp:.3f}<br>' +
              f'Outcome p = {outcome_p:.3g}, Reward modulation = {rewarded_mean:.2f} vs {non_rewarded_mean:.2f}',
        xaxis_title='Stimulus',
        yaxis_title='Firing Rate (spikes/s)',
        height=600
    )
    
    return fig

def plot_unit_psth(event_windows_data, display_window, unit_idx, sorted_pvals, unit_rank, outcome_filter="All", bin_size=0.01):
    """
    Plots PSTH for a single unit aligned to event_times, adds a vertical line at x=0, and calculates significance (Mann-Whitney U test) between pre- and post-event activity.
    
    Args:
        event_windows_data: Tuple containing (event_windows_matrix, time_axis, valid_event_indices, event_stimuli_outcome_df, metadata)
        display_window: Tuple of (start_time, end_time) for the display window
        unit_idx: int, index of the unit to plot
        sorted_pvals: array of sorted p-values
        unit_rank: int, rank of the unit in the sorted p-values
        outcome_filter: str, filter for specific outcome types ("All", "Go", "NoGo", "Hit", "Miss", "False Alarm", "CR")
    """
    event_windows_matrix, time_axis, valid_event_indices, event_stimuli_outcome_df, metadata = event_windows_data
    import streamlit as st
    # Get the unit's data
    unit_data = event_windows_matrix[unit_idx, :, :]  # Shape: [time × events]
    n_time_bins = event_windows_matrix.shape[1]
    
    # Create time axis for the window
    peri_event_window = np.linspace(display_window[0], display_window[1], n_time_bins)
    # Find the index corresponding to time 0
    zero_idx = np.argmin(np.abs(peri_event_window))
    unit_data = unit_data[zero_idx+int(display_window[0]*10):zero_idx+int(display_window[1]*10),:]

    event_outcomes = None
    if 'outcome' in event_stimuli_outcome_df.columns:
        event_outcomes = event_stimuli_outcome_df['outcome'].values
    
    # Filter data by outcome if specified
    if outcome_filter != "All" and event_outcomes is not None:
        if outcome_filter == "Go":
            hit_mask = event_outcomes == "Hit"
            miss_mask = event_outcomes == "Miss"
            outcome_mask = hit_mask | miss_mask
        elif outcome_filter == "NoGo":
            false_alarm_mask = event_outcomes == "False Alarm"
            correct_rejection_mask = event_outcomes == "CR"
            outcome_mask = false_alarm_mask | correct_rejection_mask
        else:
            outcome_mask = event_outcomes == outcome_filter
        
        if np.sum(outcome_mask) > 0:
            unit_data = unit_data[:, outcome_mask]
            filtered_outcomes = event_outcomes[outcome_mask]
        else:
            # If no events match the filter, return empty plot
            psth_fig = go.Figure()
            psth_fig.update_layout(
                title=f"PSTH - Unit {unit_idx} ({outcome_filter}) - No data available",
                xaxis_title="Time (s)",
                yaxis_title="Firing Rate (spikes/s)",
                xaxis=dict(constrain='domain'),
                margin=dict(r=80),
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            return psth_fig
    else:
        filtered_outcomes = event_outcomes
    
    # Calculate PSTH statistics
    if unit_data.shape[1] > 0:
        psth_mean = np.mean(unit_data, axis=1)  # Average across events
        psth_sem = np.std(unit_data, axis=1) / np.sqrt(unit_data.shape[1])  # SEM across events
    else:
        # If no data, create empty arrays
        psth_mean = np.zeros(unit_data.shape[0])
        psth_sem = np.zeros(unit_data.shape[0])
    
    # Create time axis for PSTH that matches the display_window
    num_time_bins = unit_data.shape[0]
    psth_time_axis = np.linspace(display_window[0], display_window[1], num_time_bins)
    
    # Calculate comprehensive PSTH metrics
    psth_metrics = calculate_psth_metrics(unit_data, psth_time_axis)
    
    # Create PSTH plot with proper time axis
    psth_fig = go.Figure()
    
    # Add main PSTH line
    psth_fig.add_trace(go.Scatter(
        x=psth_time_axis,
        y=psth_mean,
        mode='lines',
        name='Mean Firing Rate',
        line=dict(color=COLOR_ACCENT, width=3)
    ))
    
    # Add shaded area for SEM
    psth_fig.add_trace(go.Scatter(
        x=np.concatenate([psth_time_axis, psth_time_axis[::-1]]),
        y=np.concatenate([psth_mean + psth_sem, (psth_mean - psth_sem)[::-1]]),
        fill='toself',
        fillcolor=f'rgba(0,0,255,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        name='SEM'
    ))
    
    # Add vertical line at x = 0
    psth_fig.add_vline(
        x=0, 
        line_dash="dash", 
        line_color=COLOR_GRAY, 
        line_width=2
    )
    
    # Add markers for key metrics if they exist
    if not np.isnan(psth_metrics['onset_latency']):
        psth_fig.add_vline(
            x=psth_metrics['onset_latency'],
            line_dash="dot",
            line_color="orange",
            line_width=2,
            annotation_text=f"Onset",
            annotation_position="top left",
            annotation=dict(textangle=90)  # Rotate annotation 90 degrees
        )
    
    if not np.isnan(psth_metrics['peak_latency']):
        psth_fig.add_vline(
            x=psth_metrics['peak_latency'],
            line_dash="dot",
            line_color="red",
            line_width=2,
            annotation_text=f"Peak",
            annotation_position="top left",
            annotation=dict(textangle=90)  # Rotate annotation 90 degrees
        )
        
        # Add horizontal line at half-maximum for FWHM visualization
        if not np.isnan(psth_metrics['fwhm']):
            half_max = psth_metrics['baseline_rate'] + (psth_metrics['peak_rate'] - psth_metrics['baseline_rate']) / 2
            psth_fig.add_hline(
                y=half_max,
                line_dash="dot",
                line_color="purple",
                line_width=1,
                annotation_text=f"Half-Max: {half_max:.2f}",
                annotation_position="right",
                annotation=dict(textangle=90)  # Rotate annotation 90 degrees
            )
    
    # Create title with outcome filter information
    title_text = f"PSTH - Unit {unit_idx} (p={sorted_pvals[unit_rank]:.3g})"
    if outcome_filter != "All":
        title_text += f" - {outcome_filter} (n={unit_data.shape[1]})"
    else:
        title_text += f" - All trials (n={unit_data.shape[1]})"
    
    psth_fig.update_layout(
        title=title_text,
        xaxis_title="Time (s)",
        yaxis_title="Firing Rate (spikes/s)",
        xaxis=dict(constrain='domain'),
        margin=dict(r=80),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    return psth_fig, psth_metrics

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

def compute_psth_pvalues_from_event_windows(event_windows_matrix, event_times, bin_size=0.01, window=(-1, 2)):
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
    
    n_units = event_windows_matrix.shape[0]
    n_time_bins = event_windows_matrix.shape[1]
    n_events = event_windows_matrix.shape[2]
    
    # Create time axis for the window
    peri_event_window = np.linspace(window[0], window[1], n_time_bins)
    
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

def load_event_windows_data(folder):
    """
    Loads the event windows data and associated metadata.
    
    Args:
        folder (str): Directory containing the saved data
        
    Returns:
        tuple: (event_windows_matrix, time_axis, valid_event_indices, stimuli_outcome_df, metadata) or None if not found
    """
    try:
        # Load the 3D event windows matrix
        event_windows_matrix = np.load(os.path.join(folder, "event_windows_matrix.npy"))
        
        # Load the time axis
        time_axis = np.load(os.path.join(folder, "event_window_time_axis.npy"))
        
        # Load the valid event indices
        valid_event_indices = np.load(os.path.join(folder, "valid_event_indices.npy"))
        
        # Load the filtered stimuli_outcome DataFrame
        stimuli_outcome_df = pd.read_csv(os.path.join(folder, "event_windows_stimuli_outcome.csv"))
        
        # Load metadata
        metadata = {}
        metadata_file = os.path.join(folder, "event_windows_metadata.txt")
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                for line in f:
                    key, value = line.strip().split(': ')
                    metadata[key] = value
        
        return event_windows_matrix, time_axis, valid_event_indices, stimuli_outcome_df, metadata
    except Exception as e:
        print(f"Could not load event windows data: {e}")
        return None

def plot_unit_heatmap(event_windows_data, display_window, unit_idx):
    """
    Create a heatmap visualization for a single unit from event windows data.
    
    Args:
        event_windows_data: Tuple containing (event_windows_matrix, time_axis, valid_event_indices, event_stimuli_outcome_df, metadata)
        display_window: Tuple of (start_time, end_time) for the display window
        unit_idx: Index of the unit to plot
        
    Returns:
        plotly.graph_objects.Figure: The heatmap figure
    """
    event_windows_matrix, time_axis, valid_event_indices, event_stimuli_outcome_df, metadata = event_windows_data
    
    # Create a new time axis that matches the current display_window
    num_time_bins = event_windows_matrix.shape[1]
    new_time_axis = np.linspace(display_window[0], display_window[1], num_time_bins)
    
    # Get the unit's data
    unit_data = event_windows_matrix[unit_idx, :, :]  # Shape: [time × events]

    # Get outcomes for the valid events
    event_outcomes_for_raster = None
    if 'outcome' in event_stimuli_outcome_df.columns:
        event_outcomes_for_raster = event_stimuli_outcome_df['outcome'].values
    
    # Filter data by trial type
    hit_mask = event_outcomes_for_raster == 'Hit'
    miss_mask = event_outcomes_for_raster == "Miss"
    false_alarm_mask = event_outcomes_for_raster == "False Alarm"
    correct_rejection_mask = event_outcomes_for_raster == "CR"
    
    hit_data = unit_data[:, hit_mask]
    miss_data = unit_data[:, miss_mask]
    false_alarm_data = unit_data[:, false_alarm_mask]
    correct_rejection_data = unit_data[:, correct_rejection_mask]
    
    # Order data: CR, FA, Miss, Hit
    ordered_data = np.concatenate([correct_rejection_data, false_alarm_data, miss_data, hit_data], axis=1).T

    def rgba_from_hex(hex_color, alpha=0.3):
        """Convert hex color to rgba string with given alpha."""
        hex_color = hex_color.lstrip('#')
        lv = len(hex_color)
        rgb = tuple(int(hex_color[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
        return f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{alpha})"

    trial_type_colors = {
        "CR": rgba_from_hex(COLOR_CR, 0.3),
        "False Alarm": rgba_from_hex(COLOR_FA, 0.3),
        "Miss": rgba_from_hex(COLOR_MISS, 0.3),
        "Hit": rgba_from_hex(COLOR_HIT, 0.3)
    }

    # Build a list of trial types in the order of the rows in ordered_data
    trial_types_ordered = (
        ["CR"] * correct_rejection_data.shape[1] +
        ["False Alarm"] * false_alarm_data.shape[1] +
        ["Miss"] * miss_data.shape[1] +
        ["Hit"] * hit_data.shape[1]
    )

    # Create the heatmap
    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        z=ordered_data,
        x=new_time_axis,
        y=np.arange(ordered_data.shape[1]),
        colorbar=dict(title="Firing Rate", len=0.8)
    ))

    # Add colored rectangles for trial type indicators
    for i, trial_type in enumerate(trial_types_ordered):
        color = trial_type_colors.get(trial_type, "rgba(200,200,200,0.2)")
        fig.add_shape(
            type="rect",
            xref="paper", yref="y",
            x0=-0.02, x1=0,  # Just outside the heatmap
            y0=i-0.5, y1=i+0.5,
            fillcolor=color,
            line=dict(width=0),
            layer="above"
        )

    # Add legend for trial types
    legend_items = []
    for trial_type, color in trial_type_colors.items():
        legend_items.append(
            go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color=color),
                legendgroup=trial_type,
                showlegend=True,
                name=trial_type
            )
        )
    
    # Add vertical line at x = 0
    fig.add_vline(
        x=0, 
        line_dash="dash", 
        line_color=COLOR_GRAY, 
        line_width=2,
        annotation_text="Event Onset",
        annotation_position="top right"
    )
    
    fig.update_layout(
        title=f"Heatmap - Unit {unit_idx}",
        xaxis_title="Time (s)",
        yaxis_title="Trial",
        xaxis=dict(constrain='domain'),
        legend=dict(
            orientation='h',
            x=0.5,
            y=1.1,
            xanchor='center',
            yanchor='top'
        ),
        margin=dict(r=80, t=100)
    )
    
    for item in legend_items:
        fig.add_trace(item)
    
    return fig

def get_trial_statistics(event_windows_data, unit_idx):
    """
    Get trial statistics for a specific unit from event windows data.
    
    Args:
        event_windows_data: Tuple containing (event_windows_matrix, time_axis, valid_event_indices, event_stimuli_outcome_df, metadata)
        unit_idx: Index of the unit to analyze
        
    Returns:
        dict: Dictionary containing trial counts for each outcome type
    """
    event_windows_matrix, time_axis, valid_event_indices, event_stimuli_outcome_df, metadata = event_windows_data
    
    # Get the unit's data
    unit_data = event_windows_matrix[unit_idx, :, :]  # Shape: [time × events]

    # Get outcomes for the valid events
    event_outcomes_for_raster = None
    if 'outcome' in event_stimuli_outcome_df.columns:
        event_outcomes_for_raster = event_stimuli_outcome_df['outcome'].values
    
    # Filter data by trial type
    hit_mask = event_outcomes_for_raster == 'Hit'
    miss_mask = event_outcomes_for_raster == "Miss"
    false_alarm_mask = event_outcomes_for_raster == "False Alarm"
    correct_rejection_mask = event_outcomes_for_raster == "CR"
    
    hit_data = unit_data[:, hit_mask]
    miss_data = unit_data[:, miss_mask]
    false_alarm_data = unit_data[:, false_alarm_mask]
    correct_rejection_data = unit_data[:, correct_rejection_mask]
    
    return {
        'CR': correct_rejection_data.shape[1],
        'FA': false_alarm_data.shape[1],
        'Miss': miss_data.shape[1],
        'Hit': hit_data.shape[1]
    }

def single_unit_analysis_panel(
    event_windows_data, stimuli_outcome_df, selected_folder=None, bin_size=0.01, default_window=1.0):
    import streamlit as st
    import numpy as np
    from .npxl_single_unit_analysis import plot_unit_psth
    from Analysis.NPXL_analysis.population_analysis import plot_population_heatmap

    event_times = stimuli_outcome_df['time'].values 
    event_outcomes = stimuli_outcome_df['outcome'].values if 'outcome' in stimuli_outcome_df.columns else None
    
    
    # Create tabs for different analysis types
    tab1, tab2, tab3 = st.tabs(["Basic PSTH", "Advanced Analysis", "GLM Analysis"])
    
    with tab1:
        st.header("Basic Analysis")
        
        # Control Panel Section
        st.subheader("Analysis Controls")
        control_col1, control_col2 = st.columns(2)
        
        
        with control_col1:
            peri_event = st.slider(
                "PSTH Window Size", 
                0.1, 3.0, default_window, step=0.10,
                help="Time window around event onset (seconds)"
            )
            display_window = (-peri_event/2, peri_event/2)
        
        with control_col2:
            bin_size_display = st.number_input(
                "Bin Size (s)", 
                min_value=0.001, max_value=0.1, value=bin_size, step=0.001,
                help="Time bin size for PSTH calculation"
            )

        # P-value Analysis Section
        st.subheader("Statistical Analysis")
        
        # Load event windows data for accurate p-value calculation
        event_windows_data = None
        if selected_folder is not None:
            event_windows_data = load_event_windows_data(selected_folder)
        
        # Recompute p-values with current window
        if event_windows_data is not None:
            # Use event windows data for more accurate p-value calculation
            event_windows_matrix, time_axis, valid_event_indices, event_stimuli_outcome_df, metadata = event_windows_data
            # Calculate p-values using the full event windows data
            pvals = compute_psth_pvalues_from_event_windows(event_windows_matrix, event_times, bin_size=bin_size_display, window=display_window)

        
        sorted_indices = np.argsort(pvals)
        sorted_pvals = pvals[sorted_indices]
        
        # Display statistics
        stats_col1, stats_col2, stats_col3 = st.columns(3)
        with stats_col1:
            st.metric("Total Units", len(pvals))
        with stats_col2:
            significant_units = np.sum(pvals < 0.05)
            st.metric("Significant Units", f"{significant_units}/{len(pvals)}")
        with stats_col3:
            significance_rate = (significant_units / len(pvals)) * 100 if len(pvals) > 0 else 0
            st.metric("Significance Rate", f"{significance_rate:.1f}%")


        # Save p-values and metrics automatically once
        if selected_folder is not None:
            # try:
            #     save_pvalues_to_folder(pvals, selected_folder, window=display_window, bin_size=bin_size_display)
            # except Exception as e:
            #     st.error(f"Error saving p-values: {e}")

            # try:
            #     save_all_psth_metrics(event_windows_data, selected_folder, display_window, pvals)
            #     st.toast("PSTH metrics saved (auto)")
            # except Exception as e:
            #     st.error(f"Error saving PSTH metrics: {e}")

            # Add button for manual save
            if st.button("Save PSTH Metrics Again"):
                try:
                    save_all_psth_metrics(event_windows_data, selected_folder, display_window, pvals)
                    st.toast("PSTH metrics saved (manual)")
                except Exception as e:
                    st.error(f"Error saving PSTH metrics: {e}")


        # Unit Selection Section
        st.subheader("Unit Selection")
                # Unit selection controls
        unit_col1, unit_col2 = st.columns(2)
        
        with unit_col1:
            unit_rank = st.slider(
                "Unit Rank (by p-value)", 
                0, len(sorted_pvals) - 1, 0,
                help="Select unit by statistical significance rank"
            )
            unit_idx = sorted_indices[unit_rank]
        
        with unit_col2:
            st.metric("Selected Unit", unit_idx)
            st.metric("P-value", f"{sorted_pvals[unit_rank]:.3g}")
        
        # Visualization Section
        st.subheader("Visualizations")
        
        if event_windows_data is not None:
            # Create two columns for PSTH and raster plot
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:

                outcome_filter = st.selectbox(
                "Outcome Filter",
                options=["All", "Go", "NoGo", "Hit", "Miss", "False Alarm", "CR"],
                index=0,
                help="Filter PSTH by specific outcome type")
                psth_fig, psth_metrics = plot_unit_psth(event_windows_data, display_window, unit_idx, sorted_pvals, unit_rank, outcome_filter)
                st.plotly_chart(psth_fig, use_container_width=True)
                
                # Display PSTH Metrics
                st.subheader("PSTH Metrics")
                metrics_col1, metrics_col2 = st.columns(2)
                
                with metrics_col1:
                    st.metric("Onset Latency", f"{psth_metrics['onset_latency']:.3f}s" if not np.isnan(psth_metrics['onset_latency']) else "N/A")
                    st.metric("Peak Latency", f"{psth_metrics['peak_latency']:.3f}s" if not np.isnan(psth_metrics['peak_latency']) else "N/A")
                    st.metric("Response Magnitude", f"{psth_metrics['response_magnitude']:.2f} spikes/s" if not np.isnan(psth_metrics['response_magnitude']) else "N/A")
                    st.metric("FWHM", f"{psth_metrics['fwhm']:.3f}s" if not np.isnan(psth_metrics['fwhm']) else "N/A")
                    st.metric("Rise Time", f"{psth_metrics['rise_time']:.3f}s" if not np.isnan(psth_metrics['rise_time']) else "N/A")
                
                with metrics_col2:
                    st.metric("Decay Time", f"{psth_metrics['decay_time']:.3f}s" if not np.isnan(psth_metrics['decay_time']) else "N/A")
                    st.metric("Trial Variability", f"{psth_metrics['trial_variability']:.3f}" if not np.isnan(psth_metrics['trial_variability']) else "N/A")
                    st.metric("Signal-to-Noise", f"{psth_metrics['signal_to_noise']:.2f}" if not np.isnan(psth_metrics['signal_to_noise']) else "N/A")
                    st.metric("Baseline Rate", f"{psth_metrics['baseline_rate']:.2f} spikes/s")
                    st.metric("Peak Rate", f"{psth_metrics['peak_rate']:.2f} spikes/s" if not np.isnan(psth_metrics['peak_rate']) else "N/A")
                
                # Suppression metrics (if applicable)
                if psth_metrics['suppression_metrics']['magnitude'] > 0:
                    st.subheader("Suppression Metrics")
                    supp_col1, supp_col2, supp_col3 = st.columns(3)
                    with supp_col1:
                        st.metric("Suppression Magnitude", f"{psth_metrics['suppression_metrics']['magnitude']:.2f} spikes/s")
                    with supp_col2:
                        st.metric("Suppression Duration", f"{psth_metrics['suppression_metrics']['duration']:.3f}s")
                    with supp_col3:
                        st.metric("Fraction Suppressed", f"{psth_metrics['suppression_metrics']['fraction_suppressed']:.2f}")
            
            with viz_col2:
                # Create heatmap using the new function
                heatmap_fig = plot_unit_heatmap(event_windows_data, display_window, unit_idx)
                st.plotly_chart(heatmap_fig, use_container_width=True)
                
                # Heatmap statistics
                trial_stats = get_trial_statistics(event_windows_data, unit_idx)
                heatmap_stats_col1, heatmap_stats_col2, heatmap_stats_col3, heatmap_stats_col4 = st.columns(4)
                with heatmap_stats_col1:
                    st.metric("CR", trial_stats['CR'])
                with heatmap_stats_col2:
                    st.metric("FA", trial_stats['FA'])
                with heatmap_stats_col3:
                    st.metric("Miss", trial_stats['Miss'])
                with heatmap_stats_col4:
                    st.metric("Hit", trial_stats['Hit'])
        else:
            st.warning("No event windows data available. Please ensure data is loaded.")
            
            # Fallback to basic PSTH if no event windows data
            st.subheader("Basic PSTH (Fallback)")
            # Note: Fallback PSTH doesn't support metrics yet
            st.warning("PSTH metrics not available in fallback mode")
            psth_fig = plot_unit_psth(
                event_windows_data, event_times, unit_idx,
                bin_size=bin_size_display, window=display_window,
                event_outcomes=event_outcomes
            )
            if isinstance(psth_fig, tuple):
                psth_fig = psth_fig[0]  # Extract just the figure
            st.plotly_chart(psth_fig, use_container_width=True)
    
    with tab2:
        st.header("Advanced Single Unit Analysis")
        
        # Filter for significant units
        show_only_significant = st.checkbox("Show only significant units (p < 0.05)", value=False)
        
        # Get available units based on significance filter
        if show_only_significant:
            significant_mask = pvals < 0.05
            available_units = np.where(significant_mask)[0]
            if len(available_units) == 0:
                st.warning("No significant units found. Showing all units.")
                available_units = range(event_windows_matrix.shape[0])
            else:
                st.info(f"Showing {len(available_units)} significant units out of {len(pvals)} total units")
        else:
            available_units = range(event_windows_matrix.shape[0])
        
        # Unit selection
        if len(available_units) > 0:
            unit_idx_adv = st.slider(
                "Select Unit", 
                int(np.min(available_units)), 
                int(np.max(available_units)), 
                int(available_units[0])
            )
        else:
            unit_idx_adv = 0
        
        # Analysis window
        analysis_window = st.slider("Analysis window", 0.1, 2.0, 1.0, step=0.1)
        window = (-0.1, analysis_window)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Stimulus Selectivity")
            stimuli, tuning_curve, tuning_sem, best_stim = compute_stimulus_selectivity(
                event_windows_data, stimuli_outcome_df, unit_idx_adv, window
            )
            if stimuli is not None:
                st.metric("Best stimulus", f"{best_stim:.2f}")
                fig_tuning = go.Figure()
                
                # Convert to numpy arrays for arithmetic operations
                tuning_curve_array = np.array(tuning_curve)
                tuning_sem_array = np.array(tuning_sem)
                
                # Add shaded area for error bars
                fig_tuning.add_trace(go.Scatter(
                    x=np.concatenate([stimuli, stimuli[::-1]]),
                    y=np.concatenate([tuning_curve_array + tuning_sem_array, (tuning_curve_array - tuning_sem_array)[::-1]]),
                    fill='toself',
                    fillcolor=COLOR_BLUE_TRANSPARENT,
                    line=dict(color='rgba(255,255,255,0)'),
                    showlegend=False,
                    name='Error'
                ))
                
                # Add main line
                fig_tuning.add_trace(go.Scatter(
                    x=stimuli, y=tuning_curve, mode='lines+markers',
                    name=f'Best stimulus: {best_stim:.2f}',
                    line=dict(color=COLOR_BLUE, width=2),
                    marker=dict(color=COLOR_BLUE, size=6)
                ))
                
                fig_tuning.update_layout(
                    title="Stimulus Tuning Curve",
                    xaxis_title="Stimulus",
                    yaxis_title="Firing Rate (spikes/s)",
                    plot_bgcolor='white',
                    paper_bgcolor='white'
                )
                st.plotly_chart(fig_tuning, use_container_width=True)
            else:
                st.write("No stimulus data available")
        
        with col2:
            st.subheader("Go/NoGo Coding")
            d_prime, roc_auc, (go_rates, nogo_rates) = compute_go_nogo_coding(
                event_windows_data, stimuli_outcome_df, unit_idx_adv, window
            )
            if d_prime is not None:
                st.metric("d'", f"{d_prime:.3f}")
                st.metric("ROC AUC", f"{roc_auc:.3f}")
                
                # Plot Go vs NoGo rates with colors
                fig_gng = go.Figure()
                fig_gng.add_trace(go.Box(y=go_rates, name="Go", boxpoints='all', marker_color=COLOR_GO))
                fig_gng.add_trace(go.Box(y=nogo_rates, name="NoGo", boxpoints='all', marker_color=COLOR_NOGO))
                fig_gng.update_layout(
                    title="Go vs NoGo Firing Rates",
                    yaxis_title="Firing Rate (spikes/s)"
                )
                st.plotly_chart(fig_gng, use_container_width=True)
            else:
                st.write("Insufficient data for Go/NoGo analysis")
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("Outcome Modulation")
            outcome_p, (rewarded_rates, non_rewarded_rates), (rewarded_mean, non_rewarded_mean) = compute_outcome_modulation(
                event_windows_data, stimuli_outcome_df, unit_idx_adv, window
            )
            if outcome_p is not None:
                st.metric("p-value", f"{outcome_p:.3g}")
                st.metric("Rewarded mean", f"{rewarded_mean:.2f}")
                st.metric("Non-rewarded mean", f"{non_rewarded_mean:.2f}")
                
                fig_outcome = go.Figure()
                fig_outcome.add_trace(go.Box(y=rewarded_rates, name="Rewarded", boxpoints='all', marker_color=COLOR_HIT))
                fig_outcome.add_trace(go.Box(y=non_rewarded_rates, name="Non-rewarded", boxpoints='all', marker_color=COLOR_FA))
                fig_outcome.update_layout(
                    title="Rewarded vs Non-rewarded Firing Rates",
                    yaxis_title="Firing Rate (spikes/s)"
                )
                st.plotly_chart(fig_outcome, use_container_width=True)
            else:
                st.write("Insufficient data for outcome analysis")
        
        with col4:
            st.subheader("Choice Probability")
            cp, cp_corr = compute_choice_probability(
                event_windows_data, stimuli_outcome_df, unit_idx_adv, window
            )
            if cp is not None:
                st.metric("Choice Probability", f"{cp:.3f}")
                st.metric("CP Correlation", f"{cp_corr:.3f}")
                
                # Additional d' calculations
                d_hit_miss = compute_d_prime(event_windows_data, stimuli_outcome_df, unit_idx_adv, "Hit", "Miss", window)
                d_fa_cr = compute_d_prime(event_windows_data, stimuli_outcome_df, unit_idx_adv, "False Alarm", "CR", window)
                
                if d_hit_miss is not None:
                    st.metric("d' (Hit vs Miss)", f"{d_hit_miss:.3f}")
                if d_fa_cr is not None:
                    st.metric("d' (FA vs CR)", f"{d_fa_cr:.3f}")
            else:
                st.write("Insufficient data for choice probability analysis")
    
    with tab3:
        st.header("Generalized Linear Model Analysis")
        
        # Filter for significant units (same as tab2)
        show_only_significant_glm = st.checkbox("Show only significant units (p < 0.05)", value=False, key="glm_significant")
        
        # Get available units based on significance filter
        if show_only_significant_glm:
            significant_mask_glm = pvals < 0.05
            available_units_glm = np.where(significant_mask_glm)[0]
            if len(available_units_glm) == 0:
                st.warning("No significant units found. Showing all units.")
                available_units_glm = range(event_windows_matrix.shape[0])
            else:
                st.info(f"Showing {len(available_units_glm)} significant units out of {len(pvals)} total units")
        else:
            available_units_glm = range(event_windows_matrix.shape[0])
        
        unit_idx_glm = st.slider(
            "Select Unit for GLM",
            min_value=int(min(available_units_glm)),
            max_value=int(max(available_units_glm)),
            value=int(min(available_units_glm)),
            step=1,
            key="glm_unit"
        )
        
        # GLM window
        glm_window = st.slider("GLM analysis window", 0.1, 1.0, 0.5, step=0.1, key="glm_window")
        window_glm = (-glm_window/2, glm_window/2)
        
        # Fit GLM
        coefficients, (intercept, r_squared), y_pred, y = fit_glm_single_unit(
            event_windows_data, stimuli_outcome_df, unit_idx_glm, window_glm
        )
        # Plot predicted vs actual firing rates for GLM
        if y_pred is not None and y is not None:
            fig_glm_pred = go.Figure()
            fig_glm_pred.add_trace(go.Scatter(
                x=list(range(len(y))),
                y=y,
                mode='markers+lines',
                name='Actual',
                marker=dict(color='blue')
            ))
            fig_glm_pred.add_trace(go.Scatter(
                x=list(range(len(y_pred))),
                y=y_pred,
                mode='markers+lines',
                name='Predicted',
                marker=dict(color='orange')
            ))
            fig_glm_pred.update_layout(
                title="GLM: Actual vs Predicted Firing Rates",
                xaxis_title="Trial",
                yaxis_title="Firing Rate (spikes/s)",
                legend=dict(x=0.01, y=0.99)
            )
            st.plotly_chart(fig_glm_pred, use_container_width=True)
        if coefficients is not None:
            st.metric("R²", f"{r_squared:.3f}")
            st.metric("Intercept", f"{intercept:.3f}")
            
            # Display coefficients
            st.subheader("GLM Coefficients")
            feature_names = ["Stimulus", "Trial Type (Go=1)", "Outcome (Hit=1)"]
            for i, (name, coef) in enumerate(zip(feature_names, coefficients)):
                st.metric(name, f"{coef:.3f}")
            
            # Plot coefficient importance
            fig_coef = go.Figure()
            fig_coef.add_trace(go.Bar(
                x=feature_names,
                y=coefficients,
                marker_color=['blue', 'green', 'red']
            ))
            fig_coef.update_layout(
                title="GLM Coefficient Importance",
                yaxis_title="Coefficient Value",
                showlegend=False
            )
            st.plotly_chart(fig_coef, use_container_width=True)
        else:
            st.write("Could not fit GLM - insufficient data or convergence issues")
    
