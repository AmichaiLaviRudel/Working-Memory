import plotly.graph_objects as go
import numpy as np
from Analysis.GNG_bpod_analysis.stats_tests import mannwhitneyu
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
# Add color imports
from Analysis.GNG_bpod_analysis.colors import COLOR_GO, COLOR_NOGO, COLOR_HIT, COLOR_FA, COLOR_BLUE, COLOR_BLUE_TRANSPARENT, COLOR_ACCENT, COLOR_ACCENT_TRANSPARENT

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

def compute_stimulus_selectivity(spike_matrix, stimuli_outcome_df, unit_idx, window=(-0.2, 0.5)):
    """
    Compute stimulus selectivity for a single unit.
    Returns frequency tuning curve, SEM, and best frequency.
    """
    if 'stimulus' not in stimuli_outcome_df.columns:
        return None, None, None, None
    
    unique_stimuli = np.unique(stimuli_outcome_df['stimulus'])
    tuning_curve = []
    tuning_sem = []
    
    for stim in unique_stimuli:
        # Get trials with this stimulus
        stim_mask = stimuli_outcome_df['stimulus'] == stim
        stim_times = stimuli_outcome_df.loc[stim_mask, 'time'].values
        # Compute firing rates for each trial
        rates = [compute_peri_event_rate(spike_matrix, [t], unit_idx, window) for t in stim_times]
        if len(rates) > 0:
            avg_rate = np.mean(rates)
            sem = np.std(rates) / np.sqrt(len(rates))
        else:
            avg_rate = 0
            sem = 0
        tuning_curve.append(avg_rate)
        tuning_sem.append(sem)
    # Find best frequency (stimulus that elicits highest response)
    best_stim_idx = np.argmax(tuning_curve)
    best_stimulus = unique_stimuli[best_stim_idx]
    return unique_stimuli, tuning_curve, tuning_sem, best_stimulus

def compute_go_nogo_coding(spike_matrix, stimuli_outcome_df, unit_idx, window=(-0.1, 0.5)):
    """
    Compute d' and ROC AUC for Go vs NoGo discrimination.
    """
    # Separate Go and NoGo trials
    go_mask = np.isin(stimuli_outcome_df['outcome'], ['Hit', 'Miss'])
    nogo_mask = np.isin(stimuli_outcome_df['outcome'], ['False Alarm', 'CR'])
    
    go_times = stimuli_outcome_df.loc[go_mask, 'time'].values
    nogo_times = stimuli_outcome_df.loc[nogo_mask, 'time'].values
    
    if len(go_times) == 0 or len(nogo_times) == 0:
        return None, None, None
    
    # Compute firing rates for each trial
    go_rates = [compute_peri_event_rate(spike_matrix, [t], unit_idx, window) for t in go_times]
    nogo_rates = [compute_peri_event_rate(spike_matrix, [t], unit_idx, window) for t in nogo_times]
    
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

def compute_outcome_modulation(spike_matrix, stimuli_outcome_df, unit_idx, window=(-0.1, 0.5)):
    """
    Compare responses between rewarded (Hit) and non-rewarded (Miss/FA) trials.
    """
    # Separate rewarded and non-rewarded trials
    rewarded_mask = stimuli_outcome_df['outcome'] == 'Hit'
    non_rewarded_mask = np.isin(stimuli_outcome_df['outcome'], ['Miss', 'False Alarm'])
    
    rewarded_times = stimuli_outcome_df.loc[rewarded_mask, 'time'].values
    non_rewarded_times = stimuli_outcome_df.loc[non_rewarded_mask, 'time'].values
    
    if len(rewarded_times) == 0 or len(non_rewarded_times) == 0:
        return None, None, None
    
    # Compute firing rates
    rewarded_rates = [compute_peri_event_rate(spike_matrix, [t], unit_idx, window) for t in rewarded_times]
    non_rewarded_rates = [compute_peri_event_rate(spike_matrix, [t], unit_idx, window) for t in non_rewarded_times]
    
    # Statistical test
    try:
        stat, p_value = stats.mannwhitneyu(rewarded_rates, non_rewarded_rates, alternative='two-sided')
    except:
        stat, p_value = 0, 1
    
    return p_value, (rewarded_rates, non_rewarded_rates), (np.mean(rewarded_rates), np.mean(non_rewarded_rates))

def compute_choice_probability(spike_matrix, stimuli_outcome_df, unit_idx, window=(-0.1, 0.5)):
    """
    Calculate choice probability (CP) - trial-by-trial correlation between spike counts and Go/NoGo choice.
    """
    # Get Go trials only (where choice is relevant)
    go_mask = np.isin(stimuli_outcome_df['outcome'], ['Hit', 'Miss'])
    go_data = stimuli_outcome_df.loc[go_mask].copy()
    
    if len(go_data) == 0:
        return None, None
    
    # Get firing rates for each trial
    go_times = go_data['time'].values
    firing_rates = [compute_peri_event_rate(spike_matrix, [t], unit_idx, window) for t in go_times]
    
    # Create choice labels: 1 for Hit (correct Go), 0 for Miss (incorrect Go)
    choices = (go_data['outcome'] == 'Hit').astype(int)
    
    # Compute choice probability using ROC
    try:
        cp = roc_auc_score(choices, firing_rates)
        # Convert to correlation-like measure (-1 to 1)
        cp_corr = 2 * (cp - 0.5)
    except:
        cp = 0.5
        cp_corr = 0
    
    return cp, cp_corr

def compute_d_prime(spike_matrix, stimuli_outcome_df, unit_idx, condition1, condition2, window=(-0.1, 0.5)):
    """
    Compute d' between two conditions.
    """
    # Get trials for each condition
    mask1 = stimuli_outcome_df['outcome'] == condition1
    mask2 = stimuli_outcome_df['outcome'] == condition2
    
    times1 = stimuli_outcome_df.loc[mask1, 'time'].values
    times2 = stimuli_outcome_df.loc[mask2, 'time'].values
    
    if len(times1) == 0 or len(times2) == 0:
        return None
    
    # Compute firing rates
    rates1 = [compute_peri_event_rate(spike_matrix, [t], unit_idx, window) for t in times1]
    rates2 = [compute_peri_event_rate(spike_matrix, [t], unit_idx, window) for t in times2]
    
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

def fit_glm_single_unit(spike_matrix, stimuli_outcome_df, unit_idx, window=(-0.1, 0.5)):
    """
    Fit Generalized Linear Model (Poisson regression) to single neuron spike trains.
    """
    # Create design matrix
    design_matrix = []
    spike_counts = []
    
    for idx, row in stimuli_outcome_df.iterrows():
        # Get firing rate around this event
        rate = compute_peri_event_rate(spike_matrix, [row['time']], unit_idx, window)
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
        return None, None,  None, None


def plot_advanced_unit_analysis(spike_matrix, stimuli_outcome_df, unit_idx, bin_size=0.01):
    """
    Create comprehensive plots for advanced single unit analysis.
    """
    # Create subplots
    fig = go.Figure()
    
    # 1. Stimulus selectivity
    stimuli, tuning_curve, best_stim = compute_stimulus_selectivity(spike_matrix, stimuli_outcome_df, unit_idx)
    if stimuli is not None:
        fig.add_trace(go.Scatter(
            x=stimuli, y=tuning_curve, mode='lines+markers',
            name=f'Stimulus Tuning (Best: {best_stim:.2f})',
            line=dict(color='blue')
        ))
    
    # 2. Go/NoGo coding
    d_prime, roc_auc, (go_rates, nogo_rates) = compute_go_nogo_coding(spike_matrix, stimuli_outcome_df, unit_idx)
    
    # 3. Outcome modulation
    outcome_p, (rewarded_rates, non_rewarded_rates), (rewarded_mean, non_rewarded_mean) = compute_outcome_modulation(
        spike_matrix, stimuli_outcome_df, unit_idx
    )
    
    # 4. Choice probability
    cp, cp_corr = compute_choice_probability(spike_matrix, stimuli_outcome_df, unit_idx)
    
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

def plot_unit_psth(spike_matrix, event_times, unit_idx, bin_size=0.01, window=(-1, 2), event_outcomes=None, stim_filter="All"):
    """
    Plots PSTH for a single unit aligned to event_times, adds a vertical line at x=0, and calculates significance (Mann-Whitney U test) between pre- and post-event activity.
    Args:
        spike_matrix: 2D array [units x time_bins] (firing rate)
        event_times: 1D array of event times (in seconds)
        unit_idx: int, index of the unit to plot
        bin_size: float, bin size in seconds for PSTH
        window: tuple, time window around event (start, end) in seconds
    """
    # Filter event_times by stim_filter
    if event_outcomes is not None and stim_filter != "All":
        if stim_filter == "Go":
            mask = np.isin(event_outcomes, ["Hit", "Miss"])
        elif stim_filter == "NoGo":
            mask = np.isin(event_outcomes, ["False Alarm", "CR"])
        else:
            mask = np.ones(len(event_times), dtype=bool)
        event_times = np.array(event_times)[mask]
    n_bins = spike_matrix.shape[1]
    peri_event_window = np.arange(window[0], window[1], bin_size)
    peri_event_hist = np.zeros_like(peri_event_window)
    n_events = 0
    for t in event_times:
        event_bin = int(t / bin_size)
        start_bin = event_bin + int(window[0] / bin_size)
        end_bin = event_bin + int(window[1] / bin_size)
        if start_bin < 0 or end_bin > n_bins:
            continue
        segment = spike_matrix[unit_idx, start_bin:end_bin]
        if segment.shape[0] != peri_event_hist.shape[0]:
            continue
        peri_event_hist += segment
        n_events += 1
    if n_events > 0:
        peri_event_hist = peri_event_hist / n_events  # average

    # Calculate significance: pre (window[0] to 0), post (0 to window[1])
    zero_idx = np.where(peri_event_window >= 0)[0][0]
    pre = peri_event_hist[:zero_idx]
    post = peri_event_hist[zero_idx:]
    stat, p = mannwhitneyu(pre, post, alternative='two-sided')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=peri_event_window, y=peri_event_hist, mode='lines', name="PSTH"))
    fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="red")
    fig.update_layout(
        title=f"PSTH for Unit {unit_idx} (p={p:.3g})",
        xaxis_title="Time (s) from event",
        yaxis_title="Firing rate (spikes/s)",
        bargap=0
    )
    return fig

    
def compute_psth_pvalues(spike_matrix, event_times, bin_size=0.01, window=(-1, 2)):
    from Analysis.GNG_bpod_analysis.stats_tests import mannwhitneyu
    n_units = spike_matrix.shape[0]
    pvals = []
    n_bins = spike_matrix.shape[1]
    peri_event_window = np.arange(window[0], window[1], bin_size)
    for unit_idx in range(n_units):
        peri_event_hist = np.zeros_like(peri_event_window)
        n_events = 0
        for t in event_times:
            event_bin = int(t / bin_size)
            start_bin = event_bin + int(window[0] / bin_size)
            end_bin = event_bin + int(window[1] / bin_size)
            if start_bin < 0 or end_bin > n_bins:
                continue
            segment = spike_matrix[unit_idx, start_bin:end_bin]
            if segment.shape[0] != peri_event_hist.shape[0]:
                continue
            peri_event_hist += segment
            n_events += 1
        if n_events > 0:
            peri_event_hist = peri_event_hist / n_events  # average
        zero_idx = np.where(peri_event_window >= 0)[0][0]
        pre = peri_event_hist[:zero_idx]
        post = peri_event_hist[zero_idx:]
        stat, p = mannwhitneyu(pre, post, alternative='two-sided')
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


def create_raster_plot_from_event_windows(event_windows_matrix, time_axis, unit_idx, event_outcomes=None, stim_filter="All"):
    """
    Creates a raster plot from event windows data for a specific unit.
    
    Args:
        event_windows_matrix (np.ndarray): 3D matrix of shape [units × time × events]
        time_axis (np.ndarray): Time axis for the window
        unit_idx (int): Index of the unit to plot
        event_outcomes (np.ndarray): Array of event outcomes for filtering
        stim_filter (str): Filter for events ("All", "Go", "NoGo")
        
    Returns:
        plotly.graph_objects.Figure: Raster plot figure
    """
    # Get the unit's data
    unit_data = event_windows_matrix[unit_idx, :, :]  # Shape: [time × events]
    
    # Filter events based on outcomes if provided
    if event_outcomes is not None and stim_filter != "All":
        if stim_filter == "Go":
            event_mask = event_outcomes == 1
        elif stim_filter == "NoGo":
            event_mask = event_outcomes == 0
        else:
            event_mask = np.ones(len(event_outcomes), dtype=bool)
        
        unit_data = unit_data[:, event_mask]
        filtered_outcomes = event_outcomes[event_mask] if event_outcomes is not None else None
    else:
        filtered_outcomes = event_outcomes
    
    # Create raster plot
    fig = go.Figure()
    
    # For each event, find time points where firing rate > 0 and plot them
    for event_idx in range(unit_data.shape[1]):
        # Get firing rates for this event
        firing_rates = unit_data[:, event_idx]
        
        # Find time points where there are spikes (firing rate > 0)
        spike_times = time_axis[firing_rates > 0]
        
        if len(spike_times) > 0:
            # Color based on outcome if available
            if filtered_outcomes is not None:
                color = 'red' if filtered_outcomes[event_idx] == 1 else 'blue'
            else:
                color = 'black'
            
            # Add raster lines for this event
            for spike_time in spike_times:
                fig.add_trace(go.Scatter(
                    x=[spike_time, spike_time],
                    y=[event_idx - 0.4, event_idx + 0.4],
                    mode='lines',
                    line=dict(color=color, width=1),
                    showlegend=False
                ))
    
    # Update layout
    fig.update_layout(
        title=f"Raster Plot - Unit {unit_idx}",
        xaxis_title="Time (s)",
        yaxis_title="Event",
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(zeroline=True, zerolinecolor='gray'),
        yaxis=dict(zeroline=True, zerolinecolor='gray')
    )
    
    return fig


def single_unit_analysis_panel(
    spike_matrix, stimuli_outcome_df, selected_folder=None, bin_size=0.01, default_window=1.0):
    import streamlit as st
    import numpy as np
    from .npxl_single_unit_analysis import plot_unit_psth, compute_psth_pvalues
    from Analysis.NPXL_analysis.population_analysis import plot_population_heatmap

    event_times = stimuli_outcome_df['time'].values 
    event_outcomes = stimuli_outcome_df['outcome'].values if 'outcome' in stimuli_outcome_df.columns else None
    
    # Compute p-values for all units (available to all tabs)
    default_display_window = (-default_window/2, default_window/2)
    pvals = compute_psth_pvalues(spike_matrix, event_times, bin_size=bin_size, window=default_display_window)
    
    # Create tabs for different analysis types
    tab1, tab2, tab3 = st.tabs(["Basic PSTH", "Advanced Analysis", "GLM Analysis"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            stim_filter = st.radio(
                "Stimulus filter",
                options=["All", "Go", "NoGo"],
                index=0,
                horizontal=True
            )
        with col2:
            peri_event = st.slider("PSTH window size", 0.1, 3.0, default_window, step=0.10)
        display_window = (-peri_event/2, peri_event/2)

        # Recompute p-values with current window
        pvals = compute_psth_pvalues(spike_matrix, event_times, bin_size=bin_size, window=display_window)
        sorted_indices = np.argsort(pvals)
        sorted_pvals = pvals[sorted_indices]

        # Save p-values if folder is provided
        if selected_folder is not None:
            # Display p-value statistics
            significant_units = np.sum(pvals < 0.05)
            st.info(f"P-value analysis: {significant_units}/{len(pvals)} units significant (p < 0.05)")
            
            # Auto-save p-values
            save_success = save_pvalues_to_folder(pvals, selected_folder, window=display_window, bin_size=bin_size)
            if save_success:
                st.success(f"P-values auto-saved to {selected_folder}")
            else:
                st.error("Failed to auto-save p-values")
            
            # Manual save button
            if st.button("Save P-values Manually"):
                manual_save_success = save_pvalues_to_folder(pvals, selected_folder, window=display_window, bin_size=bin_size)
                if manual_save_success:
                    st.success(f"P-values manually saved to {selected_folder}")
                else:
                    st.error("Failed to manually save p-values")

        # Use a slider to select the unit by sorted order
        unit_rank = st.slider("Unit (sorted by p-value)", 0, len(sorted_pvals) - 1, 0)
        unit_idx = sorted_indices[unit_rank]

        # Load event windows data if available
        event_windows_data = None
        if selected_folder is not None:
            event_windows_data = load_event_windows_data(selected_folder)
            if event_windows_data is not None:
                st.success("Event windows data loaded successfully")
            else:
                st.warning("Event windows data not found")

        # Create two columns for PSTH and raster plot
        psth_col, raster_col = st.columns(2)
        
        with psth_col:
            st.subheader("PSTH")
            # Plot PSTH for the selected unit (filtered)
            psth_fig = plot_unit_psth(
                spike_matrix, event_times, unit_idx,
                bin_size=bin_size, window=display_window,
                event_outcomes=event_outcomes, stim_filter=stim_filter
            )
            st.plotly_chart(psth_fig, use_container_width=True)
        
        with raster_col:
            st.subheader("Raster Plot")
            if event_windows_data is not None:
                event_windows_matrix, time_axis, valid_event_indices, event_stimuli_outcome_df, metadata = event_windows_data
                
                # Get outcomes for the valid events
                event_outcomes_for_raster = None
                if 'outcome' in event_stimuli_outcome_df.columns:
                    event_outcomes_for_raster = event_stimuli_outcome_df['outcome'].values
                
                # Create raster plot
                raster_fig = create_raster_plot_from_event_windows(
                    event_windows_matrix, time_axis, unit_idx,
                    event_outcomes=event_outcomes_for_raster, stim_filter=stim_filter
                )
                st.plotly_chart(raster_fig, use_container_width=True)
                
                # Display metadata
                if metadata:
                    st.caption(f"Event windows: {metadata.get('n_events', 'N/A')} events, "
                             f"{metadata.get('n_time_bins', 'N/A')} time bins")
            else:
                st.info("Event windows data not available. Run preprocessing to generate event windows.")
    
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
                available_units = range(spike_matrix.shape[0])
            else:
                st.info(f"Showing {len(available_units)} significant units out of {len(pvals)} total units")
        else:
            available_units = range(spike_matrix.shape[0])
        
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
        analysis_window = st.slider("Analysis window", 0.1, 1.0, 0.5, step=0.1)
        window = (-analysis_window/2, analysis_window/2)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Stimulus Selectivity")
            stimuli, tuning_curve, tuning_sem, best_stim = compute_stimulus_selectivity(
                spike_matrix, stimuli_outcome_df, unit_idx_adv, window
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
                spike_matrix, stimuli_outcome_df, unit_idx_adv, window
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
                spike_matrix, stimuli_outcome_df, unit_idx_adv, window
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
                spike_matrix, stimuli_outcome_df, unit_idx_adv, window
            )
            if cp is not None:
                st.metric("Choice Probability", f"{cp:.3f}")
                st.metric("CP Correlation", f"{cp_corr:.3f}")
                
                # Additional d' calculations
                d_hit_miss = compute_d_prime(spike_matrix, stimuli_outcome_df, unit_idx_adv, "Hit", "Miss", window)
                d_fa_cr = compute_d_prime(spike_matrix, stimuli_outcome_df, unit_idx_adv, "False Alarm", "CR", window)
                
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
                available_units_glm = range(spike_matrix.shape[0])
            else:
                st.info(f"Showing {len(available_units_glm)} significant units out of {len(pvals)} total units")
        else:
            available_units_glm = range(spike_matrix.shape[0])
        
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
            spike_matrix, stimuli_outcome_df, unit_idx_glm, window_glm
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
    
