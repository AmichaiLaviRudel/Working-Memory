import plotly.graph_objects as go
from Analysis.GNG_bpod_analysis.colors import COLOR_HIT, COLOR_MISS, COLOR_FA, COLOR_CR, GO_COLORS, NOGO_COLORS, COLOR_ACCENT
import streamlit as st
import numpy as np
import pandas as pd
from Analysis.NPXL_analysis.npxl_single_unit_analysis import compute_stimulus_selectivity, compute_psth_pvalues_from_event_windows
from Analysis.GNG_bpod_analysis.colors import COLOR_ORANGE

def plot_population_heatmap(event_windows_matrix, stimuli_outcome_df, metadata):
    """
    Plot population heatmap that reconstructs events back into a time stream with navigation.
    
    Args:
        event_windows_matrix: 3D array [units × time_bins_per_window × n_events]
        stimuli_outcome_df: DataFrame with event information including 'time' column
        metadata: Dictionary with window_duration, bin_size, etc.
    """
    
    if len(event_windows_matrix.shape) != 3:
        st.error("Expected 3D event windows matrix [units × time × events]")
        return
    
    n_units, n_time_per_event, n_events = event_windows_matrix.shape
    
    # Get parameters from metadata
    window_duration = float(metadata.get('window_duration', 4.0))
    bin_size = float(metadata.get('bin_size', 0.1))
    
    # Navigation controls
    st.subheader("Heatmap Navigation 🗺️")
    display_window_duration = st.slider(
        "Display Window Duration (seconds)", 
        min_value=5.0, 
        max_value=60.0, 
        value=20.0, 
        step=1.0,
        help="Duration of the time window to display"
    )
    
    # Get original event times (in bins) from stimuli_outcome_df
    original_event_times = stimuli_outcome_df['time'].values
    
    # Calculate the range of time we need to cover
    min_event_time = np.min(original_event_times)
    max_event_time = np.max(original_event_times)
    
    # Add buffer around events to show context
    buffer_duration = window_duration  # Same as window duration
    buffer_bins = int(buffer_duration / bin_size)
    
    # Calculate the total time range needed
    # Add extra buffer to ensure all events fit completely
    extra_buffer = int(window_duration / bin_size)  # Full window duration as extra buffer
    start_time_bin = int(min_event_time - buffer_bins - extra_buffer)
    end_time_bin = int(max_event_time + buffer_bins + extra_buffer)
    
    total_time_bins = end_time_bin - start_time_bin
    total_duration_seconds = total_time_bins * bin_size
    
    # Navigation slider for time position
    max_start_time = max(0, total_duration_seconds - display_window_duration)
    
    if max_start_time > 0:
        # Add navigation buttons
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            if st.button("⏮️ Start", help="Jump to beginning"):
                st.session_state.heatmap_start_time = 0.0
        
        with col2:
            if st.button("⏪ Back", help="Go back 10 seconds"):
                current_time = st.session_state.get('heatmap_start_time', 0.0)
                st.session_state.heatmap_start_time = max(0.0, current_time - 10.0)
        
        with col3:
            if st.button("⏩ Forward", help="Go forward 10 seconds"):
                current_time = st.session_state.get('heatmap_start_time', 0.0)
                st.session_state.heatmap_start_time = min(max_start_time, current_time + 10.0)
        
        with col4:
            if st.button("⏭️ End", help="Jump to end"):
                st.session_state.heatmap_start_time = max_start_time
        
        with col5:
            if st.button("🔄 Reset", help="Reset to beginning"):
                st.session_state.heatmap_start_time = 0.0
        
        # Initialize session state if not exists
        if 'heatmap_start_time' not in st.session_state:
            st.session_state.heatmap_start_time = 0.0
        
        start_time_position = st.slider(
            "Start Time Position (seconds)",
            min_value=0.0,
            max_value=max_start_time,
            value=st.session_state.heatmap_start_time,
            step=1.0,
            help="Starting position of the display window",
            key="heatmap_time_slider"
        )
        
        # Update session state from slider
        st.session_state.heatmap_start_time = start_time_position
        
    else:
        start_time_position = 0.0
        st.info(f"Recording duration ({total_duration_seconds:.1f}s) is shorter than display window. Showing entire recording.")
    
    # Create the reconstructed matrix
    reconstructed_matrix = np.zeros((n_units, total_time_bins))
    
    # Place each event window in the correct position in the time stream
    # Use the actual window size from the event windows matrix
    actual_window_size = n_time_per_event
    window_half_duration_bins = actual_window_size // 2
    
    for event_idx, original_time in enumerate(original_event_times):
        if event_idx >= n_events:  # Safety check
            break
            
        # Calculate where this event should be placed in the reconstructed matrix
        event_center_in_reconstructed = int(original_time - start_time_bin)
        
        # Calculate start and end positions for this event window
        window_start = event_center_in_reconstructed - window_half_duration_bins
        window_end = window_start + actual_window_size  # Use actual window size
        
        # Ensure we don't go out of bounds and that the slice size matches the data
        if window_start >= 0 and window_end <= total_time_bins:
            # Verify that the slice size matches the event window size
            slice_size = window_end - window_start
            if slice_size == actual_window_size:
                try:
                    # Place the event window in the reconstructed matrix
                    reconstructed_matrix[:, window_start:window_end] = event_windows_matrix[:, :, event_idx]
                except ValueError as e:
                    st.warning(f"Error placing event {event_idx}: {e}. Shapes: reconstructed slice {reconstructed_matrix[:, window_start:window_end].shape}, event window {event_windows_matrix[:, :, event_idx].shape}")
            else:
                st.warning(f"Slice size mismatch for event {event_idx}: expected {actual_window_size}, got {slice_size}")
        else:
            # Try to clip the window to fit within bounds
            clipped_start = max(0, window_start)
            clipped_end = min(total_time_bins, window_end)
            
            # Calculate corresponding indices in the event window
            event_start_offset = clipped_start - window_start
            event_end_offset = event_start_offset + (clipped_end - clipped_start)
            
            if event_start_offset >= 0 and event_end_offset <= actual_window_size:
                try:
                    reconstructed_matrix[:, clipped_start:clipped_end] = event_windows_matrix[:, event_start_offset:event_end_offset, event_idx]
                    st.info(f"Event {event_idx} partially placed (clipped to fit bounds)")
                except ValueError as e:
                    st.warning(f"Error placing clipped event {event_idx}: {e}")
            else:
                st.warning(f"Event {event_idx} completely outside bounds or clipping failed")
    
    # Create time axis for the reconstructed matrix (in seconds relative to recording start)
    time_axis_bins = np.arange(start_time_bin, end_time_bin)
    time_axis_seconds = time_axis_bins * bin_size
    
    # Calculate display window boundaries
    display_start_bin = int(start_time_position / bin_size)
    display_end_bin = int((start_time_position + display_window_duration) / bin_size)
    display_end_bin = min(display_end_bin, total_time_bins)
    
    # Extract the subset of data for the display window
    matrix_subset = reconstructed_matrix[:, display_start_bin:display_end_bin]
    time_subset = time_axis_seconds[display_start_bin:display_end_bin]
    
    # Display information about the current view
    st.info(f"Showing {display_window_duration}s window from {start_time_position:.1f}s to {start_time_position + display_window_duration:.1f}s | Total duration: {total_duration_seconds:.1f}s 🖥️")
    
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix_subset,
            x=time_subset,
            colorscale='Greys',
            colorbar=dict(title="Spikes/sec"),
        )
    )
    outcome_color_map = {
        "Hit": COLOR_HIT,
        "Miss": COLOR_MISS,
        "False Alarm": COLOR_FA,
        "CR": COLOR_CR,
    }
    
    # Track which outcomes we've already added to legend
    legend_added = set()
    
    # Add event markers at their actual times (only those within the display window)
    if 'time' in stimuli_outcome_df.columns and 'outcome' in stimuli_outcome_df.columns:
        # Calculate display window boundaries in seconds
        display_start_seconds = start_time_position
        display_end_seconds = start_time_position + display_window_duration
        
        for t, outcome in zip(stimuli_outcome_df['time'], stimuli_outcome_df['outcome']):
            color = outcome_color_map.get(outcome, "white")
            
            # Convert event time from bins to seconds for plotting
            event_time_seconds = t * bin_size
            
            # Only show events within the current display window
            if display_start_seconds <= event_time_seconds <= display_end_seconds:
                # Show legend only once per outcome type
                showlegend = outcome not in legend_added
                if showlegend:
                    legend_added.add(outcome)
                
                fig.add_vline(
                    x=event_time_seconds, 
                    line_width=2, 
                    line_dash="solid", 
                    line_color=color,
                    name=outcome,
                    showlegend=showlegend
                )
    
    fig.update_layout(
        xaxis=dict(
            title="Time in Recording (s)",
            constrain='domain'
        ),
        yaxis_title="Unit",
        title=f"Population Activity Heatmap | {start_time_position:.1f}s - {start_time_position + display_window_duration:.1f}s",
        legend=dict(
            title="Events by Outcome",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    fig.update_layout(height=800)
    st.plotly_chart(fig, use_container_width=True)


def plot_best_stimulus_panel(event_windows_matrix, stimuli_outcome_df, metadata, window=(-0.1, 0.5)):
    """
    Plot a panel aggregating each unit's best stimulus and its firing rate.

    Args:
        event_windows_matrix: 3D array [units × time_bins × trials]
        stimuli_outcome_df: DataFrame with trial information (must include 'stimulus')
        metadata: Dictionary with recording parameters (expects 'bin_size' and 'window_duration')
        window: tuple of (start, end) time in seconds relative to event for computing responses
    """
    if 'stimulus' not in stimuli_outcome_df.columns:
        st.warning("No 'stimulus' column found; cannot compute best stimulus per unit.")
        return

    if len(event_windows_matrix.shape) != 3:
        st.error("Expected 3D event windows matrix [units × time × trials]")
        return

    n_units, n_time_bins, n_trials = event_windows_matrix.shape

    # Reconstruct time axis from metadata
    bin_size = float(metadata.get('bin_size', 0.1))
    window_duration = float(metadata.get('window_duration', 3.0))
    time_axis = np.arange(-window_duration, window_duration, bin_size)
    if time_axis.shape[0] != n_time_bins:
        # Fallback to length-consistent axis if metadata-derived axis length mismatches data
        time_axis = np.linspace(-window_duration, window_duration, n_time_bins, endpoint=False)

    # Optional filtering: only significant units
    colf1, colf2 = st.columns(2)
    with colf1:
        only_significant = st.checkbox("Only significant units (p < α)", value=False, help="Filter units by baseline vs post-event significance using Mann-Whitney U")
    with colf2:
        alpha = st.number_input("α (significance threshold)", min_value=0.001, max_value=0.2, value=0.05, step=0.001, format="%.3f")

    # Build event_windows_data tuple expected by compute_stimulus_selectivity
    valid_event_indices = np.arange(n_trials)
    event_windows_data = (event_windows_matrix, time_axis, valid_event_indices, stimuli_outcome_df, metadata)

    # Determine indices to include
    unit_indices = np.arange(n_units)
    if only_significant:
        # Use the full event window for significance computation (split around 0)
        event_times = stimuli_outcome_df['time'].values if 'time' in stimuli_outcome_df.columns else np.arange(n_trials)
        pvals = compute_psth_pvalues_from_event_windows(
            event_windows_matrix,
            event_times,
            bin_size=bin_size,
            window=(-window_duration, window_duration)
        )
        sig_mask = np.isfinite(pvals) & (pvals < float(alpha))
        unit_indices = np.where(sig_mask)[0]

    best_stimuli = []
    best_rates = []

    for unit_idx in unit_indices:
        stimuli, tuning_curve, tuning_sem, best_stim = compute_stimulus_selectivity(
            event_windows_data, stimuli_outcome_df, unit_idx, window
        )
        if stimuli is None or best_stim is None:
            continue

        # Get best rate from tuning curve
        if isinstance(tuning_curve, list):
            tuning_array = np.array(tuning_curve)
        else:
            tuning_array = tuning_curve

        if tuning_array.size == 0 or np.all(~np.isfinite(tuning_array)):
            continue

        # Use the index of the best stimulus to fetch its firing rate
        try:
            best_index = int(np.where(np.array(stimuli) == best_stim)[0][0])
            best_rate = float(tuning_array[best_index])
        except Exception:
            best_rate = float(np.nanmax(tuning_array))

        best_stimuli.append(best_stim)
        best_rates.append(best_rate)

    if len(best_stimuli) == 0:
        st.warning("No best stimulus data could be computed across units.")
        return

    # Convert to arrays
    best_stimuli_arr = np.array(best_stimuli, dtype=object)
    best_rates_arr = np.array(best_rates, dtype=float)

    # Determine jitter strategy (numeric vs categorical x)
    def _is_all_numeric(values):
        try:
            _ = np.asarray(values, dtype=float)
            return True
        except Exception:
            return False

    is_numeric_x = _is_all_numeric(best_stimuli_arr)

    if is_numeric_x:
        x_vals = np.asarray(best_stimuli_arr, dtype=float)
        # Multiplicative jitter in log space to keep positivity
        log_x = np.log10(x_vals)
        unique_sorted_log = np.unique(np.round(log_x, 12))
        if unique_sorted_log.size > 1:
            min_diff_log = np.min(np.diff(unique_sorted_log))
            log_jitter_sigma = 0.05 * float(min_diff_log)
        else:
            log_jitter_sigma = 0.02
        jitter_factors = 10 ** np.random.normal(0.0, log_jitter_sigma, size=x_vals.shape[0])
        jittered_x = x_vals * jitter_factors
        # Ensure strictly positive for log axis
        jittered_x = np.clip(jittered_x, a_min=np.finfo(float).eps, a_max=None)
        # Prepare ticks: spell out all unique stimulus values
        tick_vals = np.sort(np.unique(x_vals))
        tick_text = [f"{v:.2f}" for v in tick_vals]
    else:
        categories = list(pd.unique(best_stimuli_arr))
        cat_to_idx = {cat: i for i, cat in enumerate(categories)}
        x_idx = np.array([cat_to_idx[c] for c in best_stimuli_arr], dtype=float)
        jitter_sigma = 0.15
        jittered_x = x_idx + np.random.normal(0.0, jitter_sigma, size=x_idx.shape[0])
        tick_vals = list(range(len(categories)))
        tick_text = categories

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=jittered_x,
            y=best_rates_arr,
            mode='markers',
            name='Units',
            marker=dict(size=7, opacity=0.8, color=COLOR_ACCENT)
        )
    )

    # Add vertical reference lines at x = 1 and x = 1.5
    def _x_position_for_value(val):
        if is_numeric_x:
            return val
        # categorical: try to map exact match or numeric-close match using tick lists
        if val in (tick_text or []):
            return (tick_vals or [])[ (tick_text or []).index(val) ]
        try:
            for i, c in enumerate(tick_text or []):
                try:
                    if np.isclose(float(c), float(val)):
                        return (tick_vals or [])[i]
                except Exception:
                    continue
        except Exception:
            pass
        return None

    for vline_x in [1.0, 1.5]:
        xpos = _x_position_for_value(vline_x)
        if xpos is not None:
            fig.add_vline(x=float(xpos), line_dash="dash", line_color="black", opacity=0.6)

    # Configure x-axis: log scale when numeric, categorical otherwise
    if is_numeric_x:
        xaxis_kwargs = dict(title="Stimulus", type='log', tickmode='array', tickvals=tick_vals, ticktext=tick_text)
    else:
        xaxis_kwargs = dict(title="Stimulus", tickmode='array', tickvals=tick_vals, ticktext=tick_text)

    fig.update_layout(
        title="Best Stimulus vs Best Firing Rate per Unit",
        xaxis=xaxis_kwargs,
        yaxis_title="Firing rate at best stimulus (spikes/s)",
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    st.plotly_chart(fig, use_container_width=True)

    # Barplot of unit percentage per stimulus (based on best stimulus)
    counts_series = pd.Series(best_stimuli_arr).value_counts()

    # Determine denominator based on included units (filters)
    denom_units = len(unit_indices) if 'unit_indices' in locals() and unit_indices is not None else n_units

    if is_numeric_x:
        unique_vals = np.sort(np.unique(best_stimuli_arr.astype(float)))
        bar_x = unique_vals
        bar_counts = np.array([int(counts_series.get(v, 0)) for v in unique_vals])
        bar_y = (bar_counts.astype(float) / float(denom_units)) * 100.0
        bar_xaxis_kwargs = dict(title="Stimulus", type='log', tickmode='array', tickvals=unique_vals, ticktext=[f"{v:.2f}" for v in unique_vals])
    else:
        bar_x = list(range(len(tick_text or [])))
        bar_counts = np.array([int(np.sum(best_stimuli_arr == cat)) for cat in (tick_text or [])])
        bar_y = (bar_counts.astype(float) / float(denom_units)) * 100.0
        bar_xaxis_kwargs = dict(title="Stimulus", tickmode='array', tickvals=bar_x, ticktext=(tick_text or []))

    # Base bar: percentage of units per best stimulus
    fig_bar = go.Figure(
        data=[go.Bar(x=bar_x, y=bar_y, marker_color=COLOR_ACCENT, name='Units (%)', opacity=0.85)]
    )

    # Secondary line: average firing rate per best stimulus (sum divided by number of included units)
    if is_numeric_x:
        # Align sums to unique_vals order
        stim_vals = best_stimuli_arr.astype(float)
        sum_rates = np.array([float(np.sum(best_rates_arr[np.isclose(stim_vals, v)])) for v in unique_vals])
    else:
        # Align sums to tick_text order
        sum_rates = np.array([float(np.sum(best_rates_arr[best_stimuli_arr == cat])) for cat in (tick_text or [])])

    avg_rates = sum_rates / float(denom_units) if float(denom_units) > 0 else np.zeros_like(sum_rates)

    fig_bar.add_trace(
        go.Scatter(
            x=bar_x,
            y=avg_rates,
            name='Avg FR per unit',
            mode='lines+markers',
            line=dict(color=COLOR_ORANGE, width=2),
            marker=dict(size=6, color=COLOR_ORANGE),
            yaxis='y2'
        )
    )
    # Add vertical lines at x = 1 and x = 1.5 on the bar plot
    for vline_x in [1.0, 1.5]:
        xpos = _x_position_for_value(vline_x)
        if xpos is not None:
            fig_bar.add_vline(x=float(xpos), line_dash="dash", line_color="black", opacity=0.6)

    fig_bar.update_layout(
        title="Best Stimulus Summary",
        xaxis=bar_xaxis_kwargs,
        yaxis=dict(title="Units (%)", range=[0, 100]),
        yaxis2=dict(title="Avg FR per unit (spikes/s)", overlaying='y', side='right', showgrid=False),
        barmode='overlay',
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    st.plotly_chart(fig_bar, use_container_width=True)


def advanced_population_analysis_panel(event_windows_matrix, stimuli_outcome_df, metadata, time_axis):
    """
    Advanced population analysis panel: outcome and action decoding with efficient, balanced CV.
    """
    from sklearn.svm import SVC
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import plotly.graph_objects as go
    n_units, n_time_bins, n_trials = event_windows_matrix.shape
    key_prefix = f"adv_{abs(hash((int(n_units), int(n_time_bins), int(n_trials))))}"

    # Stimulus color map (rounded to 2 decimals strings)
    _STIM_COLOR_MAP = {
        '0.71': '#006837', '2.12': '#006837',  # Dark green
        '0.77': '#1A9850', '1.95': '#1A9850',  # Medium green
        '0.84': '#1A9850', '1.79': '#1A9850',  # Medium green (same as 0.77)
        '0.91': '#66BD63', '1.65': '#66BD63',  # Light green
        '1.08': '#FDAE61',                     # Medium orange
        '1.17': '#F46D43',                     # Coral orange
        '1.28': '#D73027',                     # Medium red
        '1.39': '#A50026',                     # Dark red
    }

    def _stim_hex_color(value):
        try:
            key = f"{float(value):.2f}"
        except Exception:
            return COLOR_ACCENT
        return _STIM_COLOR_MAP.get(key, COLOR_ACCENT)

    def _balanced_indices(labels, class_pos, class_neg):
        pos_idx = np.where(labels == class_pos)[0]
        neg_idx = np.where(labels == class_neg)[0]
        min_n = int(min(len(pos_idx), len(neg_idx)))
        if min_n < 2:
            return None, None
        rng = np.random.default_rng(seed=st.session_state.get('adv_dec_seed', 42))
        pos_sel = rng.choice(pos_idx, size=min_n, replace=False)
        neg_sel = rng.choice(neg_idx, size=min_n, replace=False)
        sel = np.concatenate([pos_sel, neg_sel])
        y = np.r_[np.ones(min_n, dtype=int), np.zeros(min_n, dtype=int)]
        return sel, y

    def _time_resolved_decode(data_u_t_trials, sel_idx, y, t_start=None, t_end=None, avg_win_sec=0.0):
        if sel_idx is None:
            return None, None
        dt = float(time_axis[1] - time_axis[0]) if len(time_axis) > 1 else 1.0
        s = 0 if t_start is None else int(np.searchsorted(time_axis, t_start, side='left'))
        e = data_u_t_trials.shape[1] if t_end is None else int(np.searchsorted(time_axis, t_end, side='right'))
        e = max(e, s + 1)
        half_w = int(max(0, round((avg_win_sec / dt) / 2.0)))

        times, acc = [], []
        n_splits = int(max(2, min(5, np.min(np.bincount(y)))))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1, gamma='scale', class_weight='balanced'))

        # Slice trials once
        data_sel = data_u_t_trials[:, :, sel_idx]
        for t in range(s, e):
            t0 = max(s, t - half_w)
            t1 = min(e, t + half_w + 1)
            X = data_sel[:, t0:t1, :].mean(axis=1).T
            scores = cross_val_score(clf, X, y, cv=cv)
            acc.append(float(np.mean(scores)))
            times.append(float(time_axis[t]))
        return np.array(times), np.array(acc)

    # TABS
    stimulus_tab, outcome_tab, action_tab = st.tabs(["Stimulus Decoder", "Outcome Decoder", "Action Decoder"])

    with stimulus_tab:
        if 'stimulus' not in stimuli_outcome_df.columns:
            st.warning("'stimulus' column not found in data.")
            return
        stimulus_labels = np.round(stimuli_outcome_df['stimulus'].values, 2)
        unique_stimuli = np.unique(stimulus_labels)
        if len(unique_stimuli) < 2:
            st.warning("Need at least two unique stimuli for decoding.")
            return
        t_range = st.slider(
            "Time range (s)", float(time_axis[0]), float(time_axis[-1]), (-2.0, 4.0), step=0.05, key=f"{key_prefix}_stim_range"
        )
        avg_win = st.number_input("Averaging window (s)", 0.0, 2.0, 0.0, 0.05, key=f"{key_prefix}_stim_avg")
        # Pair first vs last, second vs second-last, etc.
        num_pairs = len(unique_stimuli) // 2
        fig = go.Figure()
        plotted = 0
        for i in range(num_pairs):
            stim_pos, stim_neg = unique_stimuli[i], unique_stimuli[-(i+1)]
            sel_idx, y = _balanced_indices(stimulus_labels, stim_pos, stim_neg)
            if sel_idx is None:
                continue
            times, acc = _time_resolved_decode(
                event_windows_matrix, sel_idx, y,
                t_start=t_range[0], t_end=t_range[1], avg_win_sec=avg_win
            )
            if times is None:
                continue
            fig.add_trace(go.Scatter(
                x=times, y=acc, mode='lines',
                name=f"{stim_pos} vs {stim_neg}",
                line=dict(color=_stim_hex_color(stim_pos))
            ))
            plotted += 1
        if plotted == 0:
            st.warning("No valid stimulus pairs after balancing.")
            return
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray")
        fig.add_shape(
            type="line",
            x0=0, x1=2,
            y0=0, y1=0,
            line=dict(color="black", width=5, dash="solid"),
            xref="x", yref="y"
        )
        fig.update_layout(
            title="Time-resolved decoding: paired stimuli (first-vs-last, ...)",
            xaxis=dict(title="Time (s)", range=[t_range[0], t_range[1]]),
            yaxis_title="Accuracy",
            yaxis=dict(range=[0,1])
        )
        st.plotly_chart(fig, use_container_width=True)

        # New figure: compare adjacent stimulus pairs (near each other)
        if len(unique_stimuli) >= 2:
            fig_adj = go.Figure()
            plotted_adj = 0
            for i in range(len(unique_stimuli) - 1):
                stim_a, stim_b = unique_stimuli[i], unique_stimuli[i+1]
                sel_idx_adj, y_adj = _balanced_indices(stimulus_labels, stim_a, stim_b)
                if sel_idx_adj is None:
                    continue
                times_adj, acc_adj = _time_resolved_decode(
                    event_windows_matrix, sel_idx_adj, y_adj,
                    t_start=t_range[0], t_end=t_range[1], avg_win_sec=avg_win
                )
                if times_adj is None:
                    continue
                fig_adj.add_trace(go.Scatter(
                    x=times_adj, y=acc_adj, mode='lines',
                    name=f"{stim_a} vs {stim_b}",
                    line=dict(color=_stim_hex_color(stim_a))
                ))
                plotted_adj += 1
            if plotted_adj > 0:
                fig_adj.add_hline(y=0.5, line_dash="dash", line_color="gray")
                fig_adj.add_shape(
                    type="line",
                    x0=0, x1=2,
                    y0=0, y1=0,
                    line=dict(color="black", width=5, dash="solid"),
                    xref="x", yref="y"
                )
                fig_adj.update_layout(
                    title="Time-resolved decoding: adjacent stimulus pairs",
                    xaxis=dict(title="Time (s)", range=[t_range[0], t_range[1]]),
                    yaxis_title="Accuracy",
                    yaxis=dict(range=[0,1])
                )
                st.plotly_chart(fig_adj, use_container_width=True)

        # Optional: overall (time-averaged) accuracy boxplots
        show_box = st.checkbox("Show overall accuracy (boxplots)", value=False, key=f"{key_prefix}_stim_box")
        if show_box:
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import StratifiedKFold, cross_val_score
            # Build time mask once
            s = int(np.searchsorted(time_axis, t_range[0], side='left'))
            e = int(np.searchsorted(time_axis, t_range[1], side='right'))
            e = max(e, s + 1)

            def overall_scores_for_pairs(pairs):
                boxes = []
                for a, b in pairs:
                    sel_idx, y = _balanced_indices(stimulus_labels, a, b)
                    if sel_idx is None:
                        continue
                    X = event_windows_matrix[:, s:e, sel_idx].mean(axis=1).T
                    n_splits = int(max(2, min(5, np.min(np.bincount(y)))))
                    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                    clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1, gamma='scale', class_weight='balanced'))
                    scores = cross_val_score(clf, X, y, cv=cv)
                    boxes.append((f"{a} vs {b}", scores, _stim_hex_color(a)))
                return boxes

            # Extremes pairs
            extreme_pairs = [(unique_stimuli[i], unique_stimuli[-(i+1)]) for i in range(len(unique_stimuli)//2)]
            boxes_ext = overall_scores_for_pairs(extreme_pairs)
            if boxes_ext:
                fig_box_ext = go.Figure()
                for name, scores, col in boxes_ext:
                    fig_box_ext.add_trace(go.Box(y=scores, name=name, boxpoints='all', jitter=0.3, pointpos=0,
                                                  marker_color=col, line=dict(color=col)))
                fig_box_ext.add_hline(y=0.5, line_dash="dash", line_color="gray")
                fig_box_ext.update_layout(title="Overall decoding (time-averaged): extreme stimulus pairs", yaxis_title="Accuracy", yaxis=dict(range=[0,1]))
                st.plotly_chart(fig_box_ext, use_container_width=True)

            # Adjacent pairs
            adj_pairs = [(unique_stimuli[i], unique_stimuli[i+1]) for i in range(len(unique_stimuli)-1)]
            boxes_adj = overall_scores_for_pairs(adj_pairs)
            if boxes_adj:
                fig_box_adj = go.Figure()
                for name, scores, col in boxes_adj:
                    fig_box_adj.add_trace(go.Box(y=scores, name=name, boxpoints='all', jitter=0.3, pointpos=0,
                                                 marker_color=col, line=dict(color=col)))
                fig_box_adj.add_hline(y=0.5, line_dash="dash", line_color="gray")
                fig_box_adj.update_layout(title="Overall decoding (time-averaged): adjacent stimulus pairs", yaxis_title="Accuracy", yaxis=dict(range=[0,1]))
                st.plotly_chart(fig_box_adj, use_container_width=True)
    
    with outcome_tab:
        if 'outcome' not in stimuli_outcome_df.columns:
            st.warning("'outcome' column not found in data.")
            return
        # default to two most frequent outcomes for stability
        counts = stimuli_outcome_df['outcome'].value_counts()
        outcomes_order = list(counts.index)
        if len(outcomes_order) < 2:
            st.warning("Not enough unique outcomes to decode.")
            return
        selected = st.multiselect(
            "Select two outcomes:", options=outcomes_order, default=outcomes_order[:2], max_selections=2
        )
        if len(selected) != 2:
            st.info("Select exactly two outcomes.")
            return
        out_pos, out_neg = selected[0], selected[1]

        trial_labels = stimuli_outcome_df['outcome'].values
        sel_idx, y = _balanced_indices(trial_labels, out_pos, out_neg)
        if sel_idx is None:
            st.warning("Insufficient trials after balancing.")
            return

        t_range = st.slider(
            "Time range (s)", float(time_axis[0]), float(time_axis[-1]), (-2.0, 4.0), step=0.05, key=f"{key_prefix}_out_range"
        )
        avg_win = st.number_input("Averaging window (s)", 0.0, 2.0, 0.0, 0.05, key=f"{key_prefix}_out_avg")
        times, acc = _time_resolved_decode(event_windows_matrix, sel_idx, y, t_start=t_range[0], t_end=t_range[1], avg_win_sec=avg_win)
        if times is None:
            st.warning("Decoding failed.")
            return
        # Smooth for display
        def _smooth(a, k=5):
            if a is None or len(a) < 3 or k <= 1:
                return a
            k = min(k, max(1, len(a)//5))
            w = np.ones(k)/k
            return np.convolve(a, w, mode='same')
        acc_sm = _smooth(acc, 5)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=times, y=acc, mode='lines', name='Accuracy', line=dict(color='rgba(31,119,180,0.4)')))
        fig.add_trace(go.Scatter(x=times, y=acc_sm, mode='lines', name='Smoothed', line=dict(color='rgb(31,119,180)', width=3)))
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray")
        fig.add_shape(
            type="line",
            x0=0, x1=2,
            y0=0, y1=0,
            line=dict(color="black", width=5, dash="solid"),
            xref="x", yref="y"
        )
        fig.update_layout(
            title=f"Time-resolved decoding: {out_pos} vs {out_neg}",
            xaxis=dict(title="Time (s)", range=[t_range[0], t_range[1]]),
            yaxis_title="Accuracy",
            yaxis=dict(range=[0,1])
        )
        st.plotly_chart(fig, use_container_width=True)

    with action_tab:
        if 'outcome' not in stimuli_outcome_df.columns:
            st.warning("'outcome' column not found in data.")
            return
        lick_outcomes = ['Hit', 'False Alarm']
        no_lick_outcomes = ['Miss', 'CR']
        trial_outcomes = stimuli_outcome_df['outcome'].values
        action_labels = np.array([1 if o in lick_outcomes else 0 for o in trial_outcomes])
        sel_idx, y = _balanced_indices(action_labels, 1, 0)
        if sel_idx is None:
            st.warning("Insufficient trials for action decoding.")
            return

        t_range = st.slider(
            "Time range (s)", float(time_axis[0]), float(time_axis[-1]), (-2.0, 4.0), step=0.05, key=f"{key_prefix}_act_range"
        )
        avg_win = st.number_input("Averaging window (s)", 0.0, 2.0, 0.0, 0.05, key=f"{key_prefix}_act_avg")
        times, acc = _time_resolved_decode(event_windows_matrix, sel_idx, y, t_start=t_range[0], t_end=t_range[1], avg_win_sec=avg_win)
        if times is None:
            st.warning("Decoding failed.")
            return
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=times, y=acc, mode='lines', name='Accuracy'))
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray")
        fig.add_shape(
            type="line",
            x0=0, x1=2,
            y0=0, y1=0,
            line=dict(color="black", width=5, dash="solid"),
            xref="x", yref="y"
        )
        fig.update_layout(
            title="Time-resolved decoding: Lick vs No-lick",
            xaxis=dict(title="Time (s)", range=[t_range[0], t_range[1]]),
            yaxis_title="Accuracy",
            yaxis=dict(range=[0,1])
        )
        st.plotly_chart(fig, use_container_width=True)



