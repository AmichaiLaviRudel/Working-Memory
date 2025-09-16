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


def advanced_population_analysis_panel(event_windows_matrix, stimuli_outcome_df, metadata):
    """
    Advanced population analysis panel - placeholder for future implementation.
    
    Args:
        event_windows_matrix: 3D array [units × time_bins × trials]
        stimuli_outcome_df: DataFrame with trial information
        metadata: Dictionary with recording parameters
    """
    st.header("Advanced Population Analysis 🧠")
    st.info("Advanced population analysis will be implemented step by step. Coming soon!")

