import plotly.graph_objects as go
import numpy as np
from Analysis.GNG_bpod_analysis.stats_tests import mannwhitneyu

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
        title=f"PSTH for Unit {unit_idx} (Mann-Whitney U p={p:.3g})",
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


def single_unit_analysis_panel(
    spike_matrix, stimuli_outcome_df, bin_size=0.01, default_window=1.0):
    import streamlit as st
    import numpy as np
    from .npxl_single_unit_analysis import plot_unit_psth, compute_psth_pvalues
    from Analysis.NPXL_analysis.population_analysis import plot_population_heatmap



    event_times = stimuli_outcome_df['time'].values
    event_outcomes = stimuli_outcome_df['outcome'].values if 'outcome' in stimuli_outcome_df.columns else None
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

    # Compute p-values for all units and sort (filtered)
    pvals = compute_psth_pvalues(spike_matrix, event_times, bin_size=bin_size, window=display_window)
    sorted_indices = np.argsort(pvals)
    sorted_pvals = pvals[sorted_indices]

    # Use a slider to select the unit by sorted order
    unit_rank = st.slider("Unit (sorted by p-value, lowest first)", 0, len(sorted_indices) - 1, 0)
    unit_idx = sorted_indices[unit_rank]
    st.write(f"Unit index: {unit_idx}, p-value: {sorted_pvals[unit_rank]:.3g}")

    # Plot PSTH for the selected unit (filtered)
    psth_fig = plot_unit_psth(
        spike_matrix, event_times, unit_idx,
        bin_size=bin_size, window=display_window,
        event_outcomes=event_outcomes, stim_filter=stim_filter
    )
    st.plotly_chart(psth_fig, use_container_width=True)
