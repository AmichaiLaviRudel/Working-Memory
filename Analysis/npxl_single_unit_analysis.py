import plotly.graph_objects as go
import numpy as np
from Analysis.GNG_bpod_analysis.stats_tests import mannwhitneyu

def plot_unit_psth(spike_matrix, event_times, unit_idx, bin_size=0.01, window=(-1, 2)):
    """
    Plots PSTH for a single unit aligned to event_times, adds a vertical line at x=0, and calculates significance (Mann-Whitney U test) between pre- and post-event activity.
    Args:
        spike_matrix: 2D array [units x time_bins] (firing rate)
        event_times: 1D array of event times (in seconds)
        unit_idx: int, index of the unit to plot
        bin_size: float, bin size in seconds for PSTH
        window: tuple, time window around event (start, end) in seconds
    """
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
        peri_event_hist += spike_matrix[unit_idx, start_bin:end_bin]
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