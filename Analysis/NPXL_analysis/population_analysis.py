import plotly.graph_objects as go
from Analysis.GNG_bpod_analysis.colors import COLOR_HIT, COLOR_MISS, COLOR_FA, COLOR_CR
import streamlit as st

def plot_population_heatmap(spike_matrix, stimuli_outcome_df, window_size):
    max_start = max(0, spike_matrix.shape[1] - window_size)
    start_bin = st.slider("Start time bin", 0, max_start, window_size)
    end_bin = start_bin + window_size
    matrix_subset = spike_matrix[:, start_bin:end_bin]
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix_subset,
            colorscale='Viridis',
            colorbar=dict(title="Spikes/sec"),
        )
    )
    if spike_matrix.shape[1] > 1:
        total_time = stimuli_outcome_df['time'].max()
        bin_size = total_time / (spike_matrix.shape[1] - 1)
    else:
        bin_size = 1
    outcome_color_map = {
        "Hit": COLOR_HIT,
        "Miss": COLOR_MISS,
        "False Alarm": COLOR_FA,
        "CR": COLOR_CR,
    }
    if 'time' in stimuli_outcome_df.columns and 'outcome' in stimuli_outcome_df.columns:
        for t, outcome in zip(stimuli_outcome_df['time'], stimuli_outcome_df['outcome']):
            color = outcome_color_map.get(outcome, "white")
            bin_idx = int(t / bin_size)
            if start_bin <= bin_idx < end_bin:
                fig.add_vline(x=bin_idx - start_bin, line_width=3, line_dash="dash", line_color=color)
    fig.update_layout(
        xaxis=dict(
            title="Time bin",
            rangeslider=dict(visible=True),
            constrain='domain'
        ),
        yaxis_title="Unit",
        title="Spike Matrix (scrollable, with event lines)"
    )
    fig.update_layout(height=800)
    st.plotly_chart(fig, use_container_width=True) 