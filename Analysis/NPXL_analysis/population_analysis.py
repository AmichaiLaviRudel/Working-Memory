import plotly.graph_objects as go
from Analysis.GNG_bpod_analysis.colors import COLOR_HIT, COLOR_MISS, COLOR_FA, COLOR_CR
import streamlit as st
import numpy as np

def plot_population_heatmap(spike_matrix, stimuli_outcome_df, window_size):
    max_start = max(0, spike_matrix.shape[1] - window_size)
    max_start = int(max_start)
    window_size = int(window_size)

    step_value = window_size - int(0.3*window_size)  # Ensure step is at least 1

    
    start_bin = st.slider("Start time bin", 0, max_start, min(100, max_start), step=step_value)
    start_bin = int(start_bin)
    end_bin = start_bin + window_size
    matrix_subset = spike_matrix[:, start_bin:end_bin]

    # Calculate time axis for the displayed window
    if spike_matrix.shape[1] > 1:
        total_time = stimuli_outcome_df['time'].max()
        if total_time > 0:
            bin_size = total_time / (spike_matrix.shape[1] - 1)
        else:
            bin_size = 1  # Default bin size if total_time is 0
    else:
        bin_size = 1
    time_axis = np.arange(start_bin, end_bin) * bin_size
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix_subset,
            x=time_axis,
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
    
    if 'time' in stimuli_outcome_df.columns and 'outcome' in stimuli_outcome_df.columns:
        for t, outcome in zip(stimuli_outcome_df['time'], stimuli_outcome_df['outcome']):
            color = outcome_color_map.get(outcome, "white")
            # Check for division by zero
            if bin_size > 0:
                bin_idx = int(t / bin_size)
            else:
                bin_idx = 0  # Default to first bin if bin_size is 0
            if start_bin <= bin_idx < end_bin:
                # Show legend only once per outcome type
                showlegend = outcome not in legend_added
                if showlegend:
                    legend_added.add(outcome)
                
                # Ensure the index is within bounds
                time_idx = bin_idx - start_bin
                if 0 <= time_idx < len(time_axis):
                    fig.add_vline(
                        x=time_axis[time_idx], 
                        line_width=3, 
                        line_dash="dash", 
                        line_color=color,
                        name=outcome,
                        showlegend=showlegend
                    )
    
    fig.update_layout(
        xaxis=dict(
            title="Time (s)",
            rangeslider=dict(visible=True),
            constrain='domain'
        ),
        yaxis_title="Unit",
        title="Spike Matrix (scrollable, with event lines)",
        legend=dict(
            title="Tone Onset by Outcome",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    fig.update_layout(height=800)
    st.plotly_chart(fig, use_container_width=True) 