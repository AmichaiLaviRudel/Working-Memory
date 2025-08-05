import plotly.graph_objects as go
from Analysis.GNG_bpod_analysis.colors import COLOR_HIT, COLOR_MISS, COLOR_FA, COLOR_CR
import streamlit as st
import numpy as np

def plot_population_heatmap(spike_matrix, stimuli_outcome_df, metadata):
    # Handle both 2D and 3D matrices
    if len(spike_matrix.shape) == 3:
        # 3D event windows matrix: [units × time × events]
        # Average across events to get 2D matrix
        spike_matrix = np.mean(spike_matrix, axis=2)
    
    # Get display window from metadata
    display_window = metadata.get('display_window', '(-1, 2)')
    try:
        # Parse display window string like "(-1, 2)" to get start and end times
        display_window = eval(display_window)  # Safe for tuple parsing
        start_time, end_time = display_window
    except:
        # Default window if parsing fails
        start_time, end_time = -1, 2
    
    # Create time axis for the full event window
    n_time_bins = spike_matrix.shape[1]
    time_axis = np.linspace(start_time, end_time, n_time_bins)
    
    # Use the full matrix for event-aligned data (no scrolling needed)
    matrix_subset = spike_matrix
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix_subset,
            x=time_axis,
            colorscale='Greys',
            colorbar=dict(title="Spikes/sec"),
        )
    )
    
    # Add vertical line at x=0 to mark event onset
    fig.add_vline(
        x=0, 
        line_width=2, 
        line_dash="solid", 
        line_color="red",
        name="Event Onset",
        showlegend=True
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
            
            # For event-aligned data, we want to mark the event onset (time 0)
            # Find the closest time point to 0 in our time axis
            event_time = 0  # Event onset is at time 0 in event-aligned data
            
            # Show legend only once per outcome type
            showlegend = outcome not in legend_added
            if showlegend:
                legend_added.add(outcome)
            
            fig.add_vline(
                x=event_time, 
                line_width=3, 
                line_dash="dash", 
                line_color=color,
                name=outcome,
                showlegend=showlegend
            )
    
    fig.update_layout(
        xaxis=dict(
            title="Time relative to event (s)",
            constrain='domain'
        ),
        yaxis_title="Unit",
        title="Population Activity Heatmap (Event-Aligned)",
        legend=dict(
            title="Event Onset by Outcome",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    fig.update_layout(height=800)
    st.plotly_chart(fig, use_container_width=True) 