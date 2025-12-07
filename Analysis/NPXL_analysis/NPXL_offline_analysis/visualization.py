"""
Visualization functions for NPXL offline analysis.
"""
import os
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from Analysis.NPXL_analysis.single_unit_offline_analysis.config import (
    OUTCOME_COLOR_MAP,
    SUBJECT_COLORS,
    COLOR_ACCENT,
    COLOR_ACCENT_TRANSPARENT,
    COLOR_GRAY,
    COLOR_HIT,
    COLOR_FA,
    COLOR_CR,
    COLOR_MISS,
)
from Analysis.NPXL_analysis.single_unit_offline_analysis.utils import save_plot_to_html
from Analysis.GNG_bpod_analysis.colors import COLOR_LOW_BD, COLOR_HIGH_BD


def _hex_to_rgba(hex_color: str, alpha: float = 0.2) -> str:
    """
    Convert hex color to rgba string for transparency.
    """
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'


def plot_psth_by_stimulus(
    event_windows_data: tuple,
    unit_idx: int,
    display_window: tuple[float, float] = (-0.5, 1.0),
    region_name: str = "Unit",
) -> go.Figure:
    """
    Plot PSTH separated by stimulus type for a single unit.
    """
    # Handle both 5-tuple and 6-tuple formats
    if len(event_windows_data) == 6:
        event_windows_matrix, time_axis, valid_event_indices, stimuli_outcome_df, metadata, _ = event_windows_data
    else:
        event_windows_matrix, time_axis, valid_event_indices, stimuli_outcome_df, metadata = event_windows_data
    
    if 'stimulus' not in stimuli_outcome_df.columns:
        print("No stimulus information available")
        return go.Figure()
    
    # Get unit data
    unit_data = event_windows_matrix[unit_idx, :, :]  # [time × events]
    
    # Find time indices for display window
    start_idx = np.argmin(np.abs(time_axis - display_window[0]))
    end_idx = np.argmin(np.abs(time_axis - display_window[1]))
    unit_data_windowed = unit_data[start_idx:end_idx, :]
    time_axis_windowed = time_axis[start_idx:end_idx]
    
    # Get unique stimuli
    unique_stimuli = sorted(stimuli_outcome_df['stimulus'].unique())
    
    # Create figure
    fig = go.Figure()
    
    # Use SUBJECT_COLORS from colors.py (already in hex format)
    for stim_idx, stim in enumerate(unique_stimuli):
        stim_mask = (stimuli_outcome_df['stimulus'] == stim).values
        stim_trials = unit_data_windowed[:, stim_mask]
        
        if stim_trials.shape[1] > 0:
            psth_mean = np.mean(stim_trials, axis=1)
            psth_sem = np.std(stim_trials, axis=1) / np.sqrt(stim_trials.shape[1])
            
            # Get color from SUBJECT_COLORS palette
            color = SUBJECT_COLORS[stim_idx % len(SUBJECT_COLORS)]
            
            # Main trace
            fig.add_trace(go.Scatter(
                x=time_axis_windowed,
                y=psth_mean,
                mode='lines',
                name=f'Stim {stim} (n={stim_trials.shape[1]})',
                line=dict(color=color, width=2)
            ))
            
            # SEM shading using helper function
            fig.add_trace(go.Scatter(
                x=np.concatenate([time_axis_windowed, time_axis_windowed[::-1]]),
                y=np.concatenate([psth_mean + psth_sem, (psth_mean - psth_sem)[::-1]]),
                fill='toself',
                fillcolor=_hex_to_rgba(color, alpha=0.2),
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # Add vertical line at event onset
    fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)
    
    fig.update_layout(
        title=f"{region_name} Unit {unit_idx} - PSTH by Stimulus",
        xaxis_title="Time (s)",
        yaxis_title="Firing Rate (Hz)",
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig


def plot_psth_by_outcome(
    event_windows_data: tuple,
    unit_idx: int,
    display_window: tuple[float, float] = (-0.5, 1.0),
    region_name: str = "Unit",
) -> go.Figure:
    """
    Plot PSTH separated by behavioral outcome (Hit/Miss/FA/CR) for a single unit.
    """
    # Handle both 5-tuple and 6-tuple formats
    if len(event_windows_data) == 6:
        event_windows_matrix, time_axis, valid_event_indices, stimuli_outcome_df, metadata, _ = event_windows_data
    else:
        event_windows_matrix, time_axis, valid_event_indices, stimuli_outcome_df, metadata = event_windows_data
    
    if 'outcome' not in stimuli_outcome_df.columns:
        print("No outcome information available")
        return go.Figure()
    
    # Get unit data
    unit_data = event_windows_matrix[unit_idx, :, :]  # [time × events]
    
    # Find time indices for display window
    start_idx = np.argmin(np.abs(time_axis - display_window[0]))
    end_idx = np.argmin(np.abs(time_axis - display_window[1]))
    unit_data_windowed = unit_data[start_idx:end_idx, :]
    time_axis_windowed = time_axis[start_idx:end_idx]
    
    # Define outcomes - use OUTCOME_COLOR_MAP from colors.py
    outcomes = ['Hit', 'Miss', 'False Alarm', 'CR']
    
    # Create figure
    fig = go.Figure()
    
    for outcome in outcomes:
        outcome_mask = (stimuli_outcome_df['outcome'] == outcome).values
        outcome_trials = unit_data_windowed[:, outcome_mask]
        
        if outcome_trials.shape[1] > 0:
            psth_mean = np.mean(outcome_trials, axis=1)
            psth_sem = np.std(outcome_trials, axis=1) / np.sqrt(outcome_trials.shape[1])
            
            # Get color from OUTCOME_COLOR_MAP
            color = OUTCOME_COLOR_MAP.get(outcome, '#808080')  # Default gray if not found
            
            # Main trace
            fig.add_trace(go.Scatter(
                x=time_axis_windowed,
                y=psth_mean,
                mode='lines',
                name=f'{outcome} (n={outcome_trials.shape[1]})',
                line=dict(color=color, width=2)
            ))
            
            # SEM shading using helper function
            fig.add_trace(go.Scatter(
                x=np.concatenate([time_axis_windowed, time_axis_windowed[::-1]]),
                y=np.concatenate([psth_mean + psth_sem, (psth_mean - psth_sem)[::-1]]),
                fill='toself',
                fillcolor=_hex_to_rgba(color, alpha=0.2),
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # Add vertical line at event onset
    fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)
    
    fig.update_layout(
        title=f"{region_name} Unit {unit_idx} - PSTH by Outcome",
        xaxis_title="Time (s)",
        yaxis_title="Firing Rate (Hz)",
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig


def plot_raw_psth(
    event_windows_data: tuple,
    unit_idx: int,
    display_window: tuple[float, float] = (-0.5, 1.0),
    region_name: str = "Unit",
) -> go.Figure:
    """
    Plot raw PSTH (all trials averaged together) for a single unit.
    """
    # Handle both 5-tuple and 6-tuple formats
    if len(event_windows_data) == 6:
        event_windows_matrix, time_axis, valid_event_indices, stimuli_outcome_df, metadata, _ = event_windows_data
    else:
        event_windows_matrix, time_axis, valid_event_indices, stimuli_outcome_df, metadata = event_windows_data
    
    # Get unit data
    unit_data = event_windows_matrix[unit_idx, :, :]  # [time × events]
    
    # Find time indices for display window
    start_idx = np.argmin(np.abs(time_axis - display_window[0]))
    end_idx = np.argmin(np.abs(time_axis - display_window[1]))
    unit_data_windowed = unit_data[start_idx:end_idx, :]
    time_axis_windowed = time_axis[start_idx:end_idx]
    
    # Calculate PSTH (mean and SEM across all trials)
    psth_mean = np.mean(unit_data_windowed, axis=1)
    psth_sem = np.std(unit_data_windowed, axis=1) / np.sqrt(unit_data_windowed.shape[1])
    
    # Create figure
    fig = go.Figure()
    
    # Main PSTH trace
    fig.add_trace(go.Scatter(
        x=time_axis_windowed,
        y=psth_mean,
        mode='lines',
        name='Mean Firing Rate',
        line=dict(color=COLOR_ACCENT, width=2)
    ))
    
    # SEM shading
    fig.add_trace(go.Scatter(
        x=np.concatenate([time_axis_windowed, time_axis_windowed[::-1]]),
        y=np.concatenate([psth_mean + psth_sem, (psth_mean - psth_sem)[::-1]]),
        fill='toself',
        fillcolor=COLOR_ACCENT_TRANSPARENT,
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Add vertical line at event onset
    fig.add_vline(x=0, line_dash="dash", line_color=COLOR_GRAY, line_width=1)
    
    fig.update_layout(
        title=f"{region_name} Unit {unit_idx} - Raw PSTH (n={unit_data_windowed.shape[1]} trials)",
        xaxis_title="Time (s)",
        yaxis_title="Firing Rate (Hz)",
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig


def save_raw_psth_for_active_units(
    event_windows_data: tuple,
    active_units: np.ndarray,
    p_vals: np.ndarray,
    region_name: str,
    results_dir: str,
    display_window: tuple[float, float] = (-0.5, 1.0),
) -> None:
    """
    Save raw PSTH plots for all active units in a region, sorted by p-values (most significant first).
    
    Parameters:
    -----------
    event_windows_data : tuple
        Event windows data tuple
    active_units : np.ndarray
        Array of active unit indices
    p_vals : np.ndarray
        Array of p-values corresponding to active_units (for sorting)
    region_name : str
        Name of the region (e.g., "ACx", "OFC")
    results_dir : str
        Base results directory path
    display_window : tuple[float, float]
        Time window for display (start, end) in seconds
    """
    if len(active_units) == 0:
        print(f"\n=== No {region_name} active units found - skipping raw PSTH plots ===")
        return
    
    # Sort units by p-value (most significant first)
    sorted_indices = np.argsort(p_vals)
    sorted_units = active_units[sorted_indices]
    sorted_p_vals = p_vals[sorted_indices]
    
    # Create region-specific subfolder
    region_folder = region_name.lower()
    raw_psth_dir = os.path.join(results_dir, "plots", "raw_psth", region_folder)
    os.makedirs(raw_psth_dir, exist_ok=True)
    
    print(f"\n=== Saving raw PSTH plots for all {region_name} responsive units (sorted by p-value) ===")
    print(f"  Saving to: {raw_psth_dir}")
    for rank, (unit_idx, p_val) in enumerate(zip(sorted_units, sorted_p_vals), start=1):
        fig_raw_psth = plot_raw_psth(
            event_windows_data,
            int(unit_idx),
            display_window=display_window,
            region_name=region_name
        )
        # Include rank in filename for easy identification
        # Save in region-specific subfolder
        save_plot_to_html(
            fig_raw_psth,
            os.path.join(raw_psth_dir, f"unit_{unit_idx}_rank{rank:03d}_p{p_val:.4f}_raw_psth.html"),
            f"{region_name} Unit {unit_idx} (Rank {rank}, p={p_val:.4f}) Raw PSTH"
        )
    print(f"  Saved {len(active_units)} {region_name} raw PSTH plots (sorted by significance) to {raw_psth_dir}")


def plot_selectivity_summary(
    selectivity_df: pd.DataFrame,
    region_name: str = "Region",
) -> tuple[go.Figure, go.Figure]:
    """
    Create summary plots showing selectivity metrics for all active units.
    Returns two figures: (metrics_plot, classification_plot)
    """
    # Figure 1: Scatter plot of selectivity metrics
    fig_metrics = go.Figure()

    # If there are no active units or required columns are missing, return placeholder figures
    required_cols = [
        "unit_idx",
        "outcome_p_value",
        "go_nogo_dprime",
        "choice_probability_corr",
        "stimulus_selective",
        "outcome_modulated",
        "go_nogo_selective",
        "choice_coding",
    ]
    if (
        selectivity_df is None
        or len(selectivity_df) == 0
        or not all(col in selectivity_df.columns for col in required_cols)
    ):
        # Explicit placeholders avoid KeyError when there are no active units
        fig_metrics.update_layout(
            title=f"{region_name} - No selectivity metrics (no active units)",
            template="plotly_white",
            height=400,
        )
        fig_class = go.Figure()
        fig_class.update_layout(
            title=f"{region_name} - No unit classification (no active units)",
            template="plotly_white",
            height=300,
        )
        return fig_metrics, fig_class
    
    # Outcome modulation (p-value)
    outcome_p_vals = selectivity_df['outcome_p_value'].values
    valid_outcome = ~np.isnan(outcome_p_vals)
    
    if np.any(valid_outcome):
        fig_metrics.add_trace(go.Scatter(
            x=selectivity_df.loc[valid_outcome, 'unit_idx'],
            y=-np.log10(outcome_p_vals[valid_outcome] + 1e-10),  # -log10(p) with small offset
            mode='markers',
            name='Outcome Modulation (-log10 p)',
            marker=dict(color='blue', size=8, symbol='circle'),
            hovertemplate='Unit %{x}<br>-log10(p) = %{y:.2f}<extra></extra>'
        ))
    
    # Go/NoGo d'
    go_nogo_dprime = selectivity_df['go_nogo_dprime'].values
    valid_go_nogo = ~np.isnan(go_nogo_dprime)
    
    if np.any(valid_go_nogo):
        fig_metrics.add_trace(go.Scatter(
            x=selectivity_df.loc[valid_go_nogo, 'unit_idx'],
            y=go_nogo_dprime[valid_go_nogo],
            mode='markers',
            name="Go/NoGo d'",
            marker=dict(color='green', size=8, symbol='square'),
            yaxis='y2',
            hovertemplate='Unit %{x}<br>d\' = %{y:.2f}<extra></extra>'
        ))
    
    # Choice probability correlation
    cp_corr = selectivity_df['choice_probability_corr'].values
    valid_cp = ~np.isnan(cp_corr)
    
    if np.any(valid_cp):
        fig_metrics.add_trace(go.Scatter(
            x=selectivity_df.loc[valid_cp, 'unit_idx'],
            y=cp_corr[valid_cp],
            mode='markers',
            name='Choice Probability (corr)',
            marker=dict(color='red', size=8, symbol='diamond'),
            yaxis='y3',
            hovertemplate='Unit %{x}<br>CP corr = %{y:.2f}<extra></extra>'
        ))
    
    # Add significance thresholds
    fig_metrics.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="blue", 
                         annotation_text="p=0.05", annotation_position="right")
    
    fig_metrics.update_layout(
        title=f"{region_name} - Selectivity Metrics Summary",
        xaxis_title="Unit Index",
        yaxis=dict(title="-log10(Outcome p-value)", side="left"),
        yaxis2=dict(title="Go/NoGo d'", side="right", overlaying="y"),
        yaxis3=dict(title="Choice Probability (corr)", side="right", overlaying="y", position=0.95),
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    # Figure 2: Classification summary (bar chart)
    categories = {
        'Stimulus Selective': selectivity_df['stimulus_selective'].sum(),
        'Outcome Modulated': selectivity_df['outcome_modulated'].sum(),
        'Go/NoGo Selective': selectivity_df['go_nogo_selective'].sum(),
        'Choice Coding': selectivity_df['choice_coding'].sum(),
    }
    
    # Count units in each category combination
    both_stim_outcome = ((selectivity_df['stimulus_selective']) & 
                         (selectivity_df['outcome_modulated'])).sum()
    all_selective = ((selectivity_df['stimulus_selective']) | 
                     (selectivity_df['outcome_modulated']) | 
                     (selectivity_df['go_nogo_selective']) | 
                     (selectivity_df['choice_coding'])).sum()
    
    fig_class = go.Figure()
    
    fig_class.add_trace(go.Bar(
        x=list(categories.keys()),
        y=list(categories.values()),
        marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
        text=list(categories.values()),
        textposition='outside',
        hovertemplate='%{x}<br>Count: %{y}<extra></extra>'
    ))
    
    fig_class.update_layout(
        title=f"{region_name} - Unit Classification Summary (n={len(selectivity_df)} active units)",
        xaxis_title="Selectivity Type",
        yaxis_title="Number of Units",
        template='plotly_white',
        height=400
    )
    
    # Add annotation for combined categories
    fig_class.add_annotation(
        x=0.5, y=0.95,
        xref='paper', yref='paper',
        text=f"Units with any selectivity: {all_selective}<br>Units with both stimulus & outcome: {both_stim_outcome}",
        showarrow=False,
        font=dict(size=12),
        bgcolor='rgba(255,255,255,0.8)',
        bordercolor='black',
        borderwidth=1
    )
    
    return fig_metrics, fig_class


def plot_unit_heatmap(
    event_windows_data: tuple,
    unit_idx: int,
    display_window: tuple[float, float] = (-0.5, 1.0),
    region_name: str = "Unit",
) -> go.Figure:
    """
    Create a heatmap visualization for a single unit from event windows data.
    
    Parameters:
    -----------
    event_windows_data : tuple
        Event windows data tuple (5 or 6 elements)
    unit_idx : int
        Index of the unit to plot
    display_window : tuple[float, float]
        Time window for display (start, end) in seconds
    region_name : str
        Name of the brain region
    
    Returns:
    --------
    go.Figure
        Plotly figure with heatmap visualization
    """
    # Handle both 5-tuple and 6-tuple formats
    if len(event_windows_data) == 6:
        event_windows_matrix, time_axis, valid_event_indices, stimuli_outcome_df, metadata, _ = event_windows_data
    else:
        event_windows_matrix, time_axis, valid_event_indices, stimuli_outcome_df, metadata = event_windows_data
    
    # Create a new time axis that matches the current display_window
    num_time_bins = event_windows_matrix.shape[1]
    new_time_axis = np.linspace(display_window[0], display_window[1], num_time_bins)
    
    # Get the unit's data
    unit_data = event_windows_matrix[unit_idx, :, :]  # Shape: [time × events]
    
    # Get outcomes for the valid events
    event_outcomes_for_raster = None
    if 'outcome' in stimuli_outcome_df.columns:
        event_outcomes_for_raster = stimuli_outcome_df['outcome'].values
    
    if event_outcomes_for_raster is None:
        # If no outcome data, just plot all trials
        ordered_data = unit_data.T
        trial_types_ordered = ["All"] * ordered_data.shape[0]
    else:
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
        
        # Build a list of trial types in the order of the rows in ordered_data
        trial_types_ordered = (
            ["CR"] * correct_rejection_data.shape[1] +
            ["False Alarm"] * false_alarm_data.shape[1] +
            ["Miss"] * miss_data.shape[1] +
            ["Hit"] * hit_data.shape[1]
        )
    
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
        "Hit": rgba_from_hex(COLOR_HIT, 0.3),
        "All": "rgba(200,200,200,0.2)"
    }
    
    # Create the heatmap
    fig = go.Figure()
    
    fig.add_trace(go.Heatmap(
        z=ordered_data,
        x=new_time_axis,
        y=np.arange(ordered_data.shape[0]),
        colorbar=dict(title="Firing Rate (Hz)", len=0.8),
        colorscale='Viridis'
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
    
    # Add legend for trial types (only if we have outcome data)
    if event_outcomes_for_raster is not None:
        legend_items = []
        for trial_type, color in trial_type_colors.items():
            if trial_type != "All":
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
        
        for item in legend_items:
            fig.add_trace(item)
    
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
        title=f"{region_name} Unit {unit_idx} - Trial Heatmap",
        xaxis_title="Time (s)",
        yaxis_title="Trial",
        xaxis=dict(constrain='domain'),
        legend=dict(
            orientation='h',
            x=0.5,
            y=1.1,
            xanchor='center',
            yanchor='top'
        ) if event_outcomes_for_raster is not None else None,
        margin=dict(r=80, t=100),
        template='plotly_white',
        height=600
    )
    
    return fig


def get_trial_statistics(
    event_windows_data: tuple,
    unit_idx: int,
) -> dict:
    """
    Get trial statistics for a specific unit from event windows data.
    
    Parameters:
    -----------
    event_windows_data : tuple
        Event windows data tuple
    unit_idx : int
        Index of the unit to analyze
    
    Returns:
    --------
    dict
        Dictionary containing trial counts for each outcome type
    """
    # Handle both 5-tuple and 6-tuple formats
    if len(event_windows_data) == 6:
        event_windows_matrix, time_axis, valid_event_indices, stimuli_outcome_df, metadata, _ = event_windows_data
    else:
        event_windows_matrix, time_axis, valid_event_indices, stimuli_outcome_df, metadata = event_windows_data
    
    # Get the unit's data
    unit_data = event_windows_matrix[unit_idx, :, :]  # Shape: [time × events]
    
    # Get outcomes for the valid events
    event_outcomes_for_raster = None
    if 'outcome' in stimuli_outcome_df.columns:
        event_outcomes_for_raster = stimuli_outcome_df['outcome'].values
    
    if event_outcomes_for_raster is None:
        return {
            'CR': 0,
            'FA': 0,
            'Miss': 0,
            'Hit': 0,
            'Total': unit_data.shape[1]
        }
    
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
        'Hit': hit_data.shape[1],
        'Total': unit_data.shape[1]
    }


def plot_tuning_curve(
    tuning_curve_stimuli: list | str,
    tuning_curve: list | str,
    tuning_curve_sem: list | str | None = None,
    unit_idx: int | None = None,
    region_name: str = "Unit",
    use_log_scale: bool = True,
) -> go.Figure:
    """
    Plot tuning curve for a single unit.
    
    Parameters:
    -----------
    tuning_curve_stimuli : list or str
        List of stimulus values (frequencies) or JSON string representation
    tuning_curve : list or str
        List of firing rates or JSON string representation
    tuning_curve_sem : list or str or None, optional
        List of SEM values or JSON string representation. If None, no error band is shown.
    unit_idx : int or None, optional
        Unit index for title
    region_name : str, optional
        Region name for title (default: "Unit")
    use_log_scale : bool, optional
        If True, use log scale for x-axis (default: True)
    
    Returns:
    --------
    go.Figure
        Plotly figure with tuning curve
    """
    # Import streamlit to access session state
    try:
        import streamlit as st
        # Get boundaries from session state if available
        low_boundary = None
        high_boundary = None
        if hasattr(st, 'session_state'):
            if hasattr(st.session_state, "low_boundary"):
                low_boundary = st.session_state.low_boundary
            if hasattr(st.session_state, "high_boundary"):
                high_boundary = st.session_state.high_boundary
    except ImportError:
        # If streamlit is not available, boundaries will remain None
        low_boundary = None
        high_boundary = None
    # Parse JSON strings if needed
    def _parse_value(val):
        """Parse value from JSON string, Python list string, or return as-is."""
        if isinstance(val, str):
            try:
                return json.loads(val)
            except json.JSONDecodeError:
                try:
                    return eval(val)
                except Exception:
                    return val
        return val
    
    tuning_frq = _parse_value(tuning_curve_stimuli)
    tuning_fr = _parse_value(tuning_curve)
    tuning_sem = _parse_value(tuning_curve_sem) if tuning_curve_sem is not None else None
    
    # Validate data
    if not isinstance(tuning_frq, (list, tuple, np.ndarray)) or not isinstance(tuning_fr, (list, tuple, np.ndarray)):
        raise ValueError("tuning_curve_stimuli and tuning_curve must be lists or arrays")
    
    if len(tuning_frq) != len(tuning_fr):
        raise ValueError("tuning_curve_stimuli and tuning_curve must have the same length")
    
    # Create figure
    fig = go.Figure()
    
    # Main tuning curve trace
    fig.add_trace(go.Scatter(
        x=tuning_frq,
        y=tuning_fr,
        mode='lines+markers',
        name='Tuning Curve',
        line=dict(color=COLOR_ACCENT, width=2),
        marker=dict(size=8, color=COLOR_ACCENT)
    ))
    
    # Add SEM as filled error band, if available and valid length
    if (tuning_sem is not None 
        and isinstance(tuning_sem, (list, tuple, np.ndarray)) 
        and len(tuning_sem) == len(tuning_frq)):
        # Convert to numpy arrays for easier computation
        tuning_fr_arr = np.array(tuning_fr)
        tuning_sem_arr = np.array(tuning_sem)
        
        # Upper SEM boundary
        fig.add_trace(go.Scatter(
            x=tuning_frq,
            y=tuning_fr_arr + tuning_sem_arr,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip',
            name='upper_sem',
            marker=dict(color='rgba(0,0,0,0)'),
        ))
        # Lower SEM boundary with fill
        fig.add_trace(go.Scatter(
            x=tuning_frq,
            y=tuning_fr_arr - tuning_sem_arr,
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor=COLOR_ACCENT_TRANSPARENT,
            showlegend=True,
            name='SEM',
            hoverinfo='skip',
            marker=dict(color='rgba(0,0,0,0)'),
        ))
    
    # Add vertical lines at boundaries if provided
    if low_boundary is not None:
        fig.add_vline(
            x=low_boundary,
            line_dash="dash",
            line_color=COLOR_LOW_BD,
            line_width=2,

        )
    
    if high_boundary is not None:
        fig.add_vline(
            x=high_boundary,
            line_dash="dash",
            line_color=COLOR_HIGH_BD,
            line_width=2,

        )
    
    # Update layout
    title = f"{region_name} Unit {unit_idx} - Tuning Curve" if unit_idx is not None else f"{region_name} - Tuning Curve"
    
    # Configure x-axis based on scale type
    if use_log_scale:
        # Generate appropriate tick values for log scale
        # Convert to numpy array for easier manipulation
        tuning_frq_arr = np.array(tuning_frq)
        min_freq = np.min(tuning_frq_arr)
        max_freq = np.max(tuning_frq_arr)
        
        # Generate comprehensive tick values for log scale
        # Include standard round numbers that work well on log scale
        all_ticks = []
        
        # Add ticks below 1.0: 0.5, 0.6, 0.7, 0.8, 0.9
        for t in [0.5, 0.6, 0.7, 0.8, 0.9]:
            if min_freq <= t <= max_freq:
                all_ticks.append(t)
        
        # Add ticks at and above 1.0: 1.0, 1.2, 1.5, 2.0, 2.5, 3.0
        for t in [1.0, 1.2, 1.5, 2.0, 2.5, 3.0]:
            if min_freq <= t <= max_freq:
                all_ticks.append(t)
        
        # Add boundaries if they exist and are in range (with small tolerance for floating point)
        tolerance = 0.01
        if low_boundary is not None:
            if min_freq - tolerance <= low_boundary <= max_freq + tolerance:
                # Check if close to existing tick
                if not any(abs(t - low_boundary) < tolerance for t in all_ticks):
                    all_ticks.append(low_boundary)
        if high_boundary is not None:
            if min_freq - tolerance <= high_boundary <= max_freq + tolerance:
                # Check if close to existing tick
                if not any(abs(t - high_boundary) < tolerance for t in all_ticks):
                    all_ticks.append(high_boundary)
        
        # Sort and remove duplicates (within tolerance)
        all_ticks = sorted(set(all_ticks))
        
        # Format tick labels - show appropriate decimal places
        ticktext = []
        for t in all_ticks:
            if t < 1.0:
                # For values < 1, show 2 decimal places
                ticktext.append(f"{t:.2f}")
            elif t == int(t):
                # For whole numbers, show as integer
                ticktext.append(f"{int(t)}")
            else:
                # For values >= 1 with decimals, show 1 decimal place
                ticktext.append(f"{t:.1f}")
        
        fig.update_xaxes(
            type='log',
            title="Stimulus Frequency (kHz)",
            tickmode='array',
            tickvals=all_ticks,
            ticktext=ticktext,
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)'
        )
    else:
        fig.update_xaxes(
            title="Stimulus Frequency (kHz)",
            tickmode='auto',
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)'
        )
    
    fig.update_layout(
        title=title,
        yaxis_title="Firing Rate (Hz)",
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig

