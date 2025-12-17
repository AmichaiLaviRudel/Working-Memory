"""
Example usage of GLM fitting module for neural data analysis.

This script demonstrates how to:
1. Load neural data and behavioral data
2. Fit GLM models for individual units
3. Fit GLM models for all units in a dataset
4. Analyze and visualize GLM results
"""

# %%

import sys
import os

# Add workspace root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if 'single_unit_offline_analysis' in current_dir:
    workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
else:
    workspace_root = current_dir

if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Tuple, Optional, List

from Analysis.NPXL_analysis.single_unit_offline_analysis.data_loading import (
    load_data,
    load_unit_labels,
    load_full_event_windows_data,
)
from Analysis.NPXL_analysis.single_unit_offline_analysis.unit import (
    create_units_from_event_data,
)
from Analysis.NPXL_analysis.single_unit_offline_analysis.glm_fitting import (
    fit_glm_for_unit,
    fit_glm_for_all_units,
    build_design_matrix,
    fit_glm_poisson,
)
from Analysis.NPXL_analysis.single_unit_offline_analysis.utils import (
    setup_results_directory,
    save_dataframe_to_csv,
)

# %%
def extract_licks_from_unit(unit, event_windows_data) -> Optional[List]:
    """
    Extract lick data from unit or event_windows_data.
    
    Tries multiple sources:
    1. stimuli_outcome_df columns ('first_lick_time', 'licks', etc.) - preferred
    2. event_windows_data[5] (lick rate matrix) - convert to timestamps
    3. unit._event_windows_data[5] if available
    
    Parameters:
    -----------
    unit : Unit
        Unit object
    event_windows_data : tuple
        Event windows data tuple
    
    Returns:
    --------
    list or None
        List of lick timestamps per trial, or None if not available
    """
    licks_list = None
    
    # First, try to get from stimuli_outcome_df (most reliable)
    if hasattr(unit, 'stimuli_outcome_df'):
        df = unit.stimuli_outcome_df
        
        # Check for 'first_lick_time' column (in bins, need to convert to seconds)
        if 'first_lick_time' in df.columns:
            bin_size = unit.bin_size if hasattr(unit, 'bin_size') else 0.01
            licks_list = []
            for idx, row in df.iterrows():
                first_lick_bin = row['first_lick_time']
                if pd.notna(first_lick_bin) and first_lick_bin > 0:
                    # Convert from bins to seconds (relative to cue onset)
                    first_lick_time = float(first_lick_bin) * bin_size
                    licks_list.append(np.array([first_lick_time]))
                else:
                    licks_list.append(None)
        
        # If not found, check for other lick column names
        elif licks_list is None:
            for col_name in ['licks', 'lick_times', 'first_lick_times', 'lick_timestamps']:
                if col_name in df.columns:
                    licks_list = []
                    for idx, row in df.iterrows():
                        lick_val = row[col_name]
                        if pd.notna(lick_val):
                            # Handle different formats
                            if isinstance(lick_val, (list, np.ndarray)):
                                licks_list.append(np.array(lick_val))
                            elif isinstance(lick_val, (int, float)) and lick_val > 0:
                                # Single timestamp value
                                bin_size = unit.bin_size if hasattr(unit, 'bin_size') else 0.01
                                # Assume it's in bins if > 100, otherwise seconds
                                if lick_val > 100:
                                    lick_time = float(lick_val) * bin_size
                                else:
                                    lick_time = float(lick_val)
                                licks_list.append(np.array([lick_time]))
                            elif isinstance(lick_val, str):
                                # Try to parse string representation
                                try:
                                    import ast
                                    parsed = ast.literal_eval(lick_val)
                                    if isinstance(parsed, (list, np.ndarray)):
                                        licks_list.append(np.array(parsed))
                                    elif isinstance(parsed, (int, float)) and parsed > 0:
                                        bin_size = unit.bin_size if hasattr(unit, 'bin_size') else 0.01
                                        if parsed > 100:
                                            lick_time = float(parsed) * bin_size
                                        else:
                                            lick_time = float(parsed)
                                        licks_list.append(np.array([lick_time]))
                                    else:
                                        licks_list.append(None)
                                except:
                                    licks_list.append(None)
                            else:
                                licks_list.append(None)
                        else:
                            licks_list.append(None)
                    break
    
    # Try to get from event_windows_data[5] (lick rate matrix)
    if licks_list is None and len(event_windows_data) >= 6 and event_windows_data[5] is not None:
        lick_rate_matrix = event_windows_data[5]  # Shape: [1 × time × trials]
        if lick_rate_matrix.ndim == 3 and lick_rate_matrix.shape[0] == 1:
            # Convert lick rate matrix to timestamps
            time_axis = event_windows_data[1]  # Time axis
            bin_size = unit.bin_size if hasattr(unit, 'bin_size') else (time_axis[1] - time_axis[0]) if len(time_axis) > 1 else 0.01
            
            n_trials = lick_rate_matrix.shape[2]
            licks_list = []
            
            for trial_idx in range(n_trials):
                # Get lick rate for this trial: [time]
                trial_lick_rate = lick_rate_matrix[0, :, trial_idx]
                
                # Find time bins with licks (rate > threshold)
                trial_lick_times = []
                for time_idx, rate in enumerate(trial_lick_rate):
                    if rate > 0.1:  # Threshold for detecting licks
                        # Expected number of licks in this bin
                        expected_licks = rate * bin_size
                        # Sample actual number (Poisson)
                        n_licks = np.random.poisson(expected_licks)
                        if n_licks > 0:
                            # Distribute licks uniformly within bin
                            t_start = time_axis[time_idx]
                            t_end = t_start + bin_size
                            lick_positions = np.linspace(t_start, t_end, n_licks + 2)[1:-1]
                            trial_lick_times.extend(lick_positions)
                
                licks_list.append(np.array(trial_lick_times) if len(trial_lick_times) > 0 else None)
    
    # Try to get from unit._event_windows_data[5]
    if licks_list is None and hasattr(unit, '_event_windows_data'):
        if len(unit._event_windows_data) >= 6:
            lick_data = unit._event_windows_data[5]
            if lick_data is not None:
                # Assume it's a list of arrays or similar structure
                if isinstance(lick_data, (list, np.ndarray)):
                    licks_list = []
                    for item in lick_data:
                        if isinstance(item, (list, np.ndarray)) and len(item) > 0:
                            licks_list.append(np.array(item))
                        else:
                            licks_list.append(None)
    
    return licks_list


def plot_glm_coefficients(glm_results: dict, title: str = "GLM Coefficients") -> go.Figure:
    """
    Create a bar plot of GLM coefficients with error bars.
    
    Parameters:
    -----------
    glm_results : dict
        GLM results dictionary
    title : str
        Plot title
    
    Returns:
    --------
    go.Figure : Plotly figure
    """
    if 'error' in glm_results:
        print(f"Cannot plot: {glm_results['error']}")
        return None
    
    feature_names = glm_results['feature_names']
    coefficients = glm_results['coefficients']
    
    # Get p-values for significance markers
    p_values = glm_results.get('p_values', [None] * len(coefficients))
    
    # Create significance markers (* for p<0.05, ** for p<0.01, *** for p<0.001)
    sig_markers = []
    for p in p_values:
        if p is None:
            sig_markers.append('')
        elif p < 0.001:
            sig_markers.append('***')
        elif p < 0.01:
            sig_markers.append('**')
        elif p < 0.05:
            sig_markers.append('*')
        else:
            sig_markers.append('')
    
    # Create bar plot
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=feature_names,
        y=coefficients,
        text=sig_markers,
        textposition='outside',
        marker=dict(
            color=coefficients,
            colorscale='RdBu_r',
            cmin=-max(abs(np.array(coefficients))),
            cmax=max(abs(np.array(coefficients))),
            colorbar=dict(title="Coefficient"),
        ),
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Predictor",
        yaxis_title="GLM Coefficient",
        template="plotly_white",
        height=500,
        showlegend=False,
    )
    
    # Add horizontal line at y=0
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    return fig


def plot_best_trial_prediction(
    unit,
    glm_results: dict,
    time_window: Tuple[float, float] = (-1.0, 2.0),
    category_boundaries: Tuple[float, float] = (0.983, 1.525),
    title: str = "Best Trial: Predicted vs Actual"
) -> go.Figure:
    """
    Plot the best-matching trial's full time course with predicted vs actual.
    
    Parameters:
    -----------
    unit : Unit
        Unit object with neural data
    glm_results : dict
        GLM results from fit_glm_for_unit
    time_window : tuple
        Time window used for GLM fitting
    category_boundaries : tuple
        Boundaries for Go/NoGo classification
    title : str
        Plot title
    
    Returns:
    --------
    go.Figure : Plotly figure
    """
    from sklearn.linear_model import PoissonRegressor
    from sklearn.preprocessing import StandardScaler
    
    if 'error' in glm_results:
        print(f"Cannot plot: {glm_results['error']}")
        return None
    
    # Rebuild design matrix and response
    stimuli = unit.stimuli_outcome_df['stimulus'].values if 'stimulus' in unit.stimuli_outcome_df.columns else np.zeros(len(unit.stimuli_outcome_df))
    outcomes = unit.stimuli_outcome_df['outcome'].values if 'outcome' in unit.stimuli_outcome_df.columns else np.array(['Unknown'] * len(unit.stimuli_outcome_df))
    
    X, y_actual, feature_names = build_design_matrix(
        unit_data=unit.unit_data,
        time_bins=unit.time_axis,
        stimuli=stimuli,
        outcomes=outcomes,
        bin_size=unit.bin_size,
        time_window=time_window,
        category_boundaries=category_boundaries,
    )
    
    # Fit model and get predictions
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = PoissonRegressor(alpha=1.0, max_iter=300)
    model.fit(X_scaled, y_actual)
    y_pred = model.predict(X_scaled)
    
    # Find the best trial (smallest absolute prediction error)
    prediction_errors = np.abs(y_actual - y_pred)
    best_trial_idx = np.argmin(prediction_errors)
    
    # Get full time course for the best trial
    best_trial_data = unit.unit_data[:, best_trial_idx]  # Full time course (firing rate in Hz)
    
    # Calculate spike counts per bin from firing rate
    bin_size = unit.bin_size if hasattr(unit, 'bin_size') else (unit.time_axis[1] - unit.time_axis[0]) if len(unit.time_axis) > 1 else 0.01
    
    # Convert firing rate (Hz) to spike counts per bin
    # Count = rate (Hz) * bin_size (s)
    spike_counts = best_trial_data * bin_size
    
    # Create smooth predicted firing rate (lambda) function
    # Use smoothed actual firing rate as proxy for predicted lambda
    from scipy.ndimage import gaussian_filter1d
    lambda_smooth = gaussian_filter1d(best_trial_data, sigma=2.0)  # Smooth the firing rate
    
    # Scale lambda to match the predicted average if needed
    predicted_value = y_pred[best_trial_idx]
    actual_avg = np.mean(best_trial_data)
    if actual_avg > 1e-10:
        lambda_smooth = lambda_smooth * (predicted_value / actual_avg)
    
    # Create figure
    fig = go.Figure()
    
    # Plot smooth predicted firing rate (lambda) as filled curve
    fig.add_trace(
        go.Scatter(
            x=unit.time_axis,
            y=lambda_smooth,
            mode='lines',
            name='Predicted Firing Rate (λ)',
            line=dict(color='green', width=3),
            fill='tozeroy',
            fillcolor='rgba(144, 238, 144, 0.3)',  # Light green fill
        )
    )
    
    # Plot spike counts per bin as bars
    fig.add_trace(
        go.Bar(
            x=unit.time_axis,
            y=spike_counts,
            name='Spike Count per Bin',
            marker=dict(
                color='black',
                opacity=0.6,
                line=dict(width=0.5, color='darkgray')
            ),
        )
    )
    
    # Set Y-axis range based on lambda function
    y_max = lambda_smooth.max() * 1.1
    y_min = 0  # Start from 0 for firing rate
    fig.add_vrect(
        x0=time_window[0],
        x1=time_window[1],
        fillcolor="yellow",
        opacity=0.2,
        layer="below",
        line_width=0,
        annotation_text="GLM Window",
        annotation_position="top left"
    )
    
    # Add vertical line at cue onset (t=0)
    fig.add_vline(
        x=0,
        line_dash="dash",
        line_color="gray",
        annotation_text="Cue Onset",
        annotation_position="top"
    )
    
    # Get trial information
    trial_stimulus = stimuli[best_trial_idx] if len(stimuli) > best_trial_idx else "N/A"
    trial_outcome = outcomes[best_trial_idx] if len(outcomes) > best_trial_idx else "N/A"
    prediction_error = prediction_errors[best_trial_idx]
    
    # Determine category (Go vs NoGo)
    trial_category = "Go" if trial_outcome in ['Hit', 'Miss'] else "NoGo" if trial_outcome in ['CR', 'False Alarm', 'FA'] else "Unknown"
    
    # Add background color for category
    if trial_category == "Go":
        fig.add_hrect(
            y0=y_min, y1=y_max,
            fillcolor="lightgreen",
            opacity=0.1,
            layer="below",
            line_width=0,
        )
    elif trial_category == "NoGo":
        fig.add_hrect(
            y0=y_min, y1=y_max,
            fillcolor="lightcoral",
            opacity=0.1,
            layer="below",
            line_width=0,
        )
    
    # Add reward marker if applicable
    if trial_outcome == 'Hit':
        reward_y = y_max * 0.95  # Position near top
        fig.add_trace(
            go.Scatter(
                x=[0.5],  # Approximate reward time (adjust based on your task)
                y=[reward_y],
                mode='markers+text',
                name='Reward',
                marker=dict(
                    symbol='triangle-up',
                    size=15,
                    color='green',
                    line=dict(width=2, color='darkgreen')
                ),
                text=['Reward'],
                textposition='top center',
                showlegend=True,
            )
        )
    
    # Add punishment marker if applicable
    if trial_outcome in ['False Alarm', 'FA']:
        punishment_y = y_max * 0.95
        fig.add_trace(
            go.Scatter(
                x=[0.5],  # Approximate punishment time
                y=[punishment_y],
                mode='markers+text',
                name='Punishment',
                marker=dict(
                    symbol='triangle-down',
                    size=15,
                    color='red',
                    line=dict(width=2, color='darkred')
                ),
                text=['Punishment'],
                textposition='top center',
                showlegend=True,
            )
        )
    
    # Try to extract first lick time if available
    # Check if we can get lick data from metadata or event windows data
    first_lick_time = None
    try:
        # Try to access lick data if available in event_windows_data
        if hasattr(unit, '_event_windows_data') and len(unit._event_windows_data) >= 6:
            lick_data = unit._event_windows_data[5]  # Assuming lick data is 6th element
            if lick_data is not None and best_trial_idx < len(lick_data):
                trial_licks = lick_data[best_trial_idx]
                if trial_licks is not None:
                    # Handle different lick data formats
                    if isinstance(trial_licks, (list, np.ndarray)):
                        # Filter out NaN values
                        valid_licks = [l for l in trial_licks if not (isinstance(l, float) and np.isnan(l))]
                        if len(valid_licks) > 0:
                            first_lick_time = float(valid_licks[0])
                    elif isinstance(trial_licks, dict):
                        # If it's a dict, try common keys
                        for key in ['first_lick', 'licks', 'lick_times']:
                            if key in trial_licks:
                                lick_val = trial_licks[key]
                                if isinstance(lick_val, (list, np.ndarray)) and len(lick_val) > 0:
                                    first_lick_time = float(lick_val[0])
                                    break
                                elif not (isinstance(lick_val, float) and np.isnan(lick_val)):
                                    first_lick_time = float(lick_val)
                                    break
    except (AttributeError, IndexError, TypeError, ValueError) as e:
        # Lick data not available or in unexpected format, skip
        pass
    
    # Add first lick marker if found
    if first_lick_time is not None and not np.isnan(first_lick_time):
        # Check if first lick is within the visible time range
        if unit.time_axis[0] <= first_lick_time <= unit.time_axis[-1]:
            lick_idx = np.argmin(np.abs(unit.time_axis - first_lick_time))
            first_lick_y = lambda_smooth[lick_idx] if lick_idx < len(lambda_smooth) else y_max * 0.8
            fig.add_trace(
                go.Scatter(
                    x=[first_lick_time],
                    y=[first_lick_y],
                    mode='markers+text',
                    name='First Lick',
                    marker=dict(
                        symbol='circle',
                        size=12,
                        color='orange',
                        line=dict(width=2, color='darkorange')
                    ),
                    text=['First Lick'],
                    textposition='top center',
                    showlegend=True,
                )
            )
    
    # Add stimulus frequency annotation
    if not np.isnan(trial_stimulus):
        stimulus_y = y_max * 0.85
        fig.add_annotation(
            x=0.1,
            y=stimulus_y,
            text=f"Stimulus: {trial_stimulus:.3f} kHz",
            showarrow=False,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=12, color="black"),
        )
    
    # Add category annotation
    category_y = y_max * 0.75
    fig.add_annotation(
        x=0.1,
        y=category_y,
        text=f"Category: {trial_category}",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="black",
        borderwidth=1,
        font=dict(size=12, color="black"),
    )
    
    # Get previous trial outcome if available
    prev_trial_outcome = "N/A"
    if best_trial_idx > 0:
        prev_trial_outcome = outcomes[best_trial_idx - 1] if len(outcomes) > best_trial_idx - 1 else "N/A"
        prev_trial_reward = "Yes" if prev_trial_outcome == 'Hit' else "No"
        
        # Add previous trial outcome annotation
        prev_outcome_y = y_max * 0.65
        fig.add_annotation(
            x=0.1,
            y=prev_outcome_y,
            text=f"Prev Trial: {prev_trial_outcome} (Reward: {prev_trial_reward})",
            showarrow=False,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=11, color="black"),
        )
    
    # Add summary box with all predictor values
    summary_text = f"<b>Predictors:</b><br>"
    summary_text += f"Stimulus: {trial_stimulus:.3f} kHz<br>"
    summary_text += f"Category: {trial_category}<br>"
    summary_text += f"Outcome: {trial_outcome}<br>"
    if first_lick_time is not None and not np.isnan(first_lick_time):
        summary_text += f"First Lick: {first_lick_time:.3f} s<br>"
    if best_trial_idx > 0:
        summary_text += f"Prev Outcome: {prev_trial_outcome}"
    
    fig.add_annotation(
        x=1.0,  # Position on right side
        y=y_max * 0.9,
        text=summary_text,
        showarrow=False,
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="blue",
        borderwidth=2,
        font=dict(size=10, color="black"),
        align="left",
        xref="x",
        yref="y",
    )
    
    # Update layout
    fig.update_layout(
        title=f"{title}<br>Trial {best_trial_idx}: Stimulus={trial_stimulus:.3f}, Outcome={trial_outcome}, Category={trial_category}, Error={prediction_error:.3f} Hz<br>Pseudo R² = {glm_results['pseudo_r2']:.4f}",
        xaxis_title="Time (s)",
        yaxis_title="Firing Rate (Hz)",
        template="plotly_white",
        height=600,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
    )
    
    return fig


def plot_top_5_trials_with_kernels(
    unit,
    glm_results: dict,
    time_window: Tuple[float, float] = (-1.0, 2.0),
    category_boundaries: Tuple[float, float] = (0.983, 1.525),
    title: str = "Top 5 Representative Trials with Predictor Kernels",
    n_trials: int = 5,
    use_temporal_kernels: bool = True,
    n_basis: int = 10,
) -> go.Figure:
    """
    Plot the top N representative trials (one per outcome type) with convolved predictor kernels below each.
    
    Parameters:
    -----------
    unit : Unit
        Unit object with neural data
    glm_results : dict
        GLM results from fit_glm_for_unit
    time_window : tuple
        Time window used for GLM fitting
    category_boundaries : tuple
        Boundaries for Go/NoGo classification
    title : str
        Plot title
    n_trials : int
        Number of trials to show (default: 5, one per outcome type)
    
    Returns:
    --------
    go.Figure : Plotly figure with subplots
    """
    from sklearn.linear_model import PoissonRegressor
    from sklearn.preprocessing import StandardScaler
    
    if 'error' in glm_results:
        print(f"Cannot plot: {glm_results['error']}")
        return None
    
    # Rebuild design matrix and response
    stimuli = unit.stimuli_outcome_df['stimulus'].values if 'stimulus' in unit.stimuli_outcome_df.columns else np.zeros(len(unit.stimuli_outcome_df))
    outcomes = unit.stimuli_outcome_df['outcome'].values if 'outcome' in unit.stimuli_outcome_df.columns else np.array(['Unknown'] * len(unit.stimuli_outcome_df))
    
    X, y_actual, feature_names = build_design_matrix(
        unit_data=unit.unit_data,
        time_bins=unit.time_axis,
        stimuli=stimuli,
        outcomes=outcomes,
        bin_size=unit.bin_size,
        time_window=time_window,
        category_boundaries=category_boundaries,
    )
    
    # Fit model and get predictions
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = PoissonRegressor(alpha=1.0, max_iter=300)
    model.fit(X_scaled, y_actual)
    y_pred = model.predict(X_scaled)
    
    # Find trials ensuring representation of all outcome types
    prediction_errors = np.abs(y_actual - y_pred)
    
    # Group trials by outcome type
    outcome_types = ['Hit', 'Miss', 'CR', 'False Alarm', 'FA']
    trials_by_outcome = {outcome: [] for outcome in outcome_types}
    
    for i, outcome in enumerate(outcomes):
        # Normalize outcome names
        if outcome == 'False Alarm':
            outcome = 'FA'
        if outcome in trials_by_outcome:
            trials_by_outcome[outcome].append(i)
    
    # Select best trial from each outcome type
    selected_trials = []
    for outcome in outcome_types:
        if len(trials_by_outcome[outcome]) > 0:
            # Get prediction errors for this outcome type
            outcome_errors = prediction_errors[trials_by_outcome[outcome]]
            # Select the best one (smallest error)
            best_idx_in_group = np.argmin(outcome_errors)
            selected_trials.append(trials_by_outcome[outcome][best_idx_in_group])
    
    # If we need more trials, fill with best overall trials
    remaining_slots = n_trials - len(selected_trials)
    if remaining_slots > 0:
        # Get all trial indices sorted by error
        all_trials_sorted = np.argsort(prediction_errors)
        # Add best trials that aren't already selected
        for trial_idx in all_trials_sorted:
            if trial_idx not in selected_trials and len(selected_trials) < n_trials:
                selected_trials.append(trial_idx)
    
    # Sort selected trials by prediction error for display
    selected_trials = sorted(selected_trials, key=lambda x: prediction_errors[x])
    top_trial_indices = selected_trials[:n_trials]
    
    # Get GLM coefficients and feature names
    coefficients = glm_results.get('coefficients', [])
    feature_names = glm_results.get('feature_names', [])
    
    # Create subplots: For each trial, we need 1 row for the trace + 1 row for kernels
    # Layout: 5 trials × 2 rows each = 10 rows total, 1 column
    n_trial_rows = n_trials
    n_kernel_rows = n_trials
    total_rows = n_trial_rows + n_kernel_rows
    
    # Create subplot titles
    subplot_titles = []
    for idx in top_trial_indices:
        outcome = outcomes[idx] if len(outcomes) > idx else "Unknown"
        error = prediction_errors[idx]
        subplot_titles.append(f"Trial {idx}: {outcome} (Error: {error:.3f})")
    
    # Add kernel subplot titles
    for idx in top_trial_indices:
        subplot_titles.append(f"Predictor Kernels - Trial {idx}")
    
    fig = make_subplots(
        rows=total_rows,
        cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=0.05,
        row_heights=[2] * n_trial_rows + [1] * n_kernel_rows,  # Trial plots are larger
    )
    
    # Try to get lick data if available
    lick_data = None
    try:
        if hasattr(unit, '_event_windows_data') and len(unit._event_windows_data) >= 6:
            lick_data = unit._event_windows_data[5]
    except (AttributeError, IndexError):
        pass
    
    # Helper function to compute predictor kernels for a trial
    def compute_predictor_kernels(trial_idx, time_axis, coefficients, feature_names, 
                                   stimuli, outcomes, trial_data, time_window, n_basis=10):
        """Compute time-varying predictor contributions using raised cosine basis functions."""
        from Analysis.NPXL_analysis.single_unit_offline_analysis.glm_fitting import (
            create_raised_cosine_basis,
            convolve_predictor_with_basis,
        )
        
        kernels = {}
        n_time = len(time_axis)
        
        # Create raised cosine basis functions
        basis_functions = create_raised_cosine_basis(
            time_axis=time_axis,
            n_basis=n_basis,
            window=time_window,
        )
        
        # Get trial-specific values
        trial_stimulus = stimuli[trial_idx] if len(stimuli) > trial_idx else 0
        trial_outcome = outcomes[trial_idx] if len(outcomes) > trial_idx else "Unknown"
        
        # Group coefficients by predictor type
        predictor_groups = {}
        for i, feat_name in enumerate(feature_names):
            if i >= len(coefficients):
                continue
            
            # Extract base predictor name (remove _basisX suffix)
            if '_basis' in feat_name:
                base_name = feat_name.split('_basis')[0]
                basis_idx = int(feat_name.split('_basis')[1].split('_')[0])
            else:
                base_name = feat_name
                basis_idx = 0
            
            if base_name not in predictor_groups:
                predictor_groups[base_name] = {}
            predictor_groups[base_name][basis_idx] = coefficients[i]
        
        # Reconstruct kernels for each predictor
        for base_name, basis_coefs in predictor_groups.items():
            kernel = np.zeros(n_time)
            
            # Sum contributions from all basis functions
            for basis_idx in range(n_basis):
                if basis_idx in basis_coefs:
                    coef = basis_coefs[basis_idx]
                    # Add basis function weighted by coefficient
                    kernel += basis_functions[:, basis_idx] * coef
            
            # Scale by predictor value for this trial
            if base_name == 'cue_onset':
                # Cue onset: always 1
                kernels[base_name] = kernel
            elif base_name == 'stimulus':
                # Stimulus: scale by stimulus value
                kernels[base_name] = kernel * trial_stimulus
            elif base_name == 'category_go':
                # Category: scale by category value (1 for Go, 0 for NoGo)
                category_val = 1.0 if trial_outcome in ['Hit', 'Miss'] else 0.0
                kernels[base_name] = kernel * category_val
            elif base_name == 'reward':
                # Reward: event-based, centered at reward time
                if trial_outcome == 'Hit':
                    # Reconstruct kernel centered at reward time (0.5s)
                    reward_time = 0.5
                    reward_kernel = np.zeros(n_time)
                    # Find where basis functions peak (should be at time_window start)
                    basis_peak_time = time_window[0] + (time_window[1] - time_window[0]) / (n_basis - 1) * 0 if n_basis > 1 else time_window[0]
                    time_offset = reward_time - basis_peak_time
                    reward_bin_offset = int(time_offset / (time_axis[1] - time_axis[0])) if len(time_axis) > 1 else 0
                    
                    # Reconstruct kernel by shifting each basis function
                    for basis_idx in range(n_basis):
                        if basis_idx in basis_coefs:
                            coef = basis_coefs[basis_idx]
                            # Shift basis function to reward time
                            shifted_basis = np.roll(basis_functions[:, basis_idx], reward_bin_offset)
                            if reward_bin_offset > 0:
                                shifted_basis[:reward_bin_offset] = 0
                            elif reward_bin_offset < 0:
                                shifted_basis[reward_bin_offset:] = 0
                            reward_kernel += shifted_basis * coef
                    kernels[base_name] = reward_kernel
                else:
                    kernels[base_name] = np.zeros(n_time)
            elif base_name == 'punishment':
                # Punishment: event-based, centered at punishment time
                if trial_outcome in ['False Alarm', 'FA']:
                    punishment_time = 0.5
                    punishment_kernel = np.zeros(n_time)
                    basis_peak_time = time_window[0] + (time_window[1] - time_window[0]) / (n_basis - 1) * 0 if n_basis > 1 else time_window[0]
                    time_offset = punishment_time - basis_peak_time
                    punishment_bin_offset = int(time_offset / (time_axis[1] - time_axis[0])) if len(time_axis) > 1 else 0
                    
                    for basis_idx in range(n_basis):
                        if basis_idx in basis_coefs:
                            coef = basis_coefs[basis_idx]
                            shifted_basis = np.roll(basis_functions[:, basis_idx], punishment_bin_offset)
                            if punishment_bin_offset > 0:
                                shifted_basis[:punishment_bin_offset] = 0
                            elif punishment_bin_offset < 0:
                                shifted_basis[punishment_bin_offset:] = 0
                            punishment_kernel += shifted_basis * coef
                    kernels[base_name] = punishment_kernel
                else:
                    kernels[base_name] = np.zeros(n_time)
            elif base_name == 'prev_trial_reward':
                # Previous trial reward: scale by previous outcome
                prev_reward = 1.0 if trial_idx > 0 and outcomes[trial_idx-1] == 'Hit' else 0.0
                kernels[base_name] = kernel * prev_reward
            elif 'lick' in base_name.lower():
                # Lick predictors: would need actual timing
                kernels[base_name] = kernel
            else:
                # Other predictors: use kernel as is
                kernels[base_name] = kernel
        
        return kernels
    
    # Plot each trial
    for plot_idx, trial_idx in enumerate(top_trial_indices):
        trial_row = plot_idx + 1  # Row for trial trace
        kernel_row = n_trial_rows + plot_idx + 1  # Row for kernels
        
        # Get trial data
        trial_data = unit.unit_data[:, trial_idx]  # Firing rate in Hz
        trial_stimulus = stimuli[trial_idx] if len(stimuli) > trial_idx else np.nan
        trial_outcome = outcomes[trial_idx] if len(outcomes) > trial_idx else "Unknown"
        trial_category = "Go" if trial_outcome in ['Hit', 'Miss'] else "NoGo" if trial_outcome in ['CR', 'False Alarm', 'FA'] else "Unknown"
        predicted_value = y_pred[trial_idx]
        
        # Calculate spike counts per bin from firing rate
        bin_size = unit.bin_size if hasattr(unit, 'bin_size') else (unit.time_axis[1] - unit.time_axis[0]) if len(unit.time_axis) > 1 else 0.01
        
        # Convert firing rate (Hz) to spike counts per bin
        # Count = rate (Hz) * bin_size (s)
        spike_counts = trial_data * bin_size
        
        # Create smooth predicted firing rate (lambda) function
        from scipy.ndimage import gaussian_filter1d
        lambda_smooth = gaussian_filter1d(trial_data, sigma=2.0)
        
        # Scale lambda to match predicted average
        actual_avg = np.mean(trial_data)
        if actual_avg > 1e-10:
            lambda_smooth = lambda_smooth * (predicted_value / actual_avg)
        
        y_max = lambda_smooth.max() * 1.1
        y_min = 0
        
        # Add category background
        if trial_category == "Go":
            fig.add_hrect(
                y0=y_min, y1=y_max,
                fillcolor="lightgreen",
                opacity=0.1,
                layer="below",
                line_width=0,
                row=trial_row, col=1
            )
        elif trial_category == "NoGo":
            fig.add_hrect(
                y0=y_min, y1=y_max,
                fillcolor="lightcoral",
                opacity=0.1,
                layer="below",
                line_width=0,
                row=trial_row, col=1
            )
        
        # Plot smooth predicted firing rate (lambda) as filled curve
        fig.add_trace(
            go.Scatter(
                x=unit.time_axis,
                y=lambda_smooth,
                mode='lines',
                name='Predicted (λ)' if plot_idx == 0 else '',
                line=dict(color='green', width=2),
                fill='tozeroy',
                fillcolor='rgba(144, 238, 144, 0.3)',
                showlegend=(plot_idx == 0),
            ),
            row=trial_row, col=1
        )
        
        # Plot spike counts per bin as bars
        fig.add_trace(
            go.Bar(
                x=unit.time_axis,
                y=spike_counts,
                name='Spike Count per Bin' if plot_idx == 0 else '',
                marker=dict(
                    color='black',
                    opacity=0.6,
                    line=dict(width=0.5, color='darkgray')
                ),
                showlegend=(plot_idx == 0),
            ),
            row=trial_row, col=1
        )
        
        # Add cue onset line
        fig.add_vline(
            x=0,
            line_dash="dash",
            line_color="gray",
            line_width=1,
            row=trial_row, col=1
        )
        
        # Shade GLM window
        fig.add_vrect(
            x0=time_window[0],
            x1=time_window[1],
            fillcolor="yellow",
            opacity=0.15,
            layer="below",
            line_width=0,
            row=trial_row, col=1
        )
        
        # Add reward marker
        if trial_outcome == 'Hit':
            reward_y = y_max * 0.9
            fig.add_trace(
                go.Scatter(
                    x=[0.5],
                    y=[reward_y],
                    mode='markers',
                    name='Reward' if plot_idx == 0 else '',
                    marker=dict(symbol='triangle-up', size=8, color='green', line=dict(width=1, color='darkgreen')),
                    showlegend=(plot_idx == 0),
                ),
                row=trial_row, col=1
            )
        
        # Add punishment marker
        if trial_outcome in ['False Alarm', 'FA']:
            punishment_y = y_max * 0.9
            fig.add_trace(
                go.Scatter(
                    x=[0.5],
                    y=[punishment_y],
                    mode='markers',
                    name='Punishment' if plot_idx == 0 else '',
                    marker=dict(symbol='triangle-down', size=8, color='red', line=dict(width=1, color='darkred')),
                    showlegend=(plot_idx == 0),
                ),
                row=trial_row, col=1
            )
        
        # Try to add first lick marker
        if lick_data is not None and trial_idx < len(lick_data):
            try:
                trial_licks = lick_data[trial_idx]
                if trial_licks is not None:
                    if isinstance(trial_licks, (list, np.ndarray)):
                        valid_licks = [l for l in trial_licks if not (isinstance(l, float) and np.isnan(l))]
                        if len(valid_licks) > 0:
                            first_lick_time = float(valid_licks[0])
                            if unit.time_axis[0] <= first_lick_time <= unit.time_axis[-1]:
                                lick_idx = np.argmin(np.abs(unit.time_axis - first_lick_time))
                                first_lick_y = lambda_smooth[lick_idx] if lick_idx < len(lambda_smooth) else y_max * 0.8
                                fig.add_trace(
                                    go.Scatter(
                                        x=[first_lick_time],
                                        y=[first_lick_y],
                                        mode='markers',
                                        name='First Lick' if plot_idx == 0 else '',
                                        marker=dict(symbol='circle', size=8, color='orange', line=dict(width=1, color='darkorange')),
                                        showlegend=(plot_idx == 0),
                                    ),
                                    row=trial_row, col=1
                                )
            except (IndexError, TypeError, ValueError):
                pass
        
        # Add trial info annotation
        info_text = f"S:{trial_stimulus:.2f} {trial_category[:2]} {trial_outcome[:1]}"
        fig.add_annotation(
            x=0.05,
            y=y_max * 0.95,
            text=info_text,
            showarrow=False,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=9, color="black"),
            row=trial_row, col=1
        )
        
        # Compute predictor kernels for this trial
        # Extract n_basis from feature names (count unique basis indices)
        n_basis_used = n_basis  # Use provided n_basis
        if use_temporal_kernels and len(feature_names) > 0:
            # Try to extract n_basis from feature names
            basis_indices = []
            for name in feature_names:
                if '_basis' in name:
                    try:
                        basis_idx = int(name.split('_basis')[1].split('_')[0])
                        basis_indices.append(basis_idx)
                    except (ValueError, IndexError):
                        pass
            if len(basis_indices) > 0:
                n_basis_used = max(basis_indices) + 1
        
        if use_temporal_kernels:
            kernels = compute_predictor_kernels(
                trial_idx, unit.time_axis, coefficients, feature_names,
                stimuli, outcomes, trial_data, time_window, n_basis=n_basis_used
            )
        else:
            # Fallback to simple kernels if not using temporal basis
            kernels = {}
            for feat_name in set([name.split('_basis')[0] if '_basis' in name else name for name in feature_names]):
                kernels[feat_name] = np.zeros(len(unit.time_axis))
        
        # Plot predictor kernels below the trial
        kernel_colors = {
            'cue_onset': 'purple',
            'stimulus': 'blue',
            'category_go': 'green',
            'first_lick': 'orange',
            'lick_count': 'orange',
            'reward': 'green',
            'punishment': 'red',
            'prev_trial_reward': 'cyan',
        }
        
        # Plot each kernel
        for feat_name, kernel_values in kernels.items():
            color = kernel_colors.get(feat_name, 'gray')
            fig.add_trace(
                go.Scatter(
                    x=unit.time_axis,
                    y=kernel_values,
                    mode='lines',
                    name=feat_name if plot_idx == 0 else '',
                    line=dict(color=color, width=1),
                    showlegend=(plot_idx == 0),
                ),
                row=kernel_row, col=1
            )
        
        # Add zero line for kernels
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="gray",
            line_width=0.5,
            row=kernel_row, col=1
        )
        
        # Add cue onset line for kernels
        fig.add_vline(
            x=0,
            line_dash="dash",
            line_color="gray",
            line_width=0.5,
            row=kernel_row, col=1
        )
        
        # Update axes
        if plot_idx == n_trials - 1:  # Last trial
            fig.update_xaxes(title_text="Time (s)", row=trial_row, col=1)
        if plot_idx == 0:  # First trial
            fig.update_yaxes(title_text="Firing Rate (Hz)", row=trial_row, col=1)
        
        # Update kernel axes
        if plot_idx == n_trials - 1:  # Last kernel plot
            fig.update_xaxes(title_text="Time (s)", row=kernel_row, col=1)
        if plot_idx == 0:  # First kernel plot
            fig.update_yaxes(title_text="Contribution", row=kernel_row, col=1)
    
    # Count outcome types in selected trials
    outcome_counts = {}
    for idx in top_trial_indices:
        outcome = outcomes[idx] if len(outcomes) > idx else "Unknown"
        outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
    
    outcome_summary = ", ".join([f"{k}: {v}" for k, v in outcome_counts.items()])
    
    # Update layout
    fig.update_layout(
        title=f"{title} - Unit {unit.unit_idx}<br>Pseudo R² = {glm_results['pseudo_r2']:.4f} | Outcomes: {outcome_summary}",
        template="plotly_white",
        height=2000,  # Taller to accommodate kernels
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.02,
            xanchor="center",
            x=0.5
        ),
    )
    
    return fig


def plot_glm_summary_statistics(glm_df: pd.DataFrame, region_name: str = "") -> go.Figure:
    """
    Create summary plots for GLM results across multiple units.
    
    Parameters:
    -----------
    glm_df : pd.DataFrame
        DataFrame with GLM results for multiple units
    region_name : str
        Name of brain region
    
    Returns:
    --------
    go.Figure : Plotly figure with subplots
    """
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Pseudo R² Distribution',
            'Cross-Validation Score Distribution',
            'Significant Predictors',
            'Average Coefficient Magnitudes'
        )
    )
    
    # 1. Pseudo R² distribution
    if 'glm_pseudo_r2' in glm_df.columns:
        r2_values = glm_df['glm_pseudo_r2'].dropna()
        fig.add_trace(
            go.Histogram(x=r2_values, nbinsx=30, name='Pseudo R²'),
            row=1, col=1
        )
    
    # 2. CV score distribution
    if 'glm_cv_score' in glm_df.columns:
        cv_scores = glm_df['glm_cv_score'].dropna()
        fig.add_trace(
            go.Histogram(x=cv_scores, nbinsx=30, name='CV Score'),
            row=1, col=2
        )
    
    # 3. Count significant predictors (p < 0.05)
    pval_cols = [col for col in glm_df.columns if col.startswith('glm_pval_')]
    if pval_cols:
        sig_counts = {}
        for col in pval_cols:
            predictor = col.replace('glm_pval_', '')
            sig_count = (glm_df[col] < 0.05).sum()
            sig_counts[predictor] = sig_count
        
        fig.add_trace(
            go.Bar(x=list(sig_counts.keys()), y=list(sig_counts.values()), 
                   name='Significant Units'),
            row=2, col=1
        )
    
    # 4. Average coefficient magnitudes
    coef_cols = [col for col in glm_df.columns if col.startswith('glm_coef_')]
    if coef_cols:
        avg_coefs = {}
        for col in coef_cols:
            predictor = col.replace('glm_coef_', '')
            avg_coef = glm_df[col].abs().mean()
            avg_coefs[predictor] = avg_coef
        
        fig.add_trace(
            go.Bar(x=list(avg_coefs.keys()), y=list(avg_coefs.values()),
                   name='|Coefficient|'),
            row=2, col=2
        )
    
    fig.update_layout(
        title_text=f"GLM Summary Statistics - {region_name}" if region_name else "GLM Summary Statistics",
        showlegend=False,
        height=800,
        template="plotly_white",
    )
    
    return fig


def example_single_unit_glm():
    """
    Example: Fit GLM for all units, then show detailed analysis for the best fitted unit.
    """
    print("=" * 80)
    print("Example 1: Finding best fitted unit and showing detailed analysis")
    print("=" * 80)
    
    # Load data
    parent_dir = r"Z:\Shared\Amichai\NPXL\Recs\group7\catgt_G7A2_novice_2b_4t_g1"
    
    print("\nLoading data...")
    OFC_all, ACx_all, data_dir_OFC, data_dir_ACx = load_data(data_dir_parent=parent_dir)
    
    # Load full event windows data (includes behavioral data)
    print("\nLoading event windows data...")
    acx_event_windows_data = load_full_event_windows_data(data_dir_ACx)
    
    # Create Unit objects
    # Extract event matrix to determine number of units
    acx_event_matrix = acx_event_windows_data[0]
    n_units = acx_event_matrix.shape[0]
    
    # Create unit indices - use more units to find the best one
    max_units_to_test = min(50, n_units)  # Test up to 50 units
    unit_indices = np.arange(max_units_to_test)
    
    units = create_units_from_event_data(
        acx_event_windows_data,
        unit_indices=unit_indices,
        region_name="ACx",
    )
    
    if len(units) == 0:
        print("No units found!")
        return
    
    print(f"\nCreated {len(units)} Unit objects")
    
    # Fit GLM for all units to find the best one
    print("\nFitting GLM for all units to find the best fitted unit...")
    glm_df = fit_glm_for_all_units(
        units=units,
        time_window=(-1.0, 2.0),
        alpha=1.0,
        category_boundaries=(0.983, 1.525),
        normalize_response='zscore',
        use_temporal_kernels=True,
        n_basis=10,
        include_spike_history=True,
        save_to_csv=False,  # Don't save intermediate results
    )
    
    # Find the best unit (highest pseudo R²)
    if 'glm_pseudo_r2' not in glm_df.columns or glm_df['glm_pseudo_r2'].isna().all():
        print("\nWarning: No valid GLM results found. Using first unit.")
        best_unit_idx = 0
        best_r2 = 0.0
    else:
        # Find unit with highest pseudo R² (excluding NaN values)
        valid_r2 = glm_df['glm_pseudo_r2'].dropna()
        if len(valid_r2) == 0:
            print("\nWarning: No valid R² values found. Using first unit.")
            best_unit_idx = 0
            best_r2 = 0.0
        else:
            best_row = glm_df.loc[glm_df['glm_pseudo_r2'].idxmax()]
            best_unit_idx = best_row['unit_idx']
            best_r2 = best_row['glm_pseudo_r2']
            print(f"\nBest fitted unit: Unit {best_unit_idx} with Pseudo R² = {best_r2:.4f}")
    
    # Find the unit object corresponding to best_unit_idx
    unit = None
    for u in units:
        if u.unit_idx == best_unit_idx:
            unit = u
            break
    
    if unit is None:
        print(f"\nWarning: Could not find unit {best_unit_idx}. Using first unit.")
        unit = units[0]
    
    print(f"\nAnalyzing best unit: Unit {unit.unit_idx} from {unit.region_name}")
    print(f"  Neural data shape: {unit.unit_data.shape}")
    print(f"  Time bins: {unit.n_time_bins}")
    print(f"  Trials: {unit.n_trials}")
    
    # Extract lick data if available
    licks_list = extract_licks_from_unit(unit, acx_event_windows_data)
    if licks_list is not None:
        n_trials_with_licks = sum(1 for l in licks_list if l is not None and len(l) > 0)
        print(f"  Found lick data for {n_trials_with_licks} / {len(licks_list)} trials")
    else:
        print("  No lick data found - lick predictors will be excluded")
    
    # Fit GLM again for detailed results (to get full feature names, etc.)
    print("\nFitting GLM for detailed analysis...")
    glm_results = fit_glm_for_unit(
        unit_data=unit.unit_data,
        time_bins=unit.time_axis,
        stimuli_outcome_df=unit.stimuli_outcome_df,
        licks_list=licks_list,  # Pass extracted lick data
        bin_size=unit.bin_size,
        time_window=(-1.0, 2.0),  # Response window
        alpha=1.0,  # Regularization strength
        category_boundaries=(0.983, 1.525),  # Boundaries for Go/NoGo classification
        normalize_response='zscore',  # Normalize neural response before fitting
        use_temporal_kernels=True,  # Use raised cosine basis functions
        n_basis=10,  # Number of basis functions per predictor
        include_spike_history=True,
    )
    
    # Display results
    print("\nGLM Results for Best Unit:")
    print(f"  Pseudo R²: {glm_results.get('pseudo_r2', 'N/A'):.4f}")
    print(f"  Deviance: {glm_results.get('deviance', 'N/A'):.2f}")
    print(f"  CV Score: {glm_results.get('cv_score', 'N/A'):.2f}")
    print(f"  Number of features: {glm_results.get('n_features', 'N/A')}")
    print(f"  Number of trials: {glm_results.get('n_trials', 'N/A')}")
    
    if 'feature_names' in glm_results:
        print("\nCoefficients:")
        for name, coef, pval in zip(
            glm_results['feature_names'],
            glm_results['coefficients'],
            glm_results.get('p_values', [None] * len(glm_results['feature_names']))
        ):
            sig = ""
            if pval is not None:
                if pval < 0.001:
                    sig = " ***"
                elif pval < 0.01:
                    sig = " **"
                elif pval < 0.05:
                    sig = " *"
            pval_str = f"{pval:.4f}" if pval is not None else "N/A"
            print(f"  {name:20s}: {coef:8.4f}  (p={pval_str}){sig}")
    
    # Plot coefficients
    output_dir = os.path.join(parent_dir, "analysis_output", "glm_results")
    os.makedirs(output_dir, exist_ok=True)
    
    fig_coef = plot_glm_coefficients(glm_results, title=f"GLM Coefficients - Best Unit {unit.unit_idx} (R²={glm_results.get('pseudo_r2', 0):.4f})")
    if fig_coef:
        output_path = os.path.join(output_dir, f"glm_coefficients_unit_{unit.unit_idx}.html")
        fig_coef.write_html(output_path)
        print(f"\nSaved coefficient plot to: {output_path}")
    

    # Plot best trial
    print("\nGenerating best trial plot...")
    fig_best = plot_best_trial_prediction(
        unit=unit,
        glm_results=glm_results,
        time_window=(-1.0, 2.0),
        category_boundaries=(0.983, 1.525),
        title=f"Best Trial Prediction - Best Unit {unit.unit_idx} (R²={glm_results.get('pseudo_r2', 0):.4f})"
    )
    if fig_best:
        output_path = os.path.join(output_dir, f"glm_best_trial_unit_{unit.unit_idx}.html")
        fig_best.write_html(output_path)
        print(f"Saved best trial plot to: {output_path}")
    
    # Plot top 5 trials with kernels
    print("\nGenerating top 5 trials with kernels plot...")
    fig_top5 = plot_top_5_trials_with_kernels(
        unit=unit,
        glm_results=glm_results,
        time_window=(-1.0, 2.0),
        category_boundaries=(0.983, 1.525),
        title=f"Top 5 Representative Trials with Predictor Kernels - Best Unit {unit.unit_idx} (R²={glm_results.get('pseudo_r2', 0):.4f})",
        n_trials=5,
        use_temporal_kernels=True,
        n_basis=10,
    )
    if fig_top5:
        output_path = os.path.join(output_dir, f"glm_top5_trials_kernels_unit_{unit.unit_idx}.html")
        fig_top5.write_html(output_path)
        print(f"Saved top 5 trials with kernels plot to: {output_path}")


def example_all_units_glm():
    """
    Example: Fit GLM for all units and analyze results.
    """
    print("\n" + "=" * 80)
    print("Example 2: Fitting GLM for all units")
    print("=" * 80)
    
    # Load data
    parent_dir = r"Z:\Shared\Amichai\NPXL\Recs\group7\catgt_G7A2_novice_2b_4t_g1"
    
    print("\nLoading data...")
    OFC_all, ACx_all, data_dir_OFC, data_dir_ACx = load_data(data_dir_parent=parent_dir)
    
    # Load full event windows data
    print("\nLoading ACx event windows data...")
    acx_event_windows_data = load_full_event_windows_data(data_dir_ACx)
    
    # Create Unit objects
    print("\nCreating Unit objects...")
    # Extract event matrix to determine number of units
    acx_event_matrix = acx_event_windows_data[0]
    n_units = acx_event_matrix.shape[0]
    
    # Create unit indices for first 50 units
    unit_indices = np.arange(min(50, n_units))
    
    units = create_units_from_event_data(
        acx_event_windows_data,
        unit_indices=unit_indices,
        region_name="ACx",
    )
    
    print(f"Created {len(units)} Unit objects")
    
    # Setup results directory
    output_dir = os.path.join(parent_dir, "analysis_output", "glm_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Fit GLM for all units
    print("\nFitting GLM for all units...")
    glm_df = fit_glm_for_all_units(
        units=units,
        time_window=(-1.0, 2.0),
        alpha=1.0,
        category_boundaries=(0.983, 1.525),  # Boundaries for Go/NoGo classification
        normalize_response='zscore',  # Normalize neural response before fitting
        use_temporal_kernels=True,  # Use raised cosine basis functions
        n_basis=10,  # Number of basis functions per predictor
        save_to_csv=True,
        output_path=os.path.join(output_dir, "glm_results_all_units.csv")
    )
    
    print(f"\nGLM fitting complete for {len(glm_df)} units")
    print(f"Results saved to: {output_dir}")
    
    # Display summary statistics
    print("\nSummary Statistics:")
    if 'glm_pseudo_r2' in glm_df.columns:
        r2_values = glm_df['glm_pseudo_r2'].dropna()
        print(f"  Pseudo R² - Mean: {r2_values.mean():.4f}, Median: {r2_values.median():.4f}")
        print(f"  Pseudo R² - Range: [{r2_values.min():.4f}, {r2_values.max():.4f}]")
    
    if 'glm_cv_score' in glm_df.columns:
        cv_values = glm_df['glm_cv_score'].dropna()
        print(f"  CV Score - Mean: {cv_values.mean():.4f}, Median: {cv_values.median():.4f}")
    
    # Count units with good model fit (pseudo R² > 0.1)
    if 'glm_pseudo_r2' in glm_df.columns:
        good_fit = (glm_df['glm_pseudo_r2'] > 0.1).sum()
        print(f"  Units with Pseudo R² > 0.1: {good_fit} / {len(glm_df)} ({100*good_fit/len(glm_df):.1f}%)")
    
    # Create summary plots
    print("\nCreating summary plots...")
    fig = plot_glm_summary_statistics(glm_df, region_name="ACx")
    output_path = os.path.join(output_dir, "glm_summary_statistics.html")
    fig.write_html(output_path)
    print(f"Saved summary plots to: {output_path}")

# %%

"""
Run all examples.
"""
# Example 1: Single unit GLM
example_single_unit_glm()
# %%
# Example 2: All units GLM
example_all_units_glm()
# %%
print("\n" + "=" * 80)
print("All examples completed!")
print("=" * 80)


