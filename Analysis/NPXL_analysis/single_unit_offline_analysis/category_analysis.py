"""
Category sensitivity analysis functions for NPXL offline analysis.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import f_oneway
from sklearn.metrics import roc_auc_score
from Analysis.NPXL_analysis.single_unit_offline_analysis.visualization import _hex_to_rgba


def assign_stimulus_categories(
    stimuli: np.ndarray,
    low_boundary: float = 0.983,
    high_boundary: float = 1.525,
) -> np.ndarray:
    """
    Assign each stimulus to Go/NoGo categories.
    
    Categories:
    - 'Go': stimulus < low_boundary OR stimulus > high_boundary
    - 'NoGo': low_boundary <= stimulus <= high_boundary
    
    Parameters:
    -----------
    stimuli : np.ndarray
        Array of stimulus values
    low_boundary : float
        Lower category boundary (default: 0.983)
    high_boundary : float
        Upper category boundary (default: 1.525)
    
    Returns:
    --------
    np.ndarray
        Array of category labels ('Go', 'NoGo')
    """
    categories = np.full(len(stimuli), "NoGo", dtype=object)
    go_mask = (stimuli < low_boundary) | (stimuli > high_boundary)
    categories[go_mask] = "Go"
    return categories


def compute_category_sensitivity(
    event_windows_data: tuple,
    unit_idx: int,
    low_boundary: float = 0.983,
    high_boundary: float = 1.525,
    window: tuple[float, float] = (-0.2, 1),
) -> dict:
    """
    Test if a unit is sensitive to Go vs NoGo (Low+High vs Middle).
    
    Uses a two-group ANOVA (equivalent to a t-test) on Go vs NoGo rates.
    
    Parameters:
    -----------
    event_windows_data : tuple
        Event windows data tuple
    unit_idx : int
        Unit index to test
    low_boundary : float
        Lower category boundary
    high_boundary : float
        Upper category boundary
    window : tuple[float, float]
        Time window for analysis (start, end) in seconds
    
    Returns:
    --------
    dict
        Dictionary with category sensitivity metrics
    """
    # Handle both 5-tuple and 6-tuple formats
    if len(event_windows_data) == 6:
        event_windows_matrix, time_axis, valid_event_indices, stimuli_outcome_df, metadata, _ = event_windows_data
    else:
        event_windows_matrix, time_axis, valid_event_indices, stimuli_outcome_df, metadata = event_windows_data
    
    if 'stimulus' not in stimuli_outcome_df.columns:
        return {
            'category_sensitive': False,
            'category_anova_p': np.nan,
            'category_anova_f': np.nan,
            'go_mean_rate': np.nan,
            'nogo_mean_rate': np.nan,
            'go_n_trials': 0,
            'nogo_n_trials': 0,
            'best_category': None,
            'go_nogo_dprime': np.nan,
            'go_nogo_roc_auc': np.nan,
            'go_nogo_selective': False,
        }
    
    # Get unit data
    unit_data = event_windows_matrix[unit_idx, :, :]  # [time × events]
    
    # Find time indices for analysis window
    start_idx = np.argmin(np.abs(time_axis - window[0]))
    end_idx = np.argmin(np.abs(time_axis - window[1]))
    unit_data_windowed = unit_data[start_idx:end_idx, :]
    
    # Mean firing rate per trial (across time window)
    mean_rates = np.mean(unit_data_windowed, axis=0)  # [trials]
    
    # Get stimuli for each trial
    stimuli = stimuli_outcome_df['stimulus'].values
    
    # Assign categories
    categories = assign_stimulus_categories(stimuli, low_boundary, high_boundary)
    
    # Group firing rates by Go/NoGo
    go_rates = mean_rates[categories == "Go"]
    nogo_rates = mean_rates[categories == "NoGo"]
    
    # Require at least two trials per group for a meaningful test
    has_go = len(go_rates) >= 2
    has_nogo = len(nogo_rates) >= 2
    
    go_mean_rate = float(np.mean(go_rates)) if len(go_rates) > 0 else np.nan
    nogo_mean_rate = float(np.mean(nogo_rates)) if len(nogo_rates) > 0 else np.nan
    
    if not (has_go and has_nogo):
        return {
            'category_sensitive': False,
            'category_anova_p': np.nan,
            'category_anova_f': np.nan,
            'go_mean_rate': go_mean_rate,
            'nogo_mean_rate': nogo_mean_rate,
            'go_n_trials': len(go_rates),
            'nogo_n_trials': len(nogo_rates),
            'best_category': None,
            'go_nogo_dprime': np.nan,
            'go_nogo_roc_auc': np.nan,
            'go_nogo_selective': False,
        }
    
    # Two-group ANOVA (equivalent to two-sample t-test on means)
    f_stat, p_val = f_oneway(go_rates, nogo_rates)
    
    best_category = "Go" if go_mean_rate > nogo_mean_rate else "NoGo"
    
    # Compute d' and ROC AUC for Go vs NoGo
    go_nogo_dprime = np.nan
    go_nogo_roc_auc = np.nan
    go_nogo_selective = False
    
    # Safely compute statistics, handling empty or all-NaN arrays
    if len(go_rates) > 0 and len(nogo_rates) > 0:
        # Check for valid (non-NaN) values
        go_valid = go_rates[~np.isnan(go_rates)]
        nogo_valid = nogo_rates[~np.isnan(nogo_rates)]
        
        if len(go_valid) >= 2 and len(nogo_valid) >= 2:
            go_mean, go_std = np.mean(go_valid), np.std(go_valid)
            nogo_mean, nogo_std = np.mean(nogo_valid), np.std(nogo_valid)
            pooled_std = np.sqrt((go_std**2 + nogo_std**2) / 2)
            go_nogo_dprime = (go_mean - nogo_mean) / pooled_std if pooled_std > 0 else 0.0
            
            try:
                labels = np.concatenate([np.ones(len(go_valid)), np.zeros(len(nogo_valid))])
                scores = np.concatenate([go_valid, nogo_valid])
                go_nogo_roc_auc = roc_auc_score(labels, scores)
            except Exception:
                go_nogo_roc_auc = 0.5
            
            go_nogo_selective = abs(go_nogo_dprime) > 0.5
    
    return {
        'category_sensitive': p_val < 0.05,
        'category_anova_p': float(p_val),
        'category_anova_f': float(f_stat),
        'go_mean_rate': go_mean_rate,
        'nogo_mean_rate': nogo_mean_rate,
        'go_n_trials': len(go_rates),
        'nogo_n_trials': len(nogo_rates),
        'best_category': best_category,
        'go_nogo_dprime': float(go_nogo_dprime) if not np.isnan(go_nogo_dprime) else np.nan,
        'go_nogo_roc_auc': float(go_nogo_roc_auc) if not np.isnan(go_nogo_roc_auc) else np.nan,
        'go_nogo_selective': go_nogo_selective,
    }


def compute_category_sensitivity_for_all_units(
    event_windows_data: tuple,
    active_units: np.ndarray,
    low_boundary: float = 0.983,
    high_boundary: float = 1.525,
    window: tuple[float, float] = (-0.1, 0.5),
) -> pd.DataFrame:
    """
    Compute Go vs NoGo category sensitivity for all active units.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with category sensitivity metrics for each unit
    """
    results = []
    
    for unit_idx in active_units:
        unit_results = compute_category_sensitivity(
            event_windows_data,
            int(unit_idx),
            low_boundary=low_boundary,
            high_boundary=high_boundary,
            window=window,
        )
        unit_results['unit_idx'] = int(unit_idx)
        results.append(unit_results)
    
    return pd.DataFrame(results)


def plot_psth_by_category(
    event_windows_data: tuple,
    unit_idx: int,
    low_boundary: float = 0.983,
    high_boundary: float = 1.525,
    display_window: tuple[float, float] = (-0.5, 1.0),
    region_name: str = "Unit",
) -> go.Figure:
    """
    Plot PSTH separated by category (Go vs NoGo) for a single unit.
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
    
    # Get stimuli and assign categories
    stimuli = stimuli_outcome_df['stimulus'].values
    categories = assign_stimulus_categories(stimuli, low_boundary, high_boundary)
    
    # Create figure
    fig = go.Figure()
    
    # Category colors
    category_colors = {
        "Go": "#2ca02c",     # Green
        "NoGo": "#ff7f0e",   # Orange
    }
    
    category_order = ["Go", "NoGo"]
    
    for category in category_order:
        cat_mask = (categories == category)
        cat_trials = unit_data_windowed[:, cat_mask]
        
        if cat_trials.shape[1] > 0:
            # Safely compute mean and SEM, handling all-NaN cases
            valid_counts = np.sum(~np.isnan(cat_trials), axis=1)
            psth_mean = np.full(cat_trials.shape[0], np.nan)
            psth_sem = np.zeros(cat_trials.shape[0])
            
            valid_mask = valid_counts > 0
            if np.any(valid_mask):
                psth_mean[valid_mask] = np.nanmean(cat_trials[valid_mask, :], axis=1)
                # Only compute SEM where there are at least 2 valid samples
                sem_mask = valid_mask & (valid_counts >= 2)
                if np.any(sem_mask):
                    psth_sem[sem_mask] = np.nanstd(cat_trials[sem_mask, :], axis=1) / np.sqrt(valid_counts[sem_mask])
            
            color = category_colors[category]
            
            # Main trace
            fig.add_trace(go.Scatter(
                x=time_axis_windowed,
                y=psth_mean,
                mode='lines',
                name=f'{category} (n={cat_trials.shape[1]})',
                line=dict(color=color, width=2)
            ))
            
            # SEM shading
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
    
    # Add boundary lines annotation
    fig.add_annotation(
        x=0.02, y=0.98,
        xref='paper', yref='paper',
        text=(
            f'Go: <{low_boundary:.3f} or >{high_boundary:.3f}<br>'
            f'NoGo: between'
        ),
        showarrow=False,
        font=dict(size=10),
        bgcolor='rgba(255,255,255,0.8)',
        bordercolor='black',
        borderwidth=1,
        align='left'
    )
    
    fig.update_layout(
        title=f"{region_name} Unit {unit_idx} - PSTH by Category (Go vs NoGo)",
        xaxis_title="Time (s)",
        yaxis_title="Firing Rate (Hz)",
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig


def plot_category_sensitivity_summary(
    category_df: pd.DataFrame,
    region_name: str = "Region",
    low_boundary: float = 0.983,
    high_boundary: float = 1.525,
) -> go.Figure:
    """
    Create summary plot showing category sensitivity across units.
    """
    if category_df is None or len(category_df) == 0:
        fig = go.Figure()
        fig.update_layout(
            title=f"{region_name} - No category sensitivity data",
            template="plotly_white",
            height=400,
        )
        return fig
    
    fig = go.Figure()
    
    # Scatter plot: -log10(p-value) vs unit index
    valid_p = ~np.isnan(category_df['category_anova_p'])
    if np.any(valid_p):
        p_vals = category_df.loc[valid_p, 'category_anova_p']
        unit_indices = category_df.loc[valid_p, 'unit_idx']
        neg_log10_p = -np.log10(p_vals + 1e-10)
        
        # Color by significance
        colors = ['red' if p < 0.05 else 'gray' for p in p_vals]
        
        fig.add_trace(go.Scatter(
            x=unit_indices,
            y=neg_log10_p,
            mode='markers',
            name='Category Sensitivity (-log10 p)',
            marker=dict(
                color=colors,
                size=8,
                symbol='circle',
                line=dict(width=1, color='black')
            ),
            hovertemplate='Unit %{x}<br>-log10(p) = %{y:.2f}<br>p = %{customdata:.4f}<extra></extra>',
            customdata=p_vals
        ))
        
        # Add significance threshold
        fig.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="red", 
                     annotation_text="p=0.05", annotation_position="right")
    
    fig.update_layout(
        title=f"{region_name} - Category Sensitivity Summary (Boundaries: {low_boundary:.3f}, {high_boundary:.3f})",
        xaxis_title="Unit Index",
        yaxis_title="-log10(ANOVA p-value)",
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    return fig

