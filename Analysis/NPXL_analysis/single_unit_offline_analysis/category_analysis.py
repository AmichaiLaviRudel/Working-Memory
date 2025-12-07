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
    Assign each stimulus to a category based on boundaries.
    
    Categories:
    - 'Low': stimulus < low_boundary
    - 'Middle': low_boundary <= stimulus <= high_boundary
    - 'High': stimulus > high_boundary
    
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
        Array of category labels ('Low', 'Middle', 'High')
    """
    categories = np.full(len(stimuli), 'Middle', dtype=object)
    categories[stimuli < low_boundary] = 'Low'
    categories[stimuli > high_boundary] = 'High'
    return categories


def compute_category_sensitivity(
    event_windows_data: tuple,
    unit_idx: int,
    low_boundary: float = 0.983,
    high_boundary: float = 1.525,
    window: tuple[float, float] = (-0.2, 1),
) -> dict:
    """
    Test if a unit is sensitive to stimulus categories defined by boundaries.
    
    Uses one-way ANOVA to test for differences in firing rates across categories.
    
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
            'low_mean_rate': np.nan,
            'middle_mean_rate': np.nan,
            'high_mean_rate': np.nan,
            'best_category': None,
            'go_nogo_dprime': np.nan,
            'go_nogo_roc_auc': np.nan,
            'go_nogo_selective': False,
            'go_mean_rate': np.nan,
            'nogo_mean_rate': np.nan,
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
    
    # Group firing rates by category
    low_rates = mean_rates[categories == 'Low']
    middle_rates = mean_rates[categories == 'Middle']
    high_rates = mean_rates[categories == 'High']
    
    # Check if we have enough data in each category (at least 2 trials)
    has_low = len(low_rates) >= 2
    has_middle = len(middle_rates) >= 2
    has_high = len(high_rates) >= 2
    
    # Perform ANOVA if we have at least 2 categories with sufficient data
    category_groups = []
    category_names = []
    if has_low:
        category_groups.append(low_rates)
        category_names.append('Low')
    if has_middle:
        category_groups.append(middle_rates)
        category_names.append('Middle')
    if has_high:
        category_groups.append(high_rates)
        category_names.append('High')
    
    if len(category_groups) < 2:
        # Not enough categories for ANOVA, but still compute go/nogo if possible
        go_rates = np.concatenate([high_rates, low_rates]) if (has_high and has_low) else (
            high_rates if has_high else (low_rates if has_low else np.array([]))
        )
        nogo_rates = middle_rates if has_middle else np.array([])
        
        go_nogo_dprime = np.nan
        go_nogo_roc_auc = np.nan
        go_nogo_selective = False
        
        if len(go_rates) >= 2 and len(nogo_rates) >= 2:
            go_mean, go_std = np.mean(go_rates), np.std(go_rates)
            nogo_mean, nogo_std = np.mean(nogo_rates), np.std(nogo_rates)
            pooled_std = np.sqrt((go_std**2 + nogo_std**2) / 2)
            go_nogo_dprime = (go_mean - nogo_mean) / pooled_std if pooled_std > 0 else 0.0
            
            try:
                labels = np.concatenate([np.ones(len(go_rates)), np.zeros(len(nogo_rates))])
                scores = np.concatenate([go_rates, nogo_rates])
                go_nogo_roc_auc = roc_auc_score(labels, scores)
            except Exception:
                go_nogo_roc_auc = 0.5
            
            go_nogo_selective = abs(go_nogo_dprime) > 0.5
        
        return {
            'category_sensitive': False,
            'category_anova_p': np.nan,
            'category_anova_f': np.nan,
            'low_mean_rate': float(np.mean(low_rates)) if has_low else np.nan,
            'middle_mean_rate': float(np.mean(middle_rates)) if has_middle else np.nan,
            'high_mean_rate': float(np.mean(high_rates)) if has_high else np.nan,
            'best_category': None,
            'go_nogo_dprime': float(go_nogo_dprime) if not np.isnan(go_nogo_dprime) else np.nan,
            'go_nogo_roc_auc': float(go_nogo_roc_auc) if not np.isnan(go_nogo_roc_auc) else np.nan,
            'go_nogo_selective': go_nogo_selective,
            'go_mean_rate': float(np.mean(go_rates)) if len(go_rates) > 0 else np.nan,
            'nogo_mean_rate': float(np.mean(nogo_rates)) if len(nogo_rates) > 0 else np.nan,
        }
    
    # Perform one-way ANOVA
    f_stat, p_val = f_oneway(*category_groups)
    
    # Determine best category (highest mean rate)
    mean_rates_by_category = {}
    if has_low:
        mean_rates_by_category['Low'] = np.mean(low_rates)
    if has_middle:
        mean_rates_by_category['Middle'] = np.mean(middle_rates)
    if has_high:
        mean_rates_by_category['High'] = np.mean(high_rates)
    
    best_category = max(mean_rates_by_category, key=mean_rates_by_category.get) if mean_rates_by_category else None
    
    # Compute Go/NoGo selectivity: Go = High + Low, NoGo = Middle
    go_rates = np.concatenate([high_rates, low_rates]) if (has_high and has_low) else (
        high_rates if has_high else (low_rates if has_low else np.array([]))
    )
    nogo_rates = middle_rates if has_middle else np.array([])
    
    # Compute d' and ROC AUC for Go vs NoGo
    go_nogo_dprime = np.nan
    go_nogo_roc_auc = np.nan
    go_nogo_selective = False
    
    if len(go_rates) >= 2 and len(nogo_rates) >= 2:
        # Compute d'
        go_mean, go_std = np.mean(go_rates), np.std(go_rates)
        nogo_mean, nogo_std = np.mean(nogo_rates), np.std(nogo_rates)
        
        # Pooled standard deviation
        pooled_std = np.sqrt((go_std**2 + nogo_std**2) / 2)
        go_nogo_dprime = (go_mean - nogo_mean) / pooled_std if pooled_std > 0 else 0.0
        
        # Compute ROC AUC
        try:
            # Create labels: 1 for Go, 0 for NoGo
            labels = np.concatenate([np.ones(len(go_rates)), np.zeros(len(nogo_rates))])
            scores = np.concatenate([go_rates, nogo_rates])
            go_nogo_roc_auc = roc_auc_score(labels, scores)
        except Exception:
            go_nogo_roc_auc = 0.5
        
        # Consider selective if |d'| > 0.5 (same threshold as elsewhere in codebase)
        go_nogo_selective = abs(go_nogo_dprime) > 0.5
    
    return {
        'category_sensitive': p_val < 0.05,
        'category_anova_p': float(p_val),
        'category_anova_f': float(f_stat),
        'low_mean_rate': float(np.mean(low_rates)) if has_low else np.nan,
        'middle_mean_rate': float(np.mean(middle_rates)) if has_middle else np.nan,
        'high_mean_rate': float(np.mean(high_rates)) if has_high else np.nan,
        'low_n_trials': len(low_rates),
        'middle_n_trials': len(middle_rates),
        'high_n_trials': len(high_rates),
        'best_category': best_category,
        'go_nogo_dprime': float(go_nogo_dprime) if not np.isnan(go_nogo_dprime) else np.nan,
        'go_nogo_roc_auc': float(go_nogo_roc_auc) if not np.isnan(go_nogo_roc_auc) else np.nan,
        'go_nogo_selective': go_nogo_selective,
        'go_mean_rate': float(np.mean(go_rates)) if len(go_rates) > 0 else np.nan,
        'nogo_mean_rate': float(np.mean(nogo_rates)) if len(nogo_rates) > 0 else np.nan,
    }


def compute_category_sensitivity_for_all_units(
    event_windows_data: tuple,
    active_units: np.ndarray,
    low_boundary: float = 0.983,
    high_boundary: float = 1.525,
    window: tuple[float, float] = (-0.1, 0.5),
) -> pd.DataFrame:
    """
    Compute category sensitivity for all active units.
    
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
    Plot PSTH separated by category (Low/Middle/High) for a single unit.
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
        'Low': '#2ca02c',      # Green
        'Middle': '#ff7f0e',   # Orange
        'High': '#d62728',     # Red
    }
    
    category_order = ['Low', 'Middle', 'High']
    
    for category in category_order:
        cat_mask = (categories == category)
        cat_trials = unit_data_windowed[:, cat_mask]
        
        if cat_trials.shape[1] > 0:
            psth_mean = np.mean(cat_trials, axis=1)
            psth_sem = np.std(cat_trials, axis=1) / np.sqrt(cat_trials.shape[1])
            
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
        text=f'Boundaries: Low={low_boundary:.3f}, High={high_boundary:.3f}',
        showarrow=False,
        font=dict(size=10),
        bgcolor='rgba(255,255,255,0.8)',
        bordercolor='black',
        borderwidth=1,
        align='left'
    )
    
    fig.update_layout(
        title=f"{region_name} Unit {unit_idx} - PSTH by Category",
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

