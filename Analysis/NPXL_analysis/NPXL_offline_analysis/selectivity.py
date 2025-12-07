"""
Selectivity metrics computation for NPXL offline analysis.
"""
import numpy as np
import pandas as pd
from Analysis.NPXL_analysis.single_unit_offline_analysis.config import npxl_single_unit_analysis
# Note: Unit is imported lazily inside the function to avoid circular import


def compute_selectivity_metrics_for_active_units(
    event_windows_data: tuple,
    active_units: np.ndarray,
    window: tuple[float, float] = (-0.1, 0.5),
    region_name: str = "Unknown",
    use_unit_class: bool = True,
) -> pd.DataFrame:
    """
    Compute stimulus selectivity, outcome modulation, go/nogo coding, and choice probability
    for all active units.
    
    Parameters:
    -----------
    event_windows_data : tuple
        Event windows data tuple
    active_units : np.ndarray
        Array of active unit indices
    window : tuple[float, float]
        Time window for analysis (start, end) in seconds
    region_name : str
        Name of the brain region
    use_unit_class : bool
        If True, use Unit class internally (cleaner, with caching).
        If False, use direct computation (faster for one-time batch processing).
    
    Returns:
        pd.DataFrame with columns: unit_idx, stimulus_selective, outcome_p_value, 
        go_nogo_dprime, go_nogo_roc_auc, choice_probability, choice_probability_corr
    """
    if use_unit_class:
        # Use Unit class for cleaner code and caching
        # Lazy import to avoid circular dependency
        from Analysis.NPXL_analysis.single_unit_offline_analysis.unit import Unit
        
        results = []
        for unit_idx in active_units:
            unit = Unit(unit_idx, event_windows_data, region_name=region_name)
            selectivity = unit.compute_selectivity(window=window)
            selectivity["unit_idx"] = unit_idx
            results.append(selectivity)
        return pd.DataFrame(results)
    
    # Original implementation for direct computation (slightly faster for one-time use)
    # Unpack event_windows_data (may be 5 or 6 elements)
    if len(event_windows_data) == 6:
        event_windows_matrix, time_axis, valid_event_indices, stimuli_outcome_df, metadata, _ = event_windows_data
        # Create 5-tuple for analysis functions that expect it
        event_windows_data_5 = (event_windows_matrix, time_axis, valid_event_indices, stimuli_outcome_df, metadata)
    else:
        event_windows_data_5 = event_windows_data
        event_windows_matrix, time_axis, valid_event_indices, stimuli_outcome_df, metadata = event_windows_data
    
    results = []
    
    for unit_idx in active_units:
        unit_results = {"unit_idx": int(unit_idx)}
        
        # Stimulus selectivity
        if 'stimulus' in stimuli_outcome_df.columns:
            unique_stimuli, tuning_curve, tuning_sem, best_stimulus = npxl_single_unit_analysis.compute_stimulus_selectivity(
                event_windows_data_5, stimuli_outcome_df, int(unit_idx), window=window
            )
            if tuning_curve is not None and len(tuning_curve) > 1:
                # Test if there's significant variation across stimuli (ANOVA-like)
                # Simple test: check if max - min > 2*SEM
                max_response = np.max(tuning_curve)
                min_response = np.min(tuning_curve)
                max_sem = np.max(tuning_sem) if len(tuning_sem) > 0 else 0
                stimulus_selective = (max_response - min_response) > (2 * max_sem)
                unit_results["stimulus_selective"] = stimulus_selective
                unit_results["best_stimulus"] = best_stimulus
                unit_results["max_stimulus_response"] = max_response
                unit_results["min_stimulus_response"] = min_response
            else:
                unit_results["stimulus_selective"] = False
                unit_results["best_stimulus"] = None
        else:
            unit_results["stimulus_selective"] = False
            unit_results["best_stimulus"] = None
        
        # Outcome modulation
        if 'outcome' in stimuli_outcome_df.columns:
            outcome_p, outcome_rates, outcome_means = npxl_single_unit_analysis.compute_outcome_modulation(
                event_windows_data_5, stimuli_outcome_df, int(unit_idx), window=window
            )
            if outcome_p is not None:
                unit_results["outcome_p_value"] = float(outcome_p)
                unit_results["outcome_modulated"] = outcome_p < 0.05
                if outcome_means is not None:
                    unit_results["rewarded_mean_rate"] = float(outcome_means[0])
                    unit_results["non_rewarded_mean_rate"] = float(outcome_means[1])
            else:
                unit_results["outcome_p_value"] = np.nan
                unit_results["outcome_modulated"] = False
        else:
            unit_results["outcome_p_value"] = np.nan
            unit_results["outcome_modulated"] = False
        
        # Go/NoGo coding
        if 'outcome' in stimuli_outcome_df.columns:
            go_nogo_dprime, go_nogo_roc_auc, go_nogo_rates = npxl_single_unit_analysis.compute_go_nogo_coding(
                event_windows_data_5, stimuli_outcome_df, int(unit_idx), window=window
            )
            if go_nogo_dprime is not None:
                unit_results["go_nogo_dprime"] = float(go_nogo_dprime)
                unit_results["go_nogo_roc_auc"] = float(go_nogo_roc_auc)
                unit_results["go_nogo_selective"] = abs(go_nogo_dprime) > 0.5  # Threshold for selectivity
            else:
                unit_results["go_nogo_dprime"] = np.nan
                unit_results["go_nogo_roc_auc"] = np.nan
                unit_results["go_nogo_selective"] = False
        else:
            unit_results["go_nogo_dprime"] = np.nan
            unit_results["go_nogo_roc_auc"] = np.nan
            unit_results["go_nogo_selective"] = False
        
        # Choice probability
        if 'outcome' in stimuli_outcome_df.columns:
            cp, cp_corr = npxl_single_unit_analysis.compute_choice_probability(
                event_windows_data_5, stimuli_outcome_df, int(unit_idx), window=window
            )
            if cp is not None:
                unit_results["choice_probability"] = float(cp)
                unit_results["choice_probability_corr"] = float(cp_corr)
                unit_results["choice_coding"] = abs(cp_corr) > 0.1  # Threshold for choice coding
            else:
                unit_results["choice_probability"] = np.nan
                unit_results["choice_probability_corr"] = np.nan
                unit_results["choice_coding"] = False
        else:
            unit_results["choice_probability"] = np.nan
            unit_results["choice_probability_corr"] = np.nan
            unit_results["choice_coding"] = False
        
        results.append(unit_results)
    
    return pd.DataFrame(results)

