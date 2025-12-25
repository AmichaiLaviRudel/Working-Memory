"""
Design matrix building functions for GLM analysis.
"""
from typing import Tuple, List, Dict, Optional

import numpy as np
import pandas as pd
import pynapple as nap
import nemos as nmo
from patsy import dmatrix


# ============================================================================
# Categorical Feature Processing
# ============================================================================

def create_categorical_features(
    tone_onset: np.ndarray,
    stimuli_outcome_df: pd.DataFrame,
    preprocessing_bin_size: float,
    low_boundary: float,
    high_boundary: float
) -> Tuple[nap.TsdFrame, pd.DataFrame]:
    """
    Create categorical features from stimuli/outcome data.
    
    Parameters
    ----------
    tone_onset : np.ndarray
        Tone onset times
    stimuli_outcome_df : pd.DataFrame
        DataFrame with stimulus and outcome information
    preprocessing_bin_size : float
        Bin size for preprocessing
    low_boundary : float
        Lower threshold for Go/NoGo categorization
    high_boundary : float
        Upper threshold for Go/NoGo categorization
        
    Returns
    -------
    categorical_features : nap.TsdFrame
        Categorical features at tone onset times
    data : pd.DataFrame
        Original data DataFrame
    """
    stimulus = np.round(stimuli_outcome_df["stimulus"].values.astype(float), 2)
    outcome_str = stimuli_outcome_df["outcome"].astype(str).str.lower().values
    outcome_time_bins = stimuli_outcome_df['outcome_time'].values.astype(float)
    outcome_time = np.nan_to_num((outcome_time_bins) * preprocessing_bin_size + tone_onset, nan=0)
    previous_outcome = np.roll(outcome_str, 1)
    previous_outcome[0] = '0'
    
    # Category: Go (1) vs NoGo (0)
    category = np.where((stimulus < low_boundary) | (stimulus > high_boundary), 'Go', 'NoGo')
    
    # Create DataFrame
    data = pd.DataFrame({
        "stimulus_ID": stimulus,
        "category_ID": category,
        "outcome_ID": outcome_str,
        "previous_outcome": previous_outcome
    })
    
    # Create design matrix using patsy
    formula = "C(stimulus_ID) + category_ID + outcome_ID + previous_outcome"
    categorical_design_matrix = dmatrix(formula, data, return_type="dataframe")
    categorical_design_matrix.drop(columns=["Intercept"], inplace=True)
    
    categorical_features = nap.TsdFrame(
        t=tone_onset,
        d=categorical_design_matrix.values,
        columns=categorical_design_matrix.columns,
    )
    
    return categorical_features, data, outcome_time


def create_temporal_features(
    tone_onset: np.ndarray,
    licks: np.ndarray,
    outcome_time: np.ndarray,
    bin_size: float
) -> nap.TsdFrame:
    """
    Create temporal event features (binned counts).
    
    Parameters
    ----------
    tone_onset : np.ndarray
        Tone onset times
    licks : np.ndarray
        Lick times
    outcome_time : np.ndarray
        Outcome times
    bin_size : float
        Bin size in seconds
        
    Returns
    -------
    temporal_features : nap.TsdFrame
        Binned temporal features
    """
    # Find max time
    max_time = float(np.nanmax([
        np.nanmax(tone_onset),
        np.nanmax(licks),
        np.nanmax(outcome_time)
    ]))
    
    full_ep = nap.IntervalSet(start=0.0, end=max_time)
    
    # Create binned counts
    tone_onset_count = nap.Ts(tone_onset.astype(float)).count(bin_size, ep=full_ep)
    licks_count = nap.Ts(licks.astype(float)).count(bin_size, ep=full_ep)
    outcome_time_count = nap.Ts(outcome_time.astype(float)).count(bin_size, ep=full_ep)
    
    # Stack counts
    data = np.column_stack([tone_onset_count.d, licks_count.d, outcome_time_count.d])
    
    temporal_features = nap.TsdFrame(
        t=tone_onset_count.t,
        d=data,
        columns=["tone_onset", "licks", "outcome_onset"],
    )
    
    return temporal_features, full_ep


# ============================================================================
# Basis Functions
# ============================================================================

def create_basis_functions(
    n_basis_funcs: int,
    event_window_sec: float,
    acausal_before_sec: float,
    acausal_after_sec: float,
    history_window_sec: float,
    temporal_features_rate: float
) -> Tuple:
    """
    Create basis function objects for feature convolution.
    
    Parameters
    ----------
    n_basis_funcs : int
        Number of basis functions
    event_window_sec : float
        Window size for temporal events
    acausal_before_sec : float
        Acausal window before event
    acausal_after_sec : float
        Acausal window after event
    history_window_sec : float
        Window for spike history
    temporal_features_rate : float
        Sampling rate of temporal features
        
    Returns
    -------
    basis_events : nmo.basis.RaisedCosineLogConv
        Basis for temporal events
    basis_categorical : nmo.basis.RaisedCosineLinearConv
        Basis for categorical features
    basis_history : nmo.basis.RaisedCosineLogConv
        Basis for spike history
    """
    # Causal basis for temporal features
    event_window_bins = int(event_window_sec * temporal_features_rate)
    basis_events = nmo.basis.RaisedCosineLogConv(
        n_basis_funcs=n_basis_funcs,
        window_size=event_window_bins,
        label="temporal_events"
    )
    print(f"Temporal event basis (causal): {n_basis_funcs} functions over {event_window_sec}s window")
    
    # Acausal basis for categorical features
    acausal_total_sec = acausal_before_sec + acausal_after_sec
    acausal_window_bins = int(acausal_total_sec * temporal_features_rate)
    basis_categorical = nmo.basis.RaisedCosineLinearConv(
        n_basis_funcs=n_basis_funcs,
        window_size=acausal_window_bins,
        label="categorical_events"
    )
    print(f"Categorical basis (acausal): {n_basis_funcs} functions over {acausal_total_sec}s window")
    
    # Causal basis for spike history
    history_window_bins = int(history_window_sec * temporal_features_rate)
    basis_history = nmo.basis.RaisedCosineLogConv(
        n_basis_funcs=n_basis_funcs,
        window_size=history_window_bins,
        label="spike_history"
    )
    
    return basis_events, basis_categorical, basis_history, event_window_bins, acausal_window_bins, history_window_bins


def create_categorical_impulses(
    categorical_features: nap.TsdFrame,
    temporal_features: nap.TsdFrame,
    tone_onset: np.ndarray
) -> nap.TsdFrame:
    """
    Convert categorical features to impulse time series at tone onset.
    
    Parameters
    ----------
    categorical_features : nap.TsdFrame
        Categorical features at tone times
    temporal_features : nap.TsdFrame
        Temporal features (for time base)
    tone_onset : np.ndarray
        Tone onset times
        
    Returns
    -------
    categorical_impulse_tsd : nap.TsdFrame
        Categorical impulses at binned time points
    """
    tone_times = tone_onset.astype(float)
    t_bins = np.asarray(temporal_features.t, dtype=float)
    
    # Find nearest bin for each tone onset
    tone_idx = np.searchsorted(t_bins, tone_times, side="left")
    tone_idx = np.clip(tone_idx, 0, len(t_bins) - 1)
    left_idx = np.clip(tone_idx - 1, 0, len(t_bins) - 1)
    pick_left = np.abs(t_bins[left_idx] - tone_times) < np.abs(t_bins[tone_idx] - tone_times)
    tone_idx[pick_left] = left_idx[pick_left]
    
    # Create impulse time series
    categorical_impulse = np.zeros((len(t_bins), categorical_features.shape[1]), dtype=float)
    for i, idx in enumerate(tone_idx):
        if i < len(categorical_features):
            categorical_impulse[idx, :] += categorical_features.values[i, :]
    
    categorical_impulse_tsd = nap.TsdFrame(
        t=temporal_features.t,
        d=categorical_impulse,
        columns=categorical_features.columns
    )
    
    return categorical_impulse_tsd


# ============================================================================
# Design Matrix Building
# ============================================================================

def build_design_matrix(
    temporal_features: nap.TsdFrame,
    categorical_impulse_tsd: nap.TsdFrame,
    neuron_count: nap.Tsd,
    basis_events,
    basis_categorical,
    basis_history,
    n_basis_funcs: int
) -> Tuple[nap.TsdFrame, List[int]]:
    """
    Build complete design matrix with convolved features.
    
    Parameters
    ----------
    temporal_features : nap.TsdFrame
        Temporal features
    categorical_impulse_tsd : nap.TsdFrame
        Categorical impulse features
    neuron_count : nap.Tsd
        Spike count for a single neuron
    basis_events : nmo.basis
        Basis for temporal events
    basis_categorical : nmo.basis
        Basis for categorical features
    basis_history : nmo.basis
        Basis for spike history
    n_basis_funcs : int
        Number of basis functions
        
    Returns
    -------
    X : nap.TsdFrame
        Design matrix
    hist_feature_indices : list
        Indices of spike history features
    """
    # Convolve features with basis functions
    X_temporal_conv = basis_events.compute_features(temporal_features)
    X_categorical_conv = basis_categorical.compute_features(categorical_impulse_tsd)
    X_history = basis_history.compute_features(neuron_count)
    
    # Find common time support
    common_support = X_temporal_conv.time_support.intersect(
        X_categorical_conv.time_support
    ).intersect(X_history.time_support)
    
    # Restrict to common support
    X_temp_common = X_temporal_conv.restrict(common_support)
    X_cat_common = X_categorical_conv.restrict(common_support)
    X_hist_common = X_history.restrict(common_support)
    
    # Verify alignment
    assert X_temp_common.shape[0] == X_cat_common.shape[0] == X_hist_common.shape[0], \
        "Predictor time bases don't align after restriction to common support"
    
    # Build column names
    temp_cols = []
    for temp_feat in temporal_features.columns:
        for basis_idx in range(n_basis_funcs):
            temp_cols.append(f"{temp_feat}_basis{basis_idx}")
    
    cat_cols = []
    for cat_feat in categorical_impulse_tsd.columns:
        for basis_idx in range(n_basis_funcs):
            cat_cols.append(f"{cat_feat}_basis{basis_idx}")
    
    hist_cols = [f"spike_history_basis{i}" for i in range(basis_history.n_basis_funcs)]
    
    # Combine
    X = nap.TsdFrame(
        t=X_temp_common.t,
        d=np.column_stack([
            X_temp_common.values,
            X_cat_common.values,
            X_hist_common.values,
        ]),
        columns=temp_cols + cat_cols + hist_cols
    )
    
    # Remove NaN rows
    valid_mask = np.all(np.isfinite(X.values), axis=1)
    X = nap.TsdFrame(
        t=X.t[valid_mask],
        d=X.values[valid_mask],
        columns=X.columns
    )
    
    # Get spike history indices
    hist_feature_indices = [i for i, col in enumerate(X.columns) if 'spike_history' in col]
    
    print(f"Combined design matrix shape: {X.shape}")
    
    return X, hist_feature_indices


def build_population_design_matrix(
    X_shared: nap.TsdFrame,
    spike_count_population: nap.TsdFrame,
    spike_count: nap.TsdFrame,
    population_ids: np.ndarray,
    epochs: nap.IntervalSet,
    include_spike_history: bool,
    n_basis_funcs: int,
    history_acausal_before_sec: float,
    history_acausal_after_sec: float
) -> Tuple[nap.TsdFrame, int, List[int]]:
    """
    Build population design matrix with optional all-to-all spike history.
    
    Parameters
    ----------
    X_shared : nap.TsdFrame
        Shared predictors (temporal + categorical)
    spike_count_population : nap.TsdFrame
        Population spike counts
    spike_count : nap.TsdFrame
        Full spike count matrix
    population_ids : np.ndarray
        IDs of neurons in population
    epochs : nap.IntervalSet
        Epochs for restriction
    include_spike_history : bool
        Whether to include spike history features
    n_basis_funcs : int
        Number of basis functions
    history_acausal_before_sec : float
        Acausal window before spike
    history_acausal_after_sec : float
        Acausal window after spike
        
    Returns
    -------
    X_ep_pop : nap.TsdFrame
        Population design matrix
    n_history_features : int
        Number of history features
    history_feature_indices : list
        Indices of history features
    """
    X_shared_ep = X_shared.restrict(epochs)
    
    print(f"\nBuilding population design matrix:")
    print(f"  Shared predictors (temporal + categorical): {X_shared_ep.shape[1]} features")
    print(f"  Include spike history: {include_spike_history}")
    
    if include_spike_history:
        print(f"  Creating ALL-TO-ALL spike history connectivity for {len(population_ids)} neurons...")
        
        # Create acausal basis for spike history
        history_acausal_total_sec = history_acausal_before_sec + history_acausal_after_sec
        history_acausal_window_bins = int(history_acausal_total_sec * spike_count_population.rate)
        
        basis_history_pop = nmo.basis.RaisedCosineLinearConv(
            n_basis_funcs=n_basis_funcs,
            window_size=history_acausal_window_bins,
            label="spike_history_acausal"
        )
        
        print(f"  Acausal spike history basis: {n_basis_funcs} functions over {history_acausal_total_sec}s window")
        
        history_features_list = []
        for uid in population_ids:
            uid_idx = list(spike_count.columns).index(uid)
            neuron_spikes = spike_count[:, uid_idx]
            
            neuron_history = basis_history_pop.compute_features(neuron_spikes)
            neuron_history_ep = neuron_history.restrict(epochs)
            
            # Align timestamps
            if neuron_history_ep.shape[0] != X_shared_ep.shape[0]:
                hist_indices = np.searchsorted(neuron_history_ep.t, X_shared_ep.t)
                hist_indices = np.clip(hist_indices, 0, len(neuron_history_ep.t) - 1)
                neuron_history_aligned = neuron_history_ep.values[hist_indices]
            else:
                neuron_history_aligned = neuron_history_ep.values
            
            history_cols = [f"hist_from_neuron{uid}_basis{i}" for i in range(basis_history_pop.n_basis_funcs)]
            history_features_list.append((neuron_history_aligned, history_cols))
        
        # Combine all history features
        all_history_features = np.column_stack([hist for hist, _ in history_features_list])
        all_history_cols = [col for _, cols in history_features_list for col in cols]
        
        print(f"  All-to-all connectivity: {len(population_ids)} source neurons × {basis_history_pop.n_basis_funcs} basis = {all_history_features.shape[1]} history features")
        
        X_pop_combined = np.column_stack([X_shared_ep.values, all_history_features])
        X_pop_cols = list(X_shared_ep.columns) + all_history_cols
        n_history_features = all_history_features.shape[1]
    else:
        print(f"  Skipping spike history features")
        X_pop_combined = X_shared_ep.values
        X_pop_cols = list(X_shared_ep.columns)
        n_history_features = 0
    
    # Remove NaN rows
    valid_mask = np.all(np.isfinite(X_pop_combined), axis=1)
    X_ep_pop = nap.TsdFrame(
        t=X_shared_ep.t[valid_mask],
        d=X_pop_combined[valid_mask],
        columns=X_pop_cols
    )
    
    # Get history feature indices
    history_feature_indices = [i for i, col in enumerate(X_ep_pop.columns) if 'hist_from_neuron' in col]
    
    print(f"  Total features: {X_ep_pop.shape[1]} (shared: {X_shared_ep.shape[1]}, history: {n_history_features})")
    
    return X_ep_pop, n_history_features, history_feature_indices
