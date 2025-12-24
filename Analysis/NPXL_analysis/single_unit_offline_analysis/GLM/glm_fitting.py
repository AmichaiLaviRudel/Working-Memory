"""
Generalized Linear Model (GLM) fitting for neural data analysis.

This module implements GLM fitting for single units with behavioral predictors:
1. Cue Onset (tone/stimulus presentation)
2. Stimulus (frequency/category)
3. Category (Go vs NoGo trial type)
4. First lick timing
5. In-trial lick timing
6. Reward delivery
7. Punishment delivery
8. Previous trial outcome
"""
import os
import sys
from typing import Optional, Tuple, Dict, Any, List

# Add workspace root to Python path so absolute imports under `Analysis`
# work when running this file directly.
current_dir = os.path.dirname(os.path.abspath(__file__))
# GLM -> single_unit_offline_analysis -> NPXL_analysis -> Analysis -> DB (workspace root)
workspace_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
)
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

import numpy as np
import pandas as pd


import nemos as nmo  # NeMoS: JAX-based GLM framework for neuroscience

from Analysis.NPXL_analysis.single_unit_offline_analysis.GLM.design_matrix import (
    build_trial_design_matrix,
)


def build_nemos_time_series_inputs(
    unit_data: np.ndarray,
    time_axis: np.ndarray,
    design_matrix: pd.DataFrame,
    bin_size: float,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Convert per-trial spike data and behavioral design into flattened
    time-series inputs for NeMoS GLM fitting.

    This keeps the behavioral semantics (Go/NoGo, reward vs punishment,
    previous outcome, lick timing) while moving to a time-resolved model.
    """
    n_time, n_trials = unit_data.shape
    if n_trials != len(design_matrix):
        raise ValueError(
            f"unit_data has {n_trials} trials, but design_matrix has {len(design_matrix)} rows."
        )

    # Flatten spike counts across trials: [n_time * n_trials]
    spike_counts_flat: np.ndarray = unit_data.T.reshape(-1)

    n_total: int = n_time * n_trials
    stimulus = np.zeros(n_total, dtype=float)      # impulse at stimulus bin
    category_go = np.zeros(n_total, dtype=float)   # 1=Go, 0=NoGo
    reward = np.zeros(n_total, dtype=float)
    punishment = np.zeros(n_total, dtype=float)
    prev_reward = np.zeros(n_total, dtype=float)
    licks = np.zeros(n_total, dtype=float)

    for tr in range(n_trials):
        row = design_matrix.iloc[tr]
        base_idx: int = tr * n_time

        # Stimulus event
        stim_bin = int(row["stimulus_bin"])
        if 0 <= stim_bin < n_time:
            stimulus[base_idx + stim_bin] = 1.0

        # Category: constant during the trial
        category_val = float(row["stimulus_category"])
        category_go[base_idx: base_idx + n_time] = category_val

        # Current outcome: reward vs punishment
        out_code = int(row["outcome_type"])
        if out_code == 1:  # Hit
            reward[base_idx: base_idx + n_time] = 1.0
        elif out_code == 4:  # False Alarm
            punishment[base_idx: base_idx + n_time] = 1.0

        # Previous reward flag (code 1)
        if not pd.isna(row["previous_outcome"]) and int(row["previous_outcome"]) == 1:
            prev_reward[base_idx: base_idx + n_time] = 1.0

        # Licks: encode as rate (1 / bin_size) at each lick bin
        licks_bins = row["licks_bins"]
        if isinstance(licks_bins, tuple) and len(licks_bins) > 1:
            time_indices = licks_bins[1]
            valid_indices = time_indices[time_indices < n_time]
            if len(valid_indices) > 0:
                licks[base_idx + valid_indices] = 1.0 / bin_size

    predictors: Dict[str, np.ndarray] = {
        "stimulus": stimulus,
        "category_go": category_go,
        "reward": reward,
        "punishment": punishment,
        "prev_reward": prev_reward,
        "licks": licks,
    }
    return spike_counts_flat, predictors


def make_nemos_bases(
    bin_size: float,
    stimulus_kernel_config: Optional[Dict[str, Any]] = None,
    reward_punishment_kernel_config: Optional[Dict[str, Any]] = None,
    lick_kernel_config: Optional[Dict[str, Any]] = None,
    spike_history_kernel_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create NeMoS basis objects for each predictor using configurations
    analogous to the legacy raised-cosine kernels.
    """
    if stimulus_kernel_config is None:
        stimulus_kernel_config = {"n_basis": 8, "window": (0.0, 0.1)}
    if reward_punishment_kernel_config is None:
        reward_punishment_kernel_config = {"n_basis": 8, "window": (0.0, 0.6)}
    if lick_kernel_config is None:
        lick_kernel_config = {"n_basis": 7, "window": (-0.2, 0.2)}
    if spike_history_kernel_config is None:
        spike_history_kernel_config = {"n_basis": 5, "window": (0.001, 0.1)}

    def window_to_bins(window: Tuple[float, float]) -> int:
        # Convert time window (seconds) to window size in bins
        return max(1, int((window[1] - window[0]) / bin_size))

    bases: Dict[str, Any] = {}

    # Stimulus and category predictors (similar temporal structure)
    bases["stimulus"] = nmo.basis.RaisedCosineLogConv(
        n_basis_funcs=stimulus_kernel_config["n_basis"],
        window_size=window_to_bins(stimulus_kernel_config["window"]),
    )
    bases["category_go"] = nmo.basis.RaisedCosineLogConv(
        n_basis_funcs=stimulus_kernel_config["n_basis"],
        window_size=window_to_bins(stimulus_kernel_config["window"]),
    )

    # Reward and punishment events
    bases["reward"] = nmo.basis.RaisedCosineLogConv(
        n_basis_funcs=reward_punishment_kernel_config["n_basis"],
        window_size=window_to_bins(reward_punishment_kernel_config["window"]),
    )
    bases["punishment"] = nmo.basis.RaisedCosineLogConv(
        n_basis_funcs=reward_punishment_kernel_config["n_basis"],
        window_size=window_to_bins(reward_punishment_kernel_config["window"]),
    )

    # Licks around event times
    bases["licks"] = nmo.basis.RaisedCosineLogConv(
        n_basis_funcs=lick_kernel_config["n_basis"],
        window_size=window_to_bins(lick_kernel_config["window"]),
    )

    # Spike history kernel
    bases["spike_history"] = nmo.basis.RaisedCosineLogConv(
        n_basis_funcs=spike_history_kernel_config["n_basis"],
        window_size=window_to_bins(spike_history_kernel_config["window"]),
    )

    return bases


def build_nemos_feature_matrix(
    spike_counts_flat: np.ndarray,
    predictors: Dict[str, np.ndarray],
    bases: Dict[str, Any],
) -> Tuple[np.ndarray, List[str]]:
    """
    Build NeMoS feature matrix by applying basis objects to predictor
    time series and concatenating the resulting basis-expanded features.

    Returns:
        X: [n_time_total × n_features]
        feature_names: list of basis-expanded feature names.
    """
    n_time = spike_counts_flat.shape[0]
    X_blocks: List[np.ndarray] = []
    feature_names: List[str] = []

    # Primary predictors
    for name in ["stimulus", "category_go", "reward", "punishment", "prev_reward", "licks"]:
        if name not in predictors or name not in bases:
            continue
        series = predictors[name].reshape(-1, 1)  # [n_time × 1]
        basis = bases[name]

        # Each basis produces [n_time × n_basis_funcs]
        X_block = basis.compute_features(series)
        X_blocks.append(X_block)

        for b_idx in range(X_block.shape[1]):
            feature_names.append(f"{name}_basis{b_idx}")

    # Spike history uses the spike train itself as input
    if "spike_history" in bases:
        spike_series = spike_counts_flat.reshape(-1, 1)
        hist_basis = bases["spike_history"]
        X_hist = hist_basis.compute_features(spike_series)
        X_blocks.append(X_hist)
        for b_idx in range(X_hist.shape[1]):
            feature_names.append(f"spike_history_basis{b_idx}")

    if X_blocks:
        X = np.column_stack(X_blocks)
    else:
        X = np.zeros((n_time, 0))

    return X, feature_names



def fit_glm_for_unit_nemos(
    unit_data: np.ndarray,
    event_windows_data: Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        pd.DataFrame,
        Dict[str, Any],
        np.ndarray,
    ],
    alpha: float = 1.0,
) -> Dict[str, Any]:
    """
    Time-resolved NeMoS Poisson GLM for a single unit.

    This function uses the full event-windows tuple to construct a
    NeMoS-compatible design matrix over time bins, then fits a GLM
    with Poisson observations.
    """
    # Handle both full event_windows_data (6 values) and behavioral-only data (5 values)
    if len(event_windows_data) == 6:
        (
            _event_windows_matrix,
            time_axis,
            _valid_event_indices,
            stimuli_outcome_df,
            metadata,
            lick_event_windows_matrix,
        ) = event_windows_data
    else:
        (
            lick_event_windows_matrix,
            time_axis,
            _valid_event_indices,
            stimuli_outcome_df,
            metadata,
        ) = event_windows_data

    # Build per-trial behavioral design
    design_matrix, design_time_axis, bin_size = build_trial_design_matrix(
        time_axis=time_axis,
        stimuli_outcome_df=stimuli_outcome_df,
        metadata=metadata,
        lick_event_windows_matrix=lick_event_windows_matrix,
    )

    # Build time-resolved predictors and spike counts
    spike_counts_flat, predictors = build_nemos_time_series_inputs(
        unit_data=unit_data,
        time_axis=design_time_axis,
        design_matrix=design_matrix,
        bin_size=bin_size,
    )

    # Construct NeMoS basis objects using default kernel configs
    bases = make_nemos_bases(bin_size=bin_size)

    # Build feature matrix X for NeMoS GLM
    X, feature_names = build_nemos_feature_matrix(
        spike_counts_flat=spike_counts_flat,
        predictors=predictors,
        bases=bases,
    )

    if X.shape[0] < 10:
        return {
            "error": "Insufficient time bins for NeMoS GLM fitting",
            "n_time_bins": X.shape[0],
        }

    # Configure NeMoS Poisson GLM with Ridge regularization
    # Use string shortcuts for observation model and regularizer
    glm = nmo.glm.GLM(
        observation_model="Poisson",  
        regularizer="Ridge",
        regularizer_strength=alpha,
    )

    glm.fit(X, spike_counts_flat)

    # Predictions and log-likelihood
    y_pred = glm.predict(X)
    log_likelihood = glm.score(X, spike_counts_flat)

    results: Dict[str, Any] = {
        "model": glm,
        "feature_names": feature_names,
        "coefficients": getattr(glm, "coef_", None).tolist()
        if hasattr(glm, "coef_")
        else None,
        "intercept": float(getattr(glm, "intercept_", 0.0)),
        "log_likelihood": float(log_likelihood),
        "y_pred_mean": float(np.mean(y_pred)),
        "n_features": int(X.shape[1]),
        "n_time_bins": int(X.shape[0]),
        "bin_size": float(bin_size),
    }
    return results


def fit_glm_for_all_units_nemos(
    units: List,
    event_windows_data: Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        pd.DataFrame,
        Dict[str, Any],
        np.ndarray,
    ],
    alpha: float = 1.0,
    save_to_csv: bool = True,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fit NeMoS time-resolved Poisson GLM for all units and return a summary DataFrame.

    This function mirrors `fit_glm_for_all_units` but uses the NeMoS-based,
    time-resolved GLM defined in `fit_glm_for_unit_nemos`.
    """
    from Analysis.NPXL_analysis.single_unit_offline_analysis.unit import Unit
    from Analysis.NPXL_analysis.single_unit_offline_analysis.utils import (
        save_dataframe_to_csv,
    )

    results_list: List[Dict[str, Any]] = []

    for unit in units:
        if not isinstance(unit, Unit):
            continue

        glm_results = fit_glm_for_unit_nemos(
            unit_data=unit.unit_data,
            event_windows_data=event_windows_data,
            alpha=alpha,
        )

        row: Dict[str, Any] = {
            "unit_idx": unit.unit_idx,
            "region_name": unit.region_name,
        }

        if "error" in glm_results:
            row["glm_error"] = glm_results["error"]
        else:
            row.update(
                {
                    "glm_log_likelihood": glm_results.get("log_likelihood"),
                    "glm_n_features": glm_results.get("n_features"),
                    "glm_n_time_bins": glm_results.get("n_time_bins"),
                }
            )

        results_list.append(row)

    df = pd.DataFrame(results_list)

    if save_to_csv and output_path:
        save_dataframe_to_csv(df, output_path, description="NeMoS GLM results")

    return df
