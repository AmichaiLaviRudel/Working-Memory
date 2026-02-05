"""
Streamlit panel for GLM analysis visualization.

This module provides interactive visualizations for GLM analysis results,
including real vs predicted traces, weight contributions, and PSTHs.
"""
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from typing import Tuple, Optional, Dict, Any, List
from datetime import datetime
import pynapple as nap
import shutil

# Configure pynapple
nap.nap_config.suppress_conversion_warnings = True

# Local imports
from .config import (
    RATE_THRESHOLD, BIN_SIZE, PREPROCESSING_BIN_SIZE,
    EPOCH_START, EPOCH_END,
    LOW_BOUNDARY_THRESHOLD, HIGH_BOUNDARY_THRESHOLD,
    N_BASIS_FUNCS, EVENT_WINDOW_SEC, ACAUSAL_BEFORE_SEC, ACAUSAL_AFTER_SEC,
    HISTORY_WINDOW_SEC, HISTORY_ACAUSAL_BEFORE_SEC, HISTORY_ACAUSAL_AFTER_SEC,
    N_POPULATION, INCLUDE_SPIKE_HISTORY,
    GLM_SOLVER, GLM_REGULARIZER, GLM_REGULARIZER_STRENGTH
)
from .loading import load_all_probes, load_events, filter_spikes
from .design_matrix import (
    create_categorical_features, create_temporal_features,
    create_basis_functions, create_categorical_impulses,
    build_design_matrix
)

_CATEGORICAL_RAW_COLS: List[str] = ["stimulus_ID", "category_ID", "outcome_ID", "previous_outcome"]


def _unit_results_dir(save_dir: str, unit_id: int) -> str:
    # Keep per-unit artifacts separate to avoid huge single files
    return os.path.join(save_dir, "units", f"unit_{int(unit_id)}")

def _unit_cache_complete(out_dir: str) -> bool:
    # We intentionally do NOT pickle the NeMoS model object because its regularizer
    # contains non-picklable closures (raises PicklingError). Cache numeric outputs instead.
    required = [
        os.path.join(out_dir, "predictions.npy"),
        os.path.join(out_dir, "actual_rates.npy"),
        os.path.join(out_dir, "time_axis.npy"),
        os.path.join(out_dir, "coefficients.npy"),
        os.path.join(out_dir, "design_matrix_columns.npy"),
        os.path.join(out_dir, "predicted_rates_tsd.npy"),
    ]
    return all(os.path.exists(p) for p in required)


def _shared_exists(save_dir: str) -> bool:
    # Minimal shared artifacts needed to fit units later
    required = [
        os.path.join(save_dir, "metadata.json"),
        os.path.join(save_dir, "temporal_features.csv"),
        os.path.join(save_dir, "categorical_features.csv"),
        os.path.join(save_dir, "tone_onset.npy"),
        os.path.join(save_dir, "outcome_time.npy"),
        os.path.join(save_dir, "units_spike_times.npy"),
    ]
    return all(os.path.exists(p) for p in required)

def _population_dir(save_dir: str) -> str:
    # Explicitly no-history population model cache
    return os.path.join(save_dir, "population_no_history")


def _population_cache_complete(pop_dir: str) -> bool:
    required = [
        os.path.join(pop_dir, "time_axis.npy"),
        os.path.join(pop_dir, "unit_ids.npy"),
        os.path.join(pop_dir, "unit_regions.npy"),
        os.path.join(pop_dir, "design_matrix_columns.npy"),
        os.path.join(pop_dir, "coefficients.npy"),
        os.path.join(pop_dir, "y_pred.npy"),
        os.path.join(pop_dir, "y_true.npy"),
        os.path.join(pop_dir, "per_unit_scores.npy"),
    ]
    return all(os.path.exists(p) for p in required)


def _poisson_pseudo_r2_cohen(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Vectorized Cohen's pseudo-R2 for Poisson GLM, per unit.

    y_true, y_pred: shape (T, N)
    Returns: shape (N,)
    """
    eps = 1e-10  # avoids log(0) and division by zero
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Ensure strictly positive for log terms (Poisson prediction should be positive, but be defensive)
    y_pred_safe = np.maximum(y_pred, eps)
    y_true_safe = np.maximum(y_true, 0.0)  # allow zeros

    y_mean = np.mean(y_true_safe, axis=0, keepdims=True)
    y_mean_safe = np.maximum(y_mean, eps)

    dev_model = 2.0 * np.sum(
        y_true_safe * np.log((y_true_safe + eps) / y_pred_safe) - (y_true_safe - y_pred_safe),
        axis=0,
    )
    dev_null = 2.0 * np.sum(
        y_true_safe * np.log((y_true_safe + eps) / y_mean_safe) - (y_true_safe - y_mean_safe),
        axis=0,
    )

    pseudo_r2 = np.zeros_like(dev_model, dtype=float)
    valid = dev_null > 0
    pseudo_r2[valid] = 1.0 - (dev_model[valid] / dev_null[valid])
    return pseudo_r2


def _build_X_shared_no_history(
    temporal_features: nap.TsdFrame,
    categorical_features: nap.TsdFrame,
    tone_onset: np.ndarray,
) -> nap.TsdFrame:
    """
    Build the shared (temporal + categorical) design matrix WITHOUT spike-history features.

    Why: user requested PopulationGLM without unit history_features.
    """
    categorical_impulse_tsd = create_categorical_impulses(
        categorical_features, temporal_features, tone_onset
    )

    basis_events, basis_categorical, _basis_history, *_ = create_basis_functions(
        N_BASIS_FUNCS,
        EVENT_WINDOW_SEC,
        ACAUSAL_BEFORE_SEC,
        ACAUSAL_AFTER_SEC,
        HISTORY_WINDOW_SEC,
        temporal_features.rate,
    )

    X_temporal_conv = basis_events.compute_features(temporal_features)
    X_categorical_conv = basis_categorical.compute_features(categorical_impulse_tsd)

    common_support = X_temporal_conv.time_support.intersect(X_categorical_conv.time_support)
    X_temp_common = X_temporal_conv.restrict(common_support)
    X_cat_common = X_categorical_conv.restrict(common_support)

    # Align time bases
    if X_temp_common.shape[0] != X_cat_common.shape[0]:
        raise ValueError("Temporal and categorical predictors misaligned after restriction.")

    temp_cols: List[str] = []
    for temp_feat in temporal_features.columns:
        for basis_idx in range(N_BASIS_FUNCS):
            temp_cols.append(f"{temp_feat}_basis{basis_idx}")

    cat_cols: List[str] = []
    for cat_feat in categorical_impulse_tsd.columns:
        for basis_idx in range(N_BASIS_FUNCS):
            cat_cols.append(f"{cat_feat}_basis{basis_idx}")

    X_shared = nap.TsdFrame(
        t=X_temp_common.t,
        d=np.column_stack([X_temp_common.values, X_cat_common.values]),
        columns=temp_cols + cat_cols,
    )

    valid_mask = np.all(np.isfinite(X_shared.values), axis=1)
    X_shared = nap.TsdFrame(
        t=X_shared.t[valid_mask],
        d=X_shared.values[valid_mask],
        columns=X_shared.columns,
    )
    return X_shared


def _fit_population_glm_no_history(base_path: str, glm_folder: str) -> bool:
    """
    Fit PopulationGLM over ACx+OFC good units, WITHOUT spike-history features.
    Cache numeric outputs (coefficients, predictions, per-unit scores) for UI.
    """
    pop_dir = _population_dir(glm_folder)
    os.makedirs(pop_dir, exist_ok=True)

    progress = st.progress(0)
    status = st.empty()

    status.text("Loading data for PopulationGLM (no history)...")
    progress.progress(10)
    spikes, probe_path_acx, _probe_path_ofc = load_all_probes(base_path)
    spikes = filter_spikes(spikes, unit_type="good", rate_threshold=RATE_THRESHOLD)

    licks, tone_onset, stimuli_outcome_df = load_events(base_path, probe_path_acx)

    status.text("Creating features...")
    progress.progress(25)
    categorical_features, data_df, outcome_time = create_categorical_features(
        tone_onset,
        stimuli_outcome_df,
        PREPROCESSING_BIN_SIZE,
        LOW_BOUNDARY_THRESHOLD,
        HIGH_BOUNDARY_THRESHOLD,
    )
    temporal_features, full_ep = create_temporal_features(
        tone_onset, licks, outcome_time, BIN_SIZE
    )

    # Trial epochs for fitting / interactive navigation
    start = tone_onset - EPOCH_START
    end = tone_onset + EPOCH_END
    epochs = nap.IntervalSet(start=start, end=end)

    status.text("Building shared design matrix (no history)...")
    progress.progress(40)
    X_shared = _build_X_shared_no_history(temporal_features, categorical_features, tone_onset)
    X_ep = X_shared.restrict(epochs)

    # Build spike counts aligned to X_ep time base
    status.text("Building spike count matrix...")
    progress.progress(55)

    unit_ids: List[int] = list(spikes.keys())
    spike_count = spikes.count(BIN_SIZE, ep=full_ep)
    spike_count = nap.TsdFrame(
        t=spike_count.t,
        d=spike_count.values[:, np.argsort(unit_ids)],
        columns=unit_ids,
    )

    y_ep = spike_count.restrict(X_ep.time_support)
    if y_ep.shape[0] != X_ep.shape[0]:
        # Align by nearest timestamps
        pop_indices = np.searchsorted(y_ep.t, X_ep.t)
        pop_indices = np.clip(pop_indices, 0, len(y_ep.t) - 1)
        y_ep = nap.TsdFrame(
            t=X_ep.t,
            d=y_ep.values[pop_indices],
            columns=y_ep.columns,
        )

    status.text("Fitting PopulationGLM (no history)...")
    progress.progress(70)
    import nemos as nmo

    model_pop = nmo.glm.PopulationGLM(
        solver_name=GLM_SOLVER,
        regularizer=GLM_REGULARIZER,
        regularizer_strength=GLM_REGULARIZER_STRENGTH,
    )
    model_pop.fit(X_ep, y_ep)

    status.text("Computing predictions + per-unit scores...")
    progress.progress(85)
    y_pred = np.asarray(model_pop.predict(X_ep))
    y_true = np.asarray(y_ep.values)

    per_unit_scores = _poisson_pseudo_r2_cohen(y_true=y_true, y_pred=y_pred)

    # Regions (ACx / OFC) aligned to unit_ids order
    region_series = spikes.get_info("region")
    unit_regions = np.array([region_series[uid] for uid in unit_ids], dtype=object)

    # Save artifacts (no pickling of model)
    np.save(os.path.join(pop_dir, "time_axis.npy"), np.asarray(X_ep.t))
    np.save(os.path.join(pop_dir, "unit_ids.npy"), np.asarray(unit_ids))
    np.save(os.path.join(pop_dir, "unit_regions.npy"), unit_regions, allow_pickle=True)
    np.save(os.path.join(pop_dir, "design_matrix_columns.npy"), np.asarray(list(X_shared.columns), dtype=object), allow_pickle=True)
    np.save(os.path.join(pop_dir, "coefficients.npy"), np.asarray(model_pop.coef_))
    np.save(os.path.join(pop_dir, "y_pred.npy"), y_pred)
    np.save(os.path.join(pop_dir, "y_true.npy"), y_true)
    np.save(os.path.join(pop_dir, "per_unit_scores.npy"), per_unit_scores)

    # Save epochs for navigation (shared file used by other parts)
    np.save(
        os.path.join(glm_folder, "epochs.npy"),
        {"start": epochs.start, "end": epochs.end},
        allow_pickle=True,
    )

    progress.progress(100)
    status.text("PopulationGLM cache saved.")
    return True


def _load_population_cache(glm_folder: str) -> Optional[Dict[str, Any]]:
    pop_dir = _population_dir(glm_folder)
    if not _population_cache_complete(pop_dir):
        return None

    try:
        time_axis = np.load(os.path.join(pop_dir, "time_axis.npy"))
        unit_ids = np.load(os.path.join(pop_dir, "unit_ids.npy")).tolist()
        unit_regions = np.load(os.path.join(pop_dir, "unit_regions.npy"), allow_pickle=True)
        X_columns = np.load(os.path.join(pop_dir, "design_matrix_columns.npy"), allow_pickle=True).tolist()
        coefs = np.load(os.path.join(pop_dir, "coefficients.npy"))
        y_pred = np.load(os.path.join(pop_dir, "y_pred.npy"))
        y_true = np.load(os.path.join(pop_dir, "y_true.npy"))
        scores = np.load(os.path.join(pop_dir, "per_unit_scores.npy"))

        # epochs are stored at root
        epochs_data = np.load(os.path.join(glm_folder, "epochs.npy"), allow_pickle=True).item()
        epochs = nap.IntervalSet(start=epochs_data["start"], end=epochs_data["end"])

        return {
            "time_axis": time_axis,
            "unit_ids": unit_ids,
            "unit_regions": unit_regions,
            "X_columns": X_columns,
            "coefficients": coefs,
            "y_pred": y_pred,
            "y_true": y_true,
            "scores": scores,
            "epochs": epochs,
        }
    except Exception:
        # If cache is corrupted, wipe it and refit on-demand
        try:
            shutil.rmtree(pop_dir)
        except Exception:
            pass
        return None


def _load_shared(save_dir: str) -> Dict[str, Any]:
    with open(os.path.join(save_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)

    temporal_df, categorical_df = load_features(save_dir)
    tone_onset = np.load(os.path.join(save_dir, "tone_onset.npy"))
    outcome_time = np.load(os.path.join(save_dir, "outcome_time.npy"))
    unit_ids = np.load(os.path.join(save_dir, "unit_ids.npy")).tolist() if os.path.exists(os.path.join(save_dir, "unit_ids.npy")) else None
    units_spike_times = np.load(os.path.join(save_dir, "units_spike_times.npy"), allow_pickle=True).item()

    # Reconstruct nap objects for design-matrix building
    temporal_t = temporal_df.index.astype(float).to_numpy()
    temporal_features = nap.TsdFrame(
        t=temporal_t,
        d=temporal_df.to_numpy(dtype=float),
        columns=list(temporal_df.columns),
    )

    categorical_t = categorical_df.index.astype(float).to_numpy()
    design_cols = [c for c in categorical_df.columns if c not in _CATEGORICAL_RAW_COLS]
    categorical_features = nap.TsdFrame(
        t=categorical_t,
        d=categorical_df[design_cols].to_numpy(dtype=float),
        columns=design_cols,
    )

    # A reasonable full epoch: same rule as create_temporal_features (0..max_time)
    max_time = float(np.nanmax([np.nanmax(tone_onset), np.nanmax(outcome_time), float(temporal_t[-1])]))
    full_ep = nap.IntervalSet(start=0.0, end=max_time)

    # Trial epochs for plotting/navigation
    start = tone_onset - EPOCH_START
    end = tone_onset + EPOCH_END
    epochs = nap.IntervalSet(start=start, end=end)

    return {
        "metadata": metadata,
        "temporal_df": temporal_df,
        "categorical_df": categorical_df,
        "temporal_features": temporal_features,
        "categorical_features": categorical_features,
        "tone_onset": tone_onset,
        "outcome_time": outcome_time,
        "epochs": epochs,
        "full_ep": full_ep,
        "unit_ids": unit_ids,
        "units_spike_times": units_spike_times,
    }


def _fit_and_save_unit_glm(save_dir: str, unit_id: int) -> Dict[str, Any]:
    """
    Fit a GLM for a specific unit using cached shared artifacts, then save per-unit results.

    Why: fitting all units upfront can be very slow; this caches per-unit results on-demand.
    """
    shared = _load_shared(save_dir)

    temporal_features: nap.TsdFrame = shared["temporal_features"]
    categorical_features: nap.TsdFrame = shared["categorical_features"]
    tone_onset: np.ndarray = shared["tone_onset"]
    epochs: nap.IntervalSet = shared["epochs"]
    full_ep: nap.IntervalSet = shared["full_ep"]
    units_spike_times: Dict[int, np.ndarray] = shared["units_spike_times"]

    if int(unit_id) not in units_spike_times:
        raise KeyError(f"unit_id {unit_id} missing from units_spike_times.npy")

    # Recreate binned spike count for this unit
    unit_spikes = nap.Ts(units_spike_times[int(unit_id)])
    neuron_count = unit_spikes.count(BIN_SIZE, ep=full_ep)  # nap.Tsd of counts/bin

    # Create categorical impulses on the temporal time base
    categorical_impulse_tsd = create_categorical_impulses(
        categorical_features, temporal_features, tone_onset
    )

    # Basis functions + design matrix (history is always included by build_design_matrix)
    basis_events, basis_categorical, basis_history, *_ = create_basis_functions(
        N_BASIS_FUNCS,
        EVENT_WINDOW_SEC,
        ACAUSAL_BEFORE_SEC,
        ACAUSAL_AFTER_SEC,
        HISTORY_WINDOW_SEC,
        temporal_features.rate,
    )

    X, _hist_feature_indices = build_design_matrix(
        temporal_features,
        categorical_impulse_tsd,
        neuron_count,
        basis_events,
        basis_categorical,
        basis_history,
        N_BASIS_FUNCS,
    )

    X_ep = X.restrict(epochs)
    y_in_epochs = neuron_count.restrict(epochs)
    y_ep = y_in_epochs.restrict(X_ep.time_support)

    # Align timestamps if needed (rare but happens with restriction boundaries)
    if y_ep.shape[0] != X_ep.shape[0]:
        y_times = np.asarray(y_ep.t)
        x_times = np.asarray(X_ep.t)
        y_indices = np.searchsorted(y_times, x_times)
        y_indices = np.clip(y_indices, 0, len(y_times) - 1)
        y_ep = nap.Tsd(t=X_ep.t, d=y_ep.d[y_indices])

    import nemos as nmo

    glm_basis = nmo.glm.GLM(
        solver_name=GLM_SOLVER,
        regularizer=GLM_REGULARIZER,
        regularizer_strength=GLM_REGULARIZER_STRENGTH,
    )
    glm_basis.fit(X_ep, y_ep)

    pred_rate = np.asarray(glm_basis.predict(X_ep)).squeeze()
    actual_rate = np.asarray(y_ep.d).squeeze()

    # Store predicted in Hz for PSTH alignment downstream
    pred_rate_hz = pred_rate / BIN_SIZE
    predicted_rates_tsd = nap.Tsd(t=X_ep.t, d=pred_rate_hz)

    coefs = np.asarray(glm_basis.coef_).flatten()
    X_columns = list(X.columns)

    out_dir = _unit_results_dir(save_dir, unit_id)
    os.makedirs(out_dir, exist_ok=True)

    # NOTE: Do NOT joblib/pickle the NeMoS GLM object. Its regularizer contains
    # a closure that triggers PicklingError. Cache only numeric outputs.
    np.save(os.path.join(out_dir, "predictions.npy"), pred_rate)
    np.save(os.path.join(out_dir, "actual_rates.npy"), actual_rate)
    np.save(os.path.join(out_dir, "time_axis.npy"), np.asarray(X_ep.t))
    np.save(os.path.join(out_dir, "coefficients.npy"), coefs)
    np.save(os.path.join(out_dir, "design_matrix_columns.npy"), X_columns, allow_pickle=True)
    np.save(
        os.path.join(out_dir, "predicted_rates_tsd.npy"),
        {"t": np.asarray(predicted_rates_tsd.t), "d": np.asarray(predicted_rates_tsd.d)},
        allow_pickle=True,
    )

    return {
        "predictions": pred_rate,
        "actual_rates": actual_rate,
        "time_axis": np.asarray(X_ep.t),
        "coefficients": coefs,
        "design_matrix_columns": X_columns,
        "predicted_rates_tsd": predicted_rates_tsd,
    }


def _load_unit_results(save_dir: str, unit_id: int) -> Optional[Dict[str, Any]]:
    out_dir = _unit_results_dir(save_dir, unit_id)
    if not os.path.isdir(out_dir) or not _unit_cache_complete(out_dir):
        return None

    try:
        pred = np.load(os.path.join(out_dir, "predictions.npy"))
        actual = np.load(os.path.join(out_dir, "actual_rates.npy"))
        time_axis = np.load(os.path.join(out_dir, "time_axis.npy"))
        coefs = np.load(os.path.join(out_dir, "coefficients.npy"))
        X_columns = np.load(os.path.join(out_dir, "design_matrix_columns.npy"), allow_pickle=True).tolist()
        pred_tsd_data = np.load(os.path.join(out_dir, "predicted_rates_tsd.npy"), allow_pickle=True).item()
        predicted_rates_tsd = nap.Tsd(t=pred_tsd_data["t"], d=pred_tsd_data["d"])
    except Exception:
        # If cache is corrupted (EOFError is common), wipe it and refit on-demand.
        try:
            shutil.rmtree(out_dir)
        except Exception:
            pass
        return None

    return {
        "predictions": pred,
        "actual_rates": actual,
        "time_axis": time_axis,
        "coefficients": coefs,
        "design_matrix_columns": X_columns,
        "predicted_rates_tsd": predicted_rates_tsd,
    }


def run_glm_analysis(base_path: str, save_dir: str) -> bool:
    """
    Run full GLM analysis pipeline and save results.
    
    Parameters
    ----------
    base_path : str
        Base recording path
    save_dir : str
        Directory to save GLM results
        
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    try:
        os.makedirs(save_dir, exist_ok=True)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # If shared cache already exists, don't redo expensive I/O here.
        # Unit fits are performed on-demand per selected unit.
        if _shared_exists(save_dir):
            progress_bar.progress(100)
            status_text.text("GLM cache already exists.")
            return True

        # 1. Load data
        status_text.text("Loading spike data and events...")
        progress_bar.progress(10)
        
        spikes, probe_path_acx, probe_path_ofc = load_all_probes(base_path)
        spikes = filter_spikes(spikes, unit_type='good', rate_threshold=RATE_THRESHOLD)
        
        licks, tone_onset, stimuli_outcome_df = load_events(base_path, probe_path_acx)
        
        # 2. Create features
        status_text.text("Creating features...")
        progress_bar.progress(20)
        
        categorical_features, data_df, outcome_time = create_categorical_features(
            tone_onset, stimuli_outcome_df, PREPROCESSING_BIN_SIZE,
            LOW_BOUNDARY_THRESHOLD, HIGH_BOUNDARY_THRESHOLD
        )
        
        temporal_features, full_ep = create_temporal_features(
            tone_onset, licks, outcome_time, BIN_SIZE
        )
        
        # Save shared spike times + unit ids (fast to reload later)
        unit_ids = list(spikes.keys())
        np.save(os.path.join(save_dir, "unit_ids.npy"), unit_ids)
        units_dict = {uid: spikes[uid].t for uid in unit_ids}
        np.save(os.path.join(save_dir, "units_spike_times.npy"), units_dict, allow_pickle=True)
        
        # Build epochs
        start = tone_onset - EPOCH_START
        end = tone_onset + EPOCH_END
        epochs = nap.IntervalSet(start=start, end=end)
        
        # 3. Save shared results (features + event timing)
        status_text.text("Saving results...")
        progress_bar.progress(60)

        # Save event times for PSTH
        np.save(os.path.join(save_dir, "tone_onset.npy"), tone_onset)
        np.save(os.path.join(save_dir, "outcome_time.npy"), outcome_time)
        
        # Save features as CSV (following example_glm_usage.py pattern)
        temporal_df = pd.DataFrame(
            data=temporal_features.values,
            index=temporal_features.t,
            columns=temporal_features.columns
        )
        temporal_df.to_csv(os.path.join(save_dir, "temporal_features.csv"))
        
        # Save categorical feature design matrix (+ raw columns), aligned to tone onsets
        categorical_df = pd.DataFrame(
            data=categorical_features.values,
            index=categorical_features.t,
            columns=categorical_features.columns
        )
        # Add original categorical columns
        categorical_df["stimulus_ID"] = data_df["stimulus_ID"].values
        categorical_df["category_ID"] = data_df["category_ID"].values
        categorical_df["outcome_ID"] = data_df["outcome_ID"].values
        categorical_df["previous_outcome"] = data_df["previous_outcome"].values
        categorical_df.to_csv(os.path.join(save_dir, "categorical_features.csv"))
        
        # Save metadata
        region_info = spikes.get_info("region")
        n_units_by_region = pd.Series(region_info).value_counts().to_dict()
        
        base_name = base_path.split('\\')[-1]
        rec_name = base_name.replace('catgt_', '') if base_name.startswith('catgt_') else base_name
        
        metadata = {
            "recording_name": rec_name,
            "base_path": base_path,
            "save_timestamp": datetime.now().isoformat(),
            "n_units_by_region": n_units_by_region,
            "parameters": {
                "BIN_SIZE": BIN_SIZE,
                "PREPROCESSING_BIN_SIZE": PREPROCESSING_BIN_SIZE,
                "EPOCH_START": EPOCH_START,
                "EPOCH_END": EPOCH_END,
                "LOW_BOUNDARY_THRESHOLD": LOW_BOUNDARY_THRESHOLD,
                "HIGH_BOUNDARY_THRESHOLD": HIGH_BOUNDARY_THRESHOLD,
                "N_BASIS_FUNCS": N_BASIS_FUNCS,
                "RATE_THRESHOLD": RATE_THRESHOLD,
            },
            "data_shapes": {
                "temporal_features": list(temporal_df.shape),
                "categorical_features": list(categorical_df.shape),
            },
            "temporal_features_columns": list(temporal_df.columns),
            "categorical_features_columns": list(categorical_df.columns),
            "n_tone_onsets": len(tone_onset),
            "n_units": len(unit_ids),
            # The first unit shown is fit lazily; keep metadata stable
            "example_neuron_id": int(unit_ids[0]) if len(unit_ids) > 0 else None,
        }
        
        with open(os.path.join(save_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        progress_bar.progress(100)
        status_text.text("Analysis complete!")
        
        return True
        
    except Exception as e:
        st.error(f"Error running GLM analysis: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return False


def load_features(save_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load temporal and categorical features from CSV files.
    
    Parameters
    ----------
    save_dir : str
        Directory containing saved features
        
    Returns
    -------
    temporal_df : pd.DataFrame
        Temporal features DataFrame
    categorical_df : pd.DataFrame
        Categorical features DataFrame
    """
    temporal_df = pd.read_csv(os.path.join(save_dir, "temporal_features.csv"), index_col=0)
    categorical_df = pd.read_csv(os.path.join(save_dir, "categorical_features.csv"), index_col=0)
    return temporal_df, categorical_df


def load_glm_results(save_dir: str) -> Optional[Dict[str, Any]]:
    """
    Load saved GLM results.
    
    Parameters
    ----------
    save_dir : str
        Directory containing saved GLM results
        
    Returns
    -------
    dict or None
        Dictionary containing loaded results, or None if loading fails
    """
    try:
        shared = _load_shared(save_dir)
        return {
            "metadata": shared["metadata"],
            "epochs": shared["epochs"],
            "tone_onset": shared["tone_onset"],
            "outcome_time": shared["outcome_time"],
            "unit_ids": shared["unit_ids"] or list(shared["units_spike_times"].keys()),
            "units_spike_times": shared["units_spike_times"],
            "temporal_features_df": shared["temporal_df"],
            "categorical_features_df": shared["categorical_df"],
        }
        
    except Exception as e:
        st.error(f"Error loading GLM results: {str(e)}")
        return None


def plot_actual_vs_predicted_interactive(
    actual: np.ndarray,
    predicted: np.ndarray,
    epochs: nap.IntervalSet,
    epoch_idx: int,
    unit_id: int,
    time_axis: np.ndarray,
    bin_size: float
) -> go.Figure:
    """
    Create interactive Plotly plot of actual vs predicted firing rates for a specific epoch.
    
    Parameters
    ----------
    actual : np.ndarray
        Actual firing rates
    predicted : np.ndarray
        Predicted firing rates
    epochs : nap.IntervalSet
        Epoch intervals
    epoch_idx : int
        Index of epoch to plot
    unit_id : int
        Unit ID
    time_axis : np.ndarray
        Time axis for the data
    bin_size : float
        Bin size in seconds
        
    Returns
    -------
    go.Figure
        Plotly figure
    """
    if epoch_idx >= len(epochs):
        epoch_idx = 0
    
    epoch_start = epochs.start[epoch_idx]
    epoch_end = epochs.end[epoch_idx]
    
    # Find indices within epoch
    mask = (time_axis >= epoch_start) & (time_axis <= epoch_end)
    t_epoch = time_axis[mask]
    actual_epoch = actual[mask] / bin_size  # Convert to Hz
    predicted_epoch = predicted[mask] / bin_size  # Convert to Hz
    
    fig = go.Figure()
    
    # Actual trace
    fig.add_trace(go.Scatter(
        x=t_epoch,
        y=actual_epoch,
        mode='lines',
        name='Actual',
        line=dict(color='orange', width=2)
    ))
    
    # Predicted trace
    fig.add_trace(go.Scatter(
        x=t_epoch,
        y=predicted_epoch,
        mode='lines',
        name='Predicted',
        line=dict(color='green', width=2)
    ))
    
    fig.update_layout(
        title=f"Actual vs Predicted Firing Rate - Unit {unit_id} (Epoch {epoch_idx + 1})<br>"
              f"Time: {epoch_start:.2f} - {epoch_end:.2f} s",
        xaxis_title="Time (s)",
        yaxis_title="Firing Rate (Hz)",
        hovermode='x unified',
        height=400
    )
    
    return fig


def plot_weight_contributions_interactive(
    coefs: np.ndarray,
    X_columns: list,
    unit_id: int,
    show_absolute: bool = True
) -> go.Figure:
    """
    Create interactive Plotly bar chart of weight contributions.
    
    Parameters
    ----------
    coefs : np.ndarray
        GLM coefficients
    X_columns : list
        Design matrix column names
    unit_id : int
        Unit ID
    show_absolute : bool
        If True, show absolute values; if False, show signed values
        
    Returns
    -------
    go.Figure
        Plotly figure
    """
    # Group features by type
    feature_groups = {}
    
    for i, col in enumerate(X_columns):
        if 'tone_onset' in col:
            group = 'tone_onset'
        elif 'licks' in col:
            group = 'licks'
        elif 'outcome_onset' in col:
            group = 'outcome_onset'
        elif 'stimulus' in col.lower():
            group = 'stimulus'
        elif 'category' in col.lower():
            group = 'category'
        elif 'outcome' in col.lower():
            group = 'outcome'
        elif 'previous_outcome' in col.lower():
            group = 'previous_outcome'
        elif 'spike_history' in col.lower():
            group = 'spike_history'
        else:
            group = 'other'
        
        if group not in feature_groups:
            feature_groups[group] = []
        feature_groups[group].append((i, col, coefs[i]))
    
    # Calculate total contribution per group
    group_contributions = {}
    for group, items in feature_groups.items():
        if show_absolute:
            group_contributions[group] = np.sum([abs(coef) for _, _, coef in items])
        else:
            group_contributions[group] = np.sum([coef for _, _, coef in items])
    
    # Sort by contribution
    sorted_groups = sorted(group_contributions.items(), key=lambda x: abs(x[1]), reverse=True)
    
    groups = [g[0] for g in sorted_groups]
    contributions = [g[1] for g in sorted_groups]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=contributions,
        y=groups,
        orientation='h',
        marker=dict(color='steelblue'),
        text=[f"{c:.4f}" for c in contributions],
        textposition='auto'
    ))
    
    fig.update_layout(
        title=f"Weight Contributions - Unit {unit_id}",
        xaxis_title="Absolute Contribution" if show_absolute else "Signed Contribution",
        yaxis_title="Feature Group",
        height=max(400, len(groups) * 30),
        hovermode='y'
    )
    
    return fig


def compute_psth_pynapple(
    spikes: nap.Ts,
    event_times: np.ndarray,
    predicted_rates_tsd: nap.Tsd,
    window: Tuple[float, float] = (-1.0, 3.0),
    bin_size: float = 0.01
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute PSTH using pynapple framework for both real and predicted firing rates.
    
    Parameters
    ----------
    spikes : nap.Ts
        Spike times for the unit
    event_times : np.ndarray
        Event times to align to
    predicted_rates_tsd : nap.Tsd
        Predicted firing rates as TsdFrame
    window : tuple
        Time window around events (start, end) in seconds
    bin_size : float
        Bin size in seconds
        
    Returns
    -------
    time_bins : np.ndarray
        Time bin centers
    real_psth : np.ndarray
        Real PSTH (firing rate in Hz)
    predicted_psth : np.ndarray
        Predicted PSTH (firing rate in Hz)
    """
    # Create time bins
    time_bins = np.arange(window[0], window[1] + bin_size, bin_size)
    bin_centers = (time_bins[:-1] + time_bins[1:]) / 2
    
    # Initialize arrays to store counts for each event
    real_counts_list = []
    predicted_rates_list = []
    
    for event_time in event_times:
        # Create event window
        event_window = nap.IntervalSet(
            start=event_time + window[0],
            end=event_time + window[1]
        )
        
        # Real PSTH: count spikes in window
        spikes_in_window = spikes.restrict(event_window)
        if len(spikes_in_window) > 0:
            # Get relative times
            relative_times = spikes_in_window.t - event_time
            # Bin the spikes
            counts, _ = np.histogram(relative_times, bins=time_bins)
            real_counts_list.append(counts)
        else:
            real_counts_list.append(np.zeros(len(time_bins) - 1))
        
        # Predicted PSTH: extract predicted rates in window
        pred_in_window = predicted_rates_tsd.restrict(event_window)
        if len(pred_in_window) > 0:
            # Get relative times
            relative_times_pred = pred_in_window.t - event_time
            # Interpolate to time bins
            pred_interp = np.interp(bin_centers, relative_times_pred, pred_in_window.d)
            predicted_rates_list.append(pred_interp)
        else:
            predicted_rates_list.append(np.zeros(len(bin_centers)))
    
    # Stack and average across events
    real_counts_array = np.array(real_counts_list)  # [n_events × n_bins]
    predicted_rates_array = np.array(predicted_rates_list)  # [n_events × n_bins]
    
    # Average across events
    real_psth_counts = np.mean(real_counts_array, axis=0)
    predicted_psth = np.mean(predicted_rates_array, axis=0)
    
    # Convert counts to firing rate (Hz)
    real_psth = real_psth_counts / bin_size
    
    return bin_centers, real_psth, predicted_psth


def plot_psth_real_vs_predicted_interactive(
    spikes: nap.Ts,
    event_times: np.ndarray,
    predicted_rates_tsd: nap.Tsd,
    unit_id: int,
    event_type: str,
    window: Tuple[float, float] = (-1.0, 3.0),
    bin_size: float = 0.01
) -> go.Figure:
    """
    Create interactive Plotly PSTH comparing real vs predicted firing rates.
    
    Parameters
    ----------
    spikes : nap.Ts
        Spike times for the unit
    event_times : np.ndarray
        Event times to align to
    predicted_rates_tsd : nap.Tsd
        Predicted firing rates as TsdFrame
    unit_id : int
        Unit ID
    event_type : str
        Type of event ("Tone Onset" or "Outcome Onset")
    window : tuple
        Time window around events (start, end) in seconds
    bin_size : float
        Bin size in seconds
        
    Returns
    -------
    go.Figure
        Plotly figure
    """
    time_bins, real_psth, predicted_psth = compute_psth_pynapple(
        spikes, event_times, predicted_rates_tsd, window, bin_size
    )
    
    fig = go.Figure()
    
    # Real PSTH
    fig.add_trace(go.Scatter(
        x=time_bins,
        y=real_psth,
        mode='lines',
        name='Real',
        line=dict(color='orange', width=2)
    ))
    
    # Predicted PSTH
    fig.add_trace(go.Scatter(
        x=time_bins,
        y=predicted_psth,
        mode='lines',
        name='Predicted',
        line=dict(color='green', width=2)
    ))
    
    # Add vertical line at t=0
    fig.add_vline(
        x=0,
        line_dash="dash",
        line_color="red",
        annotation_text="Event",
        annotation_position="top"
    )
    
    fig.update_layout(
        title=f"PSTH: Real vs Predicted - Unit {unit_id}<br>Aligned to {event_type} (n={len(event_times)} events)",
        xaxis_title="Time from Event (s)",
        yaxis_title="Firing Rate (Hz)",
        hovermode='x unified',
        height=500
    )
    
    return fig


def glm_analysis_panel(base_path: str, selected_area: str):
    """
    Main Streamlit panel for GLM analysis.
    
    Parameters
    ----------
    base_path : str
        Base recording path
    selected_area : str
        Selected area label (for display)
    """
    header_left, header_right = st.columns([1, 0.25])
    with header_left:
        st.write("### GLM Analysis")
    with header_right:
        # Simple UX escape hatch when Streamlit state gets stuck
        if st.button("🔄 Rerun", key="glm_rerun_btn"):
            st.rerun()
    
    # Determine GLM folder path
    glm_folder = os.path.join(base_path, "GLM")
    
    if not _shared_exists(glm_folder):
        st.info("GLM analysis has not been run yet. Click the button below to run the analysis.")
        if st.button("🚀 Run GLM Analysis", type="primary"):
            with st.spinner("Running GLM analysis... This may take several minutes."):
                success = run_glm_analysis(base_path, glm_folder)
                if success:
                    st.success("✅ GLM analysis completed successfully!")
                    st.rerun()
                else:
                    st.error("❌ GLM analysis failed. Check the error messages above.")
        return
    
    # Load results
    results = load_glm_results(glm_folder)
    if results is None:
        st.error("Failed to load GLM results.")
        return
    
    # Display metadata
    metadata = results['metadata']
    with st.expander("Analysis Information", expanded=False):
        st.json(metadata)

    # PopulationGLM (no history) cache
    pop_cache = _load_population_cache(glm_folder)
    if pop_cache is None:
        st.info("PopulationGLM (no history) has not been run yet for this recording.")
        if st.button("🚀 Run PopulationGLM (no history)", type="primary", key="run_pop_glm_no_hist"):
            with st.spinner("Fitting PopulationGLM (no history)... This may take several minutes."):
                ok = _fit_population_glm_no_history(base_path=base_path, glm_folder=glm_folder)
                if ok:
                    st.success("✅ PopulationGLM cache created.")
                    st.rerun()
                else:
                    st.error("❌ PopulationGLM failed.")
        return

    unit_ids: List[int] = pop_cache["unit_ids"]
    unit_regions = np.asarray(pop_cache["unit_regions"], dtype=object)
    scores = np.asarray(pop_cache["scores"], dtype=float)

    # Unit filter + sorted selector by model score
    st.subheader("Unit selection (sorted by model score)")
    region_filter = st.selectbox(
        "Region",
        options=["All", "ACx", "OFC"],
        index=0,
        key="glm_region_filter",
    )

    indices = np.arange(len(unit_ids))
    if region_filter != "All":
        indices = indices[unit_regions == region_filter]

    # Sort by score (desc); NaNs to bottom
    sort_key = np.nan_to_num(scores[indices], nan=-np.inf)
    sorted_indices = indices[np.argsort(sort_key)[::-1]]

    def _fmt_unit(i: int) -> str:
        uid = unit_ids[i]
        reg = unit_regions[i]
        sc = scores[i]
        if np.isnan(sc):
            return f"Unit {uid} | {reg} | score=NaN"
        return f"Unit {uid} | {reg} | score={sc:.3f}"

    if len(sorted_indices) == 0:
        st.warning("No units match the selected region filter.")
        return

    selected_idx = st.selectbox(
        "Select Unit",
        options=sorted_indices.tolist(),
        format_func=_fmt_unit,
        key="glm_unit_select_sorted",
    )
    selected_unit_id = int(unit_ids[selected_idx])

    # Shared arrays for the selected unit
    time_axis = np.asarray(pop_cache["time_axis"], dtype=float)
    epochs = pop_cache["epochs"]
    y_true = np.asarray(pop_cache["y_true"], dtype=float)  # (T, N)
    y_pred = np.asarray(pop_cache["y_pred"], dtype=float)  # (T, N)
    coefs = np.asarray(pop_cache["coefficients"], dtype=float)  # (F, N)
    X_cols = pop_cache["X_columns"]

    # Create tabs for different visualizations
    trace_tab, weights_tab, psth_tab = st.tabs([
        "Real vs Predicted Trace",
        "Weight Contributions",
        "PSTH: Real vs Predicted",
    ])

    with trace_tab:
        st.write("#### Real vs Predicted firing rate (interactive epoch slider)")
        n_epochs = len(epochs)
        epoch_idx = st.slider(
            "Select Epoch",
            min_value=0,
            max_value=max(0, n_epochs - 1),
            value=0,
            key="glm_epoch_slider",
        )

        fig = plot_actual_vs_predicted_interactive(
            actual=y_true[:, selected_idx],
            predicted=y_pred[:, selected_idx],
            epochs=epochs,
            epoch_idx=epoch_idx,
            unit_id=selected_unit_id,
            time_axis=time_axis,
            bin_size=BIN_SIZE,
        )
        st.plotly_chart(fig, use_container_width=True)

    with weights_tab:
        st.write("#### Weight contributions (PopulationGLM coefficients)")
        show_absolute = st.checkbox("Show Absolute Values", value=True, key="glm_weights_absolute")
        fig = plot_weight_contributions_interactive(
            coefs=coefs[:, selected_idx],
            X_columns=X_cols,
            unit_id=selected_unit_id,
            show_absolute=show_absolute,
        )
        st.plotly_chart(fig, use_container_width=True)

    with psth_tab:
        st.write("#### PSTH: Real vs Predicted (pynapple)")
        event_type = st.selectbox(
            "Event Type",
            options=["Tone Onset", "Outcome Onset"],
            key="glm_psth_event_type",
        )

        if event_type == "Tone Onset":
            event_times = results["tone_onset"]
        else:
            event_times = results["outcome_time"]

        event_times = event_times[~np.isnan(event_times)]
        event_times = event_times[event_times > 0]
        if len(event_times) == 0:
            st.warning(f"No valid {event_type} times found.")
            return

        units_spike_times = results["units_spike_times"]
        if selected_unit_id not in units_spike_times:
            st.warning(f"Spike times not found for Unit {selected_unit_id}.")
            return

        spikes_ts = nap.Ts(units_spike_times[selected_unit_id])
        predicted_rates_hz = y_pred[:, selected_idx] / BIN_SIZE
        predicted_rates_tsd = nap.Tsd(t=time_axis, d=predicted_rates_hz)

        window_start = st.slider("Window Start (s)", -2.0, 0.0, -1.0, 0.1, key="glm_psth_window_start")
        window_end = st.slider("Window End (s)", 0.0, 5.0, 3.0, 0.1, key="glm_psth_window_end")
        window = (window_start, window_end)

        fig = plot_psth_real_vs_predicted_interactive(
            spikes=spikes_ts,
            event_times=event_times,
            predicted_rates_tsd=predicted_rates_tsd,
            unit_id=selected_unit_id,
            event_type=event_type,
            window=window,
            bin_size=BIN_SIZE,
        )
        st.plotly_chart(fig, use_container_width=True)
