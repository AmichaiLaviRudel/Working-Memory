"""Representational Similarity Analysis (RSA) for NPXL dual-probe sessions.

Quantifies how stimulus-frequency representations cluster by category (Go vs No-Go)
across ACx and OFC for three matched cohorts: Novice 2b, 1b Expert 1b, and
2b Expert 2b Categorization sessions only.

Public entry point
------------------
``regional_rsa_panel(selected_sessions_df)`` — Streamlit panel called from
``npxl_monitoring.py`` once the user has chosen a session subset.

Design notes
------------
- Self-contained: does NOT import from ``npxl_agreement_decoder`` to avoid triggering
  that page's top-level ``if _in_streamlit(): st.title(...)`` block when imported
  from another Streamlit page.
- Reuses low-level loaders from
  ``Analysis.NPXL_analysis.single_unit_offline_analysis.data_loading``.
- Per-session RSA matrices are averaged across sessions within each
  (learning_stage x area) cell. No cross-session unit alignment is required.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from sklearn.manifold import MDS

from Analysis.GNG_bpod_analysis.colors import (
    COLOR_ACX,
    COLOR_GO,
    COLOR_GRAY,
    COLOR_HIGH_BD,
    COLOR_LOW_BD,
    COLOR_NOGO,
    COLOR_OFC,
    LEARNING_STAGE_COLORS,
)
from Analysis.NPXL_analysis.single_unit_offline_analysis.data_loading import (
    load_full_event_windows_data,
    load_histology_matched_unit_indices,
    load_unit_indices_by_type,
)
# --- Constants ----------------------------------------------------------------

# Probe naming matches the rest of the NPXL stack (SpikeGLX imec0/imec1).
AREA_TO_IMEC: dict[str, str] = {"ACx": "imec0", "OFC": "imec1"}
# Bombcell unit-type codes for single units + MUA, matches agreement decoder.
GOOD_MUA_CODES: tuple[int, int] = (1, 2)
# Three matched categorization cohorts (learning stage x boundary), left-to-right in plots.
STAGE_ORDER: tuple[str, ...] = ("Novice", "1b Expert", "2b Expert")
RSA_COHORT_LABELS: dict[str, str] = {
    "Novice": "Novice — 2b Categorization",
    "1b Expert": "1b Expert — 1b Categorization",
    "2b Expert": "2b Expert — 2b Categorization",
}
# Default decode window (seconds, relative to tone onset).
DEFAULT_DECODE_WINDOW: tuple[float, float] = (-0.05, 0.3)
DEFAULT_MIN_TRIALS_PER_STIM = 3
# Minimum stimuli required to compute a meaningful RSA matrix.
MIN_STIMULI_FOR_RSA = 2
# Per-cohort stimulus caps (kHz). «21.5» = deci-kHz psychometric notation → 2.15 kHz.
DECI_KHZ_21_5_MAX_STIM_KHZ = 2.15
# Log-spaced bins per category band (matches ``_log_bin_stimulus_per_stage`` in agreement decoder).
RSA_BINS_PER_CLASS = 4


def _stage_max_stim_khz(stage: str, high_boundary: float) -> float | None:
    """Return the upper stimulus frequency (kHz) for RSA, or None for no cap."""
    if stage == "1b Expert":
        return float(high_boundary)
    if stage in ("Novice", "2b Expert"):
        return DECI_KHZ_21_5_MAX_STIM_KHZ
    return None

# Persisted outputs (under Code/DB/Results/RSA).
_RSA_MODULE_DIR = Path(__file__).resolve().parent
_DB_ROOT = _RSA_MODULE_DIR.parents[1]
RSA_RESULTS_DIR = _DB_ROOT / "Results" / "RSA"
RSA_ERROR_LOG_PATH = RSA_RESULTS_DIR / "rsa_error_log.csv"


def _category_boundary_khz() -> tuple[float, float]:
    """Tone-category boundaries (kHz) from global session state (set by Home.py)."""
    return (
        float(st.session_state.get("low_boundary", 0.983)),
        float(st.session_state.get("high_boundary", 1.525)),
    )


def rsa_cohort_stage(session_type: object) -> str | None:
    """Map ``Session Type`` to an RSA cohort key, or None if excluded.

    Only three cohorts are included (matched stage x categorization boundary):
    - Novice — 2b Categorization
    - 1b Expert — 1b Categorization
    - 2b Expert — 2b Categorization

    Other types (e.g. Novice 1b, 1b Expert 2b, FRA) return None.
    """
    s = str(session_type).strip().lower()
    if "categorization" not in s:
        return None
    if "novice" in s:
        return "Novice" if "2b categorization" in s else None
    if "1b expert" in s:
        if "2b categorization" in s:
            return None
        return "1b Expert" if "1b categorization" in s else None
    if "2b expert" in s:
        return "2b Expert" if "2b categorization" in s else None
    return None


def filter_rsa_cohort_sessions(sessions_df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows whose Session Type is one of the three RSA cohorts."""
    if sessions_df.empty or "Session Type" not in sessions_df.columns:
        return sessions_df.iloc[0:0].copy()
    mask = sessions_df["Session Type"].map(rsa_cohort_stage).notna()
    return sessions_df.loc[mask].copy()


def prepare_rsa_selection_table(source_df: pd.DataFrame) -> pd.DataFrame:
    """Session picker limited to the three RSA cohorts; eligible rows pre-checked."""
    cohort_df = filter_rsa_cohort_sessions(source_df)
    table_df = cohort_df.copy()
    if table_df.empty:
        return table_df

    if "Checkbox" not in table_df.columns:
        table_df.insert(0, "Checkbox", False)

    current_dir = table_df.get("current_dir", pd.Series("", index=table_df.index)).fillna("").astype(str).str.strip()
    table_df["Checkbox"] = current_dir.ne("")

    columns = [
        column
        for column in [
            "Checkbox",
            "Animal",
            "Date",
            "Session Type",
            "session_dprime",
            "session_hit_rate",
            "spike glx file",
            "Recording Assessment ",
            "Acx good units",
            "OFC good units",
            "current_dir",
        ]
        if column in table_df.columns
    ]
    return table_df[columns]


# --- Probe / matrix loading ---------------------------------------------------

def _find_probe_dir(session_dir: str, imec: str) -> str:
    """Locate imec0/imec1 folder containing exported ``event_windows_matrix.npy``."""
    if not os.path.isdir(session_dir):
        raise FileNotFoundError(f"Session folder not found: {session_dir}")

    direct = [
        os.path.join(session_dir, name)
        for name in os.listdir(session_dir)
        if os.path.isdir(os.path.join(session_dir, name)) and name.lower().endswith(imec.lower())
    ]
    for candidate in sorted(direct):
        if os.path.exists(os.path.join(candidate, "analysis_output", "event_windows_matrix.npy")):
            return candidate

    # CatGT / pipeline_output layouts vary; walk deeper only as a fallback.
    for root, dirs, _files in os.walk(session_dir):
        for dirname in dirs:
            if not dirname.lower().endswith(imec.lower()):
                continue
            cand = os.path.join(root, dirname)
            if os.path.exists(os.path.join(cand, "analysis_output", "event_windows_matrix.npy")):
                return cand

    raise FileNotFoundError(f"No {imec} probe folder with event windows found under {session_dir}")


@st.cache_data(show_spinner=False)
def _load_area_event_data(
    session_dir: str,
    area: str,
    *,
    use_histology: bool,
) -> dict[str, Any]:
    """Load one probe's tone-aligned matrix filtered to good+MUA (and histology-matched) units."""
    probe_dir = _find_probe_dir(session_dir, AREA_TO_IMEC[area])
    event_matrix, time_axis, valid_indices, trials_df, _meta, _lick = load_full_event_windows_data(probe_dir)
    if event_matrix.ndim != 3:
        raise ValueError(f"{area} event matrix must be 3D [units x time x trials], got {event_matrix.shape}")

    good_mua_idx = load_unit_indices_by_type(probe_dir, GOOD_MUA_CODES)
    if use_histology:
        unit_indices, _table = load_histology_matched_unit_indices(
            session_dir, probe_dir, area, GOOD_MUA_CODES
        )
        unit_indices = [i for i in unit_indices if 0 <= i < event_matrix.shape[0]]
    else:
        unit_indices = [i for i in good_mua_idx if 0 <= i < event_matrix.shape[0]]

    if not unit_indices:
        raise ValueError(f"{area}: no usable units after filtering (use_histology={use_histology}).")

    filtered = event_matrix[np.asarray(unit_indices, dtype=int), :, :]
    return {
        "matrix": filtered,                                     # [units x time x trials]
        "time_axis": np.asarray(time_axis, dtype=float),
        "valid_indices": np.asarray(valid_indices, dtype=int),
        "trials_df": trials_df.reset_index(drop=True),
        "unit_count": len(unit_indices),
    }


def _load_session_areas(
    session_dir: str,
    *,
    use_histology: bool,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Try to load ACx and OFC for a session. Returns (loaded_areas, per-area-errors)."""
    loaded: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    for area in ("ACx", "OFC"):
        try:
            loaded[area] = _load_area_event_data(session_dir, area, use_histology=use_histology)
        except (FileNotFoundError, ValueError, OSError) as exc:
            errors[area] = str(exc)
    return loaded, errors


# --- Feature extraction -------------------------------------------------------

def _feature_matrix(
    event_matrix: np.ndarray,
    time_axis: np.ndarray,
    window: tuple[float, float],
    aggregation: str,
) -> np.ndarray:
    """Collapse [units x time x trials] to [trials x units] within ``window``."""
    start, stop = window
    mask = (time_axis >= start) & (time_axis <= stop)
    if not mask.any():
        raise ValueError(f"Decode window {window} contains no time bins.")
    windowed = event_matrix[:, mask, :]
    if aggregation == "Sum":
        return windowed.sum(axis=1).T
    return windowed.mean(axis=1).T


def _stimulus_per_trial(trials_df: pd.DataFrame) -> np.ndarray:
    """Return ``stimulus_khz`` per trial; NaN where unavailable."""
    if "stimulus" not in trials_df.columns:
        return np.full(len(trials_df), np.nan)
    khz = pd.to_numeric(trials_df["stimulus"], errors="coerce").to_numpy()
    khz[~np.isfinite(khz) | (khz <= 0)] = np.nan
    return khz


def build_log_stimulus_bin_grid(
    stim_min: float,
    stim_max: float,
    low_boundary: float,
    high_boundary: float,
    *,
    bins_per_class: int = RSA_BINS_PER_CLASS,
) -> tuple[np.ndarray, np.ndarray]:
    """Geometric bin edges and centres between category boundaries (≤4 bins per band).

    Mirrors ``_log_bin_stimulus_per_stage`` in ``npxl_agreement_decoder.py`` so RSA
    pools similar kHz tones into a shared ladder across sessions.
    """
    if not np.isfinite(stim_min) or stim_min <= 0 or not np.isfinite(stim_max) or stim_max <= stim_min:
        return np.array([]), np.array([])

    class_edges = sorted(
        {float(stim_min), float(stim_max)}
        | {float(b) for b in (low_boundary, high_boundary) if stim_min < b < stim_max}
    )
    if len(class_edges) < 2:
        return np.array([]), np.array([])

    edge_set: set[float] = set()
    for left, right in zip(class_edges[:-1], class_edges[1:]):
        for edge in np.geomspace(left, right, bins_per_class + 1):
            edge_set.add(float(edge))
    edges = np.array(sorted(edge_set))
    if edges.size < 2:
        return np.array([]), np.array([])

    centers = np.sqrt(edges[:-1] * edges[1:])
    return edges, centers


def assign_stimuli_to_log_bins(
    stimuli_khz: np.ndarray,
    bin_edges: np.ndarray,
    bin_centers: np.ndarray,
) -> np.ndarray:
    """Map raw trial kHz values to log-bin centres (``side='right'`` at class edges)."""
    if bin_edges.size < 2 or bin_centers.size == 0:
        return stimuli_khz
    idx = np.clip(np.searchsorted(bin_edges, stimuli_khz, side="right") - 1, 0, len(bin_centers) - 1)
    return bin_centers[idx]


def _session_stimuli_khz(
    session_dir: str,
    *,
    use_histology: bool,
    max_stim_khz: float | None,
) -> np.ndarray:
    """Lightweight scan of one session's tone frequencies (ACx trials table)."""
    loaded, _errors = _load_session_areas(session_dir, use_histology=use_histology)
    if "ACx" not in loaded:
        return np.array([], dtype=float)
    stimuli = _stimulus_per_trial(loaded["ACx"]["trials_df"])
    stimuli = stimuli[np.isfinite(stimuli) & (stimuli > 0)]
    if max_stim_khz is not None:
        stimuli = stimuli[stimuli <= float(max_stim_khz)]
    return stimuli


def _scan_selection_stimuli_khz(
    selected_sessions_df: pd.DataFrame,
    *,
    use_histology: bool,
    low_boundary: float,
    high_boundary: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect capped trial stimuli and build a shared log-bin grid for the run."""
    pooled: list[np.ndarray] = []
    for _, row in selected_sessions_df.iterrows():
        stage = rsa_cohort_stage(row.get("Session Type"))
        if stage is None:
            continue
        session_dir = str(row.get("current_dir", "")).strip()
        if not session_dir or not os.path.isdir(session_dir):
            continue
        cap = _stage_max_stim_khz(stage, high_boundary)
        try:
            pooled.append(
                _session_stimuli_khz(session_dir, use_histology=use_histology, max_stim_khz=cap)
            )
        except (FileNotFoundError, ValueError, OSError):
            continue

    if not pooled:
        return np.array([]), np.array([])
    all_stim = np.concatenate([p for p in pooled if p.size > 0])
    if all_stim.size == 0:
        return np.array([]), np.array([])
    return build_log_stimulus_bin_grid(
        float(all_stim.min()),
        float(all_stim.max()),
        low_boundary,
        high_boundary,
        bins_per_class=RSA_BINS_PER_CLASS,
    )


# --- RSA core -----------------------------------------------------------------

def compute_session_rsa(
    session_dir: str,
    area: str,
    *,
    decode_window: tuple[float, float] = DEFAULT_DECODE_WINDOW,
    aggregation: str = "Mean",
    use_histology: bool = False,
    min_trials_per_stim: int = DEFAULT_MIN_TRIALS_PER_STIM,
    max_stim_khz: float | None = None,
    log_bin_edges: np.ndarray | None = None,
    log_bin_centers: np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute one ``[stimuli x stimuli]`` RSA matrix for a (session, area).

    Steps
    -----
    1. Load tone-aligned matrix for the area.
    2. Reduce to ``[trials x units]`` over ``decode_window``.
    3. Group trials by unique kHz; drop bins with fewer than ``min_trials_per_stim``.
    4. Z-score each unit across stimuli to remove rate-scale dominance.
    5. Pearson correlation across stimulus columns -> RSA matrix.

    Returns
    -------
    rsa_df : DataFrame indexed and columned by kHz (float), values in [-1, 1].
    info   : dict with ``n_units``, ``n_trials``, ``n_stimuli``, ``trials_per_stim``.
    """
    area_data = _load_area_event_data(session_dir, area, use_histology=use_histology)
    features = _feature_matrix(area_data["matrix"], area_data["time_axis"], decode_window, aggregation)
    n_trials, n_units = features.shape
    stimuli_khz = _stimulus_per_trial(area_data["trials_df"])
    if len(stimuli_khz) != n_trials:
        # Defensive: trials_df and matrix trial-axis should align by construction.
        m = min(len(stimuli_khz), n_trials)
        stimuli_khz = stimuli_khz[:m]
        features = features[:m]
        n_trials = m

    valid = np.isfinite(stimuli_khz)
    if not valid.any():
        raise ValueError("No trials with valid stimulus_khz.")
    features = features[valid]
    stimuli_khz = stimuli_khz[valid]

    if max_stim_khz is not None:
        cap_mask = stimuli_khz <= float(max_stim_khz)
        if not cap_mask.any():
            raise ValueError(f"No trials with stimulus_khz <= {max_stim_khz}.")
        features = features[cap_mask]
        stimuli_khz = stimuli_khz[cap_mask]

    if (
        log_bin_edges is not None
        and log_bin_centers is not None
        and log_bin_edges.size >= 2
        and log_bin_centers.size > 0
    ):
        stimuli_khz = assign_stimuli_to_log_bins(stimuli_khz, log_bin_edges, log_bin_centers)

    unique_khz, counts = np.unique(stimuli_khz, return_counts=True)
    keep_mask = counts >= int(min_trials_per_stim)
    unique_khz = unique_khz[keep_mask]
    counts = counts[keep_mask]
    if unique_khz.size < MIN_STIMULI_FOR_RSA:
        raise ValueError(
            f"Only {unique_khz.size} stimulus bins meet min_trials_per_stim={min_trials_per_stim}."
        )

    # response_by_stim: [units x stimuli] mean rate per unit per stimulus.
    response_by_stim = np.empty((n_units, unique_khz.size), dtype=float)
    for j, khz in enumerate(unique_khz):
        response_by_stim[:, j] = features[stimuli_khz == khz].mean(axis=0)

    # Z-score along the stimulus axis per unit; protects units with zero variance.
    mean_per_unit = response_by_stim.mean(axis=1, keepdims=True)
    std_per_unit = response_by_stim.std(axis=1, ddof=0, keepdims=True)
    std_per_unit[std_per_unit == 0] = 1.0
    zscored = (response_by_stim - mean_per_unit) / std_per_unit

    rsa = np.corrcoef(zscored.T) if zscored.shape[1] > 1 else np.array([[1.0]])
    rsa_df = pd.DataFrame(rsa, index=unique_khz, columns=unique_khz)
    info = {
        "n_units": int(n_units),
        "n_trials": int(features.shape[0]),
        "n_stimuli": int(unique_khz.size),
        "trials_per_stim": dict(zip(unique_khz.tolist(), counts.tolist())),
        "max_stim_khz": float(max_stim_khz) if max_stim_khz is not None else None,
        "log_binned": bool(
            log_bin_centers is not None and log_bin_centers.size > 0
        ),
    }
    return rsa_df, info


# --- Aggregation --------------------------------------------------------------

def aggregate_rsa_across_sessions(
    rsa_matrices: list[pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Element-wise nanmean of per-session RSA matrices on the union of kHz bins.

    Returns
    -------
    mean_rsa : DataFrame on the union grid (NaN where no session contributed).
    n_sessions : DataFrame of contributing session count per cell (int).
    """
    if not rsa_matrices:
        return pd.DataFrame(), pd.DataFrame()

    union = sorted({float(k) for m in rsa_matrices for k in m.index})
    if not union:
        return pd.DataFrame(), pd.DataFrame()

    # Stack reindexed matrices along a third axis, then nanmean.
    stacked = np.full((len(union), len(union), len(rsa_matrices)), np.nan, dtype=float)
    for i, m in enumerate(rsa_matrices):
        reindexed = m.reindex(index=union, columns=union)
        stacked[:, :, i] = reindexed.to_numpy(dtype=float)

    with np.errstate(invalid="ignore"):
        mean_rsa = np.nanmean(stacked, axis=2)
    n_sessions = np.sum(np.isfinite(stacked), axis=2)
    return (
        pd.DataFrame(mean_rsa, index=union, columns=union),
        pd.DataFrame(n_sessions.astype(int), index=union, columns=union),
    )


def subtract_rsa_matrices(
    minuend: pd.DataFrame,
    subtrahend: pd.DataFrame,
) -> pd.DataFrame:
    """Element-wise RSA difference (minuend − subtrahend) on the union kHz grid."""
    if minuend.empty or subtrahend.empty:
        return pd.DataFrame()

    union = sorted({float(k) for k in minuend.index} | {float(k) for k in subtrahend.index})
    if not union:
        return pd.DataFrame()

    a = minuend.reindex(index=union, columns=union).to_numpy(dtype=float)
    b = subtrahend.reindex(index=union, columns=union).to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        diff = a - b
    return pd.DataFrame(diff, index=union, columns=union)


# --- Categoricality / category-separation metrics (Panels G & H) --------------

PAIR_TYPE_LABELS: dict[str, str] = {
    "within_go": "Go-Go",
    "within_nogo": "NoGo-NoGo",
    "across": "Go-NoGo",
}
# Session-level metrics tested across learning stages (Panel H stats).
SEPARATION_TEST_METRICS: tuple[str, ...] = ("categoricality_index", "separation_index")


def _stimulus_category(khz: float, low_boundary: float, high_boundary: float) -> str:
    """Map kHz to Go (outside boundaries) / No-Go (between boundaries).

    Convention matches the agreement decoder's psychometric plot: tones below
    ``low_boundary`` or above ``high_boundary`` are Go; tones between are No-Go.
    """
    if khz < low_boundary:
        return "Go"
    if khz > high_boundary:
        return "Go"
    return "NoGo"


def _pair_type_from_categories(cat_i: str, cat_j: str) -> str:
    if cat_i == "Go" and cat_j == "Go":
        return "within_go"
    if cat_i == "NoGo" and cat_j == "NoGo":
        return "within_nogo"
    return "across"


def _nanmean(arr: np.ndarray) -> float:
    if arr.size == 0:
        return float("nan")
    with np.errstate(invalid="ignore"):
        return float(np.nanmean(arr))


def _nanvar(arr: np.ndarray) -> float:
    if arr.size < 2:
        return float("nan")
    with np.errstate(invalid="ignore"):
        return float(np.nanvar(arr, ddof=1))


def _empty_separation_summary(n_stimuli: int = 0) -> dict[str, float | int]:
    return {
        "within_go": np.nan,
        "within_nogo": np.nan,
        "within_avg": np.nan,
        "across": np.nan,
        "categoricality_index": np.nan,
        "separation_index": np.nan,
        "d_prime_pairs": np.nan,
        "var_within_go": np.nan,
        "var_within_nogo": np.nan,
        "var_across": np.nan,
        "n_pairs_within_go": 0,
        "n_pairs_within_nogo": 0,
        "n_pairs_across": 0,
        "n_stimuli": int(n_stimuli),
    }


# MDS converts similarity r in [-1, 1] -> dissimilarity in [0, 1]:
#   d(i, j) = (1 - r(i, j)) / 2
def _rsa_to_dissimilarity(rsa_df: pd.DataFrame) -> pd.DataFrame:
    """Convert correlation similarity to a [0, 1] dissimilarity matrix."""
    values = (1.0 - rsa_df.to_numpy(dtype=float)) / 2.0
    # Numerical jitter can make the diagonal slightly nonzero -> force 0.
    np.fill_diagonal(values, 0.0)
    values = np.clip(values, 0.0, None)
    # Enforce symmetry; corrcoef can drift by ~1e-16.
    values = 0.5 * (values + values.T)
    return pd.DataFrame(values, index=rsa_df.index, columns=rsa_df.columns)


def extract_category_pair_values(
    rsa_df: pd.DataFrame,
    *,
    low_boundary: float,
    high_boundary: float,
) -> dict[str, Any]:
    """Extract off-diagonal stimulus pairs and session-level separation summaries.

    Returns
    -------
    dict with keys ``pair_values`` (DataFrame) and ``summary`` (metric dict).
    """
    empty_pairs = pd.DataFrame(
        columns=["stim_i", "stim_j", "cat_i", "cat_j", "pair_type", "similarity", "dissimilarity"]
    )
    if rsa_df.empty or rsa_df.shape[0] < 2:
        return {"pair_values": empty_pairs, "summary": _empty_separation_summary(rsa_df.shape[0])}

    stim = np.asarray(rsa_df.index, dtype=float)
    cats = np.asarray([_stimulus_category(k, low_boundary, high_boundary) for k in stim])
    values = rsa_df.to_numpy(dtype=float)
    n = values.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    pair_sim = values[iu, ju]
    pair_cat_i = cats[iu]
    pair_cat_j = cats[ju]
    pair_types = np.array(
        [_pair_type_from_categories(ci, cj) for ci, cj in zip(pair_cat_i, pair_cat_j)],
        dtype=object,
    )
    pair_dissim = (1.0 - pair_sim) / 2.0

    pair_values = pd.DataFrame({
        "stim_i": stim[iu],
        "stim_j": stim[ju],
        "cat_i": pair_cat_i,
        "cat_j": pair_cat_j,
        "pair_type": pair_types,
        "similarity": pair_sim,
        "dissimilarity": pair_dissim,
    })

    within_go_mask = pair_types == "within_go"
    within_nogo_mask = pair_types == "within_nogo"
    across_mask = pair_types == "across"

    within_go_sim = pair_sim[within_go_mask]
    within_nogo_sim = pair_sim[within_nogo_mask]
    across_sim = pair_sim[across_mask]
    within_go_dissim = pair_dissim[within_go_mask]
    within_nogo_dissim = pair_dissim[within_nogo_mask]
    across_dissim = pair_dissim[across_mask]

    within_go = _nanmean(within_go_sim)
    within_nogo = _nanmean(within_nogo_sim)
    within_avg = _nanmean(np.concatenate([within_go_sim, within_nogo_sim]))
    across = _nanmean(across_sim)
    cat_idx = within_avg - across if np.isfinite(within_avg) and np.isfinite(across) else np.nan

    within_dissim_all = np.concatenate([within_go_dissim, within_nogo_dissim])
    within_dissim_mean = _nanmean(within_dissim_all)
    across_dissim_mean = _nanmean(across_dissim)
    denom = across_dissim_mean + within_dissim_mean
    separation_index = (
        (across_dissim_mean - within_dissim_mean) / denom
        if np.isfinite(denom) and denom > 0
        else np.nan
    )

    # Cohen's d on pair-level dissimilarities (within vs across).
    n_w = int(within_dissim_all.size)
    n_a = int(across_dissim.size)
    d_prime_pairs = np.nan
    if n_w >= 2 and n_a >= 2:
        var_w = _nanvar(within_dissim_all)
        var_a = _nanvar(across_dissim)
        if np.isfinite(var_w) and np.isfinite(var_a):
            pooled_var = ((n_w - 1) * var_w + (n_a - 1) * var_a) / (n_w + n_a - 2)
            if pooled_var > 0:
                d_prime_pairs = float(
                    (across_dissim_mean - within_dissim_mean) / np.sqrt(pooled_var)
                )

    summary = {
        "within_go": within_go,
        "within_nogo": within_nogo,
        "within_avg": within_avg,
        "across": across,
        "categoricality_index": cat_idx,
        "separation_index": separation_index,
        "d_prime_pairs": d_prime_pairs,
        "var_within_go": _nanvar(within_go_sim),
        "var_within_nogo": _nanvar(within_nogo_sim),
        "var_across": _nanvar(across_sim),
        "n_pairs_within_go": int(within_go_mask.sum()),
        "n_pairs_within_nogo": int(within_nogo_mask.sum()),
        "n_pairs_across": int(across_mask.sum()),
        "n_stimuli": int(n),
    }
    return {"pair_values": pair_values, "summary": summary}


def compute_categoricality_index(
    rsa_df: pd.DataFrame,
    *,
    low_boundary: float,
    high_boundary: float,
) -> dict[str, float | int]:
    """Compute within / across category similarities and separation indices.

    Backward-compatible wrapper around :func:`extract_category_pair_values`.
    """
    return extract_category_pair_values(
        rsa_df, low_boundary=low_boundary, high_boundary=high_boundary
    )["summary"]


# --- MDS embedding ------------------------------------------------------------

# This keeps anti-correlated stimuli as far apart as identical stimuli are close,
# and bounds distances in [0, 1] so optimization is well-conditioned.
def _largest_finite_stimulus_subset(dissim: pd.DataFrame) -> list[float]:
    """Return the largest stimulus set with all pairwise dissimilarities observed.

    Stage-mean RSA is built on the union of per-session tone ladders, so many
    off-diagonal cells are NaN (no session contained both frequencies). MDS needs
    a complete matrix; this picks the largest fully observed submatrix via a
    greedy max-clique search on the finite-pair graph.
    """
    if dissim.empty:
        return []

    labels = [float(x) for x in dissim.index]
    n = len(labels)
    finite = np.isfinite(dissim.to_numpy(dtype=float))
    np.fill_diagonal(finite, True)

    # Seed order by node degree so the greedy expansion finds a large clique.
    order = np.argsort(-finite.sum(axis=1))
    best_idx: list[int] = []
    for seed in order:
        selected = [int(seed)]
        for j in order:
            j = int(j)
            if j in selected:
                continue
            if all(finite[i, j] for i in selected):
                selected.append(j)
        if len(selected) > len(best_idx):
            best_idx = selected

    return [labels[i] for i in best_idx]


def compute_mds_from_rsa(
    rsa_df: pd.DataFrame,
    *,
    n_components: int = 2,
    random_state: int = 42,
    n_init: int = 8,
    max_iter: int = 500,
) -> tuple[pd.DataFrame, float, dict[str, int]]:
    """Run metric MDS on the dissimilarity matrix derived from ``rsa_df``.

    When the input matrix has NaNs (typical for stage means over sessions with
    different tone ladders), embeds the largest fully observed stimulus subset.

    Returns the 2D embedding indexed by kHz, stress, and embedding metadata.
    """
    empty_meta = {"n_stimuli_input": int(rsa_df.shape[0]), "n_stimuli_embedded": 0}
    if rsa_df.empty or rsa_df.shape[0] < 2:
        return (
            pd.DataFrame(columns=[f"mds_{i + 1}" for i in range(n_components)]),
            float("nan"),
            empty_meta,
        )

    dissim = _rsa_to_dissimilarity(rsa_df)
    kept_labels = _largest_finite_stimulus_subset(dissim)
    meta = {
        "n_stimuli_input": int(rsa_df.shape[0]),
        "n_stimuli_embedded": int(len(kept_labels)),
    }
    if len(kept_labels) < 2:
        return (
            pd.DataFrame(columns=[f"mds_{i + 1}" for i in range(n_components)]),
            float("nan"),
            meta,
        )

    label_to_pos = {float(label): pos for pos, label in enumerate(dissim.index)}
    idx = [label_to_pos[float(k)] for k in kept_labels]
    sub_dissim = dissim.to_numpy(dtype=float)[np.ix_(idx, idx)]

    mds = MDS(
        n_components=n_components,
        dissimilarity="precomputed",
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
        normalized_stress="auto",
    )
    coords = mds.fit_transform(sub_dissim)
    coord_cols = [f"mds_{i + 1}" for i in range(n_components)]
    coords_df = pd.DataFrame(coords, index=kept_labels, columns=coord_cols)
    return coords_df, float(mds.stress_), meta


# --- Plot helpers -------------------------------------------------------------

def plot_rsa_heatmap(
    rsa_df: pd.DataFrame,
    *,
    title: str,
    low_boundary: float,
    high_boundary: float,
    n_sessions: pd.DataFrame | None = None,
) -> go.Figure:
    """Heatmap of an RSA matrix with category-boundary guides."""
    if rsa_df.empty:
        fig = go.Figure()
        fig.update_layout(title=f"{title} — no data", height=320)
        return fig

    stim = [float(k) for k in rsa_df.index]
    z = rsa_df.to_numpy(dtype=float)
    text = None
    hover_extra = ""
    if n_sessions is not None and not n_sessions.empty:
        text = n_sessions.to_numpy(dtype=int).astype(str)
        hover_extra = "<br>n_sessions=%{text}"
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=stim,
            y=stim,
            zmin=-1,
            zmax=1,
            colorscale="RdBu_r",
            reversescale=False,
            zmid=0,
            colorbar=dict(title="r"),
            text=text,
            hovertemplate=(
                "stim_x=%{x:.3f} kHz<br>stim_y=%{y:.3f} kHz<br>r=%{z:.2f}" + hover_extra + "<extra></extra>"
            ),
        )
    )
    # Category boundary guides (log-kHz axes).
    for boundary, color in ((low_boundary, COLOR_LOW_BD), (high_boundary, COLOR_HIGH_BD)):
        fig.add_vline(x=boundary, line=dict(color=color, dash="dash", width=1))
        fig.add_hline(y=boundary, line=dict(color=color, dash="dash", width=1))
    fig.update_layout(
        title=title,
        height=360,
        xaxis=dict(title="Stimulus (kHz)", type="log", scaleanchor="y", constrain="domain"),
        yaxis=dict(title="Stimulus (kHz)", type="log", autorange="reversed"),
    )
    return fig


def plot_rsa_difference_heatmap(
    diff_df: pd.DataFrame,
    *,
    title: str,
    low_boundary: float,
    high_boundary: float,
    height: int = 520,
) -> go.Figure:
    """Diverging heatmap of an RSA difference matrix (e.g. 2b Expert − Novice)."""
    if diff_df.empty:
        fig = go.Figure()
        fig.update_layout(title=f"{title} — no data", height=height)
        return fig

    stim = [float(k) for k in diff_df.index]
    z = diff_df.to_numpy(dtype=float)
    z_abs = np.nanmax(np.abs(z)) if np.isfinite(z).any() else 1.0
    z_lim = float(max(z_abs, 0.05))

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=stim,
            y=stim,
            zmin=-z_lim,
            zmax=z_lim,
            colorscale="RdBu_r",
            reversescale=False,
            zmid=0,
            colorbar=dict(title="Δr"),
            hovertemplate=(
                "stim_x=%{x:.3f} kHz<br>stim_y=%{y:.3f} kHz<br>Δr=%{z:.2f}<extra></extra>"
            ),
        )
    )
    for boundary, color in ((low_boundary, COLOR_LOW_BD), (high_boundary, COLOR_HIGH_BD)):
        fig.add_vline(x=boundary, line=dict(color=color, dash="dash", width=1))
        fig.add_hline(y=boundary, line=dict(color=color, dash="dash", width=1))
    fig.update_layout(
        title=title,
        height=height,
        xaxis=dict(title="Stimulus (kHz)", type="log", scaleanchor="y", constrain="domain"),
        yaxis=dict(title="Stimulus (kHz)", type="log", autorange="reversed"),
    )
    return fig


def _per_session_rsa_lookup(
    per_session_rsa: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Map (session label, area) -> per-session RSA record."""
    return {(rec["session"], rec["area"]): rec for rec in per_session_rsa}


def render_per_session_rsa_navigator(
    per_session_rsa: list[dict[str, Any]],
    summary_df: pd.DataFrame,
    *,
    low_boundary_khz: float,
    high_boundary_khz: float,
) -> None:
    """Prev/next and dropdown navigation over individual session RSA heatmaps."""
    if not per_session_rsa:
        return

    st.subheader("Single-session RSA navigator")
    st.caption("Browse per-session RSA matrices from the current run (ACx and OFC side by side).")

    stages_present = sorted({rec["learning_stage"] for rec in per_session_rsa})
    stage_options = ["All stages"] + [s for s in STAGE_ORDER if s in stages_present]
    if len(stage_options) == 1:
        stage_options.extend(stages_present)

    filt_col1, filt_col2 = st.columns(2)
    with filt_col1:
        stage_filter = st.selectbox("Learning stage", stage_options, key="rsa_nav_stage")
    with filt_col2:
        area_view = st.selectbox("Areas shown", ["ACx and OFC", "ACx only", "OFC only"], key="rsa_nav_area")

    session_labels: list[str] = []
    seen: set[str] = set()
    for rec in per_session_rsa:
        if stage_filter != "All stages" and rec["learning_stage"] != stage_filter:
            continue
        label = rec["session"]
        if label not in seen:
            seen.add(label)
            session_labels.append(label)
    session_labels.sort()

    if not session_labels:
        st.info("No sessions match the current filters.")
        return

    idx_key = "rsa_single_session_idx"
    if idx_key not in st.session_state:
        st.session_state[idx_key] = 0
    st.session_state[idx_key] = int(np.clip(st.session_state[idx_key], 0, len(session_labels) - 1))

    nav_prev, nav_select, nav_next = st.columns([1, 8, 1])
    with nav_prev:
        if st.button("◀ Prev", key="rsa_nav_prev", disabled=st.session_state[idx_key] <= 0):
            st.session_state[idx_key] -= 1
            st.rerun()
    with nav_next:
        if st.button(
            "Next ▶",
            key="rsa_nav_next",
            disabled=st.session_state[idx_key] >= len(session_labels) - 1,
        ):
            st.session_state[idx_key] += 1
            st.rerun()
    with nav_select:
        selected_session = st.selectbox(
            "Session",
            session_labels,
            index=st.session_state[idx_key],
            key="rsa_nav_session_select",
        )
        st.session_state[idx_key] = session_labels.index(selected_session)

    lookup = _per_session_rsa_lookup(per_session_rsa)
    rec0 = next(iter(per_session_rsa))
    learning_stage = lookup.get((selected_session, "ACx"), lookup.get((selected_session, "OFC"), rec0))[
        "learning_stage"
    ]
    stage_color = LEARNING_STAGE_COLORS.get(learning_stage, (COLOR_GRAY, COLOR_GRAY))[0]
    st.markdown(
        f"<span style='color:{stage_color}'>●</span> **{learning_stage}** · "
        f"Session {st.session_state[idx_key] + 1} of {len(session_labels)}",
        unsafe_allow_html=True,
    )

    areas_to_plot = ("ACx", "OFC")
    if area_view == "ACx only":
        areas_to_plot = ("ACx",)
    elif area_view == "OFC only":
        areas_to_plot = ("OFC",)

    plot_cols = st.columns(len(areas_to_plot))
    for col, area in zip(plot_cols, areas_to_plot):
        rec = lookup.get((selected_session, area))
        with col:
            if rec is None:
                st.caption(f"{area}: no RSA matrix for this session.")
                continue
            rsa_df = rec["rsa_df"]
            n_stim = rsa_df.shape[0]
            title = f"{area} ({n_stim} stim)"
            fig = plot_rsa_heatmap(
                rsa_df,
                title=title,
                low_boundary=low_boundary_khz,
                high_boundary=high_boundary_khz,
            )
            st.plotly_chart(fig, use_container_width=True)

    if not summary_df.empty:
        sess_metrics = summary_df[summary_df["session"] == selected_session]
        if area_view == "ACx only":
            sess_metrics = sess_metrics[sess_metrics["area"] == "ACx"]
        elif area_view == "OFC only":
            sess_metrics = sess_metrics[sess_metrics["area"] == "OFC"]
        metric_cols = [
            "area", "n_units", "n_trials", "n_stimuli",
            "categoricality_index", "separation_index", "within_go", "within_nogo", "across",
        ]
        present = [c for c in metric_cols if c in sess_metrics.columns]
        if present and not sess_metrics.empty:
            st.caption("Session metrics")
            st.dataframe(
                sess_metrics[present].reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
            )


def plot_mds_embedding(
    coords_df: pd.DataFrame,
    *,
    title: str,
    low_boundary: float,
    high_boundary: float,
    stress: float | None = None,
) -> go.Figure:
    """Scatter the 2D MDS coordinates, colored by Go/No-Go category, labeled with kHz.

    Adjacent stimuli are connected with a thin grey line ordered by kHz so the
    geometric continuity (or its breakdown into category clusters) is visible.
    """
    if coords_df.empty or coords_df.shape[0] < 2:
        fig = go.Figure()
        fig.update_layout(title=f"{title} — no data", height=320)
        return fig

    # Sort by kHz to draw the connecting "frequency path" left-to-right.
    coords_sorted = coords_df.sort_index()
    stim = np.asarray(coords_sorted.index, dtype=float)
    x = coords_sorted.iloc[:, 0].to_numpy()
    y = coords_sorted.iloc[:, 1].to_numpy()
    categories = np.asarray(
        [_stimulus_category(k, low_boundary, high_boundary) for k in stim]
    )

    fig = go.Figure()
    # Connecting line in stimulus order (frequency continuity).
    fig.add_trace(
        go.Scatter(
            x=x, y=y, mode="lines",
            line=dict(color=COLOR_GRAY, width=1, dash="dot"),
            hoverinfo="skip", showlegend=False,
        )
    )
    # Markers per category so the legend is meaningful.
    for cat, color in (("Go", COLOR_GO), ("NoGo", COLOR_NOGO)):
        mask = categories == cat
        if not mask.any():
            continue
        fig.add_trace(
            go.Scatter(
                x=x[mask], y=y[mask],
                mode="markers+text",
                marker=dict(color=color, size=14, line=dict(color="white", width=1)),
                text=[f"{k:.2f}" for k in stim[mask]],
                textposition="top center",
                textfont=dict(size=10),
                name=cat,
                hovertemplate="stim=%{text} kHz<br>MDS1=%{x:.2f}<br>MDS2=%{y:.2f}<extra></extra>",
            )
        )

    plot_title = title
    if stress is not None and np.isfinite(stress):
        plot_title = f"{title} (stress={stress:.3f})"
    fig.update_layout(
        title=plot_title,
        height=360,
        xaxis=dict(title="MDS dim 1", zeroline=True, scaleanchor="y", scaleratio=1.0),
        yaxis=dict(title="MDS dim 2", zeroline=True),
        showlegend=True,
    )
    return fig


def plot_categoricality_panel(
    summary_df: pd.DataFrame,
) -> go.Figure:
    """Grouped box plot of categoricality_index per (stage x area).

    Falls back gracefully if some stages or areas have no sessions.
    """
    if summary_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No per-session categoricality data.", height=320)
        return fig

    stage_order = [s for s in STAGE_ORDER if s in summary_df["learning_stage"].unique()]
    fig = go.Figure()
    for area, color in (("ACx", COLOR_ACX), ("OFC", COLOR_OFC)):
        sub = summary_df[summary_df["area"] == area]
        if sub.empty:
            continue
        fig.add_trace(
            go.Box(
                x=sub["learning_stage"],
                y=sub["categoricality_index"],
                name=area,
                marker_color=color,
                boxpoints="all",
                jitter=0.4,
                pointpos=0,
                line=dict(width=1.2),
            )
        )
    fig.add_hline(y=0, line=dict(color=COLOR_GRAY, dash="dot", width=1))
    fig.update_layout(
        title="Categoricality Index (within - across) by stage and area",
        boxmode="group",
        xaxis=dict(title="Learning stage", categoryorder="array", categoryarray=stage_order),
        yaxis=dict(title="Categoricality index (Pearson r)"),
        height=420,
    )
    return fig


def plot_within_vs_across_panel(summary_df: pd.DataFrame) -> go.Figure:
    """Side-by-side within-category vs across-category similarity per (stage x area)."""
    if summary_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No per-session within/across data.", height=320)
        return fig

    stage_order = [s for s in STAGE_ORDER if s in summary_df["learning_stage"].unique()]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Within-category", "Across-category"))
    for area, color in (("ACx", COLOR_ACX), ("OFC", COLOR_OFC)):
        sub = summary_df[summary_df["area"] == area]
        if sub.empty:
            continue
        fig.add_trace(
            go.Box(
                x=sub["learning_stage"], y=sub["within_avg"], name=f"{area} within",
                marker_color=color, boxpoints="all", jitter=0.4, pointpos=0,
                legendgroup=area, showlegend=True,
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Box(
                x=sub["learning_stage"], y=sub["across"], name=f"{area} across",
                marker_color=color, boxpoints="all", jitter=0.4, pointpos=0,
                legendgroup=area, showlegend=False,
            ),
            row=1, col=2,
        )
    for col in (1, 2):
        fig.update_xaxes(categoryorder="array", categoryarray=stage_order, row=1, col=col)
        fig.update_yaxes(title_text="Pearson r", range=[-0.5, 1.0], row=1, col=col)
    fig.update_layout(boxmode="group", height=420, title="Within- vs Across-category similarity")
    return fig


# --- Panel H: category separation plots & stats --------------------------------

def build_pooled_pairs_df(per_session_pairs: list[dict[str, Any]]) -> pd.DataFrame:
    """Concatenate per-session pair tables with session metadata for plotting."""
    frames: list[pd.DataFrame] = []
    for rec in per_session_pairs:
        pair_df = rec.get("pair_values")
        if pair_df is None or pair_df.empty:
            continue
        tagged = pair_df.copy()
        tagged["session"] = rec["session"]
        tagged["learning_stage"] = rec["learning_stage"]
        tagged["area"] = rec["area"]
        tagged["pair_type_label"] = tagged["pair_type"].map(PAIR_TYPE_LABELS)
        frames.append(tagged)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def plot_pair_type_violin(pooled_pairs_df: pd.DataFrame) -> go.Figure:
    """Violin + strip plot of pair similarities by pair type, stage, and area."""
    if pooled_pairs_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No pair-level data for violin plot.", height=320)
        return fig

    stage_order = [s for s in STAGE_ORDER if s in pooled_pairs_df["learning_stage"].unique()]
    pair_order = [PAIR_TYPE_LABELS[k] for k in ("within_go", "within_nogo", "across")]
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("ACx", "OFC"),
        shared_yaxes=True,
    )
    pair_colors = {"Go-Go": COLOR_GO, "NoGo-NoGo": COLOR_NOGO, "Go-NoGo": COLOR_GRAY}
    for col_idx, area in enumerate(("ACx", "OFC"), start=1):
        sub = pooled_pairs_df[pooled_pairs_df["area"] == area]
        if sub.empty:
            continue
        for pair_label in pair_order:
            pair_sub = sub[sub["pair_type_label"] == pair_label]
            if pair_sub.empty:
                continue
            fig.add_trace(
                go.Violin(
                    x=pair_sub["learning_stage"],
                    y=pair_sub["similarity"],
                    name=pair_label if col_idx == 1 else pair_label,
                    legendgroup=pair_label,
                    showlegend=(col_idx == 1),
                    line_color=pair_colors.get(pair_label, COLOR_GRAY),
                    fillcolor=pair_colors.get(pair_label, COLOR_GRAY),
                    opacity=0.55,
                    box_visible=True,
                    meanline_visible=True,
                    points="all",
                    jitter=0.3,
                    pointpos=0,
                    side="positive",
                ),
                row=1, col=col_idx,
            )
        fig.update_xaxes(
            categoryorder="array", categoryarray=stage_order, title_text="Learning stage",
            row=1, col=col_idx,
        )
    fig.update_yaxes(title_text="Pearson r", range=[-0.5, 1.0], row=1, col=1)
    fig.update_layout(
        title="Pair-type similarity distributions (session-pooled pairs)",
        height=480,
        violinmode="group",
    )
    return fig


def plot_pair_type_kde(pooled_pairs_df: pd.DataFrame) -> go.Figure:
    """KDE overlay of pair similarities per (area, stage, pair type)."""
    from scipy.stats import gaussian_kde

    if pooled_pairs_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No pair-level data for KDE plot.", height=320)
        return fig

    stage_order = [s for s in STAGE_ORDER if s in pooled_pairs_df["learning_stage"].unique()]
    if not stage_order:
        fig = go.Figure()
        fig.update_layout(title="No stages in pair data.", height=320)
        return fig

    fig = make_subplots(
        rows=len(stage_order), cols=2,
        subplot_titles=[f"{stage} — {area}" for stage in stage_order for area in ("ACx", "OFC")],
        vertical_spacing=0.08,
        shared_xaxes=True,
    )
    pair_colors = {"within_go": COLOR_GO, "within_nogo": COLOR_NOGO, "across": COLOR_GRAY}
    x_grid = np.linspace(-0.5, 1.0, 200)

    for row_idx, stage in enumerate(stage_order, start=1):
        for col_idx, area in enumerate(("ACx", "OFC"), start=1):
            sub = pooled_pairs_df[
                (pooled_pairs_df["learning_stage"] == stage) & (pooled_pairs_df["area"] == area)
            ]
            for pair_type in ("within_go", "within_nogo", "across"):
                vals = pd.to_numeric(
                    sub.loc[sub["pair_type"] == pair_type, "similarity"], errors="coerce"
                ).dropna().to_numpy()
                if vals.size < 2:
                    continue
                try:
                    kde = gaussian_kde(vals)
                    y = kde(x_grid)
                except (np.linalg.LinAlgError, ValueError):
                    continue
                show_legend = row_idx == 1 and col_idx == 1
                fig.add_trace(
                    go.Scatter(
                        x=x_grid, y=y,
                        mode="lines",
                        name=PAIR_TYPE_LABELS[pair_type],
                        legendgroup=pair_type,
                        showlegend=show_legend,
                        line=dict(color=pair_colors[pair_type], width=2),
                    ),
                    row=row_idx, col=col_idx,
                )
            fig.update_xaxes(range=[-0.5, 1.0], row=row_idx, col=col_idx)
            if col_idx == 1:
                fig.update_yaxes(title_text=stage, row=row_idx, col=col_idx)
    fig.update_layout(
        title="Pair-type similarity KDEs by stage and area",
        height=140 * len(stage_order) + 80,
    )
    return fig


def plot_separation_index_panel(summary_df: pd.DataFrame) -> go.Figure:
    """Box plot of normalized separation index (CCI) per stage and area."""
    if summary_df.empty or "separation_index" not in summary_df.columns:
        fig = go.Figure()
        fig.update_layout(title="No separation index data.", height=320)
        return fig

    stage_order = [s for s in STAGE_ORDER if s in summary_df["learning_stage"].unique()]
    fig = go.Figure()
    for area, color in (("ACx", COLOR_ACX), ("OFC", COLOR_OFC)):
        sub = summary_df[summary_df["area"] == area]
        if sub.empty:
            continue
        fig.add_trace(
            go.Box(
                x=sub["learning_stage"],
                y=sub["separation_index"],
                name=area,
                marker_color=color,
                boxpoints="all",
                jitter=0.4,
                pointpos=0,
                line=dict(width=1.2),
            )
        )
    fig.add_hline(y=0, line=dict(color=COLOR_GRAY, dash="dot", width=1))
    fig.update_layout(
        title="Separation Index (CCI): (across_dissim − within_dissim) / (across_dissim + within_dissim)",
        boxmode="group",
        xaxis=dict(title="Learning stage", categoryorder="array", categoryarray=stage_order),
        yaxis=dict(title="Separation index", range=[-1, 1]),
        height=420,
    )
    return fig


def plot_within_across_detailed(summary_df: pd.DataFrame) -> go.Figure:
    """Three-way within Go-Go / NoGo-NoGo / Go-NoGo session-level means by stage."""
    if summary_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No per-session within/across data.", height=320)
        return fig

    stage_order = [s for s in STAGE_ORDER if s in summary_df["learning_stage"].unique()]
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Go-Go", "NoGo-NoGo", "Go-NoGo"),
    )
    metric_cols = ("within_go", "within_nogo", "across")
    for col_idx, metric_col in enumerate(metric_cols, start=1):
        for area, color in (("ACx", COLOR_ACX), ("OFC", COLOR_OFC)):
            sub = summary_df[summary_df["area"] == area]
            if sub.empty or metric_col not in sub.columns:
                continue
            fig.add_trace(
                go.Box(
                    x=sub["learning_stage"], y=sub[metric_col],
                    name=area if col_idx == 1 else area,
                    marker_color=color,
                    boxpoints="all", jitter=0.4, pointpos=0,
                    legendgroup=area, showlegend=(col_idx == 1),
                ),
                row=1, col=col_idx,
            )
        fig.update_xaxes(categoryorder="array", categoryarray=stage_order, row=1, col=col_idx)
        fig.update_yaxes(title_text="Pearson r", range=[-0.5, 1.0], row=1, col=col_idx)
    fig.update_layout(boxmode="group", height=420, title="Within- and across-category similarity (3 pair types)")
    return fig


def summarize_category_separation(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Cohort-level mean / SEM / median per (learning_stage, area)."""
    metric_cols = [
        "within_go", "within_nogo", "within_avg", "across",
        "categoricality_index", "separation_index", "d_prime_pairs",
        "var_within_go", "var_within_nogo", "var_across",
    ]
    if summary_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for (stage, area), group in summary_df.groupby(["learning_stage", "area"], sort=False):
        row: dict[str, Any] = {
            "learning_stage": stage,
            "area": area,
            "n_sessions": int(len(group)),
        }
        for metric in metric_cols:
            if metric not in group.columns:
                continue
            vals = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(vals)) if vals.size else np.nan
            row[f"{metric}_sem"] = float(np.std(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else np.nan
            row[f"{metric}_median"] = float(np.median(vals)) if vals.size else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _p_value_to_sig(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _separation_groups_by_stage(
    summary_df: pd.DataFrame,
    area: str,
    metric_col: str,
    *,
    min_n: int = 2,
    stages: tuple[str, ...] = STAGE_ORDER,
) -> dict[str, np.ndarray]:
    groups: dict[str, np.ndarray] = {}
    sub = summary_df[summary_df["area"] == area]
    for stage in stages:
        vals = pd.to_numeric(
            sub.loc[sub["learning_stage"] == stage, metric_col], errors="coerce"
        ).dropna()
        if len(vals) >= min_n:
            groups[stage] = vals.to_numpy(dtype=float)
    return groups


def run_separation_stage_tests(
    summary_df: pd.DataFrame,
    metric_col: str,
    *,
    min_n: int = 2,
    stages: tuple[str, ...] = STAGE_ORDER,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Kruskal-Wallis across stages + pairwise Mann-Whitney U per area (Bonferroni)."""
    from itertools import combinations

    from scipy.stats import kruskal, mannwhitneyu

    omnibus_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []
    metric_label = metric_col

    for area in sorted(summary_df["area"].dropna().astype(str).unique()):
        groups = _separation_groups_by_stage(
            summary_df, area, metric_col, min_n=min_n, stages=stages
        )
        testable = [s for s in stages if s in groups]
        if len(testable) < 2:
            continue

        if len(testable) >= 3:
            stat, p = kruskal(*(groups[g] for g in testable))
            omnibus_rows.append({
                "metric": metric_label,
                "area": area,
                "test": "Kruskal-Wallis",
                "groups": ", ".join(testable),
                "statistic": float(stat),
                "p": float(p),
                "sig": _p_value_to_sig(float(p)),
            })

        pair_stats: list[dict[str, Any]] = []
        for group_a, group_b in combinations(testable, 2):
            vals_a = groups[group_a]
            vals_b = groups[group_b]
            try:
                stat_u, p = mannwhitneyu(vals_a, vals_b, alternative="two-sided")
            except ValueError:
                stat_u, p = np.nan, np.nan
            pair_stats.append({
                "metric": metric_label,
                "area": area,
                "group_a": group_a,
                "group_b": group_b,
                "n_a": len(vals_a),
                "n_b": len(vals_b),
                "mean_a": float(np.mean(vals_a)),
                "mean_b": float(np.mean(vals_b)),
                "U": float(stat_u) if np.isfinite(stat_u) else np.nan,
                "p": float(p) if np.isfinite(p) else np.nan,
            })

        n_tests = sum(1 for row in pair_stats if np.isfinite(row["p"]))
        for row in pair_stats:
            p_raw = row["p"]
            p_adj = min(1.0, p_raw * n_tests) if np.isfinite(p_raw) and n_tests > 0 else np.nan
            pairwise_rows.append({
                **row,
                "p_adj": p_adj,
                "sig": _p_value_to_sig(p_adj) if np.isfinite(p_adj) else "",
            })

    return pd.DataFrame(omnibus_rows), pd.DataFrame(pairwise_rows)


def render_separation_stats_panel(summary_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Render omnibus and pairwise stage-comparison tables in Streamlit."""
    all_omnibus: list[pd.DataFrame] = []
    all_pairwise: list[pd.DataFrame] = []
    for metric_col in SEPARATION_TEST_METRICS:
        if metric_col not in summary_df.columns:
            continue
        omnibus_df, pairwise_df = run_separation_stage_tests(summary_df, metric_col)
        if not omnibus_df.empty:
            all_omnibus.append(omnibus_df)
        if not pairwise_df.empty:
            all_pairwise.append(pairwise_df)
        st.markdown(f"**{metric_col}**")
        if omnibus_df.empty and pairwise_df.empty:
            st.caption("Not enough sessions per stage for statistical tests.")
            continue
        if not omnibus_df.empty:
            st.caption("Omnibus (Kruskal-Wallis across learning stages, per area)")
            st.dataframe(omnibus_df, use_container_width=True, hide_index=True)
        if not pairwise_df.empty:
            st.caption("Pairwise (Mann-Whitney U, Bonferroni-corrected, per area)")
            st.dataframe(pairwise_df, use_container_width=True, hide_index=True)

    return (
        pd.concat(all_omnibus, ignore_index=True) if all_omnibus else pd.DataFrame(),
        pd.concat(all_pairwise, ignore_index=True) if all_pairwise else pd.DataFrame(),
    )


# --- Result persistence -------------------------------------------------------

def _safe_filename_token(text: str, *, max_len: int = 80) -> str:
    """Filesystem-safe slug for session labels used in per-session RSA CSV names."""
    slug = re.sub(r"[^\w.\-]+", "_", str(text).strip())
    return slug[:max_len].strip("_") or "session"


def _stage_area_slug(stage: str, area: str) -> str:
    return f"{stage.replace(' ', '_')}_{area}"


def save_rsa_results(
    *,
    summary_df: pd.DataFrame,
    failures: list[dict[str, str]],
    rsa_by_cell: dict[tuple[str, str], list[pd.DataFrame]],
    per_session_rsa: list[dict[str, Any]],
    per_session_pairs: list[dict[str, Any]] | None = None,
    run_settings: dict[str, Any],
    low_boundary_khz: float,
    high_boundary_khz: float,
) -> Path:
    """Write RSA outputs and append failures to the persistent error log.

    Files written under ``RSA_RESULTS_DIR``:
    - ``rsa_per_session_metrics.csv`` — categoricality / separation table (current run).
    - ``rsa_cohort_separation_summary.csv`` — stage × area metric summaries.
    - ``rsa_separation_stats_omnibus.csv`` / ``rsa_separation_stats_pairwise.csv``.
    - ``rsa_errors.csv`` — failures from the current run only.
    - ``rsa_error_log.csv`` — append-only error history (timestamped rows).
    - ``rsa_run_metadata.json`` — decode window, aggregation, boundaries, UTC time.
    - ``cohort_mean_rsa_{stage}_{area}.csv`` — stage-averaged matrices.
    - ``cohort_mds_{stage}_{area}.csv`` — 2D MDS coordinates + stress column.
    - ``per_session/*.csv`` — one RSA matrix per successful (session, area).
    - ``per_session/category_pairs/*.csv`` — pair-level tables per session.
    """
    RSA_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    if not summary_df.empty:
        summary_df.to_csv(RSA_RESULTS_DIR / "rsa_per_session_metrics.csv", index=False)
        cohort_summary = summarize_category_separation(summary_df)
        if not cohort_summary.empty:
            cohort_summary.to_csv(RSA_RESULTS_DIR / "rsa_cohort_separation_summary.csv", index=False)
        omnibus_df, pairwise_df = run_separation_stage_tests(summary_df, SEPARATION_TEST_METRICS[0])
        for metric_col in SEPARATION_TEST_METRICS[1:]:
            if metric_col in summary_df.columns:
                om, pw = run_separation_stage_tests(summary_df, metric_col)
                omnibus_df = pd.concat([omnibus_df, om], ignore_index=True)
                pairwise_df = pd.concat([pairwise_df, pw], ignore_index=True)
        if not omnibus_df.empty:
            omnibus_df.to_csv(RSA_RESULTS_DIR / "rsa_separation_stats_omnibus.csv", index=False)
        if not pairwise_df.empty:
            pairwise_df.to_csv(RSA_RESULTS_DIR / "rsa_separation_stats_pairwise.csv", index=False)

    errors_df = pd.DataFrame(failures)
    if not errors_df.empty:
        errors_df.to_csv(RSA_RESULTS_DIR / "rsa_errors.csv", index=False)
        log_rows = errors_df.copy()
        log_rows.insert(0, "timestamp_utc", run_ts)
        log_header = not RSA_ERROR_LOG_PATH.is_file()
        log_rows.to_csv(
            RSA_ERROR_LOG_PATH,
            mode="a",
            header=log_header,
            index=False,
        )
    else:
        errors_path = RSA_RESULTS_DIR / "rsa_errors.csv"
        if errors_path.is_file():
            errors_path.unlink()

    metadata = {
        "timestamp_utc": run_ts,
        "results_dir": str(RSA_RESULTS_DIR),
        "low_boundary_khz": low_boundary_khz,
        "high_boundary_khz": high_boundary_khz,
        **run_settings,
        "n_per_session_metrics": int(len(summary_df)),
        "n_failures": int(len(failures)),
    }
    (RSA_RESULTS_DIR / "rsa_run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    per_session_dir = RSA_RESULTS_DIR / "per_session"
    per_session_dir.mkdir(parents=True, exist_ok=True)
    for rec in per_session_rsa:
        rsa_df: pd.DataFrame = rec["rsa_df"]
        fname = (
            f"{_safe_filename_token(rec['session'])}_{rec['area']}.csv"
        )
        rsa_df.to_csv(per_session_dir / fname)

    if per_session_pairs:
        pairs_dir = per_session_dir / "category_pairs"
        pairs_dir.mkdir(parents=True, exist_ok=True)
        for rec in per_session_pairs:
            pair_df: pd.DataFrame = rec.get("pair_values", pd.DataFrame())
            if pair_df.empty:
                continue
            fname = f"{_safe_filename_token(rec['session'])}_{rec['area']}_pairs.csv"
            pair_df.to_csv(pairs_dir / fname, index=False)

    for (stage, area), matrices in rsa_by_cell.items():
        slug = _stage_area_slug(stage, area)
        mean_rsa, n_sessions = aggregate_rsa_across_sessions(matrices)
        if not mean_rsa.empty:
            mean_rsa.to_csv(RSA_RESULTS_DIR / f"cohort_mean_rsa_{slug}.csv")
            n_sessions.to_csv(RSA_RESULTS_DIR / f"cohort_mean_rsa_{slug}_n_sessions.csv")
        coords_df, stress, _mds_meta = compute_mds_from_rsa(mean_rsa)
        if not coords_df.empty:
            coords_out = coords_df.copy()
            coords_out["mds_stress"] = stress
            coords_out.to_csv(RSA_RESULTS_DIR / f"cohort_mds_{slug}.csv")

    return RSA_RESULTS_DIR


def _render_rsa_methods_section(
    *,
    decode_window: tuple[float, float],
    aggregation: str,
    use_histology: bool,
    min_trials_per_stim: int,
    use_log_stimulus_bins: bool,
    low_boundary_khz: float,
    high_boundary_khz: float,
) -> None:
    """Collapsible methods summary for the Regional RSA page."""
    histology_note = (
        "Histology-matched good units + MUA only."
        if use_histology
        else "All Bombcell good units + MUA (no histology filter)."
    )
    binning_note = (
        f"Enabled: trial tones are assigned to a shared log-spaced grid with up to "
        f"{RSA_BINS_PER_CLASS} geometric bins per category band (edges anchored at "
        f"low/high boundaries), then trials are pooled per bin centre before RSA."
        if use_log_stimulus_bins
        else "Disabled: RSA is computed on raw per-kHz stimulus labels."
    )
    st.subheader("Methods")
    with st.expander("Regional RSA — analysis pipeline", expanded=False):
        st.markdown(
            f"""
**Cohorts.** Three matched categorization groups are analysed separately:
Novice — 2b Categorization, 1b Expert — 1b Categorization, and
2b Expert — 2b Categorization. Session is the independent statistical unit.

**Neural data.** Tone-aligned spike counts are loaded from ACx (`imec0`) and OFC (`imec1`).
{histology_note}

**Feature extraction.** Population activity is collapsed to `[trials × units]` by taking the
**{aggregation.lower()}** firing rate in a **{decode_window[0]:.2f}–{decode_window[1]:.2f} s**
window relative to tone onset.

**Stimulus inclusion and binning.**
- Go / No-Go categories: tones below **{low_boundary_khz:.3f} kHz** or above
  **{high_boundary_khz:.3f} kHz** are Go; tones between are No-Go.
- Frequency caps: Novice and 2b Expert ≤ **{DECI_KHZ_21_5_MAX_STIM_KHZ:g} kHz** (21.5 deci-kHz);
  1b Expert ≤ **high boundary** ({high_boundary_khz:.3f} kHz).
- {binning_note}
- Stimuli with fewer than **{min_trials_per_stim}** trials after pooling are excluded.

**RSA matrix (per session × area).** For each surviving stimulus, the mean population vector
is computed, each unit is z-scored across stimuli, and Pearson correlation between stimulus
columns yields a symmetric representational similarity matrix (*r* ∈ [−1, 1]).

**Cohort aggregation.** Session-level RSA matrices are element-wise averaged (nan-mean) onto the
union stimulus grid within each (learning stage × area) cell.

**Panel F — stage means and learning contrast.** Heatmaps show cohort-mean RSA. When both
cohorts are present, a side panel shows **2b Expert − Novice** (element-wise Δ*r* on the
aligned grid).

**MDS embedding.** Dissimilarity *d* = (1 − *r*) / 2. When the cohort-mean matrix is sparse
(sessions used different tone ladders), metric MDS is run on the **largest fully observed
stimulus subset** (all pairwise *d* finite).

**Category separation metrics (Panels G & H).** Off-diagonal RSA pairs are classified as
Go-Go, NoGo-NoGo, or Go-NoGo. Per session × area:
- **Categoricality index** = mean(within-category *r*) − mean(across-category *r*)
- **Separation index (CCI)** = (across *d* − within *d*) / (across *d* + within *d*)
- **d′ (pairs)** = Cohen's *d* on pair-level dissimilarities (within vs across)

**Statistics.** Nonparametric tests on session-level summaries, run separately for ACx and OFC:
Kruskal-Wallis across learning stages (≥3 groups) and pairwise Mann-Whitney *U* with
Bonferroni correction for `categoricality_index` and `separation_index`.
            """.strip()
        )


# --- Streamlit entry point ----------------------------------------------------

def _run_rsa_for_selection(
    selected_sessions_df: pd.DataFrame,
    *,
    decode_window: tuple[float, float],
    aggregation: str,
    use_histology: bool,
    min_trials_per_stim: int,
    use_log_stimulus_bins: bool = True,
) -> tuple[
    dict[tuple[str, str], list[pd.DataFrame]],
    pd.DataFrame,
    list[dict[str, str]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Iterate selected sessions and accumulate per-(stage, area) RSA matrices + per-session metrics.

    Returns
    -------
    rsa_by_cell : {(stage, area): list[pd.DataFrame]}
    summary_df  : per-session metrics (one row per area where RSA succeeded).
    failures    : list of {session, area, error} for skipped sessions/areas.
    per_session_rsa : list of {session, learning_stage, area, rsa_df} for disk export.
    per_session_pairs : list of {session, learning_stage, area, pair_values} for Panel H.
    """
    rsa_by_cell: dict[tuple[str, str], list[pd.DataFrame]] = {}
    per_session_rsa: list[dict[str, Any]] = []
    per_session_pairs: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    low_b, high_b = _category_boundary_khz()

    log_bin_edges: np.ndarray | None = None
    log_bin_centers: np.ndarray | None = None
    if use_log_stimulus_bins:
        log_bin_edges, log_bin_centers = _scan_selection_stimuli_khz(
            selected_sessions_df,
            use_histology=use_histology,
            low_boundary=low_b,
            high_boundary=high_b,
        )
        if log_bin_centers is None or log_bin_centers.size == 0:
            log_bin_edges, log_bin_centers = None, None

    n_total = len(selected_sessions_df)
    progress = st.progress(0.0, text="Computing RSA per session...")
    for i, (_, row) in enumerate(selected_sessions_df.iterrows(), start=1):
        session_dir = str(row.get("current_dir", "")).strip()
        session_type = str(row.get("Session Type", "")).strip()
        stage = rsa_cohort_stage(session_type)
        animal = str(row.get("Animal", "")).strip()
        date = str(row.get("Date", "")).strip()
        label = f"{animal} | {date} | {session_type}"
        progress.progress((i - 1) / max(n_total, 1), text=f"[{i}/{n_total}] {label}")

        if stage is None:
            failures.append({
                "session": label,
                "area": "-",
                "error": "Session type not in RSA cohorts (Novice 2b / 1b Expert 1b / 2b Expert 2b)",
            })
            continue

        if not session_dir or not os.path.isdir(session_dir):
            failures.append({"session": label, "area": "-", "error": "Missing or unreadable current_dir"})
            continue

        _loaded, area_errors = _load_session_areas(session_dir, use_histology=use_histology)
        for area in ("ACx", "OFC"):
            if area in area_errors:
                failures.append({"session": label, "area": area, "error": area_errors[area]})
                continue
            try:
                rsa_df, info = compute_session_rsa(
                    session_dir, area,
                    decode_window=decode_window,
                    aggregation=aggregation,
                    use_histology=use_histology,
                    min_trials_per_stim=min_trials_per_stim,
                    max_stim_khz=_stage_max_stim_khz(stage, high_b),
                    log_bin_edges=log_bin_edges,
                    log_bin_centers=log_bin_centers,
                )
            except (ValueError, OSError, KeyError) as exc:
                failures.append({"session": label, "area": area, "error": str(exc)})
                continue

            rsa_by_cell.setdefault((stage, area), []).append(rsa_df)
            per_session_rsa.append({
                "session": label,
                "learning_stage": stage,
                "area": area,
                "rsa_df": rsa_df,
            })
            extracted = extract_category_pair_values(
                rsa_df, low_boundary=low_b, high_boundary=high_b
            )
            metrics = extracted["summary"]
            per_session_pairs.append({
                "session": label,
                "learning_stage": stage,
                "area": area,
                "pair_values": extracted["pair_values"],
            })
            rows.append({
                "session": label,
                "animal": animal,
                "date": date,
                "session_type": session_type,
                "learning_stage": stage,
                "area": area,
                "n_units": info["n_units"],
                "n_trials": info["n_trials"],
                "n_stimuli": info["n_stimuli"],
                **metrics,
            })
    progress.progress(1.0, text="Done.")
    progress.empty()
    summary_df = pd.DataFrame(rows)
    return rsa_by_cell, summary_df, failures, per_session_rsa, per_session_pairs


def regional_rsa_panel(selected_sessions_df: pd.DataFrame) -> None:
    """Render the Regional RSA (ACx vs OFC) sub-view.

    Expects rows the user checked in the monitoring sub-view, with columns
    ``current_dir``, ``Animal``, ``Date``, ``Session Type``.
    """
    st.write("### Regional RSA (ACx vs OFC)")
    st.caption(
        "Cohorts: Novice 2b Categorization, 1b Expert 1b Categorization, and "
        "2b Expert 2b Categorization only. Per-session RSA matrices are averaged within "
        "each (cohort × area) cell. Stimulus caps: Novice and 2b Expert ≤ 21.5 deci-kHz "
        f"({DECI_KHZ_21_5_MAX_STIM_KHZ:g} kHz); 1b Expert ≤ category high boundary."
    )
    if selected_sessions_df is None or selected_sessions_df.empty:
        st.info("Select sessions in the table above to compute RSA matrices.")
        return

    selected_sessions_df = filter_rsa_cohort_sessions(selected_sessions_df)
    if selected_sessions_df.empty:
        st.warning(
            "No sessions match the RSA cohorts. Expected Session Type labels containing "
            "«Novice … 2b Categorization», «1b Expert … 1b Categorization», or "
            "«2b Expert … 2b Categorization»."
        )
        return

    with st.expander("Settings", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            window_start = st.number_input(
                "Window start (s)", value=DEFAULT_DECODE_WINDOW[0], step=0.05, format="%.2f",
                key="rsa_window_start",
            )
            window_stop = st.number_input(
                "Window stop (s)", value=DEFAULT_DECODE_WINDOW[1], step=0.05, format="%.2f",
                key="rsa_window_stop",
            )
        with col2:
            aggregation = st.selectbox(
                "Aggregation", ["Mean", "Sum"], index=0, key="rsa_aggregation",
                help="How to collapse spike counts across the analysis window.",
            )
            use_histology = st.checkbox(
                "Use histology filter", value=False, key="rsa_use_histology",
                help="Keep only units confirmed in the area by histology mapping.",
            )
        with col3:
            min_trials_per_stim = st.number_input(
                "Min trials per stimulus", min_value=1, value=DEFAULT_MIN_TRIALS_PER_STIM,
                step=1, key="rsa_min_trials_per_stim",
            )
            use_log_stimulus_bins = st.checkbox(
                "Log-bin stimuli (4 per class band)",
                value=True,
                key="rsa_use_log_bins",
                help=(
                    "Pool similar kHz tones into shared log-spaced bins anchored at "
                    "category boundaries (same scheme as the agreement-decoder psychometric plots)."
                ),
            )

    if window_stop <= window_start:
        st.error("Window stop must be greater than window start.")
        return

    if not st.button("Run RSA", key="rsa_run_button"):
        st.info("Press 'Run RSA' to compute matrices for the selected sessions.")
        return

    decode_window = (float(window_start), float(window_stop))
    rsa_by_cell, summary_df, failures, per_session_rsa, per_session_pairs = _run_rsa_for_selection(
        selected_sessions_df,
        decode_window=decode_window,
        aggregation=aggregation,
        use_histology=bool(use_histology),
        min_trials_per_stim=int(min_trials_per_stim),
        use_log_stimulus_bins=bool(use_log_stimulus_bins),
    )

    low_b, high_b = _category_boundary_khz()
    saved_dir = save_rsa_results(
        summary_df=summary_df,
        failures=failures,
        rsa_by_cell=rsa_by_cell,
        per_session_rsa=per_session_rsa,
        per_session_pairs=per_session_pairs,
        run_settings={
            "decode_window_start_s": decode_window[0],
            "decode_window_stop_s": decode_window[1],
            "aggregation": aggregation,
            "use_histology": bool(use_histology),
            "min_trials_per_stimulus": int(min_trials_per_stim),
            "use_log_stimulus_bins": bool(use_log_stimulus_bins),
            "rsa_bins_per_class": int(RSA_BINS_PER_CLASS),
            "n_sessions_selected": int(len(selected_sessions_df)),
        },
        low_boundary_khz=low_b,
        high_boundary_khz=high_b,
    )
    st.success(f"Results saved to `{saved_dir}`")
    if failures:
        st.caption(
            f"Current-run errors: `{saved_dir / 'rsa_errors.csv'}` · "
            f"append-only log: `{RSA_ERROR_LOG_PATH}`"
        )
    else:
        st.caption(f"Error log (append-only): `{RSA_ERROR_LOG_PATH}`")
    n_sessions_per_cell = {
        cell: len(matrices) for cell, matrices in rsa_by_cell.items()
    }

    # Top metrics row.
    n_ok = len(summary_df)
    n_failed = len(failures)
    n_stages = summary_df["learning_stage"].nunique() if not summary_df.empty else 0
    n_areas = summary_df["area"].nunique() if not summary_df.empty else 0
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Session x area RSAs", n_ok)
    m2.metric("Stages covered", n_stages)
    m3.metric("Areas covered", n_areas)
    m4.metric("Failures", n_failed)

    render_per_session_rsa_navigator(
        per_session_rsa,
        summary_df,
        low_boundary_khz=low_b,
        high_boundary_khz=high_b,
    )

    if not rsa_by_cell:
        st.warning("No RSA matrices could be computed for the selected sessions.")
        if failures:
            st.subheader("Error log (current run)")
            st.dataframe(pd.DataFrame(failures), use_container_width=True, hide_index=True)
        _render_rsa_methods_section(
            decode_window=decode_window,
            aggregation=aggregation,
            use_histology=bool(use_histology),
            min_trials_per_stim=int(min_trials_per_stim),
            use_log_stimulus_bins=bool(use_log_stimulus_bins),
            low_boundary_khz=low_b,
            high_boundary_khz=high_b,
        )
        return

    # --- Panel F: RSA heatmaps per (area x stage) + 2b Expert − Novice side panel ---
    st.subheader("Panel F — RSA matrices (stage averages)")
    st.caption(
        f"When log-binning is enabled, similar tones are pooled into ≤{RSA_BINS_PER_CLASS} "
        "bins per category band (log-spaced between boundaries). "
        "The right-hand panel shows the stage-mean difference 2b Expert − Novice."
    )
    stage_order = [s for s in STAGE_ORDER if any(cell[0] == s for cell in rsa_by_cell)]
    if not stage_order:
        stage_order = sorted({cell[0] for cell in rsa_by_cell})

    for area in ("ACx", "OFC"):
        present = [s for s in stage_order if (s, area) in rsa_by_cell]
        if not present:
            continue
        st.markdown(f"**{area}**")
        can_diff = "2b Expert" in present and "Novice" in present
        col_widths = [1.0] * len(present) + ([1.45] if can_diff else [])
        cols = st.columns(col_widths)

        mean_rsa_by_stage: dict[str, pd.DataFrame] = {}
        for col, stage in zip(cols[: len(present)], present):
            mean_rsa, n_sessions = aggregate_rsa_across_sessions(rsa_by_cell[(stage, area)])
            mean_rsa_by_stage[stage] = mean_rsa
            with col:
                stage_color = LEARNING_STAGE_COLORS.get(stage, (COLOR_GRAY, COLOR_GRAY))[0]
                cohort_label = RSA_COHORT_LABELS.get(stage, stage)
                title = (
                    f"{cohort_label} (n={n_sessions_per_cell.get((stage, area), 0)} sessions, "
                    f"{mean_rsa.shape[0]} stim)"
                )
                fig = plot_rsa_heatmap(
                    mean_rsa,
                    title=title,
                    low_boundary=low_b,
                    high_boundary=high_b,
                    n_sessions=n_sessions,
                )
                fig.update_layout(title=dict(text=f"<span style='color:{stage_color}'>●</span> {title}"))
                st.plotly_chart(fig, use_container_width=True)

        if can_diff:
            with cols[len(present)]:
                diff_rsa = subtract_rsa_matrices(
                    mean_rsa_by_stage["2b Expert"],
                    mean_rsa_by_stage["Novice"],
                )
                n_2b = n_sessions_per_cell.get(("2b Expert", area), 0)
                n_nov = n_sessions_per_cell.get(("Novice", area), 0)
                diff_title = f"2b Expert − Novice (n={n_2b} vs {n_nov} sessions)"
                stage_color = LEARNING_STAGE_COLORS.get("2b Expert", (COLOR_GRAY, COLOR_GRAY))[0]
                fig_diff = plot_rsa_difference_heatmap(
                    diff_rsa,
                    title=diff_title,
                    low_boundary=low_b,
                    high_boundary=high_b,
                    height=520,
                )
                fig_diff.update_layout(
                    title=dict(text=f"<span style='color:{stage_color}'>●</span> {diff_title}")
                )
                st.plotly_chart(fig_diff, use_container_width=True)

    # --- MDS embeddings (2D projection of each stage's mean RSA) -------------
    st.subheader("MDS — 2D embedding of stage-mean RSA")
    st.caption(
        "Each point is a stimulus frequency; distance approximates 1 − r between stimuli "
        "in the stage-averaged RSA matrix. When sessions used different tone ladders, "
        "MDS embeds the largest stimulus subset with complete pairwise coverage "
        "(see subtitle for embedded/total counts). Tight category clusters and large "
        "between-category separation indicate strong categorical representation."
    )
    for area in ("ACx", "OFC"):
        present = [s for s in stage_order if (s, area) in rsa_by_cell]
        if not present:
            continue
        st.markdown(f"**{area}**")
        cols = st.columns(len(present))
        for col, stage in zip(cols, present):
            mean_rsa, _ = aggregate_rsa_across_sessions(rsa_by_cell[(stage, area)])
            coords_df, stress, mds_meta = compute_mds_from_rsa(mean_rsa)
            with col:
                stage_color = LEARNING_STAGE_COLORS.get(stage, (COLOR_GRAY, COLOR_GRAY))[0]
                cohort_label = RSA_COHORT_LABELS.get(stage, stage)
                title = f"{cohort_label} — {area}"
                n_in = mds_meta["n_stimuli_input"]
                n_emb = mds_meta["n_stimuli_embedded"]
                if n_emb < n_in and n_emb >= 2:
                    title = f"{title} ({n_emb}/{n_in} stims embedded)"
                fig = plot_mds_embedding(
                    coords_df,
                    title=title,
                    low_boundary=low_b,
                    high_boundary=high_b,
                    stress=stress,
                )
                fig.update_layout(
                    title=dict(text=f"<span style='color:{stage_color}'>●</span> {fig.layout.title.text}")
                )
                st.plotly_chart(fig, use_container_width=True)

    # --- Panel G: categoricality + within/across ------------------------------
    st.subheader("Panel G — Categorical clustering across regions")
    st.plotly_chart(plot_categoricality_panel(summary_df), use_container_width=True)
    st.plotly_chart(plot_within_vs_across_panel(summary_df), use_container_width=True)

    # --- Panel H: category separation (pair-type distributions + CCI + stats) ---
    st.subheader("Panel H — Category separation")
    st.caption(
        "Pair-level RSA similarities decomposed into Go-Go, NoGo-NoGo, and Go-NoGo. "
        "Separation index (CCI) is normalized on the dissimilarity scale; "
        "categoricality index (Panel G) is the unnormalized similarity contrast."
    )
    pooled_pairs_df = build_pooled_pairs_df(per_session_pairs)
    st.plotly_chart(plot_within_across_detailed(summary_df), use_container_width=True)
    st.plotly_chart(plot_separation_index_panel(summary_df), use_container_width=True)
    if not pooled_pairs_df.empty:
        st.plotly_chart(plot_pair_type_violin(pooled_pairs_df), use_container_width=True)
        st.plotly_chart(plot_pair_type_kde(pooled_pairs_df), use_container_width=True)
    else:
        st.info("No pair-level data available for distribution plots.")

    st.subheader("Panel H — Statistical tests (session-level)")
    render_separation_stats_panel(summary_df)

    cohort_summary = summarize_category_separation(summary_df)
    if not cohort_summary.empty:
        with st.expander("Cohort separation summary (mean ± SEM)", expanded=False):
            st.dataframe(cohort_summary, use_container_width=True, hide_index=True)

    # --- Per-session table + download -----------------------------------------
    st.subheader("Per-session metrics")
    if not summary_df.empty:
        display_cols = [
            "session", "learning_stage", "area", "n_units", "n_trials", "n_stimuli",
            "within_go", "within_nogo", "within_avg", "across",
            "categoricality_index", "separation_index", "d_prime_pairs",
            "var_within_go", "var_within_nogo", "var_across",
            "n_pairs_within_go", "n_pairs_within_nogo", "n_pairs_across",
        ]
        present_cols = [c for c in display_cols if c in summary_df.columns]
        st.dataframe(summary_df[present_cols], use_container_width=True, hide_index=True)
        st.download_button(
            "Download per-session metrics (CSV)",
            data=summary_df.to_csv(index=False).encode("utf-8"),
            file_name="npxl_rsa_per_session.csv",
            mime="text/csv",
            key="rsa_download_per_session",
        )
        if not pooled_pairs_df.empty:
            st.download_button(
                "Download pooled pair-level data (CSV)",
                data=pooled_pairs_df.to_csv(index=False).encode("utf-8"),
                file_name="npxl_rsa_pooled_pairs.csv",
                mime="text/csv",
                key="rsa_download_pooled_pairs",
            )

    if failures:
        with st.expander(f"{len(failures)} skipped session/area pairs (see rsa_errors.csv)", expanded=False):
            st.dataframe(pd.DataFrame(failures), use_container_width=True, hide_index=True)

    _render_rsa_methods_section(
        decode_window=decode_window,
        aggregation=aggregation,
        use_histology=bool(use_histology),
        min_trials_per_stim=int(min_trials_per_stim),
        use_log_stimulus_bins=bool(use_log_stimulus_bins),
        low_boundary_khz=low_b,
        high_boundary_khz=high_b,
    )
