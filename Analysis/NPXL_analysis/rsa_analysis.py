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
# Default decode window (seconds, relative to tone onset). Mirrors the agreement decoder.
DEFAULT_DECODE_WINDOW: tuple[float, float] = (0.0, 0.25)
DEFAULT_MIN_TRIALS_PER_STIM = 3
# Minimum stimuli required to compute a meaningful RSA matrix.
MIN_STIMULI_FOR_RSA = 2

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


# --- RSA core -----------------------------------------------------------------

def compute_session_rsa(
    session_dir: str,
    area: str,
    *,
    decode_window: tuple[float, float] = DEFAULT_DECODE_WINDOW,
    aggregation: str = "Mean",
    use_histology: bool = True,
    min_trials_per_stim: int = DEFAULT_MIN_TRIALS_PER_STIM,
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


# --- Categoricality metric (Panel G) ------------------------------------------

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


def compute_categoricality_index(
    rsa_df: pd.DataFrame,
    *,
    low_boundary: float,
    high_boundary: float,
) -> dict[str, float]:
    """Compute within / across category similarities and a categoricality index.

    Index = mean(within_go, within_nogo) - across.

    Returns NaN for any term lacking off-diagonal pairs.
    """
    if rsa_df.empty or rsa_df.shape[0] < 2:
        return {
            "within_go": np.nan,
            "within_nogo": np.nan,
            "within_avg": np.nan,
            "across": np.nan,
            "categoricality_index": np.nan,
            "n_stimuli": int(rsa_df.shape[0]),
        }

    stim = np.asarray(rsa_df.index, dtype=float)
    cats = np.asarray(
        [_stimulus_category(k, low_boundary, high_boundary) for k in stim]
    )
    values = rsa_df.to_numpy(dtype=float)
    # Off-diagonal upper-triangle mask to avoid double-counting i,j vs j,i.
    n = values.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    pair_vals = values[iu, ju]
    pair_cat_i = cats[iu]
    pair_cat_j = cats[ju]

    within_go_mask = (pair_cat_i == "Go") & (pair_cat_j == "Go")
    within_nogo_mask = (pair_cat_i == "NoGo") & (pair_cat_j == "NoGo")
    across_mask = pair_cat_i != pair_cat_j

    def _nanmean(arr: np.ndarray) -> float:
        if arr.size == 0:
            return float("nan")
        with np.errstate(invalid="ignore"):
            return float(np.nanmean(arr))

    within_go = _nanmean(pair_vals[within_go_mask])
    within_nogo = _nanmean(pair_vals[within_nogo_mask])
    within_avg = _nanmean(
        np.concatenate([pair_vals[within_go_mask], pair_vals[within_nogo_mask]])
    )
    across = _nanmean(pair_vals[across_mask])
    cat_idx = within_avg - across if np.isfinite(within_avg) and np.isfinite(across) else np.nan

    return {
        "within_go": within_go,
        "within_nogo": within_nogo,
        "within_avg": within_avg,
        "across": across,
        "categoricality_index": cat_idx,
        "n_stimuli": int(n),
    }


# --- MDS embedding ------------------------------------------------------------

# MDS converts similarity r in [-1, 1] -> dissimilarity in [0, 1]:
#   d(i, j) = (1 - r(i, j)) / 2
# This keeps anti-correlated stimuli as far apart as identical stimuli are close,
# and bounds distances in [0, 1] so optimization is well-conditioned.
def _rsa_to_dissimilarity(rsa_df: pd.DataFrame) -> pd.DataFrame:
    """Convert correlation similarity to a [0, 1] dissimilarity matrix."""
    values = (1.0 - rsa_df.to_numpy(dtype=float)) / 2.0
    # Numerical jitter can make the diagonal slightly nonzero -> force 0.
    np.fill_diagonal(values, 0.0)
    values = np.clip(values, 0.0, None)
    # Enforce symmetry; corrcoef can drift by ~1e-16.
    values = 0.5 * (values + values.T)
    return pd.DataFrame(values, index=rsa_df.index, columns=rsa_df.columns)


def compute_mds_from_rsa(
    rsa_df: pd.DataFrame,
    *,
    n_components: int = 2,
    random_state: int = 42,
    n_init: int = 8,
    max_iter: int = 500,
) -> tuple[pd.DataFrame, float]:
    """Run metric MDS on the dissimilarity matrix derived from ``rsa_df``.

    Drops stimuli with any NaN dissimilarity (otherwise sklearn raises).
    Returns the 2D embedding indexed by kHz plus the final stress value.
    """
    if rsa_df.empty or rsa_df.shape[0] < 2:
        return pd.DataFrame(columns=[f"mds_{i + 1}" for i in range(n_components)]), float("nan")

    dissim = _rsa_to_dissimilarity(rsa_df)
    # Drop any stimulus that has a NaN pair (e.g. cell only seen in a subset of sessions).
    finite_mask = np.isfinite(dissim.to_numpy()).all(axis=1)
    kept_stim = dissim.index[finite_mask]
    if len(kept_stim) < 2:
        return pd.DataFrame(columns=[f"mds_{i + 1}" for i in range(n_components)]), float("nan")
    dissim = dissim.loc[kept_stim, kept_stim]

    mds = MDS(
        n_components=n_components,
        dissimilarity="precomputed",
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
        normalized_stress="auto",
    )
    coords = mds.fit_transform(dissim.to_numpy(dtype=float))
    coord_cols = [f"mds_{i + 1}" for i in range(n_components)]
    coords_df = pd.DataFrame(coords, index=kept_stim, columns=coord_cols)
    return coords_df, float(mds.stress_)


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
    run_settings: dict[str, Any],
    low_boundary_khz: float,
    high_boundary_khz: float,
) -> Path:
    """Write RSA outputs and append failures to the persistent error log.

    Files written under ``RSA_RESULTS_DIR``:
    - ``rsa_per_session_metrics.csv`` — categoricality table (current run).
    - ``rsa_errors.csv`` — failures from the current run only.
    - ``rsa_error_log.csv`` — append-only error history (timestamped rows).
    - ``rsa_run_metadata.json`` — decode window, aggregation, boundaries, UTC time.
    - ``cohort_mean_rsa_{stage}_{area}.csv`` — stage-averaged matrices.
    - ``cohort_mds_{stage}_{area}.csv`` — 2D MDS coordinates + stress column.
    - ``per_session/*.csv`` — one RSA matrix per successful (session, area).
    """
    RSA_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    if not summary_df.empty:
        summary_df.to_csv(RSA_RESULTS_DIR / "rsa_per_session_metrics.csv", index=False)

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

    for (stage, area), matrices in rsa_by_cell.items():
        slug = _stage_area_slug(stage, area)
        mean_rsa, n_sessions = aggregate_rsa_across_sessions(matrices)
        if not mean_rsa.empty:
            mean_rsa.to_csv(RSA_RESULTS_DIR / f"cohort_mean_rsa_{slug}.csv")
            n_sessions.to_csv(RSA_RESULTS_DIR / f"cohort_mean_rsa_{slug}_n_sessions.csv")
        coords_df, stress = compute_mds_from_rsa(mean_rsa)
        if not coords_df.empty:
            coords_out = coords_df.copy()
            coords_out["mds_stress"] = stress
            coords_out.to_csv(RSA_RESULTS_DIR / f"cohort_mds_{slug}.csv")

    return RSA_RESULTS_DIR


# --- Streamlit entry point ----------------------------------------------------

def _run_rsa_for_selection(
    selected_sessions_df: pd.DataFrame,
    *,
    decode_window: tuple[float, float],
    aggregation: str,
    use_histology: bool,
    min_trials_per_stim: int,
) -> tuple[
    dict[tuple[str, str], list[pd.DataFrame]],
    pd.DataFrame,
    list[dict[str, str]],
    list[dict[str, Any]],
]:
    """Iterate selected sessions and accumulate per-(stage, area) RSA matrices + per-session metrics.

    Returns
    -------
    rsa_by_cell : {(stage, area): list[pd.DataFrame]}
    summary_df  : per-session metrics (one row per area where RSA succeeded).
    failures    : list of {session, area, error} for skipped sessions/areas.
    per_session_rsa : list of {session, learning_stage, area, rsa_df} for disk export.
    """
    rsa_by_cell: dict[tuple[str, str], list[pd.DataFrame]] = {}
    per_session_rsa: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    low_b, high_b = _category_boundary_khz()

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
            metrics = compute_categoricality_index(
                rsa_df, low_boundary=low_b, high_boundary=high_b
            )
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
    return rsa_by_cell, summary_df, failures, per_session_rsa


def regional_rsa_panel(selected_sessions_df: pd.DataFrame) -> None:
    """Render the Regional RSA (ACx vs OFC) sub-view.

    Expects rows the user checked in the monitoring sub-view, with columns
    ``current_dir``, ``Animal``, ``Date``, ``Session Type``.
    """
    st.write("### Regional RSA (ACx vs OFC)")
    st.caption(
        "Cohorts: Novice 2b Categorization, 1b Expert 1b Categorization, and "
        "2b Expert 2b Categorization only. Per-session RSA matrices are averaged within "
        "each (cohort × area) cell."
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
                "Use histology filter", value=True, key="rsa_use_histology",
                help="Keep only units confirmed in the area by histology mapping.",
            )
        with col3:
            min_trials_per_stim = st.number_input(
                "Min trials per stimulus", min_value=1, value=DEFAULT_MIN_TRIALS_PER_STIM,
                step=1, key="rsa_min_trials_per_stim",
            )

    if window_stop <= window_start:
        st.error("Window stop must be greater than window start.")
        return

    if not st.button("Run RSA", key="rsa_run_button"):
        st.info("Press 'Run RSA' to compute matrices for the selected sessions.")
        return

    decode_window = (float(window_start), float(window_stop))
    rsa_by_cell, summary_df, failures, per_session_rsa = _run_rsa_for_selection(
        selected_sessions_df,
        decode_window=decode_window,
        aggregation=aggregation,
        use_histology=bool(use_histology),
        min_trials_per_stim=int(min_trials_per_stim),
    )

    low_b, high_b = _category_boundary_khz()
    saved_dir = save_rsa_results(
        summary_df=summary_df,
        failures=failures,
        rsa_by_cell=rsa_by_cell,
        per_session_rsa=per_session_rsa,
        run_settings={
            "decode_window_start_s": decode_window[0],
            "decode_window_stop_s": decode_window[1],
            "aggregation": aggregation,
            "use_histology": bool(use_histology),
            "min_trials_per_stimulus": int(min_trials_per_stim),
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

    if not rsa_by_cell:
        st.warning("No RSA matrices could be computed for the selected sessions.")
        if failures:
            st.subheader("Error log (current run)")
            st.dataframe(pd.DataFrame(failures), use_container_width=True, hide_index=True)
        return

    # --- Panel F: RSA heatmaps per (area x stage) ------------------------------
    st.subheader("Panel F — RSA matrices (stage averages)")
    stage_order = [s for s in STAGE_ORDER if any(cell[0] == s for cell in rsa_by_cell)]
    if not stage_order:
        stage_order = sorted({cell[0] for cell in rsa_by_cell})

    for area in ("ACx", "OFC"):
        present = [s for s in stage_order if (s, area) in rsa_by_cell]
        if not present:
            continue
        st.markdown(f"**{area}**")
        cols = st.columns(len(present))
        for col, stage in zip(cols, present):
            mean_rsa, n_sessions = aggregate_rsa_across_sessions(rsa_by_cell[(stage, area)])
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
                # Highlight stage with a colored title prefix dot via annotation.
                fig.update_layout(title=dict(text=f"<span style='color:{stage_color}'>●</span> {title}"))
                st.plotly_chart(fig, use_container_width=True)

    # --- MDS embeddings (2D projection of each stage's mean RSA) -------------
    st.subheader("MDS — 2D embedding of stage-mean RSA")
    st.caption(
        "Each point is a stimulus frequency; distance approximates 1 - r between stimuli "
        "in the stage-averaged RSA matrix. Tight category clusters and large between-category "
        "separation indicate strong categorical representation."
    )
    for area in ("ACx", "OFC"):
        present = [s for s in stage_order if (s, area) in rsa_by_cell]
        if not present:
            continue
        st.markdown(f"**{area}**")
        cols = st.columns(len(present))
        for col, stage in zip(cols, present):
            mean_rsa, _ = aggregate_rsa_across_sessions(rsa_by_cell[(stage, area)])
            coords_df, stress = compute_mds_from_rsa(mean_rsa)
            with col:
                stage_color = LEARNING_STAGE_COLORS.get(stage, (COLOR_GRAY, COLOR_GRAY))[0]
                cohort_label = RSA_COHORT_LABELS.get(stage, stage)
                title = f"{cohort_label} — {area}"
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

    # --- Per-session table + download -----------------------------------------
    st.subheader("Per-session metrics")
    if not summary_df.empty:
        display_cols = [
            "session", "learning_stage", "area", "n_units", "n_trials", "n_stimuli",
            "within_go", "within_nogo", "within_avg", "across", "categoricality_index",
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

    if failures:
        with st.expander(f"{len(failures)} skipped session/area pairs (see rsa_errors.csv)", expanded=False):
            st.dataframe(pd.DataFrame(failures), use_container_width=True, hide_index=True)
