"""Streamlit page: per-session ACx/OFC decoders and agreement with behavior vs ground truth.

Loads tone-aligned event windows from NPXL preprocessing, trains separate area classifiers
on mouse Go/No-Go choice (never pooled across sessions), and visualizes trial-level agreement.
"""
from __future__ import annotations

import os
import warnings
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from Analysis.GNG_bpod_analysis.GNG_bpod_general import get_plotly_config
from Analysis.GNG_bpod_analysis.colors import (
    AREA_COLORS,
    COLOR_GO,
    COLOR_GRAY,
    COLOR_NOGO,
    LEARNING_STAGE_COLORS,
)
from Analysis.NPXL_analysis.single_unit_offline_analysis.data_loading import (
    load_full_event_windows_data,
    load_histology_matched_unit_indices,
    load_unit_indices_by_type,
)
from load_data.session_dprime import attach_session_dprime, merge_behavioral_file_from_monitoring


# Probe naming matches the rest of the NPXL monitoring stack (SpikeGLX imec0/imec1).
AREA_TO_IMEC: dict[str, str] = {"ACx": "imec0", "OFC": "imec1"}
# Behavior Y: lick = Go (Hit, FA); withhold = No-Go (Miss, CR).
GO_CHOICE_OUTCOMES = {"hit", "false alarm"}
NOGO_CHOICE_OUTCOMES = {"miss", "cr", "correct rejection", "correct reject"}
# Ground truth GT: stimulus class (Go tone vs No-Go tone), independent of whether the mouse was correct.
GO_STIM_OUTCOMES = {"hit", "miss"}
NOGO_STIM_OUTCOMES = {"false alarm", "cr", "correct rejection", "correct reject"}
# Outcomes that render as filled markers in decision scatters (vs open for Miss/FA).
_FILLED_OUTCOME_LABELS = frozenset({"hit", "cr", "correct rejection", "correct reject"})
# Normalize common abbreviations/format variants to one canonical label.
_OUTCOME_ALIASES = {
    "h": "hit",
    "hit": "hit",
    "m": "miss",
    "miss": "miss",
    "fa": "false alarm",
    "false alarm": "false alarm",
    "falsealarm": "false alarm",
    "false_alarm": "false alarm",
    "cr": "cr",
    "correct rejection": "correct rejection",
    "correct reject": "correct rejection",
    "correct_rejection": "correct rejection",
}
UNIT_THRESHOLD = 15  # Minimum good+MUA units required to run decoding for an area.
GOOD_MUA_CODES = (1, 2)  # Bombcell UnitType codes for single units + MUA.
TRAIN_TEST_FOLDS = 5  # Fixed 80/20 split via StratifiedKFold (1 fold test, 4 folds train).


def _parse_unit_summary(cell: Any) -> tuple[int, int]:
    """Parse monitoring-table cells like ``good: 18, MUA: 80, non-somatic: 50``."""
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return 0, 0
    if isinstance(cell, (int, float)) and not isinstance(cell, bool):
        return int(cell), 0
    if not isinstance(cell, str):
        return 0, 0

    good = 0
    mua = 0
    for part in cell.lower().split(","):
        if ":" not in part:
            continue
        key, value = [item.strip() for item in part.split(":", 1)]
        try:
            count = int(float(value))
        except ValueError:
            continue
        if key == "good":
            good = count
        elif key == "mua":
            mua = count
    return good, mua


def _session_label(row: pd.Series) -> str:
    animal = str(row.get("Animal", "")).strip() or "Unknown animal"
    date = str(row.get("Date", "")).strip() or "Unknown date"
    session_type = str(row.get("Session Type", "")).strip() or "Unknown session"
    return (
        f"{animal} | {date} | {session_type} "
        f"(ACx={int(row['acx_total_units'])}, OFC={int(row['ofc_total_units'])})"
    )


@st.cache_data(show_spinner=False)
def _histology_summary_for_session(session_dir: str) -> dict[str, float]:
    summary: dict[str, float] = {}
    for area in ("ACx", "OFC"):
        area_l = area.lower()
        try:
            probe_dir = _find_probe_dir(session_dir, AREA_TO_IMEC[area])
            good_mua_count = len(load_unit_indices_by_type(probe_dir, GOOD_MUA_CODES))
            matched_indices, _matched_df = load_histology_matched_unit_indices(
                session_dir,
                probe_dir,
                area,
                GOOD_MUA_CODES,
            )
            matched_count = len(matched_indices)
            summary[f"{area_l}_histology_matched"] = float(matched_count)
            summary[f"{area_l}_histology_good_mua"] = float(good_mua_count)
            summary[f"{area_l}_histology_match_pct"] = (
                100.0 * matched_count / good_mua_count if good_mua_count > 0 else np.nan
            )
        except (FileNotFoundError, ValueError, OSError):
            summary[f"{area_l}_histology_matched"] = np.nan
            summary[f"{area_l}_histology_good_mua"] = np.nan
            summary[f"{area_l}_histology_match_pct"] = np.nan
    return summary


@st.cache_data(show_spinner=False)
def load_valid_sessions(monitoring_csv_path: str) -> pd.DataFrame:
    """Sessions eligible for decoding: at least one area > UNIT_THRESHOLD and a recording path."""
    sessions_df = pd.read_csv(monitoring_csv_path, low_memory=False)
    if sessions_df.empty:
        return pd.DataFrame()

    acx_col = "Acx good units"
    ofc_col = "OFC good units"
    if acx_col not in sessions_df.columns or ofc_col not in sessions_df.columns:
        return pd.DataFrame()

    counts = sessions_df[[acx_col, ofc_col]].copy()
    acx_counts = counts[acx_col].map(_parse_unit_summary)
    ofc_counts = counts[ofc_col].map(_parse_unit_summary)
    sessions_df["acx_good_units"] = [good for good, _mua in acx_counts]
    sessions_df["acx_mua_units"] = [mua for _good, mua in acx_counts]
    sessions_df["ofc_good_units"] = [good for good, _mua in ofc_counts]
    sessions_df["ofc_mua_units"] = [mua for _good, mua in ofc_counts]
    sessions_df["acx_total_units"] = sessions_df["acx_good_units"] + sessions_df["acx_mua_units"]
    sessions_df["ofc_total_units"] = sessions_df["ofc_good_units"] + sessions_df["ofc_mua_units"]

    valid_mask = (
        (sessions_df["acx_total_units"] >= UNIT_THRESHOLD)
        | (sessions_df["ofc_total_units"] >= UNIT_THRESHOLD)
    ) & sessions_df.get("current_dir", pd.Series("", index=sessions_df.index)).fillna("").astype(str).str.strip().ne("")
    valid_sessions = sessions_df.loc[valid_mask].copy()
    valid_sessions = attach_session_dprime(valid_sessions, behavioral_file_col="behavioral file")

    histology_records: list[dict[str, float]] = []
    for session_dir in valid_sessions.get("current_dir", pd.Series("", index=valid_sessions.index)).fillna("").astype(str):
        histology_records.append(_histology_summary_for_session(session_dir.strip()))
    histology_df = pd.DataFrame(histology_records, index=valid_sessions.index)
    valid_sessions = pd.concat([valid_sessions, histology_df], axis=1)

    # Preserve original CSV row id so checkbox selection can map back after reset_index.
    valid_sessions["session_row_index"] = valid_sessions.index.astype(int)
    valid_sessions["session_label"] = valid_sessions.apply(_session_label, axis=1)
    return valid_sessions.reset_index(drop=True)


def _find_probe_dir(session_dir: str, imec: str) -> str:
    """Locate imec0/imec1 folder that contains exported ``event_windows_matrix.npy``."""
    if not os.path.isdir(session_dir):
        raise FileNotFoundError(f"Session folder not found: {session_dir}")

    direct_matches = [
        os.path.join(session_dir, name)
        for name in os.listdir(session_dir)
        if os.path.isdir(os.path.join(session_dir, name)) and name.lower().endswith(imec.lower())
    ]
    for candidate in sorted(direct_matches):
        if os.path.exists(os.path.join(candidate, "analysis_output", "event_windows_matrix.npy")):
            return candidate

    # CatGT / pipeline_output layouts vary; walk only if direct child folders miss event windows.
    for root, dirs, _files in os.walk(session_dir):
        for dirname in dirs:
            if not dirname.lower().endswith(imec.lower()):
                continue
            candidate = os.path.join(root, dirname)
            if os.path.exists(os.path.join(candidate, "analysis_output", "event_windows_matrix.npy")):
                return candidate

    raise FileNotFoundError(f"No {imec} probe folder with event windows found under {session_dir}")


@st.cache_data(show_spinner=False)
def _load_area_event_data(session_dir: str, area: str, *, use_histology: bool = True) -> dict[str, Any]:
    """Load one probe's tone-aligned matrix.

    ``use_histology=True``  — keep only good+MUA units confirmed in ``area`` by histology mapping.
    ``use_histology=False`` — keep all good+MUA units regardless of histology; faster and gives more
                              units but loses the area-label guarantee.
    """
    probe_dir = _find_probe_dir(session_dir, AREA_TO_IMEC[area])
    event_matrix, time_axis, valid_indices, trials_df, metadata, _lick_matrix = load_full_event_windows_data(probe_dir)
    if event_matrix.ndim != 3:
        raise ValueError(f"{area} event matrix must be 3D [units x time x trials], got {event_matrix.shape}")

    good_mua_indices = load_unit_indices_by_type(probe_dir, GOOD_MUA_CODES)

    if use_histology:
        unit_indices, unit_table = load_histology_matched_unit_indices(
            session_dir, probe_dir, area, GOOD_MUA_CODES
        )
        unit_indices = [idx for idx in unit_indices if 0 <= idx < event_matrix.shape[0]]
        filter_label = "histology-matched"
    else:
        # All good+MUA units — area label is assumed, not confirmed by histology.
        unit_indices = [idx for idx in good_mua_indices if 0 <= idx < event_matrix.shape[0]]
        unit_table = pd.DataFrame({"unit_idx": unit_indices, "note": "no histology filter"})
        filter_label = "all good+MUA (no histology)"

    if len(unit_indices) < UNIT_THRESHOLD:
        raise ValueError(
            f"{area} has only {len(unit_indices)} {filter_label} units "
            f"(of {len(good_mua_indices)} good+MUA in labels)."
        )

    filtered_matrix = event_matrix[np.asarray(unit_indices, dtype=int), :, :]
    return {
        "area": area,
        "probe_dir": probe_dir,
        "matrix": filtered_matrix,
        "full_matrix_shape": event_matrix.shape,
        "time_axis": np.asarray(time_axis, dtype=float),
        "valid_indices": np.asarray(valid_indices, dtype=int),
        "trials_df": trials_df.reset_index(drop=True),
        "metadata": metadata,
        "unit_count": len(unit_indices),
        "good_mua_count": len(good_mua_indices),
        "unit_table": unit_table,
        "use_histology": use_histology,
    }


@st.cache_data(show_spinner=True)
def load_session_event_data(session_dir: str, *, use_histology: bool = True) -> dict[str, Any]:
    """Load available areas; run dual-area when possible, otherwise single-area fallback."""
    area_data: dict[str, dict[str, Any]] = {}
    area_errors: dict[str, str] = {}
    for area in ("ACx", "OFC"):
        try:
            area_data[area] = _load_area_event_data(session_dir, area, use_histology=use_histology)
        except (FileNotFoundError, ValueError, OSError) as exc:
            area_errors[area] = str(exc)

    if not area_data:
        details = "; ".join(f"{k}: {v}" for k, v in area_errors.items())
        raise ValueError(f"No valid area data found for decoding. {details}")

    available_areas = [area for area in ("ACx", "OFC") if area in area_data]
    payload: dict[str, Any] = {
        "areas": available_areas,
        "missing_area_errors": area_errors,
    }

    if len(available_areas) == 1:
        single = available_areas[0]
        payload[single] = area_data[single]
        payload["time_axis"] = area_data[single]["time_axis"]
        payload["valid_indices"] = area_data[single]["valid_indices"]
        return payload

    acx_data = area_data["ACx"]
    ofc_data = area_data["OFC"]
    common_valid = np.intersect1d(acx_data["valid_indices"], ofc_data["valid_indices"])
    if common_valid.size == 0:
        # Keep the richer area so decoding can still run.
        fallback_area = "ACx" if acx_data["unit_count"] >= ofc_data["unit_count"] else "OFC"
        payload["missing_area_errors"][("OFC" if fallback_area == "ACx" else "ACx")] = (
            "Dropped because ACx/OFC had no shared valid trials for paired analysis."
        )
        payload["areas"] = [fallback_area]
        payload[fallback_area] = area_data[fallback_area]
        payload["time_axis"] = area_data[fallback_area]["time_axis"]
        payload["valid_indices"] = area_data[fallback_area]["valid_indices"]
        return payload

    def _positions_for(valid_indices: np.ndarray) -> np.ndarray:
        # Map original trial index -> column position in the saved event matrix.
        position_by_valid = {int(valid): pos for pos, valid in enumerate(valid_indices)}
        return np.asarray([position_by_valid[int(valid)] for valid in common_valid], dtype=int)

    acx_pos = _positions_for(acx_data["valid_indices"])
    ofc_pos = _positions_for(ofc_data["valid_indices"])
    acx_data["matrix"] = acx_data["matrix"][:, :, acx_pos]
    ofc_data["matrix"] = ofc_data["matrix"][:, :, ofc_pos]
    acx_data["trials_df"] = acx_data["trials_df"].iloc[acx_pos].reset_index(drop=True)
    ofc_data["trials_df"] = ofc_data["trials_df"].iloc[ofc_pos].reset_index(drop=True)

    if acx_data["time_axis"].shape != ofc_data["time_axis"].shape or not np.allclose(acx_data["time_axis"], ofc_data["time_axis"]):
        # Keep the area with more usable units when paired decoding is impossible.
        fallback_area = "ACx" if acx_data["unit_count"] >= ofc_data["unit_count"] else "OFC"
        payload["missing_area_errors"][("OFC" if fallback_area == "ACx" else "ACx")] = (
            "Dropped because ACx/OFC time axes differ."
        )
        payload["areas"] = [fallback_area]
        payload[fallback_area] = area_data[fallback_area]
        payload["time_axis"] = area_data[fallback_area]["time_axis"]
        payload["valid_indices"] = area_data[fallback_area]["valid_indices"]
        return payload

    payload["ACx"] = acx_data
    payload["OFC"] = ofc_data
    payload["time_axis"] = acx_data["time_axis"]
    payload["valid_indices"] = common_valid
    return payload


def _normalize_outcome(value: Any) -> str:
    normalized = str(value).strip().lower().replace("-", " ").replace("_", " ")
    normalized = " ".join(normalized.split())
    return _OUTCOME_ALIASES.get(normalized, normalized)


def _build_trial_labels(trials_df: pd.DataFrame) -> pd.DataFrame:
    """Derive binary behavior (Y) and stimulus ground truth (GT) from Bpod outcome strings."""
    if "outcome" not in trials_df.columns:
        raise ValueError("Trial table is missing the required 'outcome' column.")

    labels_df = trials_df.copy()
    # Keep matrix column index after dropping catch / unknown outcomes below.
    labels_df["source_trial_position"] = np.arange(len(labels_df), dtype=int)
    normalized = labels_df["outcome"].map(_normalize_outcome)
    behavior = np.full(len(labels_df), np.nan)
    ground_truth = np.full(len(labels_df), np.nan)

    behavior[normalized.isin(GO_CHOICE_OUTCOMES).to_numpy()] = 1
    behavior[normalized.isin(NOGO_CHOICE_OUTCOMES).to_numpy()] = 0
    ground_truth[normalized.isin(GO_STIM_OUTCOMES).to_numpy()] = 1
    ground_truth[normalized.isin(NOGO_STIM_OUTCOMES).to_numpy()] = 0

    labels_df["behavior_go"] = behavior
    labels_df["ground_truth_go"] = ground_truth
    return labels_df.dropna(subset=["behavior_go", "ground_truth_go"]).reset_index(drop=True)


def _feature_matrix(event_matrix: np.ndarray, time_axis: np.ndarray, window: tuple[float, float], aggregation: str) -> np.ndarray:
    """Collapse [units x time x trials] to [trials x units] using the user-selected post-onset window."""
    start, stop = window
    time_mask = (time_axis >= start) & (time_axis <= stop)
    if not time_mask.any():
        raise ValueError("Decode window does not include any time bins.")

    window_data = event_matrix[:, time_mask, :]
    if aggregation == "Sum":
        return window_data.sum(axis=1).T  # total spikes per unit in window
    return window_data.mean(axis=1).T  # mean firing rate per unit in window


def _make_classifier(model_type: str, random_state: int):
    # StandardScaler per fold avoids unit-count / firing-rate scale dominating SVM margin.
    if model_type == "RBF SVM":
        model = SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=random_state,
        )
    elif model_type == "Linear SVM":
        model = SVC(kernel="linear", class_weight="balanced", probability=True, random_state=random_state)
    else:
        model = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
            solver="liblinear",
        )
    return make_pipeline(StandardScaler(), model)


@st.cache_data(show_spinner=False)
def train_oof_decoder(
    area: str,
    session_key: str,
    x_values: np.ndarray,
    y_values: np.ndarray,
    model_type: str,
    requested_folds: int,
    random_state: int,
    min_trials_per_class: int,
) -> dict[str, Any]:
    """Stratified K-fold on one session only; returns out-of-fold probs/labels (no train-set leakage)."""
    del area, session_key  # Reserved for cache key stability if extended later.

    y = y_values.astype(int)
    class_counts = np.bincount(y, minlength=2)
    minority_count = int(class_counts.min())
    if minority_count < min_trials_per_class:
        raise ValueError(
            f"Need at least {min_trials_per_class} trials per choice class; found {class_counts.tolist()}."
        )

    # StratifiedKFold cannot use more folds than the smallest class count.
    n_splits = min(int(requested_folds), minority_count)
    if n_splits < 2:
        raise ValueError(f"Need at least 2 CV folds; found class counts {class_counts.tolist()}.")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    probabilities = np.full(len(y), np.nan, dtype=float)
    predictions = np.full(len(y), -1, dtype=int)
    warnings_seen: list[str] = []

    for train_idx, test_idx in cv.split(x_values, y):
        classifier = _make_classifier(model_type, random_state)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            classifier.fit(x_values[train_idx], y[train_idx])
        warnings_seen.extend(str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning))

        fold_probabilities = classifier.predict_proba(x_values[test_idx])[:, 1]  # P(Go choice)
        probabilities[test_idx] = fold_probabilities
        predictions[test_idx] = (fold_probabilities >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)) if len(np.unique(y)) == 2 else np.nan,
        "n_splits": n_splits,
    }
    return {"probability": probabilities, "prediction": predictions, "metrics": metrics, "warnings": warnings_seen}


def _label_binary(values: np.ndarray | pd.Series) -> list[str]:
    return ["Go" if int(value) == 1 else "No-Go" for value in values]


def _plot_crosstab_heatmap(row_values: list[str], col_values: list[str], title: str, row_label: str, col_label: str) -> go.Figure:
    """2x2 congruence heatmap with fixed Go/No-Go axis order for comparable panels."""
    categories = ["No-Go", "Go"]
    table = pd.crosstab(pd.Series(row_values, name=row_label), pd.Series(col_values, name=col_label))
    table = table.reindex(index=categories, columns=categories, fill_value=0)
    z_values = table.to_numpy()

    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=table.columns.tolist(),
            y=table.index.tolist(),
            text=z_values,
            texttemplate="%{text}",
            colorscale="Blues",
            colorbar=dict(title="Trials"),
        )
    )
    fig.update_layout(title=title, xaxis_title=col_label, yaxis_title=row_label, height=360)
    return fig


def _plot_state_vs_ground_truth(states: pd.Series, gt_labels: list[str]) -> go.Figure:
    table = pd.crosstab(states.rename("Model state"), pd.Series(gt_labels, name="Ground Truth"))
    table = table.reindex(columns=["No-Go", "Go"], fill_value=0)
    fig = go.Figure(
        data=go.Heatmap(
            z=table.to_numpy(),
            x=table.columns.tolist(),
            y=table.index.tolist(),
            text=table.to_numpy(),
            texttemplate="%{text}",
            colorscale="Purples",
            colorbar=dict(title="Trials"),
        )
    )
    fig.update_layout(title="Both Models vs Ground Truth", xaxis_title="Ground Truth", yaxis_title="Agreement State", height=420)
    return fig


def _agreement_state(acx_pred: np.ndarray, ofc_pred: np.ndarray) -> pd.Series:
    """Four-way trial label for the combined-models vs ground-truth heatmap."""
    labels = []
    for acx, ofc in zip(acx_pred, ofc_pred):
        if acx == ofc == 1:
            labels.append("Both Go")
        elif acx == ofc == 0:
            labels.append("Both No-Go")
        elif acx == 1:
            labels.append("ACx Go / OFC No-Go")
        else:
            labels.append("ACx No-Go / OFC Go")
    return pd.Series(labels)


def _error_status(acx: int, ofc: int, behavior: int, ground_truth: int) -> str:
    """Classify model agreement on trials where behavior != stimulus (Miss or FA)."""
    if acx != ofc:
        return "Models disagree"
    if acx == behavior:
        return "Both match behavior"
    if acx == ground_truth:
        return "Both match ground truth"
    return "Both miss"


def _plot_error_breakdown(results_df: pd.DataFrame) -> go.Figure | None:
    # Only behavioral mistakes: where the mouse's choice disagrees with the tone category.
    error_df = results_df[results_df["behavior_go"] != results_df["ground_truth_go"]].copy()
    if error_df.empty:
        return None

    error_df["error_type"] = np.where(error_df["behavior_go"] == 1, "False Alarm", "Miss")
    error_df["model_status"] = [
        _error_status(int(row.acx_pred), int(row.ofc_pred), int(row.behavior_go), int(row.ground_truth_go))
        for row in error_df.itertuples(index=False)
    ]
    summary = error_df.groupby(["error_type", "model_status"], observed=True).size().reset_index(name="trials")

    fig = go.Figure()
    for status, status_df in summary.groupby("model_status", observed=True):
        fig.add_trace(go.Bar(x=status_df["error_type"], y=status_df["trials"], name=status))
    fig.update_layout(
        title="Model Agreement on Behavioral Error Trials",
        xaxis_title="Behavioral Error",
        yaxis_title="Trials",
        barmode="stack",
        height=420,
    )
    return fig


# Psychometric x-axis display range (kHz), aligned with NPXL FRA density plots.
_PSYCHOMETRIC_X_MIN_KHZ = 6.0
_PSYCHOMETRIC_X_MAX_KHZ = 22.0
_PSYCHOMETRIC_MIN_TRIALS_PER_BIN = 5  # Match Streamlit plot: skip sparse frequency bins.
# Per-session sidecar export should be permissive: sparse rows can become well-supported only
# after pooling across sessions. The group-level plot applies the real display threshold later.
_PSYCHOMETRIC_EXPORT_MIN_TRIALS_PER_BIN = 1
# Group-level (pooled across sessions) psychometric curves apply this stricter floor instead.
# 1b Expert uses a fine-grained psychometric ladder with many distinct kHz values, and the
# model-disagree subset is ~5-10x sparser than all-trials. With a low threshold the per-frequency
# Bernoulli proportion swings wildly. Used for: (a) per-session model-disagree build (default),
# and (b) cross-session pooling of both agree and disagree group psychometric curves.
_PSYCHOMETRIC_GROUP_MIN_TRIALS_PER_BIN = 5


def _category_boundary_khz() -> tuple[float, float]:
    """Tone-category boundaries (kHz) from global session state."""
    return (
        float(st.session_state.get("low_boundary", 0.983)),
        float(st.session_state.get("high_boundary", 1.525)),
    )


def _build_psychometric_by_stimulus(
    results_df: pd.DataFrame,
    *,
    error_trials_only: bool = False,
    min_trials_per_bin: int = _PSYCHOMETRIC_MIN_TRIALS_PER_BIN,
) -> pd.DataFrame:
    """P(Go) for mouse, GT, and area GT-decoders at each stimulus frequency (kHz)."""
    if "stimulus" not in results_df.columns:
        return pd.DataFrame()

    work_df = results_df.copy()
    if error_trials_only:
        work_df = work_df[work_df["behavior_go"] != work_df["ground_truth_go"]].copy()
        if work_df.empty:
            return pd.DataFrame()

    work_df["stimulus_khz"] = pd.to_numeric(work_df["stimulus"], errors="coerce")
    work_df = work_df[np.isfinite(work_df["stimulus_khz"]) & (work_df["stimulus_khz"] > 0)]
    if work_df.empty:
        return pd.DataFrame()

    agg_map: dict[str, tuple[str, str]] = {
        "trials": ("stimulus_khz", "size"),
        "n_mouse_go": ("behavior_go", "sum"),
        "n_gt_go": ("ground_truth_go", "sum"),
    }
    if "acx_pred" in work_df.columns:
        agg_map["n_acx_go"] = ("acx_pred", "sum")
    if "ofc_pred" in work_df.columns:
        agg_map["n_ofc_go"] = ("ofc_pred", "sum")

    grouped = work_df.groupby("stimulus_khz", observed=True).agg(**agg_map).reset_index()
    grouped = grouped.rename(columns={"stimulus_khz": "stimulus"})
    grouped = grouped[grouped["trials"] >= min_trials_per_bin].sort_values("stimulus")
    if grouped.empty:
        return pd.DataFrame()

    grouped["p_mouse_go"] = grouped["n_mouse_go"] / grouped["trials"]
    grouped["p_gt_go"] = grouped["n_gt_go"] / grouped["trials"]
    if "n_acx_go" in grouped.columns:
        grouped["p_acx_go"] = grouped["n_acx_go"] / grouped["trials"]
    if "n_ofc_go" in grouped.columns:
        grouped["p_ofc_go"] = grouped["n_ofc_go"] / grouped["trials"]
    grouped["error_trials_only"] = error_trials_only
    return grouped.reset_index(drop=True)


def _aggregate_psychometric_across_sessions(session_psych_df: pd.DataFrame) -> pd.DataFrame:
    """Pool trial counts across sessions, then recompute psychometric P(Go) curves."""
    if session_psych_df.empty or "stimulus" not in session_psych_df.columns:
        return pd.DataFrame()

    count_cols = ["trials", "n_mouse_go", "n_gt_go"]
    if "n_acx_go" in session_psych_df.columns:
        count_cols.append("n_acx_go")
    if "n_ofc_go" in session_psych_df.columns:
        count_cols.append("n_ofc_go")
    if not set(count_cols).issubset(session_psych_df.columns):
        return pd.DataFrame()

    group_cols = ["stimulus", "error_trials_only"]
    if "error_trials_only" not in session_psych_df.columns:
        group_cols = ["stimulus"]

    grouped = (
        session_psych_df.groupby(group_cols, observed=True)[count_cols]
        .sum()
        .reset_index()
    )
    grouped = grouped[grouped["trials"] >= _PSYCHOMETRIC_MIN_TRIALS_PER_BIN].sort_values(group_cols)
    if grouped.empty:
        return pd.DataFrame()

    grouped["p_mouse_go"] = grouped["n_mouse_go"] / grouped["trials"]
    grouped["p_gt_go"] = grouped["n_gt_go"] / grouped["trials"]
    if "n_acx_go" in grouped.columns:
        grouped["p_acx_go"] = grouped["n_acx_go"] / grouped["trials"]
    if "n_ofc_go" in grouped.columns:
        grouped["p_ofc_go"] = grouped["n_ofc_go"] / grouped["trials"]
    return grouped.reset_index(drop=True)


def _plot_psychometric_summary(
    results_df: pd.DataFrame,
    *,
    error_trials_only: bool = False,
) -> go.Figure | None:
    grouped = _build_psychometric_by_stimulus(results_df, error_trials_only=error_trials_only)
    if grouped.shape[0] < 2:
        return None

    low_boundary, high_boundary = _category_boundary_khz()
    fig = go.Figure()
    for column, name in [
        ("p_mouse_go", "Mouse Go Choice"),
        ("p_acx_go", "ACx Decoder Go"),
        ("p_ofc_go", "OFC Decoder Go"),
        ("p_gt_go", "Ground Truth Go"),
    ]:
        if column not in grouped.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=grouped["stimulus"],
                y=grouped[column],
                mode="lines+markers",
                name=name,
            )
        )

    for boundary, label in (
        (low_boundary, "Low boundary"),
        (high_boundary, "High boundary"),
    ):
  
        fig.add_vline(
            x=boundary,
            line=dict(color=COLOR_GRAY, dash="dash", width=2),
        )

    title = (
        "Psychometric-Style Agreement on Error Trials by Stimulus"
        if error_trials_only
        else "Psychometric-Style Agreement by Stimulus"
    )
    fig.update_layout(
        title=title,
        xaxis_title="Stimulus (kHz, log)",
        yaxis_title="P(Go)",
        xaxis=dict(
            type="log",
        ),
        yaxis=dict(range=[0, 1]),
        height=420,
    )
    return fig


def _go_nogo_text(value: int | float) -> str:
    return "Go" if int(value) == 1 else "No-Go"


def _build_joint_conditional_probabilities(results_df: pd.DataFrame) -> pd.DataFrame:
    """Conditional outcome probabilities for each ACx/OFC prediction combination."""
    required = {"acx_pred", "ofc_pred", "behavior_go", "ground_truth_go"}
    if not required.issubset(results_df.columns):
        return pd.DataFrame()

    grouped = (
        results_df.groupby(["acx_pred", "ofc_pred"], observed=True)
        .agg(
            n_condition=("behavior_go", "size"),
            n_mouse_go=("behavior_go", "sum"),
            n_gt_go=("ground_truth_go", "sum"),
        )
        .reset_index()
    )
    grouped["n_mouse_nogo"] = grouped["n_condition"] - grouped["n_mouse_go"]
    grouped["n_gt_nogo"] = grouped["n_condition"] - grouped["n_gt_go"]
    grouped["p_mouse_go"] = grouped["n_mouse_go"] / grouped["n_condition"]
    grouped["p_mouse_nogo"] = grouped["n_mouse_nogo"] / grouped["n_condition"]
    grouped["p_gt_go"] = grouped["n_gt_go"] / grouped["n_condition"]
    grouped["p_gt_nogo"] = grouped["n_gt_nogo"] / grouped["n_condition"]
    grouped["condition"] = grouped.apply(
        lambda row: (
            f"ACx={_go_nogo_text(int(row['acx_pred']))}, "
            f"OFC={_go_nogo_text(int(row['ofc_pred']))}"
        ),
        axis=1,
    )
    grouped = grouped.sort_values(["acx_pred", "ofc_pred"], ascending=[False, False]).reset_index(drop=True)
    return grouped[
        [
            "condition",
            "acx_pred",
            "ofc_pred",
            "n_condition",
            "n_mouse_go",
            "n_mouse_nogo",
            "p_mouse_go",
            "p_mouse_nogo",
            "n_gt_go",
            "n_gt_nogo",
            "p_gt_go",
            "p_gt_nogo",
        ]
    ]


_KAPPA_SOURCE_COLUMNS: dict[str, str] = {
    "mouse": "behavior_go",
    "gt": "ground_truth_go",
    "acx": "acx_pred",
    "ofc": "ofc_pred",
}
_KAPPA_PAIR_SPECS: tuple[tuple[str, str, str], ...] = (
    ("mouse", "acx", "mouse_acx"),
    ("mouse", "ofc", "mouse_ofc"),
    ("acx", "ofc", "acx_ofc"),
    ("gt", "acx", "gt_acx"),
    ("gt", "ofc", "gt_ofc"),
    ("gt", "mouse", "gt_mouse"),
)
_KAPPA_METRIC_COLUMNS: tuple[str, ...] = tuple(f"cohen_{suffix}" for _, _, suffix in _KAPPA_PAIR_SPECS) + (
    "fleiss_mouse_acx_ofc",
    "fleiss_mouse_acx_ofc_gt",
    "var_agree_acx_ofc",  # Bernoulli variance of per-trial ACx==OFC agreement.
)


def _binary_values(series: pd.Series) -> np.ndarray:
    """Return finite 0/1 labels from a Series."""
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)].astype(int)


def _cohen_kappa_binary(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray) -> float:
    """Cohen's kappa for two binary 0/1 label sources."""
    a_values = pd.to_numeric(pd.Series(a), errors="coerce").to_numpy(dtype=float)
    b_values = pd.to_numeric(pd.Series(b), errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(a_values) & np.isfinite(b_values)
    if int(valid.sum()) == 0:
        return np.nan

    a_binary = a_values[valid].astype(int)
    b_binary = b_values[valid].astype(int)
    observed = float(np.mean(a_binary == b_binary))
    p_a_go = float(np.mean(a_binary == 1))
    p_b_go = float(np.mean(b_binary == 1))
    expected = p_a_go * p_b_go + (1.0 - p_a_go) * (1.0 - p_b_go)
    denom = 1.0 - expected
    if np.isclose(denom, 0.0):
        return np.nan
    return float((observed - expected) / denom)


def _fleiss_kappa_binary(label_matrix: pd.DataFrame | np.ndarray) -> float:
    """Fleiss' kappa for binary labels with one row per trial and one column per rater/source."""
    matrix = np.asarray(label_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] < 2:
        return np.nan

    valid_rows = np.all(np.isfinite(matrix), axis=1)
    matrix = matrix[valid_rows].astype(int)
    n_items, n_raters = matrix.shape
    if n_items == 0 or n_raters < 2:
        return np.nan

    n_go = matrix.sum(axis=1).astype(float)
    n_nogo = float(n_raters) - n_go
    p_item = (n_go * (n_go - 1.0) + n_nogo * (n_nogo - 1.0)) / (n_raters * (n_raters - 1.0))
    p_bar = float(np.mean(p_item))
    p_go = float(n_go.sum() / (n_items * n_raters))
    p_expected = p_go**2 + (1.0 - p_go) ** 2
    denom = 1.0 - p_expected
    if np.isclose(denom, 0.0):
        return np.nan
    return float((p_bar - p_expected) / denom)


def _raw_agreement(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray) -> float:
    """Raw pairwise agreement rate for context beside chance-corrected kappa."""
    a_values = pd.to_numeric(pd.Series(a), errors="coerce").to_numpy(dtype=float)
    b_values = pd.to_numeric(pd.Series(b), errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(a_values) & np.isfinite(b_values)
    if int(valid.sum()) == 0:
        return np.nan
    return float(np.mean(a_values[valid].astype(int) == b_values[valid].astype(int)))


def _kappa_summary_row(results_df: pd.DataFrame) -> dict[str, Any]:
    """Compute pairwise Cohen and multi-source Fleiss kappas for one trial table."""
    row: dict[str, Any] = {"n_trials": int(len(results_df))}
    for left, right, suffix in _KAPPA_PAIR_SPECS:
        left_col = _KAPPA_SOURCE_COLUMNS[left]
        right_col = _KAPPA_SOURCE_COLUMNS[right]
        row[f"cohen_{suffix}"] = _cohen_kappa_binary(results_df[left_col], results_df[right_col])
        row[f"p_agree_{suffix}"] = _raw_agreement(results_df[left_col], results_df[right_col])

    # Bernoulli variance of the per-trial ACx==OFC agreement indicator (var = p(1-p)).
    p_acx_ofc = row.get("p_agree_acx_ofc", np.nan)
    row["var_agree_acx_ofc"] = (
        float(p_acx_ofc * (1.0 - p_acx_ofc)) if np.isfinite(p_acx_ofc) else np.nan
    )

    row["fleiss_mouse_acx_ofc"] = _fleiss_kappa_binary(
        results_df[["behavior_go", "acx_pred", "ofc_pred"]]
    )
    row["fleiss_mouse_acx_ofc_gt"] = _fleiss_kappa_binary(
        results_df[["behavior_go", "acx_pred", "ofc_pred", "ground_truth_go"]]
    )
    return row


def _build_kappa_agreement_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """Session-level chance-corrected agreement among Mouse, GT, ACx, and OFC labels."""
    required = set(_KAPPA_SOURCE_COLUMNS.values())
    if results_df.empty or not required.issubset(results_df.columns):
        return pd.DataFrame()
    return pd.DataFrame([_kappa_summary_row(results_df)])


def _build_kappa_by_stimulus(results_df: pd.DataFrame) -> pd.DataFrame:
    """Stimulus-resolved chance-corrected agreement for later group-level binning."""
    required = {"stimulus", *_KAPPA_SOURCE_COLUMNS.values()}
    if results_df.empty or not required.issubset(results_df.columns):
        return pd.DataFrame()

    work = results_df.copy()
    work["stimulus"] = pd.to_numeric(work["stimulus"], errors="coerce")
    work = work[np.isfinite(work["stimulus"]) & (work["stimulus"] > 0)]
    if work.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for stimulus, subset in work.groupby("stimulus", observed=True):
        rows.append({"stimulus": float(stimulus), **_kappa_summary_row(subset)})
    return pd.DataFrame(rows).sort_values("stimulus").reset_index(drop=True)


def _build_single_area_conditional_probabilities(results_df: pd.DataFrame, area: str) -> pd.DataFrame:
    """Conditional probabilities for one available area and the mouse/ground-truth outcomes."""
    area_col = f"{area.lower()}_pred"
    required = {area_col, "behavior_go", "ground_truth_go"}
    if not required.issubset(results_df.columns):
        return pd.DataFrame()

    grouped = (
        results_df.groupby([area_col], observed=True)
        .agg(
            n_condition=("behavior_go", "size"),
            n_mouse_go=("behavior_go", "sum"),
            n_gt_go=("ground_truth_go", "sum"),
        )
        .reset_index()
    )
    grouped["n_mouse_nogo"] = grouped["n_condition"] - grouped["n_mouse_go"]
    grouped["n_gt_nogo"] = grouped["n_condition"] - grouped["n_gt_go"]
    grouped["p_mouse_go"] = grouped["n_mouse_go"] / grouped["n_condition"]
    grouped["p_mouse_nogo"] = grouped["n_mouse_nogo"] / grouped["n_condition"]
    grouped["p_gt_go"] = grouped["n_gt_go"] / grouped["n_condition"]
    grouped["p_gt_nogo"] = grouped["n_gt_nogo"] / grouped["n_condition"]
    grouped["condition"] = grouped[area_col].map(lambda v: f"{area}={_go_nogo_text(int(v))}")
    grouped = grouped.sort_values(area_col, ascending=False).reset_index(drop=True)
    return grouped[
        [
            "condition",
            area_col,
            "n_condition",
            "n_mouse_go",
            "n_mouse_nogo",
            "p_mouse_go",
            "p_mouse_nogo",
            "n_gt_go",
            "n_gt_nogo",
            "p_gt_go",
            "p_gt_nogo",
        ]
    ]


def _build_marginal_probabilities(results_df: pd.DataFrame, available_areas: list[str]) -> pd.DataFrame:
    """Marginal conditional probabilities like P(Mouse=Go | OFC=Go)."""
    rows: list[dict[str, Any]] = []

    def _append_rows(
        frame: pd.DataFrame,
        cond_col: str,
        cond_label: str,
        outcome_col: str,
        outcome_label: str,
    ) -> None:
        if cond_col not in frame.columns or outcome_col not in frame.columns:
            return
        grouped = (
            frame.groupby(cond_col, observed=True)
            .agg(
                n=("behavior_go", "size"),
                n_positive=(outcome_col, "sum"),
            )
            .reset_index()
        )
        for item in grouped.itertuples(index=False):
            cond_value = int(getattr(item, cond_col))
            n_count = int(item.n)
            n_positive = int(item.n_positive)
            p_positive = n_positive / n_count if n_count > 0 else np.nan
            rows.append(
                {
                    "expression": (
                        f"P({outcome_label}=Go | {cond_label}={_go_nogo_text(cond_value)})"
                    ),
                    "n_condition": n_count,
                    "n_go": n_positive,
                    "n_nogo": n_count - n_positive,
                    "p_go": p_positive,
                    "p_nogo": 1.0 - p_positive if np.isfinite(p_positive) else np.nan,
                }
            )

    for area in available_areas:
        area_col = f"{area.lower()}_pred"
        _append_rows(results_df, area_col, area, "behavior_go", "Mouse")
        _append_rows(results_df, area_col, area, "ground_truth_go", "GroundTruth")

    return pd.DataFrame(rows)


def _aggregate_joint_probabilities_across_sessions(session_joint_df: pd.DataFrame) -> pd.DataFrame:
    """Pool trial counts across sessions, then recompute joint conditional probabilities."""
    count_cols = [
        "n_condition",
        "n_mouse_go",
        "n_mouse_nogo",
        "n_gt_go",
        "n_gt_nogo",
    ]
    required = {"acx_pred", "ofc_pred", *count_cols}
    if session_joint_df.empty or not required.issubset(session_joint_df.columns):
        return pd.DataFrame()

    grouped = (
        session_joint_df.groupby(["acx_pred", "ofc_pred"], observed=True)[list(count_cols)]
        .sum()
        .reset_index()
    )
    grouped["p_mouse_go"] = grouped["n_mouse_go"] / grouped["n_condition"]
    grouped["p_mouse_nogo"] = grouped["n_mouse_nogo"] / grouped["n_condition"]
    grouped["p_gt_go"] = grouped["n_gt_go"] / grouped["n_condition"]
    grouped["p_gt_nogo"] = grouped["n_gt_nogo"] / grouped["n_condition"]
    grouped["condition"] = grouped.apply(
        lambda row: (
            f"ACx={_go_nogo_text(int(row['acx_pred']))}, "
            f"OFC={_go_nogo_text(int(row['ofc_pred']))}"
        ),
        axis=1,
    )
    return grouped.sort_values(["acx_pred", "ofc_pred"], ascending=[False, False]).reset_index(drop=True)


def _aggregate_marginal_probabilities_across_sessions(session_marginal_df: pd.DataFrame) -> pd.DataFrame:
    """Pool trial counts across sessions, then recompute marginal conditional probabilities."""
    count_cols = ["n_condition", "n_go", "n_nogo"]
    if session_marginal_df.empty or not {"expression", *count_cols}.issubset(session_marginal_df.columns):
        return pd.DataFrame()

    grouped = (
        session_marginal_df.groupby("expression", observed=True)[list(count_cols)]
        .sum()
        .reset_index()
    )
    grouped["p_go"] = grouped["n_go"] / grouped["n_condition"]
    grouped["p_nogo"] = grouped["n_nogo"] / grouped["n_condition"]
    return grouped


def _behavior_match_odds(p: float) -> float:
    """Odds of behavior matching models; NaN when p is 0, 1, or undefined."""
    if not np.isfinite(p) or p <= 0.0 or p >= 1.0:
        return np.nan
    return float(p / (1.0 - p))


def _build_model_agreement_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """P(lick matches shared GT-decoder prediction) by ACx/OFC agreement state."""
    required = {"acx_pred", "ofc_pred", "behavior_go"}
    if not required.issubset(results_df.columns):
        return pd.DataFrame()

    work = results_df.copy()
    work["behavior_matches"] = work["behavior_go"].to_numpy(dtype=int) == work["acx_pred"].to_numpy(dtype=int)
    work["models_agree"] = work["acx_pred"].to_numpy(dtype=int) == work["ofc_pred"].to_numpy(dtype=int)
    work["agreement_state"] = _agreement_state(
        work["acx_pred"].to_numpy(dtype=int),
        work["ofc_pred"].to_numpy(dtype=int),
    )

    state_frames: list[tuple[str, pd.DataFrame]] = [
        ("All trials", work),
        ("Models agree (pooled)", work[work["models_agree"]]),
        ("Both Go", work[work["agreement_state"] == "Both Go"]),
        ("Both No-Go", work[work["agreement_state"] == "Both No-Go"]),
        ("Models disagree", work[~work["models_agree"]]),
    ]
    rows: list[dict[str, Any]] = []
    for label, subset in state_frames:
        if subset.empty:
            continue
        n_trials = int(len(subset))
        n_match = int(subset["behavior_matches"].sum())
        p_match = n_match / n_trials
        rows.append(
            {
                "agreement_state": label,
                "n_trials": n_trials,
                "n_behavior_match": n_match,
                "p_behavior_match": p_match,
                "odds_behavior_match": _behavior_match_odds(p_match),
            }
        )
    return pd.DataFrame(rows)


def _model_agreement_summary_from_joint(joint_df: pd.DataFrame) -> pd.DataFrame:
    """Fallback summary when only joint-probability sidecar CSV is available."""
    required = {"acx_pred", "ofc_pred", "n_condition", "n_mouse_go", "n_mouse_nogo"}
    if joint_df.empty or not required.issubset(joint_df.columns):
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    total_n = int(joint_df["n_condition"].sum())
    total_match = 0
    agree_n = 0
    agree_match = 0

    for item in joint_df.itertuples(index=False):
        acx = int(item.acx_pred)
        ofc = int(item.ofc_pred)
        n_cond = int(item.n_condition)
        n_mouse_go = int(item.n_mouse_go)
        n_mouse_nogo = int(item.n_mouse_nogo)
        if acx == ofc:
            n_match = n_mouse_go if acx == 1 else n_mouse_nogo
            state = "Both Go" if acx == 1 else "Both No-Go"
            agree_n += n_cond
            agree_match += n_match
        else:
            n_match = n_mouse_go if acx == 1 else n_mouse_nogo
            if acx == 1 and ofc == 0:
                state = "ACx Go / OFC No-Go"
            else:
                state = "ACx No-Go / OFC Go"
        p_match = n_match / n_cond if n_cond > 0 else np.nan
        rows.append(
            {
                "agreement_state": state,
                "n_trials": n_cond,
                "n_behavior_match": n_match,
                "p_behavior_match": p_match,
                "odds_behavior_match": _behavior_match_odds(p_match),
            }
        )
        total_match += n_match

    if total_n > 0:
        rows.insert(
            0,
            {
                "agreement_state": "All trials",
                "n_trials": total_n,
                "n_behavior_match": total_match,
                "p_behavior_match": total_match / total_n,
                "odds_behavior_match": _behavior_match_odds(total_match / total_n),
            },
        )
    if agree_n > 0:
        p_agree = agree_match / agree_n
        rows.insert(
            1,
            {
                "agreement_state": "Models agree (pooled)",
                "n_trials": agree_n,
                "n_behavior_match": agree_match,
                "p_behavior_match": p_agree,
                "odds_behavior_match": _behavior_match_odds(p_agree),
            },
        )
    return pd.DataFrame(rows)


def _build_psychometric_behavior_match_by_stimulus(
    results_df: pd.DataFrame,
    *,
    models_agree_only: bool = False,
    split_by_agreement_state: bool = False,
    min_trials_per_bin: int = _PSYCHOMETRIC_MIN_TRIALS_PER_BIN,
) -> pd.DataFrame:
    """P(behavior matches shared decoder prediction) at each stimulus frequency."""
    required = {"acx_pred", "ofc_pred", "behavior_go"}
    if "stimulus" not in results_df.columns or not required.issubset(results_df.columns):
        return pd.DataFrame()

    work = results_df.copy()
    work["stimulus_khz"] = pd.to_numeric(work["stimulus"], errors="coerce")
    work = work[np.isfinite(work["stimulus_khz"]) & (work["stimulus_khz"] > 0)]
    if work.empty:
        return pd.DataFrame()

    work["behavior_matches"] = work["behavior_go"].to_numpy(dtype=int) == work["acx_pred"].to_numpy(dtype=int)
    work["models_agree"] = work["acx_pred"].to_numpy(dtype=int) == work["ofc_pred"].to_numpy(dtype=int)
    if models_agree_only or split_by_agreement_state:
        work = work[work["models_agree"]].copy()
    if work.empty:
        return pd.DataFrame()

    if split_by_agreement_state:
        work["agreement_state"] = _agreement_state(
            work["acx_pred"].to_numpy(dtype=int),
            work["ofc_pred"].to_numpy(dtype=int),
        )
        work = work[work["agreement_state"].isin(["Both Go", "Both No-Go"])].copy()
        group_cols = ["stimulus_khz", "agreement_state"]
    else:
        group_cols = ["stimulus_khz"]

    grouped = (
        work.groupby(group_cols, observed=True)
        .agg(
            trials=("behavior_matches", "size"),
            n_behavior_match=("behavior_matches", "sum"),
        )
        .reset_index()
    )
    grouped = grouped.rename(columns={"stimulus_khz": "stimulus"})
    sort_cols = ["stimulus"] + (["agreement_state"] if split_by_agreement_state else [])
    grouped = grouped[grouped["trials"] >= min_trials_per_bin].sort_values(sort_cols)
    if grouped.empty:
        return pd.DataFrame()

    grouped["p_behavior_match"] = grouped["n_behavior_match"] / grouped["trials"]
    grouped["odds_behavior_match"] = grouped["p_behavior_match"].map(_behavior_match_odds)
    grouped["models_agree_only"] = models_agree_only or split_by_agreement_state
    if not split_by_agreement_state:
        grouped["agreement_state"] = "Models agree (pooled)"
    return grouped.reset_index(drop=True)


def _build_psychometric_disagreement_area_match_by_stimulus(
    results_df: pd.DataFrame,
    *,
    min_trials_per_bin: int = _PSYCHOMETRIC_GROUP_MIN_TRIALS_PER_BIN,
) -> pd.DataFrame:
    """P(behavior matches ACx or OFC decoder) by stimulus on model-disagree trials only."""
    required = {"acx_pred", "ofc_pred", "behavior_go"}
    if "stimulus" not in results_df.columns or not required.issubset(results_df.columns):
        return pd.DataFrame()

    work = results_df.copy()
    work["stimulus_khz"] = pd.to_numeric(work["stimulus"], errors="coerce")
    work = work[np.isfinite(work["stimulus_khz"]) & (work["stimulus_khz"] > 0)]
    if work.empty:
        return pd.DataFrame()

    work["models_agree"] = work["acx_pred"].to_numpy(dtype=int) == work["ofc_pred"].to_numpy(dtype=int)
    work = work[~work["models_agree"]].copy()
    if work.empty:
        return pd.DataFrame()

    work["match_acx"] = work["behavior_go"].to_numpy(dtype=int) == work["acx_pred"].to_numpy(dtype=int)
    work["match_ofc"] = work["behavior_go"].to_numpy(dtype=int) == work["ofc_pred"].to_numpy(dtype=int)

    frames: list[pd.DataFrame] = []
    for area, match_col in (("ACx", "match_acx"), ("OFC", "match_ofc")):
        grouped = (
            work.groupby("stimulus_khz", observed=True)
            .agg(
                trials=(match_col, "size"),
                n_behavior_match=(match_col, "sum"),
            )
            .reset_index()
            .rename(columns={"stimulus_khz": "stimulus"})
        )
        grouped = grouped[grouped["trials"] >= min_trials_per_bin].sort_values("stimulus")
        if grouped.empty:
            continue
        grouped["area"] = area
        grouped["p_behavior_match"] = grouped["n_behavior_match"] / grouped["trials"]
        grouped["odds_behavior_match"] = grouped["p_behavior_match"].map(_behavior_match_odds)
        grouped["models_disagree_only"] = True
        frames.append(grouped)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _aggregate_model_agreement_across_sessions(
    session_agreement_df: pd.DataFrame,
    *,
    extra_group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Pool trial counts across sessions, then recompute behavior-match probabilities."""
    count_cols = ["n_trials", "n_behavior_match"]
    if session_agreement_df.empty or not {"agreement_state", *count_cols}.issubset(session_agreement_df.columns):
        return pd.DataFrame()

    group_cols = ["agreement_state"]
    if extra_group_cols:
        group_cols = extra_group_cols + group_cols

    grouped = (
        session_agreement_df.groupby(group_cols, observed=True)[count_cols]
        .sum()
        .reset_index()
    )
    grouped["p_behavior_match"] = grouped["n_behavior_match"] / grouped["n_trials"]
    grouped["odds_behavior_match"] = grouped["p_behavior_match"].map(_behavior_match_odds)
    return grouped


def _aggregate_psychometric_behavior_match_across_sessions(
    session_psych_df: pd.DataFrame,
    *,
    extra_group_cols: list[str] | None = None,
    min_trials_per_bin: int = _PSYCHOMETRIC_MIN_TRIALS_PER_BIN,
) -> pd.DataFrame:
    """Pool psychometric behavior-match counts across sessions.

    ``min_trials_per_bin`` is applied to the **pooled** count (after summing across sessions),
    so callers operating on a sparse subset (e.g. model-disagreement trials only) should pass a
    stricter threshold to suppress 0/3-style Bernoulli noise.
    """
    count_cols = ["trials", "n_behavior_match"]
    if session_psych_df.empty or "stimulus" not in session_psych_df.columns:
        return pd.DataFrame()
    if not set(count_cols).issubset(session_psych_df.columns):
        return pd.DataFrame()

    group_cols = ["stimulus"]
    if extra_group_cols:
        group_cols = extra_group_cols + group_cols
    for optional_col in ("models_agree_only", "models_disagree_only", "agreement_state", "area"):
        if optional_col in session_psych_df.columns:
            group_cols.append(optional_col)

    grouped = (
        session_psych_df.groupby(group_cols, observed=True)[count_cols]
        .sum()
        .reset_index()
    )
    grouped = grouped[grouped["trials"] >= min_trials_per_bin].sort_values(group_cols)
    if grouped.empty:
        return pd.DataFrame()

    grouped["p_behavior_match"] = grouped["n_behavior_match"] / grouped["trials"]
    grouped["odds_behavior_match"] = grouped["p_behavior_match"].map(_behavior_match_odds)
    return grouped.reset_index(drop=True)


def _sem(x: pd.Series) -> float:
    """Standard error of finite values."""
    vals = pd.to_numeric(x, errors="coerce").dropna()
    return float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else np.nan


def _aggregate_kappa_agreement_across_sessions(
    kappa_df: pd.DataFrame,
    *,
    extra_group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Session-level mean ± SEM for kappa metrics."""
    if kappa_df.empty or "n_trials" not in kappa_df.columns:
        return pd.DataFrame()

    metric_cols = [col for col in _KAPPA_METRIC_COLUMNS if col in kappa_df.columns]
    if not metric_cols:
        return pd.DataFrame()

    group_cols = list(extra_group_cols or [])
    if group_cols:
        grouped = kappa_df.groupby(group_cols, observed=True)
    else:
        kappa_df = kappa_df.copy()
        kappa_df["_all"] = "All selected sessions"
        group_cols = ["_all"]
        grouped = kappa_df.groupby(group_cols, observed=True)

    agg: dict[str, tuple[str, Any]] = {
        "n_sessions": ("n_trials", "count"),
        "n_trials": ("n_trials", "sum"),
    }
    for col in metric_cols:
        agg[f"mean_{col}"] = (col, "mean")
        agg[f"sem_{col}"] = (col, _sem)

    summary = grouped.agg(**agg).reset_index()
    if "_all" in summary.columns:
        summary = summary.drop(columns="_all")
    return summary


def _aggregate_kappa_by_stimulus(
    kappa_stim_df: pd.DataFrame,
    *,
    extra_group_cols: list[str] | None = None,
    min_trials_per_bin: int = _PSYCHOMETRIC_GROUP_MIN_TRIALS_PER_BIN,
) -> pd.DataFrame:
    """Weighted mean kappa by stimulus bin using session trial counts as weights."""
    if kappa_stim_df.empty or not {"stimulus", "n_trials"}.issubset(kappa_stim_df.columns):
        return pd.DataFrame()

    metric_cols = [col for col in _KAPPA_METRIC_COLUMNS if col in kappa_stim_df.columns]
    if not metric_cols:
        return pd.DataFrame()

    group_cols = list(extra_group_cols or []) + ["stimulus"]
    rows: list[dict[str, Any]] = []
    for keys, group in kappa_stim_df.groupby(group_cols, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        total_trials = int(pd.to_numeric(group["n_trials"], errors="coerce").fillna(0).sum())
        if total_trials < min_trials_per_bin:
            continue
        row["n_sessions"] = int(group[["animal", "date", "session_type"]].drop_duplicates().shape[0]) if {
            "animal",
            "date",
            "session_type",
        }.issubset(group.columns) else int(len(group))
        row["n_trials"] = total_trials
        weights = pd.to_numeric(group["n_trials"], errors="coerce").to_numpy(dtype=float)
        for col in metric_cols:
            vals = pd.to_numeric(group[col], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(vals) & np.isfinite(weights) & (weights > 0)
            row[col] = float(np.average(vals[valid], weights=weights[valid])) if valid.any() else np.nan
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def _filter_group_sidecar_to_view(sidecar_df: pd.DataFrame, view_df: pd.DataFrame) -> pd.DataFrame:
    """Keep sidecar rows for sessions present in the filtered group view."""
    if sidecar_df.empty or view_df.empty:
        return sidecar_df
    if not {"animal", "date", "session_type"}.issubset(sidecar_df.columns):
        return sidecar_df
    session_keys = view_df[["animal", "date", "session_type"]].drop_duplicates()
    return sidecar_df.merge(session_keys, on=["animal", "date", "session_type"], how="inner")


_AGREEMENT_STATE_ORDER = [
    "All trials",
    "Models agree (pooled)",
    "Both Go",
    "Both No-Go",
    "Models disagree",
    "ACx Go / OFC No-Go",
    "ACx No-Go / OFC Go",
]


def _plot_group_agreement_summary(pooled_df: pd.DataFrame) -> go.Figure | None:
    """Bar chart of P(behavior matches models) by agreement state."""
    if pooled_df.empty or "p_behavior_match" not in pooled_df.columns:
        return None

    plot_df = pooled_df.copy()
    order = [s for s in _AGREEMENT_STATE_ORDER if s in plot_df["agreement_state"].unique()]
    extra = [s for s in plot_df["agreement_state"].unique() if s not in order]
    plot_df["agreement_state"] = pd.Categorical(
        plot_df["agreement_state"], categories=order + extra, ordered=True
    )
    plot_df = plot_df.sort_values("agreement_state")

    fig = go.Figure(
        data=[
            go.Bar(
                x=plot_df["agreement_state"].astype(str),
                y=plot_df["p_behavior_match"],
                text=[f"{p:.2f}" for p in plot_df["p_behavior_match"]],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title="P(Mouse Matches Shared Decoder Prediction)",
        xaxis_title="Agreement State",
        yaxis_title="P(Behavior Match)",
        yaxis=dict(range=[0, 1.05]),
        height=420,
    )
    return fig


_PSYCHOMETRIC_EXPERT_STAGES = ("1b Expert", "2b Expert")
_MODELS_AGREE_PSYCH_STATE = "Models agree (pooled)"


def _filter_1b_psychometric_above_high_boundary(df: pd.DataFrame) -> pd.DataFrame:
    """Drop 1b Expert stimulus bins above the high category boundary (2b Expert unchanged)."""
    if df.empty or "stimulus" not in df.columns or "learning_stage" not in df.columns:
        return df
    _, high_boundary = _category_boundary_khz()
    stim_khz = pd.to_numeric(df["stimulus"], errors="coerce")
    drop = (df["learning_stage"] == "1b Expert") & (stim_khz > high_boundary)
    return df.loc[~drop].copy()


def _log_bin_stimulus_per_stage(
    df: pd.DataFrame,
    *,
    bins_per_class: int = 4,
) -> pd.DataFrame:
    """Replace raw stimulus values with log-spaced bin centres, anchored to category boundaries.

    Motivation: the per-kHz ``groupby("stimulus")`` in
    ``_aggregate_psychometric_behavior_match_across_sessions`` requires *exact* float equality to
    pool across sessions. With a fine 1b psychometric ladder (and any floating-point drift in tone
    frequencies) sessions effectively don't pool, so each plotted point is supported by ~one
    session's worth of trials and the curve looks like binomial confetti. Binning to a fixed
    geometric grid (per stage) forces every session into the same bins and gives each pooled bin
    5-10x more trials.

    Bin layout (with default ``bins_per_class=4``):
      * **1b Expert** — data already restricted to ``stimulus <= high_boundary`` by
        ``_filter_1b_psychometric_above_high_boundary``. Two bands separated by ``low_boundary``
        (Go-low and the ambiguous/No-Go segment up to ``high_boundary``) -> ``2 * 4 = 8`` bins.
      * **2b Expert** — full observed range. Three bands separated by ``low_boundary`` and
        ``high_boundary`` (Go-low, ambiguous middle, No-Go-high) -> ``3 * 4 = 12`` bins.

    A band is included only if some of the stage's stimuli fall inside it, so a stage that doesn't
    actually sample every band ends up with fewer bins (never more).
    """
    if df.empty or "stimulus" not in df.columns or "learning_stage" not in df.columns:
        return df

    low_boundary, high_boundary = _category_boundary_khz()
    work = df.copy()
    work["stimulus"] = pd.to_numeric(work["stimulus"], errors="coerce")
    work = work[np.isfinite(work["stimulus"]) & (work["stimulus"] > 0)]
    if work.empty:
        return pd.DataFrame(columns=df.columns)

    binned_frames: list[pd.DataFrame] = []
    for stage, stage_df in work.groupby("learning_stage", observed=True):
        stim = stage_df["stimulus"].to_numpy()
        stage_min, stage_max = float(stim.min()), float(stim.max())
        if not np.isfinite(stage_min) or stage_min <= 0 or stage_max <= stage_min:
            binned_frames.append(stage_df)
            continue

        # Anchor class edges to category boundaries, clipped to observed range (deduped + sorted).
        class_edges = sorted(
            {stage_min, stage_max}
            | {b for b in (low_boundary, high_boundary) if stage_min < b < stage_max}
        )
        if len(class_edges) < 2:
            binned_frames.append(stage_df)
            continue

        # Geometric (log) bin edges inside each class band, deduped where bands meet.
        all_edges_set: set[float] = set()
        for left, right in zip(class_edges[:-1], class_edges[1:]):
            for edge in np.geomspace(left, right, bins_per_class + 1):
                all_edges_set.add(float(edge))
        all_edges = np.array(sorted(all_edges_set))
        if all_edges.size < 2:
            binned_frames.append(stage_df)
            continue

        # Geometric centre per bin — renders cleanly on a log x-axis.
        centers = np.sqrt(all_edges[:-1] * all_edges[1:])
        # ``side="right"`` so a value equal to a class-boundary edge lands in the upper band.
        idx = np.clip(np.searchsorted(all_edges, stim, side="right") - 1, 0, len(centers) - 1)
        stage_df = stage_df.copy()
        stage_df["stimulus"] = centers[idx]
        binned_frames.append(stage_df)

    if not binned_frames:
        return pd.DataFrame(columns=df.columns)
    return pd.concat(binned_frames, ignore_index=True)


def _filter_models_agree_psychometric(df: pd.DataFrame) -> pd.DataFrame:
    """Keep pooled models-agree psychometric rows (exclude Both Go / Both No-Go splits)."""
    if df.empty or "agreement_state" not in df.columns:
        return df
    return df[df["agreement_state"] == _MODELS_AGREE_PSYCH_STATE].copy()


def _plot_group_agreement_psychometric(pooled_df: pd.DataFrame) -> go.Figure | None:
    """Psychometric curve: P(behavior match) vs stimulus on models-agree trials."""
    plot_df = _filter_models_agree_psychometric(pooled_df)
    if plot_df.empty or plot_df["stimulus"].nunique() < 2:
        return None

    low_boundary, high_boundary = _category_boundary_khz()
    fig = go.Figure()
    n_traces = 0
    for stage in _PSYCHOMETRIC_EXPERT_STAGES:
        if "learning_stage" not in plot_df.columns:
            break
        stage_df = plot_df[plot_df["learning_stage"] == stage]
        if stage == "1b Expert":
            stage_df = _filter_1b_psychometric_above_high_boundary(stage_df)
        stage_df = stage_df.sort_values("stimulus")
        if stage_df["stimulus"].nunique() < 2:
            continue
        line_color = LEARNING_STAGE_COLORS.get(stage, ("#888888", ""))[0]
        fig.add_trace(
            go.Scatter(
                x=stage_df["stimulus"],
                y=stage_df["p_behavior_match"],
                mode="lines+markers",
                name=stage,
                line=dict(color=line_color),
                marker=dict(color=line_color),
            )
        )
        n_traces += 1

    if n_traces == 0:
        return None

    for boundary in (low_boundary, high_boundary):
        fig.add_vline(x=boundary, line=dict(color=COLOR_GRAY, dash="dash", width=2))

    fig.update_layout(
        title="Behavior Match vs Stimulus (Models-Agree Trials)",
        xaxis_title="Stimulus (kHz, log)",
        yaxis_title="P(Behavior Matches Models)",
        xaxis=dict(type="log"),
        yaxis=dict(range=[0, 1]),
        legend_title="Learning stage",
        height=420,
    )
    return fig


def _plot_group_disagreement_area_match_psychometric(pooled_df: pd.DataFrame) -> go.Figure | None:
    """P(behavior matches ACx vs OFC decoder) by stimulus on model-disagree trials."""
    if pooled_df.empty or "area" not in pooled_df.columns:
        return None

    low_boundary, high_boundary = _category_boundary_khz()
    expert_stages = [
        stage
        for stage in _PSYCHOMETRIC_EXPERT_STAGES
        if "learning_stage" in pooled_df.columns
        and stage in pooled_df["learning_stage"].unique()
    ]
    if not expert_stages:
        return None

    valid_stages: list[str] = []
    for stage in expert_stages:
        stage_df = pooled_df[pooled_df["learning_stage"] == stage]
        if stage == "1b Expert":
            stage_df = _filter_1b_psychometric_above_high_boundary(stage_df)
        if any(
            stage_df.loc[stage_df["area"] == area, "stimulus"].nunique() >= 2
            for area in ("ACx", "OFC")
        ):
            valid_stages.append(stage)
    if not valid_stages:
        return None

    fig = make_subplots(
        rows=1,
        cols=len(valid_stages),
        subplot_titles=valid_stages,
        shared_yaxes=True,
        horizontal_spacing=0.08,
    )
    legend_areas_shown: set[str] = set()
    for col_idx, stage in enumerate(valid_stages, start=1):
        stage_df = pooled_df[pooled_df["learning_stage"] == stage]
        if stage == "1b Expert":
            stage_df = _filter_1b_psychometric_above_high_boundary(stage_df)
        for area in ("ACx", "OFC"):
            area_df = stage_df[stage_df["area"] == area].sort_values("stimulus")
            if area_df["stimulus"].nunique() < 2:
                continue
            color = AREA_COLORS.get(area, "#888888")
            show_legend = area not in legend_areas_shown
            fig.add_trace(
                go.Scatter(
                    x=area_df["stimulus"],
                    y=area_df["p_behavior_match"],
                    mode="lines+markers",
                    name=area,
                    legendgroup=area,
                    showlegend=show_legend,
                    line=dict(color=color),
                    marker=dict(color=color),
                ),
                row=1,
                col=col_idx,
            )
            legend_areas_shown.add(area)
        for boundary in (low_boundary, high_boundary):
            fig.add_vline(
                x=boundary,
                line=dict(color=COLOR_GRAY, dash="dash", width=2),
                row=1,
                col=col_idx,
            )
        fig.update_xaxes(type="log", row=1, col=col_idx)
        fig.update_yaxes(range=[0, 1], row=1, col=col_idx)

    fig.update_layout(
        title="Mouse vs ACx/OFC on Model-Disagree Trials",
        xaxis_title="Stimulus (kHz, log)",
        yaxis_title="P(Behavior Matches Area Decoder)",
        legend_title="Area",
        height=420,
    )
    return fig


_KAPPA_PLOT_METRICS: dict[str, str] = {
    "cohen_mouse_acx": "Mouse vs ACx",
    "cohen_mouse_ofc": "Mouse vs OFC",
    "cohen_acx_ofc": "ACx vs OFC",
}
_KAPPA_PLOT_COLORS: dict[str, str] = {
    "cohen_mouse_acx": AREA_COLORS.get("ACx", "#4C78A8"),
    "cohen_mouse_ofc": AREA_COLORS.get("OFC", "#2A9D8F"),
    "cohen_acx_ofc": COLOR_GRAY,
}


def _plot_group_kappa_by_stimulus(pooled_df: pd.DataFrame) -> go.Figure | None:
    """Stimulus-resolved weighted session-level kappa by learning stage."""
    if pooled_df.empty or "learning_stage" not in pooled_df.columns:
        return None

    available_metrics = [col for col in _KAPPA_PLOT_METRICS if col in pooled_df.columns]
    if not available_metrics:
        return None

    low_boundary, high_boundary = _category_boundary_khz()
    valid_stages = [
        stage
        for stage in _PSYCHOMETRIC_EXPERT_STAGES
        if stage in pooled_df["learning_stage"].unique()
        and pooled_df.loc[pooled_df["learning_stage"] == stage, "stimulus"].nunique() >= 2
    ]
    if not valid_stages:
        return None

    fig = make_subplots(
        rows=1,
        cols=len(valid_stages),
        subplot_titles=valid_stages,
        shared_yaxes=True,
        horizontal_spacing=0.08,
    )
    shown: set[str] = set()
    for col_idx, stage in enumerate(valid_stages, start=1):
        stage_df = pooled_df[pooled_df["learning_stage"] == stage].sort_values("stimulus")
        for metric in available_metrics:
            metric_df = stage_df[np.isfinite(pd.to_numeric(stage_df[metric], errors="coerce"))]
            if metric_df["stimulus"].nunique() < 2:
                continue
            show_legend = metric not in shown
            fig.add_trace(
                go.Scatter(
                    x=metric_df["stimulus"],
                    y=metric_df[metric],
                    mode="lines+markers",
                    name=_KAPPA_PLOT_METRICS[metric],
                    legendgroup=metric,
                    showlegend=show_legend,
                    line=dict(color=_KAPPA_PLOT_COLORS.get(metric, "#888888")),
                    marker=dict(color=_KAPPA_PLOT_COLORS.get(metric, "#888888")),
                    customdata=np.stack(
                        [
                            metric_df["n_trials"].to_numpy(),
                            metric_df["n_sessions"].to_numpy(),
                        ],
                        axis=-1,
                    ),
                    hovertemplate=(
                        "Stimulus=%{x:.3g} kHz<br>"
                        "Kappa=%{y:.3f}<br>"
                        "Trials=%{customdata[0]:.0f}<br>"
                        "Sessions=%{customdata[1]:.0f}<extra></extra>"
                    ),
                ),
                row=1,
                col=col_idx,
            )
            shown.add(metric)
        for boundary in (low_boundary, high_boundary):
            fig.add_vline(
                x=boundary,
                line=dict(color=COLOR_GRAY, dash="dash", width=2),
                row=1,
                col=col_idx,
            )
        fig.update_xaxes(type="log", row=1, col=col_idx)
        fig.update_yaxes(range=[-1, 1], row=1, col=col_idx)

    if not shown:
        return None
    fig.update_layout(
        title="Chance-Corrected Agreement vs Stimulus",
        xaxis_title="Stimulus (kHz, log)",
        yaxis_title="Cohen's kappa",
        legend_title="Agreement pair",
        height=420,
    )
    return fig


def _session_batch_meta(session_meta: pd.Series) -> dict[str, Any]:
    """Shared session metadata columns for batch CSV exports."""
    dprime_val = session_meta.get("session_dprime", np.nan)
    hit_rate_val = session_meta.get("session_hit_rate", np.nan)
    return {
        "animal": str(session_meta.get("Animal", "")),
        "date": str(session_meta.get("Date", "")),
        "session_type": str(session_meta.get("Session Type", "")),
        "session_dprime": float(dprime_val) if pd.notna(dprime_val) else np.nan,
        "session_hit_rate": float(hit_rate_val) if pd.notna(hit_rate_val) else np.nan,
    }


def _prob_records_from_df(
    prob_df: pd.DataFrame,
    session_meta: pd.Series,
    *,
    histology_fallback: bool,
) -> list[dict[str, Any]]:
    """Attach session metadata to per-session conditional-probability rows."""
    if prob_df.empty:
        return []
    meta = _session_batch_meta(session_meta)
    meta["histology_fallback"] = histology_fallback
    return [{**meta, **row} for row in prob_df.to_dict("records")]


def _fit_weighting_logistic_model(
    frame: pd.DataFrame,
    predictors: list[str],
    *,
    random_state: int,
) -> dict[str, Any] | None:
    """Fit Mouse~predictors logistic model on standardized predictors."""
    if not predictors:
        return None
    required_cols = predictors + ["behavior_go"]
    if not set(required_cols).issubset(frame.columns):
        return None

    work = frame[required_cols].dropna().copy()
    if work.empty or work["behavior_go"].nunique() < 2:
        return None

    x = work[predictors].to_numpy(dtype=float)
    y = work["behavior_go"].to_numpy(dtype=int)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=int(random_state),
        solver="lbfgs",
    )
    model.fit(x_scaled, y)
    prob = model.predict_proba(x_scaled)[:, 1]

    n_obs = int(len(y))
    n_params = int(len(predictors) + 1)  # intercept + coefficients
    ll_model = float(-log_loss(y, prob, normalize=False))
    p_bar = float(np.clip(np.mean(y), 1e-6, 1 - 1e-6))
    ll_null = float(np.sum(y * np.log(p_bar) + (1 - y) * np.log(1 - p_bar)))
    mcfadden_r2 = float(1.0 - (ll_model / ll_null)) if ll_null != 0 else np.nan
    aic = float(2 * n_params - 2 * ll_model)
    bic = float(np.log(max(n_obs, 1)) * n_params - 2 * ll_model)

    coef = model.coef_[0]
    coef_table = pd.DataFrame(
        {
            "predictor": predictors,
            "beta": coef,
            "odds_ratio_exp_beta": np.exp(coef),
        }
    )

    return {
        "n_obs": n_obs,
        "predictors": predictors,
        "coef_table": coef_table,
        "intercept": float(model.intercept_[0]),
        "ll_model": ll_model,
        "ll_null": ll_null,
        "mcfadden_r2": mcfadden_r2,
        "aic": aic,
        "bic": bic,
        "prob": prob,
    }


def _bootstrap_coefficients(
    frame: pd.DataFrame,
    predictors: list[str],
    *,
    random_state: int,
    n_boot: int = 300,
) -> pd.DataFrame:
    """Bootstrap CI for standardized logistic coefficients."""
    required_cols = predictors + ["behavior_go"]
    work = frame[required_cols].dropna().copy()
    if work.empty or work["behavior_go"].nunique() < 2:
        return pd.DataFrame()

    rng = np.random.default_rng(int(random_state))
    coef_samples: dict[str, list[float]] = {pred: [] for pred in predictors}
    idx_all = np.arange(len(work))

    for _ in range(int(n_boot)):
        sample_idx = rng.choice(idx_all, size=len(idx_all), replace=True)
        sample = work.iloc[sample_idx]
        if sample["behavior_go"].nunique() < 2:
            continue
        fit = _fit_weighting_logistic_model(sample, predictors, random_state=int(rng.integers(1_000_000)))
        if fit is None:
            continue
        for row in fit["coef_table"].itertuples(index=False):
            coef_samples[str(row.predictor)].append(float(row.beta))

    rows: list[dict[str, Any]] = []
    for pred in predictors:
        values = np.asarray(coef_samples[pred], dtype=float)
        if values.size == 0:
            rows.append({"predictor": pred, "beta_ci_low": np.nan, "beta_ci_high": np.nan, "n_boot_used": 0})
            continue
        rows.append(
            {
                "predictor": pred,
                "beta_ci_low": float(np.percentile(values, 2.5)),
                "beta_ci_high": float(np.percentile(values, 97.5)),
                "n_boot_used": int(values.size),
            }
        )
    return pd.DataFrame(rows)


def _nested_model_comparison(results_df: pd.DataFrame, random_state: int) -> pd.DataFrame:
    """Compare GT+area models and full GT+ACx+OFC model on fit metrics."""
    model_specs = [
        ("Mouse ~ GT + ACx", ["ground_truth_go", "acx_prob_go"]),
        ("Mouse ~ GT + OFC", ["ground_truth_go", "ofc_prob_go"]),
        ("Mouse ~ GT + ACx + OFC", ["ground_truth_go", "acx_prob_go", "ofc_prob_go"]),
    ]
    rows: list[dict[str, Any]] = []
    for name, predictors in model_specs:
        fit = _fit_weighting_logistic_model(results_df, predictors, random_state=random_state)
        if fit is None:
            continue
        rows.append(
            {
                "model": name,
                "n_obs": fit["n_obs"],
                "aic": fit["aic"],
                "bic": fit["bic"],
                "ll_model": fit["ll_model"],
                "mcfadden_r2": fit["mcfadden_r2"],
            }
        )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values("aic").reset_index(drop=True)
    out["delta_aic_vs_best"] = out["aic"] - out["aic"].min()
    return out


def _weighting_summary_dual_area(results_df: pd.DataFrame, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return coefficient table (with CI) and beta-difference summary for ACx vs OFC."""
    predictors = ["ground_truth_go", "acx_prob_go", "ofc_prob_go"]
    fit = _fit_weighting_logistic_model(results_df, predictors, random_state=random_state)
    if fit is None:
        return pd.DataFrame(), pd.DataFrame()
    ci_table = _bootstrap_coefficients(results_df, predictors, random_state=random_state)
    coef_table = fit["coef_table"].merge(ci_table, on="predictor", how="left")

    acx_beta = float(coef_table.loc[coef_table["predictor"] == "acx_prob_go", "beta"].iloc[0])
    ofc_beta = float(coef_table.loc[coef_table["predictor"] == "ofc_prob_go", "beta"].iloc[0])
    beta_diff = acx_beta - ofc_beta

    # Bootstrap CI for beta difference.
    diff_ci_low = np.nan
    diff_ci_high = np.nan
    boot = _bootstrap_coefficients(results_df, ["acx_prob_go", "ofc_prob_go", "ground_truth_go"], random_state=random_state)
    if not boot.empty:
        pass  # per-coefficient CI is already merged above

    # Compute dedicated bootstrap distribution for beta difference.
    rng = np.random.default_rng(int(random_state) + 77)
    work = results_df[["behavior_go", "ground_truth_go", "acx_prob_go", "ofc_prob_go"]].dropna().copy()
    diffs: list[float] = []
    if not work.empty and work["behavior_go"].nunique() >= 2:
        idx_all = np.arange(len(work))
        for _ in range(300):
            sample = work.iloc[rng.choice(idx_all, size=len(idx_all), replace=True)]
            if sample["behavior_go"].nunique() < 2:
                continue
            fit_boot = _fit_weighting_logistic_model(
                sample,
                ["ground_truth_go", "acx_prob_go", "ofc_prob_go"],
                random_state=int(rng.integers(1_000_000)),
            )
            if fit_boot is None:
                continue
            table = fit_boot["coef_table"].set_index("predictor")
            diffs.append(float(table.loc["acx_prob_go", "beta"] - table.loc["ofc_prob_go", "beta"]))
    if diffs:
        diff_ci_low = float(np.percentile(diffs, 2.5))
        diff_ci_high = float(np.percentile(diffs, 97.5))

    summary = pd.DataFrame(
        [
            {
                "metric": "beta_acx_minus_beta_ofc",
                "value": beta_diff,
                "ci_low": diff_ci_low,
                "ci_high": diff_ci_high,
                "interpretation": (
                    "ACx-weighted"
                    if np.isfinite(diff_ci_low) and np.isfinite(diff_ci_high) and diff_ci_low > 0
                    else "OFC-weighted"
                    if np.isfinite(diff_ci_low) and np.isfinite(diff_ci_high) and diff_ci_high < 0
                    else "No clear dominance"
                ),
            }
        ]
    )
    return coef_table, summary


def _weighting_summary_single_area(results_df: pd.DataFrame, area: str, random_state: int) -> pd.DataFrame:
    area_pred = f"{area.lower()}_prob_go"
    fit = _fit_weighting_logistic_model(results_df, ["ground_truth_go", area_pred], random_state=random_state)
    if fit is None:
        return pd.DataFrame()
    ci = _bootstrap_coefficients(results_df, ["ground_truth_go", area_pred], random_state=random_state)
    return fit["coef_table"].merge(ci, on="predictor", how="left")


def _matrix_head_tables(
    matrix: np.ndarray,
    time_axis: np.ndarray,
    *,
    max_units: int = 6,
    max_time_bins: int = 10,
    max_trials: int = 4,
) -> dict[str, pd.DataFrame]:
    """Per-trial slices of units x time for quick inspection in the UI."""
    n_units = min(max_units, matrix.shape[0])
    n_time = min(max_time_bins, matrix.shape[1])
    n_trials = min(max_trials, matrix.shape[2])
    time_labels = [f"{float(time_axis[i]):.3f}s" for i in range(n_time)]
    tables: dict[str, pd.DataFrame] = {}
    for trial_idx in range(n_trials):
        block = matrix[:n_units, :n_time, trial_idx]
        tables[f"trial_{trial_idx}"] = pd.DataFrame(
            block,
            index=[f"unit_{unit_idx}" for unit_idx in range(n_units)],
            columns=time_labels,
        )
    return tables


def _render_matrix_head_section(session_data: dict[str, Any]) -> None:
    st.subheader("Filtered Event Matrix Preview")
    st.caption("Tone-aligned firing rates after good+MUA and histology filters. Head of [units x time] per trial.")

    cols = st.columns(max(1, len(session_data["areas"])))
    for col, area in zip(cols, session_data["areas"]):
        area_data = session_data[area]
        matrix = area_data["matrix"]
        with col:
            st.markdown(f"**{area}** — shape `{matrix.shape[0]} x {matrix.shape[1]} x {matrix.shape[2]}` "
                        f"(units x time_bins x trials)")
            st.caption(
                f"From full `{area_data['full_matrix_shape']}`; "
                f"{area_data['good_mua_count']} good+MUA → {area_data['unit_count']} histology-matched."
            )
            head_tables = _matrix_head_tables(matrix, session_data["time_axis"])
            for trial_name, table in head_tables.items():
                st.markdown(f"*{trial_name}*")
                st.dataframe(table, use_container_width=True)


# Decision axes: -1 = No-Go, +1 = Go (with a small visual margin on each side).
_AXIS_MIN = -1.4
_AXIS_MAX = 1.4
_AXIS_SPLIT = 0.0  # origin splits No-Go (negative) vs Go (positive) halves


def _go_nogo_axis_values(binary: np.ndarray) -> np.ndarray:
    """Map 0/1 trial labels to -1 (No-Go) / +1 (Go) for scatter axes."""
    return np.where(binary.astype(int) == 1, 1.0, -1.0)


def _agreement_marker_colors(*axis_arrays: np.ndarray) -> list[str]:
    """Red = all No-Go (-1), green = all Go (+1), gray = any disagreement."""
    colors: list[str] = []
    for values in zip(*axis_arrays):
        vals = [int(v) for v in values]
        if all(v == -1 for v in vals):
            colors.append(COLOR_NOGO)
        elif all(v == 1 for v in vals):
            colors.append(COLOR_GO)
        else:
            colors.append(COLOR_GRAY)
    return colors


def _filled_trial_mask(results_df: pd.DataFrame) -> np.ndarray:
    """True for Hit/CR (solid circle); False for Miss/FA (open circle)."""
    if "outcome" not in results_df.columns:
        return (
            results_df["behavior_go"].to_numpy(dtype=int)
            == results_df["ground_truth_go"].to_numpy(dtype=int)
        )
    outcomes = results_df["outcome"].map(_normalize_outcome)
    return outcomes.isin(_FILLED_OUTCOME_LABELS).to_numpy()


def _jitter(values: np.ndarray, scale: float, rng: np.random.Generator) -> np.ndarray:
    # Visual spread for overlapping lattice points; magnitude << 1 so categories stay readable.
    return values.astype(float) + rng.uniform(-scale, scale, size=values.shape)


def _axis_line_shapes_2d() -> list[dict[str, Any]]:
    """Colored axis segments through origin for 2D decision scatters."""
    return [
        {"type": "line", "x0": _AXIS_MIN, "x1": _AXIS_SPLIT, "y0": 0, "y1": 0, "line": {"color": COLOR_NOGO, "width": 3}},
        {"type": "line", "x0": _AXIS_SPLIT, "x1": _AXIS_MAX, "y0": 0, "y1": 0, "line": {"color": COLOR_GO, "width": 3}},
        {"type": "line", "x0": 0, "x1": 0, "y0": _AXIS_MIN, "y1": _AXIS_SPLIT, "line": {"color": COLOR_NOGO, "width": 3}},
        {"type": "line", "x0": 0, "x1": 0, "y0": _AXIS_SPLIT, "y1": _AXIS_MAX, "line": {"color": COLOR_GO, "width": 3}},
    ]


def _axis_kwargs_2d() -> dict[str, Any]:
    return dict(
        tickmode="array",
        tickvals=[-1, 1],
        ticktext=["No-Go", "Go"],
        range=[_AXIS_MIN, _AXIS_MAX],
        zeroline=True,
        zerolinecolor="rgba(80,80,80,0.6)",
        zerolinewidth=2,
        showgrid=True,
        gridcolor="rgba(150,150,150,0.25)",
    )


def _plot_decision_pair_scatter(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    *,
    x_title: str,
    y_title: str,
    trial_colors: list[str],
    filled_mask: np.ndarray,
    hover_texts: list[str],
    random_state: int,
    jitter_scale: float,
    title: str,
    marker_size: float = 9,
) -> go.Figure:
    """One 2D Go/No-Go decision scatter for a pair of sources (e.g. Mouse vs OFC)."""
    rng = np.random.default_rng(int(random_state))
    fig = go.Figure()
    for mask, name, symbol, is_open in (
        (filled_mask, "Hit / CR", "circle", False),
        (~filled_mask, "Miss / FA", "circle", True),
    ):
        if not np.any(mask):
            continue
        idxs = np.flatnonzero(mask)
        point_colors = [trial_colors[i] for i in idxs]
        fig.add_trace(
            go.Scatter(
                x=_jitter(x_vals[mask], jitter_scale, rng),
                y=_jitter(y_vals[mask], jitter_scale, rng),
                mode="markers",
                name=name,
                marker=_scatter_marker_style(symbol, point_colors, size=marker_size, is_open=is_open),
                text=[hover_texts[i] for i in idxs],
                hoverinfo="text",
                showlegend=name == "Hit / CR",
            )
        )

    axis_kwargs = _axis_kwargs_2d()
    fig.update_layout(
        title=title,
        xaxis=dict(title=x_title, **axis_kwargs),
        yaxis=dict(title=y_title, **axis_kwargs),
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    for shape in _axis_line_shapes_2d():
        fig.add_shape(layer="below", **shape)
    return fig


def _scatter_marker_style(
    symbol: str,
    colors: list[str],
    *,
    size: float,
    is_open: bool,
) -> dict[str, Any]:
    """Hit/CR = solid fill; Miss/FA = transparent fill with colored outline."""
    if is_open:
        return dict(
            symbol=symbol,
            size=size,
            color="rgba(255,255,255,0)",
            line=dict(width=2.5, color=colors),
        )
    return dict(
        symbol=symbol,
        size=size,
        color=colors,
        line=dict(width=1, color="black"),
    )


def _plot_decisions_scatter_3d(
    results_df: pd.DataFrame,
    random_state: int,
    jitter_scale: float,
) -> go.Figure | None:
    """3D ACx x OFC x Mouse decision scatter (dual-area sessions only)."""
    if not {"acx_pred", "ofc_pred", "behavior_go", "ground_truth_go"}.issubset(results_df.columns):
        return None

    rng = np.random.default_rng(int(random_state))
    filled_mask = _filled_trial_mask(results_df)
    mouse_vals = _go_nogo_axis_values(results_df["behavior_go"].to_numpy(dtype=int))
    acx_vals = _go_nogo_axis_values(results_df["acx_pred"].to_numpy(dtype=int))
    ofc_vals = _go_nogo_axis_values(results_df["ofc_pred"].to_numpy(dtype=int))
    trial_colors = _agreement_marker_colors(acx_vals, ofc_vals, mouse_vals)
    gt_vals = results_df["ground_truth_go"].to_numpy(dtype=int)
    outcome_vals = (
        results_df["outcome"].astype(str).tolist() if "outcome" in results_df.columns else ["?"] * len(results_df)
    )

    def _hover_text(idx: int) -> str:
        return (
            f"Trial {idx}<br>"
            f"Outcome: {outcome_vals[idx]}<br>"
            f"Ground truth: {'Go' if gt_vals[idx] == 1 else 'No-Go'}<br>"
            f"Mouse: {'Go' if mouse_vals[idx] == 1 else 'No-Go'}<br>"
            f"ACx: {'Go' if acx_vals[idx] == 1 else 'No-Go'}<br>"
            f"OFC: {'Go' if ofc_vals[idx] == 1 else 'No-Go'}"
        )

    fig = go.Figure()
    axis_segments_3d = [
        ([_AXIS_MIN, _AXIS_SPLIT], [0, 0], [0, 0], COLOR_NOGO),
        ([_AXIS_SPLIT, _AXIS_MAX], [0, 0], [0, 0], COLOR_GO),
        ([0, 0], [_AXIS_MIN, _AXIS_SPLIT], [0, 0], COLOR_NOGO),
        ([0, 0], [_AXIS_SPLIT, _AXIS_MAX], [0, 0], COLOR_GO),
        ([0, 0], [0, 0], [_AXIS_MIN, _AXIS_SPLIT], COLOR_NOGO),
        ([0, 0], [0, 0], [_AXIS_SPLIT, _AXIS_MAX], COLOR_GO),
    ]
    for xs, ys, zs, color in axis_segments_3d:
        fig.add_trace(
            go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="lines",
                line=dict(color=color, width=6),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    hover_all = [_hover_text(i) for i in range(len(results_df))]
    for mask, name, symbol, is_open in (
        (filled_mask, "Hit / CR", "circle", False),
        (~filled_mask, "Miss / FA", "circle", True),
    ):
        if not np.any(mask):
            continue
        idxs = np.flatnonzero(mask)
        fig.add_trace(
            go.Scatter3d(
                x=_jitter(acx_vals[mask], jitter_scale, rng),
                y=_jitter(ofc_vals[mask], jitter_scale, rng),
                z=_jitter(mouse_vals[mask], jitter_scale, rng),
                mode="markers",
                name=name,
                marker=_scatter_marker_style(symbol, [trial_colors[i] for i in idxs], size=5, is_open=is_open),
                text=[hover_all[i] for i in idxs],
                hoverinfo="text",
            )
        )

    scene_axis = dict(
        tickmode="array",
        tickvals=[-1, 1],
        ticktext=["No-Go", "Go"],
        range=[_AXIS_MIN, _AXIS_MAX],
        showbackground=False,
        zeroline=True,
        zerolinecolor="rgba(80,80,80,0.6)",
        zerolinewidth=2,
        gridcolor="rgba(150,150,150,0.25)",
    )
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="ACx Decoder", **scene_axis),
            yaxis=dict(title="OFC Decoder", **scene_axis),
            zaxis=dict(title="Mouse", **scene_axis),
            aspectmode="cube",
        ),
        title="3D: ACx × OFC × Mouse",
        height=560,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _plot_decisions_scatter_dual_pairs(
    results_df: pd.DataFrame,
    random_state: int,
    jitter_scale: float,
) -> dict[str, go.Figure] | None:
    """Three 2D pairwise views: Mouse–OFC, Mouse–ACx, OFC–ACx."""
    if not {"acx_pred", "ofc_pred", "behavior_go", "ground_truth_go"}.issubset(results_df.columns):
        return None

    filled_mask = _filled_trial_mask(results_df)
    mouse_vals = _go_nogo_axis_values(results_df["behavior_go"].to_numpy(dtype=int))
    acx_vals = _go_nogo_axis_values(results_df["acx_pred"].to_numpy(dtype=int))
    ofc_vals = _go_nogo_axis_values(results_df["ofc_pred"].to_numpy(dtype=int))
    trial_colors = _agreement_marker_colors(acx_vals, ofc_vals, mouse_vals)
    gt_vals = results_df["ground_truth_go"].to_numpy(dtype=int)
    outcome_vals = (
        results_df["outcome"].astype(str).tolist() if "outcome" in results_df.columns else ["?"] * len(results_df)
    )
    hover_all = [
        (
            f"Trial {i}<br>Outcome: {outcome_vals[i]}<br>"
            f"GT: {'Go' if gt_vals[i] == 1 else 'No-Go'} | "
            f"Mouse: {'Go' if mouse_vals[i] == 1 else 'No-Go'} | "
            f"ACx: {'Go' if acx_vals[i] == 1 else 'No-Go'} | "
            f"OFC: {'Go' if ofc_vals[i] == 1 else 'No-Go'}"
        )
        for i in range(len(results_df))
    ]

    # Offset random_state per panel so jitter differs slightly between views.
    return {
        "mouse_ofc": _plot_decision_pair_scatter(
            ofc_vals, mouse_vals,
            x_title="OFC Decoder", y_title="Mouse",
            trial_colors=trial_colors,
            filled_mask=filled_mask,
            hover_texts=hover_all,
            random_state=random_state + 1,
            jitter_scale=jitter_scale,
            title="Mouse vs OFC",
        ),
        "mouse_acx": _plot_decision_pair_scatter(
            acx_vals, mouse_vals,
            x_title="ACx Decoder", y_title="Mouse",
            trial_colors=trial_colors,
            filled_mask=filled_mask,
            hover_texts=hover_all,
            random_state=random_state + 2,
            jitter_scale=jitter_scale,
            title="Mouse vs ACx",
        ),
        "ofc_acx": _plot_decision_pair_scatter(
            acx_vals, ofc_vals,
            x_title="ACx Decoder", y_title="OFC Decoder",
            trial_colors=trial_colors,
            filled_mask=filled_mask,
            hover_texts=hover_all,
            random_state=random_state + 3,
            jitter_scale=jitter_scale,
            title="OFC vs ACx",
        ),
    }


def _plot_decisions_scatter(
    results_df: pd.DataFrame,
    available_areas: list[str],
    random_state: int,
    jitter_scale: float,
) -> go.Figure | None:
    """Single-area: one 2D scatter (area vs Mouse). Dual-area uses dedicated 3D/pair helpers."""
    if "behavior_go" not in results_df.columns or "ground_truth_go" not in results_df.columns:
        return None
    if not available_areas:
        return None

    filled_mask = _filled_trial_mask(results_df)
    mouse_vals = _go_nogo_axis_values(results_df["behavior_go"].to_numpy(dtype=int))
    gt_vals = results_df["ground_truth_go"].to_numpy(dtype=int)
    outcome_vals = (
        results_df["outcome"].astype(str).tolist() if "outcome" in results_df.columns else ["?"] * len(results_df)
    )

    def _hover_text(idx: int, area_preds: dict[str, float]) -> str:
        area_lines = [
            f"{name}: {'Go' if int(val) == 1 else 'No-Go'}" for name, val in area_preds.items()
        ]
        return (
            f"Trial {idx}<br>"
            f"Outcome: {outcome_vals[idx]}<br>"
            f"Ground truth: {'Go' if gt_vals[idx] == 1 else 'No-Go'}<br>"
            f"Mouse: {'Go' if mouse_vals[idx] == 1 else 'No-Go'}<br>"
            + "<br>".join(area_lines)
        )

    if len(available_areas) == 2 and "acx_pred" in results_df.columns and "ofc_pred" in results_df.columns:
        return None

    # Single-area fallback: 2D scatter with mouse on the y-axis.
    area = available_areas[0]
    area_col = f"{area.lower()}_pred"
    if area_col not in results_df.columns:
        return None
    area_vals = _go_nogo_axis_values(results_df[area_col].to_numpy(dtype=int))
    trial_colors = _agreement_marker_colors(area_vals, mouse_vals)
    hover_all = [
        _hover_text(i, {area: int(area_vals[i])}) for i in range(len(results_df))
    ]
    return _plot_decision_pair_scatter(
        area_vals,
        mouse_vals,
        x_title=f"{area} Decoder",
        y_title="Mouse",
        trial_colors=trial_colors,
        filled_mask=filled_mask,
        hover_texts=hover_all,
        random_state=int(random_state),
        jitter_scale=jitter_scale,
        title=f"{area} vs Mouse",
        marker_size=10,
    )


def _prepare_session_selection_table(valid_sessions_df: pd.DataFrame) -> pd.DataFrame:
    """Subset of monitoring columns for the checkbox session picker (matches NPXL monitoring UI)."""
    table_df = valid_sessions_df.copy()
    if "Checkbox" not in table_df.columns:
        table_df.insert(0, "Checkbox", False)

    display_columns = [
        column
        for column in [
            "Checkbox",
            "Animal",
            "Date",
            "Session Type",
            "session_dprime",
            "session_hit_rate",
            "acx_total_units",
            "ofc_total_units",
            "acx_histology_matched",
            "ofc_histology_matched",
            "acx_histology_match_pct",
            "ofc_histology_match_pct",
            "session_label",
        ]
        if column in table_df.columns
    ]
    return table_df[display_columns]


def _render_metrics(area: str, decoder_result: dict[str, Any], unit_count: int, trial_count: int) -> None:
    st.markdown(f"#### {area}")
    metrics = decoder_result["metrics"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    col2.metric("Precision", f"{metrics['precision']:.3f}")
    col3.metric("Recall", f"{metrics['recall']:.3f}")
    col4, col5, col6 = st.columns(3)
    col4.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    col5.metric("Units", f"{unit_count:,}")
    col6.metric("Trials / folds", f"{trial_count:,} / {metrics['n_splits']}")
    if decoder_result["warnings"]:
        st.warning("; ".join(sorted(set(decoder_result["warnings"]))))


# --- Per-session batch helpers for group-level analysis ---

def _run_one_session_batch(
    session_dir: str,
    session_meta: pd.Series,
    *,
    decode_window: tuple[float, float],
    aggregation: str,
    classifier_type: str,
    random_state: int,
    min_trials_per_class: int,
    use_histology: bool,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Decode one session for both behavior and GT targets.

    Returns
    -------
    area_records, joint_prob_records, marginal_prob_records, single_area_prob_records,
    psychometric_prob_records, model_agreement_records, psychometric_model_agree_records,
    psychometric_model_disagree_records, kappa_agreement_records, kappa_by_stimulus_records.

    Runs both decode targets in a single data-loading pass to avoid loading event matrices twice.
    """
    empty_probs: list[dict[str, Any]] = []
    # Fall back to all good+MUA units when histology matching finds too few units in
    # any available area — ensures single-probe or no-histology-file sessions still
    # contribute their available area(s) to the group analysis.
    histology_fallback = False
    try:
        session_data = load_session_event_data(session_dir, use_histology=use_histology)
    except (ValueError, OSError):
        if use_histology:
            session_data = load_session_event_data(session_dir, use_histology=False)
            histology_fallback = True
        else:
            raise
    available_areas = session_data["areas"]
    time_axis = session_data["time_axis"]

    primary_area = "ACx" if "ACx" in available_areas else available_areas[0]
    labels_df = _build_trial_labels(session_data[primary_area]["trials_df"])
    if len(labels_df) == 0:
        return (
            [],
            empty_probs,
            empty_probs,
            empty_probs,
            empty_probs,
            empty_probs,
            empty_probs,
            empty_probs,
            empty_probs,
            empty_probs,
        )

    trial_positions = labels_df["source_trial_position"].to_numpy(dtype=int)
    session_key_prefix = f"{int(session_meta.get('session_row_index', 0))}:{session_dir}"

    # Build one feature matrix per area (shared across both decode targets).
    area_features: dict[str, np.ndarray] = {
        area: _feature_matrix(
            session_data[area]["matrix"][:, :, trial_positions],
            time_axis,
            decode_window,
            aggregation,
        )
        for area in available_areas
    }

    y_behavior = labels_df["behavior_go"].to_numpy(dtype=int)
    y_gt = labels_df["ground_truth_go"].to_numpy(dtype=int)
    behavior_results: dict[str, dict[str, Any]] = {}
    gt_results: dict[str, dict[str, Any]] = {}
    for area in available_areas:
        base_key = session_key_prefix + f":{area}"
        behavior_results[area] = train_oof_decoder(
            area, base_key + "|target=behavior_go",
            area_features[area], y_behavior,
            classifier_type, TRAIN_TEST_FOLDS, random_state, min_trials_per_class,
        )
        gt_results[area] = train_oof_decoder(
            area, base_key + "|target=ground_truth_go",
            area_features[area], y_gt,
            classifier_type, TRAIN_TEST_FOLDS, random_state, min_trials_per_class,
        )

    # Weighting betas from dual-area GT model (shared scalar per session).
    acx_beta = ofc_beta = beta_diff = diff_ci_low = diff_ci_high = np.nan
    if len(available_areas) == 2:
        weights_df = labels_df.copy()
        for area in available_areas:
            area_l = area.lower()
            weights_df[f"{area_l}_prob_go"] = gt_results[area]["probability"]
            weights_df[f"{area_l}_pred"] = gt_results[area]["prediction"]
        coef_df, diff_df = _weighting_summary_dual_area(weights_df, random_state)
        if not coef_df.empty:
            coef_idx = coef_df.set_index("predictor")
            acx_beta = float(coef_idx.loc["acx_prob_go", "beta"]) if "acx_prob_go" in coef_idx.index else np.nan
            ofc_beta = float(coef_idx.loc["ofc_prob_go", "beta"]) if "ofc_prob_go" in coef_idx.index else np.nan
            beta_diff = (acx_beta - ofc_beta) if np.isfinite(acx_beta) and np.isfinite(ofc_beta) else np.nan
        if not diff_df.empty:
            diff_ci_low = float(diff_df["ci_low"].iloc[0])
            diff_ci_high = float(diff_df["ci_high"].iloc[0])

    # Agreement probabilities use GT-decoder OOF predictions (same as Streamlit "Ground truth" mode).
    gt_results_df = labels_df.copy()
    for area in available_areas:
        area_l = area.lower()
        gt_results_df[f"{area_l}_pred"] = gt_results[area]["prediction"]
        gt_results_df[f"{area_l}_prob_go"] = gt_results[area]["probability"]

    joint_prob_records = empty_probs
    marginal_prob_records = empty_probs
    single_area_prob_records = empty_probs
    if len(available_areas) == 2:
        joint_prob_records = _prob_records_from_df(
            _build_joint_conditional_probabilities(gt_results_df),
            session_meta,
            histology_fallback=histology_fallback,
        )
        marginal_prob_records = _prob_records_from_df(
            _build_marginal_probabilities(gt_results_df, available_areas),
            session_meta,
            histology_fallback=histology_fallback,
        )
    elif len(available_areas) == 1:
        area = available_areas[0]
        single_area_prob_records = _prob_records_from_df(
            _build_single_area_conditional_probabilities(gt_results_df, area),
            session_meta,
            histology_fallback=histology_fallback,
        )
        marginal_prob_records = _prob_records_from_df(
            _build_marginal_probabilities(gt_results_df, available_areas),
            session_meta,
            histology_fallback=histology_fallback,
        )

    psychometric_prob_records = empty_probs
    for error_only in (False, True):
        psychometric_prob_records.extend(
            _prob_records_from_df(
                _build_psychometric_by_stimulus(gt_results_df, error_trials_only=error_only),
                session_meta,
                histology_fallback=histology_fallback,
            )
        )

    model_agreement_records = empty_probs
    psychometric_model_agree_records = empty_probs
    psychometric_model_disagree_records = empty_probs
    kappa_agreement_records = empty_probs
    kappa_by_stimulus_records = empty_probs
    if len(available_areas) == 2:
        model_agreement_records = _prob_records_from_df(
            _build_model_agreement_summary(gt_results_df),
            session_meta,
            histology_fallback=histology_fallback,
        )
        psychometric_model_agree_records.extend(
            _prob_records_from_df(
                _build_psychometric_behavior_match_by_stimulus(
                    gt_results_df,
                    models_agree_only=True,
                    min_trials_per_bin=_PSYCHOMETRIC_EXPORT_MIN_TRIALS_PER_BIN,
                ),
                session_meta,
                histology_fallback=histology_fallback,
            )
        )
        psychometric_model_agree_records.extend(
            _prob_records_from_df(
                _build_psychometric_behavior_match_by_stimulus(
                    gt_results_df,
                    split_by_agreement_state=True,
                    min_trials_per_bin=_PSYCHOMETRIC_EXPORT_MIN_TRIALS_PER_BIN,
                ),
                session_meta,
                histology_fallback=histology_fallback,
            )
        )
        psychometric_model_disagree_records = _prob_records_from_df(
            _build_psychometric_disagreement_area_match_by_stimulus(gt_results_df),
            session_meta,
            histology_fallback=histology_fallback,
        )
        kappa_agreement_records = _prob_records_from_df(
            _build_kappa_agreement_summary(gt_results_df),
            session_meta,
            histology_fallback=histology_fallback,
        )
        kappa_by_stimulus_records = _prob_records_from_df(
            _build_kappa_by_stimulus(gt_results_df),
            session_meta,
            histology_fallback=histology_fallback,
        )

    session_meta_cols = _session_batch_meta(session_meta)
    records: list[dict[str, Any]] = []
    for area in available_areas:
        bm = behavior_results[area]["metrics"]
        gm = gt_results[area]["metrics"]
        records.append(
            {
                **session_meta_cols,
                "area": area,
                "n_units": int(session_data[area]["unit_count"]),
                "n_trials": int(len(labels_df)),
                "n_folds": int(bm["n_splits"]),
                # Behavior-target metrics
                "accuracy": float(bm["accuracy"]),
                "precision": float(bm["precision"]),
                "recall": float(bm["recall"]),
                "roc_auc": float(bm["roc_auc"]),
                # Ground-truth-target metrics
                "accuracy_gt": float(gm["accuracy"]),
                "precision_gt": float(gm["precision"]),
                "recall_gt": float(gm["recall"]),
                "roc_auc_gt": float(gm["roc_auc"]),
                # Area weighting (dual-area sessions only; NaN for single-area)
                "acx_beta": acx_beta,
                "ofc_beta": ofc_beta,
                "beta_diff": beta_diff,
                "beta_diff_ci_low": diff_ci_low,
                "beta_diff_ci_high": diff_ci_high,
                # True when histology filter was skipped due to insufficient matched units
                "histology_fallback": histology_fallback,
            }
        )
    return (
        records,
        joint_prob_records,
        marginal_prob_records,
        single_area_prob_records,
        psychometric_prob_records,
        model_agreement_records,
        psychometric_model_agree_records,
        psychometric_model_disagree_records,
        kappa_agreement_records,
        kappa_by_stimulus_records,
    )


@st.cache_data(show_spinner=False)
def _batch_decode_all_sessions(
    monitoring_path: str,
    decode_window: tuple[float, float],
    aggregation: str,
    classifier_type: str,
    random_state: int,
    min_trials_per_class: int,
    use_histology: bool,
) -> tuple[
    pd.DataFrame,
    list[dict[str, str]],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Cached batch decoder — iterate all valid sessions; return metrics and sidecar tables."""
    sessions_df = load_valid_sessions(monitoring_path)
    records: list[dict[str, Any]] = []
    joint_prob_records: list[dict[str, Any]] = []
    marginal_prob_records: list[dict[str, Any]] = []
    single_area_prob_records: list[dict[str, Any]] = []
    psychometric_prob_records: list[dict[str, Any]] = []
    model_agreement_records: list[dict[str, Any]] = []
    psychometric_model_agree_records: list[dict[str, Any]] = []
    psychometric_model_disagree_records: list[dict[str, Any]] = []
    kappa_agreement_records: list[dict[str, Any]] = []
    kappa_by_stimulus_records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for _, row in sessions_df.iterrows():
        session_dir = str(row.get("current_dir", "")).strip()
        label = str(row.get("session_label", "?"))
        try:
            (
                area_recs,
                joint_recs,
                marginal_recs,
                single_recs,
                psych_recs,
                agree_recs,
                psych_agree_recs,
                psych_disagree_recs,
                kappa_recs,
                kappa_stim_recs,
            ) = _run_one_session_batch(
                session_dir,
                row,
                decode_window=decode_window,
                aggregation=aggregation,
                classifier_type=classifier_type,
                random_state=random_state,
                min_trials_per_class=min_trials_per_class,
                use_histology=use_histology,
            )
            records.extend(area_recs)
            joint_prob_records.extend(joint_recs)
            marginal_prob_records.extend(marginal_recs)
            single_area_prob_records.extend(single_recs)
            psychometric_prob_records.extend(psych_recs)
            model_agreement_records.extend(agree_recs)
            psychometric_model_agree_records.extend(psych_agree_recs)
            psychometric_model_disagree_records.extend(psych_disagree_recs)
            kappa_agreement_records.extend(kappa_recs)
            kappa_by_stimulus_records.extend(kappa_stim_recs)
        except Exception as exc:  # noqa: BLE001 — per-session failure must not abort the batch
            failures.append({"session": label, "error": str(exc)})
    return (
        pd.DataFrame(records),
        failures,
        pd.DataFrame(joint_prob_records),
        pd.DataFrame(model_agreement_records),
        pd.DataFrame(psychometric_model_agree_records),
        pd.DataFrame(psychometric_model_disagree_records),
        pd.DataFrame(kappa_agreement_records),
        pd.DataFrame(kappa_by_stimulus_records),
    )


# Batch CSV names produced by run_npxl_group_decoder.sbatch (see OUTPUT_CSV + --no_histology).
_GROUP_DECODER_CSV_CANDIDATES: dict[bool, tuple[str, ...]] = {
    True: (
        "npxl_group_decoder_results_with_histology_histology.csv",
        "npxl_group_decoder_results_histology.csv",
    ),
    False: (
        "npxl_group_decoder_results_no_histology.csv",
        "npxl_group_decoder_results_no_histology_no_histology.csv",
    ),
}


def _group_decoder_results_dir() -> str:
    """Directory where run_npxl_group_decoder.sbatch / npxl_group_decoder_batch.py write CSVs."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results")


def _group_decoder_results_path(use_histology: bool) -> str:
    """Preferred batch CSV path for histology-filtered vs all good+MUA runs."""
    name = _GROUP_DECODER_CSV_CANDIDATES[use_histology][0]
    return os.path.join(_group_decoder_results_dir(), name)


def _group_decoder_failures_path(results_path: str) -> str:
    return f"{os.path.splitext(results_path)[0]}_failures.csv"


@st.cache_data(show_spinner=False)
def _read_group_decoder_csv(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def _resolve_group_decoder_results_path(use_histology: bool) -> str | None:
    """Return the first existing batch CSV for the requested histology mode."""
    results_dir = _group_decoder_results_dir()
    for name in _GROUP_DECODER_CSV_CANDIDATES[use_histology]:
        path = os.path.join(results_dir, name)
        if os.path.isfile(path):
            return path
    legacy = os.path.join(results_dir, "npxl_group_decoder_results.csv")
    if os.path.isfile(legacy):
        return legacy
    return None


def _read_group_decoder_sidecar(results_path: str, suffix: str) -> pd.DataFrame:
    """Load a sidecar CSV next to the main group decoder results file."""
    path = f"{os.path.splitext(results_path)[0]}_{suffix}.csv"
    if os.path.isfile(path):
        return _read_group_decoder_csv(path)
    return pd.DataFrame()


def _group_decoder_sidecar_path(results_path: str | None, suffix: str) -> str | None:
    """Expected sidecar CSV path for a resolved group results CSV."""
    if results_path is None:
        return None
    return f"{os.path.splitext(results_path)[0]}_{suffix}.csv"


def _group_decoder_alternate_sidecar_hint(use_histology: bool, suffix: str) -> str:
    """Explain when the opposite histology mode has the requested sidecar."""
    other_results_path = _resolve_group_decoder_results_path(not use_histology)
    other_sidecar_path = _group_decoder_sidecar_path(other_results_path, suffix)
    if other_sidecar_path is None or not os.path.isfile(other_sidecar_path):
        return ""

    other_label = "histology-filtered" if not use_histology else "all good+MUA / no-histology"
    toggle_instruction = (
        "turn **off** `Use all good+MUA units (skip histology filter)`"
        if not use_histology
        else "turn **on** `Use all good+MUA units (skip histology filter)`"
    )
    return (
        f" A matching sidecar exists for the **{other_label}** result set "
        f"(`{os.path.basename(other_sidecar_path)}`); {toggle_instruction} to view it."
    )


def _load_group_decoder_batch_results(
    use_histology: bool,
) -> tuple[
    pd.DataFrame,
    list[dict[str, str]],
    str | None,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Load pre-computed group decoder output and agreement sidecars from the cluster batch job."""
    results_path = _resolve_group_decoder_results_path(use_histology)
    if results_path is None:
        return (
            pd.DataFrame(),
            [],
            None,
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    batch_df = _read_group_decoder_csv(results_path)
    failures_path = _group_decoder_failures_path(results_path)
    failures: list[dict[str, str]] = []
    if os.path.isfile(failures_path):
        failures = pd.read_csv(failures_path).to_dict("records")
    joint_probs_df = _read_group_decoder_sidecar(results_path, "joint_probs")
    model_agreement_df = _read_group_decoder_sidecar(results_path, "model_agreement")
    psychometric_model_agree_df = _read_group_decoder_sidecar(results_path, "psychometric_model_agree")
    psychometric_model_disagree_df = _read_group_decoder_sidecar(
        results_path, "psychometric_model_disagree"
    )
    kappa_agreement_df = _read_group_decoder_sidecar(results_path, "kappa_agreement")
    kappa_by_stimulus_df = _read_group_decoder_sidecar(results_path, "kappa_by_stimulus")
    return (
        batch_df,
        failures,
        results_path,
        joint_probs_df,
        model_agreement_df,
        psychometric_model_agree_df,
        psychometric_model_disagree_df,
        kappa_agreement_df,
        kappa_by_stimulus_df,
    )


def _build_group_summary_table(batch_df: pd.DataFrame) -> pd.DataFrame:
    """Mean ± SEM per (session_type × area) for decoder accuracy and ROC-AUC."""
    def _sem(x: pd.Series) -> float:
        return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan

    agg: dict[str, tuple[str, str]] = {
        "n_sessions": ("accuracy", "count"),
        "mean_accuracy": ("accuracy", "mean"),
        "sem_accuracy": ("accuracy", _sem),
        "mean_accuracy_gt": ("accuracy_gt", "mean"),
        "sem_accuracy_gt": ("accuracy_gt", _sem),
    }
    if "roc_auc" in batch_df.columns:
        agg["mean_roc_auc"] = ("roc_auc", "mean")
        agg["sem_roc_auc"] = ("roc_auc", _sem)
    if "roc_auc_gt" in batch_df.columns:
        agg["mean_roc_auc_gt"] = ("roc_auc_gt", "mean")
        agg["sem_roc_auc_gt"] = ("roc_auc_gt", _sem)
    if "session_dprime" in batch_df.columns:
        agg["mean_session_dprime"] = ("session_dprime", "mean")
        agg["sem_session_dprime"] = ("session_dprime", _sem)
    if "session_hit_rate" in batch_df.columns:
        agg["mean_hit_rate"] = ("session_hit_rate", "mean")
        agg["sem_hit_rate"] = ("session_hit_rate", _sem)

    summary = (
        batch_df.groupby(["session_type", "area"], observed=True)
        .agg(**agg)
        .reset_index()
    )
    type_order = {s: i for i, s in enumerate(_order_session_types(summary["session_type"]))}
    summary["_sort"] = summary["session_type"].map(type_order)
    return summary.sort_values(["_sort", "area"]).drop(columns="_sort").reset_index(drop=True)


_GROUP_STAGE_RANK: dict[str, int] = {"Novice": 0, "1b Expert": 1, "2b Expert": 2, "Other": 99}


def _classify_learning_stage(session_type: str) -> str:
    """Map session_type label to Novice / 1b Expert / 2b Expert (matches NPXL naming)."""
    s = str(session_type).lower()
    if "novice" in s:
        return "Novice"
    if "2b" in s:
        return "2b Expert"
    if "1b" in s:
        return "1b Expert"
    return "Other"


def _session_type_plot_label(session_type: str) -> str:
    """Compact session_type for plot axes (e.g. drop trailing 'Categorization')."""
    s = str(session_type).strip()
    suffix = " categorization"
    if s.lower().endswith(suffix):
        return s[: -len(suffix)]
    return s


def _order_session_types(session_types: Any) -> list[str]:
    """Left-to-right plot order: Novice → 1b Expert → 2b Expert → other."""
    unique = {str(s) for s in session_types if pd.notna(s) and str(s).strip()}
    return sorted(
        unique,
        key=lambda s: (_GROUP_STAGE_RANK.get(_classify_learning_stage(s), 99), s),
    )


def _p_value_to_sig(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _accuracy_groups_by_session_type(
    batch_df: pd.DataFrame,
    area: str,
    metric_col: str,
    *,
    min_n: int = 2,
) -> dict[str, np.ndarray]:
    """Session-level accuracy values per session_type for one area (unpaired groups)."""
    area_df = batch_df[batch_df["area"] == area]
    groups: dict[str, np.ndarray] = {}
    for stype in _order_session_types(area_df["session_type"]):
        vals = pd.to_numeric(
            area_df.loc[area_df["session_type"].astype(str) == stype, metric_col],
            errors="coerce",
        ).dropna()
        if len(vals) >= min_n:
            groups[stype] = vals.to_numpy(dtype=float)
    return groups


def _run_group_metric_chance_tests(
    batch_df: pd.DataFrame,
    metric_col: str,
    *,
    min_n: int = 2,
    chance_level: float = 0.5,
) -> pd.DataFrame:
    """One-sample session-level tests against chance for each session type × area."""
    from scipy.stats import wilcoxon

    rows: list[dict[str, Any]] = []

    for area in sorted(batch_df["area"].dropna().astype(str).unique()):
        groups = _accuracy_groups_by_session_type(batch_df, area, metric_col, min_n=min_n)
        for session_type, values in groups.items():
            deltas = values - chance_level
            sem = float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else np.nan
            if np.allclose(deltas, 0.0):
                stat, p = 0.0, 1.0
            else:
                nonzero_deltas = deltas[~np.isclose(deltas, 0.0)]
                if len(nonzero_deltas) < min_n:
                    stat, p = np.nan, np.nan
                else:
                    try:
                        stat, p = wilcoxon(nonzero_deltas, alternative="two-sided")
                    except ValueError:
                        stat, p = np.nan, np.nan
            rows.append(
                {
                    "area": area,
                    "session_type": session_type,
                    "n_sessions": len(values),
                    "mean_metric": float(np.mean(values)),
                    "sem_metric": sem,
                    "chance_level": chance_level,
                    "test": "Wilcoxon signed-rank",
                    "statistic": float(stat) if np.isfinite(stat) else np.nan,
                    "p": float(p) if np.isfinite(p) else np.nan,
                }
            )

    stats_df = pd.DataFrame(rows)
    if stats_df.empty:
        return stats_df

    n_tests = int(np.isfinite(stats_df["p"]).sum())
    if n_tests == 0:
        stats_df["p_adj"] = np.nan
        stats_df["sig"] = ""
        return stats_df

    stats_df["p_adj"] = stats_df["p"].map(
        lambda p: min(1.0, float(p) * n_tests) if np.isfinite(p) else np.nan
    )
    stats_df["sig"] = stats_df["p_adj"].map(lambda p: _p_value_to_sig(float(p)) if np.isfinite(p) else "")
    return stats_df


def _run_group_accuracy_session_type_tests(
    batch_df: pd.DataFrame,
    metric_col: str,
    *,
    min_n: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Kruskal-Wallis (≥3 groups) + pairwise Mann-Whitney U within each area."""
    from itertools import combinations

    from scipy.stats import kruskal, mannwhitneyu

    omnibus_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []

    for area in sorted(batch_df["area"].dropna().astype(str).unique()):
        groups = _accuracy_groups_by_session_type(batch_df, area, metric_col, min_n=min_n)
        testable = list(groups.keys())
        if len(testable) < 2:
            continue

        if len(testable) >= 3:
            stat, p = kruskal(*(groups[g] for g in testable))
            omnibus_rows.append(
                {
                    "area": area,
                    "test": "Kruskal-Wallis",
                    "groups": ", ".join(testable),
                    "statistic": float(stat),
                    "p": float(p),
                    "sig": _p_value_to_sig(float(p)),
                }
            )

        pair_stats: list[dict[str, Any]] = []
        for group_a, group_b in combinations(testable, 2):
            vals_a = groups[group_a]
            vals_b = groups[group_b]
            try:
                stat_u, p = mannwhitneyu(vals_a, vals_b, alternative="two-sided")
            except ValueError:
                stat_u, p = np.nan, np.nan
            pair_stats.append(
                {
                    "area": area,
                    "group_a": group_a,
                    "group_b": group_b,
                    "n_a": len(vals_a),
                    "n_b": len(vals_b),
                    "U": float(stat_u) if np.isfinite(stat_u) else np.nan,
                    "p": float(p) if np.isfinite(p) else np.nan,
                }
            )

        n_tests = sum(1 for row in pair_stats if np.isfinite(row["p"]))
        for row in pair_stats:
            p_raw = row["p"]
            p_adj = min(1.0, p_raw * n_tests) if np.isfinite(p_raw) and n_tests > 0 else np.nan
            pairwise_rows.append(
                {
                    **row,
                    "p_adj": p_adj,
                    "sig": _p_value_to_sig(p_adj) if np.isfinite(p_adj) else "",
                }
            )

    return pd.DataFrame(omnibus_rows), pd.DataFrame(pairwise_rows)


def _render_group_metric_chance_stats(
    batch_df: pd.DataFrame,
    metric_col: str,
    title: str,
    *,
    metric_label: str,
) -> None:
    """Show one-sample tests against chance (0.5) for each session type and area."""
    if metric_col not in batch_df.columns:
        st.caption(f"{title}: column `{metric_col}` not found in batch results.")
        return

    stats_df = _run_group_metric_chance_tests(batch_df, metric_col)
    if stats_df.empty:
        st.caption(f"{title}: not enough sessions per group (need ≥2 per session type and area).")
        return

    st.markdown(f"**{title}**")
    st.caption(
        f"Session-level cross-validated {metric_label}; Wilcoxon signed-rank tests each "
        "session type × area against 0.5 chance. Bonferroni correction across tested groups."
    )
    st.dataframe(
        stats_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "n_sessions": st.column_config.NumberColumn("N", format="%d"),
            "mean_metric": st.column_config.NumberColumn(f"Mean {metric_label}", format="%.3f"),
            "sem_metric": st.column_config.NumberColumn("SEM", format="%.3f"),
            "chance_level": st.column_config.NumberColumn("Chance", format="%.3f"),
            "statistic": st.column_config.NumberColumn("W", format="%.3f"),
            "p": st.column_config.NumberColumn(format="%.4g"),
            "p_adj": st.column_config.NumberColumn("p (Bonferroni)", format="%.4g"),
        },
    )


def _plot_group_accuracy_strip(
    batch_df: pd.DataFrame,
    metric_col: str,
    title: str,
) -> go.Figure:
    """Box + individual session dots per (session_type × area), grouped by area color."""
    fig = go.Figure()
    for area in sorted(batch_df["area"].unique()):
        area_df = batch_df[batch_df["area"] == area]
        fig.add_trace(
            go.Box(
                x=area_df["session_type"].map(_session_type_plot_label),
                y=area_df[metric_col],
                name=area,
                marker_color=AREA_COLORS.get(area, "#888888"),
                boxpoints="all",
                jitter=0.35,
                pointpos=0,
                line_width=2,
                marker_size=8,
            )
        )
    ordered_types = [
        _session_type_plot_label(t) for t in _order_session_types(batch_df["session_type"])
    ]
    fig.add_hline(
        y=0.5,
        line=dict(color=COLOR_GRAY, dash="dash", width=2),
    )
    fig.update_layout(
        title=title,
        xaxis_title="Session Type",
        yaxis_title=metric_col.replace("_", " ").title(),
        yaxis=dict(range=[0, 1.05]),
        boxmode="group",
        height=450,
        legend_title="Area",
    )
    fig.update_xaxes(categoryorder="array", categoryarray=ordered_types)
    return fig


def _plot_group_beta_diff(batch_df: pd.DataFrame) -> go.Figure | None:
    """Per-session ACx − OFC beta difference by session_type (dual-area sessions only)."""
    # beta_diff is the same for both area rows of a session; de-duplicate to ACx rows.
    dual_df = (
        batch_df[batch_df["area"] == "ACx"]
        .dropna(subset=["beta_diff"])
        .copy()
    )
    if dual_df.empty:
        return None

    session_types = _order_session_types(dual_df["session_type"])
    palette = ["#2E86AB", "#E84855", "#52B788", "#F4A261", "#9B5DE5"]

    fig = go.Figure()
    for i, stype in enumerate(session_types):
        stype_df = dual_df[dual_df["session_type"] == stype]
        plot_label = _session_type_plot_label(stype)
        fig.add_trace(
            go.Box(
                x=stype_df["session_type"].map(_session_type_plot_label),
                y=stype_df["beta_diff"],
                name=plot_label,
                marker_color=palette[i % len(palette)],
                boxpoints="all",
                jitter=0.35,
                pointpos=0,
                line_width=2,
                marker_size=8,
                showlegend=False,
            )
        )
    fig.add_hline(y=0, line=dict(color=COLOR_GRAY, dash="dash", width=2))
    plot_session_types = [_session_type_plot_label(t) for t in session_types]
    fig.update_layout(
        title="ACx vs OFC Weighting (\u03b2_ACx \u2212 \u03b2_OFC) by Session Type",
        xaxis_title="Session Type",
        yaxis_title="Beta Difference (ACx \u2212 OFC)",
        height=420,
    )
    fig.update_xaxes(categoryorder="array", categoryarray=plot_session_types)
    return fig


# Stage line colors (opaque); pulled from the shared LEARNING_STAGE_COLORS palette.
_STAGE_LINE_COLORS: dict[str, str] = {
    stage: pair[0] for stage, pair in LEARNING_STAGE_COLORS.items()
}
_STAGE_LINE_COLORS.setdefault("Other", "#888888")


def _attach_learning_stage(view_df: pd.DataFrame) -> pd.DataFrame:
    """Add a `learning_stage` column derived from session_type (cached on view_df)."""
    out = view_df.copy()
    out["learning_stage"] = out["session_type"].map(_classify_learning_stage)
    return out


_COHEN_KAPPA_STAGE_ORDER: tuple[str, ...] = ("Novice", "1b Expert", "2b Expert")


def _kappa_groups_by_learning_stage(
    kappa_df: pd.DataFrame,
    metric_col: str,
    *,
    min_n: int = 2,
    stages: tuple[str, ...] = _COHEN_KAPPA_STAGE_ORDER,
) -> dict[str, np.ndarray]:
    """Session-level Cohen's kappa values per training level."""
    groups: dict[str, np.ndarray] = {}
    for stage in stages:
        vals = pd.to_numeric(
            kappa_df.loc[kappa_df["learning_stage"] == stage, metric_col],
            errors="coerce",
        ).dropna()
        if len(vals) >= min_n:
            groups[stage] = vals.to_numpy(dtype=float)
    return groups


def _run_cohen_kappa_learning_stage_tests(
    kappa_df: pd.DataFrame,
    metric_col: str,
    *,
    min_n: int = 2,
    stages: tuple[str, ...] = _COHEN_KAPPA_STAGE_ORDER,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Kruskal-Wallis across training levels + pairwise Mann-Whitney U (Bonferroni)."""
    from itertools import combinations

    from scipy.stats import kruskal, mannwhitneyu

    groups = _kappa_groups_by_learning_stage(
        kappa_df, metric_col, min_n=min_n, stages=stages
    )
    testable = list(groups.keys())
    metric_label = _KAPPA_PLOT_METRICS.get(metric_col, metric_col)
    omnibus_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []

    if len(testable) < 2:
        return pd.DataFrame(), pd.DataFrame()

    if len(testable) >= 3:
        stat, p = kruskal(*(groups[g] for g in testable))
        omnibus_rows.append(
            {
                "metric": metric_label,
                "test": "Kruskal-Wallis",
                "groups": ", ".join(testable),
                "statistic": float(stat),
                "p": float(p),
                "sig": _p_value_to_sig(float(p)),
            }
        )

    pair_stats: list[dict[str, Any]] = []
    for group_a, group_b in combinations(testable, 2):
        vals_a = groups[group_a]
        vals_b = groups[group_b]
        try:
            stat_u, p = mannwhitneyu(vals_a, vals_b, alternative="two-sided")
        except ValueError:
            stat_u, p = np.nan, np.nan
        pair_stats.append(
            {
                "metric": metric_label,
                "group_a": group_a,
                "group_b": group_b,
                "n_a": len(vals_a),
                "n_b": len(vals_b),
                "mean_a": float(np.mean(vals_a)),
                "mean_b": float(np.mean(vals_b)),
                "U": float(stat_u) if np.isfinite(stat_u) else np.nan,
                "p": float(p) if np.isfinite(p) else np.nan,
            }
        )

    n_tests = sum(1 for row in pair_stats if np.isfinite(row["p"]))
    for row in pair_stats:
        p_raw = row["p"]
        p_adj = min(1.0, p_raw * n_tests) if np.isfinite(p_raw) and n_tests > 0 else np.nan
        pairwise_rows.append(
            {
                **row,
                "p_adj": p_adj,
                "sig": _p_value_to_sig(p_adj) if np.isfinite(p_adj) else "",
            }
        )

    return pd.DataFrame(omnibus_rows), pd.DataFrame(pairwise_rows)


def _plot_cohen_kappa_by_stage(
    kappa_stage_df: pd.DataFrame,
    metric_col: str,
    *,
    title: str,
    yaxis_title: str,
) -> go.Figure | None:
    """Box plot of per-session Cohen's kappa grouped by training level."""
    required = {metric_col, "learning_stage"}
    if kappa_stage_df.empty or not required.issubset(kappa_stage_df.columns):
        return None

    work = kappa_stage_df.dropna(subset=[metric_col]).copy()
    if work.empty:
        return None

    stages = sorted(work["learning_stage"].unique(), key=lambda s: _GROUP_STAGE_RANK.get(s, 99))
    accent_color = _KAPPA_PLOT_COLORS.get(metric_col)

    fig = go.Figure()
    for stage in stages:
        stage_df = work[work["learning_stage"] == stage]
        fig.add_trace(
            go.Box(
                x=stage_df["learning_stage"],
                y=stage_df[metric_col],
                name=stage,
                marker_color=accent_color or _STAGE_LINE_COLORS.get(stage, "#888888"),
                boxpoints="all",
                jitter=0.35,
                pointpos=0,
                line_width=2,
                marker_size=8,
                showlegend=False,
            )
        )
    fig.add_hline(y=0, line=dict(color=COLOR_GRAY, dash="dash", width=2))
    fig.update_layout(
        title=title,
        xaxis_title="Training Level",
        yaxis_title=yaxis_title,
        yaxis=dict(range=[-0.2, 1.05]),
        height=440,
    )
    fig.update_xaxes(categoryorder="array", categoryarray=stages)
    return fig


def _render_cohen_kappa_by_stage(
    kappa_stage_df: pd.DataFrame,
    metric_col: str,
    *,
    plot_key: str,
) -> None:
    """Box plot + Kruskal-Wallis / pairwise tests for one Cohen's kappa metric."""
    metric_label = _KAPPA_PLOT_METRICS.get(metric_col, metric_col)
    fig = _plot_cohen_kappa_by_stage(
        kappa_stage_df,
        metric_col,
        title=f"{metric_label} Cohen's Kappa by Training Level",
        yaxis_title=f"Cohen's \u03ba ({metric_label})",
    )
    if fig is None:
        st.info(f"Not enough per-session data for {metric_label} Cohen's kappa.")
        return

    st.markdown(f"**{metric_label} Cohen's kappa by training level**")
    st.caption(
        "Per-session chance-corrected Go/No-Go agreement. "
        "1 = perfect, 0 = chance-level, <0 = systematic disagreement. Each point is one session."
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        config=get_plotly_config(plot_key),
    )

    omnibus_df, pairwise_df = _run_cohen_kappa_learning_stage_tests(kappa_stage_df, metric_col)
    if omnibus_df.empty and pairwise_df.empty:
        st.caption(
            f"{metric_label}: not enough sessions per training level "
            f"(need \u22652 per group among {_COHEN_KAPPA_STAGE_ORDER})."
        )
        return

    if not omnibus_df.empty:
        st.caption(
            f"{metric_label}: Kruskal-Wallis across "
            f"{', '.join(_COHEN_KAPPA_STAGE_ORDER)}; "
            "pairwise Mann-Whitney U with Bonferroni correction."
        )
        st.dataframe(
            omnibus_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "statistic": st.column_config.NumberColumn("H", format="%.3f"),
                "p": st.column_config.NumberColumn(format="%.4g"),
            },
        )

    if not pairwise_df.empty:
        st.dataframe(
            pairwise_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "n_a": st.column_config.NumberColumn("N (A)", format="%d"),
                "n_b": st.column_config.NumberColumn("N (B)", format="%d"),
                "mean_a": st.column_config.NumberColumn("Mean (A)", format="%.3f"),
                "mean_b": st.column_config.NumberColumn("Mean (B)", format="%.3f"),
                "U": st.column_config.NumberColumn(format="%.3f"),
                "p": st.column_config.NumberColumn(format="%.4g"),
                "p_adj": st.column_config.NumberColumn("p (Bonferroni)", format="%.4g"),
            },
        )


def _spearman_dprime_accuracy_label(
    x: pd.Series,
    y: pd.Series,
    *,
    min_n: int = 3,
) -> str:
    """One-line Spearman label for subplot annotation."""
    from scipy.stats import spearmanr

    x_arr = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    y_arr = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    n = int(valid.sum())
    if n < min_n:
        return f"n = {n}"
    r, p = spearmanr(x_arr[valid], y_arr[valid])
    if not np.isfinite(r):
        return f"n = {n}"
    parts = [f"Spearman r = {r:.2f}", f"p = {p:.3g}" if np.isfinite(p) else None, f"n = {n}"]
    label = ", ".join(p for p in parts if p)
    if np.isfinite(p):
        sig = _p_value_to_sig(float(p))
        if sig and sig != "ns":
            label += f" ({sig})"
    return label


_DPRIME_ACCURACY_TARGETS: tuple[tuple[str, str], ...] = (
    ("accuracy", "Behavior target"),
    ("accuracy_gt", "Ground truth target"),
)
# Sessions below this hit rate are drawn as open circles (matches npxl_monitoring).
_LOW_HIT_RATE_THRESHOLD = 0.6


def _low_hit_rate_mask(df: pd.DataFrame) -> pd.Series:
    """True where session hit rate is known and below the threshold."""
    if "session_hit_rate" not in df.columns:
        return pd.Series(False, index=df.index)
    hit_rate = pd.to_numeric(df["session_hit_rate"], errors="coerce")
    return hit_rate.notna() & (hit_rate < _LOW_HIT_RATE_THRESHOLD)


def _dprime_accuracy_hover_template(area: str, *, include_hit_rate: bool) -> str:
    base = (
        f"{area}<br>"
        "Animal: %{customdata[0]}<br>"
        "Date: %{customdata[1]}<br>"
    )
    if include_hit_rate:
        base += "Hit rate: %{customdata[2]:.1%}<br>"
    return base + "d': %{x:.2f}<br>Accuracy: %{y:.3f}<extra></extra>"


def _add_dprime_accuracy_panel(
    fig: go.Figure,
    plot_df: pd.DataFrame,
    *,
    area: str,
    metric_col: str,
    row: int,
    col: int,
    x_lo: float,
    x_hi: float,
    legend_shown: set[str],
    show_regression_legend: bool,
    show_low_hit_legend: bool,
    exclude_low_hit_from_ols: bool,
) -> tuple[set[str], bool, bool]:
    """Scatter + OLS + Spearman annotation for one (target, area) panel."""
    area_df = plot_df[(plot_df["area"] == area) & plot_df[metric_col].notna()].copy()
    if area_df.empty:
        return legend_shown, show_regression_legend, show_low_hit_legend

    has_hit_rate = "session_hit_rate" in area_df.columns
    low_hit = _low_hit_rate_mask(area_df)
    has_low_hit_sessions = bool(low_hit.any())

    hover_cols = ["animal", "date"]
    if has_hit_rate:
        hover_cols.append("session_hit_rate")

    stage_order = ["Novice", "1b Expert", "2b Expert", "Other"]
    for stage in stage_order:
        stage_df = area_df[area_df["learning_stage"] == stage]
        if stage_df.empty:
            continue
        color = _STAGE_LINE_COLORS.get(stage, "#888888")
        show_legend = stage not in legend_shown

        filled_df = stage_df[~_low_hit_rate_mask(stage_df)]
        if not filled_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=filled_df["session_dprime"],
                    y=filled_df[metric_col],
                    mode="markers",
                    name=stage,
                    legendgroup=stage,
                    showlegend=show_legend,
                    marker=dict(color=color, size=10, line=dict(width=1, color="white")),
                    hovertemplate=_dprime_accuracy_hover_template(area, include_hit_rate=has_hit_rate),
                    customdata=filled_df[hover_cols].to_numpy(),
                ),
                row=row,
                col=col,
            )
            legend_shown.add(stage)

        open_df = stage_df[_low_hit_rate_mask(stage_df)]
        if not open_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=open_df["session_dprime"],
                    y=open_df[metric_col],
                    mode="markers",
                    name=stage,
                    legendgroup=stage,
                    showlegend=False,
                    marker=dict(
                        color=color,
                        size=11,
                        symbol="circle-open",
                        line=dict(width=2, color=color),
                    ),
                    hovertemplate=_dprime_accuracy_hover_template(area, include_hit_rate=has_hit_rate),
                    customdata=open_df[hover_cols].to_numpy(),
                ),
                row=row,
                col=col,
            )

    if has_low_hit_sessions and show_low_hit_legend:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(
                    size=11,
                    symbol="circle-open",
                    color="white",
                    line=dict(width=2, color=COLOR_GRAY),
                ),
                name=f"<{_LOW_HIT_RATE_THRESHOLD:.0%} hit rate",
                legendgroup="low_hit",
                showlegend=True,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )
        show_low_hit_legend = False

    ols_df = area_df[~low_hit] if exclude_low_hit_from_ols else area_df
    if len(ols_df) >= 3:
        x_vals = ols_df["session_dprime"].to_numpy(dtype=float)
        y_vals = ols_df[metric_col].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x_vals, y_vals, 1)
        line_x = np.array([x_lo, x_hi])
        line_y = slope * line_x + intercept
        fig.add_trace(
            go.Scatter(
                x=line_x,
                y=line_y,
                mode="lines",
                name="OLS fit",
                legendgroup="regression",
                showlegend=show_regression_legend,
                line=dict(color=COLOR_GRAY, width=2.5),
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )
        show_regression_legend = False

    fig.add_annotation(
        text=_spearman_dprime_accuracy_label(area_df["session_dprime"], area_df[metric_col]),
        row=row,
        col=col,
        xref="x domain",
        yref="y domain",
        x=0.05,
        y=0.95,
        xanchor="left",
        yanchor="top",
        showarrow=False,
        font=dict(size=12),
        bgcolor="rgba(255,255,255,0.85)",
        borderpad=4,
    )

    fig.add_hline(
        y=0.5,
        line=dict(color=COLOR_GRAY, dash="dash", width=2),
        row=row,
        col=col,
    )
    return legend_shown, show_regression_legend, show_low_hit_legend


def _plot_dprime_vs_accuracy(
    view_df: pd.DataFrame,
    *,
    exclude_low_hit_from_ols: bool = True,
) -> go.Figure | None:
    """Scatter of session d' vs decoder accuracy — behavior row, GT row; one column per area.

    Returns None when no session has a valid (d', accuracy) pair for any target.
    """
    if "session_dprime" not in view_df.columns:
        return None

    targets = [
        (col, label)
        for col, label in _DPRIME_ACCURACY_TARGETS
        if col in view_df.columns and not view_df.dropna(subset=["session_dprime", col]).empty
    ]
    if not targets:
        return None

    plot_df = view_df.dropna(subset=["session_dprime"]).copy()
    plot_df = _attach_learning_stage(plot_df)
    areas_present = [a for a in ("ACx", "OFC") if a in plot_df["area"].unique()]
    if not areas_present:
        return None

    x_min = float(plot_df["session_dprime"].min())
    x_max = float(plot_df["session_dprime"].max())
    x_pad = max(0.05, 0.02 * (x_max - x_min) if x_max > x_min else 0.2)
    x_lo, x_hi = x_min - x_pad, x_max + x_pad

    accuracy_cols = [col for col, _ in targets]
    acc = pd.to_numeric(plot_df[accuracy_cols].stack(), errors="coerce").dropna()
    if acc.empty:
        y_lo, y_hi = 0.3, 1.0
    else:
        y_min, y_max = float(acc.min()), float(acc.max())
        y_pad = max(0.02, 0.04 * (y_max - y_min) if y_max > y_min else 0.03)
        y_lo = max(0.3, y_min - y_pad)
        y_hi = min(1.0, y_max + y_pad)
        # Avoid an overly narrow band when points are tightly clustered.
        min_span = 0.2
        if y_hi - y_lo < min_span:
            mid = (y_lo + y_hi) / 2
            y_lo = max(0.3, mid - min_span / 2)
            y_hi = min(1.0, mid + min_span / 2)

    n_rows = len(targets)
    n_cols = len(areas_present)
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=areas_present * n_rows,
        row_titles=[label for _, label in targets],
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.10,
        horizontal_spacing=0.05,
    )

    legend_shown: set[str] = set()
    show_regression_legend = True
    show_low_hit_legend = True
    for row_idx, (metric_col, _) in enumerate(targets, start=1):
        for col_idx, area in enumerate(areas_present, start=1):
            legend_shown, show_regression_legend, show_low_hit_legend = _add_dprime_accuracy_panel(
                fig,
                plot_df,
                area=area,
                metric_col=metric_col,
                row=row_idx,
                col=col_idx,
                x_lo=x_lo,
                x_hi=x_hi,
                legend_shown=legend_shown,
                show_regression_legend=show_regression_legend,
                show_low_hit_legend=show_low_hit_legend,
                exclude_low_hit_from_ols=exclude_low_hit_from_ols,
            )

    for row_idx in range(1, n_rows + 1):
        for col_idx in range(1, n_cols + 1):
            fig.update_xaxes(
                title_text="Session d'" if row_idx == n_rows else None,
                range=[x_lo, x_hi],
                row=row_idx,
                col=col_idx,
            )
            fig.update_yaxes(
                title_text="Decoder accuracy" if col_idx == 1 else None,
                range=[y_lo, y_hi],
                row=row_idx,
                col=col_idx,
            )

    fig.update_layout(
        title="Session d' vs Decoder Accuracy",
        height=320 + 260 * n_rows,
        legend_title="Learning stage",
    )
    return fig


def _run_dprime_accuracy_correlations(
    view_df: pd.DataFrame,
    metric_col: str = "accuracy",
    *,
    min_n: int = 3,
) -> pd.DataFrame:
    """Spearman r between session d' and decoder accuracy, per (area, stage) + pooled.

    Bonferroni correction is applied only to the per-(area, stage) tests; the
    pooled-per-area rows are shown for context with raw p only.
    """
    from scipy.stats import spearmanr

    if "session_dprime" not in view_df.columns:
        return pd.DataFrame()

    plot_df = view_df.dropna(subset=["session_dprime", metric_col]).copy()
    if plot_df.empty:
        return pd.DataFrame()
    plot_df = _attach_learning_stage(plot_df)

    per_stage_rows: list[dict[str, Any]] = []
    pooled_rows: list[dict[str, Any]] = []

    for area in sorted(plot_df["area"].dropna().astype(str).unique()):
        area_df = plot_df[plot_df["area"] == area]

        # Per-stage correlations (only Novice / 1b Expert / 2b Expert).
        for stage in ("Novice", "1b Expert", "2b Expert"):
            stage_df = area_df[area_df["learning_stage"] == stage]
            n = int(len(stage_df))
            if n < min_n:
                per_stage_rows.append(
                    {
                        "area": area,
                        "stage": stage,
                        "n": n,
                        "Spearman r": np.nan,
                        "p": np.nan,
                        "p_adj": np.nan,
                        "sig": "",
                    }
                )
                continue
            r, p = spearmanr(stage_df["session_dprime"], stage_df[metric_col])
            per_stage_rows.append(
                {
                    "area": area,
                    "stage": stage,
                    "n": n,
                    "Spearman r": float(r) if np.isfinite(r) else np.nan,
                    "p": float(p) if np.isfinite(p) else np.nan,
                    "p_adj": np.nan,
                    "sig": "",
                }
            )

        # Pooled (all stages) — reported without Bonferroni since it's a separate question.
        n_pool = int(len(area_df))
        if n_pool >= min_n:
            r, p = spearmanr(area_df["session_dprime"], area_df[metric_col])
            pooled_rows.append(
                {
                    "area": area,
                    "stage": "All (pooled)",
                    "n": n_pool,
                    "Spearman r": float(r) if np.isfinite(r) else np.nan,
                    "p": float(p) if np.isfinite(p) else np.nan,
                    "p_adj": np.nan,
                    "sig": _p_value_to_sig(float(p)) if np.isfinite(p) else "",
                }
            )

    n_tests = sum(1 for row in per_stage_rows if np.isfinite(row["p"]))
    for row in per_stage_rows:
        p_raw = row["p"]
        if np.isfinite(p_raw) and n_tests > 0:
            p_adj = min(1.0, p_raw * n_tests)
            row["p_adj"] = p_adj
            row["sig"] = _p_value_to_sig(p_adj)

    return pd.DataFrame(per_stage_rows + pooled_rows)


# --- Streamlit UI (script runs top-to-bottom on each interaction) ---

def _render_single_session_tab(valid_sessions_df: pd.DataFrame, monitoring_path: str) -> None:
    """Per-session ACx/OFC decoder tab (all st.stop() replaced with return)."""
    with st.expander("Methods — what this tab does", expanded=False):
        st.markdown(
            f"""
**Goal.** For a single recording session, train independent population decoders for ACx and OFC
and compare their trial-by-trial predictions to each other, to the mouse's lick/withhold choice,
and to the stimulus ground truth (Go vs No-Go tone).

**Inputs.**
- Tone-aligned event windows produced by NPXL preprocessing (`load_full_event_windows_data`):
  a `units × time × trials` firing-rate tensor per probe (`imec0` → ACx, `imec1` → OFC).
- Per-trial behavioral table (stimulus, outcome ∈ {{Hit, Miss, FA, CR}}, trial position).
- Optional histology mapping (`load_histology_matched_unit_indices`) restricting units to those
  histologically confirmed in the target area; toggle off to use all Bombcell good+MUA units.

**Unit selection.** Bombcell `UnitType` codes {GOOD_MUA_CODES} (good + MUA). A session is included
only when at least one area has ≥{UNIT_THRESHOLD} good+MUA units.

**Trial labels.**
- *Behavior (mouse choice)*: lick = Go (Hit ∪ FA), withhold = No-Go (Miss ∪ CR).
- *Ground truth (stimulus)*: Go tone = Hit ∪ Miss, No-Go tone = FA ∪ CR. Independent of correctness.
- Trials with unknown / non-task outcomes are dropped before decoding.

**Feature matrix.** For each area, the `units × time × trials` tensor is sliced to the user-defined
decode window and aggregated across time (mean or sum) to give one `trials × units` matrix.
Features are z-scored inside each CV fold (`StandardScaler` in the sklearn pipeline) — no leakage.

**Classifier & evaluation.** One of {{RBF SVM (default), Logistic Regression, Linear SVM}} trained
per area with `StratifiedKFold(n_splits={TRAIN_TEST_FOLDS})` — a fixed 80/20 split with stratified
class balance. Out-of-fold (OOF) probabilities and predictions are collected for every trial, so
every reported number is held-out. Areas are **never pooled** and sessions are **never pooled**.

**Reported quantities.**
1. Per-area OOF metrics: accuracy, ROC-AUC, log-loss, precision/recall.
2. Agreement-space crosstabs: ACx↔OFC, ACx↔Mouse, OFC↔Mouse, and (ACx,OFC) joint state ↔ ground truth.
3. **Weighting analysis** (only when decode target = Ground truth, dual-area available): logistic
   model `Mouse ~ GT + ACx_pred + OFC_pred` gives β / odds-ratio / 95% CI per regressor plus an
   ACx−OFC contrast, with a nested-model comparison (AIC / BIC / McFadden R²) that tests whether
   ACx or OFC carries unique predictive value for the mouse's choice beyond GT.
4. Joint and marginal conditional probabilities `P(Mouse=Go | ACx, OFC)`, `P(GT=Go | ACx, OFC)`, etc.
5. Per-trial 3-D / pairwise decision scatters (Go=+1, No-Go=−1; filled=correct, open=error;
   green/red = full agreement, gray = disagreement).
6. Disagreement / error-trial breakdown and a stimulus-conditioned psychometric-style summary.
7. Downloadable OOF prediction table.
            """
        )

    st.subheader("Session Selection")
    st.caption(
        f"Sessions with >={UNIT_THRESHOLD} good+MUA units in at least one area (ACx or OFC). "
        "Check one session to decode (analysis runs on a single session at a time)."
    )

    use_histology = not st.toggle(
        "Use all good+MUA units (skip histology filter)",
        value=True,
        help=(
            "Off (default): only units confirmed to be in the target area by histology mapping are used. "
            "On: all good+MUA units from the probe are included regardless of histology — "
            "gives more units but loses the area-label guarantee."
        ),
        key="npxl_agreement_use_all_units",
    )

    selection_df = _prepare_session_selection_table(valid_sessions_df)
    edited_selection_df = st.data_editor(
        selection_df,
        height=min(360, 48 + 35 * max(len(selection_df), 1)),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Checkbox": st.column_config.CheckboxColumn(
                "Include",
                help="Select the session to analyze.",
                default=False,
            ),
            "acx_total_units": st.column_config.NumberColumn("ACx units", disabled=True),
            "ofc_total_units": st.column_config.NumberColumn("OFC units", disabled=True),
            "session_dprime": st.column_config.NumberColumn("Session d'", format="%.3f", disabled=True),
            "session_hit_rate": st.column_config.NumberColumn("Hit rate", format="%.2f", disabled=True),
            "acx_histology_matched": st.column_config.NumberColumn("ACx histology n", format="%d", disabled=True),
            "ofc_histology_matched": st.column_config.NumberColumn("OFC histology n", format="%d", disabled=True),
            "acx_histology_match_pct": st.column_config.NumberColumn("ACx histology %", format="%.1f", disabled=True),
            "ofc_histology_match_pct": st.column_config.NumberColumn("OFC histology %", format="%.1f", disabled=True),
            "session_label": st.column_config.TextColumn("Session", disabled=True),
            "Animal": st.column_config.TextColumn("Animal", disabled=True),
            "Date": st.column_config.TextColumn("Date", disabled=True),
            "Session Type": st.column_config.TextColumn("Session Type", disabled=True),
        },
        key="npxl_agreement_session_selector",
    )

    checked_rows = edited_selection_df[edited_selection_df["Checkbox"] == True]
    if checked_rows.empty:
        st.info("Check one session above to run ACx/OFC choice decoding.")
        return
    if len(checked_rows) > 1:
        st.warning("Multiple sessions are checked; using the first selected session only.")

    selected_table_idx = int(checked_rows.index[0])
    selected_row = valid_sessions_df.loc[selected_table_idx]
    session_dir = str(selected_row["current_dir"]).strip()
    # Cache key ties decoder outputs to one session + path (no cross-session pooling).
    session_key = f"{int(selected_row['session_row_index'])}:{session_dir}"



    try:
        with st.spinner("Loading selected session event windows..."):
            session_data = load_session_event_data(session_dir, use_histology=use_histology)
    except (FileNotFoundError, ValueError, OSError) as exc:
        st.error(str(exc))
        return

    available_areas = session_data["areas"]
    if session_data["missing_area_errors"]:
        for area in ("ACx", "OFC"):
            if area in session_data["missing_area_errors"] and area not in available_areas:
                st.warning(f"{area} unavailable for this run: {session_data['missing_area_errors'][area]}")
    if not use_histology:
        st.info("Histology filter is off — using all good+MUA units from each probe. Area labels are assumed, not histology-confirmed.")

    with st.expander("Histology-matched units" if use_histology else "Units (no histology filter)", expanded=False):
        for area in available_areas:
            unit_table = session_data[area]["unit_table"]
            display_cols = [
                c
                for c in [
                    "matrix_row",
                    "unit_idx",
                    "label_unitID",
                    "unit_type",
                    "cortex_group",
                    "histology_region",
                    "mapping_join_status",
                ]
                if c in unit_table.columns
            ]
            st.markdown(f"**{area}** ({len(unit_table)} units)")
            st.dataframe(unit_table[display_cols].head(20), use_container_width=True, hide_index=True)

    time_axis = session_data["time_axis"]
    min_time = float(np.min(time_axis))
    max_time = float(np.max(time_axis))
    # Default: 250 ms post-onset window starting at tone onset (0 s).
    default_start = max(0.0, min_time)
    default_stop = min(0.25, max_time)
    if default_stop <= default_start:
        default_stop = min(default_start + 0.25, max_time)

    st.subheader("Decoder Settings")
    decode_col1, decode_col2, decode_col3 = st.columns(3)
    with decode_col1:
        decode_window = st.slider(
            "Decode time window (s)",
            min_value=min_time,
            max_value=max_time,
            value=(float(default_start), float(default_stop)),
            step=0.05,
            key="npxl_agreement_decode_window_v2",
        )
        aggregation = st.selectbox(
            "Window aggregation",
            ["Mean", "Sum"],
            key="npxl_agreement_aggregation",
        )
    with decode_col2:
        classifier_type = st.selectbox(
            "Classifier",
            ["RBF SVM", "Logistic Regression", "Linear SVM"],
            index=0,
            key="npxl_agreement_classifier_v2",
        )
        st.caption("Train/Test split: fixed 80/20")
    with decode_col3:
        min_trials_per_class = st.number_input(
            "Minimum trials per class",
            min_value=2,
            max_value=50,
            value=5,
            step=1,
            key="npxl_agreement_min_trials",
        )
        random_state = st.number_input(
            "Random seed",
            min_value=0,
            max_value=9999,
            value=42,
            step=1,
            key="npxl_agreement_random_seed",
        )
        decode_target = st.radio(
            "Decode target",
            ["Mouse behavior", "Ground truth"],
            index=1,
            help=(
                "Mouse behavior: trial-by-trial lick vs withhold (Hit+FA vs Miss+CR). "
                "Ground truth: tone category (Go vs No-Go), regardless of the mouse's response."
            ),
            key="npxl_agreement_decode_target",
        )

    primary_area = "ACx" if "ACx" in available_areas else available_areas[0]
    labels_df = _build_trial_labels(session_data[primary_area]["trials_df"])
    valid_trial_count = len(labels_df)
    excluded_trials = session_data[primary_area]["trials_df"].shape[0] - valid_trial_count
    if valid_trial_count == 0:
        st.warning("No valid Hit/Miss/False Alarm/CR trials were found for this session.")
        return

    trial_positions = labels_df["source_trial_position"].to_numpy(dtype=int)
    # Select decoding target: mouse choice (default) or stimulus ground truth, per UI radio.
    if decode_target == "Ground truth":
        target_column = "ground_truth_go"
        target_label = "Ground truth (Go tone)"
    else:
        target_column = "behavior_go"
        target_label = "Mouse choice (lick=Go)"
    y_target = labels_df[target_column].to_numpy(dtype=int)
    # y_behavior keeps its name for downstream PCA labelling (it just needs the trial-level class vector).
    y_behavior = y_target

    try:
        # Independent population decoders per available area.
        area_features: dict[str, np.ndarray] = {}
        area_results: dict[str, dict[str, Any]] = {}
        for area in available_areas:
            area_features[area] = _feature_matrix(
                session_data[area]["matrix"][:, :, trial_positions],
                time_axis,
                decode_window,
                aggregation,
            )
            area_results[area] = train_oof_decoder(
                area,
                session_key + f"|target={target_column}",
                area_features[area],
                y_target,
                classifier_type,
                TRAIN_TEST_FOLDS,
                int(random_state),
                int(min_trials_per_class),
            )
    except (ValueError, RuntimeError) as exc:
        st.error(str(exc))
        return

    st.subheader("Selected Session")
    meta_cols = st.columns(5)
    meta_cols[0].metric("Animal", str(selected_row.get("Animal", "")))
    meta_cols[1].metric("Date", str(selected_row.get("Date", "")))
    meta_cols[2].metric("ACx units", f"{session_data['ACx']['unit_count']:,}" if "ACx" in available_areas else "N/A")
    meta_cols[3].metric("OFC units", f"{session_data['OFC']['unit_count']:,}" if "OFC" in available_areas else "N/A")
    meta_cols[4].metric("Valid trials", f"{valid_trial_count:,}")
    if excluded_trials > 0:
        st.info(f"Excluded {excluded_trials} trials with unknown/non-task outcomes.")

    st.subheader(f"Cross-Validated Decoding — target: {target_label}")
    metric_cols = st.columns(max(1, len(available_areas)))
    for col, area in zip(metric_cols, available_areas):
        with col:
            _render_metrics(f"{area} Decoder", area_results[area], session_data[area]["unit_count"], valid_trial_count)

    results_df = labels_df.copy()
    results_df["behavior_label"] = _label_binary(results_df["behavior_go"])
    results_df["ground_truth_label"] = _label_binary(results_df["ground_truth_go"])
    for area in available_areas:
        area_l = area.lower()
        results_df[f"{area_l}_prob_go"] = area_results[area]["probability"]
        results_df[f"{area_l}_pred"] = area_results[area]["prediction"]
        results_df[f"{area_l}_label"] = _label_binary(results_df[f"{area_l}_pred"])

    # Congruence matrices compare OOF predictions to each other, behavior, and stimulus GT.
    st.subheader("Agreement Space")
    if len(available_areas) == 2:
        results_df["agreement_state"] = _agreement_state(
            results_df["acx_pred"].to_numpy(),
            results_df["ofc_pred"].to_numpy(),
        )
        heatmap_col1, heatmap_col2 = st.columns(2)
        with heatmap_col1:
            st.plotly_chart(
                _plot_crosstab_heatmap(results_df["acx_label"].tolist(), results_df["ofc_label"].tolist(), "ACx Prediction vs OFC Prediction", "ACx", "OFC"),
                use_container_width=True,
                config=get_plotly_config("acx_vs_ofc"),
            )
        with heatmap_col2:
            st.plotly_chart(
                _plot_crosstab_heatmap(results_df["acx_label"].tolist(), results_df["behavior_label"].tolist(), "ACx Prediction vs Mouse Behavior", "ACx", "Mouse"),
                use_container_width=True,
                config=get_plotly_config("acx_vs_mouse"),
            )

        heatmap_col3, heatmap_col4 = st.columns(2)
        with heatmap_col3:
            st.plotly_chart(
                _plot_crosstab_heatmap(results_df["ofc_label"].tolist(), results_df["behavior_label"].tolist(), "OFC Prediction vs Mouse Behavior", "OFC", "Mouse"),
                use_container_width=True,
                config=get_plotly_config("ofc_vs_mouse"),
            )
        with heatmap_col4:
            st.plotly_chart(
                _plot_state_vs_ground_truth(results_df["agreement_state"], results_df["ground_truth_label"].tolist()),
                use_container_width=True,
                config=get_plotly_config("models_vs_gt"),
            )
    else:
        area = available_areas[0]
        area_l = area.lower()
        st.plotly_chart(
            _plot_crosstab_heatmap(
                results_df[f"{area_l}_label"].tolist(),
                results_df["behavior_label"].tolist(),
                f"{area} Prediction vs Mouse Behavior",
                area,
                "Mouse",
            ),
            use_container_width=True,
            config=get_plotly_config(f"{area_l}_vs_mouse"),
        )
        st.plotly_chart(
            _plot_crosstab_heatmap(
                results_df[f"{area_l}_label"].tolist(),
                results_df["ground_truth_label"].tolist(),
                f"{area} Prediction vs Ground Truth",
                area,
                "Ground Truth",
            ),
            use_container_width=True,
            config=get_plotly_config(f"{area_l}_vs_gt"),
        )

    if len(available_areas) == 2:
        st.subheader("Chance-Corrected Agreement")
        st.caption(
            "Kappa corrects raw Go/No-Go agreement for the chance agreement expected from each "
            "source's marginal Go/No-Go rate. 1 = perfect, 0 = chance-level, <0 = systematic disagreement."
        )
        kappa_summary = _build_kappa_agreement_summary(results_df)
        if not kappa_summary.empty:
            summary_cols = [
                col
                for col in [
                    "n_trials",
                    "cohen_mouse_acx",
                    "cohen_mouse_ofc",
                    "cohen_acx_ofc",
                    "cohen_gt_acx",
                    "cohen_gt_ofc",
                    "cohen_gt_mouse",
                    "fleiss_mouse_acx_ofc",
                    "fleiss_mouse_acx_ofc_gt",
                    "var_agree_acx_ofc",
                ]
                if col in kappa_summary.columns
            ]
            st.dataframe(
                kappa_summary[summary_cols],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "n_trials": st.column_config.NumberColumn("Trials", format="%d"),
                    "var_agree_acx_ofc": st.column_config.NumberColumn(
                        "ACx/OFC agree var", format="%.4f"
                    ),
                    **{
                        col: st.column_config.NumberColumn(col.replace("_", " "), format="%.3f")
                        for col in summary_cols
                        if col not in ("n_trials", "var_agree_acx_ofc")
                    },
                },
            )

        kappa_by_stimulus = _build_kappa_by_stimulus(results_df)
        if kappa_by_stimulus.empty:
            st.info("Not enough per-stimulus data to compute kappa for this session.")
        else:
            with st.expander("Per-stimulus kappa", expanded=False):
                stim_cols = [
                    col
                    for col in [
                        "stimulus",
                        "n_trials",
                        "cohen_mouse_acx",
                        "cohen_mouse_ofc",
                        "cohen_acx_ofc",
                        "cohen_gt_acx",
                        "cohen_gt_ofc",
                        "cohen_gt_mouse",
                        "fleiss_mouse_acx_ofc",
                        "fleiss_mouse_acx_ofc_gt",
                        "p_agree_mouse_acx",
                        "p_agree_mouse_ofc",
                        "p_agree_acx_ofc",
                    ]
                    if col in kappa_by_stimulus.columns
                ]
                st.dataframe(
                    kappa_by_stimulus[stim_cols],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "stimulus": st.column_config.NumberColumn("Stimulus (kHz)", format="%.3f"),
                        "n_trials": st.column_config.NumberColumn("Trials", format="%d"),
                        **{
                            col: st.column_config.NumberColumn(col.replace("_", " "), format="%.3f")
                            for col in stim_cols
                            if col not in {"stimulus", "n_trials"}
                        },
                    },
                )

    st.subheader("Agreement Probabilities And Area Weighting Stats")
    if decode_target != "Ground truth":
        st.warning(
            "Weighting stats are computed only when decoders are trained on Ground truth. "
            "Switch Decode target to Ground truth to enable this section."
        )
    else:
        if len(available_areas) == 2:
            joint_probs_df = _build_joint_conditional_probabilities(results_df)
            if joint_probs_df.empty:
                st.info("Not enough dual-area prediction data to compute joint conditional probabilities.")
            else:
                st.markdown("**Joint conditional probabilities by ACx/OFC prediction combination**")
                st.dataframe(
                    joint_probs_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "p_mouse_go": st.column_config.NumberColumn("P(Mouse=Go|cond)", format="%.3f"),
                        "p_mouse_nogo": st.column_config.NumberColumn("P(Mouse=No-Go|cond)", format="%.3f"),
                        "p_gt_go": st.column_config.NumberColumn("P(GT=Go|cond)", format="%.3f"),
                        "p_gt_nogo": st.column_config.NumberColumn("P(GT=No-Go|cond)", format="%.3f"),
                    },
                )

                # Validation check: row probabilities should each sum to 1.
                mouse_sum_ok = np.allclose(
                    (joint_probs_df["p_mouse_go"] + joint_probs_df["p_mouse_nogo"]).to_numpy(),
                    1.0,
                    atol=1e-8,
                )
                gt_sum_ok = np.allclose(
                    (joint_probs_df["p_gt_go"] + joint_probs_df["p_gt_nogo"]).to_numpy(),
                    1.0,
                    atol=1e-8,
                )
                if not (mouse_sum_ok and gt_sum_ok):
                    st.warning("Probability-row sanity check failed for one or more rows.")

            marginals_df = _build_marginal_probabilities(results_df, available_areas)
            if not marginals_df.empty:
                st.markdown("**Marginal conditional probabilities**")
                st.dataframe(
                    marginals_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "p_go": st.column_config.NumberColumn("P(Go|cond)", format="%.3f"),
                        "p_nogo": st.column_config.NumberColumn("P(No-Go|cond)", format="%.3f"),
                    },
                )

            coef_df, diff_df = _weighting_summary_dual_area(results_df, int(random_state))
            if coef_df.empty:
                st.info("Weighting model could not be fit for this session.")
            else:
                st.markdown("**Logistic weighting model (Mouse ~ GT + ACx + OFC)**")
                st.dataframe(
                    coef_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "beta": st.column_config.NumberColumn("beta", format="%.4f"),
                        "odds_ratio_exp_beta": st.column_config.NumberColumn("OR=exp(beta)", format="%.4f"),
                        "beta_ci_low": st.column_config.NumberColumn("beta_ci_low", format="%.4f"),
                        "beta_ci_high": st.column_config.NumberColumn("beta_ci_high", format="%.4f"),
                    },
                )
                st.markdown("**ACx vs OFC weighting contrast**")
                st.dataframe(
                    diff_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "value": st.column_config.NumberColumn("value", format="%.4f"),
                        "ci_low": st.column_config.NumberColumn("ci_low", format="%.4f"),
                        "ci_high": st.column_config.NumberColumn("ci_high", format="%.4f"),
                    },
                )

            nested_df = _nested_model_comparison(results_df, int(random_state))
            if not nested_df.empty:
                st.markdown("**Nested model comparison**")
                st.dataframe(
                    nested_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "aic": st.column_config.NumberColumn("AIC", format="%.2f"),
                        "bic": st.column_config.NumberColumn("BIC", format="%.2f"),
                        "ll_model": st.column_config.NumberColumn("logLik", format="%.2f"),
                        "mcfadden_r2": st.column_config.NumberColumn("McFadden R2", format="%.4f"),
                        "delta_aic_vs_best": st.column_config.NumberColumn("delta AIC", format="%.2f"),
                    },
                )
        else:
            area = available_areas[0]
            st.info(
                f"Only {area} is available for this session. Dual-area weighting comparison is not available; "
                "showing single-area fallback stats."
            )
            single_probs_df = _build_single_area_conditional_probabilities(results_df, area)
            if not single_probs_df.empty:
                st.markdown(f"**Single-area conditional probabilities ({area})**")
                st.dataframe(
                    single_probs_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "p_mouse_go": st.column_config.NumberColumn("P(Mouse=Go|cond)", format="%.3f"),
                        "p_mouse_nogo": st.column_config.NumberColumn("P(Mouse=No-Go|cond)", format="%.3f"),
                        "p_gt_go": st.column_config.NumberColumn("P(GT=Go|cond)", format="%.3f"),
                        "p_gt_nogo": st.column_config.NumberColumn("P(GT=No-Go|cond)", format="%.3f"),
                    },
                )
            single_model_df = _weighting_summary_single_area(results_df, area, int(random_state))
            if not single_model_df.empty:
                st.markdown(f"**Single-area weighting model (Mouse ~ GT + {area})**")
                st.dataframe(
                    single_model_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "beta": st.column_config.NumberColumn("beta", format="%.4f"),
                        "odds_ratio_exp_beta": st.column_config.NumberColumn("OR=exp(beta)", format="%.4f"),
                        "beta_ci_low": st.column_config.NumberColumn("beta_ci_low", format="%.4f"),
                        "beta_ci_high": st.column_config.NumberColumn("beta_ci_high", format="%.4f"),
                    },
                )

    st.subheader("Per-Trial Decision Space")
    decision_jitter = st.slider(
        "Scatter jitter",
        min_value=0.0,
        max_value=0.5,
        value=0.3,
        step=0.01,
        help="Spread overlapping trial points at the same lattice corner.",
        key="npxl_agreement_decision_jitter",
    )
    _decision_caption = (
        "Each dot is one trial (Go = +1, No-Go = -1). "
        "Filled = Hit/CR, open = Miss/FA. "
        "Color: green = all sources agree Go, red = all agree No-Go, gray = disagreement."
    )
    if len(available_areas) == 2:
        decisions_3d = _plot_decisions_scatter_3d(results_df, int(random_state), float(decision_jitter))
        pair_figs = _plot_decisions_scatter_dual_pairs(results_df, int(random_state), float(decision_jitter))
        if decisions_3d is None and pair_figs is None:
            st.info("Decision scatter unavailable for this session/area selection.")
        else:
            st.caption(_decision_caption)
            if decisions_3d is not None:
                st.plotly_chart(
                    decisions_3d,
                    use_container_width=True,
                    config=get_plotly_config("decision_scatter_3d"),
                )
            if pair_figs is not None:
                pair_col1, pair_col2, pair_col3 = st.columns(3)
                pair_specs = [
                    (pair_col1, "mouse_ofc", "decision_scatter_mouse_ofc"),
                    (pair_col2, "mouse_acx", "decision_scatter_mouse_acx"),
                    (pair_col3, "ofc_acx", "decision_scatter_ofc_acx"),
                ]
                for col, key, plot_cfg in pair_specs:
                    with col:
                        st.plotly_chart(
                            pair_figs[key],
                            use_container_width=True,
                            config=get_plotly_config(plot_cfg),
                        )
    else:
        decisions_fig = _plot_decisions_scatter(
            results_df,
            available_areas,
            int(random_state),
            float(decision_jitter),
        )
        if decisions_fig is None:
            st.info("Decision scatter unavailable for this session/area selection.")
        else:
            st.caption(_decision_caption)
            st.plotly_chart(
                decisions_fig,
                use_container_width=True,
                config=get_plotly_config("decision_scatter"),
            )

    st.subheader("Disagreement And Error Trials")
    if len(available_areas) == 2:
        error_fig = _plot_error_breakdown(results_df)
        if error_fig is None:
            st.info("No behavioral error trials found in this session.")
        else:
            st.plotly_chart(error_fig, use_container_width=True, config=get_plotly_config("error_trial_agreement"))
    else:
        error_fig = _plot_psychometric_summary(results_df, error_trials_only=True)
        if error_fig is None:
            st.info(
                "No behavioral error trials with enough repeated stimulus bins for a psychometric-style plot."
            )
        else:
            st.plotly_chart(
                error_fig,
                use_container_width=True,
                config=get_plotly_config("error_trial_psychometric"),
            )

    psycho_fig = _plot_psychometric_summary(results_df)
    if psycho_fig is None:
        st.info("Not enough repeated stimulus values for a psychometric-style summary.")
    else:
        st.plotly_chart(psycho_fig, use_container_width=True, config=get_plotly_config("agreement_by_stimulus"))

    with st.expander("Out-of-fold trial predictions", expanded=False):
        display_cols = [
            col
            for col in [
                "stimulus",
                "outcome",
                "behavior_label",
                "ground_truth_label",
                "acx_prob_go",
                "acx_label",
                "ofc_prob_go",
                "ofc_label",
                "agreement_state",
            ]
            if col in results_df.columns
        ]
        st.dataframe(results_df[display_cols], use_container_width=True, hide_index=True)
        st.download_button(
            "Download predictions CSV",
            data=results_df[display_cols].to_csv(index=False).encode("utf-8"),
            file_name=f"npxl_agreement_predictions_{int(selected_row['session_row_index'])}.csv",
            mime="text/csv",
        )


def _render_group_analysis_tab(valid_sessions_df: pd.DataFrame, monitoring_path: str) -> None:
    """Batch-decode all valid sessions and report accuracy/AUC by area × session type."""
    with st.expander("Methods — what this tab does", expanded=False):
        st.markdown(
            f"""
**Goal.** Scale the single-session decoder to the full cohort and ask how decoding performance
and ACx/OFC ↔ behavior agreement vary across sessions, learning stages, and session types.

**Inputs.**
- The same per-session event-window tensors used by the *Single Session* tab — loaded once per
  session inside the batch loop, never concatenated across sessions.
- Pre-computed cluster results from `run_npxl_group_decoder.sbatch` saved under
  `{_group_decoder_results_dir()}` (two CSV pairs: `…_with_histology.csv` and `…_no_histology.csv`).
- Session metadata (animal, date, session type, d′, hit rate) merged from the NPXL monitoring CSV
  via `attach_session_dprime` / `merge_behavioral_file_from_monitoring`.

**Pipeline (identical to the Single Session tab, run per session).**
1. Load tone-aligned tensors for ACx and OFC; apply the same Bombcell good+MUA filter and (optional)
   histology gating used in tab 1. Sessions with <{UNIT_THRESHOLD} units in an area are skipped for
   that area only.
2. Build the same `trials × units` feature matrix from the chosen decode window + aggregation.
3. Train independent ACx and OFC classifiers with `StratifiedKFold(n_splits={TRAIN_TEST_FOLDS})`
   (80/20). Areas are **never pooled** and sessions are **never pooled** — every session contributes
   its own held-out OOF predictions.
4. Repeat the whole pipeline twice per session: once with target = *Mouse behavior* (lick/withhold)
   and once with target = *Ground truth* (Go/No-Go tone). Both result sets are stored side-by-side.

**Reported quantities.**
1. *Summary by Session Type × Area*: per-area accuracy and ROC-AUC (behavior + ground truth),
   mean ± SEM, plus trial / unit counts and optional session d' / hit rate.
2. *Per-session scatter*: each dot = one session, X = behavior decoder, Y = ground-truth decoder,
   colored by area and shaped by session type / learning stage. Lets you see when ACx/OFC are
   choice-locked vs stimulus-locked.
3. *Joint conditional probabilities*: `P(Mouse | ACx, OFC)` and `P(GT | ACx, OFC)` pooled across
   sessions, broken down by the four (ACx, OFC) prediction combinations.
4. *Model agreement landscape*: distribution of (Mouse=ACx=OFC=GT) joint states across the cohort
   and how often each model disagrees with the mouse.
5. *Psychometric-style summaries*: choice/agreement curves vs stimulus, separately for model-agree
   and model-disagree trials, to ask whether disagreement trials are concentrated near threshold.
6. *Failure log*: sessions that could not be decoded (too few trials, missing histology, etc.)
   are listed with the reason so coverage is auditable.

**How to use this tab.**
- **Reload batch results**: re-reads the CSVs written by the SLURM batch job (no decoding done in
  the browser).
- **Run Group Analysis (interactive)**: re-decodes every valid session in-process with the current
  UI settings. Slow — use only for parameter sweeps not covered by the batch.
- **Histology toggle**: switches between the histology-gated and the all-good+MUA result file.
- **Session-type multiselect**: filters every plot / table downstream.
            """
        )

    st.subheader("Group-Level Decoder Analysis")
    st.caption(
        f"Batch-run ACx/OFC decoders on all {len(valid_sessions_df)} valid sessions "
        "and aggregate accuracy / ROC-AUC by area and session type. "
        "Decodes both behavior (lick) and ground truth (tone) targets in a single pass per session. "
        "Pre-computed cluster results are loaded from "
        f"`{_group_decoder_results_dir()}` (see `run_npxl_group_decoder.sbatch`)."
    )

    g_col1, g_col2, g_col3 = st.columns(3)
    with g_col1:
        g_decode_window = st.slider(
            "Decode time window (s)",
            min_value=-1.0,
            max_value=2.0,
            value=(0.0, 0.25),
            step=0.05,
            key="npxl_group_decode_window",
        )
        g_aggregation = st.selectbox(
            "Window aggregation",
            ["Mean", "Sum"],
            key="npxl_group_aggregation",
        )
    with g_col2:
        g_classifier = st.selectbox(
            "Classifier",
            ["RBF SVM", "Logistic Regression", "Linear SVM"],
            index=0,
            key="npxl_group_classifier",
        )
        g_min_trials = st.number_input(
            "Minimum trials per class",
            min_value=2,
            max_value=50,
            value=5,
            step=1,
            key="npxl_group_min_trials",
        )
    with g_col3:
        g_random_state = st.number_input(
            "Random seed",
            min_value=0,
            max_value=9999,
            value=42,
            step=1,
            key="npxl_group_random_seed",
        )
        g_use_histology = not st.toggle(
            "Use all good+MUA units (skip histology filter)",
            value=True,
            key="npxl_group_use_all_units",
            help=(
                "Off: `npxl_group_decoder_results_with_histology_histology.csv`. "
                "On: `npxl_group_decoder_results_no_histology.csv`."
            ),
        )

    histology_label = "histology-matched units" if g_use_histology else "all good+MUA (no histology filter)"
    expected_csv = _group_decoder_results_path(g_use_histology)

    load_col, run_col = st.columns([1, 1])
    with load_col:
        reload_batch = st.button("Reload batch results", key="npxl_group_reload_batch")
    with run_col:
        run_interactive = st.button("Run Group Analysis (interactive)", key="npxl_group_run")

    if reload_batch:
        _read_group_decoder_csv.clear()

    # Reload from disk unless showing a fresh interactive run (toggle/reload switches back to batch CSVs).
    histology_changed = st.session_state.get("_group_loaded_histology") != g_use_histology
    batch_source_state = st.session_state.get("_group_batch_source")
    load_from_disk = (
        batch_source_state != "interactive"
        or reload_batch
        or histology_changed
    )
    if load_from_disk:
        (
            loaded_df,
            loaded_failures,
            loaded_path,
            loaded_joint,
            loaded_model_agreement,
            loaded_psych_agree,
            loaded_psych_disagree,
            loaded_kappa,
            loaded_kappa_stimulus,
        ) = _load_group_decoder_batch_results(g_use_histology)
        if loaded_path is not None:
            st.session_state["_group_batch_df"] = loaded_df
            st.session_state["_group_batch_failures"] = loaded_failures
            st.session_state["_group_joint_probs_df"] = loaded_joint
            st.session_state["_group_model_agreement_df"] = loaded_model_agreement
            st.session_state["_group_psychometric_model_agree_df"] = loaded_psych_agree
            st.session_state["_group_psychometric_model_disagree_df"] = loaded_psych_disagree
            st.session_state["_group_kappa_agreement_df"] = loaded_kappa
            st.session_state["_group_kappa_by_stimulus_df"] = loaded_kappa_stimulus
            st.session_state["_group_batch_source"] = "batch"
            st.session_state["_group_batch_source_path"] = loaded_path
            st.session_state["_group_loaded_histology"] = g_use_histology
            st.session_state.pop("_group_settings_run", None)
        elif histology_changed or reload_batch:
            st.session_state["_group_batch_df"] = pd.DataFrame()
            st.session_state["_group_batch_failures"] = []
            st.session_state["_group_joint_probs_df"] = pd.DataFrame()
            st.session_state["_group_model_agreement_df"] = pd.DataFrame()
            st.session_state["_group_psychometric_model_agree_df"] = pd.DataFrame()
            st.session_state["_group_psychometric_model_disagree_df"] = pd.DataFrame()
            st.session_state["_group_kappa_agreement_df"] = pd.DataFrame()
            st.session_state["_group_kappa_by_stimulus_df"] = pd.DataFrame()
            st.session_state["_group_batch_source"] = "batch"
            st.session_state.pop("_group_batch_source_path", None)
            st.session_state["_group_loaded_histology"] = g_use_histology

    current_settings = (
        tuple(g_decode_window),
        g_aggregation,
        g_classifier,
        int(g_random_state),
        int(g_min_trials),
        g_use_histology,
    )

    if run_interactive:
        with st.spinner(f"Decoding {len(valid_sessions_df)} sessions — this may take a while..."):
            (
                batch_df,
                failures,
                joint_df,
                model_agreement_df,
                psych_agree_df,
                psych_disagree_df,
                kappa_df,
                kappa_stimulus_df,
            ) = _batch_decode_all_sessions(
                monitoring_path,
                *current_settings,
            )
        st.session_state["_group_batch_df"] = batch_df
        st.session_state["_group_batch_failures"] = failures
        st.session_state["_group_joint_probs_df"] = joint_df
        st.session_state["_group_model_agreement_df"] = model_agreement_df
        st.session_state["_group_psychometric_model_agree_df"] = psych_agree_df
        st.session_state["_group_psychometric_model_disagree_df"] = psych_disagree_df
        st.session_state["_group_kappa_agreement_df"] = kappa_df
        st.session_state["_group_kappa_by_stimulus_df"] = kappa_stimulus_df
        st.session_state["_group_settings_run"] = current_settings
        st.session_state["_group_batch_source"] = "interactive"
        st.session_state["_group_loaded_histology"] = g_use_histology
        st.session_state.pop("_group_batch_source_path", None)

    batch_df: pd.DataFrame = st.session_state.get("_group_batch_df", pd.DataFrame())
    failures: list[dict[str, str]] = st.session_state.get("_group_batch_failures", [])
    settings_run = st.session_state.get("_group_settings_run")
    batch_source = st.session_state.get("_group_batch_source", "batch")
    batch_source_path = st.session_state.get("_group_batch_source_path")

    if batch_source == "batch" and batch_source_path:
        st.info(f"Showing batch results ({histology_label}) from `{os.path.basename(batch_source_path)}`.")
    elif batch_source == "interactive":
        st.info(f"Showing interactive run ({histology_label}).")

    if settings_run is not None and settings_run != current_settings:
        st.warning(
            "Decoder settings have changed since the last interactive run. "
            "Click **Run Group Analysis (interactive)** to update, or toggle histology to reload batch CSVs."
        )

    if batch_df.empty and not failures:
        st.warning(
            f"No batch results found for **{histology_label}**. "
            f"Expected file: `{expected_csv}`. "
            "Submit `run_npxl_group_decoder.sbatch` once with and once without `--no_histology`, "
            "or click **Run Group Analysis (interactive)**."
        )
        return

    if not batch_df.empty:
        all_session_types = _order_session_types(batch_df["session_type"])
        selected_session_types = st.multiselect(
            "Session types to show",
            options=all_session_types,
            default=all_session_types,
            key="npxl_group_session_types_filter",
        )
        if not selected_session_types:
            st.info("Select one or more session types above to display results.")
            return

        view_df = batch_df[batch_df["session_type"].astype(str).isin(selected_session_types)].copy()
        n_sessions = view_df[["animal", "date", "session_type"]].drop_duplicates().shape[0]
        st.success(
            f"Showing {n_sessions} sessions ({len(selected_session_types)}/{len(all_session_types)} session types) — "
            f"{view_df[view_df['area'] == 'ACx'].shape[0]} ACx rows, "
            f"{view_df[view_df['area'] == 'OFC'].shape[0]} OFC rows."
        )

        # --- Summary table ---
        st.markdown("### Summary by Session Type \u00d7 Area")
        summary_df = _build_group_summary_table(view_df)
        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "n_sessions": st.column_config.NumberColumn("N", format="%d"),
                "mean_accuracy": st.column_config.NumberColumn("Mean Acc (behavior)", format="%.3f"),
                "sem_accuracy": st.column_config.NumberColumn("SEM Acc (behavior)", format="%.3f"),
                "mean_accuracy_gt": st.column_config.NumberColumn("Mean Acc (GT)", format="%.3f"),
                "sem_accuracy_gt": st.column_config.NumberColumn("SEM Acc (GT)", format="%.3f"),
                "mean_roc_auc": st.column_config.NumberColumn("Mean AUC (behavior)", format="%.3f"),
                "sem_roc_auc": st.column_config.NumberColumn("SEM AUC (behavior)", format="%.3f"),
                "mean_roc_auc_gt": st.column_config.NumberColumn("Mean AUC (GT)", format="%.3f"),
                "sem_roc_auc_gt": st.column_config.NumberColumn("SEM AUC (GT)", format="%.3f"),
                "mean_session_dprime": st.column_config.NumberColumn("Mean d'", format="%.3f"),
                "sem_session_dprime": st.column_config.NumberColumn("SEM d'", format="%.3f"),
                "mean_hit_rate": st.column_config.NumberColumn("Mean hit rate", format="%.2f"),
                "sem_hit_rate": st.column_config.NumberColumn("SEM hit rate", format="%.2f"),
            },
        )

        # --- Accuracy box plots ---
        st.markdown("### Decoder Accuracy by Area and Session Type")
        acc_col1, acc_col2 = st.columns(2)
        with acc_col1:
            st.plotly_chart(
                _plot_group_accuracy_strip(view_df, "accuracy", "Accuracy \u2014 Behavior Target"),
                use_container_width=True,
                config=get_plotly_config("group_accuracy_behavior"),
            )
        with acc_col2:
            st.plotly_chart(
                _plot_group_accuracy_strip(view_df, "accuracy_gt", "Accuracy \u2014 Ground Truth Target"),
                use_container_width=True,
                config=get_plotly_config("group_accuracy_gt"),
            )

        if "roc_auc" in view_df.columns and "roc_auc_gt" in view_df.columns:
            st.markdown("### Decoder ROC-AUC by Area and Session Type")
            auc_col1, auc_col2 = st.columns(2)
            with auc_col1:
                st.plotly_chart(
                    _plot_group_accuracy_strip(view_df, "roc_auc", "ROC-AUC \u2014 Behavior Target"),
                    use_container_width=True,
                    config=get_plotly_config("group_roc_auc_behavior"),
                )
            with auc_col2:
                st.plotly_chart(
                    _plot_group_accuracy_strip(view_df, "roc_auc_gt", "ROC-AUC \u2014 Ground Truth Target"),
                    use_container_width=True,
                    config=get_plotly_config("group_roc_auc_gt"),
                )
        else:
            st.caption(
                "ROC-AUC not in batch CSV (older batch run). Re-run the group decoder batch job to populate "
                "`roc_auc` / `roc_auc_gt` columns."
            )

        # --- Session d' vs decoder accuracy ---
        monitoring_sessions = load_valid_sessions(monitoring_path)
        view_df = merge_behavioral_file_from_monitoring(
            view_df,
            monitoring_sessions,
            animal_col="animal",
            date_col="date",
        )
        view_df = attach_session_dprime(view_df, behavioral_file_col="behavioral file")
        st.markdown("### Session d' vs Decoder Accuracy")
        exclude_low_hit_ols = st.checkbox(
            "Exclude <60% hit-rate sessions from OLS fit",
            value=True,
            key="npxl_group_dprime_exclude_low_hit_ols",
        )
        st.caption(
            "Decoder accuracy vs session d' (from behavioral .mat): behavior target (top), "
            "ground truth target (bottom). Points are colored by learning stage; open circles "
            "mark sessions with hit rate < 60%. The gray line is a single OLS fit per panel "
            "(optionally excluding low hit-rate sessions). Spearman r uses all plotted points."
        )
        dprime_fig = _plot_dprime_vs_accuracy(
            view_df,
            exclude_low_hit_from_ols=exclude_low_hit_ols,
        )
        if dprime_fig is None:
            st.info(
                "No session d' values could be computed (missing behavioral .mat path or load error)."
            )
        else:
            st.plotly_chart(
                dprime_fig,
                use_container_width=True,
                config=get_plotly_config("group_dprime_accuracy"),
            )
            # Count missing sessions on a per-session basis (one row per area).
            acx_view = view_df[view_df["area"] == "ACx"]
            n_missing = int(acx_view["session_dprime"].isna().sum()) if not acx_view.empty else int(
                view_df["session_dprime"].isna().sum()
            )
            if n_missing:
                st.caption(
                    f"{n_missing} session(s) missing d' (no behavioral .mat or could not compute)."
                )

        st.markdown("### Statistical comparisons (vs chance)")
        _render_group_metric_chance_stats(
            view_df, "accuracy", "Behavior target — accuracy", metric_label="accuracy"
        )
        _render_group_metric_chance_stats(
            view_df, "accuracy_gt", "Ground truth target — accuracy", metric_label="accuracy"
        )
        if "roc_auc" in view_df.columns:
            _render_group_metric_chance_stats(
                view_df, "roc_auc", "Behavior target — ROC-AUC", metric_label="ROC-AUC"
            )
        if "roc_auc_gt" in view_df.columns:
            _render_group_metric_chance_stats(
                view_df, "roc_auc_gt", "Ground truth target — ROC-AUC", metric_label="ROC-AUC"
            )

        # --- Agreement Space (dual-area sessions) ---
        joint_probs_df = st.session_state.get("_group_joint_probs_df", pd.DataFrame())
        model_agreement_df = st.session_state.get("_group_model_agreement_df", pd.DataFrame())
        psychometric_model_agree_df = st.session_state.get(
            "_group_psychometric_model_agree_df", pd.DataFrame()
        )
        psychometric_model_disagree_df = st.session_state.get(
            "_group_psychometric_model_disagree_df", pd.DataFrame()
        )
        kappa_agreement_df = st.session_state.get("_group_kappa_agreement_df", pd.DataFrame())
        filtered_joint = _filter_group_sidecar_to_view(joint_probs_df, view_df)
        filtered_agreement = _filter_group_sidecar_to_view(model_agreement_df, view_df)
        filtered_psych_agree = _filter_group_sidecar_to_view(psychometric_model_agree_df, view_df)
        filtered_psych_disagree = _filter_group_sidecar_to_view(
            psychometric_model_disagree_df, view_df
        )
        filtered_kappa = _filter_group_sidecar_to_view(kappa_agreement_df, view_df)

        dual_session_count = 0
        if not filtered_joint.empty:
            dual_session_count = filtered_joint[["animal", "date", "session_type"]].drop_duplicates().shape[0]
        elif not filtered_agreement.empty:
            dual_session_count = filtered_agreement[["animal", "date", "session_type"]].drop_duplicates().shape[0]
        elif not filtered_kappa.empty:
            dual_session_count = filtered_kappa[["animal", "date", "session_type"]].drop_duplicates().shape[0]

        if dual_session_count == 0:
            st.markdown("### Agreement Space (dual-area sessions)")
            st.info(
                "No dual-area agreement data for the selected session types. "
                "Re-run the cluster batch job after updating the decoder scripts, or use "
                "**Run Group Analysis (interactive)**."
            )
        else:
            st.markdown("### Agreement Space (dual-area sessions)")
            st.caption(
                f"Pooled across {dual_session_count} dual-area session(s). "
                "GT-decoder OOF predictions; behavior match = lick matches the shared ACx/OFC prediction."
            )
            if filtered_agreement.empty and not filtered_joint.empty:
                filtered_agreement = _model_agreement_summary_from_joint(filtered_joint)

            if filtered_agreement.empty:
                st.info(
                    "Model-agreement summary not found. Re-run batch to generate "
                    "`*_model_agreement.csv`, or use interactive group analysis."
                )
            else:
                pooled_agreement = _aggregate_model_agreement_across_sessions(filtered_agreement)
                agree_fig = _plot_group_agreement_summary(pooled_agreement)
                if agree_fig is not None:
                    st.plotly_chart(
                        agree_fig,
                        use_container_width=True,
                        config=get_plotly_config("group_agreement_summary"),
                    )
                st.dataframe(
                    pooled_agreement,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "n_trials": st.column_config.NumberColumn("Trials", format="%d"),
                        "n_behavior_match": st.column_config.NumberColumn("N match", format="%d"),
                        "p_behavior_match": st.column_config.NumberColumn("P(match)", format="%.3f"),
                        "odds_behavior_match": st.column_config.NumberColumn("Odds", format="%.2f"),
                    },
                )

            if filtered_kappa.empty:
                expected_kappa_path = _group_decoder_sidecar_path(batch_source_path, "kappa_agreement")
                expected_kappa_name = (
                    os.path.basename(expected_kappa_path)
                    if expected_kappa_path is not None
                    else "*_kappa_agreement.csv"
                )
                st.info(
                    f"Kappa agreement sidecar not found for the currently loaded **{histology_label}** "
                    f"results (`{expected_kappa_name}`). Re-run batch for this mode, or use "
                    f"interactive group analysis.{_group_decoder_alternate_sidecar_hint(g_use_histology, 'kappa_agreement')}"
                )
            else:
                st.markdown("**Chance-corrected agreement (session-level kappa)**")
                st.caption(
                    "Cohen's kappa is pairwise chance-corrected Go/No-Go agreement. "
                    "Fleiss' kappa summarizes agreement across Mouse+ACx+OFC, with or without GT. "
                    "Values: 1 = perfect, 0 = chance-level given label marginals, <0 = systematic disagreement."
                )
                kappa_stage_df = _attach_learning_stage(filtered_kappa)
                pooled_kappa = _aggregate_kappa_agreement_across_sessions(
                    kappa_stage_df,
                    extra_group_cols=["learning_stage"],
                )
                if pooled_kappa.empty:
                    st.info("Not enough kappa rows for the selected session types.")
                else:
                    kappa_display_cols = [
                        col
                        for col in [
                            "learning_stage",
                            "n_sessions",
                            "n_trials",
                            "mean_cohen_mouse_acx",
                            "sem_cohen_mouse_acx",
                            "mean_cohen_mouse_ofc",
                            "sem_cohen_mouse_ofc",
                            "mean_cohen_acx_ofc",
                            "sem_cohen_acx_ofc",
                            "mean_fleiss_mouse_acx_ofc",
                            "sem_fleiss_mouse_acx_ofc",
                            "mean_fleiss_mouse_acx_ofc_gt",
                            "sem_fleiss_mouse_acx_ofc_gt",
                            "mean_var_agree_acx_ofc",
                            "sem_var_agree_acx_ofc",
                        ]
                        if col in pooled_kappa.columns
                    ]
                    st.dataframe(
                        pooled_kappa[kappa_display_cols],
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "n_sessions": st.column_config.NumberColumn("Sessions", format="%d"),
                            "n_trials": st.column_config.NumberColumn("Trials", format="%d"),
                            **{
                                col: st.column_config.NumberColumn(col.replace("_", " "), format="%.3f")
                                for col in kappa_display_cols
                                if col.startswith(("mean_", "sem_"))
                            },
                        },
                    )

                for metric_col, plot_key in (
                    ("cohen_acx_ofc", "group_cohen_acx_ofc_by_stage"),
                    ("cohen_mouse_acx", "group_cohen_mouse_acx_by_stage"),
                    ("cohen_mouse_ofc", "group_cohen_mouse_ofc_by_stage"),
                ):
                    _render_cohen_kappa_by_stage(
                        kappa_stage_df,
                        metric_col,
                        plot_key=plot_key,
                    )

            if filtered_psych_agree.empty:
                st.info(
                    "Stimulus-axis agreement data not found. Re-run batch to generate "
                    "`*_psychometric_model_agree.csv`."
                )
            else:
                psych_stage_df = _filter_models_agree_psychometric(
                    _attach_learning_stage(filtered_psych_agree)
                )
                psych_stage_df = psych_stage_df[
                    psych_stage_df["learning_stage"].isin(_PSYCHOMETRIC_EXPERT_STAGES)
                ]
                psych_stage_df = _filter_1b_psychometric_above_high_boundary(psych_stage_df)
                psych_stage_df = _log_bin_stimulus_per_stage(psych_stage_df, bins_per_class=4)
                pooled_psych_agree = _aggregate_psychometric_behavior_match_across_sessions(
                    psych_stage_df,
                    extra_group_cols=["learning_stage"],
                    min_trials_per_bin=_PSYCHOMETRIC_GROUP_MIN_TRIALS_PER_BIN,
                )
                psych_agree_fig = _plot_group_agreement_psychometric(pooled_psych_agree)
                if psych_agree_fig is None:
                    st.info("Not enough stimulus bins for a psychometric-style agreement plot.")
                else:
                    st.plotly_chart(
                        psych_agree_fig,
                        use_container_width=True,
                        config=get_plotly_config("group_agreement_psychometric"),
                    )

            if filtered_psych_disagree.empty:
                st.info(
                    "Stimulus-axis model-disagreement data not found. Re-run batch to generate "
                    "`*_psychometric_model_disagree.csv`, or use **Run Group Analysis (interactive)**."
                )
            else:
                psych_disagree_stage_df = _attach_learning_stage(filtered_psych_disagree)
                psych_disagree_stage_df = psych_disagree_stage_df[
                    psych_disagree_stage_df["learning_stage"].isin(_PSYCHOMETRIC_EXPERT_STAGES)
                ]
                psych_disagree_stage_df = _filter_1b_psychometric_above_high_boundary(
                    psych_disagree_stage_df
                )
                psych_disagree_stage_df = _log_bin_stimulus_per_stage(
                    psych_disagree_stage_df, bins_per_class=4
                )
                pooled_psych_disagree = _aggregate_psychometric_behavior_match_across_sessions(
                    psych_disagree_stage_df,
                    extra_group_cols=["learning_stage"],
                    min_trials_per_bin=_PSYCHOMETRIC_GROUP_MIN_TRIALS_PER_BIN,
                )
                psych_disagree_fig = _plot_group_disagreement_area_match_psychometric(
                    pooled_psych_disagree
                )
                if psych_disagree_fig is None:
                    st.info(
                        "Not enough stimulus bins for a model-disagreement psychometric plot."
                    )
                else:
                    st.plotly_chart(
                        psych_disagree_fig,
                        use_container_width=True,
                        config=get_plotly_config("group_disagreement_psychometric"),
                    )

        st.download_button(
            "Download filtered session results CSV",
            data=view_df.to_csv(index=False).encode("utf-8"),
            file_name="npxl_group_decoder_results.csv",
            mime="text/csv",
        )

    if failures:
        with st.expander(f"Failed sessions ({len(failures)})", expanded=False):
            st.dataframe(pd.DataFrame(failures), use_container_width=True, hide_index=True)


def _in_streamlit() -> bool:
    """Return True only when executed as a live Streamlit page (not when imported)."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


# --- Top-level script (only executes when run as a Streamlit page, not when imported) ---

if _in_streamlit():
    st.title("NPXL Agreement Decoder")
    st.caption("Session-level ACx and OFC decoders for Go / No-Go behavioral choice agreement.")

    monitoring_path = st.session_state.get("npxl_monitoring_path", "")
    if not monitoring_path or not os.path.exists(monitoring_path):
        st.error("NPXL monitoring CSV was not found in session state.")
        st.stop()

    valid_sessions_df = load_valid_sessions(monitoring_path)
    if valid_sessions_df.empty:
        st.info(f"No sessions have at least {UNIT_THRESHOLD} good+MUA units in ACx or OFC.")
        st.stop()

    tab1, tab2 = st.tabs(["Single Session", "Group Analysis"])

    with tab1:
        _render_single_session_tab(valid_sessions_df, monitoring_path)

    with tab2:
        _render_group_analysis_tab(valid_sessions_df, monitoring_path)
