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
from sklearn.decomposition import PCA
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


def _category_boundary_khz() -> tuple[float, float]:
    """Tone-category boundaries (kHz) from global session state."""
    return (
        float(st.session_state.get("low_boundary", 0.983)),
        float(st.session_state.get("high_boundary", 1.525)),
    )


def _plot_psychometric_summary(
    results_df: pd.DataFrame,
    *,
    error_trials_only: bool = False,
) -> go.Figure | None:
    if "stimulus" not in results_df.columns:
        return None

    work_df = results_df.copy()
    if error_trials_only:
        work_df = work_df[work_df["behavior_go"] != work_df["ground_truth_go"]].copy()
        if work_df.empty:
            return None

    work_df["stimulus_khz"] = pd.to_numeric(work_df["stimulus"], errors="coerce")
    work_df = work_df[np.isfinite(work_df["stimulus_khz"]) & (work_df["stimulus_khz"] > 0)]
    if work_df.empty:
        return None

    agg_map: dict[str, tuple[str, str]] = {
        "trials": ("stimulus_khz", "size"),
        "mouse_go": ("behavior_go", "mean"),
        "gt_go": ("ground_truth_go", "mean"),
    }
    if "acx_pred" in work_df.columns:
        agg_map["acx_go"] = ("acx_pred", "mean")
    if "ofc_pred" in work_df.columns:
        agg_map["ofc_go"] = ("ofc_pred", "mean")

    grouped = work_df.groupby("stimulus_khz", observed=True).agg(**agg_map).reset_index()
    grouped = grouped.rename(columns={"stimulus_khz": "stimulus"})
    grouped = grouped[grouped["trials"] >= 3].sort_values("stimulus")  # avoid noisy single-trial bins
    if grouped.shape[0] < 2:
        return None

    low_boundary, high_boundary = _category_boundary_khz()
    fig = go.Figure()
    for column, name in [
        ("mouse_go", "Mouse Go Choice"),
        ("acx_go", "ACx Decoder Go"),
        ("ofc_go", "OFC Decoder Go"),
        ("gt_go", "Ground Truth Go"),
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


_PCA_GO_COLOR = "#2ECC71"     # green for Go / Hit + FA
_PCA_NOGO_COLOR = "#E74C3C"   # red for No-Go / Miss + CR


def _plot_population_pca(
    x_matrix: np.ndarray,
    y_behavior: np.ndarray,
    area: str,
    random_state: int,
) -> go.Figure:
    """2-D PCA of population activity in the decode window, coloured by behavioral choice."""
    # z-score across trials per unit before projecting (same as decoder preprocessing)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_matrix)

    # Clamp to at most 2 PCs; with < 2 units or trials PCA degrades gracefully
    n_components = min(2, x_scaled.shape[0], x_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=random_state)
    coords = pca.fit_transform(x_scaled)
    var_explained = pca.explained_variance_ratio_ * 100

    go_mask = y_behavior == 1
    nogo_mask = y_behavior == 0

    fig = go.Figure()
    for mask, label, color in (
        (go_mask, "Go (lick)", _PCA_GO_COLOR),
        (nogo_mask, "No-Go (withhold)", _PCA_NOGO_COLOR),
    ):
        if mask.any():
            x_vals = coords[mask, 0].tolist()
            y_vals = coords[mask, 1].tolist() if n_components > 1 else [0.0] * int(mask.sum())
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode="markers",
                    name=label,
                    marker=dict(color=color, size=8, opacity=0.75, line=dict(width=0.5, color="white")),
                )
            )

    pc1_label = f"PC1 ({var_explained[0]:.1f}%)" if len(var_explained) > 0 else "PC1"
    pc2_label = f"PC2 ({var_explained[1]:.1f}%)" if len(var_explained) > 1 else "PC2"
    fig.update_layout(
        title=f"{area} Population PCA — behavioral choice",
        xaxis_title=pc1_label,
        yaxis_title=pc2_label,
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


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
) -> list[dict[str, Any]]:
    """Decode one session for both behavior and GT targets; return one record per area.

    Runs both decode targets in a single data-loading pass to avoid loading event matrices twice.
    """
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
        return []

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

    records: list[dict[str, Any]] = []
    for area in available_areas:
        bm = behavior_results[area]["metrics"]
        gm = gt_results[area]["metrics"]
        dprime_val = session_meta.get("session_dprime", np.nan)
        hit_rate_val = session_meta.get("session_hit_rate", np.nan)
        records.append(
            {
                "animal": str(session_meta.get("Animal", "")),
                "date": str(session_meta.get("Date", "")),
                "session_type": str(session_meta.get("Session Type", "")),
                "session_dprime": float(dprime_val) if pd.notna(dprime_val) else np.nan,
                "session_hit_rate": float(hit_rate_val) if pd.notna(hit_rate_val) else np.nan,
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
    return records


@st.cache_data(show_spinner=False)
def _batch_decode_all_sessions(
    monitoring_path: str,
    decode_window: tuple[float, float],
    aggregation: str,
    classifier_type: str,
    random_state: int,
    min_trials_per_class: int,
    use_histology: bool,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Cached batch decoder — iterate all valid sessions; return (results_df, failures)."""
    sessions_df = load_valid_sessions(monitoring_path)
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for _, row in sessions_df.iterrows():
        session_dir = str(row.get("current_dir", "")).strip()
        label = str(row.get("session_label", "?"))
        try:
            records.extend(
                _run_one_session_batch(
                    session_dir,
                    row,
                    decode_window=decode_window,
                    aggregation=aggregation,
                    classifier_type=classifier_type,
                    random_state=random_state,
                    min_trials_per_class=min_trials_per_class,
                    use_histology=use_histology,
                )
            )
        except Exception as exc:  # noqa: BLE001 — per-session failure must not abort the batch
            failures.append({"session": label, "error": str(exc)})
    return pd.DataFrame(records), failures


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


def _load_group_decoder_batch_results(
    use_histology: bool,
) -> tuple[pd.DataFrame, list[dict[str, str]], str | None]:
    """Load pre-computed group decoder output from the cluster batch job."""
    results_path = _resolve_group_decoder_results_path(use_histology)
    if results_path is None:
        return pd.DataFrame(), [], None

    batch_df = _read_group_decoder_csv(results_path)
    failures_path = _group_decoder_failures_path(results_path)
    failures: list[dict[str, str]] = []
    if os.path.isfile(failures_path):
        failures = pd.read_csv(failures_path).to_dict("records")
    return batch_df, failures, results_path


def _build_group_summary_table(batch_df: pd.DataFrame) -> pd.DataFrame:
    """Mean ± SEM per (session_type × area) for decoder accuracy."""
    def _sem(x: pd.Series) -> float:
        return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan

    agg: dict[str, tuple[str, str]] = {
        "n_sessions": ("accuracy", "count"),
        "mean_accuracy": ("accuracy", "mean"),
        "sem_accuracy": ("accuracy", _sem),
        "mean_accuracy_gt": ("accuracy_gt", "mean"),
        "sem_accuracy_gt": ("accuracy_gt", _sem),
    }
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


def _render_group_accuracy_stats(
    batch_df: pd.DataFrame,
    metric_col: str,
    title: str,
) -> None:
    """Show omnibus and pairwise tests for session-type accuracy differences (per area)."""
    omnibus_df, pairwise_df = _run_group_accuracy_session_type_tests(batch_df, metric_col)
    if omnibus_df.empty and pairwise_df.empty:
        st.caption(f"{title}: not enough sessions per group (need ≥2 per session type and area).")
        return

    st.markdown(f"**{title}**")
    st.caption(
        "Session-level cross-validated accuracy; unpaired Mann-Whitney U between session types "
        "(within ACx or within OFC only). Bonferroni correction across pairwise tests per area."
    )
    if not omnibus_df.empty:
        st.dataframe(
            omnibus_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "statistic": st.column_config.NumberColumn(format="%.3f"),
                "p": st.column_config.NumberColumn(format="%.4g"),
            },
        )
    if not pairwise_df.empty:
        st.dataframe(
            pairwise_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "n_a": st.column_config.NumberColumn(format="%d"),
                "n_b": st.column_config.NumberColumn(format="%d"),
                "U": st.column_config.NumberColumn(format="%.1f"),
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
                x=area_df["session_type"],
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
    ordered_types = _order_session_types(batch_df["session_type"])
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
        fig.add_trace(
            go.Box(
                x=stype_df["session_type"],
                y=stype_df["beta_diff"],
                name=stype,
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
    fig.update_layout(
        title="ACx vs OFC Weighting (\u03b2_ACx \u2212 \u03b2_OFC) by Session Type",
        xaxis_title="Session Type",
        yaxis_title="Beta Difference (ACx \u2212 OFC)",
        height=420,
    )
    fig.update_xaxes(categoryorder="array", categoryarray=session_types)
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
) -> tuple[set[str], bool]:
    """Scatter + OLS + Spearman annotation for one (target, area) panel."""
    area_df = plot_df[(plot_df["area"] == area) & plot_df[metric_col].notna()].copy()
    if area_df.empty:
        return legend_shown, show_regression_legend

    stage_order = ["Novice", "1b Expert", "2b Expert", "Other"]
    for stage in stage_order:
        stage_df = area_df[area_df["learning_stage"] == stage]
        if stage_df.empty:
            continue
        color = _STAGE_LINE_COLORS.get(stage, "#888888")
        show_legend = stage not in legend_shown
        legend_shown.add(stage)

        fig.add_trace(
            go.Scatter(
                x=stage_df["session_dprime"],
                y=stage_df[metric_col],
                mode="markers",
                name=stage,
                legendgroup=stage,
                showlegend=show_legend,
                marker=dict(color=color, size=10, line=dict(width=1, color="white")),
                hovertemplate=(
                    f"{area}<br>"
                    "Animal: %{customdata[0]}<br>"
                    "Date: %{customdata[1]}<br>"
                    "d': %{x:.2f}<br>"
                    "Accuracy: %{y:.3f}<extra></extra>"
                ),
                customdata=stage_df[["animal", "date"]].to_numpy(),
            ),
            row=row,
            col=col,
        )

    if len(area_df) >= 3:
        x_vals = area_df["session_dprime"].to_numpy(dtype=float)
        y_vals = area_df[metric_col].to_numpy(dtype=float)
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
    return legend_shown, show_regression_legend


def _plot_dprime_vs_accuracy(view_df: pd.DataFrame) -> go.Figure | None:
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
    x_pad = max(0.1, 0.05 * (x_max - x_min) if x_max > x_min else 0.5)
    x_lo, x_hi = x_min - x_pad, x_max + x_pad

    n_rows = len(targets)
    n_cols = len(areas_present)
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=areas_present * n_rows,
        row_titles=[label for _, label in targets],
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.14,
        horizontal_spacing=0.08,
    )

    legend_shown: set[str] = set()
    show_regression_legend = True
    for row_idx, (metric_col, _) in enumerate(targets, start=1):
        for col_idx, area in enumerate(areas_present, start=1):
            legend_shown, show_regression_legend = _add_dprime_accuracy_panel(
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
                range=[0, 1.05],
                row=row_idx,
                col=col_idx,
            )

    fig.update_layout(
        title="Session d' vs Decoder Accuracy",
        height=380 + 320 * n_rows,
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

    st.subheader("Population PCA")
    pca_cols = st.columns(max(1, len(available_areas)))
    for col, area in zip(pca_cols, available_areas):
        with col:
            st.plotly_chart(
                _plot_population_pca(area_features[area], y_behavior, area, int(random_state)),
                use_container_width=True,
                config=get_plotly_config(f"pca_{area.lower()}"),
            )

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
        loaded_df, loaded_failures, loaded_path = _load_group_decoder_batch_results(g_use_histology)
        if loaded_path is not None:
            st.session_state["_group_batch_df"] = loaded_df
            st.session_state["_group_batch_failures"] = loaded_failures
            st.session_state["_group_batch_source"] = "batch"
            st.session_state["_group_batch_source_path"] = loaded_path
            st.session_state["_group_loaded_histology"] = g_use_histology
            st.session_state.pop("_group_settings_run", None)
        elif histology_changed or reload_batch:
            st.session_state["_group_batch_df"] = pd.DataFrame()
            st.session_state["_group_batch_failures"] = []
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
            batch_df, failures = _batch_decode_all_sessions(
                monitoring_path,
                *current_settings,
            )
        st.session_state["_group_batch_df"] = batch_df
        st.session_state["_group_batch_failures"] = failures
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
        st.caption(
            "Decoder accuracy vs session d' (from behavioral .mat): behavior target (top), "
            "ground truth target (bottom). Points are colored by learning stage; the gray line "
            "is a single OLS fit per panel. Spearman r is shown on each panel."
        )
        dprime_fig = _plot_dprime_vs_accuracy(view_df)
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

        st.markdown("### Statistical comparisons (accuracy)")
        _render_group_accuracy_stats(view_df, "accuracy", "Behavior target")
        _render_group_accuracy_stats(view_df, "accuracy_gt", "Ground truth target")

        # --- Area weighting (dual-area sessions only) ---
        beta_fig = _plot_group_beta_diff(view_df)
        if beta_fig is not None:
            st.markdown("### Area Weighting: ACx vs OFC (dual-area sessions)")
            st.plotly_chart(
                beta_fig,
                use_container_width=True,
                config=get_plotly_config("group_beta_diff"),
            )
            dual_summary = (
                view_df[view_df["area"] == "ACx"]
                .dropna(subset=["beta_diff"])
                .groupby("session_type", observed=True)
                .agg(
                    n_sessions=("beta_diff", "count"),
                    mean_beta_diff=("beta_diff", "mean"),
                    sem_beta_diff=(
                        "beta_diff",
                        lambda x: float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan,
                    ),
                )
                .reset_index()
            )
            st.dataframe(
                dual_summary,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "n_sessions": st.column_config.NumberColumn("N", format="%d"),
                    "mean_beta_diff": st.column_config.NumberColumn("Mean \u03b2_ACx \u2212 \u03b2_OFC", format="%.4f"),
                    "sem_beta_diff": st.column_config.NumberColumn("SEM \u03b2 diff", format="%.4f"),
                },
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
