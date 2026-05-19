"""Compute session-level d' and hit rate from Bpod behavioral .mat files.

Used by `npxl_monitoring.py` and `npxl_agreement_decoder.py` via the
``behavioral file`` column (or an equivalent path column). Uses the same
pipeline as the GNG analysis app: ``load_mat_file`` → ``create_single_row_with_outcome``
→ ``metric.d_prime`` (mean d' across trial bins) and ``licking_and_outcome.responses``
(session hit rate = Hit / (Hit + Miss)).
"""
from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import pandas as pd
import streamlit as st

# Session-type substrings used to classify learning stage.
_STAGE_RANK: dict[str, int] = {"Novice": 0, "1b Expert": 1, "2b Expert": 2, "Other": 99}

DEFAULT_BEHAVIORAL_FILE_COL = "behavioral file"
DEFAULT_DPRIME_WINDOW_S = 10
DEFAULT_DPRIME_COL = "session_dprime"
DEFAULT_HIT_RATE_COL = "session_hit_rate"


def _resolve_mat_path(path: object) -> str | None:
    """Return an existing .mat path, or None if missing / invalid."""
    if path is None or (isinstance(path, float) and np.isnan(path)):
        return None
    text = str(path).strip()
    if not text or text.lower() == "nan":
        return None
    normalized = os.path.normpath(text)
    if os.path.isfile(normalized):
        return normalized
    return None


def _session_df_from_mat(mat_path: str) -> pd.DataFrame | None:
    """Load one Bpod .mat session as a single-row DataFrame, or None on failure / FRA."""
    from load_data.load_bpod_data import create_single_row_with_outcome, load_mat_file

    resolved = _resolve_mat_path(mat_path)
    if resolved is None:
        return None

    try:
        (
            trial_types_df,
            raw_events_df,
            session_date,
            session_time,
            trial_settings,
            notes,
            licks,
            states,
            stimulis,
            unique_stimuli_values,
            tones_per_class,
            boundaries,
            recs,
            outcome_names,
        ) = load_mat_file(resolved)

        # FRA sessions use a different trial table — Go/NoGo metrics do not apply.
        if isinstance(stimulis, np.ndarray) and stimulis.ndim == 2:
            return None

        session_df = create_single_row_with_outcome(
            resolved,
            trial_types_df,
            raw_events_df,
            session_date,
            session_time,
            trial_settings,
            notes,
            licks,
            states,
            unique_stimuli_values,
            tones_per_class,
            boundaries,
            recs,
            outcome_names=outcome_names,
        )
        if session_df.empty:
            return None
        return session_df
    except Exception:
        return None


def _hit_rate_from_session_df(session_df: pd.DataFrame) -> float:
    """Session hit rate = Hit / (Hit + Miss) from the raw Outcomes list.

    Counts outcomes directly to avoid the ``ast.literal_eval`` path in
    ``licking_and_outcome.responses`` (which assumes Outcomes is a JSON-like string,
    while ``create_single_row_with_outcome`` stores a Python list).
    """
    if "Outcomes" not in session_df.columns or session_df.empty:
        return float("nan")

    raw = session_df["Outcomes"].values[0]
    if isinstance(raw, str):
        import ast
        try:
            outcomes_list = ast.literal_eval(raw)
        except Exception:
            return float("nan")
    elif isinstance(raw, (list, tuple, np.ndarray)):
        outcomes_list = list(raw)
    else:
        return float("nan")

    if not outcomes_list:
        return float("nan")

    hits = sum(1 for o in outcomes_list if str(o).strip().lower() == "hit")
    misses = sum(1 for o in outcomes_list if str(o).strip().lower() == "miss")
    denom = hits + misses
    if denom <= 0:
        return float("nan")
    rate = hits / denom
    return rate if np.isfinite(rate) else float("nan")


@st.cache_data(show_spinner=False)
def compute_session_metrics_from_mat(
    mat_path: str,
    t: int = DEFAULT_DPRIME_WINDOW_S,
) -> tuple[float, float]:
    """Return ``(mean_dprime, hit_rate)`` for one behavioral .mat session.

    d' and hit rate are computed independently so a failure of one does not
    blank out the other.
    """
    from Analysis.GNG_bpod_analysis.metric import d_prime

    session_df = _session_df_from_mat(mat_path)
    if session_df is None:
        return float("nan"), float("nan")

    try:
        d_vals = d_prime(session_df, index=0, t=t, plot=False)
        if d_vals is None or len(d_vals) == 0:
            mean_d = float("nan")
        else:
            mean_d = float(np.nanmean(d_vals))
            mean_d = mean_d if np.isfinite(mean_d) else float("nan")
    except Exception:
        mean_d = float("nan")

    try:
        hit_rate = _hit_rate_from_session_df(session_df)
    except Exception:
        hit_rate = float("nan")

    return mean_d, hit_rate


@st.cache_data(show_spinner=False)
def compute_session_dprime_from_mat(mat_path: str, t: int = DEFAULT_DPRIME_WINDOW_S) -> float:
    """Load one Bpod .mat session and return mean d' (same definition as ``metric.d_prime``)."""
    return compute_session_metrics_from_mat(mat_path, t=t)[0]


@st.cache_data(show_spinner=False)
def compute_session_hit_rate_from_mat(mat_path: str, t: int = DEFAULT_DPRIME_WINDOW_S) -> float:
    """Load one Bpod .mat session and return hit rate (Hit / (Hit + Miss))."""
    return compute_session_metrics_from_mat(mat_path, t=t)[1]


def attach_session_dprime(
    df: pd.DataFrame,
    *,
    behavioral_file_col: str = DEFAULT_BEHAVIORAL_FILE_COL,
    out_col: str = DEFAULT_DPRIME_COL,
    hit_rate_col: str | None = DEFAULT_HIT_RATE_COL,
    t: int = DEFAULT_DPRIME_WINDOW_S,
) -> pd.DataFrame:
    """Add ``out_col`` (and optionally ``hit_rate_col``) from each row's behavioral .mat path."""
    out = df.copy()
    if behavioral_file_col not in out.columns:
        out[out_col] = np.nan
        if hit_rate_col:
            out[hit_rate_col] = np.nan
        return out

    metrics = out[behavioral_file_col].map(lambda val: compute_session_metrics_from_mat(str(val), t=t))
    out[out_col] = metrics.map(lambda pair: pair[0])
    if hit_rate_col:
        out[hit_rate_col] = metrics.map(lambda pair: pair[1])
    return out


def merge_behavioral_file_from_monitoring(
    df: pd.DataFrame,
    monitoring_df: pd.DataFrame,
    *,
    animal_col: str = "animal",
    date_col: str = "date",
    monitoring_animal_col: str = "Animal",
    monitoring_date_col: str = "Date",
    behavioral_file_col: str = DEFAULT_BEHAVIORAL_FILE_COL,
) -> pd.DataFrame:
    """Attach ``behavioral file`` paths from the NPXL monitoring table by animal + date."""
    if behavioral_file_col not in monitoring_df.columns:
        return df.copy()

    mon = monitoring_df[[monitoring_animal_col, monitoring_date_col, behavioral_file_col]].copy()
    mon["_animal_key"] = mon[monitoring_animal_col].astype(str).str.strip()
    mon["_date_key"] = pd.to_datetime(mon[monitoring_date_col], dayfirst=True, errors="coerce").dt.normalize()
    mon = mon.dropna(subset=["_date_key"]).drop_duplicates(subset=["_animal_key", "_date_key"], keep="first")

    out = df.copy()
    if animal_col not in out.columns or date_col not in out.columns:
        out[behavioral_file_col] = pd.NA
        return out

    out["_animal_key"] = out[animal_col].astype(str).str.strip()
    out["_date_key"] = pd.to_datetime(out[date_col], dayfirst=True, errors="coerce").dt.normalize()
    merged = out.merge(
        mon[["_animal_key", "_date_key", behavioral_file_col]],
        on=["_animal_key", "_date_key"],
        how="left",
    )
    return merged.drop(columns=["_animal_key", "_date_key"])


def classify_learning_stage(session_type: object) -> str:
    """Map a session_type label to Novice / 1b Expert / 2b Expert / Other."""
    s = str(session_type).lower() if session_type is not None else ""
    if "novice" in s:
        return "Novice"
    if "2b" in s:
        return "2b Expert"
    if "1b" in s:
        return "1b Expert"
    return "Other"


def order_session_types(session_types: Iterable[object]) -> list[str]:
    """Left-to-right plot order: Novice -> 1b Expert -> 2b Expert -> other."""
    unique = {str(s) for s in session_types if pd.notna(s) and str(s).strip()}
    return sorted(unique, key=lambda s: (_STAGE_RANK.get(classify_learning_stage(s), 99), s))
