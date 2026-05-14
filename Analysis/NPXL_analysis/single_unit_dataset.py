"""Load and enrich multi-session single-unit metrics for Streamlit panels."""
from __future__ import annotations

import ast
import os
from functools import lru_cache
from typing import Any, Optional

import numpy as np
import pandas as pd


AREA_CONFIG: dict[str, dict[str, str]] = {
    "ACx": {"metric_prefix": "acx", "imec": "imec0"},
    "OFC": {"metric_prefix": "ofc", "imec": "imec1"},
}

SESSION_METADATA_COLUMNS: dict[str, tuple[str, ...]] = {
    "animal": ("Animal", "MouseName"),
    "date": ("Date", "SessionDate"),
    "session_time": ("SessionTime", "Time"),
    "session_type": ("Session Type", "SessionType", "Session_Type"),
    "spike_glx_file": ("spike glx file", "SpikeGLX", "Recording"),
    "notes": ("Notes",),
}


def _first_existing_value(row: pd.Series, names: tuple[str, ...]) -> Any:
    for name in names:
        if name not in row:
            continue

        value = row.get(name)
        if pd.isna(value):
            continue

        if isinstance(value, str):
            value = value.strip()
            if not value:
                continue

        return value

    return ""


def _normalize_unit_type(value: Any) -> str:
    if pd.isna(value):
        return "unknown"

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric_map = {
            0: "noise",
            1: "good",
            2: "mua",
            3: "non-somatic",
        }
        return numeric_map.get(int(value), "unknown")

    text = str(value).strip().lower()
    if not text:
        return "unknown"

    if text in {"1", "1.0"}:
        return "good"
    if text in {"2", "2.0"}:
        return "mua"
    if text in {"3", "3.0"}:
        return "non-somatic"
    if text in {"0", "0.0"}:
        return "noise"
    if "good" in text:
        return "good"
    if "mua" in text:
        return "mua"
    if "non" in text and ("soma" in text or "somatic" in text):
        return "non-somatic"
    if "noise" in text:
        return "noise"

    return text


def _session_metadata(row: pd.Series) -> dict[str, Any]:
    return {
        key: _first_existing_value(row, source_columns)
        for key, source_columns in SESSION_METADATA_COLUMNS.items()
    }


def _catgt_recording_name(session_dir: str) -> str:
    base_name = os.path.basename(os.path.normpath(session_dir))
    return base_name.removeprefix("catgt_")


@lru_cache(maxsize=512)
def _find_mapping_csv(session_dir: str, area: str) -> Optional[str]:
    config = AREA_CONFIG[area]
    imec = config["imec"]
    recording_name = _catgt_recording_name(session_dir)
    filename = f"unit_by_channel_mapping_{area}_probe.csv"

    direct_candidates = [
        os.path.join(session_dir, f"{recording_name}_{imec}", f"{imec}_ks4", filename),
        os.path.join(session_dir, f"{recording_name}_{imec}", f"{imec}_ks4", filename.lower()),
    ]
    for candidate in direct_candidates:
        if os.path.exists(candidate):
            return candidate

    # Mapping files are small but path layouts vary between acquisitions.
    for root, _dirs, files in os.walk(session_dir):
        if f"{imec}_ks4" not in root:
            continue

        for file_name in files:
            lowered = file_name.lower()
            if lowered.startswith("unit_by_channel_mapping_") and area.lower() in lowered:
                return os.path.join(root, file_name)

    return None


@lru_cache(maxsize=512)
def _load_mapping_csv(mapping_path: str) -> pd.DataFrame:
    mapping_df = pd.read_csv(mapping_path)
    mapping_df = mapping_df.rename(
        columns={
            "unitID": "mapping_unit_id",
            "region": "histology_region",
            "unit_type": "mapping_unit_type",
        }
    )

    if "mapping_unit_id" in mapping_df.columns:
        mapping_df["mapping_unit_id"] = pd.to_numeric(mapping_df["mapping_unit_id"], errors="coerce")

    keep_columns = [
        col
        for col in [
            "mapping_unit_id",
            "peak_channel",
            "y_pos",
            "histology_region",
            "mapping_unit_type",
            "cortex_group",
        ]
        if col in mapping_df.columns
    ]
    return mapping_df[keep_columns].drop_duplicates(subset=["mapping_unit_id"], keep="first")


@lru_cache(maxsize=512)
def _load_metrics_csv(session_dir: str, area: str) -> Optional[pd.DataFrame]:
    metric_prefix = AREA_CONFIG[area]["metric_prefix"]
    metrics_path = os.path.join(
        session_dir,
        "analysis_output",
        "tables",
        f"{metric_prefix}_all_units_metrics.csv",
    )

    if not os.path.exists(metrics_path):
        return None

    metrics_df = pd.read_csv(metrics_path, low_memory=False)
    metrics_df["metrics_path"] = metrics_path
    return metrics_df


def _join_mapping(metrics_df: pd.DataFrame, session_dir: str, area: str) -> pd.DataFrame:
    enriched_df = metrics_df.copy()
    mapping_path = _find_mapping_csv(session_dir, area)
    enriched_df["mapping_path"] = mapping_path or ""

    if not mapping_path:
        enriched_df["mapping_join_status"] = "missing_mapping_file"
        return enriched_df

    mapping_df = _load_mapping_csv(mapping_path)
    if mapping_df.empty or "mapping_unit_id" not in mapping_df.columns:
        enriched_df["mapping_join_status"] = "empty_mapping_file"
        return enriched_df

    join_source = "unit_idx"
    if "label_unitID" in enriched_df.columns:
        label_unit_ids = pd.to_numeric(enriched_df["label_unitID"], errors="coerce")
        if label_unit_ids.notna().any():
            enriched_df["_mapping_join_unit_id"] = label_unit_ids
            join_source = "label_unitID"
        else:
            enriched_df["_mapping_join_unit_id"] = pd.to_numeric(enriched_df["unit_idx"], errors="coerce")
    else:
        enriched_df["_mapping_join_unit_id"] = pd.to_numeric(enriched_df["unit_idx"], errors="coerce")

    enriched_df = enriched_df.merge(
        mapping_df,
        how="left",
        left_on="_mapping_join_unit_id",
        right_on="mapping_unit_id",
    )
    enriched_df["mapping_join_source"] = join_source
    enriched_df["mapping_join_status"] = enriched_df["mapping_unit_id"].notna().map(
        {True: "matched", False: "unmatched"}
    )
    return enriched_df.drop(columns=["_mapping_join_unit_id"], errors="ignore")


def _add_unit_type(enriched_df: pd.DataFrame) -> pd.DataFrame:
    unit_type_source = None
    for column in ("mapping_unit_type", "label_UnitType", "UnitType"):
        if column in enriched_df.columns:
            unit_type_source = column
            break

    if unit_type_source is None:
        enriched_df["unit_type"] = "unknown"
        return enriched_df

    enriched_df["unit_type"] = enriched_df[unit_type_source].map(_normalize_unit_type)
    return enriched_df


def load_multi_session_unit_metrics(selected_sessions_df: pd.DataFrame) -> pd.DataFrame:
    """Return one enriched unit-metrics table across all selected sessions."""
    rows: list[pd.DataFrame] = []
    if selected_sessions_df.empty:
        return pd.DataFrame()

    for session_index, session_row in selected_sessions_df.iterrows():
        session_dir = _first_existing_value(session_row, ("current_dir", "Current Dir", "RecordingDir"))
        if not isinstance(session_dir, str) or not session_dir.strip() or not os.path.isdir(session_dir):
            continue

        metadata = _session_metadata(session_row)
        for area in AREA_CONFIG:
            metrics_df = _load_metrics_csv(session_dir, area)
            if metrics_df is None or metrics_df.empty:
                continue

            enriched_df = _join_mapping(metrics_df, session_dir, area)
            enriched_df = _add_unit_type(enriched_df)
            enriched_df["session_index"] = session_index
            enriched_df["session_dir"] = session_dir
            enriched_df["brain_area"] = area

            for key, value in metadata.items():
                enriched_df[key] = value

            rows.append(enriched_df)

    if not rows:
        return pd.DataFrame()

    combined_df = pd.concat(rows, ignore_index=True, sort=False)
    combined_df["unit_global_id"] = (
        combined_df["session_index"].astype(str)
        + ":"
        + combined_df["brain_area"].astype(str)
        + ":"
        + combined_df["unit_idx"].astype(str)
    )
    return combined_df


# ---------------------------------------------------------------------------
# Selectivity / tuning-curve loader
# ---------------------------------------------------------------------------

@lru_cache(maxsize=512)
def _load_selectivity_csv(session_dir: str, area: str) -> Optional[pd.DataFrame]:
    metric_prefix = AREA_CONFIG[area]["metric_prefix"]
    path = os.path.join(
        session_dir,
        "analysis_output",
        "tables",
        f"{metric_prefix}_selectivity_metrics.csv",
    )
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path, low_memory=False)

    # Parse JSON-encoded list columns produced by the offline analysis pipeline.
    for col in ("tuning_curve_stimuli", "tuning_curve", "tuning_curve_sem"):
        if col not in df.columns:
            continue
        def _safe_parse(v: Any) -> Optional[list]:
            if pd.isna(v):
                return None
            try:
                parsed = ast.literal_eval(str(v))
                return parsed if isinstance(parsed, list) else None
            except (ValueError, SyntaxError):
                return None

        df[col] = df[col].map(_safe_parse)

    return df


def load_selectivity_data(selected_sessions_df: pd.DataFrame) -> pd.DataFrame:
    """Return one selectivity table (with parsed tuning curves) across all selected sessions.

    Enriches each row with mapping data (unit_type, histology_region, etc.) using
    the same pipeline as load_multi_session_unit_metrics so unit_type filters work.
    """
    rows: list[pd.DataFrame] = []
    if selected_sessions_df.empty:
        return pd.DataFrame()

    for session_index, session_row in selected_sessions_df.iterrows():
        session_dir = _first_existing_value(session_row, ("current_dir", "Current Dir", "RecordingDir"))
        if not isinstance(session_dir, str) or not session_dir.strip() or not os.path.isdir(session_dir):
            continue

        metadata = _session_metadata(session_row)
        for area in AREA_CONFIG:
            sel_df = _load_selectivity_csv(session_dir, area)
            if sel_df is None or sel_df.empty:
                continue

            sel_df = sel_df.copy()
            # Join mapping CSV so unit_type (good/mua/noise) is available for filtering
            sel_df = _join_mapping(sel_df, session_dir, area)
            sel_df = _add_unit_type(sel_df)

            sel_df["session_index"] = session_index
            sel_df["session_dir"] = session_dir
            sel_df["brain_area"] = area

            for key, value in metadata.items():
                sel_df[key] = value

            rows.append(sel_df)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True, sort=False)
