"""
Data loading functions for NPXL offline analysis.
"""
from __future__ import annotations

import glob
import os
import numpy as np
import pandas as pd
from Analysis.NPXL_analysis.NPXL_Preprocessing import load_event_windows_data
from Analysis.NPXL_analysis.single_unit_dataset import (
    AREA_CONFIG,
    _add_unit_type,
    _join_mapping,
)

# Normalized cortex_group / histology_region tokens that count as the target probe area.
_AREA_HISTOLOGY_TOKENS: dict[str, frozenset[str]] = {
    "ACx": frozenset({"acx", "auditory", "auditory cortex", "a1", "tea"}),
    "OFC": frozenset({"ofc", "orbitofrontal", "orbitofrontal cortex", "lo", "vmo", "mofc"}),
}


def _event_windows_npy_path(data_dir: str) -> str:
    return os.path.join(data_dir, "analysis_output", "event_windows_matrix.npy")


def _diagnose_missing_event_windows(data_dir_OFC: str, data_dir_ACx: str) -> str:
    """List what is present under each probe's analysis_output (helps debug pipeline skips)."""
    lines: list[str] = []
    for label, d in (("OFC probe (imec1)", data_dir_OFC), ("ACx probe (imec0)", data_dir_ACx)):
        ao = os.path.join(d, "analysis_output")
        if not os.path.exists(ao):
            lines.append(f"  {label}: missing folder {ao}")
            continue
        try:
            names = sorted(os.listdir(ao))
            preview = names[:25]
            extra = f" (+{len(names) - len(preview)} more)" if len(names) > len(preview) else ""
            lines.append(f"  {label}: {ao} contains: {preview}{extra}")
        except OSError as exc:
            lines.append(f"  {label}: could not list {ao}: {exc}")
    return "\n".join(lines)


def load_data(data_dir_parent=None, data_dir_OFC=None, data_dir_ACx=None):
    """
    Load event windows data for OFC and ACx.
    
    Parameters:
    -----------
    data_dir_parent : str, optional
        Parent directory path
    data_dir_OFC : str, optional
        OFC data directory path
    data_dir_ACx : str, optional
        ACx data directory path
    
    Returns:
    --------
    tuple : (OFC_all, ACx_all, data_dir_OFC, data_dir_ACx)

    If one probe has no ``event_windows_matrix.npy`` (e.g. preprocessing was only run on
    the other shank), the missing side returns ``(None, None)`` for the array and directory
    so callers can run single-probe analysis. A clear warning is printed.
    """
    
    # Auto-detect OFC and ACx directories based on parent directory
    if data_dir_OFC is None or data_dir_ACx is None:
        if not os.path.exists(data_dir_parent):
            raise FileNotFoundError(f"Parent directory not found: {data_dir_parent}")
        
        # Find directories ending with imec1 (OFC) and imec0 (ACx)
        subdirs = [d for d in os.listdir(data_dir_parent) 
                   if os.path.isdir(os.path.join(data_dir_parent, d))]
        
        if data_dir_OFC is None:
            ofc_dirs = [d for d in subdirs if d.endswith('imec1')]
            if not ofc_dirs:
                raise FileNotFoundError(f"No directory ending with 'imec1' found in {data_dir_parent}")
            if len(ofc_dirs) > 1:
                raise ValueError(f"Multiple directories ending with 'imec1' found: {ofc_dirs}")
            data_dir_OFC = os.path.join(data_dir_parent, ofc_dirs[0])
        
        if data_dir_ACx is None:
            acx_dirs = [d for d in subdirs if d.endswith('imec0')]
            if not acx_dirs:
                raise FileNotFoundError(f"No directory ending with 'imec0' found in {data_dir_parent}")
            if len(acx_dirs) > 1:
                raise ValueError(f"Multiple directories ending with 'imec0' found: {acx_dirs}")
            data_dir_ACx = os.path.join(data_dir_parent, acx_dirs[0])
    
    ofc_path = _event_windows_npy_path(data_dir_OFC)
    acx_path = _event_windows_npy_path(data_dir_ACx)
    ofc_ok = os.path.exists(ofc_path)
    acx_ok = os.path.exists(acx_path)

    if ofc_ok and acx_ok:
        OFC_all = np.load(ofc_path)
        ACx_all = np.load(acx_path)
        print(f"\nLoaded data:")
        print(f"  OFC_all shape: {OFC_all.shape}")
        print(f"  ACx_all shape: {ACx_all.shape}")
        return OFC_all, ACx_all, data_dir_OFC, data_dir_ACx

    if acx_ok and not ofc_ok:
        print(
            "\nWARNING: OFC (imec1) event windows not found — running ACx-only analysis.\n"
            f"  Expected: {ofc_path}\n"
            "  To analyze OFC as well, run NPXL preprocessing / event-window export for imec1.\n"
        )
        ACx_all = np.load(acx_path)
        print(f"\nLoaded data (ACx only):")
        print(f"  ACx_all shape: {ACx_all.shape}")
        return None, ACx_all, None, data_dir_ACx

    if ofc_ok and not acx_ok:
        print(
            "\nWARNING: ACx (imec0) event windows not found — running OFC-only analysis.\n"
            f"  Expected: {acx_path}\n"
            "  To analyze ACx as well, run NPXL preprocessing / event-window export for imec0.\n"
        )
        OFC_all = np.load(ofc_path)
        print(f"\nLoaded data (OFC only):")
        print(f"  OFC_all shape: {OFC_all.shape}")
        return OFC_all, None, data_dir_OFC, None

    msg = (
        "Event windows not found for either probe (need event_windows_matrix.npy under each "
        "probe's analysis_output).\n"
        f"  OFC: {ofc_path}\n"
        f"  ACx: {acx_path}\n"
        + _diagnose_missing_event_windows(data_dir_OFC, data_dir_ACx)
    )
    raise FileNotFoundError(msg)


def _resolve_probe_unit_labels_tsv(probe_dir: str) -> str:
    """
    Prefer legacy bombcell/unit_labels.tsv; else Kilosort dir
    ``<probe_dir>/imec*_ks4/cluster_bc_unitType.tsv`` (pipeline_output layout).
    """
    legacy = os.path.join(probe_dir, "bombcell", "unit_labels.tsv")
    if os.path.isfile(legacy):
        return legacy
    pattern = os.path.join(probe_dir, "imec*_ks4", "cluster_bc_unitType.tsv")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            "No unit labels found. Tried:\n"
            f"  {legacy}\n"
            f"  {pattern}"
        )
    if len(matches) == 1:
        return matches[0]
    # e.g. ..._imec1 -> prefer .../imec1_ks4/cluster_bc_unitType.tsv
    base = os.path.basename(os.path.normpath(probe_dir))
    for candidate in matches:
        ks_folder = os.path.basename(os.path.dirname(candidate))
        imec_part = ks_folder.removesuffix("_ks4")
        if base.endswith("_" + imec_part):
            return candidate
    return matches[0]


_BC_LABEL_TO_UNIT_TYPE: dict[str, int] = {
    "GOOD": 1,
    "NOISE": 0,
    "MUA": 2,
    "NON-SOMA": 3,
    "NONSOMA": 3,
    "NON-SOMATIC": 3,
}


def _labels_tsv_to_g_index_and_good_list(labels_path: str) -> tuple[pd.DataFrame, list[int]]:
    """
    Build a DataFrame row-aligned with event-matrix unit index (iloc) and list of good unit indices.

    Supports bombcell ``unit_labels.tsv`` (UnitType) and Kilosort ``cluster_bc_unitType.tsv``
    (cluster_id, bc_unitType).
    """
    raw = pd.read_csv(labels_path, header=0, sep="\t")
    fname = os.path.basename(labels_path)

    if fname == "cluster_bc_unitType.tsv" or (
        "cluster_id" in raw.columns and "bc_unitType" in raw.columns and "UnitType" not in raw.columns
    ):
        if "cluster_id" not in raw.columns or "bc_unitType" not in raw.columns:
            raise ValueError(
                f"Expected cluster_id and bc_unitType columns in {labels_path}, got {list(raw.columns)}"
            )

        def _bc_cell_to_code(v: object) -> float:
            if pd.isna(v):
                return np.nan
            key = str(v).strip().upper()
            return float(_BC_LABEL_TO_UNIT_TYPE[key]) if key in _BC_LABEL_TO_UNIT_TYPE else np.nan

        cid = raw["cluster_id"].astype(int)
        codes = raw["bc_unitType"].map(_bc_cell_to_code)
        max_id = int(cid.max())
        full = pd.DataFrame(index=pd.RangeIndex(max_id + 1))
        # Keep text labels (GOOD/MUA/...) in object dtype to avoid mixed-dtype assignment warnings.
        full["bc_unitType"] = pd.Series([None] * (max_id + 1), dtype="object")
        full["UnitType"] = np.nan
        full.loc[cid.values, "bc_unitType"] = raw["bc_unitType"].values
        full.loc[cid.values, "UnitType"] = codes.values
        good = full.index[full["UnitType"] == 1].tolist()
        return full, [int(i) for i in good]

    # Legacy bombcell unit_labels.tsv (and similar)
    df = raw
    if df.columns[0] == "unitID" and "UnitType" in df.columns:
        unit_type_mapping = {"NOISE": 0, "GOOD": 1, "MUA": 2, "NON-SOMA": 3}
        df = df.copy()
        if df["UnitType"].dtype == object:
            df["UnitType"] = df["UnitType"].map(
                lambda x: unit_type_mapping.get(str(x).strip().upper(), np.nan)
            )
        df.index = df["unitID"].astype(int)
        df.index.name = "cluster_index"
    if "UnitType" not in df.columns:
        raise ValueError(f"No UnitType (and not cluster_bc layout) in {labels_path}: columns={list(df.columns)}")
    good_mask = df["UnitType"] == 1
    good = df.index[good_mask].tolist()
    return df, [int(i) for i in good]


def _build_fallback_metrics_from_labels(probe_dir: str) -> pd.DataFrame:
    """
    Build a minimal all-units metrics table when offline analysis tables are missing.

    Why: downstream histology matching only needs event-matrix row index (unit_idx) and a
    stable per-cluster ID (label_unitID) to join against mapping_unit_id.
    """
    legacy_path = os.path.join(probe_dir, "bombcell", "unit_labels.tsv")
    ks4_pattern = os.path.join(probe_dir, "imec*_ks4", "cluster_bc_unitType.tsv")
    ks4_paths = sorted(glob.glob(ks4_pattern))

    candidate_paths: list[str] = []
    try:
        candidate_paths.append(_resolve_probe_unit_labels_tsv(probe_dir))
    except FileNotFoundError:
        pass
    candidate_paths.extend([legacy_path, *ks4_paths])
    candidate_paths = list(dict.fromkeys(candidate_paths))  # preserve order, remove duplicates

    parse_errors: list[str] = []
    best_noise_only_df: pd.DataFrame | None = None
    best_noise_only_path: str | None = None

    for labels_path in candidate_paths:
        if not os.path.isfile(labels_path):
            continue
        try:
            labels_df, _good_indices = _labels_tsv_to_g_index_and_good_list(labels_path)
        except Exception as exc:
            parse_errors.append(f"{labels_path}: {exc}")
            continue

        if "UnitType" not in labels_df.columns:
            parse_errors.append(f"{labels_path}: missing UnitType column after parsing")
            continue

        unit_type_codes = pd.to_numeric(labels_df["UnitType"], errors="coerce")
        # Preserve event-matrix ordering convention: GOOD -> MUA -> NON-SOMA -> NOISE -> other known codes.
        ordered_parts = [unit_type_codes[unit_type_codes == code] for code in (1, 2, 3, 0)]
        other_known = unit_type_codes[~unit_type_codes.isin([0, 1, 2, 3]) & unit_type_codes.notna()]
        ordered_codes = pd.concat([*ordered_parts, other_known], ignore_index=False)
        if ordered_codes.empty:
            parse_errors.append(f"{labels_path}: no parsable UnitType codes")
            continue

        fallback_df = pd.DataFrame(
            {
                "unit_idx": np.arange(len(ordered_codes), dtype=int),
                "label_unitID": ordered_codes.index.to_numpy(dtype=int),
                "label_UnitType": ordered_codes.to_numpy(dtype=float),
            }
        )

        has_good_mua_non_soma = bool(unit_type_codes.isin([1, 2, 3]).any())
        if has_good_mua_non_soma:
            if labels_path != candidate_paths[0]:
                print(
                    f"WARNING: fallback metrics used alternate labels source: {labels_path} "
                    f"(primary source lacked usable GOOD/MUA/NON-SOMA labels)."
                )
            return fallback_df

        if best_noise_only_df is None:
            best_noise_only_df = fallback_df
            best_noise_only_path = labels_path

    if best_noise_only_df is not None:
        print(
            f"WARNING: fallback metrics built from {best_noise_only_path} with no GOOD/MUA/NON-SOMA labels; "
            "only NOISE/other unit types were available."
        )
        return best_noise_only_df

    details = "\n".join(parse_errors) if parse_errors else "No readable label files were found."
    raise ValueError(
        f"Could not build fallback metrics for {probe_dir}. Tried candidate label files:\n"
        + "\n".join(f"  {p}" for p in candidate_paths)
        + f"\nDetails:\n{details}"
    )


def _histology_agrees_with_area(row: pd.Series, area: str) -> bool:
    """True when histology labels place the unit in the probe's target area (not the other cortex)."""
    area_l = area.strip().lower()
    allowed = _AREA_HISTOLOGY_TOKENS.get(area, frozenset({area_l}))

    cortex_group = row.get("cortex_group")
    if pd.notna(cortex_group) and str(cortex_group).strip():
        text = str(cortex_group).strip().lower()
        if text == area_l:
            return True
        return text in allowed or any(token in text for token in allowed)

    histology_region = row.get("histology_region")
    if pd.notna(histology_region) and str(histology_region).strip():
        text = str(histology_region).strip().lower()
        return text in allowed or any(token in text for token in allowed)

    return False


def load_histology_matched_unit_indices(
    session_dir: str,
    probe_dir: str,
    area: str,
    unit_types: tuple[int, ...] = (1, 2),
) -> tuple[list[int], pd.DataFrame]:
    """Event-matrix row indices that are good/MUA and histologically assigned to ``area``."""
    metric_prefix = AREA_CONFIG[area]["metric_prefix"]
    metrics_path = os.path.join(
        probe_dir,
        "analysis_output",
        "tables",
        f"{metric_prefix}_all_units_metrics.csv",
    )
    if os.path.exists(metrics_path):
        metrics_df = pd.read_csv(metrics_path, low_memory=False)
    else:
        metrics_df = _build_fallback_metrics_from_labels(probe_dir)
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        metrics_df.to_csv(metrics_path, index=False)
        print(
            f"WARNING: Missing {metrics_path}; built fallback metrics table from unit labels "
            f"({len(metrics_df)} rows) so histology mapping can proceed."
        )

    if "unit_idx" not in metrics_df.columns:
        raise ValueError(f"{metrics_path} has no unit_idx column (cannot align to event matrix rows).")

    enriched_df = _join_mapping(metrics_df, session_dir, area)
    enriched_df = _add_unit_type(enriched_df)

    allowed_types: list[str] = []
    if 1 in unit_types:
        allowed_types.append("good")
    if 2 in unit_types:
        allowed_types.append("mua")
    type_mask = enriched_df["unit_type"].astype(str).str.lower().isin(allowed_types)

    histology_mask = enriched_df.apply(lambda row: _histology_agrees_with_area(row, area), axis=1)
    matched_df = enriched_df.loc[type_mask & histology_mask].copy()
    matched_df["matrix_row"] = pd.to_numeric(matched_df["unit_idx"], errors="coerce")
    matched_df = matched_df.dropna(subset=["matrix_row"])
    matched_df["matrix_row"] = matched_df["matrix_row"].astype(int)

    unit_indices = sorted(matched_df["matrix_row"].unique().tolist())
    return unit_indices, matched_df


def load_unit_indices_by_type(probe_dir: str, unit_types: tuple[int, ...] = (1,)) -> list[int]:
    """Return event-matrix row positions matching Bombcell unit type codes."""
    labels_path = _resolve_probe_unit_labels_tsv(probe_dir)
    labels_df, _good_indices = _labels_tsv_to_g_index_and_good_list(labels_path)
    unit_type_codes = pd.to_numeric(labels_df["UnitType"], errors="coerce")
    ordered_codes = pd.concat(
        [unit_type_codes[unit_type_codes == code] for code in (1, 2, 3)],
        ignore_index=True,
    )
    keep_mask = ordered_codes.isin(unit_types)
    return [int(position) for position in np.flatnonzero(keep_mask.to_numpy())]


def load_unit_labels(
    data_dir_OFC: str | None,
    data_dir_ACx: str | None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, list[int], list[int]]:
    """
    Load unit labels for OFC and/or ACx. Either directory may be None for single-probe runs.

    Per-probe resolution (in order):
    - ``<probe_dir>/bombcell/unit_labels.tsv`` (legacy), or
    - ``<probe_dir>/imec*_ks4/cluster_bc_unitType.tsv`` (e.g. pipeline_output next to Kilosort).
    """
    if data_dir_OFC is None and data_dir_ACx is None:
        raise ValueError("At least one of data_dir_OFC, data_dir_ACx must be provided")

    ofc_g_index: pd.DataFrame | None = None
    acx_g_index: pd.DataFrame | None = None
    OFC_g: list[int] = []
    ACx_g: list[int] = []

    if data_dir_OFC is not None:
        ofc_labels = _resolve_probe_unit_labels_tsv(data_dir_OFC)
        ofc_g_index, OFC_g = _labels_tsv_to_g_index_and_good_list(ofc_labels)
        print("\nSuccessfully loaded unit labels for OFC")
        print(f"  OFC labels file: {ofc_labels}")
        print(f"  OFC labels shape: {ofc_g_index.shape}")
        print(f"  OFC good units: {len(OFC_g)}")

    if data_dir_ACx is not None:
        acx_labels = _resolve_probe_unit_labels_tsv(data_dir_ACx)
        acx_g_index, ACx_g = _labels_tsv_to_g_index_and_good_list(acx_labels)
        print("\nSuccessfully loaded unit labels for ACx")
        print(f"  ACx labels file: {acx_labels}")
        print(f"  ACx labels shape: {acx_g_index.shape}")
        print(f"  ACx good units: {len(ACx_g)}")

    return ofc_g_index, acx_g_index, OFC_g, ACx_g


def read_event_windows_metadata(data_dir_x: str) -> dict:
    """
    Read event windows metadata from the analysis_output folder of a given probe directory.
    """
    metadata_path = os.path.join(data_dir_x, "analysis_output", "event_windows_metadata.txt")
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    metadata: dict[str, str] = {}
    with open(metadata_path, "r") as f:
        for line in f:
            if ":" not in line:
                continue
            key, value = line.strip().split(": ", 1)
            metadata[key] = value

    # Parse and cast to numeric types
    n_units = int(float(metadata.get("n_units", 0)))
    n_time_bins = int(float(metadata.get("n_time_bins", 0)))
    n_events = int(float(metadata.get("n_events", 0)))
    window_duration = float(metadata.get("window_duration", 0.0))
    bin_size = float(metadata.get("bin_size", 0.0))

    print(f"n_units: {n_units}")
    print(f"n_time_bins: {n_time_bins}")
    print(f"n_events: {n_events}")
    print(f"window_duration: {window_duration}")
    print(f"bin_size: {bin_size}")

    return {
        "n_units": n_units,
        "n_time_bins": n_time_bins,
        "n_events": n_events,
        "window_duration": window_duration,
        "bin_size": bin_size,
    }


def load_full_event_windows_data(data_dir_x: str):
    """
    Load full event windows data including stimuli/outcome information.
    
    Returns:
        tuple: (event_windows_matrix, time_axis, valid_event_indices, 
                stimuli_outcome_df, metadata, lick_event_windows_matrix)
    """
    analysis_output_dir = os.path.join(data_dir_x, "analysis_output")
    return load_event_windows_data(analysis_output_dir)


def load_behavioral_data(folder: str):
    analysis_output_dir = os.path.join(folder, "analysis_output")
     # Load the licking event windows matrix if it exists
    lick_file_path = os.path.join(analysis_output_dir, "lick_event_windows_matrix.npy")
    if os.path.exists(lick_file_path):
        lick_event_windows_matrix = np.load(lick_file_path)
    else:
        lick_event_windows_matrix = None
    
    # Load the time axis
    time_axis = np.load(os.path.join(analysis_output_dir, "event_window_time_axis.npy"))
    
    # Load the valid event indices
    valid_event_indices = np.load(os.path.join(analysis_output_dir, "valid_event_indices.npy"))
    
    # Load the filtered stimuli_outcome DataFrame
    stimuli_outcome_df = pd.read_csv(os.path.join(analysis_output_dir, "event_windows_stimuli_outcome.csv"))
    
    # Load metadata
    metadata = {}
    metadata_file = os.path.join(analysis_output_dir, "event_windows_metadata.txt")
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            for line in f:
                key, value = line.strip().split(': ')
                metadata[key] = value

    return lick_event_windows_matrix, time_axis, valid_event_indices, stimuli_outcome_df, metadata