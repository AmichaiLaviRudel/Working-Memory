import ast
import os
import re
import traceback

import pandas as pd
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

from load_data.params_extraction import compute_metrics_for_loaded_data
from Analysis.GNG_bpod_analysis.GNG_bpod_general import render_global_early_response_filter_checkbox, get_plotly_config
from Analysis.GNG_bpod_analysis import colors as plot_colors


def _global_dataset_plotly_config(filename_prefix: str) -> dict:
    """Plotly mode-bar config with SVG download (see ``get_plotly_config``)."""
    return get_plotly_config(filename_prefix)


# Default 2B boundaries (kHz) for region-wise stats
_LOW_BOUNDARY = 0.983
_HIGH_BOUNDARY = 1.525
_OCTAVE_DISTANCE_BINS = (0.25, 0.5, 0.75, 1.0)


def _rates_by_region_2b(df: pd.DataFrame, idx, low_bound: float, high_bound: float):
    """For one 2B session row: (hit_rate_below_low, middle_CR_rate, hit_rate_above_high)."""
    try:
        if "Stimuli" not in df.columns or "Outcomes" not in df.columns:
            return np.nan, np.nan, np.nan
        row = df.loc[idx]
        stim_raw, out_raw = row.get("Stimuli"), row.get("Outcomes")
        if pd.isna(stim_raw) or pd.isna(out_raw):
            return np.nan, np.nan, np.nan
        stimuli = np.array([float(x) for x in str(stim_raw).strip("[]\n").split()]) if isinstance(stim_raw, str) else np.asarray(stim_raw, dtype=float)
        outcomes = np.array(ast.literal_eval(out_raw)) if isinstance(out_raw, str) else np.asarray(out_raw)
        if len(stimuli) != len(outcomes) or len(stimuli) == 0:
            return np.nan, np.nan, np.nan
        out_str = np.array([str(o).strip() for o in outcomes])
        hit = out_str == "Hit"
        miss = out_str == "Miss"
        cr = (out_str == "CR") | (out_str == "Correct Reject")
        fa = out_str == "False Alarm"
        below = stimuli < low_bound
        middle = (stimuli >= low_bound) & (stimuli <= high_bound)
        above = stimuli > high_bound
        n_below = np.sum(below)
        n_mid = np.sum(middle)
        n_above = np.sum(above)
        hr_below = (np.sum(hit & below) / n_below) if n_below > 0 else np.nan
        hr_above = (np.sum(hit & above) / n_above) if n_above > 0 else np.nan
        cr_mid = np.sum(cr & middle)
        fa_mid = np.sum(fa & middle)
        mid_cr_rate = (cr_mid / (cr_mid + fa_mid)) if (cr_mid + fa_mid) > 0 else np.nan
        return float(hr_below), float(mid_cr_rate), float(hr_above)
    except Exception:
        return np.nan, np.nan, np.nan


# Default path for global training CSV
DEFAULT_GLOBAL_CSV = r"Z:\Shared\Amichai\Code\DB\users_data\Amichai\global_training.csv"


def _has_metrics_columns(df: pd.DataFrame) -> bool:
    """Check if DataFrame already has computed performance metrics."""
    required_metrics = ["d_prime", "Hit_Rate", "False_Alarm_Rate"]
    return all(col in df.columns for col in required_metrics)


def _parse_array_value(value, *, numeric: bool = False) -> np.ndarray:
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            return np.asarray(parsed, dtype=float if numeric else object)
        except Exception:
            return np.fromstring(value.strip("[]\n"), sep=" ", dtype=float) if numeric else np.array([])

    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value, dtype=float if numeric else object)

    return np.array([], dtype=float if numeric else object)


def _dprime_from_outcomes(outcomes: np.ndarray) -> float:
    outcome_labels = np.asarray([str(outcome).strip() for outcome in outcomes])
    hit_n = int(np.sum(outcome_labels == "Hit"))
    miss_n = int(np.sum(outcome_labels == "Miss"))
    fa_n = int(np.sum(outcome_labels == "False Alarm"))
    cr_n = int(np.sum(np.isin(outcome_labels, ["CR", "Correct Reject", "Correct Rejection"])))

    go_n = hit_n + miss_n
    nogo_n = fa_n + cr_n
    if go_n == 0 or nogo_n == 0:
        return np.nan

    # Avoid infinite z-scores when performance is exactly 0% or 100%.
    hit_rate = np.clip(hit_n / go_n, 1e-3, 1 - 1e-3)
    fa_rate = np.clip(fa_n / nogo_n, 1e-3, 1 - 1e-3)
    return float(norm.ppf(hit_rate) - norm.ppf(fa_rate))


def _pairwise_dprime_by_octave(
    df: pd.DataFrame,
    *,
    setup_col: str = "Setup",
    octave_bins: tuple[float, ...] = _OCTAVE_DISTANCE_BINS,
) -> pd.DataFrame:
    required_cols = {setup_col, "MouseName", "Stimuli", "Outcomes"}
    if not required_cols.issubset(df.columns):
        return pd.DataFrame()

    valid_bins = set(octave_bins)
    rows = []
    for _, row in df.iterrows():
        setup = row.get(setup_col)
        mouse_name = row.get("MouseName")
        stimuli = _parse_array_value(row.get("Stimuli"), numeric=True)
        outcomes = _parse_array_value(row.get("Outcomes"))
        if pd.isna(setup) or pd.isna(mouse_name) or len(stimuli) == 0 or len(stimuli) != len(outcomes):
            continue

        unique_stims = np.sort(np.unique(np.round(stimuli.astype(float), 6)))
        for i, lo in enumerate(unique_stims[:-1]):
            if lo <= 0:
                continue

            for hi in unique_stims[i + 1:]:
                octave_dist = round(float(np.log2(hi / lo)) * 4) / 4
                if octave_dist not in valid_bins:
                    continue

                pair_mask = np.isclose(stimuli, lo, atol=1e-6) | np.isclose(stimuli, hi, atol=1e-6)
                if int(np.sum(pair_mask)) < 5:
                    continue

                dprime = _dprime_from_outcomes(outcomes[pair_mask])
                if not np.isfinite(dprime):
                    continue

                rows.append(
                    {
                        "Setup": str(setup),
                        "MouseName": str(mouse_name),
                        "Octave Distance": float(octave_dist),
                        "d_prime": dprime,
                    }
                )

    return pd.DataFrame(rows)


def render_global_dataset_page() -> None:
    """
    Render the "Global dataset" view used by `Project.py`.

    Loads data from global_training.csv and computes performance metrics on demand.
    """
    st.title("Global Training Performance Dataset")

    # CSV path input
    csv_path = st.text_input(
        "Global training CSV path",
        value=DEFAULT_GLOBAL_CSV,
        help="Path to the global_training.csv file containing all session data.",
    )

    # Check if CSV exists
    if not os.path.exists(csv_path):
        st.error(f"CSV file not found: {csv_path}")
        st.info("Please ensure the global_training.csv file exists at the specified path.")
        return

    # Load CSV (auto-load on page render if not cached)
    cached_df = st.session_state.get("global_training_perf_df", None)
    cached_path = st.session_state.get("global_training_csv_path", None)

    # Reload if path changed or not cached
    if not isinstance(cached_df, pd.DataFrame) or cached_df.empty or cached_path != csv_path:
        try:
            df_loaded = pd.read_csv(csv_path, delimiter=",", low_memory=False)
            # Map legacy "Bpod" values to "Rig" in Setup column
            if "Setup" in df_loaded.columns:
                df_loaded["Setup"] = df_loaded["Setup"].replace("Bpod", "Rig")
            st.session_state["global_training_perf_df"] = df_loaded
            st.session_state["global_training_csv_path"] = csv_path
            st.success(f"Loaded {len(df_loaded)} sessions from: {csv_path}")
        except Exception as e:
            st.error(f"Failed to load CSV.\n\n{e}")
            return

    df = st.session_state.get("global_training_perf_df", None)
    if not isinstance(df, pd.DataFrame) or df.empty:
        st.warning("No data loaded.")
        return

    # Check if metrics need to be computed
    has_metrics = _has_metrics_columns(df)

    if not has_metrics:
        st.warning("Performance metrics (d', Hit Rate, FA Rate) not found in the dataset.")
        st.info("Click 'Compute Metrics' to calculate performance metrics for each session.")

    # Compute Metrics button
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Compute Metrics", key="compute_metrics_btn", type="primary" if not has_metrics else "secondary"):
            progress = st.progress(0.0)
            status = st.empty()

            # Drop duplicate rows (by session identity or full row)
            n_before = len(df)
            key_cols = [c for c in ["MouseName", "SessionDate", "SessionTime"] if c in df.columns]
            if key_cols:
                df = df.drop_duplicates(subset=key_cols, keep="first").reset_index(drop=True)
            else:
                df = df.drop_duplicates(keep="first").reset_index(drop=True)
            n_after = len(df)
            if n_before > n_after:
                st.session_state["global_training_perf_df"] = df
                st.info(f"Dropped {n_before - n_after} duplicate row(s). Proceeding with {n_after} sessions.")

            def _progress_cb(done: int, total: int, mouse_name: str) -> None:
                frac = float(done) / float(max(total, 1))
                progress.progress(frac)
                status.text(f"Computing metrics {done}/{total}: {mouse_name}")

            with st.spinner("Computing performance metrics..."):
                df = compute_metrics_for_loaded_data(
                    df,
                    output_path=csv_path,  # Save back to same CSV
                    progress_callback=_progress_cb,
                )

            progress.progress(1.0)
            status.text("Done.")
            st.session_state["global_training_perf_df"] = df
            st.success(f"Computed metrics for {len(df)} sessions and saved to: {csv_path}")
            st.rerun()

    with col2:
        if has_metrics:
            st.success("Performance metrics are available.")

    df = st.session_state.get("global_training_perf_df", None)
    if isinstance(df, pd.DataFrame) and not df.empty:
        # Arrow-safe preview: keep full DF for download/save, but avoid pyarrow failures
        # caused by nested arrays/objects (e.g. States, Licks, TrialTypes, Outcomes, Stimuli).
        preview_drop = [
            "States",
            "Licks",
            "TrialTypes",
            "Outcomes",
            "Stimuli",
            "Unique_Stimuli_Values",
        ]
        df_preview = df.drop(columns=preview_drop, errors="ignore")
        st.dataframe(df_preview, height=450, use_container_width=True)
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download global dataset (CSV)",
            data=csv_bytes,
            file_name="all_mice_training_performance_dataset.csv",
            mime="text/csv",
        )

        if st.button("Save CSV", key="save_global_training_perf_dataset"):
            try:
                df.to_csv(csv_path, index=False)
                st.success(f"Saved: {csv_path}")
            except Exception as e:  # noqa: BLE001
                st.error(f"Failed to save CSV.\n\n{e}")

        st.divider()
        st.subheader("Training Progress Dashboard")

        # Dashboard: Check if any sessions meet criteria (not just last session)
        if "MouseName" in df.columns and "N_Boundaries" in df.columns and "Tones_per_class" in df.columns:
            # Filter controls
            with st.expander("Filter Criteria", expanded=True):
                # Row 1: Setup and Group filters
                st.markdown("**Data Filters**")
                col_setup, col_group = st.columns(2)
                
                with col_setup:
                    # Setup filter (Rig, Educage, etc.)
                    available_setups = sorted(df["Setup"].dropna().unique().tolist()) if "Setup" in df.columns else []
                    if available_setups:
                        setup_filter = st.multiselect(
                            "Setup",
                            options=available_setups,
                            default=available_setups,
                            key="dashboard_setup_filter",
                            help="Filter by setup type (e.g., Bpod, Educage)",
                        )
                    else:
                        setup_filter = None
                
                with col_group:
                    # Group filter (groupID column)
                    # Exclude pilot/test groups by default
                    EXCLUDED_GROUPS = {"Educage_headbar_pilot_15_12_2025", "Group 1", "Group 2", "Group 8", "Educage_headbar1_04_01_2026"}
                    available_groups = sorted(df["groupID"].dropna().unique().tolist()) if "groupID" in df.columns else []
                    default_groups = [g for g in available_groups if g not in EXCLUDED_GROUPS]
                    if available_groups:
                        group_filter = st.multiselect(
                            "Group",
                            options=available_groups,
                            default=default_groups,
                            key="dashboard_group_filter",
                            help="Filter by group/project ID",
                        )
                    else:
                        group_filter = None
                
                st.markdown("**Training Parameters**")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    boundaries_options = st.multiselect(
                        "N_Boundaries",
                        options=[1, 2],
                        default=[1, 2],
                        key="dashboard_boundaries",
                    )
                
                with col2:
                    tones_options = st.multiselect(
                        "Tones_per_class",
                        options=[1, 2, 3, 4],
                        default=[3, 4],
                        key="dashboard_tones",
                    )
                
                with col3:
                    dprime_threshold = st.number_input(
                        "d' threshold",
                        min_value=0.0,
                        value=1.0,
                        step=0.1,
                        key="dashboard_dprime_threshold",
                    )
                
                with col4:
                    hit_rate_threshold = st.number_input(
                        "Hit Rate threshold",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.75,
                        step=0.05,
                        key="dashboard_hit_rate_threshold",
                    )
                
                # Color coding option
                st.markdown("**Visualization**")
                color_by = st.radio(
                    "Color plots by",
                    options=["Setup", "Group", "None"],
                    horizontal=True,
                    key="dashboard_color_by",
                    help="Choose how to color-code the visualizations",
                )

            # Apply filters
            if boundaries_options and tones_options:
                # Filter by boundaries and tones
                training_mask = (
                    df["N_Boundaries"].isin(boundaries_options) & 
                    df["Tones_per_class"].isin(tones_options)
                )
                # Filter by Setup if available
                if setup_filter and "Setup" in df.columns:
                    training_mask = training_mask & df["Setup"].isin(setup_filter)
                # Filter by Group if available
                if group_filter and "groupID" in df.columns:
                    training_mask = training_mask & df["groupID"].isin(group_filter)
                
                df_filtered = df[training_mask].copy().reset_index(drop=True)

                if len(df_filtered) > 0:
                    # Use groupID column for group info (fallback to extracting from MouseName if not available)
                    if "groupID" in df_filtered.columns:
                        df_filtered["Group"] = df_filtered["groupID"].fillna("Unknown")
                    else:
                        # Fallback: extract group from MouseName (pattern: G{number}A{number})
                        def extract_group(mouse_name):
                            if pd.isna(mouse_name):
                                return None
                            match = re.match(r"G(\d+)A\d+", str(mouse_name), re.IGNORECASE)
                            return f"Group {match.group(1)}" if match else "Unknown"
                        df_filtered["Group"] = df_filtered["MouseName"].apply(extract_group)
                    
                    # Store color_by preference in session state for use in analysis functions
                    st.session_state["global_color_by"] = color_by

                    # Count total subjects and groups (all subjects with matching boundary/tones sessions)
                    total_subjects = df_filtered["MouseName"].nunique()
                    total_groups = df_filtered["Group"].nunique()

                    # Count subjects that have at least ONE session achieving d' > threshold AND Hit_Rate > threshold
                    if "d_prime" in df_filtered.columns and "Hit_Rate" in df_filtered.columns:
                        achieved_mask = (
                            pd.to_numeric(df_filtered["d_prime"], errors="coerce") > dprime_threshold
                        ) & (
                            pd.to_numeric(df_filtered["Hit_Rate"], errors="coerce") > hit_rate_threshold
                        )
                        # Get unique mice who achieved criteria (at least one session)
                        achieved_mice = df_filtered[achieved_mask]["MouseName"].unique()
                        n_achieved = len(achieved_mice)
                        pct_achieved = (n_achieved / total_subjects * 100) if total_subjects > 0 else 0.0
                    else:
                        n_achieved = 0
                        pct_achieved = 0.0
                        achieved_mice = []

                    # Calculate 1B and 2B specific metrics
                    df_1b = df_filtered[df_filtered["N_Boundaries"] == 1]
                    df_2b = df_filtered[df_filtered["N_Boundaries"] == 2]
                    
                    n_subjects_1b = df_1b["MouseName"].nunique() if len(df_1b) > 0 else 0
                    n_subjects_2b = df_2b["MouseName"].nunique() if len(df_2b) > 0 else 0
                    
                    # 1B success
                    if len(df_1b) > 0 and "d_prime" in df_1b.columns and "Hit_Rate" in df_1b.columns:
                        mask_1b = (
                            (pd.to_numeric(df_1b["d_prime"], errors="coerce") > dprime_threshold)
                            & (pd.to_numeric(df_1b["Hit_Rate"], errors="coerce") > hit_rate_threshold)
                        )
                        n_achieved_1b = len(df_1b[mask_1b]["MouseName"].unique())
                        pct_achieved_1b = (n_achieved_1b / n_subjects_1b * 100) if n_subjects_1b > 0 else 0.0
                    else:
                        n_achieved_1b = 0
                        pct_achieved_1b = 0.0
                    
                    # 2B success
                    if len(df_2b) > 0 and "d_prime" in df_2b.columns and "Hit_Rate" in df_2b.columns:
                        mask_2b = (
                            (pd.to_numeric(df_2b["d_prime"], errors="coerce") > dprime_threshold)
                            & (pd.to_numeric(df_2b["Hit_Rate"], errors="coerce") > hit_rate_threshold)
                        )
                        n_achieved_2b = len(df_2b[mask_2b]["MouseName"].unique())
                        pct_achieved_2b = (n_achieved_2b / n_subjects_2b * 100) if n_subjects_2b > 0 else 0.0
                    else:
                        n_achieved_2b = 0
                        pct_achieved_2b = 0.0

                    # Display dashboard metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Subjects", total_subjects)
                        st.metric("Total Groups", total_groups)
                    with col2:
                        st.metric(
                            f"Overall Success",
                            f"{n_achieved}/{total_subjects}",
                            delta=f"{pct_achieved:.1f}%",
                        )
                    with col3:
                        st.metric(
                            f"1B Success",
                            f"{n_achieved_1b}/{n_subjects_1b}" if n_subjects_1b > 0 else "-",
                            delta=f"{pct_achieved_1b:.1f}%" if n_subjects_1b > 0 else None,
                        )
                    with col4:
                        st.metric(
                            f"2B Success",
                            f"{n_achieved_2b}/{n_subjects_2b}" if n_subjects_2b > 0 else "-",
                            delta=f"{pct_achieved_2b:.1f}%" if n_subjects_2b > 0 else None,
                        )

                    # Detailed breakdown by Setup and Group with 1B/2B split
                    st.markdown("#### Breakdown by Setup & Group")
                    group_stats = []
                    
                    # Get unique setups (or use placeholder if not available)
                    unique_setups = sorted(df_filtered["Setup"].dropna().unique()) if "Setup" in df_filtered.columns else ["All"]
                    
                    for setup in unique_setups:
                        if "Setup" in df_filtered.columns:
                            setup_data = df_filtered[df_filtered["Setup"] == setup]
                        else:
                            setup_data = df_filtered
                        
                        for group in sorted(setup_data["Group"].unique()):
                            group_data = setup_data[setup_data["Group"] == group]
                            n_in_group = group_data["MouseName"].nunique()
                            
                            # Calculate success rate for 1B sessions
                            group_1b = group_data[group_data["N_Boundaries"] == 1]
                            n_mice_1b = group_1b["MouseName"].nunique() if len(group_1b) > 0 else 0
                            if len(group_1b) > 0 and "d_prime" in group_1b.columns and "Hit_Rate" in group_1b.columns:
                                mask_1b = (
                                    (pd.to_numeric(group_1b["d_prime"], errors="coerce") > dprime_threshold)
                                    & (pd.to_numeric(group_1b["Hit_Rate"], errors="coerce") > hit_rate_threshold)
                                )
                                achieved_1b = len(group_1b[mask_1b]["MouseName"].unique())
                                success_rate_1b = (achieved_1b / n_mice_1b * 100) if n_mice_1b > 0 else 0.0
                            else:
                                achieved_1b = 0
                                success_rate_1b = 0.0
                            
                            # Calculate success rate for 2B sessions
                            group_2b = group_data[group_data["N_Boundaries"] == 2]
                            n_mice_2b = group_2b["MouseName"].nunique() if len(group_2b) > 0 else 0
                            if len(group_2b) > 0 and "d_prime" in group_2b.columns and "Hit_Rate" in group_2b.columns:
                                mask_2b = (
                                    (pd.to_numeric(group_2b["d_prime"], errors="coerce") > dprime_threshold)
                                    & (pd.to_numeric(group_2b["Hit_Rate"], errors="coerce") > hit_rate_threshold)
                                )
                                achieved_2b = len(group_2b[mask_2b]["MouseName"].unique())
                                success_rate_2b = (achieved_2b / n_mice_2b * 100) if n_mice_2b > 0 else 0.0
                            else:
                                achieved_2b = 0
                                success_rate_2b = 0.0
                            
                            # Overall success rate
                            if "d_prime" in group_data.columns and "Hit_Rate" in group_data.columns:
                                group_achieved_mask = (
                                    (pd.to_numeric(group_data["d_prime"], errors="coerce") > dprime_threshold)
                                    & (pd.to_numeric(group_data["Hit_Rate"], errors="coerce") > hit_rate_threshold)
                                )
                                achieved_in_group = len(group_data[group_achieved_mask]["MouseName"].unique())
                            else:
                                achieved_in_group = 0
                            
                            group_stats.append({
                                "Setup": setup,
                                "Group": group,
                                "N Subjects": n_in_group,
                                "Achieved (Total)": achieved_in_group,
                                "Success Rate (%)": round((achieved_in_group / n_in_group * 100) if n_in_group > 0 else 0.0, 1),
                                "1B Achieved": f"{achieved_1b}/{n_mice_1b}" if n_mice_1b > 0 else "-",
                                "1B Success (%)": round(success_rate_1b, 1) if n_mice_1b > 0 else "-",
                                "2B Achieved": f"{achieved_2b}/{n_mice_2b}" if n_mice_2b > 0 else "-",
                                "2B Success (%)": round(success_rate_2b, 1) if n_mice_2b > 0 else "-",
                            })
                    
                    group_df = pd.DataFrame(group_stats)
                    # Sort by Setup then Group for better readability
                    group_df = group_df.sort_values(["Setup", "Group"]).reset_index(drop=True)
                    st.dataframe(group_df, use_container_width=True, hide_index=True)

                    # Detailed breakdown by Setup (Rig vs Educage comparison) with 1B/2B split
                    if "Setup" in df_filtered.columns:
                        unique_setups = df_filtered["Setup"].dropna().unique()
                        if len(unique_setups) > 1:
                            st.markdown("#### Breakdown by Setup")
                            setup_stats = []
                            for setup in sorted(unique_setups):
                                setup_data = df_filtered[df_filtered["Setup"] == setup]
                                n_subjects_setup = setup_data["MouseName"].nunique()
                                n_sessions_setup = len(setup_data)
                                
                                # Overall success for this setup
                                if "d_prime" in setup_data.columns and "Hit_Rate" in setup_data.columns:
                                    setup_achieved_mask = (
                                        (pd.to_numeric(setup_data["d_prime"], errors="coerce") > dprime_threshold)
                                        & (pd.to_numeric(setup_data["Hit_Rate"], errors="coerce") > hit_rate_threshold)
                                    )
                                    achieved_setup = len(setup_data[setup_achieved_mask]["MouseName"].unique())
                                    avg_dprime = pd.to_numeric(setup_data["d_prime"], errors="coerce").mean()
                                    avg_hit_rate = pd.to_numeric(setup_data["Hit_Rate"], errors="coerce").mean()
                                else:
                                    achieved_setup = 0
                                    avg_dprime = 0.0
                                    avg_hit_rate = 0.0
                                
                                # 1B success for this setup
                                setup_1b = setup_data[setup_data["N_Boundaries"] == 1]
                                n_mice_1b = setup_1b["MouseName"].nunique() if len(setup_1b) > 0 else 0
                                if len(setup_1b) > 0 and "d_prime" in setup_1b.columns and "Hit_Rate" in setup_1b.columns:
                                    mask_1b = (
                                        (pd.to_numeric(setup_1b["d_prime"], errors="coerce") > dprime_threshold)
                                        & (pd.to_numeric(setup_1b["Hit_Rate"], errors="coerce") > hit_rate_threshold)
                                    )
                                    achieved_1b = len(setup_1b[mask_1b]["MouseName"].unique())
                                    success_rate_1b = (achieved_1b / n_mice_1b * 100) if n_mice_1b > 0 else 0.0
                                else:
                                    achieved_1b = 0
                                    success_rate_1b = 0.0
                                
                                # 2B success for this setup
                                setup_2b = setup_data[setup_data["N_Boundaries"] == 2]
                                n_mice_2b = setup_2b["MouseName"].nunique() if len(setup_2b) > 0 else 0
                                if len(setup_2b) > 0 and "d_prime" in setup_2b.columns and "Hit_Rate" in setup_2b.columns:
                                    mask_2b = (
                                        (pd.to_numeric(setup_2b["d_prime"], errors="coerce") > dprime_threshold)
                                        & (pd.to_numeric(setup_2b["Hit_Rate"], errors="coerce") > hit_rate_threshold)
                                    )
                                    achieved_2b = len(setup_2b[mask_2b]["MouseName"].unique())
                                    success_rate_2b = (achieved_2b / n_mice_2b * 100) if n_mice_2b > 0 else 0.0
                                else:
                                    achieved_2b = 0
                                    success_rate_2b = 0.0
                                
                                setup_stats.append({
                                    "Setup": setup,
                                    "N Subjects": n_subjects_setup,
                                    "N Sessions": n_sessions_setup,
                                    "Achieved (Total)": achieved_setup,
                                    "Success Rate (%)": round((achieved_setup / n_subjects_setup * 100) if n_subjects_setup > 0 else 0.0, 1),
                                    "1B Achieved": f"{achieved_1b}/{n_mice_1b}" if n_mice_1b > 0 else "-",
                                    "1B Success (%)": round(success_rate_1b, 1) if n_mice_1b > 0 else "-",
                                    "2B Achieved": f"{achieved_2b}/{n_mice_2b}" if n_mice_2b > 0 else "-",
                                    "2B Success (%)": round(success_rate_2b, 1) if n_mice_2b > 0 else "-",
                                    "Avg d'": round(avg_dprime, 2),
                                    "Avg Hit Rate": round(avg_hit_rate, 2),
                                })
                            
                            setup_stats_df = pd.DataFrame(setup_stats)
                            st.dataframe(setup_stats_df, use_container_width=True, hide_index=True)

                            # Statistical comparison of per-mouse success between setup groups.
                            # Mann-Whitney U compares the binary per-mouse success distributions.
                            if len(setup_stats_df) == 2:
                                setup_a, setup_b = sorted(unique_setups)
                                comparisons = []
                                from scipy import stats

                                for n_boundaries, label in [(1, "1B"), (2, "2B")]:
                                    setup_a_data = df_filtered[
                                        (df_filtered["Setup"] == setup_a) & (df_filtered["N_Boundaries"] == n_boundaries)
                                    ]
                                    setup_b_data = df_filtered[
                                        (df_filtered["Setup"] == setup_b) & (df_filtered["N_Boundaries"] == n_boundaries)
                                    ]

                                    n_a = int(setup_a_data["MouseName"].nunique())
                                    n_b = int(setup_b_data["MouseName"].nunique())
                                    if n_a == 0 or n_b == 0:
                                        comparisons.append(
                                            {
                                                "Task": label,
                                                f"{setup_a} Achieved": "-",
                                                f"{setup_b} Achieved": "-",
                                                "Test": "Insufficient data",
                                                "p-value": "-",
                                            }
                                        )
                                        continue

                                    if "d_prime" in df_filtered.columns and "Hit_Rate" in df_filtered.columns:
                                        mask_a = (
                                            (pd.to_numeric(setup_a_data["d_prime"], errors="coerce") > dprime_threshold)
                                            & (pd.to_numeric(setup_a_data["Hit_Rate"], errors="coerce") > hit_rate_threshold)
                                        )
                                        mask_b = (
                                            (pd.to_numeric(setup_b_data["d_prime"], errors="coerce") > dprime_threshold)
                                            & (pd.to_numeric(setup_b_data["Hit_Rate"], errors="coerce") > hit_rate_threshold)
                                        )
                                        # Per-mouse binary success: 1 if mouse achieved criteria in any session.
                                        status_a = (
                                            setup_a_data.assign(_achieved=mask_a.astype(int))
                                            .groupby("MouseName", as_index=True)["_achieved"]
                                            .max()
                                        )
                                        status_b = (
                                            setup_b_data.assign(_achieved=mask_b.astype(int))
                                            .groupby("MouseName", as_index=True)["_achieved"]
                                            .max()
                                        )
                                        achieved_a = int(status_a.sum())
                                        achieved_b = int(status_b.sum())
                                    else:
                                        achieved_a = 0
                                        achieved_b = 0
                                        status_a = pd.Series(dtype=int)
                                        status_b = pd.Series(dtype=int)

                                    if len(status_a) == 0 or len(status_b) == 0:
                                        test_name = "Insufficient data"
                                        p_text = "-"
                                    else:
                                        _, p_val = stats.mannwhitneyu(
                                            status_a.values,
                                            status_b.values,
                                            alternative="two-sided",
                                        )
                                        test_name = "Mann-Whitney U"
                                        p_text = f"{p_val:.4g}"

                                    comparisons.append(
                                        {
                                            "Task": label,
                                            f"{setup_a} Achieved": f"{achieved_a}/{n_a}",
                                            f"{setup_b} Achieved": f"{achieved_b}/{n_b}",
                                            "Test": test_name,
                                            "p-value": p_text,
                                        }
                                    )

                                st.markdown(f"##### Setup comparison tests ({setup_a} vs {setup_b})")
                                st.dataframe(pd.DataFrame(comparisons), use_container_width=True, hide_index=True)
                    
                    # Combined comparison: d' box plot + psychometric curves in 3 columns
                    if "d_prime" in df_filtered.columns and "Stimuli" in df_filtered.columns and "Outcomes" in df_filtered.columns:
                        # Determine grouping based on color_by selection
                        if color_by == "Group" and "Group" in df_filtered.columns:
                            group_col = "Group"
                            get_color_func = plot_colors.get_group_color
                        elif "Setup" in df_filtered.columns:
                            group_col = "Setup"
                            get_color_func = plot_colors.get_setup_color
                        else:
                            group_col = None
                        
                        if group_col:
                            st.markdown(f"#### Comparison by {group_col}")
                            
                            from Analysis.GNG_bpod_analysis.licking_and_outcome import preprocess_stimuli_outcomes, compute_lick_rate
                            from scipy import stats
                            
                            # Filter options
                            filter_col1, filter_col2 = st.columns(2)
                            with filter_col1:
                                psych_only_successful = st.checkbox(
                                    "Only successful sessions", 
                                    value=False,
                                    help=f"Include only sessions with d' > {dprime_threshold} and Hit Rate > {hit_rate_threshold*100:.0f}%",
                                    key="psych_only_successful"
                                )
                            with filter_col2:
                                psych_filter_stim_range = st.checkbox(
                                    "Filter stimulus range", 
                                    value=True,
                                    help="1B: 0.7-1.6 kHz, 2B: 0.7-2.1 kHz",
                                    key="psych_filter_stim_range"
                                )
                            
                            # Apply successful sessions filter for psychometric
                            if psych_only_successful:
                                df_psych = df_filtered[achieved_mask].copy().reset_index(drop=True)
                                st.caption(f"Using {len(df_psych)} successful sessions out of {len(df_filtered)} total")
                            else:
                                df_psych = df_filtered.copy()
                            
                            # Prepare data for d' box plot
                            df_plot = df_filtered[[group_col, "N_Boundaries", "d_prime"]].copy()
                            df_plot["d_prime"] = pd.to_numeric(df_plot["d_prime"], errors="coerce")
                            df_plot = df_plot.dropna(subset=["d_prime", group_col])
                            
                            unique_categories = sorted(df_plot[group_col].unique())
                            use_gray_lines = (group_col == "Group")
                            
                            # Create 3 columns: d' box plot | 1B psychometric | 2B psychometric
                            col_dprime, col_1b, col_2b = st.columns(3)
                            
                            # Column 1: d' box plot
                            with col_dprime:
                                st.caption("d' Distribution")
                                if len(df_plot) > 0:
                                    fig = go.Figure()
                                    
                                    for boundary in [1, 2]:
                                        boundary_label = f"{boundary}B"
                                        for i, category in enumerate(unique_categories):
                                            color = get_color_func(category, i)
                                            data = df_plot[(df_plot[group_col] == category) & (df_plot["N_Boundaries"] == boundary)]["d_prime"]
                                            
                                            if len(data) > 0:
                                                fig.add_trace(go.Box(
                                                    y=data,
                                                    x=[boundary_label] * len(data),
                                                    name=category,
                                                    marker_color=color,
                                                    boxmean=True,
                                                    legendgroup=category,
                                                    showlegend=(boundary == 1),
                                                ))
                                    
                                    fig.update_layout(
                                        title=None,
                                        yaxis_title="d'",
                                        xaxis_title="Task",
                                        boxmode="group",
                                        boxgap=0.15,
                                        boxgroupgap=0.25,
                                        showlegend=True,
                                        height=350,
                                        margin=dict(l=40, r=10, t=10, b=40),
                                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                                    )
                                    plot_colors.apply_standard_font_sizes(fig)
                                    fig.add_hline(y=dprime_threshold, line_dash="dash", line_color=plot_colors.COLOR_GRAY)
                                    
                                    st.plotly_chart(fig, use_container_width=True, config=_global_dataset_plotly_config(f"dprime_{group_col.lower()}_comparison"))

                            df_oct = _pairwise_dprime_by_octave(df_psych)
                            if not df_oct.empty:
                                st.markdown("##### d' by Octave Distance (Platform Comparison)")
                                df_oct_mouse = (
                                    df_oct.groupby(["Setup", "MouseName", "Octave Distance"], as_index=False)["d_prime"]
                                    .mean()
                                )
                                st.caption("Values are averaged per mouse within each platform and octave distance before plotting/testing.")

                                fig_oct = go.Figure()
                                setup_levels = sorted(df_oct_mouse["Setup"].unique())
                                for setup_idx, setup_name in enumerate(setup_levels):
                                    setup_color = plot_colors.get_setup_color(setup_name, setup_idx)
                                    setup_df = df_oct_mouse[df_oct_mouse["Setup"] == setup_name]

                                    for dist in _OCTAVE_DISTANCE_BINS:
                                        dvals = setup_df[np.isclose(setup_df["Octave Distance"], dist)]["d_prime"]
                                        if dvals.empty:
                                            continue

                                        fig_oct.add_trace(
                                            go.Box(
                                                y=dvals,
                                                x=[f"{dist:g}"] * len(dvals),
                                                name=setup_name,
                                                marker_color=setup_color,
                                                boxmean=True,
                                                legendgroup=setup_name,
                                                showlegend=(dist == _OCTAVE_DISTANCE_BINS[0]),
                                            )
                                        )

                                fig_oct.update_layout(
                                    title=None,
                                    yaxis_title="d'",
                                    xaxis_title="Octave Distance",
                                    boxmode="group",
                                    boxgap=0.15,
                                    boxgroupgap=0.2,
                                    showlegend=True,
                                    height=360,
                                    margin=dict(l=40, r=10, t=10, b=40),
                                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                                )
                                fig_oct.add_hline(y=dprime_threshold, line_dash="dash", line_color=plot_colors.COLOR_GRAY)
                                plot_colors.apply_standard_font_sizes(fig_oct)
                                st.plotly_chart(fig_oct, use_container_width=True, config=_global_dataset_plotly_config("dprime_by_octave_platforms"))

                                if len(setup_levels) == 2:
                                    setup_a, setup_b = setup_levels
                                    distance_tests = []

                                    for dist in _OCTAVE_DISTANCE_BINS:
                                        vals_a = df_oct_mouse[
                                            (df_oct_mouse["Setup"] == setup_a)
                                            & (np.isclose(df_oct_mouse["Octave Distance"], dist))
                                        ]["d_prime"].to_numpy()
                                        vals_b = df_oct_mouse[
                                            (df_oct_mouse["Setup"] == setup_b)
                                            & (np.isclose(df_oct_mouse["Octave Distance"], dist))
                                        ]["d_prime"].to_numpy()
                                        if vals_a.size == 0 or vals_b.size == 0:
                                            distance_tests.append(
                                                {
                                                    "Octave Distance": dist,
                                                    f"{setup_a} (n)": int(vals_a.size),
                                                    f"{setup_b} (n)": int(vals_b.size),
                                                    "Test": "Insufficient data",
                                                    "U": "-",
                                                    "p-value": "-",
                                                }
                                            )
                                            continue

                                        u_stat, p_val = stats.mannwhitneyu(vals_a, vals_b, alternative="two-sided")
                                        distance_tests.append(
                                            {
                                                "Octave Distance": dist,
                                                f"{setup_a} (n)": int(vals_a.size),
                                                f"{setup_b} (n)": int(vals_b.size),
                                                "Test": "Mann-Whitney U",
                                                "U": round(float(u_stat), 2),
                                                "p-value": f"{float(p_val):.4g}",
                                            }
                                        )

                                    st.markdown(f"##### Per-distance tests ({setup_a} vs {setup_b})")
                                    st.dataframe(pd.DataFrame(distance_tests), use_container_width=True, hide_index=True)
                                else:
                                    st.caption("Per-distance Mann-Whitney tests are shown when exactly 2 setups are present.")
                            else:
                                st.caption("No valid pairwise d' values for octave bins 0.25, 0.5, 0.75, 1.0.")
                            
                            # Columns 2 & 3: Psychometric curves for 1B and 2B (multi-animal style)
                            for col, boundary, boundary_label in [(col_1b, 1, "1B"), (col_2b, 2, "2B")]:
                                with col:
                                    df_boundary = df_psych[df_psych["N_Boundaries"] == boundary]
                                    st.caption(f"{boundary_label}: {len(df_boundary)} sessions")
                                    
                                    if len(df_boundary) == 0:
                                        st.info(f"No {boundary_label} sessions.")
                                        continue
                                    
                                    fig_psych = go.Figure()
                                    stim_min, stim_max = 0.7, (1.6 if boundary == 1 else 2.1)
                                    
                                    # Collect individual session data and per-category aggregated data
                                    all_stimuli, all_lick_rates = [], []  # For overall interpolation
                                    individual_traces = []  # (unique_stims, lick_rates, category)
                                    category_session_data = {cat: {"stimuli": [], "lick_rates": []} for cat in unique_categories}
                                    
                                    for cat_idx, category in enumerate(unique_categories):
                                        cat_sessions = df_boundary[df_boundary[group_col] == category]
                                        
                                        for idx in cat_sessions.index:
                                            try:
                                                stimuli, outcomes = preprocess_stimuli_outcomes(df_psych, idx)
                                                if len(stimuli) == 0 or len(outcomes) == 0:
                                                    continue
                                                unique_stims, lick_rates, _, _ = compute_lick_rate(stimuli, outcomes)
                                                
                                                # Filter stimulus range
                                                if psych_filter_stim_range:
                                                    mask = (unique_stims >= stim_min) & (unique_stims <= stim_max)
                                                    unique_stims, lick_rates = unique_stims[mask], lick_rates[mask]
                                                
                                                if len(unique_stims) < 2:
                                                    continue
                                                
                                                # Sort by stimulus
                                                sort_idx = np.argsort(unique_stims)
                                                unique_stims, lick_rates = unique_stims[sort_idx], lick_rates[sort_idx]
                                                
                                                individual_traces.append((unique_stims, lick_rates, category, cat_idx))
                                                all_stimuli.append(unique_stims)
                                                all_lick_rates.append(lick_rates)
                                                category_session_data[category]["stimuli"].append(unique_stims)
                                                category_session_data[category]["lick_rates"].append(lick_rates)
                                            except Exception:
                                                continue
                                    
                                    if len(all_stimuli) == 0:
                                        st.info(f"No valid data for {boundary_label}.")
                                        continue
                                    
                                    # Common stimuli for interpolation
                                    common_stimuli = np.array(sorted(set(np.concatenate(all_stimuli))))
                                    
                                    # Plot individual traces in gray (multi-animal style)
                                    for unique_stims, lick_rates, category, cat_idx in individual_traces:
                                        fig_psych.add_trace(go.Scatter(
                                            x=unique_stims, y=lick_rates,
                                            mode='lines+markers',
                                            line=dict(width=plot_colors.LINE_WIDTH_MEDIUM, color=plot_colors.COLOR_GRAY, shape='spline'),
                                            marker=dict(size=6, color=plot_colors.COLOR_GRAY, symbol='circle'),
                                            hovertemplate="Stimulus: %{x:.2f} kHz<br>Lick Rate: %{y:.2f}%<extra></extra>",
                                            showlegend=False
                                        ))
                                    
                                    # Plot average line per category (bold color) when comparing by Setup
                                    if not use_gray_lines:
                                        for cat_idx, category in enumerate(unique_categories):
                                            cat_data = category_session_data[category]
                                            if len(cat_data["stimuli"]) == 0:
                                                continue
                                            # Interpolate to common stimuli and compute mean
                                            interpolated = np.array([
                                                np.interp(common_stimuli, stims, rates)
                                                for stims, rates in zip(cat_data["stimuli"], cat_data["lick_rates"])
                                            ])
                                            avg_lick = np.mean(interpolated, axis=0)
                                            color = get_color_func(category, cat_idx)
                                            fig_psych.add_trace(go.Scatter(
                                                x=common_stimuli, y=avg_lick,
                                                mode='lines+markers',
                                                line=dict(width=plot_colors.LINE_WIDTH_THICK, color=color, shape='spline'),
                                                marker=dict(size=8, color=color, symbol='circle'),
                                                name=f"{category} (n={len(cat_data['stimuli'])})",
                                                hovertemplate="Stimulus: %{x:.2f} kHz<br>Avg Lick Rate: %{y:.2f}%<extra></extra>",
                                                showlegend=False
                                            ))
                                    else:
                                        # Overall average when comparing by Group (orange line)
                                        interpolated_all = np.array([
                                            np.interp(common_stimuli, stims, rates)
                                            for stims, rates in zip(all_stimuli, all_lick_rates)
                                        ])
                                        avg_lick = np.mean(interpolated_all, axis=0)
                                        fig_psych.add_trace(go.Scatter(
                                            x=common_stimuli, y=avg_lick,
                                            mode='lines+markers',
                                            line=dict(width=plot_colors.LINE_WIDTH_THICK, color=plot_colors.COLOR_ORANGE, shape='spline'),
                                            marker=dict(size=8, color=plot_colors.COLOR_ORANGE, symbol='circle'),
                                            name=f"Average (n={len(all_stimuli)})",
                                            hovertemplate="Stimulus: %{x:.2f} kHz<br>Avg Lick Rate: %{y:.2f}%<extra></extra>",
                                            showlegend=False
                                        ))
                                    
                                    # Add boundary lines
                                    if boundary == 2:
                                        low_bd = getattr(st.session_state, "low_boundary", 0.983)
                                        high_bd = getattr(st.session_state, "high_boundary", 1.525)
                                        fig_psych.add_vline(x=low_bd, line_dash="dash", line_color=plot_colors.COLOR_VERY_SUBTLE)
                                        fig_psych.add_vline(x=high_bd, line_dash="dash", line_color=plot_colors.COLOR_VERY_SUBTLE)
                                    else:
                                        fig_psych.add_vline(x=getattr(st.session_state, "low_boundary", 1.0), line_dash="dash", line_color=plot_colors.COLOR_VERY_SUBTLE)
                                    
                                    fig_psych.update_layout(
                                        title=None,
                                        xaxis=dict(title="Frequency [kHz] (log)", type="log", showgrid=True),
                                        yaxis=dict(title="Lick Rate (%)", range=[-5, 110]),
                                        height=350,
                                        margin=dict(l=40, r=10, t=10, b=50),
                                        showlegend=False,
                                        hovermode="x unified"
                                    )
                                    plot_colors.apply_standard_font_sizes(fig_psych)
                                    st.plotly_chart(fig_psych, use_container_width=True, config=_global_dataset_plotly_config(f"mean_psychometric_{boundary_label}"))
                            
                            # Statistical analysis: 1B and 2B each use Low Go / Middle NoGo / High Go; Bonferroni correction
                            with st.expander("Statistical Analysis (Mann-Whitney U, corrected)", expanded=False):
                                from statsmodels.stats.multitest import multipletests

                                # Use same "Only successful sessions" filter as psychometric section
                                df_for_stats = df_psych
                                low_b, high_b = _LOW_BOUNDARY, _HIGH_BOUNDARY
                                region_metrics = [
                                    ("Low Go", "Hit_Rate_Low_Go"),           # hit rate for stimuli < low boundary
                                    ("Middle NoGo", "CR_Rate_Middle_NoGo"),  # CR/(CR+FA) between boundaries
                                    ("High Go", "Hit_Rate_High_Go"),        # hit rate for stimuli > high boundary
                                ]
                                stat_rows: list[dict] = []
                                p_values_raw: list[float] = []

                                def add_region_comparisons(task_label: str, df_sessions: pd.DataFrame, df_source: pd.DataFrame) -> None:
                                    if df_sessions.empty or group_col not in df_sessions.columns:
                                        return
                                    hr_low, mid_cr, hr_high = [], [], []
                                    for idx in df_sessions.index:
                                        h_l, m_c, h_h = _rates_by_region_2b(df_source, idx, low_b, high_b)
                                        hr_low.append(h_l)
                                        mid_cr.append(m_c)
                                        hr_high.append(h_h)
                                    df_sessions = df_sessions.assign(
                                        Hit_Rate_Low_Go=hr_low,
                                        CR_Rate_Middle_NoGo=mid_cr,
                                        Hit_Rate_High_Go=hr_high,
                                    )
                                    for region_name, metric_col in region_metrics:
                                        df_r = df_sessions[[group_col, metric_col]].dropna(subset=[metric_col])
                                        cat_list_r = [c for c in unique_categories if (df_r[group_col] == c).sum() > 0]
                                        for j in range(len(cat_list_r)):
                                            for k in range(j + 1, len(cat_list_r)):
                                                cat_a, cat_b = cat_list_r[j], cat_list_r[k]
                                                da = df_r[df_r[group_col] == cat_a][metric_col].values
                                                db = df_r[df_r[group_col] == cat_b][metric_col].values
                                                if len(da) >= 3 and len(db) >= 3:
                                                    stat, p = stats.mannwhitneyu(da, db, alternative="two-sided")
                                                    p_values_raw.append(p)
                                                    r = 1 - (2 * stat) / (len(da) * len(db))
                                                    stat_rows.append({
                                                        "Task": task_label,
                                                        "Metric": region_name,
                                                        "Comparison": f"{cat_a} vs {cat_b}",
                                                        "n1": len(da), "n2": len(db),
                                                        "Median1": round(float(np.median(da)), 2),
                                                        "Median2": round(float(np.median(db)), 2),
                                                        "U": round(float(stat), 1),
                                                        "p_raw": p,
                                                        "r": round(r, 3),
                                                    })

                                # 1B and 2B: use numeric N_Boundaries so 2 vs 2.0 / "2" don't drop 2B sessions
                                n_bound = pd.to_numeric(df_for_stats["N_Boundaries"], errors="coerce")
                                df_1b = df_for_stats.loc[n_bound == 1].copy()
                                df_2b = df_for_stats.loc[n_bound == 2].copy()
                                add_region_comparisons("1B", df_1b, df_for_stats)
                                add_region_comparisons("2B", df_2b, df_for_stats)

                                if p_values_raw:
                                    _, p_adj, _, _ = multipletests(p_values_raw, method="bonferroni")
                                    for i, row in enumerate(stat_rows):
                                        row["p-adj (Bonferroni)"] = f"{p_adj[i]:.4f}"
                                        p = p_adj[i]
                                        row["Sig."] = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                                    out_df = pd.DataFrame(stat_rows)
                                    display_cols = ["Task", "Metric", "Comparison", "n1", "n2", "Median1", "Median2", "U", "p-adj (Bonferroni)", "r", "Sig."]
                                    st.dataframe(out_df[[c for c in display_cols if c in out_df.columns]], use_container_width=True, hide_index=True)
                                    st.caption("Metrics: Low Go = hit rate (stim < low bound); Middle NoGo = CR/(CR+FA) (between bounds); High Go = hit rate (stim > high bound). p-adj: Bonferroni across all comparisons. *** p<0.001, ** p<0.01, * p<0.05, ns = not significant.")
                                else:
                                    st.caption("No pairwise comparisons with ≥3 samples per group.")
                                if df_2b.empty:
                                    st.caption("No 2B sessions in current selection (or none with ≥3 per group per metric). Check 'Only successful sessions' and page filters (N_Boundaries).")

                    # Show list of subjects who achieved criteria (with all their matching sessions)
                    if n_achieved > 0:
                        with st.expander(f"Subjects who achieved d' > {dprime_threshold} & Hit Rate > {hit_rate_threshold*100:.0f}%", expanded=False):
                            achieved_sessions = df_filtered[achieved_mask].copy()
                            if "SessionDate" in achieved_sessions.columns:
                                achieved_sessions["SessionDate"] = pd.to_datetime(
                                    achieved_sessions["SessionDate"], errors="coerce"
                                )
                            display_cols = ["MouseName", "Group", "d_prime", "Hit_Rate", "N_Boundaries", "Tones_per_class"]
                            # Add Setup and groupID columns if available
                            if "Setup" in achieved_sessions.columns:
                                display_cols.insert(2, "Setup")
                            if "groupID" in achieved_sessions.columns:
                                display_cols.insert(3, "groupID")
                            if "SessionDate" in achieved_sessions.columns:
                                display_cols.append("SessionDate")
                            achieved_display = achieved_sessions[display_cols].copy()
                            sort_cols = ["Group", "MouseName"]
                            if "Setup" in achieved_display.columns:
                                sort_cols.insert(0, "Setup")
                            if "SessionDate" in achieved_display.columns:
                                sort_cols.append("SessionDate")
                            achieved_display = achieved_display.sort_values(by=sort_cols)
                            st.dataframe(achieved_display, use_container_width=True, hide_index=True)
                            st.caption(f"Showing all sessions where criteria were met (not just last session)")

                    # =============================================================
                    # Analysis Section (uses filtered data)
                    # =============================================================
                    st.divider()
                    st.title("Analysis")
                    render_global_early_response_filter_checkbox()

                    # Add Checkbox column if missing (required for row selection)
                    if "Checkbox" not in df_filtered.columns:
                        df_filtered.insert(0, "Checkbox", False)

                    st.markdown("#### Select Sessions for Analysis")

                    # Only include mice that achieved criteria in at least one session
                    criteria_only = st.checkbox(
                        "Only mice that achieved criteria",
                        value=True,
                        help=f"Keep only mice with ≥1 session where d' > {dprime_threshold} & Hit Rate > {hit_rate_threshold*100:.0f}%",
                        key="criteria_mice_only",
                    )
                    if criteria_only and len(achieved_mice) > 0:
                        df_for_analysis = df_filtered[df_filtered["MouseName"].isin(achieved_mice)].reset_index(drop=True)
                    else:
                        df_for_analysis = df_filtered.reset_index(drop=True)

                    # Show filter info including setup and group
                    filter_info = f"Showing {len(df_for_analysis)} sessions from {df_for_analysis['MouseName'].nunique()} mice (N_Boundaries: {boundaries_options}, Tones_per_class: {tones_options})"
                    if "Setup" in df_for_analysis.columns:
                        setups_in_filtered = df_for_analysis["Setup"].dropna().unique().tolist()
                        if len(setups_in_filtered) > 0:
                            filter_info += f", Setups: {setups_in_filtered}"
                    if "groupID" in df_for_analysis.columns:
                        groups_in_filtered = df_for_analysis["groupID"].dropna().unique().tolist()
                        if len(groups_in_filtered) > 0:
                            filter_info += f", Groups: {len(groups_in_filtered)}"
                    st.caption(filter_info)

                    # Display editable table for session selection (include Setup and groupID columns)
                    analysis_columns = [
                        "Checkbox", "MouseName", "Group", "Setup", "groupID", "SessionDate",
                        "d_prime", "Hit_Rate", "N_Boundaries", "Tones_per_class",
                        "Psychometric_x0", "Psychometric_x0_low", "Psychometric_x0_high",
                        "Psychometric_slope_low", "Psychometric_slope_high",
                        "Psychometric_r_squared", "Psychometric_fit_type",
                    ]
                    available_cols = [c for c in analysis_columns if c in df_for_analysis.columns]

                    df_analysis_select = st.data_editor(
                        data=df_for_analysis[available_cols],
                        height=300,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Checkbox": st.column_config.CheckboxColumn(
                                "Analyse?", help="Select rows for analysis", default=False
                            ),
                            "Setup": st.column_config.Column(
                                "Setup", help="Setup type: Rig or Educage", width="small"
                            ),
                            "groupID": st.column_config.Column(
                                "Group ID", help="Project/group identifier", width="medium"
                            ),
                        },
                        key="global_analysis_data_editor",
                    )

                    # Sync checkbox selections back to the analysis df
                    df_for_analysis["Checkbox"] = df_analysis_select["Checkbox"].values

                    selected_indices = df_for_analysis.loc[df_for_analysis["Checkbox"] == True].index

                    if len(selected_indices) < 1:
                        st.info("Please select row(s) to start the analysis")
                    else:
                        # Show selection summary with setup and group breakdown
                        selected_data = df_for_analysis.loc[selected_indices]
                        selection_msg = f"Selected {len(selected_indices)} session(s) for analysis"
                        
                        breakdown_parts = []
                        if "Setup" in selected_data.columns:
                            setup_counts = selected_data["Setup"].value_counts().to_dict()
                            if len(setup_counts) > 1:
                                setup_breakdown = ", ".join([f"{s}: {c}" for s, c in setup_counts.items()])
                                breakdown_parts.append(f"Setups: {setup_breakdown}")
                        if "groupID" in selected_data.columns:
                            group_counts = selected_data["groupID"].value_counts().to_dict()
                            if len(group_counts) > 1:
                                breakdown_parts.append(f"Groups: {len(group_counts)}")
                        
                        if breakdown_parts:
                            selection_msg += f" ({'; '.join(breakdown_parts)})"
                        st.success(selection_msg)

                        try:
                            from Analysis.GNG_bpod_analysis.GNG_Bpod_Analysis import (
                                gng_bpod_analysis,
                                gng_bpod_analysis_multi_animal,
                                gng_bpod_analysis_multi_session,
                            )

                            if len(selected_indices) == 1:
                                # Single session analysis
                                index = selected_indices[0]
                                st.markdown("### Single Session Analysis")
                                gng_bpod_analysis(df_for_analysis, index)
                            else:
                                # Multi-session analysis with tabs
                                tab_single, tab_multi_animal, tab_multi_session = st.tabs([
                                    "Single Session",
                                    "Multi-Animal Analysis",
                                    "Multi-Session Analysis",
                                ])
                                
                                with tab_single:
                                    st.markdown("### Single Session Analysis")
                                    # Allow selecting one session from the selected ones
                                    session_options = {
                                        f"{df_for_analysis.loc[idx, 'MouseName']} - {df_for_analysis.loc[idx].get('SessionDate', 'N/A')}": idx
                                        for idx in selected_indices
                                    }
                                    selected_session = st.selectbox(
                                        "Select session to analyze",
                                        options=list(session_options.keys()),
                                        key="global_single_session_select",
                                    )
                                    if selected_session:
                                        gng_bpod_analysis(df_for_analysis, session_options[selected_session])
                                
                                with tab_multi_animal:
                                    st.markdown("### Multi-Animal Analysis")
                                    st.caption(f"Analyzing {len(selected_indices)} selected sessions across multiple animals")
                                    gng_bpod_analysis_multi_animal(df_for_analysis, selected_indices)
                                
                                with tab_multi_session:
                                    st.markdown("### Multi-Session Analysis")
                                    st.caption(f"Analyzing {len(selected_indices)} selected sessions")
                                    gng_bpod_analysis_multi_session(df_for_analysis, selected_indices)

                        except Exception as e:  # noqa: BLE001
                            st.error(f"Something went wrong in the analysis.\n\n{e}")
                            st.text(traceback.format_exc())

                else:
                    st.warning(f"No sessions found matching criteria: N_Boundaries in {boundaries_options}, Tones_per_class in {tones_options}")
            else:
                st.info("Please select at least one boundary option and one tones option.")
        else:
            st.warning("Missing required columns (MouseName, N_Boundaries, Tones_per_class) for dashboard.")

        st.divider()
