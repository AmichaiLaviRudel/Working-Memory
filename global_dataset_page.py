import os
import re
import traceback

import pandas as pd
import streamlit as st
import numpy as np

from load_data.params_extraction import compute_metrics_for_loaded_data
from Analysis.GNG_bpod_analysis.GNG_bpod_general import render_global_early_response_filter_checkbox


# Default path for global training CSV
DEFAULT_GLOBAL_CSV = r"Z:\Shared\Amichai\Code\DB\users_data\Amichai\global_training.csv"


def _has_metrics_columns(df: pd.DataFrame) -> bool:
    """Check if DataFrame already has computed performance metrics."""
    required_metrics = ["d_prime", "Hit_Rate", "False_Alarm_Rate"]
    return all(col in df.columns for col in required_metrics)


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
                    # Setup filter (Bpod, Educage, etc.)
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
                    available_groups = sorted(df["groupID"].dropna().unique().tolist()) if "groupID" in df.columns else []
                    if available_groups:
                        group_filter = st.multiselect(
                            "Group",
                            options=available_groups,
                            default=available_groups,
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
                        value=1.5,
                        step=0.1,
                        key="dashboard_dprime_threshold",
                    )
                
                with col4:
                    hit_rate_threshold = st.number_input(
                        "Hit Rate threshold",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.8,
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
                    # Extract group from MouseName (pattern: G{number}A{number}, e.g., G1A1, G5A2)
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

                    # Display dashboard metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Subjects", total_subjects)
                    with col2:
                        st.metric("Total Groups", total_groups)
                    with col3:
                        st.metric(
                            f"Achieved d' > {dprime_threshold} & Hit Rate > {hit_rate_threshold*100:.0f}%",
                            f"{n_achieved}",
                            delta=f"{pct_achieved:.1f}%",
                        )
                    with col4:
                        if total_subjects > 0:
                            st.metric(
                                "Success Rate",
                                f"{pct_achieved:.1f}%",
                            )

                    # Detailed breakdown by group
                    if total_groups > 1:
                        st.markdown("#### Breakdown by Group")
                        group_stats = []
                        for group in sorted(df_filtered["Group"].unique()):
                            group_data = df_filtered[df_filtered["Group"] == group]
                            n_in_group = group_data["MouseName"].nunique()
                            if "d_prime" in group_data.columns and "Hit_Rate" in group_data.columns:
                                group_achieved_mask = (
                                    (pd.to_numeric(group_data["d_prime"], errors="coerce") > dprime_threshold)
                                    & (pd.to_numeric(group_data["Hit_Rate"], errors="coerce") > hit_rate_threshold)
                                )
                                achieved_mice_in_group = group_data[group_achieved_mask]["MouseName"].unique()
                                achieved_in_group = len(achieved_mice_in_group)
                            else:
                                achieved_in_group = 0
                            group_stats.append({
                                "Group": group,
                                "N Subjects": n_in_group,
                                "Achieved Criteria": achieved_in_group,
                                "Success Rate (%)": (achieved_in_group / n_in_group * 100) if n_in_group > 0 else 0.0,
                            })
                        
                        group_df = pd.DataFrame(group_stats)
                        st.dataframe(group_df, use_container_width=True, hide_index=True)

                    # Detailed breakdown by Setup (Bpod vs Educage comparison)
                    if "Setup" in df_filtered.columns:
                        unique_setups = df_filtered["Setup"].dropna().unique()
                        if len(unique_setups) > 1:
                            st.markdown("#### Breakdown by Setup")
                            setup_stats = []
                            for setup in sorted(unique_setups):
                                setup_data = df_filtered[df_filtered["Setup"] == setup]
                                n_subjects_setup = setup_data["MouseName"].nunique()
                                n_sessions_setup = len(setup_data)
                                if "d_prime" in setup_data.columns and "Hit_Rate" in setup_data.columns:
                                    setup_achieved_mask = (
                                        (pd.to_numeric(setup_data["d_prime"], errors="coerce") > dprime_threshold)
                                        & (pd.to_numeric(setup_data["Hit_Rate"], errors="coerce") > hit_rate_threshold)
                                    )
                                    achieved_mice_setup = setup_data[setup_achieved_mask]["MouseName"].unique()
                                    achieved_setup = len(achieved_mice_setup)
                                    # Compute average d' for this setup
                                    avg_dprime = pd.to_numeric(setup_data["d_prime"], errors="coerce").mean()
                                    avg_hit_rate = pd.to_numeric(setup_data["Hit_Rate"], errors="coerce").mean()
                                else:
                                    achieved_setup = 0
                                    avg_dprime = 0.0
                                    avg_hit_rate = 0.0
                                setup_stats.append({
                                    "Setup": setup,
                                    "N Subjects": n_subjects_setup,
                                    "N Sessions": n_sessions_setup,
                                    "Achieved Criteria": achieved_setup,
                                    "Success Rate (%)": (achieved_setup / n_subjects_setup * 100) if n_subjects_setup > 0 else 0.0,
                                    "Avg d'": round(avg_dprime, 2),
                                    "Avg Hit Rate": round(avg_hit_rate, 2),
                                })
                            
                            setup_stats_df = pd.DataFrame(setup_stats)
                            st.dataframe(setup_stats_df, use_container_width=True, hide_index=True)

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
                    
                    # Show filter info including setup and group
                    filter_info = f"Showing {len(df_filtered)} filtered sessions (N_Boundaries: {boundaries_options}, Tones_per_class: {tones_options})"
                    if "Setup" in df_filtered.columns:
                        setups_in_filtered = df_filtered["Setup"].dropna().unique().tolist()
                        if len(setups_in_filtered) > 0:
                            filter_info += f", Setups: {setups_in_filtered}"
                    if "groupID" in df_filtered.columns:
                        groups_in_filtered = df_filtered["groupID"].dropna().unique().tolist()
                        if len(groups_in_filtered) > 0:
                            filter_info += f", Groups: {len(groups_in_filtered)}"
                    st.caption(filter_info)

                    # Display editable table for session selection (include Setup and groupID columns)
                    analysis_columns = ["Checkbox", "MouseName", "Group", "Setup", "groupID", "SessionDate", "d_prime", "Hit_Rate", "N_Boundaries", "Tones_per_class"]
                    available_cols = [c for c in analysis_columns if c in df_filtered.columns]

                    df_analysis_select = st.data_editor(
                        data=df_filtered[available_cols],
                        height=300,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Checkbox": st.column_config.CheckboxColumn(
                                "Analyse?", help="Select rows for analysis", default=False
                            ),
                            "Setup": st.column_config.Column(
                                "Setup", help="Setup type: Bpod or Educage", width="small"
                            ),
                            "groupID": st.column_config.Column(
                                "Group ID", help="Project/group identifier", width="medium"
                            ),
                        },
                        key="global_analysis_data_editor",
                    )

                    # Sync checkbox selections back to filtered df
                    df_filtered["Checkbox"] = df_analysis_select["Checkbox"].values

                    selected_indices = df_filtered.loc[df_filtered["Checkbox"] == True].index

                    if len(selected_indices) < 1:
                        st.info("Please select row(s) to start the analysis")
                    else:
                        # Show selection summary with setup and group breakdown
                        selected_data = df_filtered.loc[selected_indices]
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
                                gng_bpod_analysis(df_filtered, index)
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
                                        f"{df_filtered.iloc[idx]['MouseName']} - {df_filtered.iloc[idx].get('SessionDate', 'N/A')}": idx
                                        for idx in selected_indices
                                    }
                                    selected_session = st.selectbox(
                                        "Select session to analyze",
                                        options=list(session_options.keys()),
                                        key="global_single_session_select",
                                    )
                                    if selected_session:
                                        gng_bpod_analysis(df_filtered, session_options[selected_session])
                                
                                with tab_multi_animal:
                                    st.markdown("### Multi-Animal Analysis")
                                    st.caption(f"Analyzing {len(selected_indices)} selected sessions across multiple animals")
                                    gng_bpod_analysis_multi_animal(df_filtered, selected_indices)
                                
                                with tab_multi_session:
                                    st.markdown("### Multi-Session Analysis")
                                    st.caption(f"Analyzing {len(selected_indices)} selected sessions")
                                    gng_bpod_analysis_multi_session(df_filtered, selected_indices)

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
