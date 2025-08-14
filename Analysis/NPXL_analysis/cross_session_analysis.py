import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from Analysis.NPXL_analysis.npxl_single_unit_analysis import (
    compute_psth_pvalues_from_event_windows,
    compute_stimulus_selectivity,
    compute_go_nogo_coding,
    compute_choice_probability,
)
from Analysis.GNG_bpod_analysis.colors import COLOR_ACCENT, COLOR_ORANGE
from load_data.load_bpod_data import find_mat_files_in_session_data, load_mat_file
from plotly.subplots import make_subplots


def render_across_sessions_panel(project_df: pd.DataFrame) -> None:
    """
    Render the Across Sessions (Same Animal) comparison UI and plots.

    Args:
        project_df: DataFrame from the monitoring table (edited data).
    """
    st.divider()
    st.header("Across Sessions (Same Animal)")

    if len(project_df) == 0:
        st.info("No data available.")
        return

    # Auto-detect animal id column
    possible_animal_cols = [
        c for c in project_df.columns
        if c.lower() in ["animal", "animal_id", "mouse", "mouse_id", "subject", "subject_id"]
    ]
    animal_col = st.selectbox(
        "Animal ID column",
        options=possible_animal_cols if len(possible_animal_cols) > 0 else list(project_df.columns),
        index=0 if len(possible_animal_cols) > 0 else 0,
        help="Column that identifies the animal"
    )

    animal_values = sorted([str(v) for v in project_df[animal_col].dropna().unique()])
    if len(animal_values) == 0:
        st.info("No animals detected in the table.")
        return

    animal_id = st.selectbox("Select animal", options=animal_values, index=0)

    # Filter rows for this animal
    animal_df = project_df[project_df[animal_col].astype(str) == animal_id].copy()
    if len(animal_df) == 0:
        st.warning("No sessions found for this animal.")
        return
    st.dataframe(animal_df)
    # Determine session order (behavioral .mat-driven if available)
    order_cols_numeric = [c for c in animal_df.columns if c.lower() in ["session_order", "order", "session_index"]]
    order_cols_date = [c for c in animal_df.columns if c.lower() in ["date", "session_date", "recording_date"]]

    behavior_order_col = "__behavior_order__"
    animal_df[behavior_order_col] = np.nan
    for ridx, sess_row in animal_df.iterrows():
        best_ts = None
        # Prefer explicit behavioral file from table if provided
        beh_path = sess_row.get('behavioral file', None)
        if isinstance(beh_path, str) and len(beh_path) > 0 and os.path.exists(beh_path):
            try:
                _, _, session_date, session_time, *_ = load_mat_file(beh_path)
                date_str = str(session_date[0]) if isinstance(session_date, (list, np.ndarray)) else str(session_date)
                time_str = str(session_time[0]) if isinstance(session_time, (list, np.ndarray)) else str(session_time)
                ts = pd.to_datetime(f"{date_str} {time_str}", errors='coerce')
                if pd.isna(ts):
                    ts = pd.to_datetime(date_str, errors='coerce')
                if not pd.isna(ts):
                    best_ts = ts.timestamp()
            except Exception:
                pass
        if best_ts is None:
            # Fallback to scanning current_dir for behavior files
            cur_dir = sess_row.get('current_dir', None)
            if isinstance(cur_dir, str) and os.path.isdir(cur_dir):
                try:
                    mat_files = find_mat_files_in_session_data(cur_dir)
                except Exception:
                    mat_files = []
                for mf in mat_files:
                    try:
                        _, _, session_date, session_time, *_ = load_mat_file(mf)
                        date_str = str(session_date[0]) if isinstance(session_date, (list, np.ndarray)) else str(session_date)
                        time_str = str(session_time[0]) if isinstance(session_time, (list, np.ndarray)) else str(session_time)
                        ts = pd.to_datetime(f"{date_str} {time_str}", errors='coerce')
                        if pd.isna(ts):
                            ts = pd.to_datetime(date_str, errors='coerce')
                        if pd.isna(ts):
                            continue
                        ts_val = ts.timestamp()
                        if (best_ts is None) or (ts_val < best_ts):
                            best_ts = ts_val
                    except Exception:
                        continue
        if best_ts is None:
            # Last fallback to 'Date' column (day-first formats)
            date_val = sess_row.get('Date', None)
            if isinstance(date_val, str) and len(date_val) > 0:
                ts = pd.to_datetime(date_val, dayfirst=True, errors='coerce')
                if not pd.isna(ts):
                    best_ts = ts.timestamp()
        if best_ts is not None:
            animal_df.loc[ridx, behavior_order_col] = best_ts

    order_options = []
    if animal_df[behavior_order_col].notna().any():
        order_options.append("Behavior (from data)")
    order_options += order_cols_numeric + order_cols_date
    if len(order_options) == 0:
        order_options = ["Index"]

    selected_order = st.selectbox("Order sessions by", options=order_options, index=0)

    if selected_order == "Behavior (from data)":
        animal_df["__session_order__"] = pd.to_numeric(animal_df[behavior_order_col], errors='coerce')
    elif selected_order in order_cols_numeric:
        animal_df["__session_order__"] = pd.to_numeric(animal_df[selected_order], errors='coerce')
    elif selected_order in order_cols_date:
        animal_df["__session_order__"] = pd.to_datetime(animal_df[selected_order], errors='coerce')
    else:
        animal_df["__session_order__"] = np.arange(len(animal_df))

    # Sort by session order and drop FRA sessions (prefer explicit 'Session Type' column)
    animal_df = animal_df.sort_values(by="__session_order__")
    if 'Session Type' in animal_df.columns:
        animal_df = animal_df[~animal_df['Session Type'].astype(str).str.contains('FRA', case=False, na=False)]
    elif 'Session_type' in animal_df.columns:
        animal_df = animal_df[~animal_df['Session_type'].astype(str).str.contains('FRA', case=False, na=False)]
    elif 'current_dir' in animal_df.columns:
        animal_df = animal_df[~animal_df['current_dir'].astype(str).str.contains('FRA', case=False, na=False)]

    # Controls
    colA, colB, colC, colD = st.columns(4)
    with colA:
        alpha = st.number_input("α (significance)", min_value=0.001, max_value=0.2, value=0.05, step=0.001, format="%.3f")
    with colB:
        use_significant_only = st.checkbox("Significant units only", value=False)
    with colC:
        analysis_window = st.number_input("Analysis window (s)", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
    with colD:
        area_selection = st.selectbox("Area", options=["Both", "ACx", "OFC"], index=0)
    window = (-0.1, analysis_window)

    # Iterate sessions for this animal
    session_labels = []
    pct_significant = []
    median_cp = []
    median_dprime = []
    avg_best_fr = []
    # For best-stimulus distribution heatmap
    stim_values_all = set()
    per_session_best_stim_list = []
    # Per-session area-wise counts
    total_units_acx = []
    total_units_ofc = []
    sig_units_acx = []
    sig_units_ofc = []
    # Per-session area-wise metrics (when showing Both areas separately)
    pct_significant_acx = []
    pct_significant_ofc = []
    median_cp_acx = []
    median_cp_ofc = []
    median_dprime_acx = []
    median_dprime_ofc = []
    avg_best_fr_acx = []
    avg_best_fr_ofc = []
    per_session_best_stim_list_acx = []
    per_session_best_stim_list_ofc = []

    # Determine which column to use for session label
    if 'Session Type' in animal_df.columns:
        session_type_col = 'Session Type'
    elif 'Session_type' in animal_df.columns:
        session_type_col = 'Session_type'
    else:
        session_type_col = None

    for _, sess_row in animal_df.iterrows():
        # Use explicit Session Type if available; fallback to current_dir tail
        if session_type_col is not None:
            label = str(sess_row.get(session_type_col, '')).strip()
        else:
            cd = str(sess_row.get('current_dir', ''))
            label = os.path.basename(cd) if cd else ''
        session_labels.append(label)

        # Find analysis_output folder(s)
        current_dir = sess_row.get('current_dir', None)
        if not current_dir or not isinstance(current_dir, str) or not os.path.isdir(current_dir):
            pct_significant.append(np.nan)
            median_cp.append(np.nan)
            median_dprime.append(np.nan)
            avg_best_fr.append(np.nan)
            per_session_best_stim_list.append([])
            continue

        acx_dirs = []
        ofc_dirs = []
        for root, dirs, files in os.walk(current_dir):
            for d in dirs:
                if d == "analysis_output":
                    full_path = os.path.join(root, d)
                    parent_tail = os.path.basename(os.path.dirname(full_path))[-5:]
                    inferred_area = "ACx" if ("0" in parent_tail) else "OFC"
                    if inferred_area == "ACx":
                        acx_dirs.append(full_path)
                    else:
                        ofc_dirs.append(full_path)

        # Helper: compute metrics for a list of directories (area)
        def compute_session_metrics_for_dirs(dir_list):
            total_units_area = 0
            sig_units_area = 0
            cps_local = []
            dps_local = []
            best_fr_local = []
            best_stims_local = []
            for selected_folder in dir_list:
                try:
                    from Analysis.NPXL_analysis.NPXL_Preprocessing import load_event_windows_data
                    loaded_data = load_event_windows_data(selected_folder)
                except Exception:
                    loaded_data = None
                if not loaded_data:
                    continue
                event_windows_matrix, time_axis, valid_event_indices, stimuli_outcome_df, metadata = loaded_data[:5]
                event_times = stimuli_outcome_df['time'].values if 'time' in stimuli_outcome_df.columns else np.arange(event_windows_matrix.shape[2])
                pvals = compute_psth_pvalues_from_event_windows(
                    event_windows_matrix,
                    event_times,
                    bin_size=float(metadata.get('bin_size', 0.01)),
                    window=(-float(metadata.get('window_duration', 1.0)), float(metadata.get('window_duration', 1.0)))
                )
                sig_mask = np.isfinite(pvals) & (pvals < float(alpha))
                total_units_area += len(pvals)
                sig_units_area += int(np.sum(sig_mask))
                n_units = event_windows_matrix.shape[0]
                ew_data = (event_windows_matrix, time_axis, valid_event_indices, stimuli_outcome_df, metadata)
                for u in range(n_units):
                    if use_significant_only and not sig_mask[u]:
                        continue
                    cp_val, _ = compute_choice_probability(ew_data, stimuli_outcome_df, u, window)
                    if cp_val is not None and np.isfinite(cp_val):
                        cps_local.append(cp_val)
                    d_val, _, _ = compute_go_nogo_coding(ew_data, stimuli_outcome_df, u, window)
                    if d_val is not None and np.isfinite(d_val):
                        dps_local.append(d_val)
                    stimuli, tuning_curve, tuning_sem, best_stim = compute_stimulus_selectivity(ew_data, stimuli_outcome_df, u, window)
                    if stimuli is not None and best_stim is not None and tuning_curve is not None and len(tuning_curve) > 0:
                        stims_arr = np.array(stimuli)
                        tc_arr = np.array(tuning_curve)
                        try:
                            idx = int(np.where(stims_arr == best_stim)[0][0])
                            best_fr_local.append(float(tc_arr[idx]))
                            best_stims_local.append(float(best_stim))
                        except Exception:
                            pass
            median_cp_local = float(np.nanmedian(cps_local)) if len(cps_local) > 0 else np.nan
            median_dprime_local = float(np.nanmedian(dps_local)) if len(dps_local) > 0 else np.nan
            denom_local = sig_units_area if use_significant_only else total_units_area
            avg_best_fr_local = float(np.nanmean(best_fr_local)) if len(best_fr_local) > 0 and denom_local > 0 else np.nan
            pct_sig_local = 100.0 * (sig_units_area / total_units_area) if total_units_area > 0 else np.nan
            return (total_units_area, sig_units_area, pct_sig_local, median_cp_local, median_dprime_local, avg_best_fr_local, best_stims_local)

        # Compute per selection
        if area_selection == "ACx":
            tot, sig, pct, mcp, mdp, afr, bst = compute_session_metrics_for_dirs(acx_dirs)
            pct_significant.append(pct)
            median_cp.append(mcp)
            median_dprime.append(mdp)
            avg_best_fr.append(afr)
            per_session_best_stim_list.append(bst)
            total_units_acx.append(tot)
            total_units_ofc.append(0)
            sig_units_acx.append(sig)
            sig_units_ofc.append(0)
            stim_values_all.update([float(x) for x in bst])
        elif area_selection == "OFC":
            tot, sig, pct, mcp, mdp, afr, bst = compute_session_metrics_for_dirs(ofc_dirs)
            pct_significant.append(pct)
            median_cp.append(mcp)
            median_dprime.append(mdp)
            avg_best_fr.append(afr)
            per_session_best_stim_list.append(bst)
            total_units_acx.append(0)
            total_units_ofc.append(tot)
            sig_units_acx.append(0)
            sig_units_ofc.append(sig)
            stim_values_all.update([float(x) for x in bst])
        else:
            # Both: compute each area separately
            totA, sigA, pctA, mcpA, mdpA, afrA, bstA = compute_session_metrics_for_dirs(acx_dirs)
            totO, sigO, pctO, mcpO, mdpO, afrO, bstO = compute_session_metrics_for_dirs(ofc_dirs)
            total_units_acx.append(totA)
            total_units_ofc.append(totO)
            sig_units_acx.append(sigA)
            sig_units_ofc.append(sigO)
            pct_significant_acx.append(pctA)
            pct_significant_ofc.append(pctO)
            median_cp_acx.append(mcpA)
            median_cp_ofc.append(mcpO)
            median_dprime_acx.append(mdpA)
            median_dprime_ofc.append(mdpO)
            avg_best_fr_acx.append(afrA)
            avg_best_fr_ofc.append(afrO)
            per_session_best_stim_list_acx.append(bstA)
            per_session_best_stim_list_ofc.append(bstO)
            stim_values_all.update([float(x) for x in bstA])
            stim_values_all.update([float(x) for x in bstO])

    if len(session_labels) == 0:
        st.info("No non-FRA sessions to compare for this animal.")
        return

    # Build x tick positions for labeling session types
    x_idx = list(range(len(session_labels)))

    fig_summary = go.Figure()
    # Total units stacks
    fig_summary.add_trace(go.Bar(
        x=x_idx, y=total_units_acx, name='ACx total',
        marker_color=COLOR_ACCENT, offsetgroup='total', legendgroup='ACx'
    ))
    fig_summary.add_trace(go.Bar(
        x=x_idx, y=total_units_ofc, name='OFC total',
        marker_color=COLOR_ORANGE, offsetgroup='total', legendgroup='OFC'
    ))
    # Significant units stacks
    fig_summary.add_trace(go.Bar(
        x=x_idx, y=sig_units_acx, name='ACx significant',
        marker_color=COLOR_ACCENT, opacity=0.5, offsetgroup='sig', legendgroup='ACx'
    ))
    fig_summary.add_trace(go.Bar(
        x=x_idx, y=sig_units_ofc, name='OFC significant',
        marker_color=COLOR_ORANGE, opacity=0.5, offsetgroup='sig', legendgroup='OFC'
    ))
    fig_summary.update_layout(
        barmode='relative',
        title='Recording Summary: Units by Area (total vs significant)',
        xaxis_title='Session type (ordered)',
        yaxis_title='Units',
    )
    fig_summary.update_xaxes(tickmode='array', tickvals=x_idx, ticktext=session_labels)
    st.plotly_chart(fig_summary, use_container_width=True)

    # CP and d'
    fig_metrics = go.Figure()
    if area_selection == 'Both':
        fig_metrics.add_trace(go.Scatter(x=x_idx, y=median_cp_acx, mode='lines+markers', name='Median CP (ACx)', line=dict(color=COLOR_ACCENT)))
        fig_metrics.add_trace(go.Scatter(x=x_idx, y=median_cp_ofc, mode='lines+markers', name='Median CP (OFC)', line=dict(color=COLOR_ORANGE)))
        fig_metrics.add_trace(go.Scatter(x=x_idx, y=median_dprime_acx, mode='lines+markers', name="Median d' (ACx)", line=dict(color=COLOR_ACCENT, dash='dash')))
        fig_metrics.add_trace(go.Scatter(x=x_idx, y=median_dprime_ofc, mode='lines+markers', name="Median d' (OFC)", line=dict(color=COLOR_ORANGE, dash='dash')))
    else:
        fig_metrics.add_trace(go.Scatter(x=x_idx, y=median_cp, mode='lines+markers', name='Median CP'))
        fig_metrics.add_trace(go.Scatter(x=x_idx, y=median_dprime, mode='lines+markers', name="Median d'"))
    fig_metrics.update_layout(title=f"CP and d' across sessions for {animal_id}", xaxis_title="Session type (ordered)", yaxis_title="Value")
    fig_metrics.update_xaxes(tickmode='array', tickvals=x_idx, ticktext=session_labels)
    st.plotly_chart(fig_metrics, use_container_width=True)

    # Average best FR
    fig_fr = go.Figure()
    if area_selection == 'Both':
        fig_fr.add_trace(go.Scatter(x=x_idx, y=avg_best_fr_acx, mode='lines+markers', name='Avg best FR (ACx)', line=dict(color=COLOR_ACCENT)))
        fig_fr.add_trace(go.Scatter(x=x_idx, y=avg_best_fr_ofc, mode='lines+markers', name='Avg best FR (OFC)', line=dict(color=COLOR_ORANGE)))
    else:
        fig_fr.add_trace(go.Scatter(x=x_idx, y=avg_best_fr, mode='lines+markers', name='Avg best FR per unit'))
    fig_fr.update_layout(title=f"Average best-stimulus FR across sessions for {animal_id}", xaxis_title="Session type (ordered)", yaxis_title="FR (spikes/s)")
    fig_fr.update_xaxes(tickmode='array', tickvals=x_idx, ticktext=session_labels)
    st.plotly_chart(fig_fr, use_container_width=True)

    # Best stimulus composition heatmap
    if len(stim_values_all) > 0:
        stim_axis = np.array(sorted(list(stim_values_all)))
        if area_selection == 'Both':
            # Build separate heatmaps for ACx and OFC
            heatmap_acx = np.zeros((len(session_labels), len(stim_axis)))
            heatmap_ofc = np.zeros((len(session_labels), len(stim_axis)))
            for i, lst in enumerate(per_session_best_stim_list_acx):
                if len(lst) == 0:
                    heatmap_acx[i, :] = np.nan
                    continue
                for v in lst:
                    j = int(np.argmin(np.abs(stim_axis - float(v))))
                    heatmap_acx[i, j] += 1
                row_total = np.nansum(heatmap_acx[i, :])
                if row_total > 0:
                    heatmap_acx[i, :] = (heatmap_acx[i, :] / row_total) * 100.0
            for i, lst in enumerate(per_session_best_stim_list_ofc):
                if len(lst) == 0:
                    heatmap_ofc[i, :] = np.nan
                    continue
                for v in lst:
                    j = int(np.argmin(np.abs(stim_axis - float(v))))
                    heatmap_ofc[i, j] += 1
                row_total = np.nansum(heatmap_ofc[i, :])
                if row_total > 0:
                    heatmap_ofc[i, :] = (heatmap_ofc[i, :] / row_total) * 100.0

            fig_hm = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("ACx", "OFC"))
            fig_hm.add_trace(go.Heatmap(z=heatmap_acx, x=[f"{v:.2f}" for v in stim_axis], y=[str(s) for s in session_labels], colorscale='Blues', colorbar=dict(title='% Units')), row=1, col=1)
            fig_hm.add_trace(go.Heatmap(z=heatmap_ofc, x=[f"{v:.2f}" for v in stim_axis], y=[str(s) for s in session_labels], colorscale='Blues', showscale=False), row=2, col=1)
            fig_hm.update_layout(title="Best Stimulus Composition (% units)")
            fig_hm.update_xaxes(title_text="Stimulus (log scale ticks)", row=2, col=1)
            fig_hm.update_yaxes(title_text="Session type (ordered)", row=1, col=1)
            st.plotly_chart(fig_hm, use_container_width=True)
        else:
            heatmap = np.zeros((len(session_labels), len(stim_axis)))
            for i, lst in enumerate(per_session_best_stim_list):
                if len(lst) == 0:
                    heatmap[i, :] = np.nan
                    continue
                for v in lst:
                    j = int(np.argmin(np.abs(stim_axis - float(v))))
                    heatmap[i, j] += 1
                row_total = np.nansum(heatmap[i, :])
                if row_total > 0:
                    heatmap[i, :] = (heatmap[i, :] / row_total) * 100.0

            fig_hm = go.Figure(data=go.Heatmap(
                z=heatmap,
                x=[f"{v:.2f}" for v in stim_axis],
                y=[str(s) for s in session_labels],
                colorscale='Blues',
                colorbar=dict(title='% Units')
            ))
            fig_hm.update_layout(title="Best Stimulus Composition (% units)", xaxis_title="Stimulus (log scale ticks)", yaxis_title="Session type (ordered)")
            st.plotly_chart(fig_hm, use_container_width=True)



