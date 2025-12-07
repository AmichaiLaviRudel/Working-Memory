import pandas as pd
import streamlit as st
import time
import numpy as np
import os
import plotly.graph_objects as go
from Analysis.GNG_bpod_analysis.colors import COLOR_HIT, COLOR_MISS, COLOR_FA, COLOR_CR
from Analysis.NPXL_analysis.npxl_single_unit_analysis import single_unit_analysis_panel
from Analysis.NPXL_analysis.population_analysis import plot_population_heatmap, advanced_population_analysis_panel, plot_best_stimulus_panel

# Load the experimental data
project_data = pd.read_csv(st.session_state.npxl_monitoring_path, delimiter=',', low_memory=False)

# Session type is now provided explicitly by the monitoring table (column: 'Session Type')

# Streamlit App
st.title("Neuropixels Data Management")

# Display the editable data table
st_project_data = st.data_editor(
    data=project_data,
    height=400,
    column_config={
        "spike glx file": st.column_config.TextColumn(help="File Name"),
        "Session Type": st.column_config.TextColumn(help="Session Type (provided)"),
        "status": st.column_config.SelectboxColumn(
            label=None,
            help="Status of the recordings processing",
            default="Main",
            options=["Main", "Shared", "CAT", "KS", "Phy", "Tprime", "Bombcell"]
        )
    }
)

with st.expander("Utilities", expanded=False):
    st.subheader("Date Calculator")
    col1, col2 = st.columns(2)#אני ילד כאפותתתתתתתתת#
    
    with col1:
        start_date = st.date_input("Start Date", value=None, key="date_calc_start")
    
    with col2:
        end_date = st.date_input("End Date (leave empty for today)", value=None, key="date_calc_end")
    
    if start_date:
        if end_date:
            days_diff = (end_date - start_date).days
            target_date = end_date
        else:
            from datetime import date
            days_diff = (date.today() - start_date).days
            target_date = date.today()
        
        if days_diff >= 0:
            st.success(f"**{days_diff} days** from {start_date} to {target_date}")
        else:
            st.warning(f"**{abs(days_diff)} days** before {start_date} (negative difference)")
    
    st.divider()
    
    if st.button("Auto-fill unit summaries from bombcell"):
        df = st_project_data.copy()
        updated_rows = 0

        def is_empty_cell(val):
            return val is None or (isinstance(val, float) and np.isnan(val)) or (isinstance(val, str) and val.strip() == "")

        def summarize_unit_labels(tsv_path):
            try:
                tbl = pd.read_csv(tsv_path, sep='\t')
            except Exception:
                return None

            good = mua = nonsoma = 0
            # Prefer string labels if available
            if 'bc_unitType' in tbl.columns:
                col = tbl['bc_unitType'].astype(str).str.upper()
                good = (col == 'GOOD').sum()
                mua = (col == 'MUA').sum()
                nonsoma = (col == 'NON-SOMA').sum()
            elif 'UnitType' in tbl.columns:
                # UnitType may be numeric (1,2,3) or strings
                if np.issubdtype(tbl['UnitType'].dtype, np.number):
                    col = tbl['UnitType']
                    good = (col == 1).sum()
                    mua = (col == 2).sum()
                    nonsoma = (col == 3).sum()
                else:
                    col = tbl['UnitType'].astype(str).str.upper()
                    good = (col == 'GOOD').sum()
                    mua = (col == 'MUA').sum()
                    nonsoma = (col == 'NON-SOMA').sum()
            else:
                return None

            return f"good: {int(good)}, MUA: {int(mua)}, non-somatic: {int(nonsoma)}"

        for idx, row in df.iterrows():
            current_dir = row.get('current_dir', None)
            if not current_dir or not isinstance(current_dir, str) or not os.path.isdir(current_dir):
                continue

            need_acx = ('Acx good units' in df.columns) and is_empty_cell(row.get('Acx good units', None))
            need_ofc = ('OFC good units' in df.columns) and is_empty_cell(row.get('OFC good units', None))
            if not (need_acx or need_ofc):
                continue

            acx_summary = None
            ofc_summary = None

            for root, dirs, files in os.walk(current_dir):
                if 'bombcell' in dirs:
                    bc_dir = os.path.join(root, 'bombcell')
                    tsv_path = os.path.join(bc_dir, 'unit_labels.tsv')
                    if not os.path.exists(tsv_path):
                        continue

                    summary = summarize_unit_labels(tsv_path)
                    if summary is None:
                        continue

                    parent_tail = os.path.basename(root)[-5:]
                    if '0' in parent_tail:
                        acx_summary = summary
                    else:
                        ofc_summary = summary

            changed = False
            if need_acx and acx_summary is not None:
                df.at[idx, 'Acx good units'] = acx_summary
                changed = True
            if need_ofc and ofc_summary is not None:
                df.at[idx, 'OFC good units'] = ofc_summary
                changed = True

            if changed:
                updated_rows += 1

        if updated_rows > 0:
            df.to_csv(st.session_state.npxl_monitoring_path, index=False)
            st.toast(f"Filled unit summaries for {updated_rows} row(s) and saved.", icon='✅')
            time.sleep(1.0)
            st.rerun()
        else:
            st.info("No empty unit summary fields were filled.")

# Display session type summary
if 'Session Type' in st_project_data.columns:
    st.subheader("Session Type Summary")
    session_type_counts = st_project_data['Session Type'].value_counts()
    
    # Create bar plot with horizontal line at y=12
    fig = go.Figure()
    
    # Add bar plot
    fig.add_trace(go.Bar(
        x=session_type_counts.index,
        y=session_type_counts.values,
        name='Session Count',
        marker_color='lightblue'
    ))
    
    # Add horizontal line at y=12
    fig.add_hline(y=12, line_dash="dash", line_color="black", 
                  annotation_text="Target: 12", annotation_position="bottom right")
    
    # Update layout
    fig.update_layout(
        title="Session Type Distribution",
        xaxis_title="Session Type",
        yaxis_title="Count",
        showlegend=False,
        height=400
    )
    
    # Display the plot
    st.plotly_chart(fig, use_container_width=True)

# Add a save button
if st.button("Save Changes"):
    # Save the modified data to your CSV/database
    st_project_data.to_csv(st.session_state.npxl_monitoring_path, index=False)  # Adjust filename as needed
    st.toast("Changes saved successfully!",  icon='🎉')
    time.sleep(1.5)
    st.rerun()

st.divider()

def _parse_unit_summary_cell(cell):
    # Expected formats: "good: X, MUA: Y, non-somatic: Z" or just a number
    good = 0
    mua = 0
    if cell is None:
        return good, mua
    if isinstance(cell, (int, float)) and not np.isnan(cell):
        # Legacy numeric value treated as good units only
        return int(cell), 0
    if isinstance(cell, str):
        text = cell.strip().lower()
        # Try to parse key:value pairs
        try:
            parts = [p.strip() for p in text.split(',')]
            for p in parts:
                if ':' in p:
                    k, v = [q.strip() for q in p.split(':', 1)]
                    if k == 'good':
                        good = int(float(v))
                    elif k == 'mua':
                        mua = int(float(v))
            return good, mua
        except Exception:
            pass
        # Fallback: try to interpret as a single integer
        try:
            val = int(float(text))
            return val, 0
        except Exception:
            return 0, 0
    return 0, 0

# Aggregate Good & MUA units by Session Type and region (ACx vs OFC)
if 'Session Type' in st_project_data.columns:
    # session_type -> {'acx': {'good': int, 'mua': int}, 'ofc': {'good': int, 'mua': int}}
    agg = {}
    acx_col = 'Acx good units' if 'Acx good units' in st_project_data.columns else None
    ofc_col = 'OFC good units' if 'OFC good units' in st_project_data.columns else None

    for _, row in st_project_data.iterrows():
        session_type = row.get('Session Type', None)
        if session_type is None:
            continue
        if session_type not in agg:
            agg[session_type] = {'acx': {'good': 0, 'mua': 0}, 'ofc': {'good': 0, 'mua': 0}}

        if acx_col is not None:
            g, m = _parse_unit_summary_cell(row.get(acx_col, None))
            agg[session_type]['acx']['good'] += g
            agg[session_type]['acx']['mua'] += m
        if ofc_col is not None:
            g, m = _parse_unit_summary_cell(row.get(ofc_col, None))
            agg[session_type]['ofc']['good'] += g
            agg[session_type]['ofc']['mua'] += m

    if len(agg) > 0:
        st.subheader("Units by Session Type (ACx vs OFC, Good vs MUA)")
        types = list(agg.keys())
        acx_good = [agg[t]['acx']['good'] for t in types]
        acx_mua = [agg[t]['acx']['mua'] for t in types]
        ofc_good = [agg[t]['ofc']['good'] for t in types]
        ofc_mua = [agg[t]['ofc']['mua'] for t in types]

        fig_units = go.Figure()
        # ACx stacked
        fig_units.add_trace(go.Bar(x=types, y=acx_good, name='ACx Good', marker_color='green', offsetgroup='acx'))
        fig_units.add_trace(go.Bar(x=types, y=acx_mua, name='ACx MUA', marker_color='orange', offsetgroup='acx', base=None))
        # OFC stacked (side-by-side with ACx) with requested colors
        fig_units.add_trace(go.Bar(x=types, y=ofc_good, name='OFC Good', marker_color='#4C763B', offsetgroup='ofc'))
        fig_units.add_trace(go.Bar(x=types, y=ofc_mua, name='OFC MUA', marker_color='#FF9013', offsetgroup='ofc', base=None))

        fig_units.update_layout(
            barmode='stack',
            title='Sum of Good and MUA Units per Session Type (ACx vs OFC)',
            xaxis_title='Session Type',
            yaxis_title='Unit Count',
            height=440
        )
        st.plotly_chart(fig_units, use_container_width=True)


# Check if 'Checkbox' column exists and has any True values
if 'Checkbox' in st_project_data.columns and st_project_data['Checkbox'].any():
    st.subheader("Analysis")
    for idx, row in st_project_data[st_project_data['Checkbox'] == True].iterrows():
        current_dir = row.get('current_dir', None)
        if current_dir and isinstance(current_dir, str):
            analysis_output_dirs = []
            for root, dirs, files in os.walk(current_dir):
                for d in dirs:
                    if d == "analysis_output":
                        analysis_output_dirs.append(os.path.join(root, d))
    
    main_recording_dir = current_dir
    # Initialize variables for sharing between tabs
    selected_folder = None
    event_windows_matrix = None
    stimuli_outcome_df = None
    spike_matrix = None
    
    folder_options = ["imec0", "imec1"]
    folder_labels = ["ACx (imec0)", "OFC (imec1)"]
    

    selected_label = st.selectbox(
        "Select analysis output folder",
        options=folder_labels,
        key=f"selectbox_analysis_output_{idx}"
    )

    # Determine which folder to select based on the chosen label (imec0 or imec1)
    if selected_label == "ACx (imec0)":
        # Search for a directory containing 'imec0' in analysis_output_dirs
        prob_folder = next((d for d in analysis_output_dirs if "imec0" in d), None)
    else:
        # Otherwise, search for 'imec1'
        prob_folder = next((d for d in analysis_output_dirs if "imec1" in d), None)
    st.session_state[f"selected_analysis_output_{idx}"] = selected_label


    st.badge("Behavior Analysis")
    # Create 4 Streamlit tabs for analysis
    single_unit_tab, population_tab, advanced_tab, multi_tab = st.tabs(["Single Unit", "Population", "Population Adv.", "Multi"])


    with single_unit_tab:
        st.write("### Single Unit Analysis")
        single_unit_analysis_panel(selected_recording_dir=main_recording_dir, selected_area = selected_label, raw_folder=prob_folder)


        #     with population_tab:
        #         st.write("### Population Analysis")
        #         if selected_folder and event_windows_matrix is not None and stimuli_outcome_df is not None and metadata is not None:
        #             # Pass metadata instead of window_size*3
        #             plot_population_heatmap(event_windows_matrix, stimuli_outcome_df, metadata)
        #             st.divider()
        #             st.subheader("Best Stimulus Across Units")
        #             plot_best_stimulus_panel(event_windows_matrix, stimuli_outcome_df, metadata)
                    
        #         else:
        #             st.warning("Event windows data not available for population analysis")
            
        #     with advanced_tab:
        #         st.write("### Advanced Population Analysis")
        #         if selected_folder and event_windows_matrix is not None and stimuli_outcome_df is not None and metadata is not None:
        #             advanced_population_analysis_panel(event_windows_matrix, stimuli_outcome_df, metadata, time_axis_from_load)
        #         else:
        #             st.warning("Event windows data not available for advanced analysis")
            
        #     with multi_tab:
        #         st.write("### Multi Analysis")
        #         st.write("Coming soon")
        #     if FRA_session:
        #         with fra_tab:
        #             st.write("### FRA Analysis")
        #             st.write("Coming soon")
        # else:
        #     analysis_output_dirs = []

