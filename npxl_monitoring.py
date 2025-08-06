import pandas as pd
import streamlit as st
import time
import numpy as np
import os
from Analysis.GNG_bpod_analysis.colors import COLOR_HIT, COLOR_MISS, COLOR_FA, COLOR_CR
from Analysis.NPXL_analysis.npxl_single_unit_analysis import single_unit_analysis_panel
from Analysis.NPXL_analysis.population_analysis import plot_population_heatmap

# Load the experimental data
project_data = pd.read_csv(st.session_state.npxl_monitoring_path, delimiter=',', low_memory=False)

# Streamlit App
st.title("Neuropixels Data Management")

# Display the editable data table
st_project_data = st.data_editor(
    data=project_data,
    height=400,
    column_config={
        "spike glx file": st.column_config.TextColumn(help="File Name"),
        "status": st.column_config.SelectboxColumn(
            label=None,
            help="Status of the recordings processing",
            default="Main",
            options=["Main", "Shared", "CAT", "KS", "Phy", "Tprime", "Bombcell"]
        )
    }
)
# Add a save button
if st.button("Save Changes"):
    # Save the modified data to your CSV/database
    st_project_data.to_csv(st.session_state.npxl_monitoring_path, index=False)  # Adjust filename as needed
    st.toast("Changes saved successfully!",  icon='🎉')
    time.sleep(1.5)
    st.rerun()

st.divider()


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
            
            FRA_session = False
            if 'FRA' in root:
                st.badge("FRA Analysis")
                # Create 2 Streamlit tabs for analysis
                single_unit_tab, fra_tab = st.tabs(["Single Unit", "FRA"])
                FRA_session = True

            else:
                st.badge("Behavior Analysis")
                # Create 3 Streamlit tabs for analysis
                single_unit_tab, population_tab, multi_tab = st.tabs(["Single Unit", "Population", "Multi"])


            # Initialize variables for sharing between tabs
            selected_folder = None
            event_windows_matrix = None
            stimuli_outcome_df = None
            spike_matrix = None
            
            with single_unit_tab:
                st.write("### Single Unit Analysis")
                folder_options = []
                folder_labels = []
                for folder in analysis_output_dirs:
                    folder_name = os.path.basename(folder)
                    folder_parent = os.path.dirname(folder)[-5:]
                    if "0" in folder_parent:
                        folder_parent_label = "ACx"
                    else:
                        folder_parent_label = "OFC"
                    label = f"{folder_parent_label} ({folder_parent})"
                    folder_options.append(folder)
                    folder_labels.append(label)
                if folder_options:
                    default_index = 0
                    if f"selected_analysis_output_{idx}" in st.session_state:
                        try:
                            default_index = folder_options.index(st.session_state[f"selected_analysis_output_{idx}"])
                        except ValueError:
                            default_index = 0
                    selected_label = st.selectbox(
                        "Select analysis output folder",
                        options=folder_labels,
                        index=default_index,
                        key=f"selectbox_analysis_output_{idx}"
                    )
                    selected_folder = folder_options[folder_labels.index(selected_label)]
                    st.session_state[f"selected_analysis_output_{idx}"] = selected_folder
                else:
                    selected_folder = None

                if selected_folder:
                    # Use the load_event_windows_data function from npxl_single_unit_analysis
                    from Analysis.NPXL_analysis.npxl_single_unit_analysis import load_event_windows_data
                    
                    loaded_data = load_event_windows_data(selected_folder)
                    if loaded_data:
                        event_windows_matrix, time_axis_from_load, valid_event_indices, stimuli_outcome_df, metadata = loaded_data
                        single_unit_analysis_panel(event_windows_matrix, stimuli_outcome_df, selected_folder)
                    else:
                        st.error(f"Event windows data could not be loaded from: {selected_folder}")
                        st.info("Please ensure the event windows data has been generated.")
                        # Set variables to None to prevent errors in other tabs
                        event_windows_matrix = None
                        stimuli_outcome_df = None
                        metadata = None

            with population_tab:
                st.write("### Population Analysis")
                if selected_folder and event_windows_matrix is not None and stimuli_outcome_df is not None and metadata is not None:
                    # Pass metadata instead of window_size*3
                    plot_population_heatmap(event_windows_matrix, stimuli_outcome_df, metadata)
                else:
                    st.warning("Event windows data not available for population analysis")
            with multi_tab:
                st.write("### Multi Analysis")
                st.write("Coming soon")
            if FRA_session:
                with fra_tab:
                    st.write("### FRA Analysis")
                    st.write("Coming soon")
        else:
            analysis_output_dirs = []

