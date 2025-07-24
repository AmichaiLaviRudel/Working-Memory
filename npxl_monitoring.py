import pandas as pd
import streamlit as st
import time
import numpy as np
import os
from Analysis.GNG_bpod_analysis.colors import COLOR_HIT, COLOR_MISS, COLOR_FA, COLOR_CR
from Analysis.NPXL_analysis.npxl_single_unit_analysis import plot_unit_psth, compute_psth_pvalues, single_unit_analysis_panel
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
st.subheader("Analysis")

# Check if 'Checkbox' column exists and has any True values
if 'Checkbox' in st_project_data.columns and st_project_data['Checkbox'].any():
    for idx, row in st_project_data[st_project_data['Checkbox'] == True].iterrows():
        current_dir = row.get('current_dir', None)
        if current_dir and isinstance(current_dir, str):
            analysis_output_dirs = []
            for root, dirs, files in os.walk(current_dir):
                for d in dirs:
                    if d == "analysis_output":
                        analysis_output_dirs.append(os.path.join(root, d))
            
            # Create 4 Streamlit tabs for analysis
            single_unit_tab, population_tab, multi_tab, fra_tab = st.tabs(["Single Unit", "Population", "Multi", "FRA"])


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
                    analysis_type = st.selectbox("Select type", options=["good", "mua", "non_somatic", "spike"])
                    analysis_output_dirs = [selected_folder]
                    load_path = os.path.join(selected_folder, f"{analysis_type}_matrix.npy")
                    spike_matrix = np.load(load_path, allow_pickle=True)
                    stimuli_outcome_df = pd.read_csv(os.path.join(selected_folder, "stimuli_outcome_df.csv"))
                    single_unit_analysis_panel(spike_matrix, stimuli_outcome_df)

            with population_tab:
                st.write("### Population Analysis")
                plot_population_heatmap(spike_matrix, stimuli_outcome_df, window_size=1000)
            with multi_tab:
                st.write("### Multi Analysis")
                st.write("Coming soon")

            with fra_tab:
                st.write("### FRA Analysis")
                st.write("Coming soon")
        else:
            analysis_output_dirs = []

