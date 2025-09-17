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
    fig.add_hline(y=12, line_dash="dash", line_color="red", 
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
            
            # Initialize variables for sharing between tabs
            selected_folder = None
            event_windows_matrix = None
            stimuli_outcome_df = None
            spike_matrix = None
            
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
                # Use the load_event_windows_data function from NPXL_Preprocessing
                from Analysis.NPXL_analysis.NPXL_Preprocessing import load_event_windows_data    
                loaded_data = load_event_windows_data(selected_folder)

            FRA_session = False
            if 'FRA' in root:
                st.badge("FRA Analysis")
                # Create 2 Streamlit tabs for analysis
                single_unit_tab, fra_tab = st.tabs(["Single Unit", "FRA"])
                FRA_session = True

            else:
                st.badge("Behavior Analysis")
                # Create 4 Streamlit tabs for analysis
                single_unit_tab, population_tab, advanced_tab, multi_tab = st.tabs(["Single Unit", "Population", "Population Adv.", "Multi"])


            with single_unit_tab:
                st.write("### Single Unit Analysis")
                
                if loaded_data:
                    event_windows_matrix, time_axis_from_load, valid_event_indices, stimuli_outcome_df, metadata, lick_event_windows_matrix = loaded_data
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
                    st.divider()
                    st.subheader("Best Stimulus Across Units")
                    plot_best_stimulus_panel(event_windows_matrix, stimuli_outcome_df, metadata)
                    
                else:
                    st.warning("Event windows data not available for population analysis")
            
            with advanced_tab:
                st.write("### Advanced Population Analysis")
                if selected_folder and event_windows_matrix is not None and stimuli_outcome_df is not None and metadata is not None:
                    advanced_population_analysis_panel(event_windows_matrix, stimuli_outcome_df, metadata, time_axis_from_load)
                else:
                    st.warning("Event windows data not available for advanced analysis")
            
            with multi_tab:
                st.write("### Multi Analysis")
                st.write("Coming soon")
            if FRA_session:
                with fra_tab:
                    st.write("### FRA Analysis")
                    st.write("Coming soon")
        else:
            analysis_output_dirs = []

