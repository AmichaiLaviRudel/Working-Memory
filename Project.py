from functions import *  # Assumes common helper functions are defined here
import pandas as pd
import streamlit as st
import traceback


# =============================================================================
# Project Overview Section
# =============================================================================
def current_project_overview(existing_projects, selected_project, path, types):
    """
    Displays and allows editing of the current project overview.

    Args:
        existing_projects (pd.DataFrame): DataFrame with all projects.
        selected_project (str): Name of the selected project.
        path (str): Path to project files.
        types (list): List of project types.

    Returns:
        pd.DataFrame: Edited project data for further analysis.
    """
    st.title(selected_project)
    st.text(types)
    st.text("Project Overview")

    # Get the current project description from the list of projects
    current_description = existing_projects.loc[existing_projects["Project Name"] == selected_project]

    # Allow the user to edit the project description
    edited_df = st.data_editor(
        current_description,
        use_container_width = True,
        hide_index = True
    )

    # Save the edited project description
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Save Changes"):
            existing_projects.loc[existing_projects["Project Name"] == selected_project] = edited_df
            filename = save_projectlist(existing_projects, path)
            st.success(f"DataFrame saved as {filename}")

    st.divider()
    st.text("You can edit this table, but don't forget to save the changes")

    # Load the experimental data for the selected project
    project_csv = os.path.join(path, f"{selected_project}_experimental_data.csv")
    project_data = pd.read_csv(project_csv, delimiter = ',', low_memory = False)



    # Ensure that a 'Checkbox' column exists for selecting rows
    if 'Checkbox' not in project_data.columns:
        project_data.insert(0, 'Checkbox', False)

    columns_to_present = ["Checkbox", "MouseName", "SessionDate", "SessionTime", "WaterConsumption","Notes", "FilePath"]
    # Display the project data editor with column configurations
    st_project_data = st.data_editor(
        data = project_data,
        height = 400,
        # use_container_width = True,
        hide_index = True,
        column_order = columns_to_present,
        column_config = {
            "Checkbox":         st.column_config.CheckboxColumn(
                "Analyse?", help = "Select rows for analysis", default = False,
            ),
            "SessionDate":      st.column_config.Column(
                width = 100, help = "Date of the session", disabled = True
            ),
            "SessionTime":      st.column_config.Column(
                width = 100, help = "Time of the session", disabled = True
            ),
            "TrialTypes":       st.column_config.Column(
                width = 150, help = "Types of trials", disabled = True
            ),
            "Outcomes":         st.column_config.Column(
                width = 150, help = "Outcomes of trials", disabled = True
            ),
            "Stimuli":          st.column_config.Column(
                width = 150, help = "Stimuli used in trials", disabled = True
            ),
            "FilePath":         st.column_config.Column(
                width = 100, help = "Path to the file", disabled = True
            ),
            "WaterConsumption": st.column_config.Column(
                width = 60, help = "Water consumption in mL", disabled = True
            ),
            "Notes":            st.column_config.Column(
                width = 200, help = "Editable notes field", disabled = False
            ),
            "Animal":           st.column_config.Column(
                width = 250, help = "Editable animal field", disabled = False
            ),
            "Date":             st.column_config.Column(
                width = 250, help = "Editable date field", disabled = False
            ),
        }
    )
    st.divider()

    # Download button for the project data
    col1, col2, col3 = st.columns([30, 70, 25])
    with col3:
        st.download_button(
            label = "Download data",
            data = project_data.to_csv().encode("utf-8"),
            file_name = f"{selected_project}_data.csv",
            mime = 'text/csv'
        )

    # Button to save changes made to the project data
    with col1:
        if st.button("Save changes"):
            st.warning("Are you sure?")
            c1, c2 = st.columns(2)
            with c1:
                # Use a lambda so that the on_click callback is not executed immediately
                if st.button("Yes", on_click = lambda: save_changes(st_project_data, path, selected_project)):
                    st.rerun()
            with c2:
                if st.button("Cancel", key = "save_denied"):
                    st.rerun()
    return st_project_data


# =============================================================================
# Analysis Section
# =============================================================================
def analysis(project_data, analysis_type):
    """
    Perform analysis based on the project data and selected analysis type.

    Args:
        project_data (pd.DataFrame): DataFrame containing experimental data.
        analysis_type (str): Type of analysis to perform.
    """
    # Get indices of rows selected for analysis via the 'Checkbox'
    selected_indices = project_data.loc[project_data["Checkbox"] == True].index


    if len(selected_indices) == 1:
        # Single-row analysis
        index = selected_indices.values[0]
        st.markdown(f"### {analysis_type}")

        if analysis_type == 'Behavior-Bpod GUI':
            from Analysis.GNG_bpod_analysis.GNG_Bpod_Analysis import gng_bpod_analysis
            gng_bpod_analysis(project_data, index)

    elif len(selected_indices) < 1:
        st.info("Please select row(s) to start the analysis")

        # If more than one row is selected, run group analysis
    with st.expander("Multianimal Analysis"):
        st.markdown(f"### {analysis_type}")

        if analysis_type == 'Behavior-Bpod GUI':
            try:
                from Analysis.GNG_bpod_analysis.GNG_Bpod_Analysis import gng_bpod_analysis_multipule
                gng_bpod_analysis_multipule(project_data, selected_indices)
            except Exception as e:
                st.error(f"Something went wrong in group Bpod analysis.\n\n{e}")
                st.text(traceback.format_exc())

        else:
            st.info("Analysis for this project type is coming soon...")


# =============================================================================
# Main App Code
# =============================================================================

# Load the list of existing projects and the project list from session state
existing_projects = pd.read_csv(os.path.join(st.session_state.user_path, "projects_list.csv"))
project_list = st.session_state.project_list

# Sidebar project selection
st.session_state.selected_project = st.sidebar.radio("Select Project", project_list, key = "select_project")

# Process the project types for the selected project
project_types_str = existing_projects[
                        existing_projects["Project Name"] == st.session_state.selected_project
                        ]["Project Type"].values[0][1:-1]  # Remove brackets if present
project_types = [x.strip().strip("'") for x in project_types_str.split(",")]

# Display the current project overview and get edited project data
project_data = current_project_overview(
    existing_projects,
    st.session_state.selected_project,
    st.session_state.user_path,
    project_types
)
st.divider()

# Analysis section header
st.title("Analysis")

# Run analysis for each project type
for project_type in project_types:
    analysis(project_data, project_type)