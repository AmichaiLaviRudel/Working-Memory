from functions import *  # Assumes common helper functions are defined here
import pandas as pd
import streamlit as st
import traceback
import os
from Analysis.GNG_bpod_analysis.colors import get_subject_color_map
from Analysis.GNG_bpod_analysis.GNG_bpod_general import render_global_early_response_filter_checkbox
from Analysis.session_states import initialize_session_state

initialize_session_state()


# =============================================================================
# Project Overview Section
# =============================================================================

def current_project_overview(existing_projects, selected_project, path, types):
    """Display and edit the current project overview in Streamlit."""
    st.title(selected_project)
    st.text(types)
    st.text("Project Overview")

    current_description = existing_projects.loc[existing_projects["Project Name"] == selected_project]

    edited_df = st.data_editor(
        current_description,
        use_container_width=True,
        hide_index=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Save Changes"):
            existing_projects.loc[existing_projects["Project Name"] == selected_project] = edited_df
            filename = save_projectlist(existing_projects, path)
            st.success(f"DataFrame saved as {filename}")

    st.divider()
    try:
        st.text("You can edit this table, but don't forget to save the changes")

        project_csv = os.path.join(path, f"{selected_project}_experimental_data.csv")
        project_data = pd.read_csv(project_csv, delimiter=",", low_memory=False)

        if "MouseName" in project_data.columns:
            color_map = get_subject_color_map(project_data["MouseName"].fillna("Unknown"))
            st.session_state["mouse_color_map"] = color_map
            project_data["Color"] = project_data["MouseName"].fillna("Unknown").map(color_map)

        if "Checkbox" not in project_data.columns:
            project_data.insert(0, "Checkbox", False)
        if "Recording" not in project_data.columns:
            project_data.insert(0, "Recording", False)

        columns_to_present = [
            "Checkbox",
            "MouseName",
            "SessionDate",
            "SessionTime",
            "Recording",
            "WaterConsumption",
            "Notes",
            "FilePath",
        ]
        st_project_data = st.data_editor(
            data=project_data,
            height=400,
            use_container_width=False,
            hide_index=True,
            column_order=columns_to_present,
            column_config={
                "Checkbox": st.column_config.CheckboxColumn(
                    "Analyse?", help="Select rows for analysis", default=False
                ),
                "Recording": st.column_config.CheckboxColumn(
                    "Recording", width="small", help="Have a recording session?", default=False
                ),
                "SessionDate": st.column_config.Column(
                    width=None, help="Date of the session", disabled=True
                ),
                "SessionTime": st.column_config.Column(
                    width="small", help="Time of the session", disabled=True
                ),
                "TrialTypes": st.column_config.Column(
                    width="medium", help="Types of trials", disabled=True
                ),
                "Outcomes": st.column_config.Column(
                    width="medium", help="Outcomes of trials", disabled=True
                ),
                "Stimuli": st.column_config.Column(
                    width="medium", help="Stimuli used in trials", disabled=True
                ),
                "FilePath": st.column_config.Column(
                    width="medium", help="Path to the file", disabled=True
                ),
                "WaterConsumption": st.column_config.Column(
                    width=None, help="Water consumption in mL", disabled=True
                ),
                "Notes": st.column_config.Column(
                    width=None, help="Editable notes field", disabled=False
                ),
                "Animal": st.column_config.Column(
                    width=None, help="Editable animal field", disabled=False
                ),
                "Date": st.column_config.Column(
                    width="large", help="Editable date field", disabled=False
                ),
            },
        )
        st.divider()

        col1, col2, col3 = st.columns([30, 70, 25])
        with col3:
            st.download_button(
                label="Download data",
                data=project_data.to_csv().encode("utf-8"),
                file_name=f"{selected_project}_data.csv",
                mime="text/csv",
            )

        with col1:
            if st.button("Save changes"):
                st.warning("Are you sure?")
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Yes", on_click=lambda: save_changes(st_project_data, path, selected_project)):
                        st.rerun()
                with c2:
                    if st.button("Cancel", key="save_denied"):
                        st.rerun()
        return st_project_data
    except Exception as e:  # noqa: BLE001
        st.error(f"Something went wrong in the project overview.\n\n{e}")
        st.text(traceback.format_exc())
        return None


# =============================================================================
# Analysis Section
# =============================================================================

def analysis(project_data: pd.DataFrame, analysis_type: str) -> None:
    """Run the selected analysis type on the chosen rows."""
    try:
        force_idx = st.session_state.get("force_single_index", None)
    except Exception:
        force_idx = None

    if force_idx is not None and force_idx in project_data.index:
        st.session_state.selected_session = pd.Index([force_idx])
        try:
            st.session_state.force_single_index = None
        except Exception:
            pass
    else:
        st.session_state.selected_session = project_data.loc[project_data["Checkbox"] == True].index

    if len(st.session_state.selected_session) == 1:
        index = st.session_state.selected_session.values[0]
        st.markdown(f"### {analysis_type}")

        if analysis_type in ("Behavior-Bpod GUI", "Educage"):
            from Analysis.GNG_bpod_analysis.GNG_Bpod_Analysis import gng_bpod_analysis

            gng_bpod_analysis(project_data, index)

    elif len(st.session_state.selected_session) < 1:
        st.info("Please select row(s) to start the analysis")

    st.markdown(f"### {analysis_type}")
    if analysis_type in ("Behavior-Bpod GUI", "Educage"):
        try:
            from Analysis.GNG_bpod_analysis.GNG_Bpod_Analysis import (
                gng_bpod_analysis_multi_animal,
                gng_bpod_analysis_multi_session,
            )

            analysis_type_selection = st.radio(
                "Select Analysis Type",
                options=["Multi-Animal", "Multi-Session"],
                key="multi_animal_analysis_type",
            )
            if analysis_type_selection == "Multi-Animal":
                gng_bpod_analysis_multi_animal(project_data, st.session_state.selected_session)
            elif analysis_type_selection == "Multi-Session":
                gng_bpod_analysis_multi_session(project_data, st.session_state.selected_session)
        except Exception as e:  # noqa: BLE001
            st.error(f"Something went wrong in group Bpod analysis.\n\n{e}")
            st.text(traceback.format_exc())


# =============================================================================
# Main App Code
# =============================================================================

# Load the list of existing projects and the project list from session state
existing_projects = pd.read_csv(os.path.join(st.session_state.user_path, "projects_list.csv"))
project_list = st.session_state.project_list

# Sidebar project selection
st.session_state.selected_project = st.sidebar.radio(
    "Select Project", project_list, key="select_project"
)

# Process the project types for the selected project
project_types_str = existing_projects[
    existing_projects["Project Name"] == st.session_state.selected_project
]["Project Type"]

if isinstance(project_types_str, pd.Series):
    project_types_str = project_types_str.iloc[0]
elif hasattr(project_types_str, "item"):
    project_types_str = project_types_str.item(0)
else:
    project_types_str = str(project_types_str)
project_types_str = str(project_types_str)[1:-1]
project_types = [x.strip().strip("'") for x in project_types_str.split(",")]

# Try to fetch DataDir for the selected project (may be missing for older rows)
try:
    project_row = existing_projects.loc[
        existing_projects["Project Name"] == st.session_state.selected_project
    ].iloc[0]
    data_dir = project_row.get("DataDir", None)
except Exception:
    data_dir = None

# Set default analysis type in session state if available and not already set
try:
    if (not st.session_state.get("analysis_type")) and len(project_types) > 0:
        st.session_state.analysis_type = project_types[0]
except Exception:
    pass

# Explicit data-loading buttons
if any(t.lower() == "educage" for t in project_types):
    from load_data.educage_data_formmater import run_educage_formatter

    if st.button("Run Educage data formatter"):
        if not data_dir:
            st.error("Please set DataDir for this project in projects_list.csv")
        else:
            # For Educage, DataDir is a directory; the source text file is assumed
            # to be named after the last folder component (e.g. Dir\\Dir.txt).
            last_node = os.path.basename(os.path.normpath(data_dir))
            source_path = os.path.join(data_dir, f"{last_node}.txt")

            out_csv = os.path.join(
                st.session_state.user_path,
                f"{st.session_state.selected_project}_experimental_data.csv",
            )
            try:
                result_csv = run_educage_formatter(source_path, out_csv)
                st.success(f"Educage data updated: {result_csv}")
            except Exception as e:  # noqa: BLE001
                st.error(f"Educage data formatter failed.\\n\\n{e}")
                st.text(traceback.format_exc())

if any(t.lower() == "behavior-bpod gui" for t in project_types):
    from load_data.load_bpod_data import main as run_bpod_loader

    if st.button("Load / update Bpod data"):
        if not data_dir:
            st.error("Please set DataDir for this project in projects_list.csv")
        else:
            try:
                csv_path = run_bpod_loader(data_dir)
                st.success(f"Bpod data updated: {csv_path}")
            except Exception as e:  # noqa: BLE001
                st.error(f"Bpod data loader failed.\\n\\n{e}")
                st.text(traceback.format_exc())

# Display the current project overview and get edited project data
project_data = current_project_overview(
    existing_projects,
    st.session_state.selected_project,
    st.session_state.user_path,
    project_types,
)
st.divider()

# Analysis section header
st.title("Analysis")
render_global_early_response_filter_checkbox()

# Run analysis for each project type
for project_type in project_types:
    try:
        st.session_state.analysis_type = project_type
    except Exception:
        pass
    analysis(project_data, project_type)
