import streamlit as st
import os
import pandas as pd



def initialize_session_state():
    # Path and user path
    if 'path' not in st.session_state:
        st.session_state.path = r"Z:\Shared\Amichai\Code\DB"
    if 'user_path' not in st.session_state:
        st.session_state.user_path = os.path.join(st.session_state.path, "users_data", "Amichai")
    if 'npxl_monitoring_path' not in st.session_state:
        st.session_state.npxl_monitoring_path = os.path.join(st.session_state.user_path, "NPXL recordings _experimental_data.csv")

    # Project type options
    if 'Project_type_options' not in st.session_state:
        st.session_state.Project_type_options = ["", "Behavior-Bpod GUI", "Behavior-GNG GUI", "Npxls"]

    # Selected analysis type
    if 'analysis_type' not in st.session_state:
        st.session_state.analysis_type = ""

    # Project list
    try:
        existing_projects = pd.read_csv(os.path.join(st.session_state.user_path, "projects_list.csv"))
        project_list = existing_projects["Project Name"].unique().tolist()
        if 'project_list' not in st.session_state:
            st.session_state.project_list = project_list
    except Exception:
        if 'project_list' not in st.session_state:
            st.session_state.project_list = []

    # Selected project
    if 'selected_project' not in st.session_state:
        st.session_state.selected_project = ''

    # Deep-linking and selection flags
    if 'force_single_index' not in st.session_state:
        st.session_state.force_single_index = None
    if 'selected_session' not in st.session_state:
        st.session_state.selected_session = None

    # Animal selection keys (used in metric.py)
    if 'd_prime_animal_select' not in st.session_state:
        st.session_state.d_prime_animal_select = None
    if 'd_prime_low_high_animal_select' not in st.session_state:
        st.session_state.d_prime_low_high_animal_select = None

    # Low/high boundary values
    if 'low_boundary' not in st.session_state:
        st.session_state.low_boundary = 0.983
    if 'high_boundary' not in st.session_state:
        st.session_state.high_boundary = 1.525

    # Global preference: filter bins where FA rate = Hit rate = 100%
    if 'filter_fa_equal_hit' not in st.session_state:
        st.session_state.filter_fa_equal_hit = False

    st.session_state['nidq_events_times_path'] = 'nidq.xd_0_1_100.txt'


    # Optionally: change working directory
    try:
        os.chdir(st.session_state.path)
    except Exception:
        pass

# Call this function at the start of your main app file
