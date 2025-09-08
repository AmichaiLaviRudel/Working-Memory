from functions import *
import streamlit as st
from Analysis.session_states import initialize_session_state
import subprocess
import sys
import os

initialize_session_state()

# Set up the page layout
st.set_page_config(page_title="Working Memory", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")
# Main section: Display the data table for the selected project

home_page = st.Page(
    "Home.py",
    title = "Home",
    icon = ":material/cottage:",
)

projects_page = st.Page(
    "Project.py",
    title = "Projects",
    icon = ":material/school:",
)
npxl_page = st.Page(
    "npxl_monitoring.py",
    title = "Neuropixels Monitoring",
    icon = ":material/neurology:",
)

working_pages = [home_page, projects_page, npxl_page]
experiments_page = [projects_page, ]

pg = st.navigation({"Working": working_pages} )
# Add buttons to the sidebar inside a container
with st.sidebar.container():
    if st.button("Open Source Folder"):
        try:
            subprocess.Popen(f'explorer "{st.session_state.user_path}"')
        except Exception as e:
            st.error(f"Error opening folder: {e}")
    
    if st.button("Concatenate sessions and reload data"):
        # 1) Run MATLAB concatenation script
        matlab_cmd_candidates = [
            'matlab -batch "addpath(\'Z:\\Shared\\Amichai\\Code\\DB\\load_data\'); concatSessions;"',
            'matlab.exe -batch "addpath(\'Z:\\Shared\\Amichai\\Code\\DB\\load_data\'); concatSessions;"'
        ]
        with st.spinner("Running MATLAB concatenation (concatSessions.m)..."):
            matlab_ok = False
            matlab_error = None
            for cmd in matlab_cmd_candidates:
                try:
                    # Use shell=True on Windows for PATH resolution
                    completed = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
                    matlab_ok = True
                    break
                except Exception as e:
                    matlab_error = e
            if not matlab_ok:
                st.error(f"Failed to run MATLAB concatenation. Ensure MATLAB is installed and on PATH.\n{matlab_error}")
                st.stop()

        # 2) Run Python loader to rebuild CSVs
        with st.spinner("Loading Bpod data (load_bpod_data.py)..."):
            try:
                loader_script = os.path.join(os.getcwd(), 'load_data', 'load_bpod_data.py')
                subprocess.run(f'"{sys.executable}" "{loader_script}"', shell=True, check=True)
            except Exception as e:
                st.error(f"Failed to load Bpod data: {e}")
                st.stop()

        st.success("Concatenation and data reload completed. Reloading app...")
        st.rerun()
pg.run()


