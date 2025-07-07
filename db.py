from functions import *
import streamlit as st



init_session_states()

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
            import subprocess
            subprocess.Popen(f'explorer "{st.session_state.user_path}"')
        except Exception as e:
            st.error(f"Error opening folder: {e}")
pg.run()


