import streamlit as st
import os
import pandas as pd
from functions import *

init_session_states()


def fisrt_section(existing_projects, user, Project_type_options):
    st.header(f"{user.title()}'s Running Projects")

    # Convert project names to clickable links
    existing_projects['Project Name'] = existing_projects['Project Name']

    # Display the table with clickable links
    st.data_editor(existing_projects, use_container_width=True, hide_index = True)

    return existing_projects


def add_project_main(df, path):
    if st.button("Save Changes"):
        st.warning("Do you really, really, wanna do this?")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Yape"):
                cb_add_new_project(df, path)
                st.spinner("Saving Changes...")
        with c2:
            if st.button("Nope"):
                st.rerun()


def cb_add_new_project(df, path):
    df.to_csv(os.path.join(path, "projects_list.csv"), index = False)
    st.success("Changes saved!")

st.title("Hey There 🧠")
path = r"G:\My Drive\Study\Lab\Projects\Code_temp"
user = "Amichai"
user_path = os.path.join(path, 'users_data', user)
# Set up the project list
Project_type_options = ["", "Behavior-Bpod GUI", "Behavior-GNG GUI", "Npxls"]

existing_projects = pd.read_csv(os.path.join(user_path, "projects_list.csv"))
existing_projects = fisrt_section(existing_projects, user, Project_type_options)
add_project_main(existing_projects, path)