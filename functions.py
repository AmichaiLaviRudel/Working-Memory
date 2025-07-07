import streamlit as st
import scipy.io as spio
import numpy as np
import pandas as pd
from scipy.stats import norm
import os

import matplotlib.pyplot as plt
import altair as alt

def init_session_states():
    # -----------------------------------------------------------------------------
    # Set session state variables (if not already set)
    # -----------------------------------------------------------------------------
    if "path" not in st.session_state:
        st.session_state.path = r"Z:\Shared\Amichai\Code\DB"
    if "user_path" not in st.session_state:
        st.session_state.user_path = os.path.join(st.session_state.path, "users_data", "Amichai")
    if "Project_type_options" not in st.session_state:
        st.session_state.Project_type_options = ["", "Behavior-Bpod GUI", "Behavior-GNG GUI", "Npxls"]

    # Change working directory

    os.chdir(st.session_state.path)


    # -----------------------------------------------------------------------------
    # Load existing projects and project list
    # -----------------------------------------------------------------------------
    existing_projects = pd.read_csv(os.path.join(st.session_state.user_path, "projects_list.csv"))
    project_list = existing_projects["Project Name"].unique().tolist()
    if "project_list" not in st.session_state:
        st.session_state.project_list = project_list
    return st.session_state


def read_results_file(file, type):
    if type == 'Behavior-Bpod GUI':
        try:
            data = load_bpod_data(file)

    # if type == 'Etho-Odor':
    #     # Change column names
    #     # Change the column names
    #     data.columns = ["Trail", "Group", "Condition", "Subject", "Target", "Distance moved", "Mean Velocity",
    #                     "Q NE", "Q SW", "Nose Target 2", "Nose Target 1"]
    #     # # Drop the first row
    #     data = data.drop(0)


        except:
            st.warning("Can't read uploaded file")
    return data


def read_gng_file(file_path):
    data = spio.loadmat(file_path, squeeze_me=True)
    # run over data and force to be length 1
    for key in data:
        data[key] = [data[key]]

    # remove '__header__', '__version__', '__globals__'
    data.pop('__header__')
    data.pop('__version__')
    data.pop('__globals__')

    # dict to dataframe
    data = pd.DataFrame.from_dict(data)

    path, name = os.path.dirname(file_path), os.path.basename(file_path).split(".")[0]
    data.to_csv(f"{os.path.join(path, name)}.csv", index=False)

    return data

def save_changes(data, file_path, selected_project):
    save_data = pd.DataFrame(data)
    save_data.to_csv(os.path.join(file_path, f"{selected_project}_experimental_data.csv"), index=False)

def save_projectlist(df, path):
    full_filename = os.path.join(path, "projects_list.csv")
    df.to_csv(full_filename, index=False)
    return full_filename


def feed_data():
    st.session_state["sidebar_tab"] = "Project DB"
    st.session_state.selected_project = ""
    st.session_state.cb_data_entry = True
    st.balloons()


def clear(project_select):
    st.session_state["sidebar_tab"] = "Project DB"
    st.session_state.selected_project = project_select
    st.spinner


def video_analysis(project_data, index):
    # uplaod video for the experiment media file
    video_file = project_data.iloc[index]["Experiment Video"]
    vid = open(video_file, "rb")
    vid_bytes = vid.read()
    st.video(vid_bytes)


import streamlit as st
import scipy.io
import pandas as pd
import numpy as np
import tempfile
import os

def load_bpod_data(uploaded_file):
    file_name = uploaded_file.name
    # get the mouse name from the file name
    mouse_name = file_name.split("_")[0]

    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_file_path = temp_file.name


    def load_mat_file(file_path):
        # Load the .mat file
        mat_contents = scipy.io.loadmat(file_path)

        # Extract the 'SessionData' field
        session_data_content = mat_contents['SessionData'][0, 0]

        # Extract the date and time information
        session_date = session_data_content['Info']['SessionDate'][0][0]
        session_time = session_data_content['Info']['SessionStartTime_UTC'][0][0]

        # Extract 'TrialTypes' and 'RawEvents'
        trial_types = session_data_content['TrialTypes']
        raw_events = session_data_content['RawEvents']

        # Convert 'TrialTypes' to a DataFrame
        trial_types_df = pd.DataFrame({'TrialType': trial_types[0]})
        # Function to extract states and timestamps from 'Trial' data in 'RawEvents'
        def extract_states_timestamps(trial_data):
            states_info = []
            for trial in trial_data:
                states = trial['States'][0][0]
                state_names = states.dtype.names
                timestamps = {name: states[name][0] for name in state_names}
                states_info.append(timestamps)
            return states_info

        # Extract the 'Trial' field from 'RawEvents'
        trial_raw_events_data = raw_events[0, 0]['Trial'][0]
        # Extract states and timestamps from the 'Trial' data
        states_timestamps_info = extract_states_timestamps(trial_raw_events_data)
        # Convert the extracted information into a DataFrame
        raw_events_df = pd.DataFrame(states_timestamps_info)

        return trial_types_df, raw_events_df, session_date, session_time

    def create_single_row_with_outcome(trial_types_df, raw_events_df, session_date, session_time):
        trial_types_list = []
        outcomes_list = []

        def contains_nan(cell):
            if isinstance(cell, (np.ndarray, list)):
                try:
                    return np.any(np.isnan(cell))
                except TypeError:
                    return any(
                        np.any(np.isnan(sub_cell)) if isinstance(sub_cell, (np.ndarray, list)) else False for sub_cell in cell)
            return False

        rewards = raw_events_df['Reward'].apply(lambda x: not contains_nan(x))
        punishments = raw_events_df['Punishment'].apply(lambda x: not contains_nan(x))

        for idx, row in trial_types_df.iterrows():
            trial_type = row['TrialType']
            reward = rewards.iloc[idx]
            punishment = punishments.iloc[idx]

            if trial_type == 1:
                trial_type_str = 'Go'
            elif trial_type == 2:
                trial_type_str = 'NoGo'
            else:
                trial_type_str = 'Unknown'

            trial_types_list.append(trial_type_str)

            if trial_type_str == 'Go':
                outcome = 'Hit' if reward else 'Miss'
            elif trial_type_str == 'NoGo':
                outcome = 'False Alarm' if punishment else 'CR'
            else:
                outcome = 'Unknown'

            outcomes_list.append(outcome)

        combined_data = {
            'MouseName': mouse_name,
            'SessionDate': session_date[0],
            'SessionTime': session_time[0],
            'TrialTypes': trial_types_list,
            'Outcomes': outcomes_list
        }

        combined_df = pd.DataFrame([combined_data])
        return combined_df

    def save_combined_data_as_csv(file_path, combined_row_df):
        directory, filename = os.path.split(file_path)
        base_name, ext = os.path.splitext(filename)
        new_filename = f"{base_name}_DB.csv"
        new_file_path = os.path.join(directory, new_filename)

        combined_row_df.to_csv(new_file_path, index=False)

    # Process the file
    trial_types_df, raw_events_df, session_date, session_time = load_mat_file(temp_file_path)
    combined_row_df = create_single_row_with_outcome(trial_types_df, raw_events_df, session_date, session_time)
    save_combined_data_as_csv(temp_file_path, combined_row_df)

    return combined_row_df
