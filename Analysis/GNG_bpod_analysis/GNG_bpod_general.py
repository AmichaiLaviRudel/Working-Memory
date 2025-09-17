from Analysis.GNG_bpod_analysis.psychometric_curves import *
from Analysis.GNG_bpod_analysis.licking_and_outcome import *
from Analysis.GNG_bpod_analysis.metric import *


import numpy as np
import pandas as pd
import ast


def getNameAndSession(project_data, index):
    mouse_name = project_data.iloc[index]['MouseName']
    # Find all session indices where this mouse appears
    session_indices = project_data.index[project_data["MouseName"] == mouse_name].tolist()

    # Get the session number corresponding to the current index
    current_session = session_indices.index(index) + 1  # Assuming session numbers are 1-based
    return mouse_name, current_session



def get_sessions_for_animal(selected_data, animal_name):
    # Filter the data to get indices of all sessions for the selected animal
    animal_sessions = selected_data[selected_data['MouseName'] == animal_name]
    session_indices = animal_sessions.index.tolist()  # Get indices of the animal's sessions
    session_dates = animal_sessions['SessionDate'].values  # Get the session dates

    return session_indices, session_dates


# Function to filter out None and empty arrays
def filter_valid_arrays(data):
    return [arr for arr in data if isinstance(arr, np.ndarray) and arr.ndim == 1 and arr.size > 0]


# Parse stringified stimulus arrays into numeric NumPy arrays
def parse_stimuli(stim_str):
    try:
        return np.fromstring(stim_str.strip("[]"), sep = " ")
    except Exception:
        return np.array([])

