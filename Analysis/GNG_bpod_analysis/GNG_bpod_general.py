# Removed imports to avoid circular dependency

import numpy as np
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

def parse_licks(licks_str):
    import re
    try:
        licks_str = re.sub(r'array\(', 'np.array(', licks_str)
        licks = eval(licks_str, {"np": np, "None": None, "nan": None})
    except Exception:
        licks = np.array([])
    return licks
def object_to_array(obj_array, pad_value=np.nan):
    """
    Convert a 1D object array of 1D arrays/lists into
    a 2D numeric array with NaN padding.
    """
    # lengths of each sub-array
    lengths = [len(x) for x in obj_array]
    max_len = max(lengths)

    out = np.full((len(obj_array), max_len), pad_value, dtype=float)
    for i, arr in enumerate(obj_array):
        arr = np.asarray(arr, dtype=float)
        out[i, :len(arr)] = arr
    return out

def to_array(val):
    if isinstance(val, str):
        try:
            return np.array(ast.literal_eval(val))
        except Exception:
            return np.array([])
    elif isinstance(val, (list, np.ndarray)):
        return np.array(val)
    else:
        return np.array([])


def get_plotly_config(filename_prefix="plot", height=1080/2, width=1800):
    """
    Get standardized Plotly configuration with download functionality.
    
    Parameters:
    - filename_prefix: Prefix for downloaded files
    - height: Chart height in pixels
    - width: Chart width in pixels
    
    Returns:
    - config: Plotly configuration dict with download options
    """
    return {
        'displayModeBar': True,
        'modeBarButtonsToAdd': ['toImage'],
        'toImageButtonOptions': {
            'format': 'svg',  # Default to SVG
            'filename': filename_prefix,
            'height': height,
            'width': width,
            'scale': 1
        }
    }
