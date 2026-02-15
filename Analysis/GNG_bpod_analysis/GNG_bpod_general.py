# Removed imports to avoid circular dependency

import os
import numpy as np
import ast
import streamlit as st

_FILTER_EARLY_RESPONSE_KEY = "filter_early_responses_global"


def ensure_global_early_response_setting(default: bool = True) -> None:
    """Ensure the global Early Response filter flag exists in session_state."""
    if _FILTER_EARLY_RESPONSE_KEY not in st.session_state:
        st.session_state[_FILTER_EARLY_RESPONSE_KEY] = default


def render_global_early_response_filter_checkbox(label: str = "Filter Early Response trials") -> None:
    """
    Render the single checkbox that controls Early Response filtering everywhere.
    """
    ensure_global_early_response_setting()
    st.checkbox(
        label,
        value=st.session_state[_FILTER_EARLY_RESPONSE_KEY],
        key=_FILTER_EARLY_RESPONSE_KEY,
        help="Exclude trials labeled 'Early Response' from all downstream analyses.",
    )


def get_global_early_response_filter(default: bool = True) -> bool:
    """Return the current global Early Response filter flag."""
    ensure_global_early_response_setting(default)
    return bool(st.session_state[_FILTER_EARLY_RESPONSE_KEY])


def getNameAndSession(project_data, index):
    mouse_name = project_data.iloc[index]['MouseName']
    # Find all session indices where this mouse appears
    session_indices = project_data.index[project_data["MouseName"] == mouse_name].tolist()

    # Get the session number corresponding to the current index
    current_session = session_indices.index(index) + 1  # Assuming session numbers are 1-based
    return mouse_name, current_session


def normalize_workspace_path(path: str) -> str:
    """
    Map cluster-style paths to the local Z: drive while leaving Z: paths untouched.
    """
    if not path:
        return path
    if path.lower().startswith("z:"):
        return path
    unix_prefix = "/ems/elsc-labs/mizrahi-a/"
    if path.startswith(unix_prefix):
        relative = path[len(unix_prefix):]
        return os.path.join("z:\\", relative.replace("/", os.sep))
    return path



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


def get_plotly_config(filename_prefix="plot", height=None, width=None):
    """
    Get standardized Plotly configuration with download functionality.
    
    Parameters:
    - filename_prefix: Prefix for downloaded files
    - height: Chart height in pixels (default: None = use rendered height)
    - width: Chart width in pixels (default: None = use rendered/container width)
    
    Returns:
    - config: Plotly configuration dict with download options
    """
    config = {
        'displayModeBar': True,
        'modeBarButtonsToAdd': ['toImage'],
        'toImageButtonOptions': {
            'format': 'svg',  # Default to SVG
            'filename': filename_prefix,
            'scale': 1,
            'bgcolor': 'rgba(0,0,0,0)'  # Transparent background
        }
    }
    # Only set height/width if explicitly provided; otherwise use rendered dimensions
    if height is not None:
        config['toImageButtonOptions']['height'] = height
    if width is not None:
        config['toImageButtonOptions']['width'] = width
    return config

def filter_out_catch_and_early_response(outcomes, trialtypes):
    # Filter out 'Catch' trials
    catch_mask = np.array([tt != 'Catch' for tt in trialtypes], dtype=bool)
    # Filter out trials with 'Early Response' outcomes
    early_response_mask = np.array(['Early Response' not in str(outcome) for outcome in outcomes])
    # Combine both masks (keep trials that pass both filters)
    combined_mask = catch_mask & early_response_mask
    return catch_mask, early_response_mask, combined_mask