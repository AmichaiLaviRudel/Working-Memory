
import scipy.io
import pandas as pd
import numpy as np
import os as os
from tqdm import tqdm
import re as re

# file_path = "Z:/Shared/Amichai/Behavior/data/Group_4\A6\GNG\Session Data\A6_GNG_20250408_085814.mat"
def load_mat_file(file_path):
    # Load the .mat file

    mat_contents = scipy.io.loadmat(file_path)

    # Extract the 'SessionData' field
    session_data_content = mat_contents['SessionData'][0, 0]
    trial_settings = session_data_content['TrialSettings'][0]

    # Extract the date and time information
    session_date = session_data_content['Info']['SessionDate'][0][0]
    session_time = session_data_content['Info']['SessionStartTime_UTC'][0][0]

    # Extract 'TrialTypes' and 'RawEvents'
    trial_types = session_data_content['TrialTypes'][0]
    raw_events = session_data_content['RawEvents'][0, 0]


    try:
        # Extract 'Stimulus' field from 'session_data_content'
        stimuli = session_data_content['stimulus']

        # If 'stimulus' is a NumPy structured array or void object, convert it
        if isinstance(stimuli, np.void):
            stimuli = stimuli.tolist()  # Convert structured array to a list

        # If 'stimulus' is a nested array, extract the first element
        if isinstance(stimuli, (list, np.ndarray)) and len(stimuli) > 0:
            stimuli = stimuli[0]

        # Ensure stimuli is a proper NumPy array for further operations
        stimuli = np.array(stimuli)

    except Exception as e:
        print(f"Error extracting 'stimulus': {e}")
        stimuli = np.array([])  # Default to empty NumPy array

    try:
        # Extract 'Notes' correctly (handling NumPy void object)
        notes = session_data_content['Notes']

        # If 'Notes' is a NumPy void, convert it to a standard Python object
        if isinstance(notes, np.void):
            notes = notes.tolist()  # Convert structured array to list

        # Extract the first element if it's a nested list or array
        if isinstance(notes, (list, np.ndarray)) and len(notes) > 0:
            notes = notes[0]

        # Ensure 'notes' is a string (or convert to an empty string if needed)
        if not isinstance(notes, str):
            notes = ''

        # Determine task type based on the number of unique stimuli
        num_unique_stimuli = len(np.unique(stimuli))

        if num_unique_stimuli == 1:
            notes = [notes, 'TA']
        elif num_unique_stimuli == 2:
            notes = [notes, 'Discrimination']
        else:  # Covers num_unique_stimuli > 2
            notes = [notes, 'Categorization']

    except Exception as e:
        print(f"Error extracting 'Notes': {e}")

    # Determine task type based on number of unique stimuli
    Unique_Stimuli_Count = len(np.unique(stimuli))
    Unique_Stimuli_Values = np.unique(stimuli)

    # Function to check if a row contains at least one stimulus > 1.5 and one < 1
    def check_stimuli_range(unique_values):

        has_high = any(val > 1.5 for val in unique_values)  # Check for value > 1.5
        has_low = any(val < 1 for val in unique_values)  # Check for value < 1
        return has_high and has_low


    # Apply the function to create a new boolean column
    Has_High_And_Low_Stimuli = check_stimuli_range(Unique_Stimuli_Values)
    boundaries = 2 if Has_High_And_Low_Stimuli else 1
    n_classes = 3 if Has_High_And_Low_Stimuli else 2
    tones_per_class = Unique_Stimuli_Count // n_classes
    notes.append(f"{boundaries}b_{tones_per_class}t")


    # Initialize arrays
    trials = raw_events['Trial']
    n_trials = len(trial_types)
    licks = [None] * n_trials
    for i in range(n_trials):
        try:
            trial_element = trials[0, i]
            nested_structure = trial_element[0, 0]
            trial_states = nested_structure['States'][0, 0]
            states_array = np.empty((len(trial_states.dtype.names), 2), dtype = object)
            for s, state in enumerate(trial_states.dtype.names):
                states_array[s, 0] = state
                states_array[s, 1] = np.ravel(trial_states[state])

            # Convert to numpy array and remove rows where any element is NaN
            states = np.array([row for row in states_array if not any(np.isnan(np.ravel(row[1])))])

            if 'Events' in nested_structure.dtype.names:
                events = nested_structure['Events'][0, 0]

                if 'HiFi1_1' in events.dtype.names:
                    stim_t = np.min(events['HiFi1_1'][0])
                else:
                    stim_t = None

                if 'Port1In' in events.dtype.names and stim_t is not None:
                    lick = events['Port1In'][0]
                    lick_after_stim = lick[lick > stim_t]

                    if lick_after_stim.size > 0:
                        licks[i] = lick_after_stim - stim_t
        except Exception as e:
            print(f"Error processing trial {i}: {e}")

    # Convert 'TrialTypes' and stimuli to a DataFrame
    trial_types_df = pd.DataFrame(trial_types, columns = ['TrialType'])
    stimuli_df = pd.DataFrame(stimuli, columns = ['Stimuli'])
    trial_types_df = pd.concat([trial_types_df, stimuli_df], axis = 1)

    # Extract the 'Trial' field from 'RawEvents'
    trial_raw_events_data = raw_events['Trial'][0]

    # Extract states and timestamps from the 'Trial' data
    states_timestamps_info = extract_states_timestamps(trial_raw_events_data)
    # Convert the extracted information into a DataFrame
    raw_events_df = pd.DataFrame(states_timestamps_info)

    return trial_types_df, raw_events_df, session_date, session_time, trial_settings, notes, licks, states, stimuli, Unique_Stimuli_Values, tones_per_class, boundaries

# Function to extract states and timestamps from 'Trial' data in 'RawEvents'
def extract_states_timestamps(trial_data):
    states_info = []
    for trial in trial_data:
        states = trial['States'][0][0]
        state_names = states.dtype.names
        timestamps = {name: states[name][0] for name in state_names}
        states_info.append(timestamps)
    return states_info


def create_single_row_with_outcome(file_path, trial_types_df, raw_events_df, session_date, session_time, trial_settings, notes, licks, states, Unique_Stimuli_Values, tones_per_class, boundaries):
    file_name = os.path.basename(file_path)

    # Split the file name to extract the required part
    mouse_name = file_name.split('_')[0]

    trial_types_list = []
    outcomes_list = []

    def contains_nan(cell):
        if isinstance(cell, (np.ndarray, list)):
            try:
                return np.any(np.isnan(cell))
            except TypeError:
                return any(
                    np.any(np.isnan(sub_cell)) if isinstance(sub_cell, (np.ndarray, list)) else False for sub_cell in
                    cell)
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

    # Calculate water consumed in the session
    water = calculate_water_consumption(rewards, trial_settings)

    combined_data = {
        'Checkbox':         'False',
        'MouseName':        mouse_name,
        'SessionDate':      session_date[0],
        'SessionTime':      session_time[0],
        'TrialTypes':       trial_types_list,
        'States':           states,
        'Outcomes':         outcomes_list,
        'Stimuli':          trial_types_df["Stimuli"].values,
        'Licks':            licks,
        'WaterConsumption': water,
        'FilePath':         file_path,
        'Notes':            notes,
        'Unique_Stimuli_Values': Unique_Stimuli_Values,
        'Tones_per_class': tones_per_class,
        'N_Boundaries': boundaries
    }

    combined_df = pd.DataFrame([combined_data])
    return combined_df

def calculate_water_consumption(rewards, trial_settings):
    # Calculate the total water consumed based on reward times and settings
    water_consumed = 0.0  # Use float to avoid overflow
    for idx, reward in enumerate(rewards):
        if reward:
            try:
                # Extract the reward amount based on the reward time
                reward_amount = trial_settings[idx]['GUI'][0, 0]['RewardAmount'][0, 0]
                # Convert to float to ensure proper arithmetic
                reward_amount = float(reward_amount)
                water_consumed += reward_amount
            except (ValueError, TypeError, IndexError) as e:
                print(f"Warning: Could not extract reward amount for trial {idx}: {e}")
                continue
    return water_consumed


def save_combined_data_to_df(df, combined_row_df):
    df = pd.concat([df, combined_row_df], ignore_index = True)
    return df



def find_mat_files_in_session_data(directory):
    # List to store paths of .mat files
    mat_files = []

    # Walk through the directory
    for root, dirs, files in os.walk(directory):
        # Check if 'Session Data' is in the current directory path
        if 'Session Data' in root and 'GNG' in os.path.dirname(root):
            for file in files:
                if file.endswith('.mat'):
                    # Add the full path of the .mat file to the list
                    mat_files.append(os.path.join(root, file))
    return mat_files


def is_processed(file_path):
    # Check if the file has already been processed
    # This is a placeholder for your logic to determine if the file was processed
    # Example: Check if a .done file exists for this .mat file
    processed_flag = file_path + '.done'
    return os.path.exists(processed_flag)


def mark_as_processed(file_path):
    # Create a flag file to mark the .mat file as processed
    processed_flag = file_path + '.done'
    with open(processed_flag, 'w') as f:
        f.write('Processed')


# the path of the folder to scan
import json
import os
from tkinter import Tk, filedialog

# Path to the configuration file
config_file_path = "last_directory.json"


def get_last_directory(default_path=None):
    """Retrieve the last chosen directory from the config file."""
    if os.path.exists(config_file_path):
        with open(config_file_path, 'r') as f:
            config = json.load(f)
            return config.get("last_directory", default_path)
    return default_path


def save_last_directory(directory_path):
    """Save the chosen directory to the config file."""
    with open(config_file_path, 'w') as f:
        json.dump({"last_directory": directory_path}, f)


def choose_directory(default_path):
    """Open a directory chooser dialog with the default path."""
    root = Tk()
    root.withdraw()  # Hide the main Tkinter window
    root.attributes("-topmost", True)  # Bring the dialog to the front
    directory_path = filedialog.askdirectory(initialdir = default_path, title = "Select Directory")
    root.destroy()
    return directory_path


# Main logic to handle directory selection
default_directory = r"Z:\Shared\Amichai\Behavior\data"  # Replace with your hardcoded default
last_directory = get_last_directory(default_directory)

print(f"Last directory: {last_directory}")
new_directory_path = choose_directory(last_directory)

if new_directory_path:
    directory_path = new_directory_path
    save_last_directory(directory_path)
    print(f"New directory selected: {directory_path}")
else:
    directory_path = last_directory
    print(f"Using last directory: {directory_path}")


def get_csv_path(directory_path):
    # get group name the last folder in the path
    group_name = os.path.split(directory_path)[-1]
    # if there is a underscore in the group name, replace it with a space
    if '_' in group_name:
        group_name = group_name.replace('_', ' ')
    csv_path = fr'Z:\Shared\Amichai\Code\DB\users_data\Amichai\{group_name}_experimental_data.csv'
    return group_name, csv_path



if __name__ == "__main__":
    group_name, csv_path = get_csv_path(directory_path)
    # Load the existing CSV into a DataFrame
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        # Create an empty DataFrame with a 'file_path' column if the CSV doesn't exist
        df = pd.DataFrame(columns = ['Checkbox'])

        # add the group to the project list
        project_list = pd.read_csv(r"Z:\Shared\Amichai\Code\DB\users_data\Amichai\\projects_list.csv")

        # check if the group is already in the project list
        if group_name in project_list['Project Name'].values:
            pass
        else:
            new_project = pd.DataFrame({'Project Name':        group_name, 'Project Type': "['Behavior-Bpod GUI']",
                                        'Project Description': 'Automatically added group'}, index = [0])
            project_list = pd.concat([project_list, new_project], ignore_index = True)
            project_list.to_csv(r"Z:\Shared\Amichai\Code\DB\users_data\Amichai\projects_list.csv",
                                index = False)

    # Get all .mat files in 'Session Data' directories
    mat_files_list = find_mat_files_in_session_data(directory_path)

    # Process each .mat file if not already done
    for mat_file in tqdm(mat_files_list, desc = "Processing .mat files"):
        if not is_processed(mat_file):
            trial_types_df, raw_events_df, session_date, session_time, trial_settings, notes, licks, states,  stimuli, Unique_Stimuli_Values, tones_per_class, boundaries = load_mat_file(mat_file)
            if len(trial_types_df) < 50 or "Fake" in mat_file:
                continue

            combined_row_df = create_single_row_with_outcome(mat_file, trial_types_df, raw_events_df, session_date,
                                                             session_time, trial_settings, notes, licks, states,
                                                             Unique_Stimuli_Values, tones_per_class, boundaries)
            df = save_combined_data_to_df(df, combined_row_df)
            mark_as_processed(mat_file)  # Mark the file as processed
    #
    # Drop rows that have NaN values in the 'SessionDate' column
    df = df.dropna(subset = ['SessionDate'])
    # Convert 'SessionDate' to regular Python string type before converting to datetime
    df['SessionDate'] = df['SessionDate'].apply(lambda x: str(x))

    # Convert to datetime
    df['SessionDate'] = pd.to_datetime(df['SessionDate'])

    # Sort the DataFrame by session date and time
    df = df.sort_values(by = ['SessionDate', 'SessionTime']).reset_index(drop = True)

    # Save the updated DataFrame back to the CSV
    df.to_csv(csv_path, index = False)

