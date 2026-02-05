import scipy.io
import pandas as pd
import numpy as np
import os as os
import shutil
from tqdm import tqdm
import re as re
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
from tkinter import Tk, filedialog
from typing import Any, Dict, List, Tuple, Optional


# =============================================================================
# Low-level MAT loading and parsing
# =============================================================================

def load_mat_file(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, Any, Any, Any, Any, List[np.ndarray], np.ndarray, Any, Any, Any, Any, bool, Optional[np.ndarray]]:
    """Load a single .mat Bpod file and extract per-trial data.

    The logic is unchanged from the original script; only typing was added.
    """
    try:
        mat_contents = scipy.io.loadmat(file_path)
    except Exception as e:
        print(f"[WARN] Failed to load .mat file {file_path}: {e}")
        raise
    
    # Check if SessionData exists
    if "SessionData" not in mat_contents:
        raise KeyError(f"File {file_path} does not contain 'SessionData' key - may not be a valid Bpod session file")

    # Check if FRA is in file path or in the directory path
    file_dir = os.path.dirname(file_path) if os.path.dirname(file_path) else ""
    is_fra = "FRA" in file_path or "FRA" in file_dir
    if is_fra:
        session_data_content = mat_contents["SessionData"][0, 0]
        try:
            _ = session_data_content["trialTable"]  # probe structure
            trial_settings = [session_data_content["SettingsFile"][0]]
            session_date = session_data_content["Info"]["SessionDate"][0][0]
            session_time = session_data_content["Info"]["SessionStartTime_UTC"][0][0]
            raw_events = session_data_content["RawEvents"][0, 0]
            stimuli = session_data_content["trialTable"]
            notes = ["FRA"]
            licks: List[np.ndarray] = []
            states: np.ndarray = np.array([])
            Unique_Stimuli_Values = np.unique(session_data_content["freq"])
            tones_per_class = np.unique(session_data_content["atten"])
            boundaries: List[Any] = []
            trial_types_df = pd.DataFrame()
            raw_events_df = pd.DataFrame()
            recs = False
            return (
                trial_types_df,
                raw_events_df,
                session_date,
                session_time,
                trial_settings,
                notes,
                licks,
                states,
                stimuli,
                Unique_Stimuli_Values,
                tones_per_class,
                boundaries,
                recs,
                None,  # OutcomeName not available for FRA files
            )
        except (KeyError, ValueError, IndexError):
            # Not actually an FRA file, fall through to regular processing
            pass

    # If we get here, either it's not FRA or the FRA file didn't have the expected structure
    session_data_content = mat_contents["SessionData"][0, 0]
    trial_settings = session_data_content["TrialSettings"][0]

    # Extract the date and time information
    session_date = session_data_content["Info"]["SessionDate"][0][0]
    session_time = session_data_content["Info"]["SessionStartTime_UTC"][0][0]

    # Extract 'TrialTypes' and 'RawEvents'
    trial_types = session_data_content["TrialTypes"][0]
    raw_events = session_data_content["RawEvents"][0, 0]
    
    # Extract 'OutcomeName' if available
    outcome_names: Optional[np.ndarray] = None
    try:
        outcome_names = session_data_content["OutcomeName"][0]
    except (KeyError, IndexError):
        pass  # OutcomeName not available, will use calculated outcomes

    count_recs = 0
    recs = False
    if ["Recording"] in trial_settings[0]["GUI"][0, 0]:
        for i in range(len(trial_settings)):
            count_recs += trial_settings[i]["GUI"][0, 0]["Recording"][0, 0]
            if count_recs > 10:
                recs = True
                break
    else:
        recs = False

    try:
        stimuli = session_data_content["stimulus"]
        if isinstance(stimuli, np.void):
            stimuli = stimuli.tolist()
        if isinstance(stimuli, (list, np.ndarray)) and len(stimuli) > 0:
            stimuli = stimuli[0]
        stimuli = np.array(stimuli)
    except Exception as e:  # noqa: F841
        print(f"Error extracting 'stimulus': {e}")
        stimuli = np.array([])

    try:
        notes = session_data_content["Notes"]
        if isinstance(notes, np.void):
            notes = notes.tolist()
        if isinstance(notes, (list, np.ndarray)) and len(notes) > 0:
            notes = notes[0]
        if not isinstance(notes, str):
            notes = ""

        num_unique_stimuli = len(np.unique(stimuli))
        if num_unique_stimuli == 1:
            notes = [notes, "TA"]
        elif num_unique_stimuli == 2:
            notes = [notes, "Discrimination"]
        else:
            notes = [notes, "Categorization"]
    except Exception:
        num_unique_stimuli = len(np.unique(stimuli))
        if num_unique_stimuli == 1:
            notes = ["TA", "TA"]
        elif num_unique_stimuli == 2:
            notes = ["Discrimination"]
        else:
            notes = ["Categorization"]

    Unique_Stimuli_Count = len(np.unique(stimuli))
    Unique_Stimuli_Values = np.unique(stimuli)

    def check_stimuli_range(unique_values: np.ndarray) -> bool:
        has_high = any(val > 1.5 for val in unique_values)
        has_low = any(val < 1 for val in unique_values)
        return has_high and has_low

    Has_High_And_Low_Stimuli = check_stimuli_range(Unique_Stimuli_Values)
    boundaries = 2 if Has_High_And_Low_Stimuli else 1
    n_classes = 3 if Has_High_And_Low_Stimuli else 2
    tones_per_class = Unique_Stimuli_Count // n_classes
    notes.append(f"{boundaries}b_{tones_per_class}t")

    trials = raw_events["Trial"]
    n_trials = len(trial_types)
    n_trials_available = trials.shape[1]
    n_trials_to_process = min(n_trials, n_trials_available)
    if n_trials != n_trials_available:
        print(
            f"Warning: Trial count mismatch - trial_types: {n_trials}, raw_events trials: {n_trials_available}. "
            f"Processing {n_trials_to_process} trials."
        )

    licks: List[Optional[np.ndarray]] = [None] * n_trials_to_process
    for i in range(n_trials_to_process):
        try:
            trial_element = trials[0, i]
            nested_structure = trial_element[0, 0]
            trial_states = nested_structure["States"][0, 0]
            states_array = np.empty((len(trial_states.dtype.names), 2), dtype=object)
            for s, state in enumerate(trial_states.dtype.names):
                states_array[s, 0] = state
                states_array[s, 1] = np.ravel(trial_states[state])

            states = np.array(
                [row for row in states_array if not any(np.isnan(np.ravel(row[1])))]
            )

            if "Events" in nested_structure.dtype.names:
                events = nested_structure["Events"][0, 0]

                if "HiFi1_1" in events.dtype.names:
                    stim_t = np.min(events["HiFi1_1"][0])
                else:
                    stim_t = None

                if "Port1In" in events.dtype.names and stim_t is not None:
                    lick = events["Port1In"][0]
                    lick_after_stim = lick[lick > stim_t]
                    if lick_after_stim.size > 0:
                        licks[i] = lick_after_stim - stim_t
        except Exception as e:  # noqa: F841
            print(f"Error processing trial {i}: {e}")

    trial_types_df = pd.DataFrame(trial_types[:n_trials_to_process], columns=["TrialType"])
    stimuli_df = pd.DataFrame(stimuli[:n_trials_to_process], columns=["Stimuli"])
    trial_types_df = pd.concat([trial_types_df, stimuli_df], axis=1)

    trial_raw_events_data = raw_events["Trial"][0]
    states_timestamps_info = extract_states_timestamps(trial_raw_events_data)
    raw_events_df = pd.DataFrame(states_timestamps_info)

    return (
        trial_types_df,
        raw_events_df,
        session_date,
        session_time,
        trial_settings,
        notes,
        licks,
        states,
        stimuli,
        Unique_Stimuli_Values,
        tones_per_class,
        boundaries,
        recs,
        outcome_names,
    )


def extract_states_timestamps(trial_data: np.ndarray) -> List[Dict[str, np.ndarray]]:
    states_info: List[Dict[str, np.ndarray]] = []
    for trial in trial_data:
        states = trial["States"][0][0]
        state_names = states.dtype.names
        timestamps = {name: states[name][0] for name in state_names}
        states_info.append(timestamps)
    return states_info


# =============================================================================
# Aggregation helpers
# =============================================================================

def create_single_row_with_outcome(
    file_path: str,
    trial_types_df: pd.DataFrame,
    raw_events_df: pd.DataFrame,
    session_date: Any,
    session_time: Any,
    trial_settings: Any,
    notes: Any,
    licks: List[Optional[np.ndarray]],
    states: np.ndarray,
    Unique_Stimuli_Values: Any,
    tones_per_class: Any,
    boundaries: Any,
    recs: bool,
    outcome_names: Optional[np.ndarray] = None,
    group_name: str = "",
) -> pd.DataFrame:
    file_name = os.path.basename(file_path)
    mouse_name = file_name.split("_")[0]

    trial_types_list: List[str] = []
    outcomes_list: List[str] = []

    def contains_nan(cell: Any) -> bool:
        if isinstance(cell, (np.ndarray, list)):
            try:
                return bool(np.any(np.isnan(cell)))
            except TypeError:
                return any(
                    np.any(np.isnan(sub_cell)) if isinstance(sub_cell, (np.ndarray, list)) else False
                    for sub_cell in cell
                )
        return False

    rewards = raw_events_df["Reward"].apply(lambda x: not contains_nan(x))
    punishments = raw_events_df["Punishment"].apply(lambda x: not contains_nan(x))

    min_length = min(len(trial_types_df), len(raw_events_df))
    if len(trial_types_df) != len(raw_events_df):
        print(
            f"Warning: Length mismatch - trial_types_df: {len(trial_types_df)}, raw_events_df: {len(raw_events_df)}. "
            f"Using first {min_length} trials."
        )

    for i in range(min_length):
        trial_type = trial_types_df.iloc[i]["TrialType"]
        reward = rewards.iloc[i]
        punishment = punishments.iloc[i]

        if trial_type == 1:
            trial_type_str = "Go"
        elif trial_type == 2:
            trial_type_str = "NoGo"
        else:
            trial_type_str = "Unknown"

        trial_types_list.append(trial_type_str)

        # Use OutcomeName from SessionData if available, otherwise calculate
        if outcome_names is not None and i < len(outcome_names):
            try:
                outcome = outcome_names[i]
                # Handle numpy array/string conversion
                if isinstance(outcome, np.ndarray):
                    outcome = outcome.item() if outcome.size > 0 else None
                if outcome is None or (isinstance(outcome, str) and len(outcome) == 0):
                    raise ValueError("Empty outcome")
                # Convert to string if needed
                outcome = str(outcome).strip()
            except (ValueError, IndexError, AttributeError):
                # Fall back to calculated outcome
                if trial_type_str == "Go":
                    outcome = "Hit" if reward else "Miss"
                elif trial_type_str == "NoGo":
                    outcome = "False Alarm" if punishment else "CR"
                else:
                    outcome = "Unknown"
        else:
            # Calculate outcome from trial type and rewards/punishments
            if trial_type_str == "Go":
                outcome = "Hit" if reward else "Miss"
            elif trial_type_str == "NoGo":
                outcome = "False Alarm" if punishment else "CR"
            else:
                outcome = "Unknown"

        outcomes_list.append(outcome)

    water = calculate_water_consumption(rewards, trial_settings)

    combined_data: Dict[str, Any] = {
        "Checkbox": "False",
        "MouseName": mouse_name,
        "SessionDate": session_date[0],
        "SessionTime": session_time[0],
        "TrialTypes": trial_types_list,
        "States": states,
        "Outcomes": outcomes_list,
        "Stimuli": trial_types_df["Stimuli"].iloc[:min_length].values,
        "Licks": licks[:min_length],
        "WaterConsumption": water,
        "FilePath": file_path,
        "Notes": notes,
        "Recording": recs,
        "Unique_Stimuli_Values": Unique_Stimuli_Values,
        "Tones_per_class": tones_per_class,
        "N_Boundaries": boundaries,
        "groupID": group_name,
        "Setup": "Bpod",
    }

    combined_df = pd.DataFrame([combined_data])
    return combined_df


def calculate_water_consumption(rewards: pd.Series, trial_settings: Any) -> float:
    water_consumed: float = 0.0
    for idx, reward in enumerate(rewards):
        if reward:
            try:
                reward_amount = trial_settings[idx]["GUI"][0, 0]["RewardAmount"][0, 0]
                reward_amount = float(reward_amount)
                water_consumed += reward_amount
            except (ValueError, TypeError, IndexError) as e:
                print(f"Warning: Could not extract reward amount for trial {idx}: {e}")
                continue
    return water_consumed


def save_combined_data_to_df(df: pd.DataFrame, combined_row_df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return combined_row_df
    if combined_row_df.empty:
        return df
    return pd.concat([df, combined_row_df], ignore_index=True)


def extract_subject_id_from_path(file_path: str) -> Optional[str]:
    """Extract subject ID (e.g., G7A1) from file path or name."""
    subject_regex = re.compile(r"G\d+A\d+", re.IGNORECASE)
    path = Path(file_path)
    for part in [path.name] + [p.name for p in path.parents]:
        m = subject_regex.search(part)
        if m:
            return m.group(0).upper()
    return None


def infer_datetime_from_bpod_file(file_path: str) -> Optional[datetime]:
    """
    Infer full session datetime for a Bpod .mat file.
    Preferred layout: {id}_{protocol}_{YYYYMMDD}_{HHMMSS}.mat
    Example: G7A1_FRA_20251112_102435.mat
    Falls back to file modification time if pattern doesn't match.
    """
    path = Path(file_path)
    stem = path.stem
    
    # Strict pattern for the documented format
    m = re.match(
        r"^(?P<id>G\d+A\d+)_(?P<protocol>[A-Za-z0-9]+)_(?P<ymd>20\d{2}\d{2}\d{2})_(?P<hms>\d{6})$",
        stem,
    )
    if m:
        ymd = m.group("ymd")
        hms = m.group("hms")
        try:
            y = int(ymd[0:4])
            mon = int(ymd[4:6])
            d = int(ymd[6:8])
            h = int(hms[0:2])
            mi = int(hms[2:4])
            s = int(hms[4:6])
            return datetime(y, mon, d, h, mi, s)
        except ValueError:
            pass
    
    # Fallback: use file modification time
    try:
        mtime = os.path.getmtime(file_path)
        return datetime.fromtimestamp(mtime)
    except OSError:
        return None


def session_label_from_datetime(session_dt: datetime, *, bin_seconds: int = 90) -> str:
    """
    Compute a session folder label from a datetime, quantized into bins
    of `bin_seconds`. This ensures behavior and ephys sessions that occur
    within the same ~90s window share the same session folder.
    """
    seconds = session_dt.hour * 3600 + session_dt.minute * 60 + session_dt.second
    bin_index = seconds // bin_seconds
    label_seconds = bin_index * bin_seconds
    h = label_seconds // 3600
    rem = label_seconds % 3600
    m = rem // 60
    s = rem % 60
    return f"{h:02d}-{m:02d}-{s:02d}"


def find_mat_files_in_session_data(directory: str, organize: bool = True) -> List[str]:
    """
    Find .mat files in session data directory and optionally move them to organized structure.
    
    Args:
        directory: Source directory to search for .mat files
        organize: If True, move files to organized structure (Z:\\Shared\\Amichai\\Data)
    
    Returns:
        List of paths to .mat files (from organized structure if moved, otherwise from source)
    """
    organized_root = Path(r"Z:\Shared\Amichai\Data")
    mat_files: List[str] = []
    files_moved = 0
    files_skipped = 0
    
    # Find files in source directory
    source_files: List[str] = []
    for root, dirs, files in os.walk(directory):  # noqa: B007
        if "Session Data" in root and "GNG" in os.path.dirname(root) and "Original_Files" not in root:
            for file in files:
                if file.endswith(".mat") and file != "DefaultSettings.mat":
                    source_files.append(os.path.join(root, file))
    
    if not organize:
        # Return source files without moving
        return source_files
    
    # Move files to organized structure
    organized_root.mkdir(parents=True, exist_ok=True)
    
    for mat_file_path in source_files:
        mat_file = Path(mat_file_path)
        
        # Extract metadata
        subject_id = extract_subject_id_from_path(str(mat_file))
        if subject_id is None:
            files_skipped += 1
            continue
        
        session_dt = infer_datetime_from_bpod_file(str(mat_file))
        if session_dt is None:
            files_skipped += 1
            continue
        
        # Skip files that are less than 90 minutes old (recently modified/created)
        time_diff = datetime.now() - session_dt
        if time_diff < timedelta(minutes=90):
            files_skipped += 1
            continue
        
        # Compute destination path: [Root]/[Subject_ID]/[YYYY-MM-DD]/[Session_Label]/behavior/
        date_str = session_dt.date().isoformat()
        session_label = session_label_from_datetime(session_dt)
        dest_dir = organized_root / subject_id / date_str / session_label / "behavior"
        dest_file = dest_dir / mat_file.name
        
        # Skip if destination already exists
        if dest_file.exists():
            # File already organized, add to return list
            mat_files.append(str(dest_file))
            files_skipped += 1
            continue
        
        # Move the file
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(mat_file), str(dest_file))
            mat_files.append(str(dest_file))
            files_moved += 1
        except (OSError, shutil.Error) as e:
            print(f"[WARN] Failed to move {mat_file}: {e}")
            files_skipped += 1
            # Add source file to return list if move failed
            mat_files.append(str(mat_file))
    
    if files_moved > 0 or files_skipped > 0:
        print(f"Moved {files_moved} files to organized structure, skipped {files_skipped} files")
    
    return mat_files


def find_mat_files_in_organized_structure(group_number: str) -> List[str]:
    """
    Search for .mat files in the organized data structure (Z:\\Shared\\Amichai\\Data)
    for a specific group number.
    
    Args:
        group_number: Group number as string (e.g., "1", "5")
    
    Returns:
        List of paths to .mat files found in folders matching G{group_number}A* pattern
    """
    organized_root = r"Z:\Shared\Amichai\Data"
    mat_files: List[str] = []
    
    if not os.path.exists(organized_root):
        return mat_files
    
    # Pattern to match subject folders: G{group_number}A{animal_number}
    # e.g., G1A1, G1A2, G5A1, etc.
    pattern = re.compile(rf"^G{group_number}A\d+$", re.IGNORECASE)
    
    # Walk through the organized structure
    for root, dirs, files in os.walk(organized_root):  # noqa: B007
        # Check if current directory name matches the group pattern
        current_dir = os.path.basename(root)
        if pattern.match(current_dir):
            # Found a subject folder for this group, search for .mat files
            for file in files:
                if file.endswith(".mat") and file != "DefaultSettings.mat":
                    mat_files.append(os.path.join(root, file))
        else:
            # Check parent directories in the path
            path_parts = root.split(os.sep)
            for part in path_parts:
                if pattern.match(part):
                    # Found a matching subject folder in the path, search for .mat files
                    for file in files:
                        if file.endswith(".mat") and file != "DefaultSettings.mat":
                            full_path = os.path.join(root, file)
                            if full_path not in mat_files:  # Avoid duplicates
                                mat_files.append(full_path)
                    break
    
    return mat_files



# =============================================================================
# Directory & CSV helpers
# =============================================================================

def get_csv_path(directory_path: str) -> Tuple[str, str]:
    group_name = os.path.split(directory_path)[-1]
    if "_" in group_name:
        group_name = group_name.replace("_", " ")
    csv_path = rf"Z:\Shared\Amichai\Code\DB\users_data\Amichai\{group_name}_experimental_data.csv"
    return group_name, csv_path


CONFIG_FILE_PATH = "last_directory.json"


def get_last_directory(default_path: Optional[str] = None) -> Optional[str]:
    """Retrieve the last chosen directory from the config file."""
    if os.path.exists(CONFIG_FILE_PATH):
        with open(CONFIG_FILE_PATH, "r") as f:
            config = json.load(f)
            return config.get("last_directory", default_path)
    return default_path


def save_last_directory(directory_path: str) -> None:
    """Save the chosen directory to the config file."""
    with open(CONFIG_FILE_PATH, "w") as f:
        json.dump({"last_directory": directory_path}, f)


def choose_directory(default_path: str) -> str:
    """Open a directory chooser dialog with the default path."""
    root = Tk()
    root.withdraw()  # Hide the main Tkinter window
    root.attributes("-topmost", True)  # Bring the dialog to the front
    directory_path = filedialog.askdirectory(initialdir=default_path, title="Select Directory")
    root.destroy()
    return directory_path


# =============================================================================
# Public processing API
# =============================================================================

def process_bpod_directory(directory_path: str, group_number: str) -> str:
    """Process all Bpod .mat files under the given directory and update the per-group CSV.

    Returns the path to the updated CSV file.
    """
    group_name, csv_path = get_csv_path(directory_path)

    # Always start with fresh DataFrame (overwrite previous CSV)
    df = pd.DataFrame(columns=["Checkbox"])

    # Add the group to the project list if it does not exist yet
    project_list_path = r"Z:\Shared\Amichai\Code\DB\users_data\Amichai\projects_list.csv"
    project_list = pd.read_csv(project_list_path)

    if group_name not in project_list["Project Name"].values:
        new_project = pd.DataFrame(
            {
                "Project Name": [group_name],
                "Project Type": ["['Behavior-Bpod GUI']"],
                "Project Description": ["Automatically added group"],
            }
        )
        project_list = pd.concat([project_list, new_project], ignore_index=True)
        project_list.to_csv(project_list_path, index=False)
    
    # First, move files from session data to organized structure
    print(f"Organizing files from {directory_path}...")
    find_mat_files_in_session_data(directory_path, organize=True)
    
    # Then, find files in the organized structure for this group
    print(f"Searching for files in organized structure for Group {group_number}...")
    mat_files_list = find_mat_files_in_organized_structure(group_number)
    
    for mat_file in tqdm(mat_files_list, desc="Processing .mat files"):
        try:
            (
                trial_types_df,
                raw_events_df,
                session_date,
                session_time,
                trial_settings,
                notes,
                licks,
                states,
                stimuli,
                Unique_Stimuli_Values,
                tones_per_class,
                boundaries,
                recs,
                outcome_names,
            ) = load_mat_file(mat_file)
        except (KeyError, ValueError, IndexError, OSError) as e:
            print(f"[WARN] Skipping file {mat_file}: {e}")
            continue
        except Exception as e:
            print(f"[WARN] Unexpected error processing {mat_file}: {e}")
            continue


        # Skip short or fake sessions
        if len(trial_types_df) < 50 or "Fake" in mat_file:
            continue

        combined_row_df = create_single_row_with_outcome(
            mat_file,
            trial_types_df,
            raw_events_df,
            session_date,
            session_time,
            trial_settings,
            notes,
            licks,
            states,
            Unique_Stimuli_Values,
            tones_per_class,
            boundaries,
            recs,
            outcome_names,
            group_name,
        )
        df = save_combined_data_to_df(df, combined_row_df)

    # Final cleanup and sort
    df = df.dropna(subset=["SessionDate"])
    df["SessionDate"] = df["SessionDate"].apply(lambda x: str(x))
    df["SessionDate"] = pd.to_datetime(df["SessionDate"], format="mixed")
    df = df.sort_values(by=["SessionDate", "SessionTime"]).reset_index(drop=True)
    df.to_csv(csv_path, index=False)

    return csv_path


def main(directory_path: Optional[str] = None, group_number: Optional[str] = None) -> str:
    """Main entry point used by CLI and Streamlit.

    If directory_path is None, fall back to the original Tk-based chooser flow.
    """
    if directory_path is None:
        default_directory = r"Z:\Shared\Amichai\Behavior\data"
        last_directory = get_last_directory(default_directory)

        print(f"Last directory: {last_directory}")
        new_directory_path = choose_directory(last_directory or default_directory)

        if new_directory_path:
            directory_path = new_directory_path
            save_last_directory(directory_path)
            print(f"New directory selected: {directory_path}")
        else:
            directory_path = last_directory
            print(f"Using last directory: {directory_path}")

    if directory_path is None:
        raise ValueError("No directory_path provided or selected.")

    return process_bpod_directory(directory_path, group_number)


if __name__ == "__main__":
    cli_directory = sys.argv[1] if len(sys.argv) > 1 else None
    main(cli_directory)
