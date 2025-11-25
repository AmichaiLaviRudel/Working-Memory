import sys
import os
import re

# Add the workspace root to Python path to enable imports
workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from load_data.load_bpod_data import load_mat_file, create_single_row_with_outcome
import numpy as np
import pandas as pd
from scipy.stats import norm

import shutil


# --- Utility Functions ---
def find_ks_folders(root_directory, target_dir="bombcell"):
    """
    Recursively search for all folders within root_directory that contain a subdirectory named target_dir (default: 'bombcell').
    Returns a list of paths to these parent folders.
    """
    ks_folders = []
    for dirpath, dirnames, filenames in os.walk(root_directory):
        if target_dir in dirnames:
            ks_folders.append(dirpath)
    return ks_folders

def read_sample_rate(folder):
    """
    Reads the sample rate from a .meta file in the given folder. If not found, returns a default of 30000 Hz.
    Parameters:
        folder (str): Path to the folder containing the .meta file.
    Returns:
        float: The sample rate in Hz.
    """
    import glob
    meta_files = glob.glob(os.path.join(folder, "*.meta"))
    if meta_files:
        with open(meta_files[0], 'r') as f:
            for line in f:
                if 'imSampRate' in line:
                    return float(line.strip().split('=')[-1])
    return 30000  # fallback default

def read_bin_size_from_metadata(folder, default_bin_size=0.005):
    """
    Reads the bin size from the event_windows_metadata.txt file created by NPXL_Preprocessing.py.
    If not found, returns a default bin size.
    
    Parameters:
        folder (str): Path to the folder containing the event_windows_metadata.txt file.
        default_bin_size (float): Default bin size in seconds if metadata file not found.
    Returns:
        float: The bin size in seconds.
    """
    metadata_file = os.path.join(folder, "event_windows_metadata.txt")
    
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r') as f:
                for line in f:
                    if 'bin_size:' in line:
                        bin_size = float(line.strip().split(': ')[1])
                        print(f"Read bin size from metadata: {bin_size:.6f}s")
                        return bin_size
        except Exception as e:
            print(f"Error reading bin size from metadata: {e}")
    
    print(f"Using default bin size: {default_bin_size:.6f}s")
    return default_bin_size

def load_cluster_info(working_dir, data_dirs):
    """
    Loads cluster information from a bombcell unit_labels.tsv or cluster_info.tsv file in the working directory.

    Parameters:
        working_dir (str): Path to the working directory (e.g., imec1_ks4).
        data_dirs (str): Path to the data directory (used as fallback).

    Returns:
        pd.DataFrame or None: DataFrame with cluster info, or None if not found.
    """
    # Normalize paths to handle both forward and backward slashes
    working_dir = working_dir.replace("\\", "/")
    data_dirs = data_dirs.replace("\\", "/") if data_dirs else None
    
    # working_dir, data_dirs = current_ks_folder, ks4_data_dir 
    # The bombcell folder is inside the working_dir (imec*_ks4)
    label_dir = os.path.join(working_dir, "bombcell", "unit_labels.tsv").replace("\\", "/")
    if os.path.isfile(label_dir):
        return pd.read_csv(label_dir, sep='\t')
    elif "cluster_bc_unitType.tsv" in os.listdir(data_dirs):
        data = pd.read_csv(os.path.join(data_dirs, "cluster_bc_unitType.tsv").replace("\\", "/"), sep='\t')
        # save the data to the working_dir
        data_to_bc = data.copy()
        data_to_bc.columns = ["unitID", "UnitType"]
        data_to_bc.index = data_to_bc["unitID"]
        data_to_bc.index.name = "cluster_index"
        
        # Map unit types to numeric values
        unit_type_mapping = {
            'NOISE': 0,
            'GOOD': 1,
            'MUA': 2,
            'NON-SOMA': 3
        }
        data_to_bc['UnitType'] = data_to_bc['UnitType'].map(unit_type_mapping)
        
        data_to_bc.to_csv(label_dir, sep='\t', index=False)
        return data
    elif "cluster_info.tsv" in os.listdir(data_dirs):
        label_dir = os.path.join(data_dirs, "cluster_info.tsv")
        if os.path.isfile(label_dir):
            return pd.read_csv(label_dir, sep='\t')
    else:
        print(f"No cluster info found in {working_dir} or {data_dirs}")
        return None

def extract_spike_matrices(spikes_times, spikes_clusters, cluster_info):
    """
    Extracts spike times for good, MUA, and non-somatic clusters from the provided cluster info.
    Returns a [cluster x spike times] matrix, padded with 0 (not np.nan), and the cluster metadata for each row.
    Parameters:
        spikes_times (np.ndarray): Array of spike times (seconds).
        spikes_clusters (np.ndarray): Array of cluster assignments for each spike.
        cluster_info (pd.DataFrame): DataFrame with cluster metadata and UnitType.
    Returns:
        np.ndarray: [n_clusters x max_spikes] array, padded with 0.
        list of dict: cluster metadata (including index) for each row of the spike_matrix.
    """

    # spikes_times,spikes_clusters,cluster_info = spike_timestamps_seconds, spike_cluster_assignments, cluster_metadata
    try:
        unit_type_column = cluster_info.columns[1]
        # Get cluster indices for each type
        good_clusters_index = cluster_info[cluster_info[unit_type_column] == 'GOOD'].index
        mua_clusters_index = cluster_info[cluster_info[unit_type_column] == "MUA"].index
        non_somatic_index = cluster_info[cluster_info[unit_type_column] == 'NON-SOMA'].index
        ordered_indices = list(good_clusters_index) + list(mua_clusters_index) + list(non_somatic_index)
        if len(ordered_indices) == 0:
            unit_type_column = cluster_info.columns[1]
            good_clusters_index = cluster_info[cluster_info[unit_type_column] == 1].index
            mua_clusters_index = cluster_info[cluster_info[unit_type_column] == 2].index
            non_somatic_index = cluster_info[cluster_info[unit_type_column] == 3].index
            ordered_indices = list(good_clusters_index) + list(mua_clusters_index) + list(non_somatic_index)
    except Exception:
        print(f"Warning: No good, MUA, or non-somatic clusters found in {cluster_info}")
        return None, None

    # Collect spike times for each cluster
    spike_lists = []
    all_cluster_indices = []
    for idx in ordered_indices:
        cluster_spike_times = spikes_times[spikes_clusters == idx]
        spike_lists.append(cluster_spike_times)
        meta = cluster_info.loc[idx].to_dict()
        meta['cluster_index'] = idx
        all_cluster_indices.append(meta)
    # Find the maximum number of spikes in any cluster
    max_len = max((len(s) for s in spike_lists), default=0)
    # Pad each cluster's spike times with 0 to max_len
    spike_matrix = np.zeros((len(spike_lists), max_len))
    for i, spikes in enumerate(spike_lists):
        if len(spikes) > 0:
            spike_matrix[i, :len(spikes)] = spikes
    return spike_matrix, all_cluster_indices

def bin_spike_matrix(spike_matrix, bins):
    """
    Bins spike times for each cluster into the provided time bins.
    Parameters:
        spike_matrix (list of np.ndarray): List of spike time arrays for each cluster.
        bins (np.ndarray): Array of bin edges (seconds).
    Returns:
        np.ndarray: 2D array (clusters x bins) of spike counts.
    """
    n_bins = len(bins) - 1
    binned = np.zeros((len(spike_matrix), n_bins), dtype=int)
    for i, cluster_spikes in enumerate(spike_matrix):
        if len(cluster_spikes) > 0:
            counts, _ = np.histogram(cluster_spikes, bins=bins)
            binned[i, :] = counts
    return binned

def save_analysis_data(folder, spike_matrix, stimuli_outcome, all_cluster_indices, licking_times=None):
    """
    Saves all analysis data including spike matrices (split by cell type), stimuli/outcome data, and licking data.
    Splits the spike_matrix into good, MUA, and non-somatic cell types based on all_cluster_indices (list of dicts),
    and saves each as a separate .npy file. Also saves the full spike_matrix, the stimuli_outcome_df, and licking data.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    # folder, spike_matrix, stimuli_outcome, all_cluster_indices, licking_times = analysis_output_dir, firing_rate_matrix, stimuli_outcome_df, cluster_metadata_list, licking_timestamps_in_bins
    # Convert all_cluster_indices to DataFrame for easy filtering
    meta_df = pd.DataFrame(all_cluster_indices)
    try:
        good_mask = meta_df['UnitType'] == 1
        mua_mask = meta_df['UnitType'] == 2
        non_somatic_mask = meta_df['UnitType'] == 3
    except Exception:
        good_mask = meta_df['bc_unitType'] == 'GOOD'
        mua_mask = meta_df['bc_unitType'] == 'MUA'
        non_somatic_mask = meta_df['bc_unitType'] == 'NON-SOMA'

    good_matrix = spike_matrix[good_mask.values, :] if good_mask.any() else np.empty((0, spike_matrix.shape[1]))
    mua_matrix = spike_matrix[mua_mask.values, :] if mua_mask.any() else np.empty((0, spike_matrix.shape[1]))
    non_somatic_matrix = spike_matrix[non_somatic_mask.values, :] if non_somatic_mask.any() else np.empty((0, spike_matrix.shape[1]))

    np.save(os.path.join(folder, "good_matrix.npy"), good_matrix)
    np.save(os.path.join(folder, "mua_matrix.npy"), mua_matrix)
    np.save(os.path.join(folder, "non_somatic_matrix.npy"), non_somatic_matrix)
    np.save(os.path.join(folder, "spike_matrix.npy"), spike_matrix)
    stimuli_outcome.to_csv(os.path.join(folder, "stimuli_outcome_df.csv"), index=False)

    # Save licking data if provided
    if licking_times is not None:
        np.save(os.path.join(folder, "licking_timestamps_seconds.npy"), licking_times)

def load_spike_matrices(folder, type="good"):
    """
    Loads the good, MUA, non-somatic spike matrices, the full spike_matrix, and the stimuli_outcome DataFrame from the specified files.
    Returns:
        tuple: (good, mua, non_somatic, spike_matrix, stimuli_outcome_df)
    """
    if type == "good":
        load_path = os.path.join(folder, "good_matrix.npy")
    elif type == "mua":
        load_path = os.path.join(folder, "mua_matrix.npy")
    elif type == "non_somatic":
        load_path = os.path.join(folder, "non_somatic_matrix.npy")
    elif type == "spike":
        load_path = os.path.join(folder, "spike_matrix.npy")
    else:
        raise ValueError(f"Invalid type: {type}, the only options are 'good', 'mua', 'non_somatic', 'spike'")
    
    stimuli_outcome_df = pd.read_csv(os.path.join(folder, "stimuli_outcome_df.csv"))
    if os.path.isfile(load_path):
        spike_matrix = np.load(load_path, allow_pickle=True)
    else:
        spike_matrix = None
    return spike_matrix, stimuli_outcome_df 

def load_spike_data(data_dir):
    """
    Loads spike_times.npy and spike_clusters.npy from the given data_dir.
    Args:
        data_dir (str): Directory containing spike_times.npy and spike_clusters.npy
    Returns:
        tuple: (spike_times, spike_clusters) as numpy arrays
    """
    try:
        spike_times_sec= np.load(os.path.join(data_dir, 'spike_times_sec_adj.npy').replace("\\", "/")) 
    except Exception:
        spike_times_sec = np.load(os.path.join(data_dir, 'spike_times_seconds.npy').replace("\\", "/"))
        
    spike_clusters = np.load(os.path.join(data_dir, 'spike_clusters.npy'))
    len(spike_times_sec)
    len(spike_clusters) 
    return spike_times_sec, spike_clusters

def couple_stimuli_outcome_and_times(parent_dir, stimuli, atten, outcomes, bin_size):
    """
    Finds the file ending with 'nidq.xd_0_1_100.txt' in the directory, loads times, and couples with stimuli values and outcomes.
    Trims all arrays to the shortest length if they mismatch.
    """
    # Define the expected file suffix
    expected_suffix = "nidq.xd_0_1_100.txt"
    
    times = load_nidq_stream(parent_dir, stream_suffix=expected_suffix)
    if len(times) == 0:
        expected_suffix = "nidq.xd_0_1_0.txt"
        times = load_nidq_stream(parent_dir, stream_suffix=expected_suffix)

    # check if the times are in the same length as the stimuli_values and outcomes
    if len(times) != len(stimuli):
        print(f"Warning: tone onset times {len(times)}, and stimuli {len(stimuli)} are not in the same length")

    # Find the minimum length
    min_len = min(len(times), len(stimuli), len(atten), len(outcomes))
    times = times[:min_len]*(1/bin_size) # convert to bins
    stimuli = stimuli[:min_len]
    atten = atten[:min_len]
    outcomes = outcomes[:min_len]

    df = pd.DataFrame({'time': times, 'stimulus': stimuli, 'atten': atten, 'outcome': outcomes})
    return df

# --- Behavioral File Copy Function  ---
def copy_behavioral_file_for_dir(parent_dir, df, spike_glx_col='spike glx file', behavioral_file_col='behavioral file', csv_path=None, force_rerun=False):
    """
    For a given parent_dir, check if any spikeglx file in the CSV is present in that directory.
    If so, find the corresponding behavioral file and copy it to the parent_dir if not already present.
    Also updates the 'current_dir' column in the DataFrame for the corresponding row and saves the DataFrame if csv_path is provided.
    Parameters:
        parent_dir (str): Directory to check for spikeglx file and to copy behavioral file into.
        df (pd.DataFrame): DataFrame containing experiment metadata.
        spike_glx_col (str): Column name for spikeglx file in df.
        behavioral_file_col (str): Column name for behavioral file in df.
        csv_path (str, optional): Path to save the updated DataFrame.
        force_rerun (bool): If True, force copy the behavioral file even if it already exists.
    Returns:
        str: Path to the behavioral file in the parent_dir (if copied or found), or None if not found.
    """
    behav_file = None
   # parent_dir, df, csv_path=nidq_dir, experiment_metadata_df, experiment_metadata_csv_path
            
    # Check if parent_dir exists
    if not os.path.exists(parent_dir):
        print(f"Warning: Parent directory does not exist: {parent_dir}")
        return None

    # Get list of directories in parent_dir to check for spikeglx matches
    try:
        entries_in_parent = os.listdir(parent_dir)
    except (OSError, PermissionError) as e:
        print(f"Warning: Could not list directory {parent_dir}: {e}")
        return None

    for idx, row in df.iterrows():
        # Use parameter column names instead of hardcoded ones
        spikeglx_base = str(row[spike_glx_col]).strip() if spike_glx_col in row else None
        behavioral_full_name = str(row[behavioral_file_col]).strip() if behavioral_file_col in row else None
        
        # Skip if values are invalid or NaN
        if not spikeglx_base or not behavioral_full_name or spikeglx_base.lower() == 'nan' or behavioral_full_name.lower() == 'nan':
            continue
        
        # Check if source behavioral file exists
        if not os.path.exists(behavioral_full_name):
            continue
        
        # Construct spikeglx directory name (e.g., "catgt_G7A1_novice_FRA_g0")
        spikeglx_name = f"catgt_{spikeglx_base}"
        
        if not spikeglx_base in parent_dir:
            continue
        else:
            print(f"Found spikeglx directory: {spikeglx_base} in {parent_dir}")
            # Get behavioral file name
            behavioral_name = os.path.basename(behavioral_full_name)
            behav_file_in_parent = os.path.join(parent_dir, behavioral_name).replace("\\", "/")
            
            # Copy behavioral file to parent_dir if it doesn't exist or if force_rerun is True
            try:
                if not os.path.exists(behav_file_in_parent) or force_rerun:
                    shutil.copy2(behavioral_full_name, parent_dir)
                    print(f"Copied {behavioral_name} to {parent_dir}")
                    behav_file = behav_file_in_parent
                else:
                    behav_file = behav_file_in_parent
                    print(f"Behavioral file {behavioral_name} already exists in {parent_dir}")
            except Exception as e:
                print(f"Error copying behavioral file {behavioral_name} to {parent_dir}: {e}")

                
               
        # Save the DataFrame if a path is provided (with error handling for permission issues)
        if csv_path is not None:
            try:
                df.to_csv(csv_path, index=False)
            except PermissionError as e:
                print(f"Warning: Could not save CSV file (file may be open in another program): {e}")
                # Continue processing even if CSV save fails
            except Exception as e:
                print(f"Warning: Error saving CSV file: {e}")
        
        # Found a match, return the behavioral file path
        return behav_file
        
    # No matching spikeglx directory found in parent_dir
    return None

def spike_times_to_binary_matrix(spike_matrix, t_start, t_end, bin_size=0.01):
    """
    Converts a list of spike times per unit to a binary matrix [units x time_bins].
    Each entry is 1 if there is a spike in the bin, 0 otherwise.

    Args:
        spike_matrix (list of np.ndarray): List of arrays, each with spike times for a unit (in seconds).
        t_start (float): Start time (in seconds).
        t_end (float): End time (in seconds).
        bin_size (float): Bin size in seconds (default: 1 ms).

    Returns:
        np.ndarray: Binary matrix of shape [units x time_bins].
        np.ndarray: The bin edges.
    """
    n_units = len(spike_matrix)
    bins = np.arange(t_start, t_end + bin_size, bin_size)
    n_bins = len(bins) - 1
    binary_matrix = np.zeros((n_units, n_bins), dtype=int)
    for i, spikes in enumerate(spike_matrix):
        if len(spikes) > 0:
            counts, _ = np.histogram(spikes, bins=bins)
            binary_matrix[i, :] = (counts > 0).astype(int)
    return binary_matrix, bins

def spike_times_to_firing_rate_matrix(spike_matrix, t_start, t_end, bin_size=0.1):
    """
    Converts a spike times matrix to a firing rate matrix [units x time_bins].
    Each entry is the firing rate (spikes per second) in that bin.

    Args:
        spike_matrix (np.ndarray): Matrix of shape [n_units x max_spikes] with spike times (padded with 0).
        t_start (float): Start time (in seconds).
        t_end (float): End time (in seconds).
        bin_size (float): Bin size in seconds.

    Returns:
        np.ndarray: Firing rate matrix of shape [units x time_bins].
        np.ndarray: The bin edges.
    """
    n_units = spike_matrix.shape[0]
    bins = np.arange(t_start, t_end + bin_size, bin_size)
    n_bins = len(bins) - 1
    firing_rate_matrix = np.zeros((n_units, n_bins), dtype=float)
    
    for i in range(n_units):
        # Get non-zero spike times for this unit (remove padding zeros)
        unit_spikes = spike_matrix[i, :]
        unit_spikes = unit_spikes[unit_spikes > 0]  # Remove zero padding
        
        if len(unit_spikes) > 0:
            counts, _ = np.histogram(unit_spikes, bins=bins)
            firing_rate_matrix[i, :] = counts / bin_size  # spikes per second
    
    return firing_rate_matrix, bins

def load_nidq_stream(working_dir, stream_suffix):
    """
    Loads any NIDQ stream data from a file ending with the specified suffix in the working directory.
    
    Parameters:
        working_dir (str): Path to the working directory containing the NIDQ stream file.
        stream_suffix (str): The file suffix to search for (e.g., 'nidq.xd_0_2_0.txt' for licking data).
        
    Returns:
        np.ndarray: Array of stream timestamps (in seconds).
        
    Raises:
        FileNotFoundError: If no file ending with the specified suffix is found in the directory.
    """
    # Find the NIDQ stream file
    stream_file = None
    for fname in os.listdir(working_dir):
        if fname.endswith(stream_suffix):
            stream_file = os.path.join(working_dir, fname)
            break
    
    if stream_file is None:
        raise FileNotFoundError(f"No file ending with {stream_suffix} found in directory: {working_dir}")
    
    # Load the stream timestamps
    stream_timestamps = np.loadtxt(stream_file)
    
    # Ensure it's a 1D array
    if stream_timestamps.ndim > 1:
        stream_timestamps = stream_timestamps.flatten()
    
    return stream_timestamps

def reshape_firing_rate_to_event_windows(firing_rate_matrix, stimuli_outcome_df, window_duration=1.0, bin_size=0.1):
    """
    Reshapes firing rate matrix to extract windows around each event.
    
    Args:
        firing_rate_matrix (np.ndarray): Firing rate matrix of shape [units × time_bins]
        stimuli_outcome_df (pd.DataFrame): DataFrame containing 'time' column with event timestamps
        window_duration (float): Duration of window before and after event in seconds (default: 1.0)
        bin_size (float): Bin size in seconds (default: 0.1)
        
    Returns:
        np.ndarray: 3D matrix of shape [units × time_bins_per_window × n_events]
        np.ndarray: Time axis for the window (relative to event onset)
        list: List of valid event indices that were successfully extracted
    """
    n_units, n_time_bins = firing_rate_matrix.shape
    
    # Calculate number of bins in the window
    bins_per_window = int(2 * window_duration / bin_size)  # Before + after event
    
    # Get event times from DataFrame
    if 'time' not in stimuli_outcome_df.columns:
        raise ValueError("stimuli_outcome_df must contain 'time' column")
    
    event_times = stimuli_outcome_df['time'].values

    # Initialize output matrix
    event_windows = []
    valid_event_indices = []
    
    # Time axis for the window (relative to event onset)
    time_axis = np.arange(-window_duration, window_duration, bin_size)
    
    for event_idx, event_time in enumerate(event_times):
        # event_time = event_times[4]
        event_bin = int(event_time)  # Already in bins times, just convert to int
        
        # Calculate start and end bins for the window
        start_bin = event_bin - int(window_duration / bin_size)
        end_bin = event_bin + int(window_duration / bin_size)
        
        # Check if window fits within the firing rate matrix
        if start_bin >= 0 and end_bin <= n_time_bins:
            # Extract window for all units
            window_data = firing_rate_matrix[:, start_bin:end_bin]
            
            # Ensure the window has the expected size
            if window_data.shape[1] == bins_per_window:
                event_windows.append(window_data)
                valid_event_indices.append(event_idx)
            else:
                print(f"Warning: Event {event_idx} window size mismatch. Expected {bins_per_window}, got {window_data.shape[1]}")
        else:
            print(f"Warning: Event {event_idx} window ({start_bin}:{end_bin}) outside matrix bounds (0:{n_time_bins})")
    
    if not event_windows:
        raise ValueError("No valid event windows could be extracted")
    
    # Stack all windows into a 3D array
    event_windows_matrix = np.stack(event_windows, axis=2)  # Shape: [units × time × events]
    
    print(f"Successfully extracted {len(valid_event_indices)} event windows")
    print(f"Output shape: {event_windows_matrix.shape}")
    
    return event_windows_matrix, time_axis, valid_event_indices

def reshape_licking_to_event_windows(licking_timestamps_in_bins, stimuli_outcome_df, recording_duration_bins, window_duration=1.0, bin_size=0.1):
    """
    Reshapes licking timestamps to extract lick rate windows around each event.
    
    Args:
        licking_timestamps_in_bins (np.ndarray): Array of licking timestamps in bins
        stimuli_outcome_df (pd.DataFrame): DataFrame containing 'time' column with event timestamps
        recording_duration_bins (int): Total duration of recording in bins
        window_duration (float): Duration of window before and after event in seconds (default: 1.0)
        bin_size (float): Bin size in seconds (default: 0.1)
        
    Returns:
        np.ndarray: 3D matrix of shape [1 × time_bins_per_window × n_events] (lick rate)
        np.ndarray: Time axis for the window (relative to event onset)
        list: List of valid event indices that were successfully extracted
    """
    if licking_timestamps_in_bins is None or len(licking_timestamps_in_bins) == 0:
        raise ValueError("No licking data provided")
    
    # Calculate number of bins in the window
    bins_per_window = int(2 * window_duration / bin_size)  # Before + after event
    
    # Get event times from DataFrame
    if 'time' not in stimuli_outcome_df.columns:
        raise ValueError("stimuli_outcome_df must contain 'time' column")
    
    event_times = stimuli_outcome_df['time'].values
    
    # Convert licking timestamps to lick rate matrix
    # Create time bins for the entire recording
    time_bins = np.arange(0, recording_duration_bins + 1)
    
    # Calculate lick counts per bin
    lick_counts, _ = np.histogram(licking_timestamps_in_bins, bins=time_bins)
    
    # Convert to lick rate (licks per second)
    lick_rate_matrix = lick_counts / bin_size  # Shape: [n_time_bins]
    lick_rate_matrix = lick_rate_matrix.reshape(1, -1)  # Shape: [1 × n_time_bins]
    
    n_time_bins = lick_rate_matrix.shape[1]
    
    # Initialize output matrix
    event_windows = []
    valid_event_indices = []
    
    # Time axis for the window (relative to event onset)
    time_axis = np.arange(-window_duration, window_duration, bin_size)
    
    for event_idx, event_time in enumerate(event_times):
        event_bin = int(event_time)  # Already in bins times, just convert to int
        
        # Calculate start and end bins for the window
        start_bin = event_bin - int(window_duration / bin_size)
        end_bin = event_bin + int(window_duration / bin_size)
        
        # Check if window fits within the lick rate matrix
        if start_bin >= 0 and end_bin <= n_time_bins:
            # Extract window for lick rate (single channel)
            window_data = lick_rate_matrix[:, start_bin:end_bin]
            
            # Ensure the window has the expected size
            if window_data.shape[1] == bins_per_window:
                event_windows.append(window_data)
                valid_event_indices.append(event_idx)
            else:
                print(f"Warning: Lick event {event_idx} window size mismatch. Expected {bins_per_window}, got {window_data.shape[1]}")
        else:
            print(f"Warning: Lick event {event_idx} window ({start_bin}:{end_bin}) outside matrix bounds (0:{n_time_bins})")
    
    if not event_windows:
        raise ValueError("No valid licking event windows could be extracted")
    
    # Stack all windows into a 3D array
    lick_event_windows_matrix = np.stack(event_windows, axis=2)  # Shape: [1 × time × events]
    
    print(f"Successfully extracted {len(valid_event_indices)} licking event windows")
    print(f"Licking output shape: {lick_event_windows_matrix.shape}")
    
    return lick_event_windows_matrix, time_axis, valid_event_indices

def save_event_windows_data(folder, event_windows_matrix, time_axis, valid_event_indices, stimuli_outcome_df, lick_event_windows_matrix=None):
    """
    Saves the event windows data and associated metadata.
    
    Args:
        folder (str): Directory to save the data
        event_windows_matrix (np.ndarray): 3D matrix of shape [units × time × events]
        time_axis (np.ndarray): Time axis for the window
        valid_event_indices (list): List of valid event indices
        stimuli_outcome_df (pd.DataFrame): Original stimuli_outcome DataFrame
        lick_event_windows_matrix (np.ndarray, optional): 3D matrix of shape [1 × time × events] for lick rates
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Save the 3D event windows matrix
    np.save(os.path.join(folder, "event_windows_matrix.npy"), event_windows_matrix)
    
    # Save the licking event windows matrix if provided
    if lick_event_windows_matrix is not None:
        np.save(os.path.join(folder, "lick_event_windows_matrix.npy"), lick_event_windows_matrix)
    
    # Save the time axis
    np.save(os.path.join(folder, "event_window_time_axis.npy"), time_axis)
    
    # Save the valid event indices
    np.save(os.path.join(folder, "valid_event_indices.npy"), valid_event_indices)
    
    # Save the filtered stimuli_outcome DataFrame (only valid events)
    valid_events_df = stimuli_outcome_df.iloc[valid_event_indices].copy()
    valid_events_df.to_csv(os.path.join(folder, "event_windows_stimuli_outcome.csv"), index=False)
    
    # Save metadata
    metadata = {
        'n_units': event_windows_matrix.shape[0],
        'n_time_bins': event_windows_matrix.shape[1],
        'n_events': event_windows_matrix.shape[2],
        'window_duration': abs(time_axis[0]),  # Assuming symmetric window
        'bin_size': time_axis[1] - time_axis[0] if len(time_axis) > 1 else 0.1
    }
    
    metadata_file = os.path.join(folder, "event_windows_metadata.txt")
    with open(metadata_file, 'w') as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Event windows data saved to {folder}")

def load_event_windows_data(folder):
    """
    Loads the event windows data and associated metadata.
    
    Args:
        folder (str): Directory containing the saved data
        
    Returns:
        tuple: (event_windows_matrix, time_axis, valid_event_indices, stimuli_outcome_df, metadata, lick_event_windows_matrix)
    """
    # Load the 3D event windows matrix
    event_windows_matrix = np.load(os.path.join(folder, "event_windows_matrix.npy"))
    
    # Load the licking event windows matrix if it exists
    lick_file_path = os.path.join(folder, "lick_event_windows_matrix.npy")
    if os.path.exists(lick_file_path):
        lick_event_windows_matrix = np.load(lick_file_path)
    else:
        lick_event_windows_matrix = None
    
    # Load the time axis
    time_axis = np.load(os.path.join(folder, "event_window_time_axis.npy"))
    
    # Load the valid event indices
    valid_event_indices = np.load(os.path.join(folder, "valid_event_indices.npy"))
    
    # Load the filtered stimuli_outcome DataFrame
    stimuli_outcome_df = pd.read_csv(os.path.join(folder, "event_windows_stimuli_outcome.csv"))
    
    # Load metadata
    metadata = {}
    metadata_file = os.path.join(folder, "event_windows_metadata.txt")
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            for line in f:
                key, value = line.strip().split(': ')
                metadata[key] = value
    
    return event_windows_matrix, time_axis, valid_event_indices, stimuli_outcome_df, metadata, lick_event_windows_matrix

# --- Main Analysis Loop ---
def main():
    """
    Main analysis workflow. Finds KS folders, copies behavioral files, and (optionally) runs further analysis for each folder.
    """
    force_rerun = True
    
    bin_size=0.005 # in seconds

    # recordings_root_directory = r"/ems/elsc-labs/mizrahi-a/Shared/Amichai/NPXL/Recs/group5"
    # experiment_metadata_csv_path = r"/ems/elsc-labs/mizrahi-a/Code\DB\users_data\Amichai\NPXL recordings _experimental_data.csv".replace("\\", "/")
    
    recordings_root_directory = r"Z:/Shared/Amichai/NPXL/Recs/group7".replace("\\", "/")
    experiment_metadata_csv_path = r"Z:\Shared\Amichai/Code\DB\users_data\Amichai\NPXL recordings _experimental_data.csv".replace("\\", "/")
    

    experiment_metadata_df = pd.read_csv(experiment_metadata_csv_path)
    ks_analysis_folders = find_ks_folders(recordings_root_directory, target_dir="bombcell")      # ks for kilosort, the folder name is the probe identifier
    for idx, current_ks_folder in enumerate(ks_analysis_folders):
        print(f"\n\nProcessing {idx+1} of {len(ks_analysis_folders)}: {current_ks_folder}")
        try:
            # current_ks_folder = ks_analysis_folders[-3].replace("\\", "/")

            recording_session_dir = os.path.dirname(current_ks_folder)
            os.chdir(recording_session_dir)
            nidq_dir = recording_session_dir
            
            # Extract probe identifier from folder name (e.g., "imec1_ks4" -> "1")
            # The current_ks_folder is already the imec*_ks4 folder, so spike data is directly in it
            probe_match = re.search(r'imec(\d+)', os.path.basename(current_ks_folder))
            if probe_match:
                probe_identifier = probe_match.group(1)
            else:
                # Fallback: try to extract from last character (old method)
                probe_identifier = current_ks_folder[-1] if current_ks_folder[-1].isdigit() else "0"

            ks4_data_dir = os.path.join(current_ks_folder, f"imec{probe_identifier}_ks4").replace("\\", "/")
            analysis_output_dir = os.path.join(current_ks_folder, "analysis_output").replace("\\", "/")
            behavioral_data_file_path = copy_behavioral_file_for_dir(parent_dir=recording_session_dir, df = experiment_metadata_df, csv_path=experiment_metadata_csv_path, force_rerun=False)
            
            # Skip if no behavioral file was found
            if behavioral_data_file_path is None:
                print(f"Warning: No behavioral file found for {recording_session_dir}. Skipping...")
                continue
            
            if not os.path.exists(analysis_output_dir) or force_rerun:

                # Process behavioral data
                try:
                    trial_types_df, raw_events_df, session_date, session_time, trial_settings, notes, licks, states, stimulis, Unique_Stimuli_Values, tones_per_class, boundaries, recs = load_mat_file(behavioral_data_file_path)
                    
                    # Check if this is actually an FRA file by checking if stimulis is a 2D array (FRA) or 1D (regular)
                    is_fra_file = "FRA" in current_ks_folder and isinstance(stimulis, np.ndarray) and len(stimulis.shape) == 2
                    
                    if is_fra_file:
                        try:
                            stimuli = stimulis[:, 0]
                            atten = stimulis[:, 1]
                            outcomes = np.zeros(len(stimuli))
                        except (IndexError, TypeError) as e:
                            print(f"Warning: File in FRA folder but doesn't have FRA structure. Processing as regular file: {e}")
                            # Fall back to regular processing
                            session_summary_df = create_single_row_with_outcome(behavioral_data_file_path, trial_types_df, raw_events_df, session_date,
                                                                    session_time, trial_settings, notes, licks, states,
                                                                    Unique_Stimuli_Values, tones_per_class, boundaries, recs)
                            stimuli = stimulis
                            atten = np.ones(len(stimuli))*(-60)
                            outcomes = session_summary_df["Outcomes"].iloc[0]
                    else:
                        session_summary_df = create_single_row_with_outcome(behavioral_data_file_path, trial_types_df, raw_events_df, session_date,
                                                                session_time, trial_settings, notes, licks, states,
                                                                Unique_Stimuli_Values, tones_per_class, boundaries, recs)
                        stimuli = stimulis
                        atten = np.ones(len(stimuli))*(-60)
                        outcomes = session_summary_df["Outcomes"].iloc[0]
                except Exception as e:
                    print(f"Error processing behavioral data {behavioral_data_file_path}: {e}")
                    continue

                # --- Load Spikeglx Data ---
                try:
                    spike_timestamps_seconds, spike_cluster_assignments = load_spike_data(ks4_data_dir)
                    
                    if len(spike_timestamps_seconds) != len(spike_cluster_assignments):
                        print(f"{current_ks_folder}: Warning: spike times and cluster assignments not in the same length")
                        continue
                        
                    # check if spike_timestamps_seconds and spike_cluster_assignments are in the same length
                    cluster_metadata = load_cluster_info(current_ks_folder, ks4_data_dir)
                    
                    spike_times_matrix, cluster_metadata_list = extract_spike_matrices(spike_timestamps_seconds, spike_cluster_assignments, cluster_metadata) 
                    
                    # Convert spike times to firing rate matrix
                    recording_start_time = np.floor(spike_timestamps_seconds.min()) 
                    recording_end_time = np.ceil(spike_timestamps_seconds.max()) 
                    firing_rate_matrix, time_bins = spike_times_to_firing_rate_matrix(spike_times_matrix, recording_start_time, recording_end_time, bin_size)
                    
                    stimuli_outcome_df = couple_stimuli_outcome_and_times(nidq_dir, stimuli, atten, outcomes, bin_size)
                    
                    # Load licking data
                    try:
                        licking_timestamps_seconds = load_nidq_stream(nidq_dir, stream_suffix="nidq.xd_0_2_0.txt")
                        licking_timestamps_in_bins = licking_timestamps_seconds*(1/bin_size)

                    except FileNotFoundError:
                        print(f"Warning: No licking data found for {current_ks_folder}")
                        licking_timestamps_in_bins = None
                    
                    save_analysis_data(analysis_output_dir, firing_rate_matrix, stimuli_outcome_df, cluster_metadata_list, licking_timestamps_in_bins)
                    
                    # Create event windows data
                    try:
                        window_duration = 4.0
                        event_windows_matrix, time_axis, valid_event_indices = reshape_firing_rate_to_event_windows(
                            firing_rate_matrix, stimuli_outcome_df, window_duration, bin_size=bin_size)
                        
                        # Create licking event windows if licking data is available
                        lick_event_windows_matrix = None
                        if licking_timestamps_in_bins is not None:
                            try:
                                recording_duration_bins = firing_rate_matrix.shape[1]
                                lick_event_windows_matrix, lick_time_axis, lick_valid_indices = reshape_licking_to_event_windows(
                                    licking_timestamps_in_bins, stimuli_outcome_df, recording_duration_bins, 
                                    window_duration, bin_size=bin_size)
                            except Exception as e:
                                print(f"Warning: Could not create licking event windows: {e}")
                                lick_event_windows_matrix = None
                        
                        # Save event windows data (including licking if available)
                        save_event_windows_data(analysis_output_dir, event_windows_matrix, time_axis, valid_event_indices, 
                                               stimuli_outcome_df, lick_event_windows_matrix)
                        print(f"Event windows data created successfully: {event_windows_matrix.shape}")
                        
                    except Exception as e:
                        print(f"Warning: Could not create event windows data: {e}")
                    
                except Exception as e:
                    print(f"Error processing spike data for {current_ks_folder}: {e}")
                    continue
        except Exception as e:
            print(f"Error processing {current_ks_folder}: {e}")
            continue

if __name__ == "__main__":
    main()

