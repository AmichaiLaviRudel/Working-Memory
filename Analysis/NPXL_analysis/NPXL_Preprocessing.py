import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from load_data.load_bpod_data import load_mat_file, create_single_row_with_outcome

import streamlit as st
import scipy.io as spio
import numpy as np
import pandas as pd
from scipy.stats import norm

import pandas as pd
import matplotlib.pyplot as plt

import plotly.graph_objects as go

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

def load_cluster_info(working_dir, data_dirs):
    """
    Loads cluster information from a unit_labels.tsv or cluster_info.tsv file in the working directory.
    Parameters:
        working_dir (str): Path to the working directory.
        data_dirs (str): Path to the data directory (used as fallback).
    Returns:
        pd.DataFrame or None: DataFrame with cluster info, or None if not found.
    """
    try:
        label_dir = os.path.join(working_dir, "bombcell", "unit_labels.tsv")
    except Exception:
        label_dir = os.path.join(data_dirs, "cluster_info.tsv")
    if os.path.isfile(label_dir):
        return pd.read_csv(label_dir, sep='\t')
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
    # Get cluster indices for each type
    good_clusters_index = cluster_info[cluster_info['UnitType'] == 1].index
    mua_clusters_index = cluster_info[cluster_info['UnitType'] == 2].index
    non_somatic_index = cluster_info[cluster_info['UnitType'] == 3].index
    ordered_indices = list(good_clusters_index) + list(mua_clusters_index) + list(non_somatic_index)
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

def save_spike_matrices(folder, spike_matrix, stimuli_outcome_df, all_cluster_indices):
    """
    Splits the spike_matrix into good, MUA, and non-somatic cell types based on all_cluster_indices (list of dicts),
    and saves each as a separate .npy file. Also saves the full spike_matrix and the stimuli_outcome_df.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Convert all_cluster_indices to DataFrame for easy filtering
    import pandas as pd
    meta_df = pd.DataFrame(all_cluster_indices)
    good_mask = meta_df['UnitType'] == 1
    mua_mask = meta_df['UnitType'] == 2
    non_somatic_mask = meta_df['UnitType'] == 3

    good_matrix = spike_matrix[good_mask.values, :] if good_mask.any() else np.empty((0, spike_matrix.shape[1]))
    mua_matrix = spike_matrix[mua_mask.values, :] if mua_mask.any() else np.empty((0, spike_matrix.shape[1]))
    non_somatic_matrix = spike_matrix[non_somatic_mask.values, :] if non_somatic_mask.any() else np.empty((0, spike_matrix.shape[1]))

    np.save(os.path.join(folder, "good_matrix.npy"), good_matrix)
    np.save(os.path.join(folder, "mua_matrix.npy"), mua_matrix)
    np.save(os.path.join(folder, "non_somatic_matrix.npy"), non_somatic_matrix)
    np.save(os.path.join(folder, "spike_matrix.npy"), spike_matrix)
    stimuli_outcome_df.to_csv(os.path.join(folder, "stimuli_outcome_df.csv"), index=False)

def load_spike_matrices(folder, type="good"):
    """
    Loads the good, MUA, non-somatic spike matrices, the full spike_matrix, and the stimuli_outcome DataFrame from the specified files.
    Returns:
        tuple: (good, mua, non_somatic, spike_matrix, stimuli_outcome_df)
    """
    if type == "good":
        load_path   = os.path.join(folder, "good.npy")
    elif type == "mua":
        load_path = os.path.join(folder, "mua.npy")
    elif type == "non_somatic":
        load_path = os.path.join(folder, "non_somatic.npy")
    elif type == "all":
        load_path = os.path.join(folder, "spike_matrix.npy")
    else:
        raise ValueError(f"Invalid type: {type}, the only options are 'good', 'mua', 'non_somatic', 'all'")
    
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
        sr (float): Sample rate (not used here, but kept for compatibility)
    Returns:
        tuple: (spike_times, spike_clusters) as numpy arrays
    """
    spike_times_ms = np.load(os.path.join(data_dir, 'spike_times.npy')) 
    spike_clusters = np.load(os.path.join(data_dir, 'spike_clusters.npy'))
    return spike_times_ms, spike_clusters

def couple_stimuli_outcome_and_times(directory, stimuli_values, outcomes):
    """
    Finds the file ending with 'nidq.xd_0_1_100.txt' in the directory, loads times, and couples with stimuli values and outcomes.
    Trims all arrays to the shortest length if they mismatch.
    Also stores the path to the found txt file in Streamlit session state under the key 'nidq_txt_path'.
    """
    # Store the path in Streamlit session state
    try:
        import streamlit as st
        if 'nidq_events_times_path' not in st.session_state:
            st.session_state['nidq_events_times_path'] = "nidq.xd_0_1_100.txt"
    except ImportError:
        pass  # If Streamlit is not available, skip
    # Find the file
    txt_file = None
    for fname in os.listdir(directory):
        if fname.endswith(st.session_state['nidq_events_times_path']):
            txt_file = os.path.join(directory, fname)
            break
    if txt_file is None:
        raise FileNotFoundError(f"No file ending with {st.session_state['nidq_events_times_path']} found in directory.")



    import numpy as np
    times = np.loadtxt(txt_file)
    if times.ndim > 1:
        times = times.flatten()

    # Find the minimum length
    min_len = min(len(times), len(stimuli_values), len(outcomes))
    times = times[:min_len]
    stimuli_values = stimuli_values[:min_len]
    outcomes = outcomes[:min_len]

    import pandas as pd
    df = pd.DataFrame({'time': times, 'stimulus': stimuli_values, 'outcome': outcomes})
    return df

# --- Behavioral File Copy Function  ---
def copy_behavioral_file_for_dir(parent_dir, df, spike_glx_col='spike glx file', behavioral_file_col='behavioral file', csv_path=None):
    """
    For a given parent_dir, check if any spikeglx file in the CSV is present in that directory.
    If so, find the corresponding behavioral file and copy it to parent_dir if not already present.
    Also updates the 'parent_dir' column in the DataFrame for the corresponding row and saves the DataFrame if csv_path is provided.
    Parameters:
        parent_dir (str): Directory to check for spikeglx file and to copy behavioral file into.
        df (pd.DataFrame): DataFrame containing experiment metadata.
        spike_glx_col (str): Column name for spikeglx file in df.
        behavioral_file_col (str): Column name for behavioral file in df.
        csv_path (str, optional): Path to save the updated DataFrame.
    Returns:
        str: Path to the behavioral file in the parent_dir (if copied or found).
    """
    import os
    behav_file = None
    for idx, row in df.iterrows():
        spikeglx_base = str(row[spike_glx_col]).strip()
        spikeglx_name = f"catgt_{spikeglx_base}"
        behavioral_full_name = str(row[behavioral_file_col]).strip()
        behavioral_name = os.path.basename(behavioral_full_name)
        if not spikeglx_name or not behavioral_full_name:
            continue
        if any(spikeglx_name in entry for entry in os.listdir(parent_dir)):

            for dirpath, _, filenames in os.walk(parent_dir):
                if behavioral_name in filenames:
                    behav_file = os.path.join(dirpath, behavioral_name)
                    break
            if not behav_file:
                dest_path = parent_dir
                shutil.copy2(behavioral_full_name, dest_path)
                print(f"Copied {behavioral_name} to {spikeglx_name}")
                behav_file = os.path.join(dest_path, behavioral_name)
            # Update the parent_dir column for this row
            if 'current_dir' in df.columns:
                df.at[idx, 'current_dir'] = parent_dir
            else:
                df.loc[idx, 'current_dir'] = parent_dir
            # Save the DataFrame if a path is provided
            if csv_path is not None:
                df.to_csv(csv_path, index=False)
            break
    return behav_file

def spike_times_to_binary_matrix(spike_matrix, t_start, t_end, bin_size=0.001):
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

def spike_times_to_firing_rate_matrix(spike_matrix, t_start, t_end, bin_size=0.01):
    """
    Converts a list of spike times per unit to a firing rate matrix [units x time_bins].
    Each entry is the firing rate (spikes per second) in that bin.

    Args:
        spike_matrix (list of np.ndarray): List of arrays, each with spike times for a unit (in seconds).
        t_start (float): Start time (in seconds).
        t_end (float): End time (in seconds).
        bin_size (float): Bin size in seconds.

    Returns:
        np.ndarray: Firing rate matrix of shape [units x time_bins].
        np.ndarray: The bin edges.
    """
    n_units = len(spike_matrix)
    bins = np.arange(t_start, t_end + bin_size, bin_size)
    n_bins = len(bins) - 1
    firing_rate_matrix = np.zeros((n_units, n_bins), dtype=float)
    for i, spikes in enumerate(spike_matrix):
        if len(spikes) > 0:
            counts, _ = np.histogram(spikes, bins=bins)
            firing_rate_matrix[i, :] = counts / bin_size  # spikes per second
    return firing_rate_matrix, bins



# --- Main Analysis Loop ---
def main():
    """
    Main analysis workflow. Finds KS folders, copies behavioral files, and (optionally) runs further analysis for each folder.
    """
    rerun = True
    root_directory = r"Z:\Shared\Amichai\NPXL\Recs\group5"
    try:
        csv_path = st.session_state.npxl_monitoring_path
    except Exception:
        csv_path = r"Z:\Shared\Amichai\Code\DB\users_data\Amichai\NPXL recordings _experimental_data.csv"
    df_csv = pd.read_csv(csv_path)
    ks_folders = find_ks_folders(root_directory)
    for working_dir in ks_folders:
        try:
            os.chdir(working_dir)
            parent_dir = os.path.dirname(working_dir)
            probe_id = working_dir[-1]
            data_dir = os.path.join(working_dir, f"imec{probe_id}_ks4")
            output_dir = os.path.join(working_dir, "analysis_output")
            behavioral_file_path = copy_behavioral_file_for_dir(parent_dir, df_csv, csv_path=csv_path)
            
            if not os.path.exists(output_dir) or rerun:

                try:
                    trial_types_df, raw_events_df, session_date, session_time, trial_settings, notes , licks, states, stimuli, Unique_Stimuli_Values, tones_per_class, boundaries = load_mat_file(behavioral_file_path)
                    combined_row_df = create_single_row_with_outcome(behavioral_file_path, trial_types_df, raw_events_df, session_date,
                                                                session_time, trial_settings, notes, licks, states,
                                                                Unique_Stimuli_Values, tones_per_class, boundaries)
                except Exception as e:
                    print(f"Error processing {behavioral_file_path}: {e}")

                # --- Load Spikeglx Data ---
                sr = read_sample_rate(working_dir)
                spike_times_sr, spike_clusters = load_spike_data(data_dir)
                cluster_info = load_cluster_info(working_dir, data_dir)
                spike_times_sec = spike_times_sr / sr # convert to seconds
                spike_matrix, all_cluster_indices = extract_spike_matrices(spike_times_sec, spike_clusters, cluster_info) 
                # Convert spike times to firing rate matrix
                all_spikes = np.concatenate([s for s in spike_matrix if len(s) > 0])
                t_start = np.floor(all_spikes.min()) 
                t_end = np.ceil(all_spikes.max()) 
                firing_rate_matrix, bins = spike_times_to_firing_rate_matrix(spike_matrix, t_start, t_end, bin_size=0.01)
                stimuli_outcome_df = couple_stimuli_outcome_and_times(parent_dir, stimuli, combined_row_df["Outcomes"].iloc[0])

                save_spike_matrices(output_dir, firing_rate_matrix, stimuli_outcome_df, all_cluster_indices)
        except Exception as e:
            print(f"Error processing {working_dir}: {e}")
            continue

if __name__ == "__main__":
    main()

