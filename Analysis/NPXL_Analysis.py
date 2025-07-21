from load_data.load_bpod_data import load_mat_file, create_single_row_with_outcome

import streamlit as st
import scipy.io as spio
import numpy as np
import pandas as pd
from scipy.stats import norm

import os
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
    Parameters:
        spikes_times (np.ndarray): Array of spike times (seconds).
        spikes_clusters (np.ndarray): Array of cluster assignments for each spike.
        cluster_info (pd.DataFrame): DataFrame with cluster metadata and UnitType.
    Returns:
        tuple: (spike_matrix, good_spike_matrix, mua_spike_matrix, non_somatic_spike_matrix, all_cluster_indices)
    """
    good_clusters_index = cluster_info[cluster_info['UnitType'] == 1].index
    mua_clusters_index = cluster_info[cluster_info['UnitType'] == 2].index
    non_somatioc_index = cluster_info[cluster_info['UnitType'] == 3].index
    all_cluster_indices = (
        list(good_clusters_index) +
        list(mua_clusters_index) +
        list(non_somatioc_index)
    )
    spike_matrix = []
    good_spike_matrix = []
    mua_spike_matrix = []
    non_somatic_spike_matrix = []
    for idx, cluster_id in enumerate(all_cluster_indices):
        cluster_spike_times = spikes_times[spikes_clusters == cluster_id]
        spike_matrix.append(cluster_spike_times)
        if idx < len(good_clusters_index):
            good_spike_matrix.append(cluster_spike_times)
        elif idx < len(good_clusters_index) + len(mua_clusters_index):
            mua_spike_matrix.append(cluster_spike_times)
        else:
            non_somatic_spike_matrix.append(cluster_spike_times)
    return spike_matrix, good_spike_matrix, mua_spike_matrix, non_somatic_spike_matrix, all_cluster_indices

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

def save_spike_matrices(good, mua, non_somatic, good_path, mua_path, non_somatic_path):
    """
    Saves the good, MUA, and non-somatic spike matrices as numpy object arrays to the specified paths.
    """
    np.save(good_path, np.array(good, dtype=object))
    np.save(mua_path, np.array(mua, dtype=object))
    np.save(non_somatic_path, np.array(non_somatic, dtype=object))

def load_spike_matrices(good_path, mua_path, non_somatic_path):
    """
    Loads the good, MUA, and non-somatic spike matrices from the specified numpy files.
    Returns:
        tuple: (good, mua, non_somatic) spike matrices as numpy arrays.
    """
    good = np.load(good_path, allow_pickle=True)
    mua = np.load(mua_path, allow_pickle=True)
    non_somatic = np.load(non_somatic_path, allow_pickle=True)
    return good, mua, non_somatic

def peri_event_spike_matrix(spike_matrix, events_times, pre_event_ms=500, post_event_ms=500, bin_size_ms=1):
    """
    Extracts peri-event spike counts for each unit and event.
    Parameters:
        spike_matrix (list of np.ndarray): List of spike times for each unit.
        events_times (np.ndarray): Array of event times (seconds).
        pre_event_ms (int): Time before event (ms).
        post_event_ms (int): Time after event (ms).
        bin_size_ms (int): Bin size in ms.
    Returns:
        np.ndarray: Array of shape (n_units, n_events, n_timepoints) with spike counts per ms bin.
    """
    peri_event_window_ms = pre_event_ms + post_event_ms
    n_timepoints = peri_event_window_ms // bin_size_ms
    n_units = len(spike_matrix)
    n_events = len(events_times)
    peri_event_spikes = np.zeros((n_units, n_events, n_timepoints), dtype=int)
    for unit_idx, spike_times in enumerate(spike_matrix):
        for event_idx, event_time in enumerate(events_times):
            window_start = event_time - pre_event_ms / 1000
            window_end = event_time + post_event_ms / 1000
            spikes_in_window = spike_times[(spike_times >= window_start) & (spike_times < window_end)]
            spike_times_ms = ((spikes_in_window - event_time) * 1000).astype(int) + pre_event_ms
            spike_times_ms = spike_times_ms[(spike_times_ms >= 0) & (spike_times_ms < n_timepoints)]
            for t in spike_times_ms:
                peri_event_spikes[unit_idx, event_idx, t] += 1
    return peri_event_spikes

def load_spike_data(data_dir, sr):
    """
    Loads spike_times.npy and spike_clusters.npy from the given data_dir.
    Args:
        data_dir (str): Directory containing spike_times.npy and spike_clusters.npy
        sr (float): Sample rate (not used here, but kept for compatibility)
    Returns:
        tuple: (spike_times, spike_clusters) as numpy arrays
    """
    spike_times_ms = np.load(os.path.join(data_dir, 'spike_times.npy')) / sr
    spike_clusters = np.load(os.path.join(data_dir, 'spike_clusters.npy'))
    return spike_times_ms, spike_clusters

# --- Behavioral File Copy Function  ---
def copy_behavioral_file_for_dir(parent_dir, df, spike_glx_col='spike glx file', behavioral_file_col='behavioral file'):
    """
    For a given parent_dir, check if any spikeglx file in the CSV is present in that directory.
    If so, find the corresponding behavioral file and copy it to parent_dir if not already present.
    Parameters:
        parent_dir (str): Directory to check for spikeglx file and to copy behavioral file into.
        df (pd.DataFrame): DataFrame containing experiment metadata.
        spike_glx_col (str): Column name for spikeglx file in df.
        behavioral_file_col (str): Column name for behavioral file in df.
        root_directory (str): Root directory to search for behavioral files.
    Returns:
        str: Path to the behavioral file in the parent_dir (if copied or found).
    """
    
    for idx, row in df.iterrows():
        spikeglx_base = str(row[spike_glx_col]).strip()
        spikeglx_name = f"catgt_{spikeglx_base}"
        behavioral_full_name = str(row[behavioral_file_col]).strip()
        behavioral_name = os.path.basename(behavioral_full_name)
        if not spikeglx_name or not behavioral_full_name:
            continue
        if any(spikeglx_name in entry for entry in os.listdir(parent_dir)):
            print(f"Spikeglx file found in {parent_dir}")

            behav_file = None
            for dirpath, _, filenames in os.walk(parent_dir):
                if behavioral_name in filenames:
                    behav_file = os.path.join(dirpath, behavioral_name)
                    print(f"Behavioral file found in {dirpath}")
                    break
            if not behav_file:
                dest_path = parent_dir
                shutil.copy2(behavioral_full_name, dest_path)
                print(f"Copied {behavioral_full_name} to {dest_path}")
                behav_file = os.path.join(dest_path, behavioral_name)
                break

    return behav_file
# /ems/elsc-labs/mizrahi-a/Shared/
# --- Main Analysis Loop ---
def main():
    """
    Main analysis workflow. Finds KS folders, copies behavioral files, and (optionally) runs further analysis for each folder.
    """
    root_directory = r"Z:\Shared\Amichai\NPXL\Recs\group5"
    csv_path = r"Z:\Shared\Amichai\Code\DB\users_data\Amichai\NPXL recordings _experimental_data.csv"
    df_csv = pd.read_csv(csv_path)
    ks_folders = find_ks_folders(root_directory)
    for working_dir in ks_folders:
        try:
            working_dir = ks_folders[3]
            os.chdir(working_dir)
            print(working_dir)
            import glob
            parent_dir = os.path.dirname(working_dir)
            probe_id = working_dir[-1]
            data_dir = os.path.join(working_dir, f"imec{probe_id}_ks4")
            behavioral_file_path = copy_behavioral_file_for_dir(parent_dir, df_csv)
            try:
                trial_types_df, raw_events_df, session_date, session_time, trial_settings, notes , licks, states, stimuli, Unique_Stimuli_Values, tones_per_class, boundaries = load_mat_file(behavioral_file_path)
                combined_row_df = create_single_row_with_outcome(behavioral_file_path, trial_types_df, raw_events_df, session_date,
                                                             session_time, trial_settings, notes, licks, states,
                                                             Unique_Stimuli_Values, tones_per_class, boundaries)
                print(combined_row_df)
            except Exception as e:
                print(f"Error processing {behavioral_file_path}: {e}")

            # --- Load Spikeglx Data ---
            sr = read_sample_rate(working_dir)
            spike_times, spike_clusters = load_spike_data(data_dir, sr)
            spike_matrix, good_spike_matrix, mua_spike_matrix, non_somatic_spike_matrix, all_cluster_indices = extract_spike_matrices(spike_times, spike_clusters, cluster_info)
            peri_event_spikes = peri_event_spike_matrix(spike_matrix, stimuli['stimulus_onset'], pre_event_ms=500, post_event_ms=500, bin_size_ms=1)
            print(peri_event_spikes)

        except Exception as e:
            print(f"Error processing {working_dir}: {e}")
            continue

if __name__ == "__main__":
    main()

