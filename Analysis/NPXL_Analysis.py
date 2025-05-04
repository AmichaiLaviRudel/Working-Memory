from functions import *

import streamlit as st
import scipy.io as spio
import numpy as np
import pandas as pd
from scipy.stats import norm
import os

import matplotlib.pyplot as plt
import tqdm
import os



def preprocess(working_dir, spike_sample_rate=30000, events_sample_rate=20000, active_unit_threshold=15,
               window_in_sec=1, rerun=False):
    """this function takes the data from phy and the GNG for each working directory and preprocess it to a matrix of
    spikes for each cluster and event. the function saves the matrix in the analysis_output folder"""
    import numpy as np
    from tqdm import tqdm
    import os
    root_directory = r"Z:\Shared\Amichai\NPXL\Recs\Group2"
    ks_folders = []
    # Walk through all directories and files
    for dirpath, dirnames, filenames in os.walk(root_directory):
        try:
            # Check if params.py and phy.log exist in the current directory
            if 'params.py' in filenames and 'phy.log' in filenames:
                working_dir = dirpath
                print(working_dir)

                # Path to b_file.txt
                b_file_path = os.path.join(working_dir, 'b_file.txt')

                while True:
                    # If b_file.txt exists, read its content
                    if os.path.exists(b_file_path):
                        with open(b_file_path, 'r') as f:
                            behavior_dir = f.read().strip()  # Read and strip whitespace

                        # If the stored path is not a valid .mat file, delete b_file.txt and ask again
                        if not os.path.isfile(behavior_dir) or not behavior_dir.lower().endswith('.mat'):
                            os.remove(b_file_path)
                            print("Invalid or missing .mat file. b_file.txt deleted.")
                        else:
                            break  # Valid file, exit loop

                    # Prompt user for a valid MATLAB behavior file
                    behavior_file = input("Enter the full path to the MATLAB behavior file (*.mat): ").strip()

                    if os.path.isfile(behavior_file) and behavior_file.lower().endswith('.mat'):
                        with open(b_file_path, 'w') as f:
                            f.write(behavior_file)  # Save the path inside b_file.txt
                        print(f"Behavior file saved in: {b_file_path}")
                        behavior_dir = behavior_file
                        break
                    else:
                        print("Invalid file. Please enter a valid MATLAB file (*.mat).")

                # Load the MATLAB behavior file
                behavior_data = load_mat_file(behavior_dir)

                meta_columns = ['group', 'animal', 'recording', 'acronym', 'event']  # columns to add to the psth data

                # Change to the directory where the data is located
                os.chdir(working_dir)

                # checking if the analysis folder exist or not.
                if not os.path.exists("analysis_output"):
                    # Create a new directory because it does not exist
                    os.makedirs("analysis_output")

                # load the data
                spikes_times = np.load(r"spike_times.npy").flatten()
                spikes_clusters = np.load(r"spike_clusters.npy").flatten()
                cluster_group = pd.read_csv(r"cluster_group.tsv", sep = '\t')
                # border_table = pd.read_csv(r"border_table.csv")
                cluster_info = pd.read_csv(r"cluster_info.tsv", sep = '\t')
                channel_positions = np.load(r"channel_positions.npy")

                spikes_times_in_sec = spikes_times / spike_sample_rate

                # Parameters:
                ms = 1000

                # ___________________________________________________________________________#
                # Find unique clusters and their maximum time
                n_clusters = np.max(spikes_clusters)

                # Find the good clusters
                good_clusters = cluster_info[cluster_info['group'] == 'good'].to_numpy()[:, 0]
                mua_clusters = cluster_info[cluster_info['group'] == 'mua'].to_numpy()[:, 0]

                all_clusters_size = len(good_clusters) + len(mua_clusters)
                # create an array of zeros with the size of the number of good clusters that can be in different sizes
                spike_matrix = np.empty(all_clusters_size, dtype = object)

                # fill the spike matrix with the spike times of each cluster
                for i in range(all_clusters_size):
                    if i < len(good_clusters):
                        query = np.argwhere(spikes_clusters == good_clusters[i]).flatten()
                    else:
                        query = np.argwhere(spikes_clusters == mua_clusters[i - len(good_clusters)]).flatten()

                    spike_matrix[i] = spikes_times_in_sec[query]

                # ___________________________________________________________________________#
                # get units position and acronym

                # Here I am assume that the max depth is approximatly the length of the probe
                cluster_info['depth_corr'] = np.abs(cluster_info['depth'] - cluster_info['depth'].max())

                # get the acronym of the good clusters
                good_clusters_acronym = cluster_info[cluster_info['group'] == 'good'].to_numpy()[:, -1]
                np.save(f"analysis_output/good_units_acronym.npy", good_clusters_acronym)

                # ___________________________________________________________________________#

                # Ensure output directory exists
                os.makedirs("analysis_output", exist_ok = True)

                # Loop through each unique outcome type
                for outcome in behavior_data["Outcome"].unique():
                    # Filter events based on the current outcome and reset index
                    outcome_matrix = behavior_data[behavior_data["Outcome"] == outcome].reset_index(drop = True)

                    if outcome_matrix.empty:
                        continue  # Skip empty outcome matrices

                    outcome_name = outcome_matrix["OutcomeName"][0][0]

                    # Define the window size in seconds and the number of time points (assuming ms is defined)
                    window_in_sec = 1
                    num_timepoints = window_in_sec  # Ensure ms is correctly defined

                    try:
                        # Pre-allocate a NaN-filled array for spike data
                        spikes_matrix_window = np.full((all_clusters_size, len(outcome_matrix), num_timepoints), np.nan)
                    except Exception as error:
                        print(f"An error occurred while initializing spike matrix for {outcome_name}: {error}")
                        continue  # Skip this outcome if an error occurs

                    # Process each event
                    for i, row in tqdm(outcome_matrix.iterrows(), total = len(outcome_matrix),
                                       desc = f"Processing {outcome_name}"):
                        # Compute event time
                        event_time = (row["TrialStartTimestamp"] + row[
                            "ToneOnset"]) / 60  # Ensure correct unit conversion
                        window_floor = event_time - window_in_sec
                        window_ceil = event_time + window_in_sec

                        # Loop over each cluster
                        for j in range(all_clusters_size):  # Fix iteration over clusters
                            cluster_spikes = spike_matrix[j]  # Assume spike times are sorted

                            # Efficiently find spikes within the time window using searchsorted
                            start_idx = np.searchsorted(cluster_spikes, window_floor, side = 'left')
                            end_idx = np.searchsorted(cluster_spikes, window_ceil, side = 'right')
                            spikes_in_window = cluster_spikes[start_idx:end_idx]

                            # Compute relative spike times
                            relative_spikes = spikes_in_window - event_time

                            # Clip to available window length
                            n_spikes = min(len(relative_spikes), num_timepoints)
                            if n_spikes > 0:
                                spikes_matrix_window[j, i, :n_spikes] = relative_spikes[:n_spikes]

                    # Save the matrix locally
                    spikes_matrix_good_units = spikes_matrix_window[:len(good_clusters), :, :]
                    np.save(f"analysis_output/{outcome_name}_matrix.npy", spikes_matrix_window)
                    np.save(f"analysis_output/{outcome_name}_good_units_matrix.npy", spikes_matrix_good_units)

                    significant_good_units = np.full((len(good_clusters), 3), np.nan)
                    for j in range(len(good_clusters)):
                        p_value, response_higher = analyze_aligned_neuron(spikes_matrix_good_units[j, :, :], j)
                        significant_good_units[j] = int(j), p_value, response_higher
                    np.save(f"analysis_output/{outcome_name}_significant_good_units.npy", significant_good_units)
        except Exception as error:
            print(f"An error occurred in {dirpath}: {error}")
            continue  # Skip this outcome if an error occurs


# ___________________________________________________________________________#


def load_mat_file(file_path):
    import numpy as np
    import scipy
    import pandas as pd
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
        outcomeName = session_data_content['OutcomeName'].flatten()
        outcome = session_data_content['Outcome'].flatten()
        trialStartTimestamp = session_data_content['TrialStartTimestamp'].flatten()

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

    # Initialize arrays
    trials = raw_events['Trial']
    n_trials = len(trial_types)
    licks = [None] * n_trials
    play_tone_time = [None] * n_trials

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
            play_tone_time
            if 'Events' in nested_structure.dtype.names:
                events = nested_structure['Events'][0, 0]

                if 'HiFi1_1' in events.dtype.names:
                    stim_t = np.min(events['HiFi1_1'][0])
                else:
                    stim_t = None
                play_tone_time[i] = stim_t
                if 'Port1In' in events.dtype.names and stim_t is not None:
                    lick = events['Port1In'][0]
                    lick_after_stim = lick[lick > stim_t]

                    if lick_after_stim.size > 0:
                        licks[i] = lick_after_stim - stim_t
        except Exception as e:
            print(f"Error processing trial {i}: {e}")

    # Convert 'TrialTypes' and stimuli to a DataFrame
    toneOnset_df = pd.DataFrame(play_tone_time, columns = ['ToneOnset'])
    trialStartTimestamp_df = pd.DataFrame(trialStartTimestamp, columns = ['TrialStartTimestamp'])
    trial_types_df = pd.DataFrame(trial_types, columns = ['TrialType'])
    stimuli_df = pd.DataFrame(stimuli, columns = ['Stimuli'])
    outcome_df = pd.DataFrame(outcome, columns = ['Outcome'])
    outcomeName_df = pd.DataFrame(outcomeName, columns = ['OutcomeName'])

    # Convert licks to a DataFrame, handling None values properly
    licks_array = np.array([x if isinstance(x, np.ndarray) else np.nan for x in licks], dtype = object)
    licks_df = pd.DataFrame(licks_array, columns = ['licks'])

    trial_df = pd.concat(
        [trialStartTimestamp_df, trial_types_df, stimuli_df, outcome_df, outcomeName_df, toneOnset_df, licks_df],
        axis = 1)

    return trial_df


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel


def analyze_aligned_neuron(aligned_spike_trials, unit_idx,
                           baseline_window=(-1, 0), response_window=(0, 1),
                           bin_size=0.01, alpha=0.05):
    """
    Analyzes a single neuron's response using spikes that are already aligned to event onset (time 0).

    Parameters:
        aligned_spike_trials : list of 1D arrays or lists
            Each element contains the spike times (in seconds) for one trial, already aligned
            so that event onset is at 0.
        baseline_window : tuple
            Time window (in seconds) for the baseline period (default is (-1, 0)).
        response_window : tuple
            Time window (in seconds) for the response period (default is (0, 1)).
        bin_size : float
            Bin size in seconds for the PSTH (default is 0.01 sec).
        alpha : float
            Significance level for the test (default is 0.05).

    Returns:
        p_value : float
            The p-value from the paired t-test comparing response and baseline firing rates.
        response_higher : str
            "higher" if response firing rate is greater, "lower" if less, "no significant difference" if p > alpha.
    """
    baseline_rates = []
    response_rates = []

    # Loop over each trial
    for trial_spikes in aligned_spike_trials:
        trial_spikes = np.array(trial_spikes)
        # Count spikes in the baseline and response windows
        baseline_count = np.sum((trial_spikes >= baseline_window[0]) & (trial_spikes < baseline_window[1]))
        response_count = np.sum((trial_spikes >= response_window[0]) & (trial_spikes < response_window[1]))

        # Convert counts to firing rates (Hz) by dividing by the window duration
        baseline_rate = baseline_count / (baseline_window[1] - baseline_window[0])
        response_rate = response_count / (response_window[1] - response_window[0])

        baseline_rates.append(baseline_rate)
        response_rates.append(response_rate)

    baseline_rates = np.array(baseline_rates)
    response_rates = np.array(response_rates)

    # Perform a paired t-test comparing response and baseline firing rates
    t_stat, p_value = ttest_rel(response_rates, baseline_rates)

    # Determine if response rate is higher or lower
    if p_value < alpha:
        response_higher = "1" if np.mean(response_rates) > np.mean(baseline_rates) else "-1"
    else:
        response_higher = "0"

    # # --- Plotting ---
    # if p_value < 0.00005:
    #     fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (10, 8), sharex = True)
    #
    #     # Raster Plot
    #     for i, trial_spikes in enumerate(aligned_spike_trials):
    #         ax1.vlines(trial_spikes, i + 0.5, i + 1.5)
    #     ax1.axvline(0, color = 'red', linestyle = '--', label = 'Event onset')
    #     ax1.set_ylabel('Trial')
    #     ax1.set_title(f'Raster Plot, unit: {unit_idx}')
    #     ax1.legend()
    #
    #     # PSTH (Peri-Stimulus Time Histogram)
    #     t_min = baseline_window[0]
    #     t_max = response_window[1]
    #     bins = np.arange(t_min, t_max + bin_size, bin_size)
    #
    #     # Concatenate spikes from all trials
    #     all_spikes = np.concatenate([np.array(trial) for trial in aligned_spike_trials])
    #     counts, edges = np.histogram(all_spikes, bins = bins)
    #     # Convert counts to firing rate (Hz)
    #     psth = counts / (len(aligned_spike_trials) * bin_size)
    #
    #     ax2.bar(edges[:-1], psth, width = bin_size, align = 'edge')
    #     ax2.axvline(0, color = 'red', linestyle = '--', label = 'Event onset')
    #     ax2.set_xlabel('Time (s)')
    #     ax2.set_ylabel('Firing Rate (Hz)')
    #     ax2.set_title(f'PSTH (p = {p_value:.3g}, {response_higher})')
    #     ax2.legend()
    #
    #     plt.tight_layout()
    #     plt.show()

    return p_value, response_higher


working_dir = r"Z:\Shared\Amichai\NPXL\Recs\Group3"

preprocess(working_dir, spike_sample_rate = 30000, events_sample_rate = 20000, active_unit_threshold = 15,
           window_in_sec = 1, rerun = False)


