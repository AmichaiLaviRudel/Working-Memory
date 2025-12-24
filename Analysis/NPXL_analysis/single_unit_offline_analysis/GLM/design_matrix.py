
# %%
import sys
import os



# Add workspace root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up 4 levels: GLM -> single_unit_offline_analysis -> NPXL_analysis -> Analysis -> DB (workspace root)
workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))

if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any

# Configure matplotlib backend for inline display
try:
    from IPython import get_ipython
    ipython = get_ipython()
    if ipython is not None:
        # In Jupyter/IPython, use inline backend
        ipython.run_line_magic('matplotlib', 'inline')
    else:
        # In VS Code or other environments, try to use a GUI backend
        try:
            matplotlib.use('TkAgg')  # Try TkAgg first
        except:
            try:
                matplotlib.use('Qt5Agg')  # Fallback to Qt5
            except:
                pass  # Use default backend
except ImportError:
    # Not in IPython, try GUI backend
    try:
        matplotlib.use('TkAgg')
    except:
        pass

from Analysis.NPXL_analysis.single_unit_offline_analysis.data_loading import (
    load_data,
    load_unit_labels,
    load_full_event_windows_data,
    load_behavioral_data,
)
from Analysis.NPXL_analysis.single_unit_offline_analysis.unit import (
    create_units_from_event_data,
)
from Analysis.NPXL_analysis.single_unit_offline_analysis.utils import (
    setup_results_directory,
    save_dataframe_to_csv,
)

def build_trial_design_matrix(
    time_axis: np.ndarray,
    stimuli_outcome_df: pd.DataFrame,
    metadata: Dict[str, Any],
    lick_event_windows_matrix: np.ndarray,
) -> Tuple[pd.DataFrame, np.ndarray, float]:
    """
    Build a per-trial behavioral design matrix from lightweight inputs.

    Args:
        time_axis: 1D array of time points for each trial window.
        stimuli_outcome_df: DataFrame with at least 'stimulus', 'outcome',
            and 'outcome_time' columns.
        metadata: Dict containing at least 'bin_size' (and optionally
            'window_duration').
        lick_event_windows_matrix: 3D array [channels × time_bins × trials]
            with non-zero entries at lick times.

    Returns:
        design_matrix: DataFrame with one row per trial and columns:
            - 'stimulus_type'
            - 'stimulus_category' (Go/NoGo, 1/0)
            - 'previous_outcome' (integer outcome code, NaN for first trial)
            - 'outcome_type' (integer outcome code)
            - 'licks_bins' (tuple from np.where on lick matrix)
            - 'outcomes_bins' (outcome time bin within the window, NaN if none)
            - 'stimulus_bin' (stimulus time bin within the window)
        time_axis: Time axis array shared by all trials.
        bin_size: Time bin size in seconds.
    """

    bin_size: float = float(metadata.get("bin_size", 0.005))  # default 5 ms

    stimuli_type: np.ndarray = stimuli_outcome_df["stimulus"].values
    outcomes_time_bins: np.ndarray = stimuli_outcome_df["outcome_time"].values
    outcomes: np.ndarray = stimuli_outcome_df["outcome"].values

    # Map outcome strings to numeric codes for downstream modeling
    outcome_mapping: Dict[str, int] = {
        "hit": 1,
        "miss": 2,
        "cr": 3,
        "false alarm": 4,
        "catch": 5,
    }
    outcomes_numeric = np.array(
        [outcome_mapping.get(str(o).lower(), 0) for o in outcomes],
        dtype=int,
    )

    n_trials: int = len(stimuli_type)
    stim_bin: float = len(time_axis) / 2  # approximate stimulus bin (center)
    # Outcome times are relative to stimulus; shift into window coordinates
    outcomes_time_bins = outcomes_time_bins + stim_bin

    stimulus_types = []
    stimulus_categories = []
    previous_outcomes = []
    outcome_types = []
    licks_bins_list = []
    outcomes_bins_list = []
    stimulus_bins = []

    for trial_idx in range(n_trials):
        # Stimulus type and category (Go/NoGo based on frequency bounds)
        stim_type = float(stimuli_type[trial_idx])
        stimulus_types.append(stim_type)

        if stim_type < 0.983 or stim_type > 1.525:
            stim_category = 1  # Go
        else:
            stim_category = 0  # NoGo
        stimulus_categories.append(stim_category)

        # Previous outcome code; NaN for the first trial
        if trial_idx > 0:
            previous_outcome = int(outcomes_numeric[trial_idx - 1])
        else:
            previous_outcome = np.nan
        previous_outcomes.append(previous_outcome)

        # Current outcome code
        outcome_types.append(int(outcomes_numeric[trial_idx]))

        # Outcome time bin within the event window (if present)
        outcome_time_bin = outcomes_time_bins[trial_idx]
        if np.isnan(outcome_time_bin):
            outcomes_bins_list.append(np.nan)
        else:
            if outcome_time_bin < 0 or outcome_time_bin > len(time_axis):
                # Outcome is outside the window; keep as NaN for clarity
                outcomes_bins_list.append(np.nan)
            else:
                outcomes_bins_list.append(int(outcome_time_bin))

        # Lick bins: indices where lick_event_windows_matrix is non-zero
        licks_bins = np.where(lick_event_windows_matrix[:, :, trial_idx] != 0)
        licks_bins_list.append(licks_bins)

        # Stimulus bin is the same for all trials (center of window)
        stimulus_bins.append(stim_bin)

    design_matrix = pd.DataFrame(
        {
            "stimulus_type": stimulus_types, 
            "stimulus_category": stimulus_categories,
            "previous_outcome": previous_outcomes,
            "outcome_type": outcome_types,
            "licks_bins": licks_bins_list,
            "outcomes_bins": outcomes_bins_list,
            "stimulus_bin": stimulus_bins,
        }
    )

    return design_matrix, time_axis, bin_size


# Configure matplotlib for inline display
try:
    from IPython import get_ipython
    ipython = get_ipython()
    if ipython is not None:
        # In Jupyter/IPython, use inline backend via magic command
        ipython.run_line_magic('matplotlib', 'inline')
    else:
        # In VS Code or other environments, use inline backend directly
        matplotlib.use('module://matplotlib_inline.backend_inline')
except (ImportError, AttributeError):
    # Fallback: try to set inline backend
    try:
        matplotlib.use('module://matplotlib_inline.backend_inline')
    except:
        # Last resort: use Agg backend (non-interactive, but won't error)
        matplotlib.use('Agg')
        print("Warning: Using non-interactive backend. Plots will not display inline.")

# %%
# parent_dir = r"Z:\Shared\Amichai\NPXL\Recs\group7\catgt_G7A3_novice_2b_4t_g0"
# data_dir_ACx =r"Z:\Shared\Amichai\NPXL\Recs\group7\catgt_G7A3_novice_2b_4t_g0\G7A3_novice_2b_4t_g0_imec0"
# data_dir_OFC =r"Z:\Shared\Amichai\NPXL\Recs\group7\catgt_G7A3_novice_2b_4t_g0\G7A3_novice_2b_4t_g0_imec1"



# # Load behavioral data
# acx_event_windows_data = load_behavioral_data(data_dir_ACx)

# # Build per-trial design matrix using helper (for interactive exploration)
# # Handle both full event_windows_data (6 values) and behavioral-only data (5 values)
# if len(acx_event_windows_data) == 6:
#     event_windows_matrix, time_axis, valid_event_indices, stimuli_outcome_df, metadata, lick_event_windows_matrix = acx_event_windows_data
# else:
#     lick_event_windows_matrix, time_axis, valid_event_indices, stimuli_outcome_df, metadata = acx_event_windows_data

# design_matrix, time_axis, bin_size = build_trial_design_matrix(
#     time_axis=time_axis,
#     stimuli_outcome_df=stimuli_outcome_df,
#     metadata=metadata,
#     lick_event_windows_matrix=lick_event_windows_matrix,
# )


# %%
# Plot to visualize events for one row of the design matrix
def plot_trial_events(trial_idx: int, design_matrix: pd.DataFrame, time_axis: np.ndarray, unit_idx: int = 0, event_windows_matrix: np.ndarray = None):
    """
    Create a static plot showing events for a single trial.
    
    Args:
        trial_idx: Index of the trial to plot
        design_matrix: DataFrame with trial data
        time_axis: Time axis array
        unit_idx: Index of the unit to plot spikes for
        event_windows_matrix: 3D array of shape [units × time × events] for spike data
    """
    if trial_idx >= len(design_matrix):
        print(f"Trial index {trial_idx} out of range. Max index: {len(design_matrix) - 1}")
        return
    
    row = design_matrix.iloc[trial_idx]
    
    # Outcome name mapping
    outcome_names = {1: 'Hit', 2: 'Miss', 3: 'CR', 4: 'FA', 5: 'Catch', 0: 'Unknown'}
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Extract event times
    stim_bin = int(row['stimulus_bin'])
    stim_time = time_axis[stim_bin] if stim_bin < len(time_axis) else None
    
    # Extract outcome bin - handle NaN and type conversion
    outcome_bin_val = row['outcomes_bins']
    if pd.isna(outcome_bin_val):
        outcome_bin = None
        outcome_time = None
    else:
        try:
            outcome_bin = int(outcome_bin_val)
            if 0 <= outcome_bin < len(time_axis):
                outcome_time = time_axis[outcome_bin]
            else:
                outcome_time = None
                print(f"Warning: Outcome bin {outcome_bin} out of bounds [0, {len(time_axis)})")
        except (ValueError, TypeError) as e:
            outcome_bin = None
            outcome_time = None
            print(f"Warning: Could not convert outcome_bin {outcome_bin_val} to int: {e}")
    
    # Debug output
    print(f"Trial {trial_idx}: outcome_bin={outcome_bin}, outcome_time={outcome_time}, outcome_type={row['outcome_type']}")
    
    # Extract lick events
    licks_bins = row['licks_bins']
    # licks_bins is a tuple from np.where: (channel_indices, time_indices)
    if len(licks_bins) > 1 and len(licks_bins[1]) > 0:
        lick_time_indices = licks_bins[1]  # Second element contains time indices
        # Ensure indices are within bounds
        valid_indices = lick_time_indices[lick_time_indices < len(time_axis)]
        lick_times = time_axis[valid_indices] if len(valid_indices) > 0 else np.array([])
    else:
        lick_times = np.array([])
    
    # Extract spike events for the specified unit
    spike_times = np.array([])
    if event_windows_matrix is not None:
        try:
            if unit_idx < event_windows_matrix.shape[0] and trial_idx < event_windows_matrix.shape[2]:
                # Get spike data for this unit and trial: [time_bins]
                unit_trial_data = event_windows_matrix[unit_idx, :, trial_idx]
                # Find bins with spikes (non-zero values)
                spike_bins = np.where(unit_trial_data > 0)[0]
                if len(spike_bins) > 0:
                    # Ensure indices are within bounds
                    valid_spike_bins = spike_bins[spike_bins < len(time_axis)]
                    spike_times = time_axis[valid_spike_bins] if len(valid_spike_bins) > 0 else np.array([])
        except (IndexError, AttributeError) as e:
            print(f"Warning: Could not extract spikes for unit {unit_idx}, trial {trial_idx}: {e}")
    
    # Plot stimulus event
    if stim_time is not None:
        ax.scatter(stim_time, 3, s=150, c='blue', marker='v', label='Stimulus', zorder=3)
    
    # Plot outcome event
    if outcome_time is not None:
        outcome_name = outcome_names.get(int(row['outcome_type']), 'Unknown')
        ax.scatter(outcome_time, 2, s=200, c='red', marker='s', edgecolors='darkred', linewidths=2, 
                   label=f'Outcome ({outcome_name})', zorder=3)
    else:
        print(f"Warning: Outcome time is None for trial {trial_idx}. outcome_bin={outcome_bin_val}")
    
    # Plot lick events
    if len(lick_times) > 0:
        ax.scatter(lick_times, np.ones(len(lick_times)), s=50, c='green', marker='o', label='Licks', zorder=3)
    
    # Plot spike events
    if len(spike_times) > 0:
        ax.scatter(spike_times, np.ones(len(spike_times)) * 0.5, s=30, c='purple', marker='|', linewidths=2, 
                   label=f'Spikes (Unit {unit_idx})', zorder=2)
    
    # Add time axis reference line at stimulus time
    if stim_time is not None:
        ax.axvline(x=stim_time, color='gray', linestyle='--', alpha=0.5, label='Stimulus time')
    
    # Set labels and title
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Event Type')
    ax.set_yticks([0.5, 1, 2, 3])
    ax.set_yticklabels(['Spikes', 'Licks', 'Outcome', 'Stimulus'])
    ax.set_ylim(0, 4)
    # Map outcome type to string
    outcome_type_str = outcome_names.get(int(row['outcome_type']), 'Unknown')
    
    # Map previous outcome to string
    if pd.isna(row['previous_outcome']):
        prev_outcome_str = 'NaN'
    else:
        prev_outcome_str = outcome_names.get(int(row['previous_outcome']), 'Unknown')
    
    # Map stimulus category to string (0 = NoGo, 1 = Go based on threshold)
    stim_category_str = 'Go' if row['stimulus_category'] == 1 else 'NoGo'
    
    ax.set_title(f'Trial {trial_idx} Events (Unit {unit_idx}) | Stimulus: {row["stimulus_type"]:.3f} (Cat: {stim_category_str}) | '
                 f'Outcome: {outcome_type_str} | Prev Outcome: {prev_outcome_str}')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Display plot - use plt.show() which works in both IPython and regular Python
    # The matplotlib inline backend will handle display automatically
    plt.show()








# %%
# # Plot trial 0 by default (user can change trial_idx)
# trial_idx = 150
# unit_idx = 15
# plot_trial_events(trial_idx, design_matrix, time_axis, unit_idx)


# %%
