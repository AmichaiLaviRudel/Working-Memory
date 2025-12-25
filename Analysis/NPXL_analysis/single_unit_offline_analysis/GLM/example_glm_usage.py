#%%
from typing import Any
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap

import nemos as nmo

# some helper plotting functions
from nemos import _documentation_utils as doc_plots
from patsy import dmatrix
from matplotlib.patches import FancyBboxPatch
# configure pynapple to ignore conversion warning
nap.nap_config.suppress_conversion_warnings = True

# configure plots some
plt.style.use(nmo.styles.plot_style)

# Set modern scientific font (Lato or Inter)
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
# Try Lato first, then Inter, then fallback to system sans-serif
matplotlib.rcParams['font.sans-serif'] = ['Lato', 'Inter', 'DejaVu Sans', 'Helvetica', 'Liberation Sans']
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.labelsize'] = 11
matplotlib.rcParams['axes.titlesize'] = 12
matplotlib.rcParams['xtick.labelsize'] = 9
matplotlib.rcParams['ytick.labelsize'] = 9
matplotlib.rcParams['legend.fontsize'] = 9
matplotlib.rcParams['figure.titlesize'] = 14



# %%
# General spike loader for multiple probes
# imec0 = ACx (Auditory Cortex)
# imec1 = OFC (Orbitofrontal Cortex)

base_path = r"Z:\Shared\Amichai\NPXL\Recs\group5\catGTGroup5\catgt_G5A3_2b_4t_new2_g0"


def load_probe_spikes(base_path, imec_name, region_name):
    """
    Load spikes from a single probe with unit type labels.
    
    Parameters
    ----------
    base_path : str
        Base recording path
    imec_name : str
        Probe name (e.g., 'imec0', 'imec1')
    region_name : str
        Brain region name (e.g., 'ACx', 'OFC')
        
    Returns
    -------
    spike_list : list of nap.Ts
        List of spike trains
    unit_types : list of str
        List of unit type labels
    regions : list of str
        List of region labels
    cluster_ids : list of int
        List of original cluster IDs
    """
    # Extract recording name without 'catgt_' prefix for subdirectory naming
    base_name = base_path.split('\\')[-1]
    # Remove 'catgt_' prefix if present
    rec_name = base_name.replace('catgt_', '') if base_name.startswith('catgt_') else base_name
    probe_path = f"{base_path}\\{rec_name}_{imec_name}"
    
    # Load spike data
    spike_times = np.load(f"{probe_path}\\{imec_name}_ks4\\spike_times_sec_adj.npy")
    spike_clusters = np.load(f"{probe_path}\\{imec_name}_ks4\\spike_clusters.npy")
    unit_labels = pd.read_csv(f"{probe_path}\\bombcell\\unit_labels.tsv", sep="\t")
    

    # Numeric to string unit type mapping
    unit_type_numeric_map = {
        0: 'noise',
        1: 'good',
        2: 'mua',
        3: 'non-somatic'
    }

    # Create unit type map
    unit_type_map = {}
    for unit_id, row in unit_labels.iterrows():
        label_raw = row.iloc[0]
        
        # Handle both numeric and string formats
        if isinstance(label_raw, (int, float, np.integer, np.floating)):
            label = unit_type_numeric_map.get(int(label_raw), 'unknown')
        else:
            label_str = str(label_raw).upper()
            if 'GOOD' in label_str:
                label = 'good'
            elif 'MUA' in label_str:
                label = 'mua'
            elif 'NON-SOMA' in label_str or 'NONSOMATIC' in label_str:
                label = 'non-somatic'
            elif 'NOISE' in label_str:
                label = 'noise'
            else:
                label = 'unknown'
        
        unit_type_map[int(unit_id)] = label
    
    # Build lists for reindexing
    spike_list = []
    unit_types = []
    regions = []
    cluster_ids = []
    
    for cl in np.unique(spike_clusters):
        unit_times = spike_times[spike_clusters == cl]
        spike_list.append(nap.Ts(unit_times))
        unit_types.append(unit_type_map.get(int(cl), 'unknown'))
        regions.append(region_name)
        cluster_ids.append(int(cl))
    
    print(f"Loaded {len(spike_list)} units from {region_name} ({imec_name})")
    print(f"  Unit types: {pd.Series(unit_types).value_counts().to_dict()}")
    
    return spike_list, unit_types, regions, cluster_ids, probe_path


# Load both probes
spike_list_acx, unit_types_acx, regions_acx, cluster_ids_acx, probe_path_acx = load_probe_spikes(
    base_path, 'imec0', 'ACx'
)

spike_list_ofc, unit_types_ofc, regions_ofc, cluster_ids_ofc, probe_path_ofc = load_probe_spikes(
    base_path, 'imec1', 'OFC'
)

# Combine into single lists
all_spikes = spike_list_acx + spike_list_ofc
all_unit_types = unit_types_acx + unit_types_ofc
all_regions = regions_acx + regions_ofc
all_cluster_ids = cluster_ids_acx + cluster_ids_ofc

spike_dict_reindexed = {i: spk for i, spk in enumerate(all_spikes)}
spikes = nap.TsGroup(spike_dict_reindexed)

# Add metadata (unit_type, region, original cluster_id)
spikes.set_info(
    unit_type=all_unit_types,
    region=all_regions,
    cluster_id=all_cluster_ids
)

print(f"\nCombined: {len(spikes)} total units")
print(f"By region: {pd.Series(all_regions).value_counts().to_dict()}")
print(f"By unit type: {pd.Series(all_unit_types).value_counts().to_dict()}")


#%% Parameters
# Minimum average firing rate threshold (Hz) for selecting units
RATE_TH = 1

# Bin size for binning spike counts and predictors (in seconds)
BIN_SIZE = 0.01

# Window for peri-event epoching (in seconds, relative to event)
EPOCH_START = -1
EPOCH_END = 3

# Example epoch (for plotting): here from 236s to 242s
EXAMPLE_EPOCH = nap.IntervalSet(start=30, end=90)

# Which unit to show as example when plotting
EXAMPLE_NEURON_ID = 1

# Bin size for preprocessing continuous data used in feature extraction (in seconds)
PREROCEESING_BIN_SIZE = 0.005

# Thresholds for categorizing stimulus into "Go" or "NoGo"
LOW_BOUNDARY_TH = 0.983
HIGH_BOUNDARY_TH = 1.525

# Window size (in seconds) after event, used for temporal event-aligned features (causal)
EVENT_WINDOW_SEC = 4  # seconds after event

# Acausal window for categorical features (captures effects before and after trial characteristics)
ACAUSAL_BEFORE_SEC = 1  # seconds before event
ACAUSAL_AFTER_SEC = 3   # seconds after event

# Spike history window size (in seconds) for spike history predictor
HISTORY_WINDOW_SEC = 1

# Acausal window for spike history in population GLM (captures effects before and after spike)
HISTORY_ACAUSAL_BEFORE_SEC = 1  # seconds before spike
HISTORY_ACAUSAL_AFTER_SEC = 1   # seconds after spike

# Number of raised cosine basis functions for all event-locked convolutions
N_BASIS_FUNCS = 8

# Parameter to control whether to include spike history
N_POPULATION = 300

INCLUDE_SPIKE_HISTORY = True  # Set to False to exclude spike history
# Note: When True, creates ALL-TO-ALL connectivity - each neuron's spike history 
#       features are included and can affect ALL neurons via PopulationGLM coefficients

#%% filtering
spikes = spikes.getby_category("unit_type")["good"]
spikes = spikes.getby_threshold("rate", RATE_TH)


# %% 
base_name = base_path.split('\\')[-1]
# Remove 'catgt_' prefix if present
rec_name = base_name.replace('catgt_', '') if base_name.startswith('catgt_') else base_name

licks = np.loadtxt(os.path.join(base_path, f"{rec_name}_tcat.nidq.xd_0_2_0.txt"))
tone_onset = np.loadtxt(os.path.join(base_path, f"{rec_name}_tcat.nidq.xd_0_1_100.txt"))
stimuli_outcome_df = pd.read_csv(os.path.join(probe_path_acx, "analysis_output", "stimuli_outcome_df.csv"))


stimulus = np.round(stimuli_outcome_df["stimulus"].values.astype(float), 2)
outcome_str = stimuli_outcome_df["outcome"].astype(str).str.lower().values
outcome_time_bins = (stimuli_outcome_df['outcome_time'] ).values.astype(float) 
outcome_time = np.nan_to_num((outcome_time_bins) * PREROCEESING_BIN_SIZE + tone_onset, nan=0)
previous_outcome = np.roll(outcome_str, 1)
previous_outcome[0] = '0'
# Category: Go (1) vs NoGo (0)
category = np.where((stimulus < LOW_BOUNDARY_TH) | (stimulus > HIGH_BOUNDARY_TH), 'Go', 'NoGo')


# Stack features as 2D array [n_times × 3]
data = pd.DataFrame({"stimulus_ID": stimulus,
                    "category_ID": category,
                    "outcome_ID": outcome_str, 
                    "previous_outcome": previous_outcome})

#%%
formula = "C(stimulus_ID) + category_ID + outcome_ID + previous_outcome"
categorical_design_matrix = dmatrix(formula, data, return_type="dataframe")
categorical_design_matrix.drop(columns=["Intercept"], inplace=True)

categorical_features = nap.TsdFrame(
    t=tone_onset,
    d=categorical_design_matrix.values,
    columns=categorical_design_matrix.columns,
)


#%%

# Example: lick / stimulus / outcome times as Ts (point processes)
max_time = float(
    np.nanmax(
        np.array(
            [
                np.nanmax(tone_onset),  # last tone onset in seconds
                np.nanmax(licks),       # last lick in seconds
                np.nanmax(outcome_time) # last outcome event in seconds
            ]
        )
    )
)

full_ep = nap.IntervalSet(start=0.0, end=max_time)  # count from t=0 to the last event

tone_onset_count = nap.Ts(tone_onset.astype(float)).count(BIN_SIZE, ep=full_ep)
licks_count = nap.Ts(licks.astype(float)).count(BIN_SIZE, ep=full_ep)
outcome_time_count = nap.Ts(outcome_time.astype(float)).count(BIN_SIZE, ep=full_ep)

# All three counts now share the same time axis (full_ep); stack their counts
arrays = [tone_onset_count.d, licks_count.d, outcome_time_count.d]
data = np.column_stack(arrays)

temporal_features = nap.TsdFrame(
    t=tone_onset_count.t,  # common time base from 0 to max_time
    d=data,
    columns=["tone_onset", "licks", "outcome_onset"],
)

#%% spikecount
unit_ids = list(spikes.keys())
spike_count = spikes.count(BIN_SIZE, ep=full_ep)

spike_count = nap.TsdFrame(
    t=spike_count.t,
    d=spike_count.values[:, np.argsort(unit_ids)],
    columns=unit_ids,
)

#%% Build epochs starting at each tone onset
start = tone_onset - EPOCH_START
end = tone_onset + EPOCH_END # 3 seconds after tone onset
epochs = nap.IntervalSet(start=start, end=end)


# %%
# select a neuron's spike count time series
neuron_count = spike_count[:, EXAMPLE_NEURON_ID]

# restrict to a smaller time interval
epoch_one_spk = EXAMPLE_EPOCH
fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)

# Convert spike count to firing rate and smooth
neuron_count_restricted = neuron_count.restrict(epoch_one_spk)
firing_rate_tsd = nap.Tsd(t=neuron_count_restricted.t, d=neuron_count_restricted.d / BIN_SIZE)
firing_rate_smooth = firing_rate_tsd.smooth(std=0.05, windowsize=0.25)

# Create regular time grid for filled area plot
t_start = epoch_one_spk.start[0]
t_end = epoch_one_spk.end[0]
t_full = np.arange(t_start, t_end + BIN_SIZE, BIN_SIZE)
fr_filled = np.zeros_like(t_full, dtype=float)

# Fill in smoothed values
smooth_times = firing_rate_smooth.t
smooth_values = firing_rate_smooth.d
for i, t_val in enumerate(t_full):
    dists = np.abs(smooth_times - t_val)
    if len(dists) > 0 and np.min(dists) < BIN_SIZE:
        fr_filled[i] = smooth_values[np.argmin(dists)]

# Create gradient area plot for firing rate
n_gradient_layers = 15
for i in range(n_gradient_layers):
    y_bottom = fr_filled * (i / n_gradient_layers)
    y_top = fr_filled * ((i + 1) / n_gradient_layers)
    alpha = 0.4 * ((i + 1) / n_gradient_layers)
    axes[0].fill_between(t_full, y_bottom, y_top, alpha=alpha, color='tab:blue', linewidth=0)

axes[0].plot(t_full, fr_filled, color='tab:blue', linewidth=2, alpha=0.7, label="Firing Rate")
axes[0].set_ylabel("Firing Rate (Hz)")
axes[0].set_title("Spike Count Time Series (Smoothed)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Tone onset (keep as step plot for events)
tone_data = temporal_features["tone_onset"].restrict(epoch_one_spk)
axes[1].step(tone_data.t, tone_data.d, where="post", color="red", alpha=0.7, linewidth=1.5, label="Tone Onset")
axes[1].set_ylabel("Event")
axes[1].set_title("Tone Onset")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Licks
lick_data = temporal_features["licks"].restrict(epoch_one_spk)
axes[2].step(lick_data.t, lick_data.d, where="post", color="tab:blue", alpha=0.7, linewidth=1.5, label="Licks")
axes[2].set_ylabel("Event")
axes[2].set_title("Licks")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

# Outcome Onset
outcome_data = temporal_features["outcome_onset"].restrict(epoch_one_spk)
axes[3].step(outcome_data.t, outcome_data.d, where="post", color="green", alpha=0.7, linewidth=1.5, label="Outcome Onset")
axes[3].set_xlabel("Time (sec)")
axes[3].set_ylabel("Event")
axes[3].set_title("Outcome Onset")
axes[3].legend()
axes[3].grid(True, alpha=0.3)

plt.tight_layout()


#%% NeMoS GLM
# check that the dimensionality matches NeMoS expectation
print(f"predictor shape: {temporal_features.shape}")
print(f"count shape: {neuron_count.shape}")


# construct the train and test epochs
duration = temporal_features.time_support.tot_length("s")
start = temporal_features.time_support["start"]
end = temporal_features.time_support["end"]
first_half = nap.IntervalSet(start, start + duration / 2)
second_half = nap.IntervalSet(start + duration / 2, end)


# %%
# Build NeMoS predictors using basis functions:
# - Convolve temporal features (tone_onset, licks, outcome_onset) with RaisedCosineLog basis
# - Convolve categorical features with RaisedCosineLog basis
# - Add spike-history predictor with basis
# Then fit GLM restricted to tone epochs and plot actual vs predicted FR

# --- Convert categorical features (trial-level) into binned impulses ---
# Map categorical features from tone_onset times to the binned temporal axis
tone_times = tone_onset.astype(float)
t_bins = np.asarray(temporal_features.t, dtype=float)

# Find nearest bin for each tone onset
tone_idx = np.searchsorted(t_bins, tone_times, side="left")
tone_idx = np.clip(tone_idx, 0, len(t_bins) - 1)
left_idx = np.clip(tone_idx - 1, 0, len(t_bins) - 1)
pick_left = np.abs(t_bins[left_idx] - tone_times) < np.abs(t_bins[tone_idx] - tone_times)
tone_idx[pick_left] = left_idx[pick_left]

# Create impulse time series for categorical features
categorical_impulse = np.zeros((len(t_bins), categorical_features.shape[1]), dtype=float)
for i, idx in enumerate(tone_idx):
    if i < len(categorical_features):
        categorical_impulse[idx, :] += categorical_features.values[i, :]

categorical_impulse_tsd = nap.TsdFrame(
    t=temporal_features.t,
    d=categorical_impulse,
    columns=categorical_features.columns
)

#
#%% --- Define basis functions for event predictors ---
# Causal basis for temporal features (events affect future spikes)
event_window_bins = int(EVENT_WINDOW_SEC * temporal_features.rate)

# RaisedCosineLog basis: good for capturing fast onset then decay
basis_events = nmo.basis.RaisedCosineLogConv(
    n_basis_funcs=N_BASIS_FUNCS,
    window_size=event_window_bins,
    label="temporal_events"
)

print(f"Temporal event basis (causal): {N_BASIS_FUNCS} functions over {EVENT_WINDOW_SEC}s window")

# Acausal basis for categorical features (capture effects before and after trial characteristics)
acausal_total_sec = ACAUSAL_BEFORE_SEC + ACAUSAL_AFTER_SEC
acausal_window_bins = int(acausal_total_sec * temporal_features.rate)

# Create acausal basis using RaisedCosineBasis (symmetric)
basis_categorical = nmo.basis.RaisedCosineLinearConv(
    n_basis_funcs=N_BASIS_FUNCS,
    window_size=acausal_window_bins,
    label="categorical_events"
)

print(f"Categorical basis (acausal): {N_BASIS_FUNCS} functions over {acausal_total_sec}s window ({-ACAUSAL_BEFORE_SEC}s to +{ACAUSAL_AFTER_SEC}s)")

#%% --- Convolve temporal features with causal basis ---
X_temporal_conv = basis_events.compute_features(temporal_features)
print(f"Convolved temporal features shape: {X_temporal_conv.shape}")

#%% --- Convolve categorical impulses with acausal basis ---
# Need to shift impulses by acausal_before_sec to account for lookback
categorical_impulse_shifted = nap.TsdFrame(
    t=categorical_impulse_tsd.t,
    d=categorical_impulse_tsd.values,
    columns=categorical_impulse_tsd.columns
)

X_categorical_conv = basis_categorical.compute_features(categorical_impulse_shifted)
print(f"Convolved categorical features shape: {X_categorical_conv.shape}")

#%% --- Spike-history predictor with basis ---
history_window_bins = int(HISTORY_WINDOW_SEC * neuron_count.rate)

basis_history = nmo.basis.RaisedCosineLogConv(
    n_basis_funcs= N_BASIS_FUNCS,
    window_size=history_window_bins,
    label="spike_history"
)
X_history = basis_history.compute_features(neuron_count)
print(f"Spike history features shape: {X_history.shape}")

#%% --- Combine all predictors ---
# NeMoS basis convolution preserves time base, but may introduce edge NaNs
# Find common time support by intersecting all three
common_support = X_temporal_conv.time_support.intersect(
    X_categorical_conv.time_support
).intersect(X_history.time_support)

#%% Restrict all to common support, then verify they share timestamps
X_temp_common = X_temporal_conv.restrict(common_support)
X_cat_common = X_categorical_conv.restrict(common_support)
X_hist_common = X_history.restrict(common_support)

# Verify time alignment (they should match after restrict to common support)
assert X_temp_common.shape[0] == X_cat_common.shape[0] == X_hist_common.shape[0], \
    "Predictor time bases don't align after restriction to common support"

# Build descriptive column names from original feature names + basis index
temp_cols = []
for temp_feat in temporal_features.columns:
    for basis_idx in range(N_BASIS_FUNCS):
        temp_cols.append(f"{temp_feat}_basis{basis_idx}")

cat_cols = []
for cat_feat in categorical_features.columns:
    for basis_idx in range(N_BASIS_FUNCS):
        cat_cols.append(f"{cat_feat}_basis{basis_idx}")

hist_cols = [f"spike_history_basis{i}" for i in range(basis_history.n_basis_funcs)]

X = nap.TsdFrame(
    t=X_temp_common.t,
    d=np.column_stack([
        X_temp_common.values,
        X_cat_common.values,
        X_hist_common.values,
    ]),
    columns=temp_cols + cat_cols + hist_cols
)

# Remove any NaN rows (edge effects from convolution)
valid_mask = np.all(np.isfinite(X.values), axis=1)
X = nap.TsdFrame(
    t=X.t[valid_mask],
    d=X.values[valid_mask],
    columns=X.columns
)

print(f"Combined design matrix shape: {X.shape}")

#%% --- Restrict to tone epochs ---
X_ep = X.restrict(epochs)

# Align target (neuron_count) to X_ep timestamps
# Restrict neuron_count to epochs, then to X_ep's exact time support for alignment
y_in_epochs = neuron_count.restrict(epochs)
y_ep = y_in_epochs.restrict(X_ep.time_support)

# If shapes still don't match, subsample y to X's timestamps
if y_ep.shape[0] != X_ep.shape[0]:
    # Find nearest timestamp in y for each X timestamp
    y_times = np.asarray(y_ep.t)
    x_times = np.asarray(X_ep.t)
    y_indices = np.searchsorted(y_times, x_times)
    y_indices = np.clip(y_indices, 0, len(y_times) - 1)
    y_ep = nap.Tsd(t=X_ep.t, d=y_ep.d[y_indices])

print(f"X_ep shape (in tone epochs): {X_ep.shape}")
print(f"y_ep shape (in tone epochs): {y_ep.shape}")



#%% --- Fit GLM with basis-convolved predictors ---
glm_basis = nmo.glm.GLM(solver_name="LBFGS", regularizer="Ridge", regularizer_strength=0.1)
glm_basis.fit(X_ep, y_ep)

#%%

print(f"Train pseudo-R2: {glm_basis.score(X_ep, y_ep, score_type='pseudo-r2-Cohen'):.4f}")

#%% --- Predict firing rate ---
pred_rate_hz = np.squeeze(glm_basis.predict(X_ep)) / BIN_SIZE
actual_rate_hz = np.squeeze(y_ep.d) / BIN_SIZE



#%% --- Visualize: actual vs predicted FR ---
# Use EXAMPLE_EPOCH for consistent plotting
plot_ep = EXAMPLE_EPOCH
# NeMoS helper plot - restrict data to plot window first
y_plot = y_ep.restrict(plot_ep)
pred_plot_mask = (y_ep.t >= plot_ep.start[0]) & (y_ep.t <= plot_ep.end[0])

# Create a pynapple Tsd for predicted rate in plot window
pred_rate_tsd = nap.Tsd(t=y_ep.t[pred_plot_mask], d=pred_rate_hz[pred_plot_mask])
# Smooth predicted rate using Tsd smooth method (0.25s window = 5 bins at BIN_SIZE=0.05s)
pred_rate_tsd = pred_rate_tsd.smooth(std=0.05, windowsize=0.25)

# Also smooth actual rate
actual_rate_tsd_raw = nap.Tsd(t=y_ep.t[pred_plot_mask], d=actual_rate_hz[pred_plot_mask])
actual_rate_tsd = actual_rate_tsd_raw.smooth(std=0.05, windowsize=0.25)

# Fill missing time points with zeros to ensure continuous area chart
# Create regular time grid at BIN_SIZE resolution
t_start = plot_ep.start[0]
t_end = plot_ep.end[0]
t_full = np.arange(t_start, t_end + BIN_SIZE, BIN_SIZE)
# Initialize with zeros
pred_rate_filled = np.zeros_like(t_full, dtype=float)
actual_rate_filled = np.zeros_like(t_full, dtype=float)

# Fill in actual values by finding nearest neighbors
if len(actual_rate_tsd) > 0:
    actual_times = actual_rate_tsd.t
    actual_values = actual_rate_tsd.d
    for i, t_val in enumerate(t_full):
        # Find nearest time point
        dists = np.abs(actual_times - t_val)
        if len(dists) > 0 and np.min(dists) < BIN_SIZE:
            actual_rate_filled[i] = actual_values[np.argmin(dists)]

# Fill in predicted values by finding nearest neighbors
if len(pred_rate_tsd) > 0:
    pred_times = pred_rate_tsd.t
    pred_values = pred_rate_tsd.d
    for i, t_val in enumerate(t_full):
        # Find nearest time point
        dists = np.abs(pred_times - t_val)
        if len(dists) > 0 and np.min(dists) < BIN_SIZE:
            pred_rate_filled[i] = pred_values[np.argmin(dists)]

# Direct matplotlib comparison with area chart and gradients
fig, ax = plt.subplots(1, 1, figsize=(14, 5))

# Use filled time series with zeros at all time points
t_plot = t_full
actual_plot = actual_rate_filled
pred_plot = pred_rate_filled

# Create gradient effect for area charts using multiple overlapping fills
# Orange gradient for observed (actual) firing rate - darker at top, lighter at bottom
n_gradient_layers = 15
for i in range(n_gradient_layers):
    # Create gradient from bottom (transparent) to top (opaque)
    y_bottom = actual_plot * (i / n_gradient_layers)
    y_top = actual_plot * ((i + 1) / n_gradient_layers)
    alpha = 0.4 * ((i + 1) / n_gradient_layers)  # Increasing opacity toward top
    ax.fill_between(t_plot, y_bottom, y_top, alpha=alpha, color='orange', linewidth=0)

# Green gradient for predicted firing rate - darker at top, lighter at bottom
for i in range(n_gradient_layers):
    # Create gradient from bottom (transparent) to top (opaque)
    y_bottom = pred_plot * (i / n_gradient_layers)
    y_top = pred_plot * ((i + 1) / n_gradient_layers)
    alpha = 0.4 * ((i + 1) / n_gradient_layers)  # Increasing opacity toward top
    ax.fill_between(t_plot, y_bottom, y_top, alpha=alpha, color='green', linewidth=0)

# Add semi-transparent outlines for better visibility
ax.plot(t_plot, actual_plot, color='orange', linewidth=2, alpha=0.7, label="Actual FR")
ax.plot(t_plot, pred_plot, color='green', linewidth=2, alpha=0.7, label="Predicted FR")

ax.set_xlabel("Time (sec)", fontsize=11)
ax.set_ylabel("Firing rate (Hz)", fontsize=11)
ax.set_title(f"Actual vs Predicted Firing Rate - Neuron {EXAMPLE_NEURON_ID}", fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()

#%% Calculate and plot partial contribution of each predictor (stacked bar)
import re

# Get GLM coefficients
coefs = np.asarray(glm_basis.coef_).flatten()

# Group coefficients by feature type and sum absolute values
feature_contributions = {}

# Temporal features (tone_onset, licks, outcome_onset)
for temp_feat in temporal_features.columns:
    feature_indices = [i for i, col in enumerate(X.columns) if col.startswith(f"{temp_feat}_basis")]
    if len(feature_indices) > 0:
        feature_contributions[temp_feat] = np.sum(np.abs(coefs[feature_indices]))

# Categorical features
for cat_feat in categorical_features.columns:
    # Find all columns that contain this categorical feature
    feature_indices = [i for i, col in enumerate(X.columns) if cat_feat in col and 'basis' in col]
    if len(feature_indices) > 0:
        # Handle stimulus_ID formatting first (before other replacements)
        # Check if this is a stimulus-related column in the original name
        short_name = cat_feat
        if 'stimulus' in cat_feat.lower():
            # Try to extract the numeric value from various patsy patterns
            # Pattern 1: C(stimulus_ID)[T.1.17] or C(stimulus_ID)[T.1.17]
            stim_match = re.search(r'\[T?\.?([\d.]+)\]', cat_feat)
            if not stim_match:
                # Pattern 2: Look for any number in the string (fallback)
                stim_match = re.search(r'([\d.]+)', cat_feat)
            
            if stim_match:
                stim_value = stim_match.group(1)
                short_name = f'stim:{stim_value}'
            else:
                # If no number found, just use a generic label
                short_name = 'stimulus'
        else:
            # Apply other replacements for non-stimulus features
            short_name = cat_feat.replace('category_ID[T.', 'cat:').replace('outcome_ID[T.', 'out:').replace('previous_outcome[T.', 'prev:').replace(']', '')
        
        feature_contributions[short_name] = np.sum(np.abs(coefs[feature_indices]))

# Spike history - excluded from partial contribution plot
# hist_indices = [i for i, col in enumerate(X.columns) if 'spike_history' in col and 'basis' in col]
# if len(hist_indices) > 0:
#     feature_contributions['spike_history'] = np.sum(np.abs(coefs[hist_indices]))

# Sort by contribution (descending)
sorted_features = sorted(feature_contributions.items(), key=lambda x: x[1], reverse=True)
feature_names = [name for name, _ in sorted_features]
contributions = [val for _, val in sorted_features]

# Normalize to percentages (sum to 100%)
total_contribution = sum(contributions) if contributions else 1
percentages = [(c / total_contribution) * 100 for c in contributions]

# Create vertical stacked bar plot
fig, ax = plt.subplots(1, 1, figsize=(8, 10))

# Color scheme: different colors for different feature types
colors = []
for name in feature_names:
    if name in ['tone_onset', 'licks', 'outcome_onset']:
        colors.append('#4A90E2')  # Blue for temporal
    elif name.startswith('cat:') or name.startswith('out:') or name.startswith('prev:') or name == 'stimulus':
        colors.append('#50C878')  # Green for categorical
    elif name == 'spike_history':
        colors.append('#E74C3C')  # Red for spike history
    else:
        colors.append('#95A5A6')  # Gray for others

# Create stacked bar (single vertical bar with all segments)
bottom = 0
bars = []
x_pos = 0
bar_width = 0.6

for i, (name, pct, color) in enumerate(zip(feature_names, percentages, colors)):
    bar = ax.bar(x_pos, pct, bottom=bottom, width=bar_width, color=color, 
                edgecolor='white', linewidth=1.5, label=name)
    bars.append(bar)
    
    # Add percentage and name label beside the segment
    segment_center = bottom + pct / 2
    
    # Percentage label inside segment if large enough, otherwise outside
    if pct >= 3:
        # Text inside the segment (white)
        ax.text(x_pos, segment_center, f'{pct:.1f}%',
               ha='center', va='center', fontsize=10, fontweight='bold',
               color='white')
        # Name label outside to the right (with spacing)
        ax.text(x_pos + bar_width/2 + 0.15, segment_center, name,
               ha='left', va='center', fontsize=9, color='black')
    else:
        # For small segments, place percentage on the left side and name on the right side
        # Percentage on the left
        ax.text(x_pos - bar_width/2 - 0.05, segment_center, f'{pct:.1f}%',
               ha='right', va='center', fontsize=9, fontweight='bold',
               color='black')
        # Name on the right
        ax.text(x_pos + bar_width/2 + 0.15, segment_center, name,
               ha='left', va='center', fontsize=9, color='black')
    
    bottom += pct

# Customize plot
ax.set_ylim(0, 100)
ax.set_xlim(-0.5, 1.2)  # Increased left padding for spacing between bar and y-axis
ax.set_ylabel('Relative Contribution (%)', fontsize=11)
ax.set_title('Partial Contribution of Each Predictor to GLM\n(Sum of Absolute Coefficients)', 
            fontsize=12, fontweight='bold', pad=20)
ax.set_xticks([])
ax.set_yticks([0, 25, 50, 75, 100])
ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=9)
ax.grid(False, axis='x')
ax.grid(True, axis='y', alpha=0.2, linestyle='--', linewidth=0.5)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#CCCCCC')
ax.spines['bottom'].set_visible(False)



#%% Helper function for rounded bars
def draw_rounded_bars(ax, x, heights, colors, alpha=0.7, edgecolor='black', boxstyle='round,pad=0.01'):
    """
    Draw rounded bars on an axis.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw on
    x : array-like
        X positions of bars
    heights : array-like
        Heights of bars
    colors : array-like
        Colors for each bar
    alpha : float
        Transparency
    edgecolor : str
        Edge color
    boxstyle : str
        Box style for rounded corners
    """
    for i, (xi, h, c) in enumerate(zip(x, heights, colors)):
        if h >= 0:
            # Positive bars: start from 0, go up
            y_bottom = 0
            y_top = h
        else:
            # Negative bars: start from 0, go down
            y_bottom = h
            y_top = 0
        
        width = 0.6  # Bar width
        x_left = xi - width / 2
        x_right = xi + width / 2
        
        # Create rounded rectangle
        rect = FancyBboxPatch(
            (x_left, y_bottom), x_right - x_left, y_top - y_bottom,
            boxstyle=boxstyle,
            facecolor=c,
            edgecolor=edgecolor,
            alpha=alpha,
            linewidth=1
        )
        ax.add_patch(rect)

#%% Visualize basis functions and how they reconstruct temporal kernels for all feature types
# Evaluate basis functions on a grid (for temporal event features - causal)
time_event, basis_kernels_event = basis_events.evaluate_on_grid(event_window_bins)
time_event_sec = time_event * EVENT_WINDOW_SEC

# Evaluate basis functions for categorical features (acausal)
time_cat, basis_kernels_cat = basis_categorical.evaluate_on_grid(acausal_window_bins)
time_cat_sec = time_cat * acausal_total_sec - ACAUSAL_BEFORE_SEC  # Shift so 0 is at event

# Evaluate basis functions for spike history
time_hist, basis_kernels_hist = basis_history.evaluate_on_grid(history_window_bins)
time_hist_sec = time_hist * HISTORY_WINDOW_SEC

# Get feature names
temporal_feature_names = list(temporal_features.columns)
categorical_feature_names = list(categorical_features.columns)

# Calculate grid layout - now 2 columns per feature (kernel + coefficients)
n_temporal = len(temporal_feature_names)
n_categorical = len(categorical_feature_names)
n_total_features = n_temporal + n_categorical + 1  # temporal + categorical + spike_history

# Create subplots: first 2 rows for basis functions, then 3 columns per feature (weighted, kernel, coefficients)
fig = plt.figure(figsize=(18, 3 * (n_total_features)))
gs = fig.add_gridspec(n_total_features + 2, 3, hspace=0.4, wspace=0.3)

# Row 0: Causal basis functions for temporal features (spans all columns)
ax_basis_temp = fig.add_subplot(gs[0, :])
for i in range(N_BASIS_FUNCS):
    ax_basis_temp.plot(time_event_sec, basis_kernels_event[:, i], alpha=0.7, linewidth=2, label=f"Basis {i}")
ax_basis_temp.set_xlabel("Time from event (s)")
ax_basis_temp.set_ylabel("Basis function value")
ax_basis_temp.set_title("Temporal Event Basis Functions (Causal - RaisedCosineLog)", fontsize=12, fontweight='bold')
ax_basis_temp.legend(fontsize=8, ncol=4, loc='upper right')
ax_basis_temp.axhline(0, color='k', linewidth=0.5, linestyle='--')
ax_basis_temp.axvline(0, color='red', linewidth=1, linestyle='--', alpha=0.5)
ax_basis_temp.grid(True, alpha=0.3)

# Row 1: Acausal basis functions for categorical features (spans all columns)
ax_basis_cat = fig.add_subplot(gs[1, :])
for i in range(N_BASIS_FUNCS):
    ax_basis_cat.plot(time_cat_sec, basis_kernels_cat[:, i], alpha=0.7, linewidth=2, label=f"Basis {i}")
ax_basis_cat.set_xlabel("Time from event (s)")
ax_basis_cat.set_ylabel("Basis function value")
ax_basis_cat.set_title("Categorical Basis Functions (Acausal - RaisedCosineLinear)", fontsize=12, fontweight='bold')
ax_basis_cat.legend(fontsize=8, ncol=4, loc='upper right')
ax_basis_cat.axhline(0, color='k', linewidth=0.5, linestyle='--')
ax_basis_cat.axvline(0, color='red', linewidth=1, linestyle='--', alpha=0.5, label='Event')
ax_basis_cat.grid(True, alpha=0.3)

row_idx = 2

# Temporal features
for feature_name in temporal_feature_names:
    feature_idx = [i for i, col in enumerate(X.columns) if feature_name in col and 'basis' in col]
    
    if len(feature_idx) == N_BASIS_FUNCS:
        feature_coefs = glm_basis.coef_.flatten()[np.array(feature_idx)]
        reconstructed_kernel = np.dot(basis_kernels_event, feature_coefs)
        
        # Column 1: Weighted basis functions
        ax1 = fig.add_subplot(gs[row_idx, 0])
        for i in range(N_BASIS_FUNCS):
            ax1.plot(time_event_sec, basis_kernels_event[:, i] * feature_coefs[i], 
                    alpha=0.6, linewidth=1.5, label=f"B{i}")
        ax1.set_xlabel("Time (s)", fontsize=9)
        ax1.set_ylabel("Weighted basis", fontsize=9)
        ax1.set_title(f"Temporal: {feature_name}\nWeighted Basis", fontsize=10)
        ax1.axhline(0, color='k', linewidth=0.5, linestyle='--')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=6, ncol=2, loc='best')
        
        # Column 2: Reconstructed kernel with gradient area
        ax2 = fig.add_subplot(gs[row_idx, 1])
        # Create gradient area plot
        n_gradient_layers = 15
        for i in range(n_gradient_layers):
            y_bottom = reconstructed_kernel * (i / n_gradient_layers)
            y_top = reconstructed_kernel * ((i + 1) / n_gradient_layers)
            alpha = 0.4 * ((i + 1) / n_gradient_layers)
            ax2.fill_between(time_event_sec, y_bottom, y_top, alpha=alpha, color='tab:blue', linewidth=0)
        ax2.plot(time_event_sec, reconstructed_kernel, 'b-', linewidth=2, alpha=0.7)
        ax2.set_xlabel("Time (s)", fontsize=9)
        ax2.set_ylabel("Kernel weight", fontsize=9)
        ax2.set_title("Reconstructed Kernel", fontsize=10)
        ax2.axhline(0, color='k', linewidth=0.5, linestyle='--')
        ax2.grid(True, alpha=0.3)
        
        # Column 3: Coefficient bars
        ax3 = fig.add_subplot(gs[row_idx, 2])
        bars = ax3.bar(range(N_BASIS_FUNCS), feature_coefs, 
                       color=['tab:blue' if c >= 0 else 'tab:red' for c in feature_coefs],
                       alpha=0.7, edgecolor='black')
        ax3.set_xlabel("Basis index", fontsize=9)
        ax3.set_ylabel("Coefficient", fontsize=9)
        ax3.set_title("Basis Coefficients", fontsize=10)
        ax3.set_xticks(range(N_BASIS_FUNCS))
        ax3.axhline(0, color='k', linewidth=0.5, linestyle='--')
        ax3.grid(True, alpha=0.3, axis='y')
    
    row_idx += 1

# Categorical features (using acausal basis)
for feature_name in categorical_feature_names:
    feature_idx = [i for i, col in enumerate(X.columns) if feature_name in col and 'basis' in col]
    
    if len(feature_idx) == N_BASIS_FUNCS:
        feature_coefs = glm_basis.coef_.flatten()[np.array(feature_idx)]
        reconstructed_kernel = np.dot(basis_kernels_cat, feature_coefs)
        
        # Shorten name for display
        short_name = feature_name.replace('category_ID[T.', 'cat:').replace('outcome_ID[T.', 'out:').replace('previous_outcome[T.', 'prev:').replace(']', '').replace('stimulus_ID', 'stimulus')
        
        # Column 1: Weighted basis functions
        ax1 = fig.add_subplot(gs[row_idx, 0])
        for i in range(N_BASIS_FUNCS):
            ax1.plot(time_cat_sec, basis_kernels_cat[:, i] * feature_coefs[i], 
                    alpha=0.6, linewidth=1.5, label=f"B{i}")
        ax1.set_xlabel("Time from event (s)", fontsize=9)
        ax1.set_ylabel("Weighted basis", fontsize=9)
        ax1.set_title(f"Categorical: {short_name}\nWeighted Basis (Acausal)", fontsize=10)
        ax1.axhline(0, color='k', linewidth=0.5, linestyle='--')
        ax1.axvline(0, color='red', linewidth=0.5, linestyle='--', alpha=0.5)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=6, ncol=2, loc='best')
        
        # Column 2: Reconstructed kernel with gradient area
        ax2 = fig.add_subplot(gs[row_idx, 1])
        # Create gradient area plot
        n_gradient_layers = 15
        for i in range(n_gradient_layers):
            y_bottom = reconstructed_kernel * (i / n_gradient_layers)
            y_top = reconstructed_kernel * ((i + 1) / n_gradient_layers)
            alpha = 0.4 * ((i + 1) / n_gradient_layers)
            ax2.fill_between(time_cat_sec, y_bottom, y_top, alpha=alpha, color='tab:green', linewidth=0)
        ax2.plot(time_cat_sec, reconstructed_kernel, 'g-', linewidth=2, alpha=0.7)
        ax2.set_xlabel("Time from event (s)", fontsize=9)
        ax2.set_ylabel("Kernel weight", fontsize=9)
        ax2.set_title("Reconstructed Kernel (Acausal)", fontsize=10)
        ax2.axhline(0, color='k', linewidth=0.5, linestyle='--')
        ax2.axvline(0, color='red', linewidth=0.5, linestyle='--', alpha=0.5)
        ax2.grid(True, alpha=0.3)
        
        # Column 3: Coefficient bars
        ax3 = fig.add_subplot(gs[row_idx, 2])
        bars = ax3.bar(range(N_BASIS_FUNCS), feature_coefs, 
                       color=['tab:green' if c >= 0 else 'tab:orange' for c in feature_coefs],
                       alpha=0.7, edgecolor='black')
        ax3.set_xlabel("Basis index", fontsize=9)
        ax3.set_ylabel("Coefficient", fontsize=9)
        ax3.set_title("Basis Coefficients", fontsize=10)
        ax3.set_xticks(range(N_BASIS_FUNCS))
        ax3.axhline(0, color='k', linewidth=0.5, linestyle='--')
        ax3.grid(True, alpha=0.3, axis='y')
    
    row_idx += 1

# Spike history
hist_feature_idx = [i for i, col in enumerate(X.columns) if 'spike_history' in col and 'basis' in col]

if len(hist_feature_idx) == basis_history.n_basis_funcs:
    hist_coefs = glm_basis.coef_.flatten()[np.array(hist_feature_idx)]
    reconstructed_hist = np.dot(basis_kernels_hist, hist_coefs)
    
    # Column 1: Weighted basis functions
    ax1 = fig.add_subplot(gs[row_idx, 0])
    for i in range(basis_history.n_basis_funcs):
        ax1.plot(time_hist_sec, basis_kernels_hist[:, i] * hist_coefs[i], 
                alpha=0.6, linewidth=1.5, label=f"B{i}")
    ax1.set_xlabel("Time (s)", fontsize=9)
    ax1.set_ylabel("Weighted basis", fontsize=9)
    ax1.set_title("Spike History\nWeighted Basis", fontsize=10)
    ax1.axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=6, ncol=2, loc='best')
    
    # Column 2: Reconstructed kernel with gradient area
    ax2 = fig.add_subplot(gs[row_idx, 1])
    # Create gradient area plot
    n_gradient_layers = 15
    for i in range(n_gradient_layers):
        y_bottom = reconstructed_hist * (i / n_gradient_layers)
        y_top = reconstructed_hist * ((i + 1) / n_gradient_layers)
        alpha = 0.4 * ((i + 1) / n_gradient_layers)
        ax2.fill_between(time_hist_sec, y_bottom, y_top, alpha=alpha, color='tab:red', linewidth=0)
    ax2.plot(time_hist_sec, reconstructed_hist, 'r-', linewidth=2, alpha=0.7)
    ax2.set_xlabel("Time from spike (s)", fontsize=9)
    ax2.set_ylabel("Kernel weight", fontsize=9)
    ax2.set_title("Reconstructed Kernel", fontsize=10)
    ax2.axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax2.grid(True, alpha=0.3)
    
    # Column 3: Coefficient bars
    ax3 = fig.add_subplot(gs[row_idx, 2])
    bars = ax3.bar(range(basis_history.n_basis_funcs), hist_coefs, 
                   color=['tab:red' if c >= 0 else 'tab:purple' for c in hist_coefs],
                   alpha=0.7, edgecolor='black')
    ax3.set_xlabel("Basis index", fontsize=9)
    ax3.set_ylabel("Coefficient", fontsize=9)
    ax3.set_title("Basis Coefficients", fontsize=10)
    ax3.set_xticks(range(basis_history.n_basis_funcs))
    ax3.axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax3.grid(True, alpha=0.3, axis='y')

plt.suptitle("GLM Temporal Kernels: Causal (Temporal) + Acausal (Categorical) + Spike History", 
            fontsize=14, fontweight='bold', fontfamily='sans-serif')



plt.tight_layout()

#%% Create population design matrix with optional per-neuron spike history (self-coupling)

# Get shared predictors (temporal + categorical) without spike history
non_history_indices = [i for i, col in enumerate(X.columns) if 'spike_history' not in col]
X_shared = nap.TsdFrame(
    t=X.t,
    d=X.values[:, non_history_indices],
    columns=[col for col in X.columns if 'spike_history' not in col]
)
X_shared_ep = X_shared.restrict(epochs)

print(f"\nBuilding population design matrix:")
print(f"  Shared predictors (temporal + categorical): {X_shared_ep.shape[1]} features")
print(f"  Include spike history: {INCLUDE_SPIKE_HISTORY}")

# Create POPULATION target matrix (multiple neurons) for PopulationGLM
# PopulationGLM requires y to be 2D: (n_timebins, n_neurons)
# Select neurons for population analysis
N_POPULATION = min(N_POPULATION, len(spikes))
population_ids = np.random.choice(list(spikes.keys()), N_POPULATION, replace=False)

# Build population spike count matrix and per-neuron spike history features
spikes_population = spikes[population_ids]

spike_count_population = spikes_population.count(BIN_SIZE, ep=epochs)

spike_count_population = nap.TsdFrame(
    t=spike_count_population.t,
    d=spike_count_population.values[:, np.argsort(population_ids)],
    columns=population_ids,
)

# Create spike history features for EACH neuron (ALL-TO-ALL connectivity) - optional
# ALL-TO-ALL: Each neuron's spike history features are included in the design matrix.
# PopulationGLM learns separate coefficients for how each neuron's history affects each target neuron.
# This enables cross-neuron coupling: neuron i's history can affect neuron j's firing.
if INCLUDE_SPIKE_HISTORY:
    print(f"  Creating ALL-TO-ALL spike history connectivity for {len(population_ids)} neurons...")
    print(f"    Each neuron's history can affect all neurons (cross-coupling enabled)")
    
    # Create acausal basis for spike history in population GLM
    history_acausal_total_sec = HISTORY_ACAUSAL_BEFORE_SEC + HISTORY_ACAUSAL_AFTER_SEC
    history_acausal_window_bins = int(history_acausal_total_sec * spike_count_population.rate)
    
    basis_history_pop = nmo.basis.RaisedCosineLinearConv(
        n_basis_funcs=N_BASIS_FUNCS,
        window_size=history_acausal_window_bins,
        label="spike_history_acausal"
    )
    
    print(f"  Acausal spike history basis: {N_BASIS_FUNCS} functions over {history_acausal_total_sec}s window ({-HISTORY_ACAUSAL_BEFORE_SEC}s to +{HISTORY_ACAUSAL_AFTER_SEC}s)")
    
    history_features_list = []

    for uid in population_ids:
        # Get this neuron's spike count from full matrix
        uid_idx = list(spike_count.columns).index(uid)
        neuron_spikes = spike_count[:, uid_idx]
        
        # Compute spike history features for this neuron using acausal basis
        neuron_history = basis_history_pop.compute_features(neuron_spikes)
        neuron_history_ep = neuron_history.restrict(epochs)
        
        # Align to X_shared_ep timestamps
        if neuron_history_ep.shape[0] != X_shared_ep.shape[0]:
            hist_indices = np.searchsorted(neuron_history_ep.t, X_shared_ep.t)
            hist_indices = np.clip(hist_indices, 0, len(neuron_history_ep.t) - 1)
            neuron_history_aligned = neuron_history_ep.values[hist_indices]
        else:
            neuron_history_aligned = neuron_history_ep.values
        
        # Label columns with source neuron ID (this history can affect all target neurons)
        history_cols = [f"hist_from_neuron{uid}_basis{i}" for i in range(basis_history_pop.n_basis_funcs)]
        history_features_list.append((neuron_history_aligned, history_cols))

    # Combine all history features from all source neurons
    # Structure: [neuron0_history | neuron1_history | ... | neuronN_history]
    # Each history can affect all target neurons via PopulationGLM coefficients
    all_history_features = np.column_stack([hist for hist, _ in history_features_list])
    all_history_cols = [col for _, cols in history_features_list for col in cols]

    print(f"  All-to-all connectivity: {len(population_ids)} source neurons × {basis_history_pop.n_basis_funcs} basis = {all_history_features.shape[1]} history features")
    print(f"    Total possible connections: {len(population_ids)}² = {len(population_ids)**2} neuron pairs")
    
    # Combine shared predictors + all neuron-specific history features
    X_pop_combined = np.column_stack([X_shared_ep.values, all_history_features])
    X_pop_cols = list(X_shared_ep.columns) + all_history_cols
    n_history_features = all_history_features.shape[1]
else:
    print(f"  Skipping spike history features")
    # Use only shared predictors
    X_pop_combined = X_shared_ep.values
    X_pop_cols = list(X_shared_ep.columns)
    n_history_features = 0

# Remove NaN rows
valid_mask = np.all(np.isfinite(X_pop_combined), axis=1)
X_ep_pop = nap.TsdFrame(
    t=X_shared_ep.t[valid_mask],
    d=X_pop_combined[valid_mask],
    columns=X_pop_cols
)

print(f"  Total features: {X_ep_pop.shape[1]} (shared: {X_shared_ep.shape[1]}, history: {n_history_features})")

# Align population spike counts to X_ep_pop timestamps
if spike_count_population.shape[0] != X_ep_pop.shape[0]:
    pop_indices = np.searchsorted(spike_count_population.t, X_ep_pop.t)
    pop_indices = np.clip(pop_indices, 0, len(spike_count_population.t) - 1)
    y_ep_pop = nap.TsdFrame(
        t=X_ep_pop.t,
        d=spike_count_population.values[pop_indices],
        columns=spike_count_population.columns
    )
else:
    y_ep_pop = spike_count_population.restrict(X_ep_pop.time_support)

# Get region info for each neuron
population_regions = spikes_population.get_info('region').values

print(f"y_ep_pop shape: {y_ep_pop.shape}")


#%% Fit Population GLM
model_pop = nmo.glm.PopulationGLM(
    solver_name="LBFGS",
    regularizer="Ridge",
    regularizer_strength=0.1
)

print("\nFitting Population GLM...")
model_pop.fit(X_ep_pop, y_ep_pop)
#%%
# Score the model
pop_score = model_pop.score(X_ep_pop, y_ep_pop, score_type='pseudo-r2-Cohen')
print(f"Population GLM Cohen's pseudo-R2: {pop_score:.4f}")

#%% Compute per-neuron scores from predictions
# Get predictions for all neurons
y_pred_pop = model_pop.predict(X_ep_pop)  # Shape: (n_timebins, n_neurons)
y_actual_pop = y_ep_pop.values

# Compute per-neuron pseudo-R² (Cohen) manually
print("\nPer-neuron pseudo-R²:")
per_neuron_scores = []

# Convert predictions to numpy array (in case they're JAX arrays)
y_pred_np = np.asarray(y_pred_pop)
y_actual_np = np.asarray(y_actual_pop)
per_unit_scores = {}
unit_regions = {}
for i, uid in enumerate(population_ids):
    y_pred_i = y_pred_np[:, i]
    y_actual_i = y_actual_np[:, i]
    
    # Compute pseudo-R² (Cohen): 1 - (deviance_model / deviance_null)
    y_mean = np.mean(y_actual_i)
    eps = 1e-10  # Avoid log(0)
    
    # Model deviance (Poisson)
    dev_model = 2 * np.sum(
        y_actual_i * np.log((y_actual_i + eps) / (y_pred_i + eps)) - (y_actual_i - y_pred_i)
    )
    
    # Null deviance
    dev_null = 2 * np.sum(
        y_actual_i * np.log((y_actual_i + eps) / (y_mean + eps)) - (y_actual_i - y_mean)
    )
    
    pseudo_r2 = 1 - (dev_model / dev_null) if dev_null > 0 else 0
    per_unit_scores[uid] = pseudo_r2
    unit_regions[uid] = population_regions[i]
    
    print(f"  Neuron {uid} ({unit_regions[uid]}): {pseudo_r2:.4f}")

# Convert to firing rates for plotting (use numpy arrays)
predicted_firing_rate = y_pred_np / BIN_SIZE
actual_firing_rate = y_actual_np / BIN_SIZE


#%% Plot actual vs predicted firing rate for each neuron with event markers
# Create boolean mask for EXAMPLE_EPOCH
# Select top 3 from ACx and top 3 from OFC
acx_ids = [uid for uid in population_ids if population_regions[list(population_ids).index(uid)] == 'ACx']
ofc_ids = [uid for uid in population_ids if population_regions[list(population_ids).index(uid)] == 'OFC']

# Sort by score and take top 3 from each region
acx_sorted = sorted(acx_ids, key=lambda x: per_unit_scores[x], reverse=True)[:3]
ofc_sorted = sorted(ofc_ids, key=lambda x: per_unit_scores[x], reverse=True)[:3]

# Combine: ACx first, then OFC
population_ids_sorted = acx_sorted + ofc_sorted
UNIT_TO_PLOT = len(population_ids_sorted)

plot_mask = (X_ep_pop.t >= EXAMPLE_EPOCH.start[0]) & (X_ep_pop.t <= EXAMPLE_EPOCH.end[0])
t_plot_raw = X_ep_pop.t[plot_mask]

# Get event times that fall within the plot window
tone_times_plot = tone_onset[(tone_onset >= EXAMPLE_EPOCH.start[0]) & (tone_onset <= EXAMPLE_EPOCH.end[0])]
lick_times_plot = licks[(licks >= EXAMPLE_EPOCH.start[0]) & (licks <= EXAMPLE_EPOCH.end[0])]
outcome_times_plot = outcome_time[(outcome_time >= EXAMPLE_EPOCH.start[0]) & (outcome_time <= EXAMPLE_EPOCH.end[0])]

# Create regular time grid at BIN_SIZE resolution for filled area plot
t_start = EXAMPLE_EPOCH.start[0]
t_end = EXAMPLE_EPOCH.end[0]
t_full = np.arange(t_start, t_end + BIN_SIZE, BIN_SIZE)

fig, axes = plt.subplots(UNIT_TO_PLOT, 1, figsize=(12, 3*UNIT_TO_PLOT), sharex=True)
if UNIT_TO_PLOT == 1:
    axes = [axes]

for plot_idx, uid in enumerate(population_ids_sorted[:UNIT_TO_PLOT]):
    ax = axes[plot_idx]
    
    # Find original index of this neuron in population_ids
    original_idx = list(population_ids).index(uid)
    
    # Get firing rates (restricted to epoch)
    actual_fr_raw = actual_firing_rate[plot_mask, original_idx]
    pred_fr_raw = predicted_firing_rate[plot_mask, original_idx]
    
    # Smooth firing rates using pynapple
    actual_tsd = nap.Tsd(t=t_plot_raw, d=actual_fr_raw)
    pred_tsd = nap.Tsd(t=t_plot_raw, d=pred_fr_raw)
    actual_smooth = actual_tsd.smooth(std=0.05, windowsize=0.25)
    pred_smooth = pred_tsd.smooth(std=0.05, windowsize=0.25)
    
    # Fill in smoothed values by finding nearest neighbors
    actual_filled = np.zeros_like(t_full, dtype=float)
    pred_filled = np.zeros_like(t_full, dtype=float)
    
    actual_times = actual_smooth.t
    actual_values = actual_smooth.d
    pred_times = pred_smooth.t
    pred_values = pred_smooth.d
    
    for i, t_val in enumerate(t_full):
        # Find nearest time point for actual
        dists_actual = np.abs(actual_times - t_val)
        if len(dists_actual) > 0 and np.min(dists_actual) < BIN_SIZE:
            actual_filled[i] = actual_values[np.argmin(dists_actual)]
        
        # Find nearest time point for predicted
        dists_pred = np.abs(pred_times - t_val)
        if len(dists_pred) > 0 and np.min(dists_pred) < BIN_SIZE:
            pred_filled[i] = pred_values[np.argmin(dists_pred)]
    
    # Create gradient effect for area charts using multiple overlapping fills
    n_gradient_layers = 15
    
    # Orange gradient for actual firing rate - darker at top, lighter at bottom
    for i in range(n_gradient_layers):
        y_bottom = actual_filled * (i / n_gradient_layers)
        y_top = actual_filled * ((i + 1) / n_gradient_layers)
        alpha = 0.4 * ((i + 1) / n_gradient_layers)  # Increasing opacity toward top
        ax.fill_between(t_full, y_bottom, y_top, alpha=alpha, color='orange', linewidth=0)
    
    # Green gradient for predicted firing rate - darker at top, lighter at bottom
    for i in range(n_gradient_layers):
        y_bottom = pred_filled * (i / n_gradient_layers)
        y_top = pred_filled * ((i + 1) / n_gradient_layers)
        alpha = 0.4 * ((i + 1) / n_gradient_layers)  # Increasing opacity toward top
        ax.fill_between(t_full, y_bottom, y_top, alpha=alpha, color='green', linewidth=0)
    
    # Add semi-transparent outlines for better visibility
    ax.plot(t_full, actual_filled, color='orange', linewidth=2, alpha=0.7, label='Actual FR')
    ax.plot(t_full, pred_filled, color='green', linewidth=2, alpha=0.7, label='Predicted FR')
    
    # Add event markers
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    
    # Tone onsets (red vertical lines)
    for tone_t in tone_times_plot:
        ax.axvline(tone_t, color='red', alpha=0.3, linewidth=1, linestyle='--', zorder=5)
    
    # Licks (black circle markers at bottom of plot)
    if len(lick_times_plot) > 0:
        lick_y_position = y_min + 0.05 * y_range  # 5% from bottom
        ax.scatter(lick_times_plot, 
                  np.ones(len(lick_times_plot)) * lick_y_position,
                  marker='o', s=30, color='black', alpha=0.6, 
                  edgecolors='black', linewidths=0.5, zorder=10)
    
    # Outcomes (green vertical lines)
    for outcome_t in outcome_times_plot:
        if outcome_t > 0:  # Skip invalid outcome times
            ax.axvline(outcome_t, color='green', alpha=0.3, linewidth=1, linestyle='--', zorder=5)
    
    ax.set_ylabel('Firing Rate (Hz)')
    ax.set_title(f'Neuron {uid} ({population_regions[original_idx]}) - Pseudo-R2: {per_unit_scores[uid]:.4f}')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Add legend for events (only on first subplot)
    if plot_idx == 0:
        from matplotlib.lines import Line2D
        event_legend = [
            Line2D([0], [0], color='red', linewidth=1, linestyle='--', alpha=0.5, label='Tone onset'),
            Line2D([0], [0], marker='o', color='black', markersize=6, linestyle='', 
                   markeredgecolor='black', markeredgewidth=0.5, alpha=0.6, label='Licks'),
            Line2D([0], [0], color='green', linewidth=1, linestyle='--', alpha=0.5, label='Outcome')
        ]
        ax.legend(handles=ax.get_legend_handles_labels()[0] + event_legend, 
                 loc='upper right', fontsize=8, ncol=2)

axes[-1].set_xlabel('Time (s)')
plt.suptitle('Population GLM: Actual vs Predicted Firing Rates with Events', fontsize=14, fontweight='bold')
plt.tight_layout()


#%% Analyze all-to-all connectivity structure (if spike history is included)
if INCLUDE_SPIKE_HISTORY:
    print(f"\n{'='*60}")
    print("ALL-TO-ALL CONNECTIVITY ANALYSIS")
    print(f"{'='*60}")
    
    # Extract coefficients for spike history features
    coef_matrix = np.asarray(model_pop.coef_)  # Shape: (n_features, n_neurons)
    
    # Find history feature indices
    history_feature_indices = [i for i, col in enumerate(X_ep_pop.columns) if 'hist_from_neuron' in col]
    
    if len(history_feature_indices) > 0:
        # Extract history coefficients: (n_history_features, n_target_neurons)
        history_coefs = coef_matrix[history_feature_indices, :]
        
        # Reshape to (n_source_neurons, n_basis, n_target_neurons)
        n_source = len(population_ids)
        n_basis = basis_history_pop.n_basis_funcs
        n_target = len(population_ids)
        
        history_coefs_reshaped = history_coefs.reshape(n_source, n_basis, n_target)
        
        # Reconstruct kernels for each connection and compute normalized coupling strength
        # This accounts for both coefficients and basis function shapes
        # Use L2 norm of reconstructed kernel as coupling strength (normalized metric)
        # This is more interpretable than summing raw coefficients
        # Shape: (n_source_neurons, n_target_neurons)
        coupling_strength = np.zeros((n_source, n_target))
        coupling_signed = np.zeros((n_source, n_target))
        
        # Compute basis kernels for spike history (using acausal basis for population GLM)
        # Use the acausal window size as when basis_history_pop was created
        history_acausal_total_sec_pop = HISTORY_ACAUSAL_BEFORE_SEC + HISTORY_ACAUSAL_AFTER_SEC
        # Use the rate from the binned data (1/BIN_SIZE)
        history_acausal_window_bins = int(history_acausal_total_sec_pop / BIN_SIZE)
        time_hist, basis_kernels_hist = basis_history_pop.evaluate_on_grid(history_acausal_window_bins)
        # Shift time axis so 0 is at the spike (acausal: negative before, positive after)
        time_hist_sec = time_hist * history_acausal_total_sec_pop - HISTORY_ACAUSAL_BEFORE_SEC
        
        for source_idx in range(n_source):
            for target_idx in range(n_target):
                # Get coefficients for this connection
                coefs = history_coefs_reshaped[source_idx, :, target_idx]
                # Reconstruct kernel using acausal basis
                reconstructed_kernel = np.dot(basis_kernels_hist, coefs)
                # Compute coupling strength as L2 norm of reconstructed kernel (magnitude)
                coupling_strength[source_idx, target_idx] = np.linalg.norm(reconstructed_kernel)
                # Compute signed coupling as sum of reconstructed kernel (net effect: +excitatory, -inhibitory)
                coupling_signed[source_idx, target_idx] = np.sum(reconstructed_kernel)
        
        # Normalize signed coupling to -1 to 1 range to preserve excitation/inhibition
        # Divide by maximum absolute value to scale to [-1, 1]
        max_abs_coupling = np.max(np.abs(coupling_signed))
        coupling_normalized = coupling_signed / (max_abs_coupling + 1e-10)
        
        print(f"Coupling matrix shape: {coupling_strength.shape} (source × target)")
        print(f"Signed coupling: Sum of reconstructed kernels (preserves excitation/inhibition)")
        print(f"  Raw - Mean: {np.mean(coupling_signed):.4f}, Max: {np.max(coupling_signed):.4f}, Min: {np.min(coupling_signed):.4f}")
        print(f"  Normalized (-1 to 1) - Mean: {np.mean(coupling_normalized):.4f}, Max: {np.max(coupling_normalized):.4f}, Min: {np.min(coupling_normalized):.4f}")
        
        # Use normalized signed coupling for visualization and analysis
        # Positive values = excitatory, negative values = inhibitory
        # For visualization, use signed values to show excitation/inhibition
        coupling_strength = coupling_normalized  # Signed normalized coupling (-1 to 1)
        coupling_signed = coupling_normalized  # Same as coupling_strength (signed)
        
        # Separate neurons by region
        acx_indices = [i for i, r in enumerate(population_regions) if r == 'ACx']
        ofc_indices = [i for i, r in enumerate(population_regions) if r == 'OFC']
        
        # Sort within each region by total outgoing coupling strength (sum of absolute values)
        # This orders units by their overall influence on other units
        def sort_by_coupling_strength(indices):
            if len(indices) == 0:
                return indices
            # Compute total outgoing coupling strength for each unit
            outgoing_strength = np.sum(np.abs(coupling_strength[np.ix_(indices, range(n_target))]), axis=1)
            # Sort indices by descending coupling strength
            sorted_idx = np.argsort(outgoing_strength)[::-1]
            return [indices[i] for i in sorted_idx]
        
        acx_indices_sorted = sort_by_coupling_strength(acx_indices)
        ofc_indices_sorted = sort_by_coupling_strength(ofc_indices)
        
        print(f"\nRegion breakdown:")
        print(f"  ACx neurons: {len(acx_indices_sorted)} (sorted by coupling strength)")
        print(f"  OFC neurons: {len(ofc_indices_sorted)} (sorted by coupling strength)")
        
        # Use sorted indices for all subsequent analysis
        acx_indices = acx_indices_sorted
        ofc_indices = ofc_indices_sorted
        
        # Extract submatrices for different connection types
        # ACx → ACx
        acx_to_acx = coupling_strength[np.ix_(acx_indices, acx_indices)] if len(acx_indices) > 0 else np.array([])
        # ACx → OFC
        acx_to_ofc = coupling_strength[np.ix_(acx_indices, ofc_indices)] if len(acx_indices) > 0 and len(ofc_indices) > 0 else np.array([])
        # OFC → ACx
        ofc_to_acx = coupling_strength[np.ix_(ofc_indices, acx_indices)] if len(ofc_indices) > 0 and len(acx_indices) > 0 else np.array([])
        # OFC → OFC
        ofc_to_ofc = coupling_strength[np.ix_(ofc_indices, ofc_indices)] if len(ofc_indices) > 0 else np.array([])
        
        # Same for signed coupling
        acx_to_acx_signed = coupling_signed[np.ix_(acx_indices, acx_indices)] if len(acx_indices) > 0 else np.array([])
        acx_to_ofc_signed = coupling_signed[np.ix_(acx_indices, ofc_indices)] if len(acx_indices) > 0 and len(ofc_indices) > 0 else np.array([])
        ofc_to_acx_signed = coupling_signed[np.ix_(ofc_indices, acx_indices)] if len(ofc_indices) > 0 and len(acx_indices) > 0 else np.array([])
        ofc_to_ofc_signed = coupling_signed[np.ix_(ofc_indices, ofc_indices)] if len(ofc_indices) > 0 else np.array([])
        
        # Print statistics by connection type
        print(f"\nConnection statistics:")
        if acx_to_acx.size > 0:
            acx_acx_self = np.diag(acx_to_acx)
            acx_acx_cross = acx_to_acx.copy()
            np.fill_diagonal(acx_acx_cross, 0)
            acx_acx_cross = acx_acx_cross[acx_acx_cross != 0]
            acx_acx_cross_mean = np.mean(acx_acx_cross) if len(acx_acx_cross) > 0 else 0.0
            print(f"  ACx → ACx: mean={np.mean(acx_to_acx):.4f}, self={np.mean(acx_acx_self):.4f}, cross={acx_acx_cross_mean:.4f}")
        
        if acx_to_ofc.size > 0:
            print(f"  ACx → OFC: mean={np.mean(acx_to_ofc):.4f}, max={np.max(acx_to_ofc):.4f}")
        
        if ofc_to_acx.size > 0:
            print(f"  OFC → ACx: mean={np.mean(ofc_to_acx):.4f}, max={np.max(ofc_to_acx):.4f}")
        
        if ofc_to_ofc.size > 0:
            ofc_ofc_self = np.diag(ofc_to_ofc)
            ofc_ofc_cross = ofc_to_ofc.copy()
            np.fill_diagonal(ofc_ofc_cross, 0)
            ofc_ofc_cross = ofc_ofc_cross[ofc_ofc_cross != 0]
            ofc_ofc_cross_mean = np.mean(ofc_ofc_cross) if len(ofc_ofc_cross) > 0 else 0.0
            print(f"  OFC → OFC: mean={np.mean(ofc_to_ofc):.4f}, self={np.mean(ofc_ofc_self):.4f}, cross={ofc_ofc_cross_mean:.4f}")
        
        # Reorder neurons by region (already sorted by coupling strength within each region)
        region_order = acx_indices + ofc_indices
        
        # Figure 1: Full connectivity matrix (square figure)
        fig1 = plt.figure(figsize=(8, 8))
        ax_full = fig1.add_subplot(1, 1, 1)
        coupling_ordered = coupling_strength[np.ix_(region_order, region_order)]
        # Use diverging colormap to show excitation (red/positive) and inhibition (blue/negative)
        # Set aspect ratio to match matrix dimensions (square matrix = square plot)
        aspect_ratio = coupling_ordered.shape[0] / coupling_ordered.shape[1] if coupling_ordered.shape[1] > 0 else 1
        im = ax_full.imshow(coupling_ordered, aspect=aspect_ratio, cmap='RdBu_r', vmin=-1, vmax=1)
        ax_full.set_xlabel('Target Neuron', fontsize=11)
        ax_full.set_ylabel('Source Neuron', fontsize=11)
        ax_full.set_title('Full Connectivity Matrix (Grouped by Region, Sorted by Coupling Strength)', 
                         fontsize=12, fontweight='bold')
        
        # Add region boundaries
        if len(acx_indices) > 0 and len(ofc_indices) > 0:
            ax_full.axvline(len(acx_indices) - 0.5, color='black', linewidth=2, linestyle='--')
            ax_full.axhline(len(acx_indices) - 0.5, color='black', linewidth=2, linestyle='--')
            ax_full.text(len(acx_indices)/2, -1, 'ACx', ha='center', va='top', fontsize=10, fontweight='bold', color='black')
            ax_full.text(len(acx_indices) + len(ofc_indices)/2, -1, 'OFC', ha='center', va='top', fontsize=10, fontweight='bold', color='black')
            ax_full.text(-1, len(acx_indices)/2, 'ACx', ha='right', va='center', fontsize=10, fontweight='bold', color='black', rotation=90)
            ax_full.text(-1, len(acx_indices) + len(ofc_indices)/2, 'OFC', ha='right', va='center', fontsize=10, fontweight='bold', color='black', rotation=90)
        
        # Make the colorbar half the size and center it beneath the matrix
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax_full)
        # cax_width is half the width of the plot, position below center
        cax = divider.append_axes("bottom", size="1%", pad=0.45)  # "5%" is bar thickness, pad <0.5 centers better
        cb = plt.colorbar(im, cax=cax, orientation='horizontal')
        cb.set_label('Coupling Strength (Normalized -1 to 1)')
        # Make the colorbar half the width of the axes and centered
        box = ax_full.get_position()
        cbox = cax.get_position()
        new_width = (box.width) * 0.5
        cax.set_position([
            box.x0 + (box.width - new_width) / 2,
            cbox.y0,
            new_width,
            cbox.height
        ])
        plt.suptitle('All-to-All Connectivity Matrix', fontsize=14, fontweight='bold', 
                    fontfamily='sans-serif', y=0.98)
        plt.tight_layout()
        
        # Figure 2: Breakdown by connection type
        fig2 = plt.figure(figsize=(16, 8))
        gs = fig2.add_gridspec(2, 4, hspace=0.3, wspace=0.3, height_ratios=[3, 1])
        
        # ACx → ACx
        if acx_to_acx.size > 0:
            ax = fig2.add_subplot(gs[0, 0])
            # Set aspect ratio to square (1:1)
            im = ax.imshow(acx_to_acx, aspect=1, cmap='RdBu_r', vmin=-1, vmax=1)
            ax.set_title('ACx → ACx', fontweight='bold')
            ax.set_xlabel('Target ACx')
            ax.set_ylabel('Source ACx')
            ax.plot([-0.5, len(acx_indices)-0.5], [-0.5, len(acx_indices)-0.5], 'r--', linewidth=1, alpha=0.5)
        
        # ACx → OFC
        if acx_to_ofc.size > 0:
            ax = fig2.add_subplot(gs[0, 1])
            # Set aspect ratio to square (1:1)
            im = ax.imshow(acx_to_ofc, aspect=1, cmap='RdBu_r', vmin=-1, vmax=1)
            ax.set_title('ACx → OFC', fontweight='bold')
            ax.set_xlabel('Target OFC')
            ax.set_ylabel('Source ACx')
        
        # OFC → ACx
        if ofc_to_acx.size > 0:
            ax = fig2.add_subplot(gs[0, 2])
            # Set aspect ratio to square (1:1)
            im = ax.imshow(ofc_to_acx, aspect=1, cmap='RdBu_r', vmin=-1, vmax=1)
            ax.set_title('OFC → ACx', fontweight='bold')
            ax.set_xlabel('Target ACx')
            ax.set_ylabel('Source OFC')
        
        # OFC → OFC
        if ofc_to_ofc.size > 0:
            ax = fig2.add_subplot(gs[0, 3])
            # Set aspect ratio to square (1:1)
            im = ax.imshow(ofc_to_ofc, aspect=1, cmap='RdBu_r', vmin=-1, vmax=1)
            ax.set_title('OFC → OFC', fontweight='bold')
            ax.set_xlabel('Target OFC')
            ax.set_ylabel('Source OFC')
            plt.colorbar(im, ax=ax)
            ax.plot([-0.5, len(ofc_indices)-0.5], [-0.5, len(ofc_indices)-0.5], 'r--', linewidth=1, alpha=0.5)
        
        # Comparison: Within-region vs cross-region (spans full width of first row)
        ax = fig2.add_subplot(gs[1, :])
        within_region = []
        cross_region = []
        
        if acx_to_acx.size > 0:
            acx_acx_flat = acx_to_acx.flatten()
            within_region.extend(acx_acx_flat)
        if ofc_to_ofc.size > 0:
            ofc_ofc_flat = ofc_to_ofc.flatten()
            within_region.extend(ofc_ofc_flat)
        if acx_to_ofc.size > 0:
            acx_ofc_flat = acx_to_ofc.flatten()
            cross_region.extend(acx_ofc_flat)
        if ofc_to_acx.size > 0:
            ofc_acx_flat = ofc_to_acx.flatten()
            cross_region.extend(ofc_acx_flat)
        
        # Plot separate histograms for within-region and cross-region
        if len(within_region) > 0 or len(cross_region) > 0:
            # Determine common bin range for both distributions
            all_values = within_region + cross_region if len(within_region) > 0 and len(cross_region) > 0 else (within_region if len(within_region) > 0 else cross_region)
            bins = 50
            bin_range = (np.min(all_values), np.max(all_values))
            
            # Plot within-region distribution
            if len(within_region) > 0:
                within_counts, within_bins = np.histogram(within_region, bins=bins, range=bin_range, density=True)
                within_centers = (within_bins[:-1] + within_bins[1:]) / 2
                
                # Create gradient area fills for within-region
                n_gradient_layers = 15
                for i in range(n_gradient_layers):
                    y_bottom = within_counts * (i / n_gradient_layers)
                    y_top = within_counts * ((i + 1) / n_gradient_layers)
                    alpha = 0.4 * ((i + 1) / n_gradient_layers)
                    ax.fill_between(within_centers, y_bottom, y_top, alpha=alpha, color='tab:blue', linewidth=0)
                
                # Add highlighted line on top
                ax.plot(within_centers, within_counts, color='tab:blue', linewidth=2, alpha=0.8, label='Within Region')
            
            # Plot cross-region distribution
            if len(cross_region) > 0:
                cross_counts, cross_bins = np.histogram(cross_region, bins=bins, range=bin_range, density=True)
                cross_centers = (cross_bins[:-1] + cross_bins[1:]) / 2
                
                # Create gradient area fills for cross-region
                n_gradient_layers = 15
                for i in range(n_gradient_layers):
                    y_bottom = cross_counts * (i / n_gradient_layers)
                    y_top = cross_counts * ((i + 1) / n_gradient_layers)
                    alpha = 0.4 * ((i + 1) / n_gradient_layers)
                    ax.fill_between(cross_centers, y_bottom, y_top, alpha=alpha, color='tab:orange', linewidth=0)
                
                # Add highlighted line on top
                ax.plot(cross_centers, cross_counts, color='tab:orange', linewidth=2, alpha=0.8, label='Cross Region')
            
            ax.set_xlabel('Coupling Strength')
            ax.set_ylabel('Density')
            ax.set_title('Coupling Strength Distribution', fontweight='bold')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            
            # Add statistics text for both distributions
            stats_text = []
            if len(within_region) > 0:
                stats_text.append(f'Within: μ={np.mean(within_region):.4f}, σ={np.std(within_region):.4f}')
            if len(cross_region) > 0:
                stats_text.append(f'Cross: μ={np.mean(cross_region):.4f}, σ={np.std(cross_region):.4f}')
            
            if stats_text:
                ax.text(0.05, 0.95, '\n'.join(stats_text), 
                       transform=ax.transAxes, verticalalignment='top', 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('Connectivity Breakdown by Region', fontsize=14, fontweight='bold', 
                    fontfamily='sans-serif', y=0.98)
        plt.tight_layout()
        
        # Print top connections by type
        print(f"\nTop connections by type:")
        best_connections = {}
        if acx_to_acx.size > 0:
            top_acx_acx = np.unravel_index(np.argmax(acx_to_acx), acx_to_acx.shape)
            source_id = population_ids[acx_indices[top_acx_acx[0]]]
            target_id = population_ids[acx_indices[top_acx_acx[1]]]
            best_connections['ACx→ACx'] = (source_id, target_id, acx_to_acx[top_acx_acx])
            print(f"  ACx→ACx: Neuron {source_id} → {target_id}: {acx_to_acx[top_acx_acx]:.4f}")
        if acx_to_ofc.size > 0:
            top_acx_ofc = np.unravel_index(np.argmax(acx_to_ofc), acx_to_ofc.shape)
            source_id = population_ids[acx_indices[top_acx_ofc[0]]]
            target_id = population_ids[ofc_indices[top_acx_ofc[1]]]
            best_connections['ACx→OFC'] = (source_id, target_id, acx_to_ofc[top_acx_ofc])
            print(f"  ACx→OFC: Neuron {source_id} → {target_id}: {acx_to_ofc[top_acx_ofc]:.4f}")
        if ofc_to_acx.size > 0:
            top_ofc_acx = np.unravel_index(np.argmax(ofc_to_acx), ofc_to_acx.shape)
            source_id = population_ids[ofc_indices[top_ofc_acx[0]]]
            target_id = population_ids[acx_indices[top_ofc_acx[1]]]
            best_connections['OFC→ACx'] = (source_id, target_id, ofc_to_acx[top_ofc_acx])
            print(f"  OFC→ACx: Neuron {source_id} → {target_id}: {ofc_to_acx[top_ofc_acx]:.4f}")
        if ofc_to_ofc.size > 0:
            top_ofc_ofc = np.unravel_index(np.argmax(ofc_to_ofc), ofc_to_ofc.shape)
            source_id = population_ids[ofc_indices[top_ofc_ofc[0]]]
            target_id = population_ids[ofc_indices[top_ofc_ofc[1]]]
            best_connections['OFC→OFC'] = (source_id, target_id, ofc_to_ofc[top_ofc_ofc])
            print(f"  OFC→OFC: Neuron {source_id} → {target_id}: {ofc_to_ofc[top_ofc_ofc]:.4f}")

#%% Plot actual firing rates of best connected units by type
if INCLUDE_SPIKE_HISTORY and len(best_connections) > 0:
    # Create boolean mask for EXAMPLE_EPOCH
    plot_mask = (X_ep_pop.t >= EXAMPLE_EPOCH.start[0]) & (X_ep_pop.t <= EXAMPLE_EPOCH.end[0])
    t_plot_raw = X_ep_pop.t[plot_mask]
    
    # Get event times that fall within the plot window
    tone_times_plot = tone_onset[(tone_onset >= EXAMPLE_EPOCH.start[0]) & (tone_onset <= EXAMPLE_EPOCH.end[0])]
    lick_times_plot = licks[(licks >= EXAMPLE_EPOCH.start[0]) & (licks <= EXAMPLE_EPOCH.end[0])]
    outcome_times_plot = outcome_time[(outcome_time >= EXAMPLE_EPOCH.start[0]) & (outcome_time <= EXAMPLE_EPOCH.end[0])]
    
    # Create 4 subplots for each connection type
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    axes = axes.flatten()
    
    connection_types = ['ACx→ACx', 'ACx→OFC', 'OFC→ACx', 'OFC→OFC']
    colors = {'source': 'tab:blue', 'target': 'tab:orange'}
    
    for plot_idx, conn_type in enumerate(connection_types):
        ax = axes[plot_idx]
        
        if conn_type in best_connections:
            source_id, target_id, coupling_strength = best_connections[conn_type]
            
            # Find indices in population_ids
            source_idx = list(population_ids).index(source_id)
            target_idx = list(population_ids).index(target_id)
            
            # Get firing rates for source and target neurons (restricted to epoch)
            source_fr_raw = actual_firing_rate[plot_mask, source_idx]
            target_fr_raw = actual_firing_rate[plot_mask, target_idx]
            
            # Smooth firing rates using pynapple
            source_tsd = nap.Tsd(t=t_plot_raw, d=source_fr_raw)
            target_tsd = nap.Tsd(t=t_plot_raw, d=target_fr_raw)
            source_smooth = source_tsd.smooth(std=0.05, windowsize=0.25)
            target_smooth = target_tsd.smooth(std=0.05, windowsize=0.25)
            
            # Create regular time grid at BIN_SIZE resolution for filled area plot
            t_start = EXAMPLE_EPOCH.start[0]
            t_end = EXAMPLE_EPOCH.end[0]
            t_full = np.arange(t_start, t_end + BIN_SIZE, BIN_SIZE)
            
            # Fill in smoothed values by finding nearest neighbors
            source_filled = np.zeros_like(t_full, dtype=float)
            target_filled = np.zeros_like(t_full, dtype=float)
            
            source_times = source_smooth.t
            source_values = source_smooth.d
            target_times = target_smooth.t
            target_values = target_smooth.d
            
            for i, t_val in enumerate(t_full):
                # Find nearest time point for source
                dists_source = np.abs(source_times - t_val)
                if len(dists_source) > 0 and np.min(dists_source) < BIN_SIZE:
                    source_filled[i] = source_values[np.argmin(dists_source)]
                
                # Find nearest time point for target
                dists_target = np.abs(target_times - t_val)
                if len(dists_target) > 0 and np.min(dists_target) < BIN_SIZE:
                    target_filled[i] = target_values[np.argmin(dists_target)]
            
            # Create gradient effect for area charts using multiple overlapping fills
            n_gradient_layers = 15
            
            # Blue gradient for source neuron - darker at top, lighter at bottom
            for i in range(n_gradient_layers):
                y_bottom = source_filled * (i / n_gradient_layers)
                y_top = source_filled * ((i + 1) / n_gradient_layers)
                alpha = 0.4 * ((i + 1) / n_gradient_layers)  # Increasing opacity toward top
                ax.fill_between(t_full, y_bottom, y_top, alpha=alpha, color=colors['source'], linewidth=0)
            
            # Orange gradient for target neuron - darker at top, lighter at bottom
            for i in range(n_gradient_layers):
                y_bottom = target_filled * (i / n_gradient_layers)
                y_top = target_filled * ((i + 1) / n_gradient_layers)
                alpha = 0.4 * ((i + 1) / n_gradient_layers)  # Increasing opacity toward top
                ax.fill_between(t_full, y_bottom, y_top, alpha=alpha, color=colors['target'], linewidth=0)
            
            # Add semi-transparent outlines for better visibility
            ax.plot(t_full, source_filled, color=colors['source'], linewidth=2, alpha=0.7, 
                   label=f'Source: Neuron {source_id}')
            ax.plot(t_full, target_filled, color=colors['target'], linewidth=2, alpha=0.7, 
                   label=f'Target: Neuron {target_id}')
            
            # Add event markers
            y_min, y_max = ax.get_ylim()
            y_range = y_max - y_min
            
            # Tone onsets (red vertical lines)
            for tone_t in tone_times_plot:
                ax.axvline(tone_t, color='red', alpha=0.3, linewidth=1, linestyle='--', zorder=5)
            
            # Licks (black circle markers at bottom of plot)
            if len(lick_times_plot) > 0:
                lick_y_position = y_min + 0.05 * y_range  # 5% from bottom
                ax.scatter(lick_times_plot, 
                          np.ones(len(lick_times_plot)) * lick_y_position,
                          marker='o', s=30, color='black', alpha=0.6, 
                          edgecolors='black', linewidths=0.5, zorder=10)
            
            # Outcomes (green vertical lines)
            for outcome_t in outcome_times_plot:
                if outcome_t > 0:  # Skip invalid outcome times
                    ax.axvline(outcome_t, color='green', alpha=0.3, linewidth=1, linestyle='--', zorder=5)
            
            ax.set_ylabel('Firing Rate (Hz)')
            ax.set_title(f'{conn_type}: Neuron {source_id} → {target_id}\n(Coupling: {coupling_strength:.4f})', 
                        fontweight='bold')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'No {conn_type} connections found', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{conn_type}', fontweight='bold')
            ax.set_ylabel('Firing Rate (Hz)')
    
    axes[2].set_xlabel('Time (s)')
    axes[3].set_xlabel('Time (s)')
    plt.suptitle(f'Actual Firing Rates of Best Connected Units by Type\n(Time window: {EXAMPLE_EPOCH.start[0]:.1f}s - {EXAMPLE_EPOCH.end[0]:.1f}s)', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()


# %%
# Select non-history coefficient rows and group categorical features
feature_names_all = list(X_ep_pop.columns)
non_history_indices = [i for i in range(len(feature_names_all)) if i not in history_feature_indices]

# Group categorical features together
# Define feature groups
feature_groups = {
    'tone_onset': [],
    'licks': [],
    'outcome_onset': [],
    'stimulus': [],  # All stimulus_ID values grouped together
    'category': [],
    'outcome': [],
    'previous_outcome': []
}

# Map each feature to its group
for idx in non_history_indices:
    feat_name = feature_names_all[idx]
    
    # Temporal features
    if feat_name.startswith('tone_onset_basis'):
        feature_groups['tone_onset'].append(idx)
    elif feat_name.startswith('licks_basis'):
        feature_groups['licks'].append(idx)
    elif feat_name.startswith('outcome_onset_basis'):
        feature_groups['outcome_onset'].append(idx)
    # Categorical features
    elif 'stimulus' in feat_name.lower():
        feature_groups['stimulus'].append(idx)
    elif 'category_ID' in feat_name:
        feature_groups['category'].append(idx)
    elif 'outcome_ID' in feat_name:
        feature_groups['outcome'].append(idx)
    elif 'previous_outcome' in feat_name:
        feature_groups['previous_outcome'].append(idx)

# Create grouped coefficient matrix and feature names
grouped_feature_names = []
grouped_coef_indices = []

for group_name, indices in feature_groups.items():
    if len(indices) > 0:
        grouped_feature_names.append(group_name)
        grouped_coef_indices.append(indices)

# Create grouped coefficient matrix by summing absolute values within each group
n_neurons = coef_matrix.shape[1]
n_groups = len(grouped_feature_names)
grouped_coefs = np.zeros((n_groups, n_neurons))

for group_idx, (group_name, indices) in enumerate(zip(grouped_feature_names, grouped_coef_indices)):
    # Sum absolute coefficients for this group across all neurons
    grouped_coefs[group_idx, :] = np.sum(np.abs(coef_matrix[indices, :]), axis=0)

print(f"Grouped coefficients shape: {grouped_coefs.shape}")
print(f"Grouped feature names: {grouped_feature_names}")
print(f"Number of features per group: {[len(indices) for indices in grouped_coef_indices]}")

#%%

ofc_coefs = grouped_coefs[:, ofc_indices]
acx_coefs = grouped_coefs[:, acx_indices]

print(ofc_coefs.shape)
print(acx_coefs.shape)

#%% Half violin plots for each feature group by region using Plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Calculate relative contributions (percentages) for each unit
# Normalize grouped coefficients to percentages for each unit
n_neurons = grouped_coefs.shape[1]
grouped_coefs_pct = np.zeros_like(grouped_coefs)

for unit_idx in range(n_neurons):
    unit_total = np.sum(grouped_coefs[:, unit_idx])
    if unit_total > 0:
        grouped_coefs_pct[:, unit_idx] = (grouped_coefs[:, unit_idx] / unit_total) * 100
    else:
        grouped_coefs_pct[:, unit_idx] = 0

# Get relative contributions by region
ofc_coefs_pct = grouped_coefs_pct[:, ofc_indices]
acx_coefs_pct = grouped_coefs_pct[:, acx_indices]

# Filter to only features we want to plot
features_to_plot = ['stimulus', 'category', 'licks', 'outcome', 'previous_outcome', 'tone_onset', 'outcome_onset']
features_to_plot = [f for f in features_to_plot if f in grouped_feature_names]

colors = {'ACx': '#4A90E2', 'OFC': '#E74C3C'}  # Blue for ACx, Red for OFC

feature_labels = {
    'stimulus': 'Stimulus ID',
    'category': 'Category',
    'licks': 'Licks',
    'outcome': 'Outcome',
    'previous_outcome': 'Previous Outcome',
    'tone_onset': 'Tone Onset',
    'outcome_onset': 'Outcome Onset'
}

# Create subplots
fig = make_subplots(
    rows=1, 
    cols=len(features_to_plot),
    subplot_titles=[feature_labels.get(f, f) for f in features_to_plot],
    shared_yaxes=True,
    horizontal_spacing=0.05
)

for col_idx, feature in enumerate(features_to_plot):
    # Get feature index
    feat_idx = grouped_feature_names.index(feature)
    
    # Get relative contributions for this feature by region
    acx_data = acx_coefs_pct[feat_idx, :].flatten()
    ofc_data = ofc_coefs_pct[feat_idx, :].flatten()
    
    # Plot half violins using plotly
    # ACx on left (negative side)
    if len(acx_data) > 0:
        fig.add_trace(
            go.Violin(
                y=acx_data,
                x=[col_idx] * len(acx_data),
                name='ACx',
                side='negative',
                box_visible=True,
                meanline_visible=True,
                fillcolor=colors['ACx'],
                line_color=colors['ACx'],
                opacity=0.6,
                showlegend=(col_idx == 0),  # Only show legend for first subplot
                legendgroup='ACx'
            ),
            row=1, col=col_idx+1
        )
    
    # OFC on right (positive side)
    if len(ofc_data) > 0:
        fig.add_trace(
            go.Violin(
                y=ofc_data,
                x=[col_idx] * len(ofc_data),
                name='OFC',
                side='positive',
                box_visible=True,
                meanline_visible=True,
                fillcolor=colors['OFC'],
                line_color=colors['OFC'],
                opacity=0.6,
                showlegend=(col_idx == 0),  # Only show legend for first subplot
                legendgroup='OFC'
            ),
            row=1, col=col_idx+1
        )

# Update layout
fig.update_layout(
    title={
        'text': 'Distribution of Relative Contributions by Region<br><sub>Excluding Spike History</sub>',
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 16, 'family': 'sans-serif'}
    },
    height=600,
    showlegend=True,
    violinmode='overlay',
    violingroupgap=0,
    violingap=0,
    font=dict(family='sans-serif', size=11)
)

# Update y-axis label (only on first subplot)
fig.update_yaxes(title_text='Relative Contribution (%)', row=1, col=1)

# Update x-axes to remove tick labels
for col_idx in range(1, len(features_to_plot) + 1):
    fig.update_xaxes(showticklabels=False, row=1, col=col_idx)

# Save plot as HTML file in the GLM folder (same directory as this script)
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
output_path = os.path.join(script_dir, "partial_contributions_by_region.html")
fig.write_html(output_path)
print(f"Plot saved to: {output_path}")

#%% Time to react: Find basis with maximum absolute coefficient for each feature
# Ensure we have region indices
if 'acx_indices' not in locals() or 'ofc_indices' not in locals():
    acx_indices = [i for i, r in enumerate(population_regions) if r == 'ACx']
    ofc_indices = [i for i, r in enumerate(population_regions) if r == 'OFC']

# Evaluate basis functions on grid to get time values
time_event, basis_kernels_event = basis_events.evaluate_on_grid(event_window_bins)
time_event_sec = time_event * EVENT_WINDOW_SEC

time_cat, basis_kernels_cat = basis_categorical.evaluate_on_grid(acausal_window_bins)
acausal_total_sec = ACAUSAL_BEFORE_SEC + ACAUSAL_AFTER_SEC
time_cat_sec = time_cat * acausal_total_sec - ACAUSAL_BEFORE_SEC  # Shift so 0 is at event

# Find the peak time for each basis function (time at which each basis function has maximum value)
# For temporal (causal) basis functions
basis_peak_times_event = np.zeros(N_BASIS_FUNCS)
for i in range(N_BASIS_FUNCS):
    peak_idx = np.argmax(np.abs(basis_kernels_event[:, i]))
    basis_peak_times_event[i] = time_event_sec[peak_idx]

# For categorical (acausal) basis functions
basis_peak_times_cat = np.zeros(N_BASIS_FUNCS)
for i in range(N_BASIS_FUNCS):
    peak_idx = np.argmax(np.abs(basis_kernels_cat[:, i]))
    basis_peak_times_cat[i] = time_cat_sec[peak_idx]

# Extract time to react for each feature group and neuron
# For each feature group, find which basis has the max abs coefficient, then get its time
time_to_react = {}  # {feature_name: array of shape (n_neurons,)}

for group_idx, (group_name, indices) in enumerate(zip(grouped_feature_names, grouped_coef_indices)):
    if len(indices) == 0:
        continue
    
    # Determine if this is a temporal or categorical feature
    is_temporal = group_name in ['tone_onset', 'licks', 'outcome_onset']
    basis_peak_times = basis_peak_times_event if is_temporal else basis_peak_times_cat
    
    # For each neuron, find the basis with max abs coefficient
    neuron_times = np.zeros(n_neurons)
    
    for neuron_idx in range(n_neurons):
        # Get coefficients for this feature group and neuron
        coefs_for_group = coef_matrix[indices, neuron_idx]
        abs_coefs = np.abs(coefs_for_group)
        
        if np.max(abs_coefs) == 0:
            # No significant coefficient, set to NaN
            neuron_times[neuron_idx] = np.nan
            continue
        
        # Find the index of the maximum absolute coefficient
        max_idx_in_group = np.argmax(abs_coefs)
        feature_idx_in_group = indices[max_idx_in_group]
        feature_name = feature_names_all[feature_idx_in_group]
        
        # Extract basis index from feature name (e.g., "tone_onset_basis3" -> 3)
        # Pattern: feature_name_basis{basis_idx}
        basis_match = re.search(r'basis(\d+)$', feature_name)
        if basis_match:
            basis_idx = int(basis_match.group(1))
            if 0 <= basis_idx < len(basis_peak_times):
                neuron_times[neuron_idx] = basis_peak_times[basis_idx]
            else:
                neuron_times[neuron_idx] = np.nan
        else:
            neuron_times[neuron_idx] = np.nan
    
    time_to_react[group_name] = neuron_times

# Filter to only features we want to plot
features_to_plot_time = ['stimulus', 'category', 'licks', 'outcome', 'previous_outcome', 'tone_onset', 'outcome_onset']
features_to_plot_time = [f for f in features_to_plot_time if f in time_to_react]

# Create matplotlib figure with boxplots for time to react
fig_time, axes_time = plt.subplots(1, len(features_to_plot_time), figsize=(5 * len(features_to_plot_time), 6), sharey=True)
if len(features_to_plot_time) == 1:
    axes_time = [axes_time]

for col_idx, feature in enumerate(features_to_plot_time):
    if feature not in time_to_react:
        continue
    
    ax = axes_time[col_idx]
    
    times_all = time_to_react[feature]
    
    # Get times by region
    acx_times = times_all[acx_indices]
    ofc_times = times_all[ofc_indices]
    
    # Remove NaN values
    acx_times_clean = acx_times[~np.isnan(acx_times)]
    ofc_times_clean = ofc_times[~np.isnan(ofc_times)]
    
    # Prepare data for boxplot (list of arrays)
    box_data = []
    if len(acx_times_clean) > 0:
        box_data.append(acx_times_clean)
    if len(ofc_times_clean) > 0:
        box_data.append(ofc_times_clean)
    
    if len(box_data) > 0:
        # Create boxplot with custom positions and widths
        bp = ax.boxplot(
            box_data,
            positions=[1, 2] if len(box_data) == 2 else ([1] if len(acx_times_clean) > 0 else [2]),
            widths=0.6,  # Large box width
            patch_artist=True,
            showmeans=True,  # Show mean
            meanline=True,  # Mean as line
            showfliers=True  # Show outliers
        )
        
        # Color the boxes
        box_colors = []
        if len(acx_times_clean) > 0:
            box_colors.append(colors['ACx'])
        if len(ofc_times_clean) > 0:
            box_colors.append(colors['OFC'])
        
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor(color)
            patch.set_linewidth(2)
        
        # Style the other elements
        for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
            if element in bp:
                for item in bp[element]:
                    if element == 'fliers':
                        item.set_markerfacecolor(box_colors[0] if len(box_colors) > 0 else 'black')
                        item.set_markeredgecolor(box_colors[0] if len(box_colors) > 0 else 'black')
                    else:
                        item.set_color(box_colors[0] if len(box_colors) > 0 else 'black')
                        item.set_linewidth(2)
    
    ax.set_title(feature_labels.get(feature, feature), fontsize=12, fontweight='bold')
    ax.set_xticks([1, 2] if len(box_data) == 2 else ([1] if len(acx_times_clean) > 0 else [2]))
    ax.set_xticklabels(['ACx' if len(acx_times_clean) > 0 else '', 'OFC' if len(ofc_times_clean) > 0 else ''][:len(box_data)])
    ax.set_ylabel('Time to React (s)' if col_idx == 0 else '')
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle('Time to React by Feature and Region\n(Time at which maximum absolute coefficient occurs)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()

# Save plot
output_path_time = os.path.join(script_dir, "time_to_react_by_region.png")
fig_time.savefig(output_path_time, dpi=300, bbox_inches='tight')
print(f"Time to react plot saved to: {output_path_time}")
plt.show()

#%%
