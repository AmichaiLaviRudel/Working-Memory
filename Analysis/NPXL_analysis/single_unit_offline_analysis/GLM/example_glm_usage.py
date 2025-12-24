#%%
from typing import Any
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap

import nemos as nmo

# some helper plotting functions
from nemos import _documentation_utils as doc_plots
from patsy import dmatrix
# configure pynapple to ignore conversion warning
nap.nap_config.suppress_conversion_warnings = True

# configure plots some
plt.style.use(nmo.styles.plot_style)



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
RATE_TH = 5

# Bin size for binning spike counts and predictors (in seconds)
BIN_SIZE = 0.05

# Window for peri-event epoching (in seconds, relative to event)
EPOCH_START = -1
EPOCH_END = 3

# Example epoch (for plotting): here from 236s to 242s
EXAMPLE_EPOCH = nap.IntervalSet(start=30, end=40)

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

# Number of raised cosine basis functions for all event-locked convolutions
N_BASIS_FUNCS = 10

# Parameter to control whether to include spike history
N_POPULATION = 200

INCLUDE_SPIKE_HISTORY = True  # Set to False to exclude spike history
# Note: When True, creates ALL-TO-ALL connectivity - each neuron's spike history 
#       features are included and can affect ALL neurons via PopulationGLM coefficients

#%% filtering
spikes = spikes.getby_category("unit_type")["good"]
spikes = spikes.getby_threshold("rate", RATE_TH)


# %% 
licks = np.loadtxt(os.path.join(base_path, "G5A3_2b_4t_new2_g0_tcat.nidq.xd_0_2_0.txt"))
tone_onset = np.loadtxt(os.path.join(base_path, "G5A3_2b_4t_new2_g0_tcat.nidq.xd_0_1_100.txt"))
stimuli_outcome_df = pd.read_csv(os.path.join(probe_path_acx, "analysis_output", "stimuli_outcome_df.csv"))


stimulus = stimuli_outcome_df["stimulus"].values.astype(float)
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
formula = "stimulus_ID + category_ID + outcome_ID + previous_outcome"
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
fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

# Spike count
axes[0].step(
    neuron_count.restrict(epoch_one_spk).t,
    neuron_count.restrict(epoch_one_spk).d,
    where="post",
    label="Spike Count"
)
axes[0].set_ylabel("Counts")
axes[0].set_title("Spike Count Time Series")
axes[0].legend()

# Tone onset
axes[1].step(
    temporal_features["tone_onset"].restrict(epoch_one_spk).t,
    temporal_features["tone_onset"].restrict(epoch_one_spk).d,
    color="red",
    alpha=0.5,
    label="Tone Onset"
)
axes[1].set_ylabel("Event")
axes[1].set_title("Tone Onset")
axes[1].legend()

# Licks
axes[2].step(
    temporal_features["licks"].restrict(epoch_one_spk).t,
    temporal_features["licks"].restrict(epoch_one_spk).d,
    color="blue",
    alpha=0.5,
    label="Licks"
)
axes[2].set_ylabel("Event")
axes[2].set_title("Licks")
axes[2].legend()

# Outcome Onset
axes[3].step(
    temporal_features["outcome_onset"].restrict(epoch_one_spk).t,
    temporal_features["outcome_onset"].restrict(epoch_one_spk).d,
    color="green",
    alpha=0.5,
    label="Outcome Onset"
)
axes[3].set_xlabel("Time (sec)")
axes[3].set_ylabel("Event")
axes[3].set_title("Outcome Onset")
axes[3].legend()

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
history_window_sec = 0.8
history_window_bins = int(history_window_sec * neuron_count.rate)

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
# Choose a representative window
plot_ep = nap.IntervalSet(start=epochs.start[0]+2.5*60, end=epochs.start[0]+3*60)
# NeMoS helper plot - restrict data to plot window first
y_plot = y_ep.restrict(plot_ep)
pred_plot_mask = (y_ep.t >= plot_ep.start[0]) & (y_ep.t <= plot_ep.end[0])

# Create a pynapple Tsd for predicted rate in plot window
pred_rate_tsd = nap.Tsd(t=y_ep.t[pred_plot_mask], d=pred_rate_hz[pred_plot_mask])
# Smooth predicted rate using Tsd smooth method (0.25s window = 5 bins at BIN_SIZE=0.05s)
pred_rate_tsd = pred_rate_tsd.smooth(std=0.05, windowsize=0.25)

# Fill missing time points with zeros to ensure continuous area chart
# Create regular time grid at BIN_SIZE resolution
t_start = plot_ep.start[0]
t_end = plot_ep.end[0]
t_full = np.arange(t_start, t_end + BIN_SIZE, BIN_SIZE)
# Initialize with zeros
pred_rate_filled = np.zeros_like(t_full, dtype=float)
actual_rate_filled = np.zeros_like(t_full, dtype=float)

# Fill in actual values by finding nearest neighbors
actual_mask = pred_plot_mask
if np.any(actual_mask):
    actual_times = y_ep.t[actual_mask]
    actual_values = actual_rate_hz[actual_mask]
    for i, t_val in enumerate(t_full):
        # Find nearest time point
        dists = np.abs(actual_times - t_val)
        if np.min(dists) < BIN_SIZE:
            actual_rate_filled[i] = actual_values[np.argmin(dists)]

# Fill in predicted values by finding nearest neighbors
if len(pred_rate_tsd) > 0:
    pred_times = pred_rate_tsd.t
    pred_values = pred_rate_tsd.d
    for i, t_val in enumerate(t_full):
        # Find nearest time point
        dists = np.abs(pred_times - t_val)
        if np.min(dists) < BIN_SIZE:
            pred_rate_filled[i] = pred_values[np.argmin(dists)]

# Create filled Tsd objects
pred_rate_tsd = nap.Tsd(t=t_full, d=pred_rate_filled)
actual_rate_tsd = nap.Tsd(t=t_full, d=actual_rate_filled)


# Direct matplotlib comparison with area chart and gradients
fig, ax = plt.subplots(1, 1, figsize=(12, 4))

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

ax.set_xlabel("Time (sec)")
ax.set_ylabel("Firing rate (Hz)")
ax.set_title(f"Actual vs Predicted Firing Rate - Neuron {EXAMPLE_NEURON_ID}")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()

#%% Visualize basis functions and how they reconstruct temporal kernels for all feature types
# Evaluate basis functions on a grid (for temporal event features - causal)
time_event, basis_kernels_event = basis_events.evaluate_on_grid(event_window_bins)
time_event_sec = time_event * EVENT_WINDOW_SEC

# Evaluate basis functions for categorical features (acausal)
time_cat, basis_kernels_cat = basis_categorical.evaluate_on_grid(acausal_window_bins)
time_cat_sec = time_cat * acausal_total_sec - ACAUSAL_BEFORE_SEC  # Shift so 0 is at event

# Evaluate basis functions for spike history
time_hist, basis_kernels_hist = basis_history.evaluate_on_grid(history_window_bins)
time_hist_sec = time_hist * history_window_sec

# Get feature names
temporal_feature_names = list(temporal_features.columns)
categorical_feature_names = list(categorical_features.columns)

# Calculate grid layout - now 2 columns per feature (kernel + coefficients)
n_temporal = len(temporal_feature_names)
n_categorical = len(categorical_feature_names)
n_total_features = n_temporal + n_categorical + 1  # temporal + categorical + spike_history

# Create subplots: first 2 rows for basis functions, then 3 columns per feature (weighted, kernel, coefficients)
fig = plt.figure(figsize=(18, 3 * (n_total_features + 2)))
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
        
        # Column 2: Reconstructed kernel
        ax2 = fig.add_subplot(gs[row_idx, 1])
        ax2.plot(time_event_sec, reconstructed_kernel, 'b-', linewidth=3)
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
        
        # Column 2: Reconstructed kernel
        ax2 = fig.add_subplot(gs[row_idx, 1])
        ax2.plot(time_cat_sec, reconstructed_kernel, 'g-', linewidth=3)
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
    
    # Column 2: Reconstructed kernel
    ax2 = fig.add_subplot(gs[row_idx, 1])
    ax2.plot(time_hist_sec, reconstructed_hist, 'r-', linewidth=3)
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

plt.suptitle("GLM Temporal Kernels: Causal (Temporal) + Acausal (Categorical) + Spike History", fontsize=14, fontweight='bold')
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
#%%
# Select neurons for population analysis
N_POPULATION = min(N_POPULATION, len(spikes))
population_ids = np.random.choice(list(spikes.keys()), N_POPULATION, replace=False)


print(f"\nBuilding population target matrix:")
print(f"  {N_POPULATION} neurons: {population_ids}")


#%% Build population spike count matrix and per-neuron spike history features
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
    history_features_list = []

    for uid in population_ids:
        # Get this neuron's spike count from full matrix
        uid_idx = list(spike_count.columns).index(uid)
        neuron_spikes = spike_count[:, uid_idx]
        
        # Compute spike history features for this neuron
        neuron_history = basis_history.compute_features(neuron_spikes)
        neuron_history_ep = neuron_history.restrict(epochs)
        
        # Align to X_shared_ep timestamps
        if neuron_history_ep.shape[0] != X_shared_ep.shape[0]:
            hist_indices = np.searchsorted(neuron_history_ep.t, X_shared_ep.t)
            hist_indices = np.clip(hist_indices, 0, len(neuron_history_ep.t) - 1)
            neuron_history_aligned = neuron_history_ep.values[hist_indices]
        else:
            neuron_history_aligned = neuron_history_ep.values
        
        # Label columns with source neuron ID (this history can affect all target neurons)
        history_cols = [f"hist_from_neuron{uid}_basis{i}" for i in range(basis_history.n_basis_funcs)]
        history_features_list.append((neuron_history_aligned, history_cols))

    # Combine all history features from all source neurons
    # Structure: [neuron0_history | neuron1_history | ... | neuronN_history]
    # Each history can affect all target neurons via PopulationGLM coefficients
    all_history_features = np.column_stack([hist for hist, _ in history_features_list])
    all_history_cols = [col for _, cols in history_features_list for col in cols]

    print(f"  All-to-all connectivity: {len(population_ids)} source neurons × {basis_history.n_basis_funcs} basis = {all_history_features.shape[1]} history features")
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
print(f"Regions: {dict(zip(population_ids, population_regions))}")

# %%
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
        n_basis = basis_history.n_basis_funcs
        n_target = len(population_ids)
        
        history_coefs_reshaped = history_coefs.reshape(n_source, n_basis, n_target)
        
        # Reconstruct kernels for each connection and compute normalized coupling strength
        # This accounts for both coefficients and basis function shapes
        # Use L2 norm of reconstructed kernel as coupling strength (normalized metric)
        # This is more interpretable than summing raw coefficients
        # Shape: (n_source_neurons, n_target_neurons)
        coupling_strength = np.zeros((n_source, n_target))
        coupling_signed = np.zeros((n_source, n_target))
        
        # Compute basis kernels for spike history
        # Use the same window size as when basis_history was created
        history_window_sec = 0.8
        # Use the rate from the binned data (1/BIN_SIZE)
        history_window_bins = int(history_window_sec / BIN_SIZE)
        time_hist, basis_kernels_hist = basis_history.evaluate_on_grid(history_window_bins)
        
        for source_idx in range(n_source):
            for target_idx in range(n_target):
                # Get coefficients for this connection
                coefs = history_coefs_reshaped[source_idx, :, target_idx]
                # Reconstruct kernel (same as for temporal features)
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
        
        # Visualize by region
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Full matrix with region boundaries (top row)
        ax_full = fig.add_subplot(gs[0, :])
        # Reorder neurons by region (already sorted by coupling strength within each region)
        region_order = acx_indices + ofc_indices
        coupling_ordered = coupling_strength[np.ix_(region_order, region_order)]
        # Use diverging colormap to show excitation (red/positive) and inhibition (blue/negative)
        im = ax_full.imshow(coupling_ordered, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
        ax_full.set_xlabel('Target Neuron')
        ax_full.set_ylabel('Source Neuron')
        ax_full.set_title('Full Connectivity Matrix (Grouped by Region, Sorted by Coupling Strength)', fontsize=12, fontweight='bold')
        
        # Add region boundaries
        if len(acx_indices) > 0 and len(ofc_indices) > 0:
            ax_full.axvline(len(acx_indices) - 0.5, color='white', linewidth=2, linestyle='--')
            ax_full.axhline(len(acx_indices) - 0.5, color='white', linewidth=2, linestyle='--')
            ax_full.text(len(acx_indices)/2, -1, 'ACx', ha='center', va='top', fontsize=10, fontweight='bold', color='white')
            ax_full.text(len(acx_indices) + len(ofc_indices)/2, -1, 'OFC', ha='center', va='top', fontsize=10, fontweight='bold', color='white')
            ax_full.text(-1, len(acx_indices)/2, 'ACx', ha='right', va='center', fontsize=10, fontweight='bold', color='white', rotation=90)
            ax_full.text(-1, len(acx_indices) + len(ofc_indices)/2, 'OFC', ha='right', va='center', fontsize=10, fontweight='bold', color='white', rotation=90)
        
        plt.colorbar(im, ax=ax_full, label='Coupling Strength (Normalized -1 to 1)\nRed=Excitatory, Blue=Inhibitory')
        
        # ACx → ACx
        if acx_to_acx.size > 0:
            ax = fig.add_subplot(gs[1, 0])
            im = ax.imshow(acx_to_acx, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
            ax.set_title('ACx → ACx', fontweight='bold')
            ax.set_xlabel('Target ACx')
            ax.set_ylabel('Source ACx')
            plt.colorbar(im, ax=ax)
            ax.plot([-0.5, len(acx_indices)-0.5], [-0.5, len(acx_indices)-0.5], 'r--', linewidth=1, alpha=0.5)
        
        # ACx → OFC
        if acx_to_ofc.size > 0:
            ax = fig.add_subplot(gs[1, 1])
            im = ax.imshow(acx_to_ofc, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
            ax.set_title('ACx → OFC', fontweight='bold')
            ax.set_xlabel('Target OFC')
            ax.set_ylabel('Source ACx')
            plt.colorbar(im, ax=ax)
        
        # OFC → ACx
        if ofc_to_acx.size > 0:
            ax = fig.add_subplot(gs[1, 2])
            im = ax.imshow(ofc_to_acx, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
            ax.set_title('OFC → ACx', fontweight='bold')
            ax.set_xlabel('Target ACx')
            ax.set_ylabel('Source OFC')
            plt.colorbar(im, ax=ax)
        
        # OFC → OFC
        if ofc_to_ofc.size > 0:
            ax = fig.add_subplot(gs[2, 0])
            im = ax.imshow(ofc_to_ofc, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
            ax.set_title('OFC → OFC', fontweight='bold')
            ax.set_xlabel('Target OFC')
            ax.set_ylabel('Source OFC')
            plt.colorbar(im, ax=ax)
            ax.plot([-0.5, len(ofc_indices)-0.5], [-0.5, len(ofc_indices)-0.5], 'r--', linewidth=1, alpha=0.5)
        
        # Comparison: Within-region vs cross-region
        ax = fig.add_subplot(gs[2, 1:])
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
        
        if len(within_region) > 0 and len(cross_region) > 0:
            ax.hist(within_region, bins=30, alpha=0.6, label='Within-region', color='tab:blue', density=True)
            ax.hist(cross_region, bins=30, alpha=0.6, label='Cross-region', color='tab:orange', density=True)
            ax.set_xlabel('Coupling Strength')
            ax.set_ylabel('Density')
            ax.set_title('Within-Region vs Cross-Region Coupling', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add statistics text
            ax.text(0.05, 0.95, f'Within: μ={np.mean(within_region):.4f}\nCross: μ={np.mean(cross_region):.4f}', 
                   transform=ax.transAxes, verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('All-to-All Connectivity: ACx vs OFC Analysis', fontsize=16, fontweight='bold', y=0.995)
        
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

#%% Plot actual vs predicted firing rate for each neuron with event markers
# Create boolean mask for EXAMPLE_EPOCH
UNIT_TO_PLOT = min(20, len(population_ids))
population_ids_sorted = sorted(population_ids, key=lambda x: per_unit_scores[x], reverse=True)
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

# %%
