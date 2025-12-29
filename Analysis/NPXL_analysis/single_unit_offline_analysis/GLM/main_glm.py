"""
Main GLM analysis script.

This script orchestrates the full GLM analysis pipeline:
1. Load spike data and events
2. Build design matrices
3. Fit single-neuron and population GLMs
4. Generate visualizations

Usage:
    python main_glm.py
"""
#%%
import os
import numpy as np
import pynapple as nap
import nemos as nmo
import matplotlib.pyplot as plt

# Configure pynapple
nap.nap_config.suppress_conversion_warnings = True

# Local imports
from config import (
    BASE_PATH, RATE_THRESHOLD, BIN_SIZE, PREPROCESSING_BIN_SIZE,
    EPOCH_START, EPOCH_END, EXAMPLE_EPOCH, EXAMPLE_NEURON_ID,
    LOW_BOUNDARY_THRESHOLD, HIGH_BOUNDARY_THRESHOLD,
    N_BASIS_FUNCS, EVENT_WINDOW_SEC, ACAUSAL_BEFORE_SEC, ACAUSAL_AFTER_SEC,
    HISTORY_WINDOW_SEC, HISTORY_ACAUSAL_BEFORE_SEC, HISTORY_ACAUSAL_AFTER_SEC,
    N_POPULATION, INCLUDE_SPIKE_HISTORY,
    GLM_SOLVER, GLM_REGULARIZER, GLM_REGULARIZER_STRENGTH
)
from colors import configure_fonts
from loading import load_all_probes, load_events, filter_spikes
from design_matrix import (
    create_categorical_features, create_temporal_features,
    create_basis_functions, create_categorical_impulses,
    build_design_matrix, build_population_design_matrix
)
from plotting import (
    plot_spike_count_with_events, plot_actual_vs_predicted,
    plot_partial_contributions, plot_basis_and_kernels,
    plot_connectivity_matrix, plot_connectivity_breakdown,
    plot_contributions_by_region_plotly, plot_time_to_react_boxplot,
    smooth_and_fill, find_units_with_high_temporal_coefficients,
    plot_perievent_raster_peth
)

# Configure fonts and plot style
configure_fonts()
plt.style.use(nmo.styles.plot_style)

#%%
# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("=" * 60)
print("LOADING DATA")
print("=" * 60)

BASE_PATH =  r"Z:\Shared\Amichai\NPXL\Recs\group7\catgt_G7A2_novice_2b_4t_g1"
INCLUDE_SPIKE_HISTORY = False
# Load spikes from all probes
spikes, probe_path_acx, probe_path_ofc = load_all_probes(BASE_PATH)

# Filter to good units with sufficient firing rate
spikes = filter_spikes(spikes, unit_type='good', rate_threshold=RATE_THRESHOLD)
print(f"\nFiltered to {len(spikes)} good units with rate >= {RATE_THRESHOLD} Hz")

# Load behavioral events
licks, tone_onset, stimuli_outcome_df = load_events(BASE_PATH, probe_path_acx)

#%%
# ============================================================================
# 2. CREATE FEATURES
# ============================================================================
print("\n" + "=" * 60)
print("CREATING FEATURES")
print("=" * 60)

# Create categorical features
categorical_features, data_df, outcome_time = create_categorical_features(
    tone_onset, stimuli_outcome_df, PREPROCESSING_BIN_SIZE,
    LOW_BOUNDARY_THRESHOLD, HIGH_BOUNDARY_THRESHOLD
)
print(f"Categorical features: {categorical_features.shape}")

# Create temporal features
temporal_features, full_ep = create_temporal_features(
    tone_onset, licks, outcome_time, BIN_SIZE
)
print(f"Temporal features: {temporal_features.shape}")

# Build spike counts
unit_ids = list(spikes.keys())
spike_count = spikes.count(BIN_SIZE, ep=full_ep)
spike_count = nap.TsdFrame(
    t=spike_count.t,
    d=spike_count.values[:, np.argsort(unit_ids)],
    columns=unit_ids,
)

# Build epochs
start = tone_onset - EPOCH_START
end = tone_onset + EPOCH_END
epochs = nap.IntervalSet(start=start, end=end)


# ============================================================================
# 3. CREATE BASIS FUNCTIONS
# ============================================================================
print("\n" + "=" * 60)
print("CREATING BASIS FUNCTIONS")
print("=" * 60)

basis_events, basis_categorical, basis_history, event_window_bins, acausal_window_bins, history_window_bins = create_basis_functions(
    N_BASIS_FUNCS, EVENT_WINDOW_SEC, ACAUSAL_BEFORE_SEC, ACAUSAL_AFTER_SEC,
    HISTORY_WINDOW_SEC, temporal_features.rate
)


# ============================================================================
# 4. BUILD DESIGN MATRIX FOR SINGLE NEURON
# ============================================================================
print("\n" + "=" * 60)
print("BUILDING SINGLE-NEURON DESIGN MATRIX")
print("=" * 60)

# Select example neuron
neuron_count = spike_count[:, EXAMPLE_NEURON_ID]

# Create categorical impulses
categorical_impulse_tsd = create_categorical_impulses(
    categorical_features, temporal_features, tone_onset
)

# Build design matrix
X, hist_feature_indices = build_design_matrix(
    temporal_features, categorical_impulse_tsd, neuron_count,
    basis_events, basis_categorical, basis_history, N_BASIS_FUNCS
)


# ============================================================================
# 5. FIT SINGLE-NEURON GLM
# ============================================================================
print("\n" + "=" * 60)
print("FITTING SINGLE-NEURON GLM")
print("=" * 60)

# Restrict to epochs
X_ep = X.restrict(epochs)
y_in_epochs = neuron_count.restrict(epochs)
y_ep = y_in_epochs.restrict(X_ep.time_support)

# Align timestamps if needed
if y_ep.shape[0] != X_ep.shape[0]:
    y_times = np.asarray(y_ep.t)
    x_times = np.asarray(X_ep.t)
    y_indices = np.searchsorted(y_times, x_times)
    y_indices = np.clip(y_indices, 0, len(y_times) - 1)
    y_ep = nap.Tsd(t=X_ep.t, d=y_ep.d[y_indices])

print(f"X_ep shape: {X_ep.shape}")
print(f"y_ep shape: {y_ep.shape}")

# Fit GLM
glm_basis = nmo.glm.GLM(solver_name=GLM_SOLVER, regularizer=GLM_REGULARIZER, 
                        regularizer_strength=GLM_REGULARIZER_STRENGTH)
glm_basis.fit(X_ep, y_ep)

# Evaluate
train_score = glm_basis.score(X_ep, y_ep, score_type='pseudo-r2-Cohen')
print(f"Train pseudo-R2: {train_score:.4f}")


# ============================================================================
# 6. SINGLE-NEURON VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 60)
print("GENERATING SINGLE-NEURON VISUALIZATIONS")
print("=" * 60)

#%% Plot spike count with events
fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
plot_spike_count_with_events(axes, neuron_count, temporal_features, EXAMPLE_EPOCH, BIN_SIZE)
plt.tight_layout()
plt.show()

#%% Plot actual vs predicted firing rate
pred_rate_hz = np.squeeze(glm_basis.predict(X_ep)) / BIN_SIZE
actual_rate_hz = np.squeeze(y_ep.d) / BIN_SIZE

plot_ep = EXAMPLE_EPOCH
plot_mask = (y_ep.t >= plot_ep.start[0]) & (y_ep.t <= plot_ep.end[0])

t_start = plot_ep.start[0]
t_end = plot_ep.end[0]
t_full = np.arange(t_start, t_end + BIN_SIZE, BIN_SIZE)

actual_filled = smooth_and_fill(t_full, y_ep.t[plot_mask], actual_rate_hz[plot_mask], BIN_SIZE)
pred_filled = smooth_and_fill(t_full, y_ep.t[plot_mask], pred_rate_hz[plot_mask], BIN_SIZE)

fig, ax = plt.subplots(1, 1, figsize=(14, 5))
plot_actual_vs_predicted(ax, t_full, actual_filled, pred_filled, 
                        f"Actual vs Predicted Firing Rate - Neuron {EXAMPLE_NEURON_ID}")
plt.tight_layout()
plt.show()

#%% Plot partial contributions
coefs = np.asarray(glm_basis.coef_).flatten()
fig = plot_partial_contributions(
    coefs, list(X.columns), 
    list(temporal_features.columns), 
    list(categorical_features.columns)
)
plt.tight_layout()
plt.show()

#%% Plot basis functions and kernels
acausal_total_sec = ACAUSAL_BEFORE_SEC + ACAUSAL_AFTER_SEC
fig = plot_basis_and_kernels(
    glm_basis.coef_,
    list(X.columns),
    list(temporal_features.columns),
    list(categorical_features.columns),
    basis_events,
    basis_categorical,
    basis_history,
    event_window_bins,
    EVENT_WINDOW_SEC,
    acausal_window_bins,
    ACAUSAL_BEFORE_SEC,
    acausal_total_sec,
    history_window_bins,
    HISTORY_WINDOW_SEC,
    N_BASIS_FUNCS
)
plt.show()


# ============================================================================
# 7. POPULATION GLM
# ============================================================================
print("\n" + "=" * 60)
print("FITTING POPULATION GLM")
print("=" * 60)

# Select population
N_POP = min(N_POPULATION, len(spikes))
population_ids = np.random.choice(list(spikes.keys()), N_POP, replace=False)
print(f"Selected {N_POP} neurons for population analysis")

# Build population spike counts
spikes_population = spikes[population_ids]
spike_count_population = spikes_population.count(BIN_SIZE, ep=epochs)
spike_count_population = nap.TsdFrame(
    t=spike_count_population.t,
    d=spike_count_population.values[:, np.argsort(population_ids)],
    columns=population_ids,
)

# Get shared predictors (without spike history)
non_history_indices = [i for i, col in enumerate(X.columns) if 'spike_history' not in col]
X_shared = nap.TsdFrame(
    t=X.t,
    d=X.values[:, non_history_indices],
    columns=[col for col in X.columns if 'spike_history' not in col]
)

# Build population design matrix
X_ep_pop, n_history_features, history_feature_indices = build_population_design_matrix(
    X_shared, spike_count_population, spike_count, population_ids, epochs,
    INCLUDE_SPIKE_HISTORY, N_BASIS_FUNCS,
    HISTORY_ACAUSAL_BEFORE_SEC, HISTORY_ACAUSAL_AFTER_SEC
)

# Align target
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

# Get region info
population_regions = spikes_population.get_info('region').values

# Fit population GLM
model_pop = nmo.glm.PopulationGLM(
    solver_name=GLM_SOLVER,
    regularizer=GLM_REGULARIZER,
    regularizer_strength=GLM_REGULARIZER_STRENGTH
)

print("\nFitting Population GLM...")
model_pop.fit(X_ep_pop, y_ep_pop)

pop_score = model_pop.score(X_ep_pop, y_ep_pop, score_type='pseudo-r2-Cohen')
print(f"Population GLM Cohen's pseudo-R2: {pop_score:.4f}")

#%%
# ============================================================================
# 8. POPULATION VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 60)
print("GENERATING POPULATION VISUALIZATIONS")
print("=" * 60)

# Get predictions
y_pred_pop = model_pop.predict(X_ep_pop)
y_actual_pop = y_ep_pop.values
y_pred_np = np.asarray(y_pred_pop)
y_actual_np = np.asarray(y_actual_pop)

# Compute per-neuron scores
per_unit_scores = {}
unit_regions = {}
print("\nPer-neuron pseudo-R²:")
for i, uid in enumerate(population_ids):
    y_pred_i = y_pred_np[:, i]
    y_actual_i = y_actual_np[:, i]
    
    y_mean = np.mean(y_actual_i)
    eps = 1e-10
    
    dev_model = 2 * np.sum(
        y_actual_i * np.log((y_actual_i + eps) / (y_pred_i + eps)) - (y_actual_i - y_pred_i)
    )
    dev_null = 2 * np.sum(
        y_actual_i * np.log((y_actual_i + eps) / (y_mean + eps)) - (y_actual_i - y_mean)
    )
    
    pseudo_r2 = 1 - (dev_model / dev_null) if dev_null > 0 else 0
    per_unit_scores[uid] = pseudo_r2
    unit_regions[uid] = population_regions[i]

# Get region indices
acx_indices = [i for i, r in enumerate(population_regions) if r == 'ACx']
ofc_indices = [i for i, r in enumerate(population_regions) if r == 'OFC']

print(f"\nACx neurons: {len(acx_indices)}")
print(f"OFC neurons: {len(ofc_indices)}")

#%%
# ============================================================================
# 9. CONNECTIVITY ANALYSIS (if spike history included)
# ============================================================================
if INCLUDE_SPIKE_HISTORY and n_history_features > 0:
    print("\n" + "=" * 60)
    print("ANALYZING CONNECTIVITY")
    print("=" * 60)
    
    coef_matrix = np.asarray(model_pop.coef_)
    n_source = len(population_ids)
    n_target = len(population_ids)
    
    coupling_strength = np.zeros((n_source, n_target))
    
    for source_idx, source_uid in enumerate(population_ids):
        for target_idx, target_uid in enumerate(population_ids):
            hist_pattern = f"hist_from_neuron{source_uid}_basis"
            hist_cols = [i for i, col in enumerate(X_ep_pop.columns) if hist_pattern in col]
            
            if len(hist_cols) > 0:
                coupling_coefs = coef_matrix[hist_cols, target_idx]
                coupling_strength[source_idx, target_idx] = np.sum(coupling_coefs)
    
    # Normalize
    max_abs = np.max(np.abs(coupling_strength))
    coupling_normalized = coupling_strength / (max_abs + 1e-10)
    
    # Sort indices by coupling strength
    def sort_by_coupling(indices):
        if len(indices) == 0:
            return indices
        outgoing = np.sum(np.abs(coupling_normalized[np.ix_(indices, range(n_target))]), axis=1)
        return [indices[i] for i in np.argsort(outgoing)[::-1]]
    
    acx_indices_sorted = sort_by_coupling(acx_indices)
    ofc_indices_sorted = sort_by_coupling(ofc_indices)
    region_order = acx_indices_sorted + ofc_indices_sorted
    
    # Plot connectivity matrix
    fig1 = plot_connectivity_matrix(
        coupling_normalized, acx_indices_sorted, ofc_indices_sorted, region_order
    )
    plt.show()
    
    # Get submatrices
    acx_to_acx = coupling_normalized[np.ix_(acx_indices_sorted, acx_indices_sorted)] if len(acx_indices_sorted) > 0 else np.array([])
    acx_to_ofc = coupling_normalized[np.ix_(acx_indices_sorted, ofc_indices_sorted)] if len(acx_indices_sorted) > 0 and len(ofc_indices_sorted) > 0 else np.array([])
    ofc_to_acx = coupling_normalized[np.ix_(ofc_indices_sorted, acx_indices_sorted)] if len(ofc_indices_sorted) > 0 and len(acx_indices_sorted) > 0 else np.array([])
    ofc_to_ofc = coupling_normalized[np.ix_(ofc_indices_sorted, ofc_indices_sorted)] if len(ofc_indices_sorted) > 0 else np.array([])
    
    # Plot breakdown
    fig2 = plot_connectivity_breakdown(
        acx_to_acx, acx_to_ofc, ofc_to_acx, ofc_to_ofc,
        acx_indices_sorted, ofc_indices_sorted
    )
    plt.show()

#%%
# ============================================================================
# 10. FEATURE CONTRIBUTION ANALYSIS
# ============================================================================
print("\n" + "=" * 60)
print("ANALYZING FEATURE CONTRIBUTIONS BY REGION")
print("=" * 60)

coef_matrix = np.asarray(model_pop.coef_)

# Group features
feature_names_all = list(X_ep_pop.columns)
non_history_idx = [i for i in range(len(feature_names_all)) if i not in history_feature_indices]

feature_groups = {
    'tone_onset': [], 'licks': [], 'outcome_onset': [],
    'stimulus': [], 'category': [], 'outcome': [], 'previous_outcome': []
}

for idx in non_history_idx:
    feat_name = feature_names_all[idx]
    if feat_name.startswith('tone_onset_basis'):
        feature_groups['tone_onset'].append(idx)
    elif feat_name.startswith('licks_basis'):
        feature_groups['licks'].append(idx)
    elif feat_name.startswith('outcome_onset_basis'):
        feature_groups['outcome_onset'].append(idx)
    elif 'stimulus' in feat_name.lower():
        feature_groups['stimulus'].append(idx)
    elif 'category_ID' in feat_name:
        feature_groups['category'].append(idx)
    elif 'outcome_ID' in feat_name:
        feature_groups['outcome'].append(idx)
    elif 'previous_outcome' in feat_name:
        feature_groups['previous_outcome'].append(idx)

# Create grouped coefficients
grouped_feature_names = []
grouped_coef_indices = []
for group_name, indices in feature_groups.items():
    if len(indices) > 0:
        grouped_feature_names.append(group_name)
        grouped_coef_indices.append(indices)

n_neurons = coef_matrix.shape[1]
n_groups = len(grouped_feature_names)
grouped_coefs = np.zeros((n_groups, n_neurons))

for group_idx, indices in enumerate(grouped_coef_indices):
    grouped_coefs[group_idx, :] = np.sum(np.abs(coef_matrix[indices, :]), axis=0)

# Normalize to percentages
grouped_coefs_pct = np.zeros_like(grouped_coefs)
for unit_idx in range(n_neurons):
    unit_total = np.sum(grouped_coefs[:, unit_idx])
    if unit_total > 0:
        grouped_coefs_pct[:, unit_idx] = (grouped_coefs[:, unit_idx] / unit_total) * 100

# Save Plotly plot
script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, "partial_contributions_by_region.html")
plot_contributions_by_region_plotly(
    grouped_coefs_pct, grouped_feature_names, acx_indices, ofc_indices, output_path
)

#%%
# ============================================================================
# 11. TIME TO REACT ANALYSIS
# ============================================================================
print("\n" + "=" * 60)
print("ANALYZING TIME TO REACT")
print("=" * 60)

import re

# Evaluate basis functions on grid to get time values
time_event, basis_kernels_event = basis_events.evaluate_on_grid(event_window_bins)
time_event_sec = time_event * EVENT_WINDOW_SEC

time_cat, basis_kernels_cat = basis_categorical.evaluate_on_grid(acausal_window_bins)
acausal_total_sec = ACAUSAL_BEFORE_SEC + ACAUSAL_AFTER_SEC
time_cat_sec = time_cat * acausal_total_sec - ACAUSAL_BEFORE_SEC

# Find the peak time for each basis function
basis_peak_times_event = np.zeros(N_BASIS_FUNCS)
for i in range(N_BASIS_FUNCS):
    peak_idx = np.argmax(np.abs(basis_kernels_event[:, i]))
    basis_peak_times_event[i] = time_event_sec[peak_idx]

basis_peak_times_cat = np.zeros(N_BASIS_FUNCS)
for i in range(N_BASIS_FUNCS):
    peak_idx = np.argmax(np.abs(basis_kernels_cat[:, i]))
    basis_peak_times_cat[i] = time_cat_sec[peak_idx]

# Extract time to react for each feature group and neuron
time_to_react = {}
feature_names_all = list(X_ep_pop.columns)

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
            neuron_times[neuron_idx] = np.nan
            continue
        
        # Find the index of the maximum absolute coefficient
        max_idx_in_group = np.argmax(abs_coefs)
        feature_idx_in_group = indices[max_idx_in_group]
        feature_name = feature_names_all[feature_idx_in_group]
        
        # Extract basis index from feature name
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

# Plot time to react
output_path_time = os.path.join(script_dir, "time_to_react_by_region.png")
plot_time_to_react_boxplot(time_to_react, acx_indices, ofc_indices, output_path_time)
plt.show()

#%%
# ============================================================================
# 12. PERI-EVENT RASTER AND PETH FOR HIGH COEFFICIENT UNITS
# ============================================================================
print("\n" + "=" * 60)
print("PLOTTING PERI-EVENT RASTER AND PETH")
print("=" * 60)

# Temporal features to analyze
temporal_features_to_plot = ['tone_onset', 'licks', 'outcome_onset']

# Get coefficient matrix from population GLM
coef_matrix_pop = np.asarray(model_pop.coef_)

# For each temporal feature, find top units and plot
for temp_feat in temporal_features_to_plot:
    print(f"\nFinding units with high coefficients for {temp_feat}...")
    
    # Find top units for this feature
    top_units = find_units_with_high_temporal_coefficients(
        coef_matrix_pop,
        list(X_ep_pop.columns),
        temp_feat,
        population_ids,
        top_n=3
    )
    
    if len(top_units) == 0:
        print(f"  No units found for {temp_feat}")
        continue
    
    print(f"  Top units for {temp_feat}:")
    for unit_id, coef_val in top_units:
        print(f"    Unit {unit_id}: max |coef| = {coef_val:.4f}")
    
    # Get event times for this feature
    if temp_feat == 'tone_onset':
        event_times = tone_onset
    elif temp_feat == 'licks':
        event_times = licks
    elif temp_feat == 'outcome_onset':
        event_times = outcome_time
    else:
        continue
    
    # Filter out invalid event times
    event_times = event_times[~np.isnan(event_times)]
    event_times = event_times[event_times > 0]
    
    if len(event_times) == 0:
        print(f"  No valid event times for {temp_feat}")
        continue
    
    # Get category and outcome information for matching events
    # data_df was created in create_categorical_features and contains category_ID and outcome_ID
    category_array = None
    outcome_array = None
    
    if temp_feat == 'tone_onset':
        # For tone_onset, we can directly use the category and outcome from data_df
        # data_df has the same length as tone_onset
        category_array = data_df['category_ID'].values if 'data_df' in locals() else None
        outcome_array = data_df['outcome_ID'].values if 'data_df' in locals() else None
    elif temp_feat in ['licks', 'outcome_onset']:
        # For licks and outcome_onset, we need to match to nearest tone_onset
        # The plotting function will handle this matching
        category_array = data_df['category_ID'].values if 'data_df' in locals() else None
        outcome_array = data_df['outcome_ID'].values if 'data_df' in locals() else None
    
    # Plot for each top unit
    for unit_id, coef_val in top_units:
        # Check if unit is in the filtered spikes (not just population)
        if unit_id in spikes:
            print(f"  Plotting peri-event raster and PETH for Unit {unit_id}...")
            fig = plot_perievent_raster_peth(
                spikes,
                event_times,
                unit_id,
                temp_feat,
                window=(EPOCH_START, EPOCH_END),
                bin_size=BIN_SIZE,
                tone_onset=tone_onset if temp_feat != 'tone_onset' else None,
                category=category_array,
                outcome=outcome_array
            )
            
            if fig is not None:
                # Save figure
                output_path_peth = os.path.join(
                    script_dir, 
                    f"perievent_{temp_feat}_unit{unit_id}.png"
                )
                fig.savefig(output_path_peth, dpi=300, bbox_inches='tight')
                print(f"    Saved to: {output_path_peth}")
                plt.show()
        else:
            print(f"  Unit {unit_id} not found in filtered spikes")


print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)

#%%

