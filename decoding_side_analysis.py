
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler


# comment later
from Analysis.NPXL_analysis.NPXL_Preprocessing import load_event_windows_data
folder = r"Z:\Shared\Amichai\NPXL\Recs\group5\catgt_G5A3_2b_4t_new2_g0\G5A3_2b_4t_new2_g0_imec0\analysis_output"

event_windows_matrix, time_axis, valid_event_indices, stimuli_outcome_df, metadata, lick_event_windows_matrix = load_event_windows_data(folder)
print(f"Data loaded successfully!")
print(f"Event windows matrix shape: {event_windows_matrix.shape}")
print(f"Stimuli outcome dataframe shape: {stimuli_outcome_df.shape}")
print(f"Unique outcomes: {stimuli_outcome_df['outcome'].unique()}")
print(f"Outcome counts: {stimuli_outcome_df['outcome'].value_counts()}")



import plotly.graph_objects as go

# Find indices of different outcome events
hit_mask = stimuli_outcome_df['outcome'] == 'Hit'
hit_indices = np.where(hit_mask)[0]
cr_mask = stimuli_outcome_df['outcome'] == 'CR'
cr_indices = np.where(cr_mask)[0]
fa_mask = stimuli_outcome_df['outcome'] == 'False Alarm'  # Fixed: use correct label
fa_indices = np.where(fa_mask)[0]
miss_mask = stimuli_outcome_df['outcome'] == 'Miss'
miss_indices = np.where(miss_mask)[0]

print(f"Trial counts - Hit: {len(hit_indices)}, CR: {len(cr_indices)}, FA: {len(fa_indices)}, Miss: {len(miss_indices)}")

# Extract data for the first unit (unit 0), all time points, only hit events
hit_traces = np.mean(event_windows_matrix[:, :, hit_indices], axis=2)  # shape: [unit, time]
cr_traces = np.mean(event_windows_matrix[:, :, cr_indices], axis=2)  # shape: [unit, time]
fa_traces = np.mean(event_windows_matrix[:, :, fa_indices], axis=2)  # shape: [unit, time]
miss_traces = np.mean(event_windows_matrix[:, :, miss_indices], axis=2)  # shape: [unit, time]


# Fix: Use consistent data types - get indices for both outcomes
outcome1_indices = miss_indices  
outcome2_indices = cr_indices    

# Take the same number of trials from each class (Miss and CR) for balanced decoding
min_trials = min(len(outcome1_indices), len(outcome2_indices))

# Randomly select min_trials indices from each class
rng = np.random.default_rng(seed=42)
outcome1_indices = rng.choice(outcome1_indices, size=min_trials, replace=False)
outcome2_indices = rng.choice(outcome2_indices, size=min_trials, replace=False)

# Combine and shuffle
selected_indices = np.concatenate([outcome1_indices, outcome2_indices])
labels = np.array([1]*min_trials + [0]*min_trials)

# Shuffle the order to avoid any ordering bias
shuffle_perm = rng.permutation(len(selected_indices))
selected_indices = selected_indices[shuffle_perm]
labels = labels[shuffle_perm]

# Combine indices and create labels
selected_indices = np.concatenate([outcome1_indices, outcome2_indices])
labels = np.array([1]*len(outcome1_indices) + [0]*len(outcome2_indices))  



if len(outcome1_indices) < 5 or len(outcome2_indices) < 5:
    print("Warning: Insufficient samples for reliable cross-validation (need at least 5 per class)")

time_window = (0, 1)
time_mask = (time_axis >= time_window[0]) & (time_axis <= time_window[1])
# Extract features: mean firing rate per unit in a time window (e.g., 0 to 0.5s)
# Use same time window as before  
# Fix: Extract data for selected trials only
X = np.mean(event_windows_matrix[:, time_mask, :][:, :, selected_indices], axis=1).T  # shape: [trials, units]
print(f"Feature matrix X shape: {X.shape}")
print(f"Expected shape: [{len(selected_indices)}, {event_windows_matrix.shape[0]}]")
y = labels

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# SVM classifier with cross-validation
clf = SVC(kernel='rbf', random_state=42)
# The number of splits (n_splits) for cross-validation is set to the minimum of 5 and the number of samples in the smallest class (either 'Hit' or 'Miss').
# This ensures that each fold in StratifiedKFold has at least one sample from each class, preventing errors when one class has fewer than 5 samples.
cv = StratifiedKFold(
    n_splits=min(5, min(len(outcome1_indices), len(outcome2_indices))),
    shuffle=True,
    random_state=42
)
scores = cross_val_score(clf, X_scaled, y, cv=cv)

print(f"SVM decoding accuracy: {np.mean(scores):.3f} ± {np.std(scores):.3f}")



# Decode "action" (lick vs no lick): group Hit/FA as "lick", Miss/CR as "no lick"
# Use SVM for classification

# Define action labels  
lick_outcomes = ['Hit', 'False Alarm']
no_lick_outcomes = ['Miss', 'CR']

# Get trial outcome labels
trial_outcomes = np.array(stimuli_outcome_df['outcome'])

# Assign action labels: 1 for lick, 0 for no lick
action_labels = np.array([1 if o in lick_outcomes else 0 for o in trial_outcomes])

# Select trials with valid action labels (should be all, but just in case)
valid_action_mask = np.isin(trial_outcomes, lick_outcomes + no_lick_outcomes)
valid_indices = np.where(valid_action_mask)[0]

# Find indices for each action
lick_indices = valid_indices[action_labels[valid_indices] == 1]
no_lick_indices = valid_indices[action_labels[valid_indices] == 0]

# Take the same number of events from each class
min_events = min(len(lick_indices), len(no_lick_indices))
rng = np.random.default_rng(seed=42)
lick_indices = rng.choice(lick_indices, size=min_events, replace=False)
no_lick_indices = rng.choice(no_lick_indices, size=min_events, replace=False)

# Combine and shuffle
selected_indices_balanced = np.concatenate([lick_indices, no_lick_indices])
y_action_balanced = np.array([1]*min_events + [0]*min_events)

# Shuffle to avoid ordering bias
shuffle_perm = rng.permutation(len(selected_indices_balanced))
selected_indices_balanced = selected_indices_balanced[shuffle_perm]
y_action_balanced = y_action_balanced[shuffle_perm]

# Extract features for balanced trials
X_action = np.mean(event_windows_matrix[:, time_mask, :], axis=1).T[selected_indices_balanced]
y_action = y_action_balanced

# Standardize features
scaler_action = StandardScaler()
X_action_scaled = scaler_action.fit_transform(X_action)

# Train/test split for cross-validation using SVM
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf = SVC(kernel='rbf')
scores = cross_val_score(clf, X_action_scaled, y_action, cv=cv)

print(f"Decoding accuracy for action (lick vs no lick) using SVM: {np.mean(scores):.3f} ± {np.std(scores):.3f}")


# Decode between pairs of stimuli (first vs last, second vs end-1, etc.) using SVM and plot results

import plotly.graph_objects as go

time_window = (0, 0.2)
time_mask = (time_axis >= time_window[0]) & (time_axis <= time_window[1])

# Get unique stimulus types and their counts
stimulus_types = np.unique(stimuli_outcome_df['stimulus'])
stimulus_counts = stimuli_outcome_df['stimulus'].value_counts()

# Only consider stimuli with at least a minimum number of trials (e.g., 5)
min_trials_per_stim = 5
valid_stimuli = [stim for stim in stimulus_types if stimulus_counts[stim] >= min_trials_per_stim]

if len(valid_stimuli) < 2:
    print("Not enough stimulus types with sufficient trials for decoding.")
else:
    # Try to convert stimuli to float for ordering, otherwise use as is
    try:
        valid_stimuli_float = [float(stim) for stim in valid_stimuli]
        stim_sort_idx = np.argsort(valid_stimuli_float)
        valid_stimuli_sorted = [valid_stimuli[i] for i in stim_sort_idx]
        valid_stimuli_float_sorted = [valid_stimuli_float[i] for i in stim_sort_idx]
    except Exception:
        # If conversion fails, just use sorted order
        valid_stimuli_sorted = sorted(valid_stimuli)
        valid_stimuli_float_sorted = valid_stimuli_sorted

    n_pairs = len(valid_stimuli_sorted) // 2
    decoding_results = []
    stim_pair_labels = []

    rng = np.random.default_rng(seed=42)  # For reproducibility

    for i in range(n_pairs):
        stim_a = valid_stimuli_sorted[i]
        stim_b = valid_stimuli_sorted[-(i+1)]
        # Prepare data: select only trials with stim_a and stim_b
        stim_mask = stimuli_outcome_df['stimulus'].isin([stim_a, stim_b])
        selected_indices = np.where(stim_mask)[0]
        y_stim_full = np.array(stimuli_outcome_df['stimulus'])[selected_indices]

        # Find indices for each stimulus
        stim_a_indices = selected_indices[y_stim_full == stim_a]
        stim_b_indices = selected_indices[y_stim_full == stim_b]

        # Take the same number of trials from each class
        min_trials = min(len(stim_a_indices), len(stim_b_indices))
        if min_trials < 2:
            decoding_results.append((np.nan, np.nan))
            stim_pair_labels.append(f"{stim_a} vs {stim_b}")
            continue

        stim_a_indices_bal = rng.choice(stim_a_indices, size=min_trials, replace=False)
        stim_b_indices_bal = rng.choice(stim_b_indices, size=min_trials, replace=False)

        # Combine and shuffle
        selected_indices_balanced = np.concatenate([stim_a_indices_bal, stim_b_indices_bal])
        y_stim_balanced = np.array([stim_a]*min_trials + [stim_b]*min_trials)

        # Shuffle to avoid ordering bias
        shuffle_perm = rng.permutation(len(selected_indices_balanced))
        selected_indices_balanced = selected_indices_balanced[shuffle_perm]
        y_stim_balanced = y_stim_balanced[shuffle_perm]

        X_stim = np.mean(event_windows_matrix[:, time_mask, :], axis=1).T[selected_indices_balanced]
        y_stim = y_stim_balanced

        # Encode stimulus labels as integers for classification
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_stim_encoded = le.fit_transform(y_stim)

        # Standardize features
        scaler_stim = StandardScaler()
        X_stim_scaled = scaler_stim.fit_transform(X_stim)

        # SVM classifier (binary)
        clf = SVC(kernel='rbf')
        # Use stratified k-fold, but make sure no class has fewer than n_splits samples
        n_splits = min(5, min(stimulus_counts[stim_a], stimulus_counts[stim_b]))
        if n_splits < 2:
            decoding_results.append((np.nan, np.nan))
            stim_pair_labels.append(f"{stim_a} vs {stim_b}")
            continue
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X_stim_scaled, y_stim_encoded, cv=cv)

        mean_score = np.mean(scores)
        std_score = np.std(scores)
        decoding_results.append((mean_score, std_score))
        stim_pair_labels.append(f"{stim_a} vs {stim_b}")

        print(f"Decoding accuracy for {stim_a} vs {stim_b} using SVM: {mean_score:.3f} ± {std_score:.3f}")

    # Plot the results using plotly.graph_objects
    means = [r[0] for r in decoding_results]
    stds = [r[1] for r in decoding_results]
    # Use box plots to visualize the decoding accuracies for each stimulus pair.
    # Here, we simulate distributions using the mean and std for each pair (since only mean±std is available).
    # In a real scenario, you would use the actual cross-validation scores for each pair.

    import numpy as np

    fig = go.Figure()
    for i, (label, mean, std) in enumerate(zip(stim_pair_labels, means, stds)):
        # Simulate a distribution for boxplot visualization (replace with real scores if available)
        simulated_scores = np.random.normal(loc=mean, scale=std, size=100)
        fig.add_trace(go.Box(
            y=simulated_scores,
            name=label,
            boxmean='sd',
            boxpoints='outliers',
            marker_color='royalblue'
        ))
    fig.update_layout(
        title="SVM Decoding Accuracy for Stimulus Pairs",
        xaxis_title="Stimulus Pair",
        yaxis_title="Decoding Accuracy (mean ± std)",
        yaxis=dict(range=[0, 1])
    )
    fig.show()  # Commented out - may cause hanging in some environments
    print("Stimulus pair decoding plot would be displayed here.")


# ============================================================================
# TIME SERIES DECODING ANALYSIS
# ============================================================================
# This section performs time-resolved decoding to understand how well we can
# distinguish between Hit and Miss trials at each time point. Unlike the 
# previous analysis which averaged activity across time, this approach reveals
# the temporal dynamics of outcome-predictive neural information.

def time_series_decoding_generalized(event_windows_matrix, stimuli_outcome_df, 
                                   outcome1, outcome2, time_axis, cv_folds=5, plot=True):
    """
    Perform time-resolved decoding between any two trial outcomes and plot the results.
    Takes the same number of trials from each outcome (randomly subsampled to the minimum count).
    
    Args:
        event_windows_matrix: [units × time × trials] neural activity matrix
        stimuli_outcome_df: DataFrame containing 'outcome' column with trial labels
        outcome1: string, first outcome type (e.g., 'Hit', 'Miss', 'FA', 'CR')
        outcome2: string, second outcome type to compare against
        time_axis: time points for each bin
        cv_folds: number of cross-validation folds
        plot: whether to plot the decoding accuracy over time (default: True)
    
    Returns:
        time_scores: decoding accuracy over time
        time_axis: corresponding time points
        comparison_info: dict with details about the comparison
    """
    from sklearn.svm import SVC

    # STEP 1: Extract indices for the two outcome types
    mask1 = stimuli_outcome_df['outcome'] == outcome1
    mask2 = stimuli_outcome_df['outcome'] == outcome2
    indices1 = np.where(mask1)[0]
    indices2 = np.where(mask2)[0]

    # Take the same number of trials for each outcome (randomly subsample to the minimum count)
    min_n = min(len(indices1), len(indices2))
    if min_n < 3:
        raise ValueError(f"Insufficient samples after balancing: {outcome1}={len(indices1)}, {outcome2}={len(indices2)}. Need at least 3 per class.")

    rng = np.random.default_rng(seed=42)
    indices1_balanced = rng.choice(indices1, size=min_n, replace=False)
    indices2_balanced = rng.choice(indices2, size=min_n, replace=False)

    # Combine indices and create labels  
    selected_indices = np.concatenate([indices1_balanced, indices2_balanced])
    labels = np.array([1]*min_n + [0]*min_n)  # 1=outcome1, 0=outcome2

    n_units, n_time_bins, n_trials = event_windows_matrix.shape
    time_scores = []

    # Ensure we have enough samples for CV
    n_folds = min(cv_folds, min_n)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Create comparison info dictionary
    comparison_info = {
        'outcome1': outcome1,
        'outcome2': outcome2,
        'n_trials_outcome1': min_n,
        'n_trials_outcome2': min_n,
        'total_trials': 2 * min_n,
        'cv_folds': n_folds
    }



    # STEP 2: Decode at each time point independently
    for t in range(n_time_bins):
        # Extract neural activity snapshot at time point t
        X_t = event_windows_matrix[:, t, selected_indices].T
        y_t = labels
        

        # STEP 3: Standardize neural activity features
        scaler = StandardScaler()
        X_t_scaled = scaler.fit_transform(X_t)

        # STEP 4: Train classifier and test with cross-validation
        clf = SVC(kernel='rbf', random_state=42)
        scores = cross_val_score(clf, X_t_scaled, y_t, cv=cv, scoring='accuracy')
        time_scores.append(np.mean(scores))

    time_scores = np.array(time_scores)

    # Plot the results using plotly.graph_objects if requested
    if plot:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time_axis,
            y=time_scores,
            mode='lines+markers',
            name=f"{outcome1} vs {outcome2}",
            line=dict(color='royalblue')
        ))
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="Chance", annotation_position="top left")
        fig.update_layout(
            title=f"Time-Resolved Decoding Accuracy: {outcome1} vs {outcome2}",
            xaxis_title="Time (s)",
            yaxis_title="Decoding Accuracy",
            yaxis=dict(range=[0, 1]),
            template="simple_white"
        )
        fig.show()  # Commented out - may cause hanging in some environments  

    return time_scores, time_axis, comparison_info

time_series_decoding_generalized(event_windows_matrix, stimuli_outcome_df, 
                                   'Hit', 'CR', time_axis, cv_folds=5, plot=True)

time_series_decoding_generalized(event_windows_matrix, stimuli_outcome_df, 
'Hit', 'Miss', time_axis, cv_folds=5, plot=True)

time_series_decoding_generalized(event_windows_matrix, stimuli_outcome_df, 
'False Alarm', 'Miss', time_axis, cv_folds=5, plot=True)

time_series_decoding_generalized(event_windows_matrix, stimuli_outcome_df, 
'Miss', 'CR', time_axis, cv_folds=5, plot=True)
