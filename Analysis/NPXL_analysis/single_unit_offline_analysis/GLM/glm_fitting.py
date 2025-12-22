"""
Generalized Linear Model (GLM) fitting for neural data analysis.

This module implements GLM fitting for single units with behavioral predictors:
1. Cue Onset (tone/stimulus presentation)
2. Stimulus (frequency/category)
3. Category (Go vs NoGo trial type)
4. First lick timing
5. In-trial lick count
6. Reward delivery
7. Punishment delivery
8. Previous trial outcome
9. Trial states (wait, play_sound, reinforcement_delay, response_window)
"""
import os
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any, List
from scipy import stats
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score


def extract_lick_features(licks_list: List, states_list: List, 
                          time_bins: np.ndarray, bin_size: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract first lick timing and in-trial lick count from behavioral data.
    
    Parameters:
    -----------
    licks_list : list
        List of lick timestamps for each trial
    states_list : list
        List of state dictionaries for each trial
    time_bins : np.ndarray
        Time axis for binning (relative to cue onset)
    bin_size : float
        Size of time bins in seconds
    
    Returns:
    --------
    tuple : (first_lick_bins, lick_counts)
        - first_lick_bins: bin index of first lick per trial (NaN if no licks)
        - lick_counts: total number of licks per trial
    """
    n_trials = len(licks_list)
    first_lick_bins = np.full(n_trials, np.nan)
    lick_counts = np.zeros(n_trials)
    
    for i, licks in enumerate(licks_list):
        if licks is None or len(licks) == 0:
            continue
        
        # Handle various lick data formats
        if isinstance(licks, (list, np.ndarray)):
            lick_array = np.array(licks).flatten()
            lick_array = lick_array[~np.isnan(lick_array)]
            
            if len(lick_array) > 0:
                lick_counts[i] = len(lick_array)
                # Convert first lick to bin index (relative to cue onset at t=0)
                first_lick_time = lick_array[0]
                first_lick_bins[i] = np.argmin(np.abs(time_bins - first_lick_time))
    
    return first_lick_bins, lick_counts


def create_raised_cosine_basis(
    time_axis: np.ndarray,
    n_basis: int = 10,
    window: Tuple[float, float] = (0.0, 2.0),
    peak_spacing: Optional[float] = None,
) -> np.ndarray:
    """
    Create raised cosine basis functions for temporal convolution.
    
    Raised cosine basis functions provide smooth, localized temporal kernels.
    Each basis function is a raised cosine (cos^2) centered at different time points.
    
    Parameters:
    -----------
    time_axis : np.ndarray
        Time axis in seconds
    n_basis : int
        Number of basis functions
    window : tuple
        Time window (start, end) for basis functions
    peak_spacing : float, optional
        Spacing between basis function peaks. If None, automatically determined.
    
    Returns:
    --------
    np.ndarray
        Basis functions [n_time_bins × n_basis]
    """
    n_time = len(time_axis)
    basis_functions = np.zeros((n_time, n_basis))
    
    # Determine peak spacing
    if peak_spacing is None:
        # Space peaks evenly across window with some overlap
        window_duration = window[1] - window[0]
        if n_basis > 1:
            # Overlap basis functions for smooth coverage
            peak_spacing = window_duration / (n_basis - 1)
        else:
            peak_spacing = window_duration
    
    # Width of each basis function (controls overlap)
    width = 2 * peak_spacing
    
    for i in range(n_basis):
        # Peak position
        peak_time = window[0] + i * peak_spacing
        
        # Compute distance from peak
        t_centered = time_axis - peak_time
        
        # Raised cosine: cos^2(π * t / (2*width)) for |t| < width, else 0
        # Formula: 0.5 * (1 + cos(π * t / width)) for |t| < width
        t_scaled = t_centered / width
        mask = np.abs(t_scaled) < 1.0
        
        # Raised cosine: cos^2(π * t_scaled / 2)
        basis_functions[mask, i] = np.cos(np.pi * t_scaled[mask] / 2) ** 2
    
    # Normalize each basis function to have unit integral (optional, for interpretability)
    # This makes coefficients more comparable across basis functions
    for i in range(n_basis):
        basis_integral = np.trapz(basis_functions[:, i], time_axis)
        if basis_integral > 1e-10:
            basis_functions[:, i] = basis_functions[:, i] / basis_integral
    
    return basis_functions


def convolve_predictor_with_basis(
    predictor_values: np.ndarray,
    basis_functions: np.ndarray,
    event_times: Optional[np.ndarray] = None,
    time_axis: Optional[np.ndarray] = None,
    time_window: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    Convolve predictor values with temporal basis functions.
    
    Creates temporal features by integrating predictor * basis_function over time.
    For static predictors: predictor_value * integral(basis_function_i)
    For event-based predictors: basis_function_i evaluated at event time
    
    Parameters:
    -----------
    predictor_values : np.ndarray
        Predictor values [n_trials]
    basis_functions : np.ndarray
        Basis functions [n_time_bins × n_basis]
    event_times : np.ndarray, optional
        Event times for each trial (for event-based predictors)
    time_axis : np.ndarray, optional
        Time axis for event timing
    time_window : tuple, optional
        Time window for integration
    
    Returns:
    --------
    np.ndarray
        Convolved features [n_trials × n_basis]
        Each column is: predictor_value * integral(basis_function_i) or basis_function_i(event_time)
    """
    n_trials = len(predictor_values)
    n_time, n_basis = basis_functions.shape
    
    # Result: [n_trials × n_basis]
    convolved = np.zeros((n_trials, n_basis))
    
    # Get time window mask if provided
    if time_window is not None and time_axis is not None:
        time_mask = (time_axis >= time_window[0]) & (time_axis <= time_window[1])
    else:
        time_mask = np.ones(n_time, dtype=bool)
    
    # Compute bin size for integration
    if time_axis is not None and len(time_axis) > 1:
        bin_size = np.mean(np.diff(time_axis))
    else:
        bin_size = 1.0
    
    for trial_idx in range(n_trials):
        pred_val = predictor_values[trial_idx]
        
        if event_times is not None and time_axis is not None:
            # Event-based predictor: evaluate basis functions at event time
            event_time = event_times[trial_idx]
            if not np.isnan(event_time) and pred_val > 0:
                # Find closest time bin to event
                event_bin = np.argmin(np.abs(time_axis - event_time))
                
                # Evaluate each basis function at event time (or nearby)
                for basis_idx in range(n_basis):
                    # Use value at event bin, or interpolate
                    if 0 <= event_bin < n_time:
                        basis_val = basis_functions[event_bin, basis_idx]
                    else:
                        basis_val = 0.0
                    convolved[trial_idx, basis_idx] = pred_val * basis_val
            # If no event, leave as zeros
        else:
            # Static predictor: integrate basis functions over time window
            # Feature = predictor_value * integral(basis_function_i)
            for basis_idx in range(n_basis):
                # Integrate basis function over time window
                basis_integral = np.sum(basis_functions[time_mask, basis_idx]) * bin_size
                convolved[trial_idx, basis_idx] = pred_val * basis_integral
    
    return convolved


def extract_spike_history_features(
    unit_data: np.ndarray,
    time_bins: np.ndarray,
    bin_size: float,
    history_window: Tuple[float, float] = (0.001, 0.1),
) -> np.ndarray:
    """
    Extract spike history features for GLM.
    
    For each time bin, computes the firing rate in previous time bins
    within the history window. This captures refractoriness and bursting.
    
    Parameters:
    -----------
    unit_data : np.ndarray
        Neural data [time_bins × trials]
    time_bins : np.ndarray
        Time axis for data
    bin_size : float
        Time bin size in seconds
    history_window : tuple
        Time window for spike history (start, end) in seconds
        Default: (0.001, 0.1) = 1ms to 100ms
    
    Returns:
    --------
    np.ndarray
        Spike history features [time_bins × trials]
        Each value is the average firing rate in the history window
    """
    n_time, n_trials = unit_data.shape
    history_features = np.zeros((n_time, n_trials))
    
    # Convert history window to bin indices
    history_start_bins = int(history_window[0] / bin_size)
    history_end_bins = int(history_window[1] / bin_size)
    
    for t in range(n_time):
        # Look back from current time bin
        start_idx = max(0, t - history_end_bins)
        end_idx = max(0, t - history_start_bins)
        
        if end_idx > start_idx:
            # Average firing rate in history window
            history_features[t, :] = np.mean(unit_data[start_idx:end_idx, :], axis=0)
    
    return history_features


def extract_trial_state_features(states_list: List, time_bins: np.ndarray,
                                 bin_size: float) -> Dict[str, np.ndarray]:
    """
    Extract trial state timing features from behavioral data.
    
    Parameters:
    -----------
    states_list : list
        List of state dictionaries for each trial
    time_bins : np.ndarray
        Time axis for binning
    bin_size : float
        Size of time bins in seconds
    
    Returns:
    --------
    dict : Dictionary with state timing arrays
        - 'wait_duration': Duration of wait state (seconds)
        - 'sound_duration': Duration of sound presentation (seconds)
        - 'reinforcement_delay': Duration of reinforcement delay (seconds)
        - 'response_window': Duration of response window (seconds)
    """
    n_trials = len(states_list)
    state_features = {
        'wait_duration': np.zeros(n_trials),
        'sound_duration': np.zeros(n_trials),
        'reinforcement_delay': np.zeros(n_trials),
        'response_window': np.zeros(n_trials),
    }
    
    for i, states in enumerate(states_list):
        if states is None or not isinstance(states, dict):
            continue
        
        # Extract state durations if available
        # States may contain timing information for different trial phases
        for state_name, key in [
            ('WaitForResponse', 'wait_duration'),
            ('PlaySound', 'sound_duration'),
            ('ReinforcementDelay', 'reinforcement_delay'),
            ('ResponseWindow', 'response_window'),
        ]:
            if state_name in states:
                state_times = states[state_name]
                if isinstance(state_times, (list, np.ndarray)) and len(state_times) >= 2:
                    # Duration is difference between start and end times
                    state_features[key][i] = float(state_times[1] - state_times[0])
    
    return state_features


def build_design_matrix(
    unit_data: np.ndarray,
    time_bins: np.ndarray,
    stimuli: np.ndarray,
    outcomes: np.ndarray,
    licks_list: Optional[List] = None,
    states_list: Optional[List] = None,
    bin_size: float = 0.005,
    time_window: Tuple[float, float] = (0.0, 2.0),
    category_boundaries: Optional[Tuple[float, float]] = None,
    normalize_response: Optional[str] = None,
    use_temporal_kernels: bool = True,
    n_basis: int = 10,
    # Predictor-specific kernel configurations
    stimulus_kernel_config: Optional[Dict[str, Any]] = None,
    reward_punishment_kernel_config: Optional[Dict[str, Any]] = None,
    lick_kernel_config: Optional[Dict[str, Any]] = None,
    spike_history_kernel_config: Optional[Dict[str, Any]] = None,
    include_cue_onset: bool = True,
    include_stimulus: bool = True,
    include_category: bool = True,
    include_licks: bool = True,
    include_reward_punishment: bool = True,
    include_prev_outcome: bool = True,
    include_trial_states: bool = True,
    include_spike_history: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build design matrix (X) and response vector (y) for GLM fitting.
    
    Parameters:
    -----------
    unit_data : np.ndarray
        Neural data [time_bins × trials]
    time_bins : np.ndarray
        Time axis for data
    stimuli : np.ndarray
        Stimulus values per trial
    outcomes : np.ndarray
        Trial outcomes per trial
    licks_list : list, optional
        List of lick timestamps per trial
    states_list : list, optional
        List of state dictionaries per trial
    bin_size : float
        Time bin size in seconds
    time_window : tuple
        Time window for averaging neural response (start, end) in seconds
    category_boundaries : tuple, optional
        Boundaries (low, high) for classifying Go vs NoGo
        If None, tries to infer from outcomes
    normalize_response : str, optional
        Normalization method for neural response:
        - 'zscore': Z-score normalization (mean=0, std=1)
        - 'minmax': Min-max normalization (0-1 range)
        - 'per_trial': Normalize each trial's time course before averaging
        - None: No normalization
    use_temporal_kernels : bool
        If True, use raised cosine basis functions for temporal convolution
        If False, use static predictors (one value per trial)
    n_basis : int
        Default number of raised cosine basis functions per predictor
    stimulus_kernel_config : dict, optional
        Configuration for stimulus kernel: {'n_basis': int, 'window': (start, end)}
        Default: {'n_basis': 8, 'window': (0.0, 0.6)} (0-600ms)
    reward_punishment_kernel_config : dict, optional
        Configuration for reward/punishment kernel: {'n_basis': int, 'window': (start, end)}
        Default: {'n_basis': 8, 'window': (0.0, 0.6)} (0-600ms)
    lick_kernel_config : dict, optional
        Configuration for lick kernel: {'n_basis': int, 'window': (start, end)}
        Default: {'n_basis': 7, 'window': (-0.2, 0.2)} (-200ms to +200ms, acausal)
    spike_history_kernel_config : dict, optional
        Configuration for spike history kernel: {'n_basis': int, 'window': (start, end)}
        Default: {'n_basis': 5, 'window': (0.001, 0.1)} (1ms to 100ms)
    include_* : bool
        Flags for which predictors to include
    
    Returns:
    --------
    tuple : (X, y, feature_names)
        - X: Design matrix [n_trials × n_features]
        - y: Neural response vector [n_trials]
        - feature_names: List of feature names
    """
    n_trials = unit_data.shape[1]
    
    # Extract neural response in specified time window
    start_idx = np.argmin(np.abs(time_bins - time_window[0]))
    end_idx = np.argmin(np.abs(time_bins - time_window[1]))
    y = np.mean(unit_data[start_idx:end_idx, :], axis=0)
    
    # Normalize response if requested
    if normalize_response == 'zscore':
        # Z-score normalization: (y - mean) / std
        y_mean = np.mean(y)
        y_std = np.std(y)
        if y_std > 1e-10:  # Avoid division by zero
            y = (y - y_mean) / y_std
            # Shift to make non-negative for Poisson (add minimum to all values)
            y_min = np.min(y)
            if y_min < 0:
                y = y - y_min + 1e-6  # Shift so minimum is slightly above zero
    elif normalize_response == 'minmax':
        # Min-max normalization: (y - min) / (max - min)
        y_min = np.min(y)
        y_max = np.max(y)
        if y_max - y_min > 1e-10:  # Avoid division by zero
            y = (y - y_min) / (y_max - y_min)
            # Ensure non-negative (should already be, but add small epsilon)
            y = y + 1e-6
    elif normalize_response == 'per_trial':
        # Normalize each trial's time course before averaging
        # This normalizes within-trial variability
        normalized_data = np.zeros_like(unit_data)
        for trial_idx in range(n_trials):
            trial_trace = unit_data[:, trial_idx]
            trial_mean = np.mean(trial_trace)
            trial_std = np.std(trial_trace)
            if trial_std > 1e-10:
                normalized_data[:, trial_idx] = (trial_trace - trial_mean) / trial_std
            else:
                normalized_data[:, trial_idx] = trial_trace - trial_mean
        
        # Now average the normalized traces
        y = np.mean(normalized_data[start_idx:end_idx, :], axis=0)
        # Shift to make non-negative for Poisson
        y_min = np.min(y)
        if y_min < 0:
            y = y - y_min + 1e-6
    
    feature_list = []
    feature_names = []
    
    # Set default kernel configurations (Go/No-Go GLM standard)
    if stimulus_kernel_config is None:
        stimulus_kernel_config = {'n_basis': 8, 'window': (0.0, 0.1)}  # 0-600ms
    if reward_punishment_kernel_config is None:
        reward_punishment_kernel_config = {'n_basis': 8, 'window': (0.0, 0.6)}  # 0-600ms
    if lick_kernel_config is None:
        lick_kernel_config = {'n_basis': 7, 'window': (-0.2, 0.2)}  # -200ms to +200ms (acausal)
    if spike_history_kernel_config is None:
        spike_history_kernel_config = {'n_basis': 5, 'window': (0.001, 0.1)}  # 1ms to 100ms
    
    # Create predictor-specific basis functions if using temporal kernels
    if use_temporal_kernels:
        # Stimulus/reward/punishment kernels (0-600ms)
        stimulus_basis = create_raised_cosine_basis(
            time_axis=time_bins,
            n_basis=stimulus_kernel_config['n_basis'],
            window=stimulus_kernel_config['window'],
        )
        reward_punishment_basis = create_raised_cosine_basis(
            time_axis=time_bins,
            n_basis=reward_punishment_kernel_config['n_basis'],
            window=reward_punishment_kernel_config['window'],
        )
        # Lick kernel (acausal: -200ms to +200ms)
        lick_basis = create_raised_cosine_basis(
            time_axis=time_bins,
            n_basis=lick_kernel_config['n_basis'],
            window=lick_kernel_config['window'],
        )
        # Spike history kernel (1ms to 100ms)
        spike_history_basis = create_raised_cosine_basis(
            time_axis=time_bins,
            n_basis=spike_history_kernel_config['n_basis'],
            window=spike_history_kernel_config['window'],
        )
        # Default basis for other predictors
        default_basis = create_raised_cosine_basis(
            time_axis=time_bins,
            n_basis=n_basis,
            window=time_window,
        )
    else:
        stimulus_basis = reward_punishment_basis = lick_basis = spike_history_basis = default_basis = None
    
    # 1. Cue Onset - Binary indicator (always 1 since data is aligned to cue)
    if include_cue_onset:
        if use_temporal_kernels:
            # Convolve with basis: impulse at t=0 (use stimulus kernel config)
            cue_onset_events = np.zeros(n_trials)  # Event at t=0 for all trials
            convolved = convolve_predictor_with_basis(
                predictor_values=np.ones(n_trials),
                basis_functions=stimulus_basis,
                event_times=cue_onset_events,  # All at t=0
                time_axis=time_bins,
                time_window=stimulus_kernel_config['window'],
            )
            # Add each basis function as a separate feature
            for basis_idx in range(stimulus_kernel_config['n_basis']):
                feature_list.append(convolved[:, basis_idx])
                feature_names.append(f'cue_onset_basis{basis_idx}')
        else:
            feature_list.append(np.ones(n_trials))
            feature_names.append('cue_onset')
    
    # 2. Stimulus - Can be continuous frequency or categorical
    if include_stimulus:
        stimulus_values = stimuli.astype(float)
        # Handle potential NaN values
        stimulus_values[np.isnan(stimulus_values)] = np.nanmean(stimulus_values)
        
        if use_temporal_kernels:
            # Convolve stimulus with basis functions (0-600ms window)
            convolved = convolve_predictor_with_basis(
                predictor_values=stimulus_values,
                basis_functions=stimulus_basis,
                time_axis=time_bins,
                time_window=stimulus_kernel_config['window'],
            )
            # Add each basis function as a separate feature
            for basis_idx in range(stimulus_kernel_config['n_basis']):
                feature_list.append(convolved[:, basis_idx])
                feature_names.append(f'stimulus_basis{basis_idx}')
        else:
            feature_list.append(stimulus_values)
            feature_names.append('stimulus')
    
    # 3. Category - Go vs NoGo (binary)
    if include_category:
        category = np.zeros(n_trials)
        
        # Try to infer category from outcomes if boundaries not provided
        if category_boundaries is None:
            # Use outcomes to determine category
            for i, outcome in enumerate(outcomes):
                if outcome in ['Hit', 'Miss']:
                    category[i] = 1  # Go trial
                elif outcome in ['CR', 'False Alarm', 'FA']:
                    category[i] = 0  # NoGo trial
        else:
            # Use stimulus boundaries to determine category
            low_boundary, high_boundary = category_boundaries
            stimulus_values = stimuli.astype(float)
            for i, stim in enumerate(stimulus_values):
                if np.isnan(stim):
                    # If stimulus is NaN, try to infer from outcome
                    if outcomes[i] in ['Hit', 'Miss']:
                        category[i] = 1
                    else:
                        category[i] = 0
                elif stim < low_boundary or stim > high_boundary:
                    category[i] = 1  # Go trial (outside boundaries)
                else:
                    category[i] = 0  # NoGo trial (inside boundaries)
        
        if use_temporal_kernels:
            # Convolve category with basis functions (use stimulus kernel config)
            convolved = convolve_predictor_with_basis(
                predictor_values=category,
                basis_functions=stimulus_basis,
                time_axis=time_bins,
                time_window=stimulus_kernel_config['window'],
            )
            # Add each basis function as a separate feature
            for basis_idx in range(stimulus_kernel_config['n_basis']):
                feature_list.append(convolved[:, basis_idx])
                feature_names.append(f'category_go_basis{basis_idx}')
        else:
            feature_list.append(category)
            feature_names.append('category_go')
    
    # 4 & 5. Lick features - using delta functions
    if include_licks and licks_list is not None:
        if use_temporal_kernels:
            # Create delta function representation for each trial
            # For each trial, create a time series with delta functions at lick times
            n_time = len(time_bins)
            lick_delta_matrix = np.zeros((n_trials, n_time))  # [trials × time]
            
            for trial_idx, licks in enumerate(licks_list):
                if licks is None or len(licks) == 0:
                    continue
                
                # Handle various lick data formats
                if isinstance(licks, (list, np.ndarray)):
                    lick_array = np.array(licks).flatten()
                    lick_array = lick_array[~np.isnan(lick_array)]
                    
                    # For each lick, place a delta function (impulse) at the lick time
                    for lick_time in lick_array:
                        # Find closest time bin
                        lick_bin = np.argmin(np.abs(time_bins - lick_time))
                        if 0 <= lick_bin < n_time:
                            # Delta function: impulse at lick time
                            # Normalize by bin_size to maintain proper scaling
                            lick_delta_matrix[trial_idx, lick_bin] += 1.0 / bin_size
            
            # Convolve delta functions with basis functions
            # For each basis function, compute the convolution for each trial
            for basis_idx in range(lick_kernel_config['n_basis']):
                basis_func = lick_basis[:, basis_idx]  # [time]
                
                # For each trial, convolve delta function with basis
                convolved_features = np.zeros(n_trials)
                for trial_idx in range(n_trials):
                    # Convolution: sum over time of delta(t) * basis(t - tau)
                    # Since delta is non-zero only at lick times, this becomes:
                    # sum over lick times of basis(t_lick)
                    trial_delta = lick_delta_matrix[trial_idx, :]  # [time]
                    
                    # Find where delta is non-zero (lick times)
                    lick_bins = np.where(trial_delta > 0)[0]
                    
                    if len(lick_bins) > 0:
                        # Sum basis function values at lick times
                        # Weight by delta magnitude (which is 1/bin_size)
                        convolved_features[trial_idx] = np.sum(
                            trial_delta[lick_bins] * basis_func[lick_bins]
                        ) * bin_size  # Multiply by bin_size to get proper integral
                    else:
                        convolved_features[trial_idx] = 0.0
                
                feature_list.append(convolved_features)
                feature_names.append(f'lick_delta_basis{basis_idx}')
        else:
            # Non-temporal: extract first lick and count
            first_lick_bins, lick_counts = extract_lick_features(
                licks_list, states_list, time_bins, bin_size
            )
            # Replace NaN in first_lick_bins with mean (for trials without licks)
            first_lick_bins[np.isnan(first_lick_bins)] = np.nanmean(first_lick_bins)
            feature_list.append(first_lick_bins)
            feature_names.append('first_lick')
            feature_list.append(lick_counts)
            feature_names.append('lick_count')
    
    # 6 & 7. Reward and Punishment
    if include_reward_punishment:
        reward = np.zeros(n_trials)
        punishment = np.zeros(n_trials)
        reward_times = np.full(n_trials, np.nan)  # Event times for reward
        punishment_times = np.full(n_trials, np.nan)  # Event times for punishment
        
        for i, outcome in enumerate(outcomes):
            if outcome == 'Hit':
                reward[i] = 1
                reward_times[i] = 0.5  # Approximate reward time (adjust based on task)
            elif outcome == 'False Alarm' or outcome == 'FA':
                punishment[i] = 1
                punishment_times[i] = 0.5  # Approximate punishment time
        
        if use_temporal_kernels:
            # Reward: event-based convolution (0-600ms window)
            convolved_reward = convolve_predictor_with_basis(
                predictor_values=reward,
                basis_functions=reward_punishment_basis,
                event_times=reward_times,
                time_axis=time_bins,
                time_window=reward_punishment_kernel_config['window'],
            )
            for basis_idx in range(reward_punishment_kernel_config['n_basis']):
                feature_list.append(convolved_reward[:, basis_idx])
                feature_names.append(f'reward_basis{basis_idx}')
            
            # Punishment: event-based convolution (0-600ms window)
            convolved_punishment = convolve_predictor_with_basis(
                predictor_values=punishment,
                basis_functions=reward_punishment_basis,
                event_times=punishment_times,
                time_axis=time_bins,
                time_window=reward_punishment_kernel_config['window'],
            )
            for basis_idx in range(reward_punishment_kernel_config['n_basis']):
                feature_list.append(convolved_punishment[:, basis_idx])
                feature_names.append(f'punishment_basis{basis_idx}')
        else:
            feature_list.append(reward)
            feature_names.append('reward')
            feature_list.append(punishment)
            feature_names.append('punishment')
    
    # 8. Previous trial outcome
    if include_prev_outcome:
        prev_reward = np.zeros(n_trials)
        prev_reward[1:] = (outcomes[:-1] == 'Hit').astype(float)
        
        if use_temporal_kernels:
            # Convolve previous reward with basis functions (use default basis)
            convolved = convolve_predictor_with_basis(
                predictor_values=prev_reward,
                basis_functions=default_basis,
                time_axis=time_bins,
                time_window=time_window,
            )
            for basis_idx in range(n_basis):
                feature_list.append(convolved[:, basis_idx])
                feature_names.append(f'prev_trial_reward_basis{basis_idx}')
        else:
            feature_list.append(prev_reward)
            feature_names.append('prev_trial_reward')
    
    # 9. Trial state durations
    if include_trial_states and states_list is not None:
        state_features = extract_trial_state_features(states_list, time_bins, bin_size)
        for state_name, state_values in state_features.items():
            feature_list.append(state_values)
            feature_names.append(state_name)
    
    # 10. Spike History - captures refractoriness and bursting
    if include_spike_history and use_temporal_kernels:
        # Extract spike history features for each trial
        spike_history_features = extract_spike_history_features(
            unit_data=unit_data,
            time_bins=time_bins,
            bin_size=bin_size,
            history_window=spike_history_kernel_config['window'],
        )
        
        # Integrate history with basis functions over the time window
        start_idx = np.argmin(np.abs(time_bins - time_window[0]))
        end_idx = np.argmin(np.abs(time_bins - time_window[1]))
        
        # For each basis function, compute integral of history * basis for all trials
        for basis_idx in range(spike_history_kernel_config['n_basis']):
            # Get basis function in time window
            basis_in_window = spike_history_basis[start_idx:end_idx, basis_idx]
            
            # For each trial, integrate: history(t) * basis(t) over time window
            history_features = np.zeros(n_trials)
            for trial_idx in range(n_trials):
                history_in_window = spike_history_features[start_idx:end_idx, trial_idx]
                integral = np.sum(basis_in_window * history_in_window) * bin_size
                history_features[trial_idx] = integral
            
            feature_list.append(history_features)
            feature_names.append(f'spike_history_basis{basis_idx}')
    
    # Validate all features have correct shape before stacking
    validated_features = []
    validated_names = []
    for i, (feat, name) in enumerate(zip(feature_list, feature_names)):
        feat_array = np.asarray(feat)
        # Ensure feature is 1D array with n_trials elements
        if feat_array.ndim == 0:
            # Scalar - broadcast to array
            feat_array = np.full(n_trials, float(feat_array))
        elif feat_array.ndim > 1:
            # Multi-dimensional - flatten if possible
            feat_array = feat_array.flatten()
        
        # Check length matches n_trials
        if len(feat_array) != n_trials:
            print(f"Warning: Feature '{name}' has length {len(feat_array)}, expected {n_trials}. Skipping.")
            continue
        
        validated_features.append(feat_array)
        validated_names.append(name)
    
    # Stack all features into design matrix
    if validated_features:
        X = np.column_stack(validated_features)
    else:
        X = np.zeros((n_trials, 0))
    
    return X, y, validated_names


def fit_glm_poisson(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    alpha: float = 1.0,
    standardize: bool = True,
    cv_folds: int = 5,
) -> Dict[str, Any]:
    """
    Fit a Poisson GLM to neural data.
    
    Parameters:
    -----------
    X : np.ndarray
        Design matrix [n_trials × n_features]
    y : np.ndarray
        Neural response [n_trials]
    feature_names : list
        Names of features
    alpha : float
        Regularization strength (L2 penalty)
    standardize : bool
        Whether to standardize features before fitting
    cv_folds : int
        Number of cross-validation folds
    
    Returns:
    --------
    dict : GLM results containing:
        - coefficients: Feature coefficients
        - intercept: Model intercept
        - feature_names: Names of features
        - deviance: Model deviance
        - pseudo_r2: McFadden's pseudo R²
        - cv_score: Cross-validated score (mean)
        - cv_std: Cross-validated score (std)
        - standardized: Whether features were standardized
    """
    # Handle edge cases
    if X.shape[0] < 10:
        return {
            'error': 'Insufficient trials for GLM fitting',
            'n_trials': X.shape[0]
        }
    
    # Ensure y is non-negative (required for Poisson)
    y = np.maximum(y, 0)
    
    # Check if unit has any activity (all zeros or very low)
    if np.sum(y) < 1e-6 or np.all(y < 1e-6):
        return {
            'error': 'Unit has no activity in time window',
            'n_trials': X.shape[0],
            'mean_firing_rate': float(np.mean(y))
        }
    
    # Standardize features if requested
    scaler = None
    if standardize and X.shape[1] > 0:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.copy()
    
    # Fit Poisson GLM
    try:
        model = PoissonRegressor(alpha=alpha, max_iter=300, tol=1e-4)
        model.fit(X_scaled, y)
        
        # Compute deviance
        y_pred = model.predict(X_scaled)
        # Ensure predictions are strictly positive (add small epsilon)
        y_pred = np.maximum(y_pred, 1e-10)
        
        # Poisson deviance
        deviance = 2 * np.sum(y * np.log((y + 1e-10) / (y_pred + 1e-10)) - (y - y_pred))
        
        # McFadden's pseudo R² (1 - deviance/null_deviance)
        null_model = PoissonRegressor(alpha=0, fit_intercept=True, max_iter=100)
        null_model.fit(np.zeros((len(y), 1)), y)
        y_null = null_model.predict(np.zeros((len(y), 1)))
        y_null = np.maximum(y_null, 1e-10)  # Ensure positive
        null_deviance = 2 * np.sum(y * np.log((y + 1e-10) / (y_null + 1e-10)) - (y - y_null))
        pseudo_r2 = 1 - (deviance / null_deviance) if null_deviance != 0 else 0
        
        # Cross-validation score with error handling
        try:
            # Custom scoring function that handles edge cases
            # Matches sklearn's neg_mean_poisson_deviance behavior
            def safe_poisson_scorer(y_true, y_pred):
                y_pred = np.maximum(y_pred, 1e-10)  # Ensure strictly positive
                y_true = np.maximum(y_true, 0)  # Ensure non-negative
                # Compute mean poisson deviance (negative for sklearn convention)
                # Formula: mean(2 * (y_true * log(y_true/y_pred) - (y_true - y_pred)))
                # Use small epsilon to avoid log(0)
                deviance = 2 * np.mean(y_true * np.log((y_true + 1e-10) / y_pred) - (y_true - y_pred))
                return -deviance  # Negative because sklearn uses neg_mean_poisson_deviance
            
            # Use custom scorer
            from sklearn.metrics import make_scorer
            custom_scorer = make_scorer(safe_poisson_scorer, greater_is_better=True)
            
            # Determine number of CV folds (at least 2, but not more than data allows)
            n_folds = min(cv_folds, max(2, len(y) // 5))
            if n_folds < 2:
                raise ValueError("Not enough data for cross-validation")
            
            cv_scores = cross_val_score(
                model, X_scaled, y, 
                cv=n_folds,
                scoring=custom_scorer,
                error_score='nan'  # Return NaN on error instead of raising
            )
            
            # Filter out NaN scores
            cv_scores = cv_scores[~np.isnan(cv_scores)]
            
            if len(cv_scores) == 0:
                cv_mean = np.nan
                cv_std = np.nan
            else:
                # Scores are already negative deviance, so we negate to get positive deviance
                cv_mean = float(-cv_scores.mean())
                cv_std = float(cv_scores.std())
        except Exception as cv_error:
            # If CV fails, set to NaN
            cv_mean = np.nan
            cv_std = np.nan
        
        results = {
            'coefficients': model.coef_.tolist(),
            'intercept': float(model.intercept_),
            'feature_names': feature_names,
            'deviance': float(deviance),
            'pseudo_r2': float(pseudo_r2),
            'cv_score': cv_mean,
            'cv_std': cv_std,
            'standardized': standardize,
            'n_features': X.shape[1],
            'n_trials': X.shape[0],
        }
        
        # Add p-values using Wald test (approximate)
        # Standard errors from Fisher information (Hessian)
        y_pred = model.predict(X_scaled)
        y_pred = np.maximum(y_pred, 1e-10)  # Ensure positive for Fisher information
        # For Poisson: SE ≈ sqrt(diag(X'WX)^-1) where W = diag(y_pred)
        try:
            W = np.diag(y_pred)
            fisher_info = X_scaled.T @ W @ X_scaled
            # Add small regularization to ensure invertibility
            cov_matrix = np.linalg.inv(fisher_info + alpha * np.eye(X_scaled.shape[1]) + 1e-8 * np.eye(X_scaled.shape[1]))
            std_errors = np.sqrt(np.diag(cov_matrix))
            
            # Wald test: z = coef / SE
            z_scores = model.coef_ / (std_errors + 1e-10)
            p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))
            
            results['p_values'] = p_values.tolist()
            results['std_errors'] = std_errors.tolist()
        except (np.linalg.LinAlgError, ValueError):
            # Singular matrix or other numerical issue - cannot compute p-values
            results['p_values'] = None
            results['std_errors'] = None
        
        return results
        
    except Exception as e:
        return {
            'error': f'GLM fitting failed: {str(e)}',
            'n_trials': X.shape[0],
            'n_features': X.shape[1],
        }


def fit_glm_for_unit(
    unit_data: np.ndarray,
    time_bins: np.ndarray,
    stimuli_outcome_df: pd.DataFrame,
    licks_list: Optional[List] = None,
    states_list: Optional[List] = None,
    bin_size: float = 0.01,
    time_window: Tuple[float, float] = (0.0, 0.5),
    alpha: float = 1.0,
    category_boundaries: Optional[Tuple[float, float]] = (0.983, 1.525),
    normalize_response: Optional[str] = 'zscore',
    use_temporal_kernels: bool = True,
    n_basis: int = 10,
    include_spike_history: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Fit GLM for a single unit with all behavioral predictors.
    
    Parameters:
    -----------
    unit_data : np.ndarray
        Neural data [time_bins × trials]
    time_bins : np.ndarray
        Time axis
    stimuli_outcome_df : pd.DataFrame
        DataFrame with 'stimulus' and 'outcome' columns
    licks_list : list, optional
        Lick data per trial
    states_list : list, optional
        State data per trial
    bin_size : float
        Time bin size in seconds
    time_window : tuple
        Time window for neural response
    alpha : float
        Regularization parameter
    category_boundaries : tuple, optional
        Boundaries (low, high) for Go/NoGo classification
        Default: (0.983, 1.525)
    normalize_response : str, optional
        Normalization method for neural response:
        - 'zscore': Z-score normalization (default)
        - 'minmax': Min-max normalization
        - 'per_trial': Normalize each trial's time course before averaging
        - None: No normalization
    use_temporal_kernels : bool
        If True, use raised cosine basis functions for temporal convolution (default: True)
    n_basis : int
        Default number of raised cosine basis functions per predictor (default: 10)
    include_spike_history : bool
        If True, include spike history predictor (default: True)
    **kwargs : Additional parameters for build_design_matrix
    
    Returns:
    --------
    dict : GLM results
    """
    # Extract predictors from DataFrame
    stimuli = stimuli_outcome_df['stimulus'].values if 'stimulus' in stimuli_outcome_df.columns else np.zeros(len(stimuli_outcome_df))
    outcomes = stimuli_outcome_df['outcome'].values if 'outcome' in stimuli_outcome_df.columns else np.array(['Unknown'] * len(stimuli_outcome_df))
    
    # Build design matrix
    X, y, feature_names = build_design_matrix(
        unit_data=unit_data,
        time_bins=time_bins,
        stimuli=stimuli,
        outcomes=outcomes,
        licks_list=licks_list,
        states_list=states_list,
        bin_size=bin_size,
        time_window=time_window,
        category_boundaries=category_boundaries,
        normalize_response=normalize_response,
        use_temporal_kernels=use_temporal_kernels,
        n_basis=n_basis,
        include_spike_history=include_spike_history,
        **kwargs
    )
    
    # Fit GLM
    results = fit_glm_poisson(X, y, feature_names, alpha=alpha)
    
    # Add metadata
    results['time_window'] = time_window
    results['bin_size'] = bin_size
    
    return results


def fit_glm_for_all_units(
    units: List,
    time_window: Tuple[float, float] = (0.0, 0.5),
    alpha: float = 1.0,
    category_boundaries: Optional[Tuple[float, float]] = (0.983, 1.525),
    normalize_response: Optional[str] = 'zscore',
    use_temporal_kernels: bool = True,
    n_basis: int = 10,
    include_spike_history: bool = True,
    save_to_csv: bool = True,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fit GLM for all units and return results as DataFrame.
    
    Parameters:
    -----------
    units : list
        List of Unit objects
    time_window : tuple
        Time window for neural response
    alpha : float
        Regularization parameter
    category_boundaries : tuple, optional
        Boundaries (low, high) for Go/NoGo classification
        Default: (0.983, 1.525)
    normalize_response : str, optional
        Normalization method for neural response:
        - 'zscore': Z-score normalization (default)
        - 'minmax': Min-max normalization
        - 'per_trial': Normalize each trial's time course before averaging
        - None: No normalization
    use_temporal_kernels : bool
        If True, use raised cosine basis functions for temporal convolution (default: True)
    n_basis : int
        Number of raised cosine basis functions per predictor (default: 10)
    save_to_csv : bool
        Whether to save results to CSV
    output_path : str, optional
        Path to save CSV file
    
    Returns:
    --------
    pd.DataFrame : GLM results for all units
    """
    from Analysis.NPXL_analysis.single_unit_offline_analysis.unit import Unit
    
    results_list = []
    
    for unit in units:
        if not isinstance(unit, Unit):
            continue
        
        # Extract lick data if available (try to get from unit or event_windows_data)
        licks_list = None
        try:
            # Try to get from unit's event_windows_data
            if hasattr(unit, '_event_windows_data') and len(unit._event_windows_data) >= 6:
                lick_data = unit._event_windows_data[5]
                if lick_data is not None:
                    # If it's a lick rate matrix, we'd need to convert it
                    # For now, try to use it directly if it's already in the right format
                    if isinstance(lick_data, (list, np.ndarray)):
                        licks_list = []
                        for item in lick_data:
                            if isinstance(item, (list, np.ndarray)) and len(item) > 0:
                                licks_list.append(np.array(item))
                            else:
                                licks_list.append(None)
        except (AttributeError, IndexError, TypeError):
            pass
        
        # Fit GLM
        glm_results = fit_glm_for_unit(
            unit_data=unit.unit_data,
            time_bins=unit.time_axis,
            stimuli_outcome_df=unit.stimuli_outcome_df,
            licks_list=licks_list,  # Pass extracted lick data
            states_list=None,  # Extract from metadata if available
            bin_size=unit.bin_size,
            time_window=time_window,
            alpha=alpha,
            category_boundaries=category_boundaries,
            normalize_response=normalize_response,
            use_temporal_kernels=use_temporal_kernels,
            n_basis=n_basis,
            include_spike_history=include_spike_history,
        )
        
        # Create result row
        row = {
            'unit_idx': unit.unit_idx,
            'region_name': unit.region_name,
        }
        
        # Add GLM metrics
        if 'error' not in glm_results:
            row.update({
                'glm_pseudo_r2': glm_results.get('pseudo_r2'),
                'glm_deviance': glm_results.get('deviance'),
                'glm_cv_score': glm_results.get('cv_score'),
                'glm_intercept': glm_results.get('intercept'),
                'glm_n_features': glm_results.get('n_features'),
                'glm_n_trials': glm_results.get('n_trials'),
            })
            
            # Add individual coefficients
            for i, (feat_name, coef) in enumerate(zip(
                glm_results.get('feature_names', []),
                glm_results.get('coefficients', [])
            )):
                row[f'glm_coef_{feat_name}'] = coef
                
                # Add p-values if available
                if glm_results.get('p_values') is not None:
                    row[f'glm_pval_{feat_name}'] = glm_results['p_values'][i]
        else:
            row['glm_error'] = glm_results.get('error')
        
        results_list.append(row)
    
    # Convert to DataFrame
    df = pd.DataFrame(results_list)
    
    # Save to CSV if requested
    if save_to_csv and output_path:
        from Analysis.NPXL_analysis.single_unit_offline_analysis.utils import save_dataframe_to_csv
        save_dataframe_to_csv(df, output_path, description="GLM results")
    
    return df
