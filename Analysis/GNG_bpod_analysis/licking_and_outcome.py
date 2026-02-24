# Removed imports to avoid circular dependency
from typing import Any
from pathlib import Path
from Analysis.GNG_bpod_analysis.GNG_bpod_general import (
    filter_valid_arrays,
    parse_stimuli,
    filter_out_catch_and_early_response,
    get_global_early_response_filter,
)
import Analysis.GNG_bpod_analysis.colors as colors
from Analysis.GNG_bpod_analysis.colors import (
    COLOR_FA,
    OUTCOME_COLOR_MAP,
    COLOR_ACCENT,
    COLOR_GRAY,
    COLOR_GO,
    COLOR_NOGO,
    GO_COLORS,
    COLOR_D_PRIME,
    COLOR_HIT,
    COLOR_CR,
    COLOR_LOW_BD,
    COLOR_HIGH_BD,
)

import re
import ast
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import altair as alt
from plotly.subplots import make_subplots
import streamlit as st
from Analysis.GNG_bpod_analysis.GNG_bpod_general import get_plotly_config
from Analysis.GNG_bpod_analysis.latency_map import _zscore_by_session

# Default boundaries for first-lick distance plot (match psychometric_curves fallback)
_DEFAULT_LOW_BOUNDARY_FTL = 0.983
_DEFAULT_HIGH_BOUNDARY_FTL = 1.525


def _get_boundaries_ftl() -> tuple[float, float]:
    """Return (low_boundary, high_boundary) from session state or defaults for first-lick plots."""
    low = getattr(st.session_state, "low_boundary", _DEFAULT_LOW_BOUNDARY_FTL)
    high = getattr(st.session_state, "high_boundary", _DEFAULT_HIGH_BOUNDARY_FTL)
    return float(low), float(high)


def _dist_oct_from_low_boundary(stim: float | np.ndarray, low_boundary: float) -> float | np.ndarray:
    """Distance from low boundary in octaves: log2(stim / low_boundary). Preserves low/NoGo/high ordering."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log2(np.asarray(stim, dtype=float) / low_boundary)


def _dist_oct_to_closest_boundary(
    stim: float | np.ndarray,
    low_boundary: float,
    high_boundary: float,
) -> float | np.ndarray:
    """
    Distance to the closest boundary (low or high) in octaves.
    - stim < low: log2(low/stim) (positive).
    - low <= stim <= high: min(log2(stim/low), log2(high/stim)) (0 at both boundaries).
    - stim > high: log2(stim/high) (positive).
    """
    stim = np.asarray(stim, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        below = np.log2(low_boundary / stim)
        above = np.log2(stim / high_boundary)
        in_zone_low = np.log2(stim / low_boundary)
        in_zone_high = np.log2(high_boundary / stim)
        in_zone = np.minimum(in_zone_low, in_zone_high)
    out = np.where(stim < low_boundary, below, np.where(stim > high_boundary, above, in_zone))
    return out


def _read_last_reinforcement_delay_time_seconds(parameters_txt: str | Path) -> float | None:
    """
    Read the *last session* value of `reinforcement_delay_time` from an Educage `parameters.txt`.

    Why: `parameters.txt` can contain multiple sessions separated by timestamps; we want the last one.
    """
    try:
        p = Path(parameters_txt)
        if not p.exists() or not p.is_file():
            return None
        text = p.read_text(encoding="utf-8", errors="ignore")
        matches = re.findall(r"reinforcement_delay_time\s*:\s*([0-9]*\.?[0-9]+)", text, flags=re.IGNORECASE)
        if not matches:
            return None
        return float(matches[-1])
    except Exception:
        return None


def _find_parameters_txt_for_row(project_data: pd.DataFrame, index: int) -> Path | None:
    """
    Try to locate an Educage `parameters.txt` near the current session, best-effort.

    We intentionally keep this permissive and fast: if we can't find it, caller falls back.
    """
    candidate_cols = [
        "parameters_path",
        "ParametersPath",
        "params_path",
        "ParamsPath",
        "current_dir",
        "CurrentDir",
        "FilePath",
        "file_path",
        "path",
    ]
    for col in candidate_cols:
        if col not in project_data.columns:
            continue
        val = project_data.iloc[index].get(col, None)
        if not val or not isinstance(val, str):
            continue
        try:
            p = Path(val)
        except Exception:
            continue

        candidates: list[Path] = []
        if p.is_dir():
            candidates.extend([p / "parameters.txt", p.parent / "parameters.txt"])
        else:
            candidates.extend([p.parent / "parameters.txt", p.parent.parent / "parameters.txt"])

        for c in candidates:
            if c.exists() and c.is_file():
                return c

    return None


def responses(selected_data, index=0):
    """Compute responses"""
    # Extract the outcomes list (make sure it's in list format, not a string)
    outcomes = selected_data["Outcomes"].values[index]
    # If outcomes is a string representing a list, use ast.literal_eval to convert it
    outcomes_list = ast.literal_eval(outcomes)
    # Check if we have valid data
    if not outcomes_list or len(outcomes_list) == 0:
        # Return empty DataFrame with proper structure
        return pd.DataFrame({"Hit": [], "CR": [], "FA": [], "Miss": [], "Catch - No Response": [], "Catch - Response": []})

    # Define all unique outcomes in the list
    # Note: "Correct Reject" is treated the same as "CR" and not calculated separately
    unique_outcomes = {'Hit', 'CR', 'False Alarm', 'Miss', 'Catch - No Response', 'Catch - Response'}
    # Dictionary to store cumulative counts for each outcome
    cumulative_counts = {}
    # Calculate cumulative counts for each unique outcome
    for outcome_type in unique_outcomes:
        # Special handling: treat both "CR" and "Correct Reject" as "CR"
        if outcome_type == "CR":
            # Create a binary array that matches either "CR" or "Correct Reject"
            binary_outcome = np.array([1 if (outcome == "CR" or outcome == "Correct Reject") else 0 for outcome in outcomes_list])
        else:
            # Create a binary array for the current outcome type
            binary_outcome = np.array([1 if outcome == outcome_type else 0 for outcome in outcomes_list])

        # Calculate the cumulative sum for this outcome type
        cumulative_sum = np.cumsum(binary_outcome)

        # Store the cumulative sum in the dictionary
        cumulative_counts[outcome_type] = cumulative_sum

    # Create a DataFrame for responses
    responses = pd.DataFrame({
        "Hit":  cumulative_counts["Hit"],
        "CR":   cumulative_counts["CR"],  # Includes both "CR" and "Correct Reject"
        "FA":   cumulative_counts["False Alarm"],  # Corrected the label to match 'False Alarm'
        "Miss": cumulative_counts["Miss"],
        "Catch - No Response": cumulative_counts["Catch - No Response"],
        "Catch - Response": cumulative_counts["Catch - Response"]
    })
    return responses
    
# Function to calculate the licking rate
def licking_rate(selected_data, index=0, t=10, plot=True):
    # Extract the outcomes
    
    responses_data = responses(selected_data, index)

    # Check if we have valid data
    if responses_data.empty:
        # Return empty DataFrames with proper structure
        rates = pd.DataFrame({"Hit": [], "Miss": [], "CR": [], "FA": [], "Catch - No Response": [], "Catch - Response": []})
        frac = pd.DataFrame({"Go": [], "NoGo": [], "Catch": []})
        return rates, frac

    # Fix column names for consistency
    hit_bin = responses_data["Hit"].diff().rolling(t).sum()
    miss_bin = responses_data["Miss"].diff().rolling(t).sum()
    cr_bin = responses_data["CR"].diff().rolling(t).sum()
    fa_bin = responses_data["FA"].diff().rolling(t).sum()
    catch_no_response_bin = responses_data["Catch - No Response"].diff().rolling(t).sum()
    catch_response_bin = responses_data["Catch - Response"].diff().rolling(t).sum()

    rates = pd.DataFrame({
        "Hit":  hit_bin,
        "Miss": miss_bin,
        "CR":   cr_bin,
        "FA":   fa_bin,
        "Catch - No Response": catch_no_response_bin,
        "Catch - Response": catch_response_bin
    }).dropna()
    # Check if we still have data after dropna
    if rates.empty:
        frac = pd.DataFrame({"Go": [], "NoGo": [], "Catch": []})
        return rates, frac

    # Avoid division by zero
    hit_rate = 100 * hit_bin / (hit_bin + miss_bin).replace(0, np.nan)
    fa_rate = 100 * fa_bin / (cr_bin + fa_bin).replace(0, np.nan)
    catch_rate = 100 * catch_response_bin / (catch_no_response_bin + catch_response_bin).replace(0, np.nan)
    frac = pd.DataFrame({"Go": hit_rate, "NoGo": fa_rate, "Catch": catch_rate})

   
    c_go = colors.COLOR_GO 
    c_nogo = colors.COLOR_NOGO  
    c_catch = colors.COLOR_LOW_BD
    if plot:
        st.subheader("Licking rate")
        st.line_chart(frac, color=[c_catch, c_go, c_nogo])

    return rates, frac

### Function: Compute Lick Rate ###
def compute_lick_rate(stimuli, outcomes, trialtypes=None):
    """
    Computes the lick rate (as a percentage) for each unique stimulus level.
    Separates Catch trials and computes their response rate separately.
    
    Parameters:
    - stimuli: array of stimulus values
    - outcomes: array of trial outcomes
    - trialtypes: optional array of trial types (used to filter out 'Catch' trials)
    
    Returns:
    - unique_stimuli: stimulus values for Go/NoGo trials
    - lick_rates: lick rates for Go/NoGo trials
    - catch_stimuli: stimulus values for Catch trials
    - catch_lick_rates: response rates for Catch trials
    """

    # Detect catch trials: either from trialtypes ('Catch') or from outcomes ('Catch - Response', 'Catch - No Response')
    if trialtypes is not None:
        catch_mask = np.array(['catch' in str(tt).lower() for tt in trialtypes], dtype=bool)
    else:
        # Detect catch trials from outcomes (for Educage data where outcomes contain 'Catch - Response' etc.)
        catch_mask = np.array(['catch' in str(outcome).lower() for outcome in outcomes], dtype=bool)
    
    # Separate catch and non-catch trials
    non_catch_mask = ~catch_mask
    
    # Get stimuli and outcomes for non-catch trials
    non_catch_stimuli = stimuli[non_catch_mask]
    non_catch_outcomes = outcomes[non_catch_mask]  

    # Get stimuli and outcomes for catch trials
    catch_trial_stimuli = stimuli[catch_mask]
    catch_trial_outcomes = outcomes[catch_mask]
    
    # Compute lick rates for non-catch (Go/NoGo) trials
    unique_stimuli = np.unique(non_catch_stimuli) if len(non_catch_stimuli) > 0 else np.array([])
    lick_rates = []
    for stimulus in unique_stimuli:
        mask = non_catch_stimuli == stimulus
        relevant_outcomes = non_catch_outcomes[mask]
        licks = sum(outcome in {"Hit", "False Alarm"} for outcome in relevant_outcomes)
        lick_rates.append((licks / len(relevant_outcomes) * 100) if len(relevant_outcomes) > 0 else 0)
    
    # Compute response rates for catch trials
    catch_stimuli = np.unique(catch_trial_stimuli) if len(catch_trial_stimuli) > 0 else np.array([])
    catch_lick_rates = []
    for stimulus in catch_stimuli:
        mask = catch_trial_stimuli == stimulus
        relevant_outcomes = catch_trial_outcomes[mask]
        # Count responses: 'Catch - Response' means the animal responded
        responses = sum('response' in str(outcome).lower() and 'no response' not in str(outcome).lower() 
                       for outcome in relevant_outcomes)
        catch_lick_rates.append((responses / len(relevant_outcomes) * 100) if len(relevant_outcomes) > 0 else 0)

    return unique_stimuli, np.array(lick_rates), catch_stimuli, np.array(catch_lick_rates)

def lick_rate_multipule_sessions(selected_data, t=10, plot=True,  animal_name = "None"):
    from Analysis.GNG_bpod_analysis.colors import COLOR_HIT, COLOR_FA
    from Analysis.GNG_bpod_analysis.GNG_bpod_general import get_sessions_for_animal
    # Step 2: Automatically select all sessions for the chosen animal
    session_indices, session_dates = get_sessions_for_animal(selected_data, animal_name)

    rates = np.zeros([len(session_indices), 4])  # mean and std for hit and FA rates for each session

    for idx, i in enumerate(session_indices):
        rate, frac = licking_rate(selected_data, index=i, t=t, plot=False)
        hit_rate = frac["Go"]
        fa_rate = frac["NoGo"]

        # Calculate mean and standard deviation for hit and FA rates
        mean_hit_rate = np.nanmean(hit_rate)
        std_hit_rate = np.nanstd(hit_rate)
        mean_fa_rate = np.nanmean(fa_rate)
        std_fa_rate = np.nanstd(fa_rate)

        rates[idx, :] = mean_hit_rate, std_hit_rate, mean_fa_rate, std_fa_rate

    # Create DataFrame for plotting
    data = pd.DataFrame({
        'Session Index': np.arange(1, len(session_indices) + 1),  # Sequential session index
        'Session Date': pd.to_datetime(session_dates),  # Convert session dates to datetime format
        'Hit Rate': rates[:, 0],  # Mean hit rate
        'Hit Error': rates[:, 1],  # Std hit rate (for error bars)
        'FA Rate': rates[:, 2],  # Mean false alarm rate
        'FA Error': rates[:, 3]  # Std false alarm rate
    })

    # Plot using Altair with formatted session dates
    st.title(f"Lick Rate Progress for {animal_name}")

    # Base chart for hit rate
    hit_chart = alt.Chart(data).mark_line(color = COLOR_HIT).encode(
        x=alt.X('Session Index:Q', title='Session Index', axis=alt.Axis(format='.0f', tickCount=len(session_indices))),
        y=alt.Y('Hit Rate:Q', title='Rate (%)', scale=alt.Scale(domain=[0, 100])),
        tooltip=['Session Index', 'Hit Rate', 'Hit Error']
    )

    # Error bars for hit rate
    hit_error_bars = hit_chart.mark_errorbar().encode(
        x='Session Index:Q',
        y=alt.Y('Hit Rate:Q'),
        yError='Hit Error:Q'
    )

    # Line chart for FA rate
    fa_chart = alt.Chart(data).mark_line(color= COLOR_FA).encode(
        x=alt.X('Session Index:Q'),
        y=alt.Y('FA Rate:Q', title=None),
        tooltip=['Session Index', 'FA Rate', 'FA Error']
    )

    # Error bars for FA rate
    fa_error_bars = fa_chart.mark_errorbar().encode(
        x='Session Index:Q',
        y=alt.Y('FA Rate:Q'),
        yError='FA Error:Q'
    )

    # Combine charts
    combined_chart = (hit_chart + hit_error_bars + fa_chart + fa_error_bars).properties(
        width=600,
        height=300
    )

    # Display chart
    st.altair_chart(combined_chart, use_container_width=True)

### Function: Clean and Convert Data ###
def preprocess_stimuli_outcomes(selected_data, index=0):
    """
    Extracts and processes stimuli and outcomes from the selected session.
    Converts them from string representations to NumPy arrays.
    """
    
    try:
        stimuli = selected_data["Stimuli"].values[index].strip("[]\n").split()
        stimuli = np.array([float(num) for num in stimuli])
    except Exception:
        stimuli = selected_data["Stimuli"].values[index]

    outcomes = np.array(ast.literal_eval(selected_data["Outcomes"].values[index]))

    return stimuli, outcomes

def process_and_plot_lick_data(project_data, index, plot=False, filter_early_response: bool | None = None):
    """
    Processes lick data from a DataFrame and generates raster and histogram plots using Plotly subplots.

    Args:
        project_data (pd.DataFrame): DataFrame containing 'Licks' and 'TrialTypes' columns.
        index (int): Index of the trial to process.
        plot (bool): Whether to display the plot
    """
    from Analysis.GNG_bpod_analysis.colors import COLOR_GO, COLOR_NOGO
    # Define colors
    c_go = COLOR_GO
    c_nogo = COLOR_NOGO
    outcomes = ast.literal_eval(project_data["Outcomes"].values[index])
    outcomes = np.array(outcomes)



    # Extract data from DataFrame
    licks_str = project_data.iloc[index]["Licks"]
    trials_str = project_data.iloc[index]["TrialTypes"]

    trials = ast.literal_eval(trials_str) if isinstance(trials_str, str) else trials_str
    trials = np.array(trials)

    stimuli_str = project_data.iloc[index]["Stimuli"]
    if isinstance(stimuli_str, str):
        stimuli = np.array([float(x) for x in stimuli_str.strip("[]").split()])
    else:
        stimuli = np.array(stimuli_str)

    try:
        states_str = (project_data.iloc[index]["States"])
        # Replace 'array(' with 'np.array(' so Python can evaluate it correctly
        # Regular expression to extract state names and arrays
        pattern = r"\['(.*?)' array\(\[(.*?)\]\)\]"

        # Extract matches
        matches = re.findall(pattern, states_str)

        # Convert matches to structured numpy array
        data = [(name, np.array(list(map(float, values.split(','))))) for name, values in matches]

        # Convert to numpy array with dtype=object
        states_array = np.array(data, dtype = object)
        index = np.where(states_array[:, 0] == "ReinforsmentDelay")[0]
        index_end_trial = np.where(states_array[:, 0] == "ResponseWindow")[0]
        tone_onset = states_array[index-1,1][0][0]
        reinforsment_delay_end = round(states_array[index,1][0][1] - tone_onset,3)
        response_window_end = round(states_array[index_end_trial,1][0][1] - tone_onset,3)
        response_window_end = max(response_window_end,reinforsment_delay_end+2)
    except Exception as e:
        # Fallback for datasets without Bpod state timing (e.g. Educage exports).
        # Prefer using Educage parameters.txt (last session) when available.
        stim_dur = 0.3
        reinforsment_delay_dur = 0.001
        response_window_dur = 2

        params_path = _find_parameters_txt_for_row(project_data, index)
        reinforcement_delay_time = _read_last_reinforcement_delay_time_seconds(params_path) if params_path else None

        # User request: use reinforcement_delay_time as reinforsment_delay_end (absolute time, sec)
        if reinforcement_delay_time is not None:
            reinforsment_delay_end = float(reinforcement_delay_time)
        else:
            reinforsment_delay_end = stim_dur + reinforsment_delay_dur

        response_window_end = response_window_dur+reinforsment_delay_end
        response_window_end = max(response_window_end,reinforsment_delay_end+3)



    # Convert licks from string to array safely
    if isinstance(licks_str, str):
        licks_str = re.sub(r'array\(', 'np.array(', licks_str)
        licks = eval(licks_str, {"np": np, "None": None, "nan": None})
    else:
        licks = licks_str

    
    # Convert licks to NumPy array
    licks = np.array(licks, dtype=object)
    # Ensure all elements in licks are numpy ndarrays
    licks = np.array([
        np.array(l, dtype=float) if not isinstance(l, np.ndarray) and l is not None and l != [] else
        (l if isinstance(l, np.ndarray) else np.array([]))
        for l in licks
    ], dtype=object)

    licks = np.array([_trim_on_decrease(trial) for trial in licks], dtype=object)

    if filter_early_response is None:
        filter_early_response = get_global_early_response_filter()

    # Optional filtering of Early Response trials
    if filter_early_response:
        early_response_mask = np.array(
            ['Early Response' not in str(outcome) for outcome in outcomes],
            dtype=bool
        )

        outcomes = outcomes[early_response_mask]
        trials = trials[early_response_mask]
        stimuli = stimuli[early_response_mask]
        licks = licks[early_response_mask]

    # Identify 'Go' and 'No-Go' trials
    no_go_trial = np.where(trials == 'NoGo')[0]
    go_trial = np.where(trials == 'Go')[0]

    hits = np.where(outcomes == "Hit")[0]
    misses = np.where(outcomes == "Miss")[0]
    fas = np.where(outcomes == "False Alarm")[0]
    crs = np.where(outcomes == "Correct Rejection")[0]
    
    # Check for data consistency issues and fix if possible
    if len(stimuli) != len(outcomes):
        print(f"Warning: Data inconsistency detected - stimuli length ({len(stimuli)}) != outcomes length ({len(outcomes)})")
        print(f"Stimuli: {stimuli}")
        print(f"Outcomes: {outcomes}")
        print(f"Trial indices - hits: {hits}, misses: {misses}, fas: {fas}, crs: {crs}")
        
        # Try to fix by padding stimuli array with NaN values if it's shorter
        if len(stimuli) < len(outcomes):
            print(f"Attempting to fix by padding stimuli array with NaN values")
            padding_size = len(outcomes) - len(stimuli)
            stimuli = np.concatenate([stimuli, np.full(padding_size, np.nan)])
        # If stimuli is longer, truncate it
        elif len(stimuli) > len(outcomes):
            print(f"Attempting to fix by truncating stimuli array")
            stimuli = stimuli[:len(outcomes)]

    # Extract licks
    go_licks = licks[hits]
    no_go_licks = licks[fas]

    go_stimuli = stimuli[hits]
    no_go_stimuli = stimuli[fas]


    # Filter valid Go and No-Go licks
    filtered_go_licks = filter_valid_arrays(go_licks)
    filtered_no_go_licks = filter_valid_arrays(no_go_licks)
    
    # Concatenate valid licks
    concatenated_go = np.concatenate(filtered_go_licks) if filtered_go_licks else np.array([])
    concatenated_no_go = np.concatenate(filtered_no_go_licks) if filtered_no_go_licks else np.array([])

    # Generate new trial indices for Go and No-Go
    df_go_raster = prepare_raster_data(filtered_go_licks, "Go", go_stimuli, start_index=1)
    df_nogo_raster = prepare_raster_data(filtered_no_go_licks, "No-Go", no_go_stimuli, start_index=len(filtered_go_licks) + 1)
    # Parse stimulus data for stimulus IDs
    try:
        stimuli_str = project_data.iloc[index]["Stimuli"]
        if isinstance(stimuli_str, str):
            stimuli = parse_stimuli(stimuli_str)
        else:
            stimuli = np.array(stimuli_str)
    except Exception:
        stimuli = np.array([])

    # Calculate first lick times for each trial
    go_first_lick_times = []
    no_go_first_lick_times = []
    
    # Process Go trials
    if not df_go_raster.empty:
        for trial_idx in df_go_raster["Trial Index"].unique():
            trial_licks = df_go_raster[df_go_raster["Trial Index"] == trial_idx]["Time"].values
            
            if len(trial_licks) > 0:
                first_lick_time = trial_licks[0]  # First lick in this trial
                # Get stimulus ID for this trial
                stim_id = df_go_raster[df_go_raster["Trial Index"] == trial_idx]["Trial Stim"].values[0]
                go_first_lick_times.append({
                    "Trial Index": trial_idx,
                    "Trial Type": "Go",
                    "First Lick Time (s)": first_lick_time,
                    "Stimulus ID": stim_id
                })
    
    # Process NoGo trials
    if not df_nogo_raster.empty:
        for trial_idx in df_nogo_raster["Trial Index"].unique():
            trial_licks = df_nogo_raster[df_nogo_raster["Trial Index"] == trial_idx]["Time"].values
            if len(trial_licks) > 0:
                first_lick_time = trial_licks[0]  # First lick in this trial
                # Get stimulus ID for this trial
                stim_id = df_nogo_raster[df_nogo_raster["Trial Index"] == trial_idx]["Trial Stim"].values[0]
                no_go_first_lick_times.append({
                    "Trial Index": trial_idx,
                    "Trial Type": "NoGo",
                    "First Lick Time (s)": first_lick_time,
                    "Stimulus ID": stim_id
                })
    
    # Create DataFrames for first lick times
    df_go_first_licks = pd.DataFrame(go_first_lick_times)
    df_no_go_first_licks = pd.DataFrame(no_go_first_lick_times)
    
    # Combine both trial types
    df_all_first_licks = pd.concat([df_go_first_licks, df_no_go_first_licks], ignore_index=True)
    df_all_first_licks = df_all_first_licks.sort_values("Trial Index")

    # Create the Plotly subplot figure
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0,

    )

    # Raster Plot
    if not df_go_raster.empty or not df_nogo_raster.empty:
        if not df_go_raster.empty:
            fig.add_trace(
                go.Scatter(
                    x=df_go_raster["Time"],
                    y=df_go_raster["Trial Index"],
                    mode="markers",
                    marker=dict(color=c_go, size=5),
                    name="Go Trials"
                ),
                row=1, col=1
            )
        if not df_nogo_raster.empty:
            fig.add_trace(
                go.Scatter(
                    x=df_nogo_raster["Time"],
                    y=df_nogo_raster["Trial Index"],
                    mode="markers",
                    marker=dict(color=c_nogo, size=5),
                    name="No-Go Trials"
                ),
                row=1, col=1
            )
        # Add vertical reference line at Time = 0
        fig.add_vline(x=0.0, line=dict(color=COLOR_GRAY, width=2), opacity=0.4, row=1, col=1)
        fig.add_vline(x=reinforsment_delay_end, line=dict(color=COLOR_GRAY, width=2), opacity=0.2, row=1, col=1)
        fig.add_vline(x=response_window_end, line=dict(color=COLOR_GRAY, width=2), opacity=0.4, row=1, col=1)
    else:
        st.warning("No valid lick raster data to plot.")

    # Histogram Plot
    df_go_hist = pd.DataFrame({"Time": concatenated_go, "Trial Type": "Go"})
    df_nogo_hist = pd.DataFrame({"Time": concatenated_no_go, "Trial Type": "No-Go"})
    df_hist = pd.concat([df_go_hist, df_nogo_hist])

    if not df_hist.empty:
        # Compute histogram bins and plot as lines
        # Define bin edges for consistent comparison
        all_times = np.concatenate([concatenated_go, concatenated_no_go])
        bin_edges = np.linspace(min(all_times), max(all_times), 30)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Compute histogram for Go trials
        go_density, _ = np.histogram(concatenated_go, bins=bin_edges, density=True)
        fig.add_trace(
            go.Scatter(
                x=bin_centers,
                y=go_density,
                mode='lines',
                name="Go",
                line=dict(color=c_go, width=2),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Compute histogram for No-Go trials (density)
        nogo_density, _ = np.histogram(concatenated_no_go, bins=bin_edges, density=True)
        fig.add_trace(
            go.Scatter(
                x=bin_centers,
                y=nogo_density,
                mode='lines',
                name="No-Go",
                line=dict(color=c_nogo, width=2),
                showlegend=False
            ),
            row=2, col=1
        )
        fig.add_vline(x = 0.0, line = dict(color = "gray", width=2), opacity=0.4 , row=2, col=1)
        fig.add_vline(x = reinforsment_delay_end, line = dict(color = "gray", width = 2), opacity=0.2, row = 2, col = 1)
        fig.add_vline(x = response_window_end, line = dict(color = "gray", width = 2), opacity=0.4, row = 2, col = 1)
    else:
        st.warning("No valid lick data to plot.")

    # Update layout and styling
    fig.update_layout(
        height=500,
        width = 1000,
        showlegend=True,
        xaxis_title="",
        yaxis_title="Trial Index",
        xaxis2_title="Time from Tone (s)",
        yaxis2_title="Density",
        plot_bgcolor="white",
        title_text=f"Hits: {len(filtered_go_licks)}, FAs: {len(filtered_no_go_licks)} / {len(trials)}",

    )

    # Display the subplot figure in Streamlit
    if plot:
        colors.apply_standard_font_sizes(fig)
        st.plotly_chart(fig, use_container_width=False, key="raster_histogram_plot")


   # Create a DataFrame to contain lick and stimulus info for each outcome type
    outcome_data = {
        "Outcome": [],
        "Licks": [],
        "Stimuli": []
    }

    # Hits
    outcome_data["Outcome"].append("Hit")
    outcome_data["Licks"].append(go_licks)
    # Add bounds checking for hits
    if len(hits) > 0 and len(stimuli) > 0:
        valid_hit_indices = hits[hits < len(stimuli)]
        if len(valid_hit_indices) > 0:
            outcome_data["Stimuli"].append(stimuli[valid_hit_indices])
        else:
            outcome_data["Stimuli"].append(np.array([]))
    else:
        outcome_data["Stimuli"].append(np.array([]))

    # Misses
    outcome_data["Outcome"].append("Miss")
    outcome_data["Licks"].append(licks[misses])
    # Add bounds checking to prevent index out of bounds error
    if len(misses) > 0 and len(stimuli) > 0:
        # Only include stimuli for misses that are within bounds
        valid_miss_indices = misses[misses < len(stimuli)]
        if len(valid_miss_indices) > 0:
            outcome_data["Stimuli"].append(stimuli[valid_miss_indices])
        else:
            outcome_data["Stimuli"].append(np.array([]))
    else:
        outcome_data["Stimuli"].append(np.array([]))

    # False Alarms
    outcome_data["Outcome"].append("False Alarm")
    outcome_data["Licks"].append(no_go_licks)
    # Add bounds checking for false alarms
    if len(fas) > 0 and len(stimuli) > 0:
        valid_fa_indices = fas[fas < len(stimuli)]
        if len(valid_fa_indices) > 0:
            outcome_data["Stimuli"].append(stimuli[valid_fa_indices])
        else:
            outcome_data["Stimuli"].append(np.array([]))
    else:
        outcome_data["Stimuli"].append(np.array([]))

    # Correct Rejections
    outcome_data["Outcome"].append("Correct Rejection")
    outcome_data["Licks"].append(licks[crs])
    # Add bounds checking for correct rejections
    if len(crs) > 0 and len(stimuli) > 0:
        valid_cr_indices = crs[crs < len(stimuli)]
        if len(valid_cr_indices) > 0:
            outcome_data["Stimuli"].append(stimuli[valid_cr_indices])
        else:
            outcome_data["Stimuli"].append(np.array([]))
    else:
        outcome_data["Stimuli"].append(np.array([]))


    outcome_df = pd.DataFrame(outcome_data)

    return df_go_first_licks, df_no_go_first_licks, outcome_df

# Prepare raster plot data
def prepare_raster_data(licks_list, trial_type, trial_stim, start_index=1):

    """Formats raster data for Plotly scatter plot."""
    data = []
    for i, licks_in_trial in enumerate(licks_list):
        if isinstance(licks_in_trial, np.ndarray) and licks_in_trial.size > 0:
            trial_idx = start_index + i  # Assigns sequential index
            for lick in licks_in_trial:
                data.append({"Time": lick, "Trial Index": trial_idx, "Trial Type": trial_type, "Trial Stim": trial_stim[i]})
    return pd.DataFrame(data)


def _build_first_lick_distance_df(
    project_data: pd.DataFrame,
    indices: list[int],
    filter_early_response: bool,
) -> pd.DataFrame | None:
    """
    Build combined first-lick DataFrame across sessions with SessionID, Dist_oct, and Latency_Z.
    Returns None if no valid rows after filtering.
    """
    low_boundary, high_boundary = _get_boundaries_ftl()
    rows = []
    for idx in indices:
        try:
            df_go, df_nogo, _ = process_and_plot_lick_data(
                project_data, idx, plot=False, filter_early_response=filter_early_response
            )
            part = pd.concat([df_go, df_nogo], ignore_index=True)
            part = part[
                (part["First Lick Time (s)"] >= 0)
                & (part["First Lick Time (s)"] <= FIRST_LICK_LATENCY_MAX_S)
            ]
            if part.empty:
                continue
            part["SessionID"] = idx
            rows.append(part)
        except Exception:
            continue
    if not rows:
        return None
    ftl_df = pd.concat(rows, ignore_index=True)
    ftl_df["Dist_oct"] = _dist_oct_from_low_boundary(
        ftl_df["Stimulus ID"].values, low_boundary
    )
    ftl_df["Dist_oct_closest"] = _dist_oct_to_closest_boundary(
        ftl_df["Stimulus ID"].values, low_boundary, high_boundary
    )
    ftl_df = ftl_df[np.isfinite(ftl_df["Dist_oct"]) & np.isfinite(ftl_df["Dist_oct_closest"])].copy()
    if ftl_df.empty:
        return None
    # Z-score Go and NoGo separately within each session
    ftl_df = _zscore_by_session(
        ftl_df,
        session_col=["SessionID", "Trial Type"],
        latency_col="First Lick Time (s)",
        z_col="Latency_Z",
    )
    return ftl_df


def _get_phase_per_session(project_data: pd.DataFrame) -> dict[int, str]:
    """
    Map session row index -> Phase (Novice / 1B Expert / 2B Expert).

    Criteria (per animal, sorted chronologically):
      - Novice:    first 2 sessions with N_Boundaries == 1 AND Tones_per_class > 1.
      - 1B Expert: last  2 sessions with N_Boundaries == 1 AND Tones_per_class >= 3.
      - 2B Expert: last  2 sessions with N_Boundaries == 2 AND Tones_per_class >= 3.

    A session CAN carry multiple phase labels (e.g. an early 1B session that is
    both Novice and 1B Expert when the animal has very few sessions). The caller
    explodes "|"-separated labels into multiple rows.
    """
    required = {"MouseName", "SessionDate", "N_Boundaries", "Tones_per_class"}
    if required - set(project_data.columns):
        return {}
    # Coerce to numeric to handle string / float dtype from CSV
    n_boundaries = pd.to_numeric(project_data["N_Boundaries"], errors="coerce")
    tones_per_class = pd.to_numeric(project_data["Tones_per_class"], errors="coerce")

    idx_to_phases: dict[int, list[str]] = {}

    for _mouse, grp in project_data.groupby("MouseName", sort=False):
        grp = grp.sort_values("SessionDate")
        indices = grp.index.tolist()

        # Novice: first 2 sessions with 1 boundary & more than 1 tone per class
        novice_candidates = [
            i for i in indices
            if n_boundaries.loc[i] == 1 and tones_per_class.loc[i] > 1
        ]
        for idx in novice_candidates[:2]:
            idx_to_phases.setdefault(idx, []).append("Novice")

        # 1B Expert: last 2 sessions with 1 boundary & >= 3 tones per class
        one_b_idx = [
            i for i in indices
            if n_boundaries.loc[i] == 1 and tones_per_class.loc[i] >= 3
        ]
        for idx in one_b_idx[-2:]:
            idx_to_phases.setdefault(idx, []).append("1B Expert")

        # 2B Expert: last 2 sessions with 2 boundaries & >= 3 tones per class
        two_b_idx = [
            i for i in indices
            if n_boundaries.loc[i] == 2 and tones_per_class.loc[i] >= 3
        ]
        for idx in two_b_idx[-2:]:
            idx_to_phases.setdefault(idx, []).append("2B Expert")

    # Flatten: "|"-separated string so the caller can explode later
    return {idx: "|".join(phases) for idx, phases in idx_to_phases.items()}


def _build_phase_ftl_and_aggregate(
    project_data: pd.DataFrame,
    indices: list[int],
    filter_early_response: bool,
    boundary_bin_oct: float = 0.1,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    """
    Build FTL with Phase, then aggregate Mean_Z, SEM_Z per (Phase, Dist_oct_closest).
    X-axis is distance to closest boundary (octaves), sorted ascending.
    Returns (ftl_with_phase, agg_grand_mean, agg_per_animal) or (None, None, None).
    """
    ftl_df = _build_first_lick_distance_df(project_data, indices, filter_early_response)
    if ftl_df is None or ftl_df.empty:
        return None, None, None
    ftl_df = ftl_df[np.isfinite(ftl_df["Latency_Z"])].copy()
    if ftl_df.empty:
        return None, None, None

    phase_map = _get_phase_per_session(project_data)
    ftl_df["Phase_raw"] = ftl_df["SessionID"].map(phase_map)
    ftl_df = ftl_df.dropna(subset=["Phase_raw"]).copy()
    if ftl_df.empty:
        return None, None, None
    # Explode "|"-separated phases so a session in multiple phases
    # contributes a row to each (e.g. Novice AND 1B Expert).
    ftl_df["Phase"] = ftl_df["Phase_raw"].str.split("|")
    ftl_df = ftl_df.explode("Phase")
    ftl_df = ftl_df[ftl_df["Phase"].isin(["Novice", "1B Expert", "2B Expert"])].copy()
    ftl_df.drop(columns=["Phase_raw"], inplace=True)
    if ftl_df.empty:
        return None, None, None

    MAX_DIST_OCT = 0.6   # drop stimuli farther than this from any boundary
    BIN_WIDTH_OCT = 0.05  # bin width for combining nearby distances

    ftl_df = ftl_df[ftl_df["Dist_oct_closest"].abs() <= MAX_DIST_OCT].copy()
    if ftl_df.empty:
        return None, None, None

    # Bin distances so nearby stimuli are combined
    ftl_df["Dist_bin"] = (ftl_df["Dist_oct_closest"] / BIN_WIDTH_OCT).round(0) * BIN_WIDTH_OCT
    ftl_df["Dist_bin"] = ftl_df["Dist_bin"].round(4)

    # Add MouseName for per-animal lines
    mouse_map = project_data["MouseName"].to_dict() if "MouseName" in project_data.columns else {}
    ftl_df["MouseName"] = ftl_df["SessionID"].map(mouse_map)

    # --- Per-animal aggregation (for individual lines) ---
    animal_agg = (
        ftl_df.groupby(["Phase", "Trial Type", "MouseName", "Dist_bin"], as_index=False)["Latency_Z"]
        .mean()
        .rename(columns={"Latency_Z": "Mean_Z", "Dist_bin": "Dist_oct_closest"})
    )

    # --- Grand mean aggregation (across all animals) ---
    agg_rows = []
    for (phase, ttype, dist_bin), sub in ftl_df.groupby(["Phase", "Trial Type", "Dist_bin"]):
        z = sub["Latency_Z"].astype(float)
        n = z.size
        mean_z = float(np.nanmean(z)) if n else np.nan
        sem_z = float(np.nanstd(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
        agg_rows.append({
            "Phase": phase,
            "Trial Type": ttype,
            "Dist_oct_closest": float(dist_bin),
            "Mean_Z": mean_z,
            "SEM_Z": sem_z,
            "N": n,
        })
    agg_df = pd.DataFrame(agg_rows)

    return ftl_df, agg_df, animal_agg


def _run_phase_latency_stats(
    ftl_with_phase: pd.DataFrame,
    project_data: pd.DataFrame,
    boundary_bin_oct: float = 0.1,
) -> tuple[str, pd.DataFrame | None]:
    """
    For each (Phase, Trial Type) group test whether latency near the boundary
    differs from latency far from the boundary.

    Per-animal metric: mean Latency_Z in the *closest* bin
    vs. mean Latency_Z in the 0.25 oct bin (or nearest available).
    Paired Wilcoxon signed-rank across animals; Bonferroni correction.
    Returns (report_string, results_df).
    """
    from scipy.stats import wilcoxon, mannwhitneyu

    if "MouseName" not in ftl_with_phase.columns:
        if "MouseName" not in project_data.columns:
            return "Statistics skipped (no MouseName).", None
        ftl = ftl_with_phase.copy()
        ftl["MouseName"] = ftl["SessionID"].map(project_data["MouseName"].to_dict())
    else:
        ftl = ftl_with_phase.copy()
    ftl = ftl.dropna(subset=["MouseName", "Latency_Z"])
    if ftl.empty:
        return "No valid data for statistics.", None

    phases = [p for p in ["Novice", "1B Expert", "2B Expert"] if p in ftl["Phase"].unique()]

    # Go trials only
    ftl_go = ftl[ftl["Trial Type"] == "Go"]
    if ftl_go.empty:
        return "No Go trials for statistics.", None

    # Use Dist_bin (binned distances) so all animals sharing a bin are grouped together.
    # Fall back to Dist_oct_closest if Dist_bin wasn't computed upstream.
    dist_col = "Dist_bin" if "Dist_bin" in ftl_go.columns else "Dist_oct_closest"

    FAR_BIN = 0.25  # fixed "far" reference bin (octaves)

    stat_rows = []
    for phase in phases:
        sub = ftl_go[ftl_go["Phase"] == phase]
        if sub.empty:
            continue
        sub_abs = sub[dist_col].abs()
        min_dist = sub_abs.min()

        # Find the available bin closest to FAR_BIN
        available_bins = sub_abs.unique()
        far_dist = float(available_bins[np.argmin(np.abs(available_bins - FAR_BIN))])
        if min_dist == far_dist:
            continue
        close_summary = (
            sub.loc[sub_abs == min_dist]
            .groupby("MouseName")["Latency_Z"].mean()
        )
        far_summary = (
            sub.loc[sub_abs == far_dist]
            .groupby("MouseName")["Latency_Z"].mean()
        )
        common = close_summary.index.intersection(far_summary.index)
        n = len(common)
        close_vals = close_summary.loc[common].values if n else np.array([])
        far_vals = far_summary.loc[common].values if n else np.array([])

        row = {
            "Phase": phase,
            "n_animals": n,
            "Dist_close": float(min_dist),
            "Dist_far": float(far_dist),
            "Mean_Z_close": float(np.nanmean(close_vals)) if n else np.nan,
            "Mean_Z_far": float(np.nanmean(far_vals)) if n else np.nan,
            "statistic": np.nan,
            "p": np.nan,
        }
        if n >= 2:
            try:
                # One-sided: H1 = close > far
                stat, p = wilcoxon(close_vals, far_vals, alternative="greater")
            except Exception:
                stat, p = mannwhitneyu(close_vals, far_vals, alternative="greater")
            row["statistic"] = float(stat)
            row["p"] = float(p)
        stat_rows.append(row)

    if not stat_rows:
        return "No phases with enough data for close-vs-far test.", None

    results_df = pd.DataFrame(stat_rows)
    # Bonferroni correction across all tests
    n_tests = results_df["p"].notna().sum()
    if n_tests > 1:
        results_df["p_adj"] = results_df["p"].apply(lambda x: min(1.0, x * n_tests) if np.isfinite(x) else np.nan)
    else:
        results_df["p_adj"] = results_df["p"]
    results_df["sig"] = results_df["p_adj"].apply(
        lambda x: "***" if x < 0.001 else "**" if x < 0.01 else "*" if x < 0.05 else "ns" if np.isfinite(x) else ""
    )

    lines = [
        f"**Close vs Far from boundary (Go only)** — per phase, comparing closest bin vs ~{FAR_BIN} oct bin.",
        "One-sided Wilcoxon signed-rank (H1: close > far), Bonferroni-corrected.",
    ]
    return "\n\n".join(lines), results_df


def _show_phase_debug_table(
    project_data: pd.DataFrame,
    phase_map: dict[int, str],
    indices: list[int],
) -> None:
    """Display a compact table of session -> phase assignments inside an expander."""
    rows_for_table = []
    for idx in indices:
        if idx not in project_data.index:
            continue
        row = project_data.loc[idx]
        phases = phase_map.get(idx, "—")
        rows_for_table.append({
            "Index": idx,
            "Mouse": row.get("MouseName", "?"),
            "Date": str(row.get("SessionDate", "?")),
            "N_Boundaries": row.get("N_Boundaries", "?"),
            "Tones_per_class": row.get("Tones_per_class", "?"),
            "Notes": str(row.get("Notes", "")),
            "Phase(s)": phases,
        })
    if rows_for_table:
        debug_df = pd.DataFrame(rows_for_table)
        with st.expander("Phase assignment per session (debug)", expanded=False):
            st.dataframe(debug_df, use_container_width=True, hide_index=True)


def _run_between_phase_stats_at_closest_bin(
    ftl_df: pd.DataFrame,
    project_data: pd.DataFrame,
    phases: list[str],
) -> None:
    """Compare phases at the closest distance bin (Go only). Kruskal-Wallis + pairwise Mann-Whitney U."""
    from itertools import combinations
    from scipy.stats import kruskal, mannwhitneyu

    dist_col = "Dist_bin" if "Dist_bin" in ftl_df.columns else "Dist_oct_closest"

    go = ftl_df[(ftl_df["Trial Type"] == "Go")].copy()
    if go.empty:
        return

    if "MouseName" not in go.columns:
        if "MouseName" not in project_data.columns:
            return
        go["MouseName"] = go["SessionID"].map(project_data["MouseName"].to_dict())

    closest_bin = go[dist_col].abs().min()
    go_close = go[go[dist_col].abs() == closest_bin]

    # Per-animal mean at this bin
    animal_means = (
        go_close.groupby(["Phase", "MouseName"])["Latency_Z"]
        .mean()
        .reset_index()
    )
    animal_means = animal_means[animal_means["Phase"].isin(phases)]

    groups = {ph: grp["Latency_Z"].values for ph, grp in animal_means.groupby("Phase") if ph in phases}
    present = [ph for ph in phases if ph in groups and len(groups[ph]) >= 1]
    if len(present) < 2:
        return

    lines: list[str] = [
        f"**Between-phase comparison at closest bin (|dist| = {closest_bin:.2f} oct, Go only)**",
        "",
        "Per-animal mean Latency Z at this bin:",
    ]
    for ph in present:
        vals = groups[ph]
        lines.append(f"- **{ph}**: n={len(vals)}, mean={np.nanmean(vals):.3f}, SD={np.nanstd(vals, ddof=1):.3f}")
    lines.append("")

    # Omnibus Kruskal-Wallis (if ≥3 groups with ≥2 animals each)
    testable = [ph for ph in present if len(groups[ph]) >= 2]
    if len(testable) >= 3:
        stat, p = kruskal(*(groups[ph] for ph in testable))
        lines.append(f"Kruskal-Wallis: H={stat:.2f}, p={p:.3g}")
    elif len(testable) == 2:
        lines.append("Only 2 groups with n≥2 — skipping omnibus test, running pairwise only.")
    else:
        lines.append("Too few animals per group for statistical testing.")
        with st.expander("Between-phase comparison at closest bin"):
            st.markdown("\n\n".join(lines))
        return

    # Pairwise Mann-Whitney U (two-sided)
    pair_rows = []
    for a, b in combinations(testable, 2):
        try:
            stat, p = mannwhitneyu(groups[a], groups[b], alternative="two-sided")
        except ValueError:
            stat, p = np.nan, np.nan
        pair_rows.append({"Group A": a, "Group B": b, "U": stat, "p": p})
    if pair_rows:
        pair_df = pd.DataFrame(pair_rows)
        n_tests = pair_df["p"].notna().sum()
        pair_df["p_adj"] = pair_df["p"].apply(
            lambda x: min(1.0, x * n_tests) if np.isfinite(x) else np.nan
        )
        pair_df["sig"] = pair_df["p_adj"].apply(
            lambda x: "***" if x < 0.001 else "**" if x < 0.01 else "*" if x < 0.05 else "ns" if np.isfinite(x) else ""
        )
        lines.append("")
        lines.append("Pairwise Mann-Whitney U (two-sided, Bonferroni-corrected):")

    with st.expander("Between-phase comparison at closest bin"):
        st.markdown("\n\n".join(lines))
        if pair_rows:
            fmt = {"U": "{:.1f}", "p": "{:.3g}", "p_adj": "{:.3g}"}
            st.dataframe(
                pair_df.style.format(fmt, na_rep="—"),
                use_container_width=True,
                hide_index=True,
            )


def plot_first_lick_by_distance_by_phase(
    project_data: pd.DataFrame,
    index: int | list[int] | None = None,
    plot: bool = True,
    filter_early_response: bool | None = None,
    boundary_bin_oct: float = 0.1,
    show_2b: bool = True,
    min_n: int = 1,
) -> None:
    """
    First-lick latency by distance to closest boundary, compared across phases.
    X-axis: distance to closest boundary (octaves). Y-axis: Mean Z-score ± SEM.
    """
    if filter_early_response is None:
        filter_early_response = get_global_early_response_filter()
    if index is None:
        indices = project_data.index.tolist()
    elif isinstance(index, (int, np.integer)):
        indices = [index]
    else:
        indices = list(index)
    if not indices:
        if plot:
            st.warning("No sessions to plot.")
        return

    # Show phase assignment for debugging / verification
    phase_map = _get_phase_per_session(project_data)
    if plot:
        _show_phase_debug_table(project_data, phase_map, indices)

    ftl_df, agg_df, animal_agg = _build_phase_ftl_and_aggregate(
        project_data, indices, filter_early_response, boundary_bin_oct=boundary_bin_oct
    )
    if ftl_df is None or agg_df is None or animal_agg is None:
        if plot:
            st.warning("No valid first-lick data for phase comparison.")
        return

    phases_to_plot = ["Novice", "1B Expert"]
    if show_2b:
        phases_to_plot.append("2B Expert")

    # Filter to requested phases
    ftl_df = ftl_df[ftl_df["Phase"].isin(phases_to_plot)].copy()
    agg_df = agg_df[agg_df["Phase"].isin(phases_to_plot)].copy()
    animal_agg = animal_agg[animal_agg["Phase"].isin(phases_to_plot)].copy()
    if agg_df.empty:
        if plot:
            st.warning("No aggregated data for the selected phases.")
        return

    # Drop distance bins with fewer than min_n trials
    if min_n > 1:
        agg_df = agg_df[agg_df["N"] >= min_n].copy()
    if agg_df.empty:
        if plot:
            st.warning("No data remaining after min-N filter.")
        return

    color_map = {"Novice": "gray", "1B Expert": COLOR_LOW_BD, "2B Expert": COLOR_HIGH_BD}
    # Go trials only — single plot; drop the 0.3 edge bin (sparse / unreliable)
    go_agg = agg_df[(agg_df["Trial Type"] == "Go") & (agg_df["Dist_oct_closest"] < 0.3)]
    fig = go.Figure()
    for phase in phases_to_plot:
        sub = go_agg[go_agg["Phase"] == phase].sort_values("Dist_oct_closest")
        if sub.empty:
            continue
        x = sub["Dist_oct_closest"].values
        y = sub["Mean_Z"].values
        sem = sub["SEM_Z"].values
        color = color_map.get(phase, "black")
        # Mean line
        fig.add_trace(
            go.Scatter(
                x=x, y=y, mode="lines+markers",
                name=phase, legendgroup=phase,
                line=dict(color=color, width=3),
                marker=dict(size=5),
            )
        )
        # SEM shading
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([x, x[::-1]]),
                y=np.concatenate([y - sem, (y + sem)[::-1]]),
                fill="toself", fillcolor=color, opacity=0.15,
                line=dict(color="rgba(0,0,0,0)"),
                legendgroup=phase, showlegend=False, hoverinfo="skip",
            )
        )


    fig.update_layout(
        title="First Lick by Distance to Closest Boundary — Go (by learning phase)",
        xaxis_title="Distance to closest boundary (oct)",
        yaxis_title="First Lick Time (Z-score)",
        template="simple_white",
    )
    if plot:
        st.plotly_chart(fig, use_container_width=True, config=get_plotly_config())

    # Summary table of plotted values per phase
    if plot:
        summary_rows = []
        for phase in phases_to_plot:
            sub = go_agg[go_agg["Phase"] == phase].sort_values("Dist_oct_closest")
            if sub.empty:
                continue
            for _, row in sub.iterrows():
                summary_rows.append({
                    "Phase": phase,
                    "Dist (oct)": row["Dist_oct_closest"],
                    "Mean Z": row["Mean_Z"],
                    "SEM Z": row["SEM_Z"],
                    "N": int(row["N"]),
                })
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            with st.expander("Summary table (plotted values)"):
                fmt = {"Dist (oct)": "{:.2f}", "Mean Z": "{:.3f}", "SEM Z": "{:.3f}"}
                st.dataframe(
                    summary_df.style.format(fmt, na_rep="—"),
                    use_container_width=True,
                    hide_index=True,
                )

    # Between-group comparison at the closest distance bin
    if plot:
        _run_between_phase_stats_at_closest_bin(ftl_df, project_data, phases_to_plot)

    # Statistics: close vs far from boundary, per (Phase x Trial Type)
    report, stats_df = _run_phase_latency_stats(ftl_df, project_data, boundary_bin_oct=boundary_bin_oct)
    if plot:
        with st.expander("Statistics (close vs far from boundary)"):
            st.markdown(report)
            if stats_df is not None and not stats_df.empty:
                fmt = {
                    "Dist_close": "{:.2f}",
                    "Dist_far": "{:.2f}",
                    "Mean_Z_close": "{:.3f}",
                    "Mean_Z_far": "{:.3f}",
                    "statistic": "{:.2f}",
                    "p": "{:.3g}",
                    "p_adj": "{:.3g}",
                }
                st.dataframe(stats_df.style.format(fmt, na_rep="—"), use_container_width=True, hide_index=True)


def plot_first_lick_by_stimulus(
    project_data: pd.DataFrame,
    index: int | list[int] | None = None,
    plot: bool = True,
    filter_early_response: bool | None = None,
    normalize: bool = True,
) -> None:
    """
    Plot first lick times by distance from boundary (octaves), with optional in-session Z-score.

    Single-session: pass int index. Multi-session/multi-animal: pass index=None to use all rows.
    When normalize=True (default), y-axis is First Lick Time (Z-score) per session for comparability.
    """
    from Analysis.GNG_bpod_analysis.colors import COLOR_GO, COLOR_NOGO

    if filter_early_response is None:
        filter_early_response = get_global_early_response_filter()

    if index is None:
        indices = project_data.index.tolist()
    elif isinstance(index, (int, np.integer)):
        indices = [index]
    else:
        indices = list(index)

    if not indices:
        if plot:
            st.warning("No sessions to plot.")
        return

    ftl_df = _build_first_lick_distance_df(
        project_data, indices, filter_early_response
    )
    if ftl_df is None or ftl_df.empty:
        if plot:
            st.warning("No valid first-lick data after filtering.")
        return

    low_boundary, high_boundary = _get_boundaries_ftl()
    dist_high_oct = np.log2(high_boundary / low_boundary)
    y_col = "Latency_Z" if normalize else "First Lick Time (s)"
    ftl_plot = ftl_df[np.isfinite(ftl_df[y_col])].copy()
    if ftl_plot.empty:
        if plot:
            st.warning("No finite values to plot.")
        return

    # One box per unique distance (round to avoid float noise)
    ftl_plot["Dist_oct_round"] = np.round(ftl_plot["Dist_oct"], 3)
    distances_unique = np.sort(ftl_plot["Dist_oct_round"].unique())

    fig = go.Figure()
    for dist in distances_unique:
        subset = ftl_plot[ftl_plot["Dist_oct_round"] == dist]
        y_vals = subset[y_col].values
        # Go/NoGo color from stimulus: use first stimulus in group (all same region for same dist)
        stim_repr = subset["Stimulus ID"].iloc[0]
        if low_boundary < stim_repr < high_boundary:
            color = COLOR_NOGO
            name_prefix = "NoGo"
        else:
            color = COLOR_GO
            name_prefix = "Go"
        fig.add_trace(
            go.Box(
                x=[dist] * len(y_vals),
                y=y_vals,
                name=f"{name_prefix} {dist:.3f}",
                fillcolor=None,
                opacity=0.7,
                line_color=color,
                showlegend=True,
                boxpoints="outliers",
                jitter=0.1,
                pointpos=0,
            )
        )

    fig.add_vline(
        x=0.0,
        line_width=2,
        line_dash="dash",
        line_color=COLOR_GRAY,
    )
    fig.add_vline(
        x=dist_high_oct,
        line_width=2,
        line_dash="dash",
        line_color=COLOR_GRAY,
    )
    if normalize:
        fig.add_hline(y=0, line_width=1, line_dash="dot", line_color=COLOR_GRAY)
        fig.add_hline(y=1, line_width=1, line_dash="dot", line_color=COLOR_GRAY)
        fig.add_hline(y=-1, line_width=1, line_dash="dot", line_color=COLOR_GRAY)
        yaxis_title = "First Lick Time (Z-score)"
        title_suffix = " (Z-score per session)"
    else:
        fig.add_hline(
            y=0.2,
            line_width=2,
            line_dash="dash",
            line_color=COLOR_GRAY,
            annotation_text="Reinforcement Delay",
        )
        fig.add_hline(
            y=2.2,
            line_width=2,
            line_dash="dash",
            line_color=COLOR_GRAY,
            annotation_text="Response Window",
        )
        yaxis_title = "First Lick Time (s)"
        title_suffix = ""

    fig.update_layout(
        title=f"First Lick Time by Distance from Boundary{title_suffix}",
        xaxis_title="Distance from low boundary (octaves)",
        yaxis_title=yaxis_title,
        template="simple_white",
        showlegend=False,
        xaxis=dict(
            tickmode="array",
            tickvals=distances_unique,
            ticktext=[f"{x:.2f}" for x in distances_unique],
        ),
    )

    if plot:
        st.plotly_chart(fig, use_container_width=True, config=get_plotly_config())


def plot_n_lick_by_stimulus(project_data, index, plot=True, filter_early_response: bool | None = None):

    # Build per-trial lick counts and stimulus IDs
    try:
        licks_str = project_data.iloc[index]["Licks"]
        stimuli_str = project_data.iloc[index]["Stimuli"]

        if isinstance(licks_str, str):
            licks_str = re.sub(r'array\(', 'np.array(', licks_str)
            licks = eval(licks_str, {"np": np, "None": None, "nan": None})
        else:
            licks = licks_str

        licks = np.array(licks, dtype=object)
        licks = np.array([
            np.array(l, dtype=float) if not isinstance(l, np.ndarray) and l is not None and l != [] else
            (l if isinstance(l, np.ndarray) else np.array([]))
            for l in licks
        ], dtype=object)
        
        licks = np.array([_trim_on_decrease(trial) for trial in licks], dtype=object)
    
        if isinstance(stimuli_str, str):
            stimuli = np.array([float(x) for x in stimuli_str.strip("[]").split()])
        else:
            stimuli = np.array(stimuli_str)

        # Optional Early Response filtering uses Outcomes to mask trials
        if filter_early_response is None:
            filter_early_response = get_global_early_response_filter()

        if filter_early_response:
            try:
                outcomes = np.array(ast.literal_eval(project_data["Outcomes"].values[index]))
                early_response_mask = np.array(
                    ['Early Response' not in str(outcome) for outcome in outcomes],
                    dtype=bool
                )
                # Guard against length mismatch
                if len(early_response_mask) == len(stimuli) == len(licks):
                    stimuli = stimuli[early_response_mask]
                    licks = licks[early_response_mask]
            except Exception:
                # If anything goes wrong, fall back to unfiltered data
                pass

        n_licks = np.array([len(l) if isinstance(l, np.ndarray) else 0 for l in licks])

        per_trial_df = pd.DataFrame({
            "Trial": np.arange(1, len(n_licks) + 1),
            "Stimulus ID": stimuli,
            "N Licks": n_licks
        })

        grouped = per_trial_df.groupby("Stimulus ID").agg(
            N_Trials=("N Licks", "size"),
            Mean_N_Licks=("N Licks", "mean"),
            Median_N_Licks=("N Licks", "median"),
            Std_N_Licks=("N Licks", "std"),
            Sum_N_Licks=("N Licks", "sum")
        ).reset_index()

    except Exception:
        per_trial_df = pd.DataFrame(columns=["Trial", "Stimulus ID", "N Licks"]) 
        grouped = pd.DataFrame(columns=["Stimulus ID", "N_Trials", "Mean_N_Licks", "Median_N_Licks", "Std_N_Licks", "Sum_N_Licks"]) 

    fig = go.Figure()
    if not per_trial_df.empty:
        
        # Sort stimulus IDs for consistent x-axis ordering
        stim_ids_sorted = sorted(per_trial_df["Stimulus ID"].unique())
        
        # Color by stimulus type (Go vs NoGo)
        for stim_id in stim_ids_sorted:
            stim_data = per_trial_df[per_trial_df["Stimulus ID"] == stim_id]
            
            # Determine color based on stimulus type
            if  (st.session_state.low_boundary > stim_id) or (stim_id > st.session_state.high_boundary):
                color = COLOR_GO
                name = f"Go {stim_id}"
                showlegend_stim = False
            elif round(st.session_state.low_boundary,2) == round(stim_id, 2):
                color = colors.COLOR_LOW_BD
                name = f"Catch - Low Boundary"
                showlegend_stim = True
            elif round(st.session_state.high_boundary,2) == round(stim_id, 2):
                color = colors.COLOR_HIGH_BD
                name = f"Catch - High Boundary"
                showlegend_stim = True
            else:
                color = COLOR_NOGO
                name = f"NoGo {stim_id}"
                showlegend_stim = False
            fig.add_trace(
                go.Violin(
                    x=stim_data["Stimulus ID"].astype(str),
                    y=stim_data["N Licks"],
                    name=name,
                    box_visible=False,
                    meanline_visible=True,
                    line_color=color,
                    opacity=0.6,
                    legendgroup=name,
                    showlegend=showlegend_stim,

                )
            )
            
        # Add line connecting the means per stimulus
        try:
            grouped_sorted = grouped.sort_values("Stimulus ID")
            fig.add_trace(
                go.Scatter(
                    x=grouped_sorted["Stimulus ID"].astype(str),
                    y=grouped_sorted["Median_N_Licks"],
                    mode='lines',
                    name='Median',
                    line=dict(color="black", width=3, shape='spline'),  # Use 'spline' for smoother interpolation
                    showlegend=True,
                )

            )
            # Add annotation under each tick with the count of trials
            for i, (_, row) in enumerate(grouped_sorted.iterrows()):
                fig.add_annotation(
                    x=i,  # use index position on categorical axis
                    y=-2,  # slightly below the min y
                    text=f"n={int(row['N_Trials'])}",
                    showarrow=False,
                    font=dict(
                        size=colors.LABEL_FONT_SIZE,
                        color="black"
                    ),
                    xanchor="center",
                    yanchor="top"
                )

        except Exception:
            pass

        fig.update_layout(
            title="N Licks by Stimulus",
            xaxis_title="Stimulus ID",
            yaxis_title="N Licks",
            template="simple_white",
            showlegend=True,
            xaxis=dict(
                categoryorder='array',
                categoryarray=[str(x) for x in stim_ids_sorted]
            )
        )
        colors.apply_standard_font_sizes(fig)

    if plot:
        st.subheader("N Licks by Stimulus")
        with st.expander("Licks", expanded=False):
            st.dataframe(licks)

        with st.expander("Per-trial counts", expanded=False):
            st.dataframe(per_trial_df)

        with st.expander("Grouped summary", expanded=False):
            st.dataframe(grouped)
        st.plotly_chart(fig, use_container_width=True, config=get_plotly_config())
        return per_trial_df, grouped, fig
    else:
        return per_trial_df, grouped, fig

# Function to create learning curve with interactivity
def learning_curve(selected_data, index=0):
    # Get the data from the responses function
    data = responses(selected_data, index)
    # Melt the data to long format for Altair
    data_melted = pd.melt(data.reset_index(), id_vars = "index", var_name = "Response Type", value_name = "Value")

    # Create the Altair line chart with custom colors and interactivity
    chart = alt.Chart(data_melted).mark_line().encode(
        x = alt.X('index:Q', title = 'Trials'),
        y = alt.Y('Value:Q', title = 'Cumulative Responses'),
        color = alt.Color('Response Type:N', scale = alt.Scale(
            domain = list(OUTCOME_COLOR_MAP.keys()),
            range = [OUTCOME_COLOR_MAP[k] for k in OUTCOME_COLOR_MAP.keys()]
        )),
        tooltip = ['index', 'Response Type', 'Value']
    ).properties(
        title = "Mouse Performance Learning Curve"
    ).interactive()  # Enable zooming and panning

    # Remove the grid lines for a clean look
    chart = chart.configure_axis(grid = False)

    # Display the interactive chart in Streamlit
    st.altair_chart(chart, use_container_width = True)

# First-lick latency analysis: only use values in [0, 2.5] s (response window)
FIRST_LICK_LATENCY_MAX_S = 2.5


def get_reinforcement_delay_seconds(project_data: pd.DataFrame, row_idx: int) -> float | None:
    """
    Return reinforcement delay in seconds for the given session (row).
    Uses Bpod States when available, else Educage parameters.txt or default.
    """
    try:
        if "States" not in project_data.columns:
            raise ValueError("No States column")
        states_str = project_data.iloc[row_idx]["States"]
        pattern = r"\['(.*?)' array\(\[(.*?)\]\)\]"
        matches = re.findall(pattern, states_str)
        data = [(name, np.array(list(map(float, values.split(","))))) for name, values in matches]
        states_array = np.array(data, dtype=object)
        state_idx = np.where(states_array[:, 0] == "ReinforsmentDelay")[0]
        if state_idx.size == 0:
            raise ValueError("ReinforsmentDelay state not found")
        tone_onset = states_array[state_idx[0] - 1, 1][0][0]
        reinforsment_delay_end = round(states_array[state_idx[0], 1][0][1] - tone_onset, 3)
        return float(reinforsment_delay_end)
    except Exception:
        pass
    # Educage fallback
    stim_dur = 0.3
    reinforsment_delay_dur = 0.001
    params_path = _find_parameters_txt_for_row(project_data, row_idx)
    reinforcement_delay_time = _read_last_reinforcement_delay_time_seconds(params_path) if params_path else None
    if reinforcement_delay_time is not None:
        return float(reinforcement_delay_time)
    return stim_dur + reinforsment_delay_dur


def hellinger_first_lick_distributions(
    go_times: np.ndarray, nogo_times: np.ndarray, n_bins: int = 15
) -> float:
    """
    Hellinger distance between Go and NoGo first-lick latency distributions.
    Bins samples on [0, FIRST_LICK_LATENCY_MAX_S], normalizes to probabilities,
    then H = (1/sqrt(2)) * sqrt(sum((sqrt(p_i) - sqrt(q_i))^2)). Returns np.nan if either sample is empty.
    """
    if go_times is None or nogo_times is None:
        return np.nan
    go_times = np.asarray(go_times, dtype=float)
    nogo_times = np.asarray(nogo_times, dtype=float)
    go_times = go_times[np.isfinite(go_times)]
    nogo_times = nogo_times[np.isfinite(nogo_times)]
    if go_times.size == 0 or nogo_times.size == 0:
        return np.nan
    bins = np.linspace(0, FIRST_LICK_LATENCY_MAX_S, n_bins + 1)
    p_hist, _ = np.histogram(go_times, bins=bins)
    q_hist, _ = np.histogram(nogo_times, bins=bins)
    p = p_hist / p_hist.sum()
    q = q_hist / q_hist.sum()
    # Small epsilon to avoid sqrt(0) gradient issues; renormalize
    eps = 1e-10
    p = (p + eps) / (p + eps).sum()
    q = (q + eps) / (q + eps).sum()
    H = np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))
    return float(H)


def plot_first_lick_latency(
    selected_data,
    index: int = 0,
    df_go_first_licks: pd.DataFrame | None = None,
    df_no_go_first_licks: pd.DataFrame | None = None,
    plot: bool = True,
    filter_early_response: bool | None = None,
):
    """
    Measures the latency of the first lick in each trial and compares Go vs NoGo trials.
    Creates a half violin plot to visualize the distribution of latencies.
    
    Args:
        selected_data (pd.DataFrame): DataFrame containing experiment data
        index (int): Index of the session to analyze
        df_go_first_licks (pd.DataFrame): Go trial first lick data
        df_no_go_first_licks (pd.DataFrame): NoGo trial first lick data
        plot (bool): Whether to display the plot
    """
    import numpy as np
    import ast
    from Analysis.GNG_bpod_analysis.colors import COLOR_GO, COLOR_NOGO, COLOR_GRAY
    if filter_early_response is None:
        filter_early_response = get_global_early_response_filter()

    # If first lick data is not provided, calculate it
    if df_go_first_licks is None or df_no_go_first_licks is None:
        # Get the data from process_and_plot_lick_data (respect Early Response filter flag)
        df_go_first_licks, df_no_go_first_licks, _ = process_and_plot_lick_data(
            selected_data, index, plot=False, filter_early_response=filter_early_response
        )
    
    # Check if we have valid data
    if df_go_first_licks is None or df_no_go_first_licks is None:
        st.warning("No valid first lick data available for analysis.")
        return None
    
    df_first_licks = pd.concat([df_go_first_licks, df_no_go_first_licks])
    # Restrict to 0–2.5 s for first-lick latency analysis
    col = "First Lick Time (s)"
    df_first_licks = df_first_licks[(df_first_licks[col] >= 0) & (df_first_licks[col] <= FIRST_LICK_LATENCY_MAX_S)]
    
    # Display the first lick data
    st.write("**First Lick Times by Trial:**")


    
    # Create half violin plot using plotly
    fig = go.Figure()
    
    # Get separate data for each trial type
    go_data = df_first_licks[df_first_licks["Trial Type"] == "Go"]["First Lick Time (s)"].values
    nogo_data = df_first_licks[df_first_licks["Trial Type"] == "NoGo"]["First Lick Time (s)"].values
    
    if len(go_data) > 0:
        # Create left side for Go trials
        fig.add_trace(go.Violin(
            y=go_data,
            x=[0] * len(go_data),  # Go data at x=0
            name="Go",
            box_visible=True,
            meanline_visible=True,
            fillcolor=COLOR_GO,
            line_color=COLOR_GO,
            opacity=0.7,
            side='negative'  # Left side for Go trials
        ))
    
    if len(nogo_data) > 0:
        # Create right side for NoGo trials
        fig.add_trace(go.Violin(
            y=nogo_data,
            x=[0] * len(nogo_data),  # NoGo data at x=0
            name="NoGo",
            box_visible=True,
            meanline_visible=True,
            fillcolor=COLOR_NOGO,
            line_color=COLOR_NOGO,
            opacity=0.7,
            side='positive'  # Right side for NoGo trials
        ))
    fig.update_traces(meanline_visible=True,
                  points='all', # show all points
                  jitter=0.1,  # add some jitter on points for better visibility
                  scalemode='count') #scale violin plot area with total count
    # Update layout
    fig.update_layout(
        title="First Lick Latency Distribution",
        yaxis_title="Latency (s)",
        xaxis_title="",
        showlegend=True,
        height=500,
        violinmode='overlay',
        violingroupgap=0,
        violingap=0,
        xaxis=dict(
            showticklabels=False,
            range=[-1, 1],
            showgrid=False
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=COLOR_GRAY,
            gridwidth=0.5,
            range=[0, FIRST_LICK_LATENCY_MAX_S]
        )
    )
    
    # Add statistics
    go_latencies = df_first_licks[df_first_licks["Trial Type"] == "Go"]["First Lick Time (s)"].values
    no_go_latencies = df_first_licks[df_first_licks["Trial Type"] == "NoGo"]["First Lick Time (s)"].values
    
    if len(go_latencies) > 0 and len(no_go_latencies) > 0:
        from scipy.stats import ks_2samp
        ks_stat, ks_p = ks_2samp(go_latencies, no_go_latencies, alternative="two-sided")
        h = hellinger_first_lick_distributions(go_latencies, no_go_latencies)

        st.write(f"**Statistics:**")
        st.write(f"- Go trials: n={len(go_latencies)}, mean={np.mean(go_latencies):.3f}s ± {np.std(go_latencies):.3f}s")
        st.write(f"- NoGo trials: n={len(no_go_latencies)}, mean={np.mean(no_go_latencies):.3f}s ± {np.std(no_go_latencies):.3f}s")
        st.write(f"- Kolmogorov-Smirnov: D={ks_stat:.3f}, p={ks_p:.3g}")
        st.write(f"- Hellinger distance (Go vs NoGo) first-lick latency: {h:.4f}")

    if len(go_latencies) > 0:
        reinforcement_delay = get_reinforcement_delay_seconds(selected_data, index)
        if reinforcement_delay is not None:
            mean_go = float(np.mean(go_latencies))
            dist = abs(mean_go - reinforcement_delay)
            st.write(f"**Convergence to reinforcement delay:**")
            st.write(f"- Reinforcement delay: {reinforcement_delay:.3f} s")
            st.write(f"- Mean Go first lick: {mean_go:.3f} s")
            st.write(f"- Distance (mean Go first lick to reinforcement delay): {dist:.3f} s")
        else:
            st.caption("Reinforcement delay not available for this session.")
    
    colors.apply_standard_font_sizes(fig)
    st.plotly_chart(fig, use_container_width=True, config=get_plotly_config())
    
    # return df_first_licks

def plot_first_lick_latency_multiple_sessions(selected_data, animal_name="None", plot=True):
    """
    Calculate and plot the mean and standard deviation of first lick latency across multiple sessions.
    Uses half violin plots to show the distribution for each session.
    
    Args:
        selected_data (pd.DataFrame): DataFrame containing experiment data
        animal_name (str): Name of the animal to analyze
        plot (bool): Whether to display the plot
    """
    from Analysis.GNG_bpod_analysis.GNG_bpod_general import get_sessions_for_animal
    from Analysis.GNG_bpod_analysis.colors import COLOR_GO, COLOR_NOGO, COLOR_GRAY
    import pandas as pd
    import numpy as np
    import streamlit as st
    import plotly.graph_objects as go

    session_indices, session_dates = get_sessions_for_animal(selected_data, animal_name)
    session_results = []

    for idx, session_idx in enumerate(session_indices):
        try:
            df_go_first_licks, df_no_go_first_licks, _ = process_and_plot_lick_data(selected_data, session_idx, plot=False)
            
            go_latencies = df_go_first_licks["First Lick Time (s)"].values if not df_go_first_licks.empty else np.array([])
            go_latencies = go_latencies[(go_latencies >= 0) & (go_latencies <= FIRST_LICK_LATENCY_MAX_S)] if go_latencies.size else go_latencies
            go_mean = np.mean(go_latencies) if len(go_latencies) > 0 else np.nan
            go_std = np.std(go_latencies) if len(go_latencies) > 0 else np.nan
            go_count = len(go_latencies)
            
            nogo_latencies = df_no_go_first_licks["First Lick Time (s)"].values if not df_no_go_first_licks.empty else np.array([])
            nogo_latencies = nogo_latencies[(nogo_latencies >= 0) & (nogo_latencies <= FIRST_LICK_LATENCY_MAX_S)] if nogo_latencies.size else nogo_latencies
            nogo_mean = np.mean(nogo_latencies) if len(nogo_latencies) > 0 else np.nan
            nogo_std = np.std(nogo_latencies) if len(nogo_latencies) > 0 else np.nan
            nogo_count = len(nogo_latencies)
            
            session_results.append({
                'Session Index': idx + 1,
                'Session Date': session_dates[idx],
                'Go Mean': go_mean,
                'Go Std': go_std,
                'Go Count': go_count,
                'NoGo Mean': nogo_mean,
                'NoGo Std': nogo_std,
                'NoGo Count': nogo_count,
                'Go Latencies': go_latencies,
                'NoGo Latencies': nogo_latencies
            })
        except Exception as e:
            print(f"Error processing session {session_idx}: {e}")
            continue
    
    if not session_results:
        st.warning(f"No valid sessions found for {animal_name}")
        return None
    
    results_df = pd.DataFrame(session_results)
    
    if plot:
        st.title(f"First Lick Latency Distribution for {animal_name}")
        fig = go.Figure()
        
        # Plot half violin plots for each session
        for idx, row in results_df.iterrows():
            session_idx = row['Session Index']
            
            # Go trials - left side
            go_data = row['Go Latencies']
            if len(go_data) > 0:
                fig.add_trace(go.Violin(
                    y=go_data,
                    x=[session_idx] * len(go_data),
                    name=f"Go (Session {session_idx})",
                    box_visible=True,
                    meanline_visible=True,
                    fillcolor=COLOR_GO,
                    line_color=COLOR_GO,
                    opacity=0.15,
                    side='negative',
                    legendgroup=f"session_{session_idx}",
                    showlegend=False
                ))
            
            # NoGo trials - right side
            nogo_data = row['NoGo Latencies']
            if len(nogo_data) > 0:
                fig.add_trace(go.Violin(
                    y=nogo_data,
                    x=[session_idx] * len(nogo_data),
                    name=f"NoGo (Session {session_idx})",
                    box_visible=True,
                    meanline_visible=True,
                    fillcolor=COLOR_NOGO,
                    line_color=COLOR_NOGO,
                    opacity=0.15,
                    side='positive',
                    legendgroup=f"session_{session_idx}",
                    showlegend=False
                ))
        
        # Add legend entries for Go and NoGo
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(color=COLOR_GO, size=10),
            name='Go Trials',
            showlegend=True
        ))
        
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(color=COLOR_NOGO, size=10),
            name='NoGo Trials',
            showlegend=True
        ))
        
        # Add average lines on top of violin plots
        go_means = results_df['Go Mean'].dropna()
        nogo_means = results_df['NoGo Mean'].dropna()
        
        if len(go_means) > 0:
            fig.add_trace(go.Scatter(
                x=go_means.index + 1,  # Session indices
                y=go_means.values,
                mode='lines+markers',
                name='Go Average',
                line=dict(color=COLOR_GO, width=2),
                marker=dict(color=COLOR_GO, size=6),
                showlegend=False
            ))
        
        if len(nogo_means) > 0:
            fig.add_trace(go.Scatter(
                x=nogo_means.index + 1,  # Session indices
                y=nogo_means.values,
                mode='lines+markers',
                name='NoGo Average',
                line=dict(color=COLOR_NOGO, width=2),
                marker=dict(color=COLOR_NOGO, size=6),
                showlegend=False
            ))
        
        # Add gray vertical line at 0.3 seconds
        fig.add_hline(
            y=0.3,
            line_dash="dash",
            line_color="gray",
            line_width=1,
            annotation_text="Reinforcement Delay",
            annotation_position="bottom right"
        )
        
        fig.update_layout(
            title=f"First Lick Latency Distribution for {animal_name}",
            xaxis_title="Session Index",
            yaxis_title="First Lick Latency (s)",
            showlegend=True,
            height=600,
            violinmode='overlay',
            violingroupgap=0,
            violingap=0,
            xaxis=dict(
                tickmode='linear',
                tick0=1,
                dtick=1,
                showgrid=True,
                gridcolor=COLOR_GRAY,
                gridwidth=0.2
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor=COLOR_GRAY,
                gridwidth=0.2,
                range=[0, FIRST_LICK_LATENCY_MAX_S]
            )
        )
        colors.apply_standard_font_sizes(fig)
        st.plotly_chart(fig, use_container_width=True, config=get_plotly_config())
        
    
    return results_df


def plot_first_lick_hellinger_first_vs_last_day(project_data):
    """
    Multi-animal: compare Hellinger(Go vs NoGo) first-lick on first day vs last day.
    First day = first session per animal where the NoGo first-lick array is non-empty;
    last day = last session per animal. Groups all animals together and runs paired
    statistical comparison (Wilcoxon signed-rank).
    """
    from Analysis.GNG_bpod_analysis.GNG_bpod_general import get_sessions_for_animal
    from scipy.stats import wilcoxon

    animals = project_data["MouseName"].unique()
    filter_early = get_global_early_response_filter()
    rows = []

    for animal in animals:
        session_indices, session_dates = get_sessions_for_animal(project_data, animal)
        if len(session_indices) == 0:
            continue

        # First day = first session with non-empty NoGo first-lick array (after 0–2.5 s filter)
        first_idx = None
        for idx in session_indices:
            try:
                _, g_nogo, _ = process_and_plot_lick_data(
                    project_data, idx, plot=False, filter_early_response=filter_early
                )
                nogo_t = g_nogo["First Lick Time (s)"].values if g_nogo is not None and not g_nogo.empty else np.array([])
                nogo_t = nogo_t[(nogo_t >= 0) & (nogo_t <= FIRST_LICK_LATENCY_MAX_S)] if nogo_t.size else nogo_t
                if nogo_t.size > 0:
                    first_idx = idx
                    break
            except Exception:
                continue
        if first_idx is None:
            continue

        last_idx = session_indices[-1]
        if first_idx == last_idx:
            continue  # need two distinct days for comparison

        first_w = np.nan
        last_w = np.nan
        try:
            g_go, g_nogo, _ = process_and_plot_lick_data(
                project_data, first_idx, plot=False, filter_early_response=filter_early
            )
            go_t = g_go["First Lick Time (s)"].values if g_go is not None and not g_go.empty else np.array([])
            nogo_t = g_nogo["First Lick Time (s)"].values if g_nogo is not None and not g_nogo.empty else np.array([])
            go_t = go_t[(go_t >= 0) & (go_t <= FIRST_LICK_LATENCY_MAX_S)] if go_t.size else go_t
            nogo_t = nogo_t[(nogo_t >= 0) & (nogo_t <= FIRST_LICK_LATENCY_MAX_S)] if nogo_t.size else nogo_t
            first_h = hellinger_first_lick_distributions(go_t, nogo_t)
        except Exception:
            pass
        try:
            g_go, g_nogo, _ = process_and_plot_lick_data(
                project_data, last_idx, plot=False, filter_early_response=filter_early
            )
            go_t = g_go["First Lick Time (s)"].values if g_go is not None and not g_go.empty else np.array([])
            nogo_t = g_nogo["First Lick Time (s)"].values if g_nogo is not None and not g_nogo.empty else np.array([])
            go_t = go_t[(go_t >= 0) & (go_t <= FIRST_LICK_LATENCY_MAX_S)] if go_t.size else go_t
            nogo_t = nogo_t[(nogo_t >= 0) & (nogo_t <= FIRST_LICK_LATENCY_MAX_S)] if nogo_t.size else nogo_t
            last_h = hellinger_first_lick_distributions(go_t, nogo_t)
        except Exception:
            pass

        rows.append({
            "MouseName": animal,
            "first_day_hellinger": first_h,
            "last_day_hellinger": last_h,
        })

    if not rows:
        st.warning("No animals with at least 2 sessions and valid first-lick data.")
        return

    df = pd.DataFrame(rows)
    valid = df.dropna(subset=["first_day_hellinger", "last_day_hellinger"])
    if valid.empty:
        st.warning("No animals with both first-day and last-day Hellinger.")
        return

    # Grouped comparison: all animals together, First day vs Last day
    first_vals = valid["first_day_hellinger"].values
    last_vals = valid["last_day_hellinger"].values
    try:
        stat, p_value = wilcoxon(first_vals, last_vals, alternative="two-sided")
    except Exception:
        stat, p_value = np.nan, np.nan

    st.write("**Grouped comparison (all animals)**")
    st.write(f"- First day: n={len(first_vals)}, median={np.nanmedian(first_vals):.4f}, mean={np.nanmean(first_vals):.4f}")
    st.write(f"- Last day: n={len(last_vals)}, median={np.nanmedian(last_vals):.4f}, mean={np.nanmean(last_vals):.4f}")
    st.write(f"- Wilcoxon signed-rank (first vs last): statistic={stat:.4f}, p={p_value:.3g}")

    # Box plot: First day vs Last day (all animals pooled)
    fig = go.Figure()
    fig.add_trace(go.Box(
        y=first_vals,
        name="First day",
        marker_color=COLOR_ACCENT,
        boxpoints="all",
        jitter=0.2,              # small jitter for separation
        pointpos=0,              # points on top of the box
    ))
    fig.add_trace(go.Box(
        y=last_vals,
        name="Last day",
        marker_color=COLOR_ACCENT,
        boxpoints="all",
        jitter=0.2,              # small jitter for separation
        pointpos=0,              # points on top of the box
    ))
    fig.update_layout(
        title="First Lick Hellinger: First Day vs Last Day (all animals)",
        yaxis_title="Hellinger distance",
        showlegend=False,
        height=600,
    )
    colors.apply_standard_font_sizes(fig)
    st.plotly_chart(fig, use_container_width=True, config=get_plotly_config())

    # Per-animal grouped bars
    st.write("**Per-animal values**")
    fig2 = go.Figure()
    x = np.arange(len(df))
    width = 0.35
    fig2.add_trace(go.Bar(
        x=x - width / 2,
        y=df["first_day_hellinger"],
        name="First day",
        marker_color=COLOR_GO,
        width=width,
    ))
    fig2.add_trace(go.Bar(
        x=x + width / 2,
        y=df["last_day_hellinger"],
        name="Last day",
        marker_color=COLOR_NOGO,
        width=width,
    ))
    fig2.update_layout(
        title="First Lick Hellinger: First Day vs Last Day (per animal)",
        xaxis_title="Animal",
        yaxis_title="Hellinger distance",
        barmode="group",
        xaxis=dict(tickvals=x, ticktext=df["MouseName"].tolist()),
        showlegend=False,
        height=400,
    )
    colors.apply_standard_font_sizes(fig2)
    st.plotly_chart(fig2, use_container_width=True, config=get_plotly_config())
    st.dataframe(df, use_container_width=True, hide_index=True)


def plot_go_first_lick_distance_to_reinforcement_first_vs_last_day(project_data):
    """
    Multi-animal: distance from mean(Go first lick) to reinforcement delay on first day vs last day.
    First day = first session with non-empty Go first licks (0–2.5 s); last day = last session.
    Groups all animals and runs Wilcoxon signed-rank; box plot + per-animal bars + table.
    """
    from Analysis.GNG_bpod_analysis.GNG_bpod_general import get_sessions_for_animal
    from scipy.stats import wilcoxon

    animals = project_data["MouseName"].unique()
    filter_early = get_global_early_response_filter()
    rows = []

    for animal in animals:
        session_indices, _ = get_sessions_for_animal(project_data, animal)
        if len(session_indices) == 0:
            continue

        # First day = first session with non-empty Go first licks (0–2.5 s)
        first_idx = None
        for idx in session_indices:
            try:
                g_go, _, _ = process_and_plot_lick_data(
                    project_data, idx, plot=False, filter_early_response=filter_early
                )
                go_t = g_go["First Lick Time (s)"].values if g_go is not None and not g_go.empty else np.array([])
                go_t = go_t[(go_t >= 0) & (go_t <= FIRST_LICK_LATENCY_MAX_S)] if go_t.size else go_t
                if go_t.size > 0:
                    first_idx = idx
                    break
            except Exception:
                continue
        if first_idx is None:
            continue

        last_idx = session_indices[-1]
        if first_idx == last_idx:
            continue

        first_dist = np.nan
        last_dist = np.nan
        try:
            g_go, _, _ = process_and_plot_lick_data(
                project_data, first_idx, plot=False, filter_early_response=filter_early
            )
            go_t = g_go["First Lick Time (s)"].values if g_go is not None and not g_go.empty else np.array([])
            go_t = go_t[(go_t >= 0) & (go_t <= FIRST_LICK_LATENCY_MAX_S)] if go_t.size else go_t
            mean_go = float(np.mean(go_t)) if go_t.size else np.nan
            rd = get_reinforcement_delay_seconds(project_data, first_idx)
            if rd is not None and not np.isnan(mean_go):
                first_dist = abs(mean_go - rd)
        except Exception:
            pass
        try:
            g_go, _, _ = process_and_plot_lick_data(
                project_data, last_idx, plot=False, filter_early_response=filter_early
            )
            go_t = g_go["First Lick Time (s)"].values if g_go is not None and not g_go.empty else np.array([])
            go_t = go_t[(go_t >= 0) & (go_t <= FIRST_LICK_LATENCY_MAX_S)] if go_t.size else go_t
            mean_go = float(np.mean(go_t)) if go_t.size else np.nan
            rd = get_reinforcement_delay_seconds(project_data, last_idx)
            if rd is not None and not np.isnan(mean_go):
                last_dist = abs(mean_go - rd)
        except Exception:
            pass

        rows.append({
            "MouseName": animal,
            "first_day_distance": first_dist,
            "last_day_distance": last_dist,
        })

    if not rows:
        st.warning("No animals with at least 2 sessions and valid Go first-lick data.")
        return

    df = pd.DataFrame(rows)
    valid = df.dropna(subset=["first_day_distance", "last_day_distance"])
    if valid.empty:
        st.warning("No animals with both first-day and last-day distance to reinforcement delay.")
        return

    first_vals = valid["first_day_distance"].values
    last_vals = valid["last_day_distance"].values
    try:
        stat, p_value = wilcoxon(first_vals, last_vals, alternative="two-sided")
    except Exception:
        stat, p_value = np.nan, np.nan

    st.write("**Grouped comparison (all animals)**")
    st.write(f"- First day: n={len(first_vals)}, median={np.nanmedian(first_vals):.3f} s, mean={np.nanmean(first_vals):.3f} s")
    st.write(f"- Last day: n={len(last_vals)}, median={np.nanmedian(last_vals):.3f} s, mean={np.nanmean(last_vals):.3f} s")
    st.write(f"- Wilcoxon signed-rank (first vs last): statistic={stat:.4f}, p={p_value:.3g}")

    fig = go.Figure()
    # Show points on top of the box, with a small jitter for better visibility
    fig.add_trace(go.Box(
        y=first_vals,
        name="First day",
        marker_color=COLOR_GO,
        boxpoints="all",         # show all points
        jitter=0.2,              # small jitter for separation
        pointpos=0,              # points on top of the box
        marker=dict(size=6),
    ))
    fig.add_trace(go.Box(
        y=last_vals,
        name="Last day",
        marker_color=COLOR_GO,
        boxpoints="all",
        jitter=0.2,              # small jitter for separation
        pointpos=0,              # points on top of the box
        marker=dict(size=6),
    ))
    fig.update_layout(
        title="Go first-lick distance to reinforcement delay: First vs Last Day (all animals)",
        yaxis_title="Distance (s)",
        showlegend=False,
        height=600,
    )
    fig.update_layout(
        title="Go first-lick distance to reinforcement delay: First vs Last Day (all animals)",
        yaxis_title="Distance (s)",
        showlegend=False,
        height=600,
    )
    colors.apply_standard_font_sizes(fig)
    st.plotly_chart(fig, use_container_width=True, config=get_plotly_config())

    st.write("**Per-animal values**")
    fig2 = go.Figure()
    x = np.arange(len(df))
    width = 0.35
    fig2.add_trace(go.Bar(x=x - width / 2, y=df["first_day_distance"], name="First day", marker_color=COLOR_GO, width=width))
    fig2.add_trace(go.Bar(x=x + width / 2, y=df["last_day_distance"], name="Last day", marker_color=COLOR_NOGO, width=width))
    fig2.update_layout(
        title="Go first-lick distance to reinforcement delay (per animal)",
        xaxis_title="Animal",
        yaxis_title="Distance (s)",
        barmode="group",
        xaxis=dict(tickvals=x, ticktext=df["MouseName"].tolist()),
        showlegend=True,
        height=400,
    )
    colors.apply_standard_font_sizes(fig2)
    st.plotly_chart(fig2, use_container_width=True, config=get_plotly_config())
    st.dataframe(df, use_container_width=True, hide_index=True)


def _parse_start_times(start_times):
    """
    Parse start times from various formats and return list of time objects.
    """
    if pd.isna(start_times) or not start_times:
        return []
    
    try:
        if isinstance(start_times, str):
            start_times_list = ast.literal_eval(start_times)
        else:
            start_times_list = start_times
        
        # Convert to datetime objects
        times = []
        for time_str in start_times_list:
            try:
                # Parse time string like '11:49:57.236554'
                time_obj = pd.to_datetime(time_str, format='%H:%M:%S.%f').time()
                times.append(time_obj)
            except:
                try:
                    # Fallback for different format
                    time_obj = pd.to_datetime(time_str).time()
                    times.append(time_obj)
                except:
                    continue
        
        return times
    except Exception:
        return []

def _create_time_bins(bin_size_minutes=30):
    """
    Create time bins for the day and return bins and labels.
    """
    bins = []
    bin_labels = []
    
    for hour in range(24):
        for minute in range(0, 60, bin_size_minutes):
            start_time = pd.Timestamp.combine(pd.Timestamp.today().date(), 
                                            pd.Timestamp(f"{hour:02d}:{minute:02d}:00").time())
            end_time = start_time + pd.Timedelta(minutes=bin_size_minutes)
            bins.append((start_time.time(), end_time.time()))
            bin_labels.append(f"{hour:02d}:{minute:02d}")
    
    return bins, bin_labels

def _count_trials_in_bins(times, bins):
    """
    Count trials in each time bin.
    """
    bin_counts = [0] * len(bins)
    
    for time_obj in times:
        for i, (bin_start, bin_end) in enumerate(bins):
            if bin_start <= time_obj < bin_end:
                bin_counts[i] += 1
                break
    
    return bin_counts

def _create_x_values(bin_labels):
    """
    Convert bin labels to datetime for proper x-axis.
    """
    return [pd.Timestamp(f"2024-01-01 {label}:00") for label in bin_labels]

def daily_activity_single_animal(project_data, index, plot=False):
    """
    Plot daily activity for a single animal showing trial counts over time of day (15-min bins).
    """
    if project_data is None or project_data.empty:
        st.info("No data loaded.")
        return
    
    if "StartTime" not in project_data.columns:
        st.info("No StartTime data available for activity analysis.")
        return
    
    # Get the specific session data
    session_data = project_data.iloc[index]
    start_times = session_data["StartTime"]
    
    if pd.isna(start_times) or not start_times:
        st.info("No start time data available for this session.")
        return
    
    # Parse start times using helper function
    times = _parse_start_times(start_times)
    if not times:
        st.info("Could not parse start times.")
        return
    
    # Create time bins using helper function
    bin_size_minutes = 30
    bins, bin_labels = _create_time_bins(bin_size_minutes)
    
    # Count trials in each bin using helper function
    bin_counts = _count_trials_in_bins(times, bins)
    
    if plot:
        # Create the plot
        fig = go.Figure()
        
        # Convert bin labels to datetime for proper x-axis
        x_values = _create_x_values(bin_labels)
        
        fig.add_trace(go.Scatter(
            x=x_values,
            y=bin_counts,
            mode='lines+markers',
            name="Trial Count",
            line=dict(color=COLOR_ACCENT, width=2),
            marker=dict(size=4, color=COLOR_ACCENT),
            opacity=0.3
        ))
        
        # Add average line
        avg_count = np.mean(bin_counts)
        fig.add_hline(
            y=avg_count,
            line_dash="solid",
            line_color=COLOR_ACCENT,
            line_width=4,
            opacity=0.8,
            annotation_text=f"Average: {avg_count:.1f}",
            annotation_position="top right"
        )
        
        # Update layout
        fig.update_layout(
            title=f"Daily Activity Pattern - {session_data['MouseName']} ({session_data['SessionDate']})",
            xaxis_title="Time of Day",
            yaxis_title=f"Number of Trials ({bin_size_minutes}-min bins)",
            xaxis=dict(
                tickformat="%H:%M",
                tickmode='array',
                tickvals=x_values[::4],  # Show every 4th tick (hourly)
                ticktext=[label for i, label in enumerate(bin_labels) if i % 4 == 0]
            ),
            height=500,
            width=900,
            showlegend=False
        )
        colors.apply_standard_font_sizes(fig)
        st.plotly_chart(fig, use_container_width=True, config=get_plotly_config())

def daily_activity_multi_animal(project_data):
    """
    Plot daily activity for multiple animals on a selected date, showing trial counts over time of day.
    """
    if project_data is None or project_data.empty:
        st.info("No data loaded.")
        return
    
    if "StartTime" not in project_data.columns:
        st.info("No StartTime data available for activity analysis.")
        return
    
    # Get unique dates - force refresh by creating a new list
    dates = sorted(project_data["SessionDate"].astype(str).unique())
    if len(dates) == 0:
        st.info("No dates found in data.")
        return
    
    # Use a unique key to force refresh
    selected_date = st.selectbox("Select a date", options=dates, 
                                index=max(0, len(dates) - 1), 
                                key=f"daily_activity_date_{len(project_data)}")
    
    # Filter data for selected date
    date_data = project_data[project_data["SessionDate"].astype(str) == str(selected_date)]
    
    if date_data.empty:
        st.info(f"No data found for date {selected_date}")
        return
    
    # Get unique mice for this date
    mice = sorted(date_data["MouseName"].unique())
    if len(mice) == 0:
        st.info("No animals found for selected date.")
        return
    
    # Create time bins using helper function
    bin_size_minutes = 30
    bins, bin_labels = _create_time_bins(bin_size_minutes)
    
    fig = go.Figure()
    
    # Convert bin labels to datetime for proper x-axis
    x_values = _create_x_values(bin_labels)
    
    # Determine per-mouse colors
    try:
        color_map = st.session_state.get('mouse_color_map', {})
        if not color_map:
            from Analysis.GNG_bpod_analysis.colors import get_subject_color_map
            color_map = get_subject_color_map(project_data['MouseName'])
    except Exception:
        color_map = {}
    
    # Store all bin counts for average calculation
    all_bin_counts = []
    
    for i, mouse in enumerate(mice):
        mouse_data = date_data[date_data["MouseName"] == mouse]
        if len(mouse_data) == 0:
            continue
            
        # Get start times for this mouse
        start_times = mouse_data.iloc[0]["StartTime"]
        
        # Parse start times using helper function
        times = _parse_start_times(start_times)
        if not times:
            continue
        
        # Count trials in each bin using helper function
        bin_counts = _count_trials_in_bins(times, bins)
        all_bin_counts.append(bin_counts)
        
        # Add trace for this mouse as stacked bar
        fig.add_trace(go.Bar(
            x=x_values,
            y=bin_counts,
            name=str(mouse),
            marker=dict(color=color_map.get(str(mouse), colors.COLOR_SUBTLE)),
            opacity=0.7
        ))
    
    if len(fig.data) == 0:
        st.info("No activity data found for any animals on selected date.")
        return
    
    # Calculate and add average line across all animals
    if all_bin_counts:
        avg_bin_counts = np.mean(all_bin_counts, axis=0)
        fig.add_trace(go.Scatter(
            x=x_values,
            y=avg_bin_counts,
            mode='lines',
            name='Average',
            line=dict(color='black', width=4),
            opacity=0.9
        ))
    
    # Update layout
    fig.update_layout(
        title=f"Daily Activity Pattern by Animal — {selected_date} ({bin_size_minutes}-min bins)",
        xaxis_title="Time of Day",
        yaxis_title="Number of Trials",
        barmode='stack',  # Stack the bars
        xaxis=dict(
            tickformat="%H:%M",
            tickmode='array',
            tickvals=x_values[::4],  # Show every 4th tick (hourly)
            ticktext=[label for i, label in enumerate(bin_labels) if i % 4 == 0]
        ),
        height=500,
        width=900,
        showlegend=True
    )
    colors.apply_standard_font_sizes(fig)
    st.plotly_chart(fig, use_container_width=True, config=get_plotly_config())

def daily_multi_animal_lick_rate(project_data, t=15):
    """
    Plot lick rate data for all unique mice on a selected date, overlaid on the same plot.
    Uses the same logic as the licking_rate function but for multiple animals.
    """
    if project_data is None or project_data.empty:
        st.info("No data loaded.")
        return

    # Get unique dates - force refresh by creating a new list
    dates = sorted(project_data["SessionDate"].astype(str).unique())
    if len(dates) == 0:
        st.info("No dates found in data.")
        return

    # Use a unique key to force refresh
    selected_date = st.selectbox("Select a date", options=dates, 
                                index=max(0, len(dates) - 1), 
                                key=f"daily_multi_lick_date_{len(project_data)}")

    # Filter data for selected date
    date_data = project_data[project_data["SessionDate"].astype(str) == str(selected_date)]
    
    if date_data.empty:
        st.info(f"No data found for date {selected_date}")
        return
    # Get unique mice for this date
    mice = sorted(date_data["MouseName"].unique())
    if len(mice) == 0:
        st.info("No animals found for selected date.")
        return
    
    fig = go.Figure()

    # Determine per-mouse colors
    try:
        color_map = st.session_state.get('mouse_color_map', {})
        if not color_map:
            from Analysis.GNG_bpod_analysis.colors import get_subject_color_map
            color_map = get_subject_color_map(date_data['MouseName'])
    except Exception:
        color_map = {}

    for mouse in mice:
        mouse_data = date_data[date_data["MouseName"] == mouse]
        if len(mouse_data) == 0:
            continue
        # Compute Go hit-rate series using existing function and selected bin size t
        _, frac = licking_rate(mouse_data, index=0, t=t, plot=False)
        go_series = frac["Go"].dropna()
        if len(go_series) == 0:
            continue
        x = np.arange(1, len(go_series) + 1)
        fig.add_trace(go.Scatter(
            x=x,
            y=go_series.values,
            mode='lines',
            name=str(mouse),
            line=dict(width=2, color=color_map.get(str(mouse), colors.COLOR_SUBTLE))
        ))

    if len(fig.data) == 0:
        st.info("No lick data found for any animals on selected date.")
        return

    fig.update_layout(
        title=f"Go Hit Rate by Animal — {selected_date} (rolling window={t})",
        xaxis_title="Trial index",
        yaxis_title="Hit rate (%)",
        yaxis=dict(range=[0, 100]),
        height=500,
        width=900,
        showlegend=True
    )
    colors.apply_standard_font_sizes(fig)
    st.plotly_chart(fig, use_container_width=True, config=get_plotly_config())

def cumulative_number_of_trials_vs_daily_dprime(project_data, t=15):
    """
    Plot cumulative number of trials vs daily d' progression for all mice.
    
    Args:
        project_data (pd.DataFrame): DataFrame containing experiment data
        t (int): Bin size for d' calculation
    """    

    from Analysis.GNG_bpod_analysis.metric import d_prime_multiple_sessions
    # Prepare data: for each mouse, sum up number of trials over days, and plot daily d' vs cumulative trials
    mice = sorted(project_data["MouseName"].unique())

    fig = go.Figure()
    
    # Add horizontal line at y=1
    # Draw the horizontal line as a shape in the layout, so it appears in the background of the plot
    fig.update_layout(
        shapes=[
            dict(
                type="line",
                xref="paper", yref="y",
                x0=0, x1=1,
                y0=1, y1=1,
                line=dict(
                    color=COLOR_GRAY,
                    width=5,
                    dash="solid"
                ),
                layer="below"  # ensures it is in the background
            )
        ]
    )
    for mouse in mice:
        mouse_data = project_data[project_data["MouseName"] == mouse].sort_values("SessionDate")
        session_dates = mouse_data["SessionDate"].astype(str).values
        original_indices = mouse_data.index.values
        n_trials_per_day = []
        d_prime_per_day = []
        mouse_colors = []
        for i in range(len(mouse_data)):
            color = mouse_data.iloc[i]["Color"] if "Color" in mouse_data.columns else "gray"
            mouse_colors.append(color)
            stimuli, outcomes = preprocess_stimuli_outcomes(mouse_data, i)
            n_trials_per_day.append(len(stimuli))
        # Cumulative number of trials
        cumulative_trials = np.cumsum(n_trials_per_day)
        # Force recalculation by passing a fresh copy of the data
        data = d_prime_multiple_sessions(project_data.copy(), t=t, animal_name=mouse, plot=False)
        d_prime_per_day = data["d_prime"]
        n_t = data["tones_per_class"]
        n_b = data["Boundaries"]

        # Determine marker symbols: 'circle' if n_b == 1, 'square' if n_b == 2
        marker_symbols = ['circle' if nb == 1 else 'square' for nb in n_b]
        # Use shared helpers for marker sizes and legends
        marker_sizes = colors.marker_sizes_from_tones(n_t, scale=5.0, default_size=6.0)
        # Attach rich customdata for hover/click: [orig_index, session_date, tones, boundaries]
        customdata = np.column_stack([
            original_indices[:len(d_prime_per_day)],
            np.array(session_dates[:len(d_prime_per_day)], dtype=object),
            np.array(n_t[:len(d_prime_per_day)], dtype=object),
            np.array(n_b[:len(d_prime_per_day)], dtype=object)
        ])
        fig.add_trace(go.Scatter(
            x=cumulative_trials,
            y=d_prime_per_day,
            mode='lines+markers',
            customdata=customdata,
            marker=dict(
                color=mouse_colors[0],
                size=marker_sizes,
                symbol=marker_symbols,
                line=dict(
                    width=2,
                    color=mouse_colors[0]
                )
            ),
            name=str(mouse),
            text=[f"{n_t[i]}T | {n_b[i]}B" for i in range(len(d_prime_per_day))],
            hovertemplate=(
                "Mouse: %{name}<br>"
                "Date: %{customdata[1]} (idx %{customdata[0]})<br>"
                "Cumulative Trials: %{x}<br>"
                "d': %{y:.2f}<br>"
                "T|B: %{text}<extra></extra>"
            ),
            textposition="top center",
            showlegend=True
        ))

        # Add legend entries for marker shape (number of boundaries)
        if mouse == mice[0]:
            # Only add these once to avoid duplicate legend entries
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(symbol='circle', color='gray', size=8),
                name="1 Boundary",
                showlegend=True,
                hoverinfo='skip'
            ))
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(symbol='square', color='gray', size=8),
                name="2 Boundaries",
                showlegend=True,
                hoverinfo='skip'
            ))
            # Add legend entries for marker sizes via shared helper
            colors.add_marker_legends(fig, n_b, n_t, scale=5.0)

    fig.update_layout(
        xaxis_title="Cumulative Number of Trials",
        yaxis_title="Daily d'",
        title="Daily d' vs Cumulative Number of Trials per Mouse",
        plot_bgcolor="white",
        legend_title_text="Legend"
    )

    colors.apply_standard_font_sizes(fig)
    st.plotly_chart(fig, use_container_width=True, config=get_plotly_config())

    # --- Correlation plot: cumulative trials vs d', split by 1B / 2B ---
    _plot_cumulative_trials_dprime_correlation(project_data, t=t)


def _plot_cumulative_trials_dprime_correlation(project_data, t: int = 15):
    """
    Assess whether task experience predicts discriminability.

    For each mouse, daily d' (sensitivity index, computed as the mean d' over
    a sliding window of *t* trials within each session) is paired with the
    cumulative number of trials the animal has completed up to and including
    that session.

    A one-sided Pearson correlation test (H₁: r > 0) is used to evaluate
    whether cumulative trial exposure is positively associated with d'.
    A least-squares linear fit is overlaid on the scatter plot, and the
    Pearson r and one-sided p-value are annotated on the figure.
    """
    from Analysis.GNG_bpod_analysis.metric import d_prime_multiple_sessions
    from scipy.stats import pearsonr

    mice = sorted(project_data["MouseName"].unique())

    cum_all, dp_all, names_all = [], [], []

    for mouse in mice:
        mouse_data = project_data[project_data["MouseName"] == mouse].sort_values("SessionDate")

        n_trials_per_day = []
        for i in range(len(mouse_data)):
            stimuli, _ = preprocess_stimuli_outcomes(mouse_data, i)
            n_trials_per_day.append(len(stimuli))
        cumulative_trials = np.cumsum(n_trials_per_day)

        data = d_prime_multiple_sessions(project_data.copy(), t=t, animal_name=mouse, plot=False)
        d_primes = data["d_prime"].values

        n = min(len(cumulative_trials), len(d_primes))
        for j in range(n):
            if np.isnan(d_primes[j]):
                continue
            cum_all.append(cumulative_trials[j])
            dp_all.append(d_primes[j])
            names_all.append(mouse)

    cum_all = np.array(cum_all, dtype=float)
    dp_all = np.array(dp_all, dtype=float)

    if len(cum_all) < 3:
        return

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=cum_all, y=dp_all,
        mode="markers",
        marker=dict(color=COLOR_D_PRIME, size=6, opacity=0.6),
        text=names_all,
        hovertemplate="Mouse: %{text}<br>Cum. Trials: %{x}<br>d': %{y:.2f}<extra></extra>",
        name="Data",
        showlegend=True,
    ))

    # Linear fit
    slope, intercept = np.polyfit(cum_all, dp_all, 1)
    x_fit = np.linspace(cum_all.min(), cum_all.max(), 100)
    y_fit = slope * x_fit + intercept
    # One-sided test: H1 = positive correlation (d' improves with more trials)
    r, p = pearsonr(cum_all, dp_all, alternative='greater')

    fig.add_trace(go.Scatter(
        x=x_fit, y=y_fit,
        mode="lines",
        line=dict(color=COLOR_D_PRIME, width=2, dash="dash"),
        name=f"Fit (r={r:.2f}, p={p:.3f})",
        showlegend=True,
    ))

    p_str = f"p = {p:.1e}" if p < 0.001 else f"p = {p:.3f}"
    fig.add_annotation(
        text=f"r = {r:.2f}, {p_str} (one-sided)",
        xref="paper", yref="paper",
        x=0.98, y=0.98,
        showarrow=False,
        font=dict(size=13),
        xanchor="right", yanchor="top",
    )

    fig.update_layout(
        title="Correlation: Cumulative Trials vs d'",
        xaxis_title="Cumulative Number of Trials",
        yaxis_title="d'",
        plot_bgcolor="white",
    )

    colors.apply_standard_font_sizes(fig)
    st.plotly_chart(fig, use_container_width=True, config=get_plotly_config())


def _trim_on_decrease(sequence: np.ndarray) -> np.ndarray:
    if not isinstance(sequence, np.ndarray) or sequence.size == 0:
        return sequence if isinstance(sequence, np.ndarray) else np.array([])
    diffs = np.diff(sequence.astype(float))
    dec_indices = np.where(diffs < 0)[0]
    if dec_indices[0:1].size == 0:
        return sequence
    cut = int(dec_indices[0])
    return sequence[cut+1:]


    
    