from Analysis.GNG_bpod_analysis.psychometric_curves import *
from Analysis.GNG_bpod_analysis.metric import *
from Analysis.GNG_bpod_analysis.GNG_bpod_general import *
from Analysis.GNG_bpod_analysis.colors import COLOR_FA, OUTCOME_COLOR_MAP

import re
import ast
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import altair as alt
from plotly.subplots import make_subplots
import streamlit as st

def responses(selected_data, index=0):
    import numpy as np
    import ast


    # Extract the outcomes list (make sure it's in list format, not a string)
    outcomes = selected_data["Outcomes"].values[index]

    # If outcomes is a string representing a list, use ast.literal_eval to convert it
    outcomes_list = ast.literal_eval(outcomes)

    # Define all unique outcomes in the list
    unique_outcomes = {'Hit', 'CR', 'False Alarm', 'Miss'}


    # Dictionary to store cumulative counts for each outcome
    cumulative_counts = {}

    # Calculate cumulative counts for each unique outcome
    for outcome_type in unique_outcomes:
        # Create a binary array for the current outcome type
        binary_outcome = np.array([1 if outcome == outcome_type else 0 for outcome in outcomes_list])

        # Calculate the cumulative sum for this outcome type
        cumulative_sum = np.cumsum(binary_outcome)

        # Store the cumulative sum in the dictionary
        cumulative_counts[outcome_type] = cumulative_sum

    # Create a DataFrame for responses
    responses = pd.DataFrame({
        "Hit":  cumulative_counts["Hit"],
        "CR":   cumulative_counts["CR"],
        "FA":   cumulative_counts["False Alarm"],  # Corrected the label to match 'False Alarm'
        "Miss": cumulative_counts["Miss"]
    })


    return responses

# Function to calculate the licking rate
def licking_rate(selected_data, index=0, t=10, plot=True):
    data = responses(selected_data, index)

    # Fix column names for consistency
    hit_bin = data["Hit"].diff().rolling(t).sum()
    miss_bin = data["Miss"].diff().rolling(t).sum()
    cr_bin = data["CR"].diff().rolling(t).sum()
    fa_bin = data["FA"].diff().rolling(t).sum()

    rates = pd.DataFrame({
        "Hit":  hit_bin,
        "Miss": miss_bin,
        "CR":   cr_bin,
        "FA":   fa_bin
    }).dropna()

    # Avoid division by zero
    hit_rate = 100 * hit_bin / (hit_bin + miss_bin).replace(0, np.nan)
    fa_rate = 100 * fa_bin / (cr_bin + fa_bin).replace(0, np.nan)

    frac = pd.DataFrame({"Go": hit_rate, "NoGo": fa_rate})

    c_go = "#006837"  # Dark Green
    c_nogo = "#A50026"  # Dark Red

    if plot:
        st.subheader("Licking rate")
        st.line_chart(frac, color=[c_go, c_nogo])

    return rates, frac

### Function: Compute Lick Rate ###
def compute_lick_rate(stimuli, outcomes):
    """
    Computes the lick rate (as a percentage) for each unique stimulus level.
    """
    from Analysis.GNG_bpod_analysis.colors import COLOR_HIT, COLOR_FA

    unique_stimuli = np.unique(stimuli)
    lick_rates = []

    for stimulus in unique_stimuli:
        mask = stimuli == stimulus
        relevant_outcomes = outcomes[mask]
        licks = sum(outcome in {"Hit", "False Alarm"} for outcome in relevant_outcomes)
        lick_rates.append((licks / len(relevant_outcomes) * 100) if len(relevant_outcomes) > 0 else 0)

    return unique_stimuli, np.array(lick_rates)

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
def preprocess_stimuli_outcomes(selected_data, index):
    """
    Extracts and processes stimuli and outcomes from the selected session.
    Converts them from string representations to NumPy arrays.
    """
    stimuli = selected_data["Stimuli"].values[index].strip("[]\n").split()
    stimuli = np.array([float(num) for num in stimuli])

    outcomes = np.array(ast.literal_eval(selected_data["Outcomes"].values[index]))

    return stimuli, outcomes

def process_and_plot_lick_data(project_data, index, plot=True):
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

    # Extract data from DataFrame
    licks_str = project_data.iloc[index]["Licks"]
    trials_str = project_data.iloc[index]["TrialTypes"]
    states_str = (project_data.iloc[index]["States"])

    # Convert trial types from string to list safely
    trials = ast.literal_eval(trials_str) if isinstance(trials_str, str) else trials_str
    trials = np.array(trials)


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

    # Convert licks from string to array safely
    if isinstance(licks_str, str):
        licks_str = re.sub(r'array\(', 'np.array(', licks_str)
        licks = eval(licks_str, {"np": np, "None": None})
    else:
        licks = licks_str



    # Convert licks to NumPy array
    licks = np.array(licks, dtype=object)

    # Identify 'Go' and 'No-Go' trials
    no_go_trial = np.where(trials == 'NoGo')[0]
    go_trial = np.where(trials == 'Go')[0]

    # Extract licks
    go_licks = licks[go_trial]
    no_go_licks = licks[no_go_trial]

    # Filter valid Go and No-Go licks
    filtered_go_licks = filter_valid_arrays(go_licks)
    filtered_no_go_licks = filter_valid_arrays(no_go_licks)

    # Concatenate valid licks
    concatenated_go = np.concatenate(filtered_go_licks) if filtered_go_licks else np.array([])
    concatenated_no_go = np.concatenate(filtered_no_go_licks) if filtered_no_go_licks else np.array([])

    # Generate new trial indices for Go and No-Go
    df_go_raster = prepare_raster_data(filtered_go_licks, "Go", start_index=1)
    df_nogo_raster = prepare_raster_data(filtered_no_go_licks, "No-Go", start_index=len(filtered_go_licks) + 1)

    # Calculate first lick times for each trial
    go_first_lick_times = []
    no_go_first_lick_times = []
    
    # Process Go trials
    if not df_go_raster.empty:
        for trial_idx in df_go_raster["Trial Index"].unique():
            trial_licks = df_go_raster[df_go_raster["Trial Index"] == trial_idx]["Time"].values
            if len(trial_licks) > 0:
                first_lick_time = trial_licks[0]  # First lick in this trial
                go_first_lick_times.append({
                    "Trial Index": trial_idx,
                    "Trial Type": "Go",
                    "First Lick Time (s)": first_lick_time
                })
    
    # Process NoGo trials
    if not df_nogo_raster.empty:
        for trial_idx in df_nogo_raster["Trial Index"].unique():
            trial_licks = df_nogo_raster[df_nogo_raster["Trial Index"] == trial_idx]["Time"].values
            if len(trial_licks) > 0:
                first_lick_time = trial_licks[0]  # First lick in this trial
                no_go_first_lick_times.append({
                    "Trial Index": trial_idx,
                    "Trial Type": "NoGo",
                    "First Lick Time (s)": first_lick_time
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

    # ✅ Add Raster Plot (Scatter)
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
        fig.add_vline(x = 0.0, line = dict(color = "gray", width=2), opacity=0.4 , row=1, col=1)
        fig.add_vline(x = reinforsment_delay_end, line = dict(color = "gray", width = 2), opacity=0.2, row = 1, col = 1)
        fig.add_vline(x = response_window_end, line = dict(color = "gray", width = 2), opacity=0.4, row = 1, col = 1)


    else:
        st.warning("No valid lick raster data to plot.")

    # ✅ Add Histogram Plot (Bar Chart)
    df_go_hist = pd.DataFrame({"Time": concatenated_go, "Trial Type": "Go"})
    df_nogo_hist = pd.DataFrame({"Time": concatenated_no_go, "Trial Type": "No-Go"})
    df_hist = pd.concat([df_go_hist, df_nogo_hist])

    if not df_hist.empty:
        fig.add_trace(
            go.Histogram(
                x=df_go_hist["Time"],
                name="Go",
                marker_color=c_go,
                opacity=0.7,
                showlegend = False
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Histogram(
                x=df_nogo_hist["Time"],
                name="No-Go",
                marker_color=c_nogo,
                opacity=0.7,
                showlegend = False
            ),
            row=2, col=1
        )
        fig.add_vline(x = 0.0, line = dict(color = "gray", width=2), opacity=0.4 , row=2, col=1)
        fig.add_vline(x = reinforsment_delay_end, line = dict(color = "gray", width = 2), opacity=0.2, row = 2, col = 1)
        fig.add_vline(x = response_window_end, line = dict(color = "gray", width = 2), opacity=0.4, row = 2, col = 1)

    else:
        st.warning("No valid lick data to plot.")

    # ✅ Update layout and styling
    fig.update_layout(
        height=500,
        width = 1000,
        showlegend=True,
        xaxis_title="",
        yaxis_title="Trial Index",
        xaxis2_title="Time from Tone (s)",
        yaxis2_title="Lick Count",
        plot_bgcolor="white",
        title_text=f"Hits: {len(filtered_go_licks)}, FAs: {len(filtered_no_go_licks)} / {len(trials)}",

    )

    # Display the subplot figure in Streamlit
    if plot:
        st.plotly_chart(fig, use_container_width=False)

    return df_go_first_licks, df_no_go_first_licks

# Prepare raster plot data
def prepare_raster_data(licks_list, trial_type, start_index=1):

    """Formats raster data for Plotly scatter plot."""
    data = []
    for i, licks_in_trial in enumerate(licks_list):
        if isinstance(licks_in_trial, np.ndarray) and licks_in_trial.size > 0:
            trial_idx = start_index + i  # Assigns sequential index
            for lick in licks_in_trial:
                data.append({"Time": lick, "Trial Index": trial_idx, "Trial Type": trial_type})
    return pd.DataFrame(data)

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

def plot_first_lick_latency(selected_data, index=0, df_go_first_licks=None, df_no_go_first_licks= None, plot=True):
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
    
    # If first lick data is not provided, calculate it
    if df_go_first_licks is None or df_no_go_first_licks is None:
        # Get the data from process_and_plot_lick_data
        df_go_first_licks, df_no_go_first_licks = process_and_plot_lick_data(selected_data, index, plot=False)
    
    # Check if we have valid data
    if df_go_first_licks is None or df_no_go_first_licks is None:
        st.warning("No valid first lick data available for analysis.")
        return None
    
    df_first_licks = pd.concat([df_go_first_licks, df_no_go_first_licks])
    
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
            gridwidth=0.5
        )
    )
    
    # Add statistics
    go_latencies = df_first_licks[df_first_licks["Trial Type"] == "Go"]["First Lick Time (s)"].values
    no_go_latencies = df_first_licks[df_first_licks["Trial Type"] == "NoGo"]["First Lick Time (s)"].values
    
    if len(go_latencies) > 0 and len(no_go_latencies) > 0:
        from scipy.stats import mannwhitneyu
        stat, p_value = mannwhitneyu(go_latencies, no_go_latencies, alternative='two-sided')
        
        st.write(f"**Statistics:**")
        st.write(f"- Go trials: n={len(go_latencies)}, mean={np.mean(go_latencies):.3f}s ± {np.std(go_latencies):.3f}s")
        st.write(f"- NoGo trials: n={len(no_go_latencies)}, mean={np.mean(no_go_latencies):.3f}s ± {np.std(no_go_latencies):.3f}s")
        st.write(f"- Mann-Whitney U test: p={p_value:.3g}")
    
    st.plotly_chart(fig, use_container_width=True)
    
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
            df_go_first_licks, df_no_go_first_licks = process_and_plot_lick_data(selected_data, session_idx, plot=False)
            
            go_latencies = df_go_first_licks["First Lick Time (s)"].values if not df_go_first_licks.empty else []
            go_mean = np.mean(go_latencies) if len(go_latencies) > 0 else np.nan
            go_std = np.std(go_latencies) if len(go_latencies) > 0 else np.nan
            go_count = len(go_latencies)
            
            nogo_latencies = df_no_go_first_licks["First Lick Time (s)"].values if not df_no_go_first_licks.empty else []
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
                gridwidth=0.2
            )
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display summary statistics
        st.write("**Summary Statistics:**")
        summary_df = results_df[['Session Index', 'Session Date', 'Go Mean', 'Go Std', 'Go Count', 'NoGo Mean', 'NoGo Std', 'NoGo Count']].copy()
        st.dataframe(summary_df, use_container_width=True)
    
    return results_df
