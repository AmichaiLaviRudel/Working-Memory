import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import Analysis.GNG_bpod_analysis.colors as colors

from Analysis.GNG_bpod_analysis.licking_and_outcome import preprocess_stimuli_outcomes, compute_lick_rate
from Analysis.GNG_bpod_analysis.GNG_bpod_general import get_sessions_for_animal, getNameAndSession

def plot_psychometric_curves_with_boundaries(project_data, N_Boundaries, n_indices = 2):
    """
    Plots psychometric curves for individual trials in grayscale and an average curve in blue.

    Parameters:
    - project_data: DataFrame containing preprocessed data.
    - stimulus_range: Tuple (min_value, max_value) defining the stimulus range to include.
    """



    # Initialize a single figure
    fig = go.Figure()

    # Define a grayscale color palette
    gray_shades = [colors.COLOR_GRAY]

    # Filter rows where N_Boundaries matches
    filtered_df_reset = project_data[project_data["N_Boundaries"] == N_Boundaries].reset_index(drop = True)

    # Parse stringified stimulus arrays into numeric NumPy arrays
    def parse_stimuli(stim_str):
        try:
            return np.fromstring(stim_str.strip("[]"), sep = " ")
        except Exception:
            return np.array([])

    filtered_df_reset["Parsed_Stimuli"] = filtered_df_reset["Unique_Stimuli_Values"].apply(parse_stimuli)

    # Further filter if N_Boundaries == 1 (exclude rows with any stimulus > 1.5)
    if N_Boundaries == 1:
        filtered_df_reset = filtered_df_reset[filtered_df_reset["Parsed_Stimuli"].apply(lambda x: np.all(x <= 1.5))].reset_index(drop = True)


    # Get the sessions for each mouse
    mouse_sessions = {}
    for index, row in filtered_df_reset.iterrows():
        name, session = getNameAndSession(filtered_df_reset, index)
        if name not in mouse_sessions:
            mouse_sessions[name] = []
        mouse_sessions[name].append((session, index))


    # Keep only the last n sessions for each mouse

    last_n_indices = []
    for name, sessions in mouse_sessions.items():
        # Sort sessions by session number and take last n_indices
        sorted_sessions = sorted(sessions, key = lambda x: x[0])[-n_indices:]
        last_n_indices.extend([idx for _, idx in sorted_sessions])

    # Filter the dataframe to keep only the last n sessions per mouse
    filtered_df_reset = filtered_df_reset.iloc[last_n_indices].reset_index()

    # Store data for computing the average line
    all_lick_rates = []
    all_stimuli = []

    #
    # Loop over the filtered dataframe
    for i, (index, row) in enumerate(filtered_df_reset.iterrows()):
        try:
            name, session = getNameAndSession(filtered_df_reset, index)
            stimuli, outcomes = preprocess_stimuli_outcomes(filtered_df_reset, index)
            unique_stimuli, lick_rates = compute_lick_rate(stimuli, outcomes)


            # Store data for averaging
            all_stimuli.append(unique_stimuli)
            all_lick_rates.append(lick_rates)

            # Line plot with markers (grayscale)
            fig.add_trace(go.Scatter(
                x = unique_stimuli, y = lick_rates,
                mode = 'lines',
                line = dict(width = colors.LINE_WIDTH_MEDIUM, color = gray_shades[i % len(gray_shades)]),  # Cycle through grayscale colors
                marker = dict(size = 6, color = gray_shades[i % len(gray_shades)]),
                name = f"{name}, #{session}",
                hovertemplate = "Stimulus: %{x:.2f} kHz<br>Lick Rate: %{y:.2f}%<extra></extra>"
            ))

        except Exception as e:
            print(f"Error processing index {index}: {e}")

    # Compute the average lick rate for each unique stimulus
    if all_stimuli and all_lick_rates:
        # Find common stimulus values (union of all)
        common_stimuli = sorted(set(np.concatenate(all_stimuli)))

        # Interpolate individual lick rates onto the common stimulus values
        interpolated_lick_rates = np.array([
            np.interp(common_stimuli, unique_stimuli, lick_rates)
            for unique_stimuli, lick_rates in zip(all_stimuli, all_lick_rates)
        ])

        # Compute the average lick rate across all trials
        avg_lick_rate = np.mean(interpolated_lick_rates, axis = 0)

        # Add the average line
        fig.add_trace(go.Scatter(
            x = common_stimuli, y = avg_lick_rate,
            mode = 'lines',
            line = dict(width = colors.LINE_WIDTH_THICK, color = colors.COLOR_LOW_BD if N_Boundaries == 1 else colors.COLOR_HIGH_BD),
            name = "Average Response",
            hovertemplate = "Stimulus: %{x:.2f} kHz<br>Avg Lick Rate: %{y:.2f}%<extra></extra>"
        ))

    if N_Boundaries == 2:
        # Add vertical boundary lines
        for x_val, name in zip([1, 1.5], ["Low Boundary", "High Boundary"]):
            if x_val > 0:
                fig.add_trace(go.Scatter(
                    x = [x_val, x_val], y = [0, 100],
                    mode = "lines", line = dict(dash = "dash", width = colors.LINE_WIDTH_MEDIUM, color = 'gray'),
                    name = name,
                    hoverinfo = "skip"
                ))

        # Layout settings
        fig.update_layout(
            title = "Psychometric Curve, Two Boundaries",
            xaxis = dict(
                title = "Stimulus Value [kHz] <br> (log scale)", type = "log",
                tickmode = "array", tickvals = [1, 1.5] + sorted(np.round(common_stimuli, 2).tolist()),
                showgrid = True
            ),
            yaxis = dict(title = "Lick Rate (%)", range = [-5, 110]),
            legend = dict(x = 1.01, y = 0.99, bgcolor = "rgba(255,255,255,0.4)"),
            margin = dict(l = 40, r = 40, t = 60, b = 40),
            hovermode = "x unified"
        )

    elif N_Boundaries == 1:
        # Add vertical boundary line at 1.5 kHz
        fig.add_trace(go.Scatter(
            x = [1, 1], y = [0, 100],
            mode = "lines", line = dict(dash = "dash", width = colors.LINE_WIDTH_MEDIUM, color = 'gray'),
            name = "Low Boundary",
            hoverinfo = "skip"
        ))

        # Layout settings
        fig.update_layout(
            title = "Psychometric Curve, One Boundary",
            xaxis = dict(
                title = "Stimulus Value [kHz] <br> (log scale)", type = "log",
                tickmode = "array", tickvals = [1.5] + sorted(np.round(common_stimuli, 2).tolist()),
                showgrid = True
            ),
            yaxis = dict(title = "Lick Rate (%)", range = [-5, 110]),
            legend = dict(x = 1.01, y = 0.99, bgcolor = "rgba(255,255,255,0.4)"),
            margin = dict(l = 40, r = 40, t = 60, b = 40),
            hovermode = "x unified"
        )
    # Display final figure
    st.plotly_chart(fig, use_container_width = False)


def plot_psychometric_curve(unique_stimuli, lick_rates, x_fit, y_fit, x0, slope_at_midpoint):
    """
    Creates an interactive Plotly graph of the psychometric curve with:
    - A log-scaled x-axis.
    - Scatter points for actual data.
    - A fitted sigmoid curve.
    - Vertical dashed lines for x0, x=1, and x=1.5.
    - Interactive legend (toggle data series).
    """
    try:
        # Ensure unique_stimuli and x_fit are valid (positive for log scale)
        if np.any(unique_stimuli <= 0):
            st.error("Error: unique_stimuli contains non-positive values. Log scale requires all values > 0.")
            return

        if np.any(x_fit <= 0):
            st.error("Error: x_fit contains non-positive values. Log scale requires all values > 0.")
            return

        # st.table(pd.DataFrame({
        #     "x": unique_stimuli,
        #     "y": lick_rates,
        # }))

        max_dp, min_dp = lick_rates.max(), lick_rates.min()

        # Define figure
        fig = go.Figure()

        # Scatter plot for actual data
        fig.add_trace(go.Scatter(
            x = unique_stimuli, y = lick_rates,
            mode = 'markers', marker = dict(size = 8, color = '#1E90FA'),
            name = "Data Points",
            hovertemplate = "Stimulus: %{x:.2f} kHz<br>Lick Rate: %{y:.2f}%<extra></extra>"
        ))
        # ((data_points - min_dp) / (max_dp - min_dp)) * 100
        # Fitted sigmoid curve
        fig.add_trace(go.Scatter(
            x = x_fit, y = ((y_fit+min_dp)/(max_dp+min_dp)*100),
            mode = 'lines', line = dict(width = colors.LINE_WIDTH_THICK, color = '#9699A7'),
            name = "Fitted Curve",
            hovertemplate = "Stimulus: %{x:.2f} kHz<br>Fitted Lick Rate: %{y:.2f}%<extra></extra>"
        ))

        # Vertical lines at x=1, x=1.5, and x0
        for x_val, name in zip([1, 1.5], ["Boundary Low", "Boundary High"]):
            if x_val > 0:  # Avoid issues with log scale
                fig.add_trace(go.Scatter(
                    x = [x_val, x_val], y = [0, 100],
                    mode = "lines", line = dict(dash = "dash", width = colors.LINE_WIDTH_THICK, color = 'gray'),
                    name = name,
                    hoverinfo = "skip"
                ))

        # Layout settings
        fig.update_layout(
            title = "Psychometric Curve",
            xaxis = dict(
                title = "Stimulus Value [kHz] <br> (log scale)", type = "log",
                tickmode = "array", tickvals = [1, 1.5] + sorted(np.round(unique_stimuli,2).tolist()),
                showgrid = True
            ),
            yaxis = dict(title = "Lick Rate (%)", range = [-5, 110]),
            legend = dict(x = 1.01, y = 0.99, bgcolor = "rgba(255,255,255,0.4)"),
            margin = dict(l = 40, r = 40, t = 60, b = 40),
            hovermode = "x unified"
        )

        # Display in Streamlit
        st.plotly_chart(fig, use_container_width = False)

    except Exception as e:
        st.error(f"Unexpected error in plot_psychometric_curve: {e}")
