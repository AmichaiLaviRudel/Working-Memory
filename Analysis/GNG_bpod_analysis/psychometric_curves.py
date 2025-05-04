from Analysis.GNG_bpod_analysis.licking_and_outcome import *
from Analysis.GNG_bpod_analysis.metric import *

import plotly.graph_objects as go
import numpy as np
import streamlit as st
import pandas as pd
from scipy.optimize import curve_fit



### Function: Fit Psychometric Curve ###
def psychometric_fitting(unique_stims, data_points):
    """
    Fits a sigmoid curve to the psychometric data and extracts key parameters.
    Returns:
    - x0: Inflection point (boundary)
    - slope_at_midpoint: Slope at x0
    - slope_at_boundary: Slope at the x = 1
    - x_fit, y_fit: Fitted curve
    """

    def sigmoid(x, L, x0, k):
        return L / (1 + np.exp(-k * (x - x0)))

    # Handle NaNs before fitting
    valid_mask = np.isfinite(unique_stims) & np.isfinite(data_points)
    unique_stims, data_points = unique_stims[valid_mask], data_points[valid_mask]
    max_dp, min_dp = data_points.max(), data_points.min()
    data_points = ((data_points-min_dp)/(max_dp-min_dp))*100

    if len(unique_stims) < 3:
        raise ValueError("Insufficient valid data for curve fitting.")

    try:
        popt, _ = curve_fit(sigmoid, unique_stims, data_points,
                            p0 = [max(data_points), np.median(unique_stims), 1], maxfev = 10000)
    except RuntimeError:
        raise RuntimeError("Curve fitting failed. Check data for validity.")

    L, x0, k = popt
    slope_at_midpoint = (L * k) / 4
    x_median = 1
    slope_at_boundary = (L * k * np.exp(-k * (x_median - x0))) / ((1 + np.exp(-k * (x_median - x0))) ** 2)

    x_fit = np.linspace(min(unique_stims), max(unique_stims), 100)
    y_fit = sigmoid(x_fit, *popt)

    return x0, slope_at_midpoint, slope_at_boundary, x_fit, y_fit



### Main Function: Run the Full Psychometric Analysis ###
def psychometric_curve(selected_data, index, plot=True):
    """
    Processes psychometric data, fits a sigmoid curve, and plots the psychometric curve.
    """
    try:
        # Extract and preprocess data
        stimuli, outcomes = preprocess_stimuli_outcomes(selected_data, index)
        unique_stimuli, lick_rates = compute_lick_rate(stimuli, outcomes)

        # Fit the psychometric curve
        x0, slope_at_midpoint, slope_at_boundary, x_fit, y_fit = psychometric_fitting(unique_stimuli, lick_rates)

        # Display results
        if plot:
            st.subheader("Psychometric Curve with Fit")
            st.text(f"x0: {round(x0, 2)}, Slope at Boundary: {round(slope_at_boundary, 2)}")
            plot_psychometric_curve(unique_stimuli, lick_rates, x_fit, y_fit, x0, slope_at_boundary)

        return x0, slope_at_midpoint, slope_at_boundary

    except Exception as e:

        return None, None, None

def psychometric_curve_multiple_sessions(selected_data, animal_name = "None", plot=False):
    """
    Plots the progression of the slope at the boundary across multiple sessions for a selected animal.
    """
    if animal_name == "None":
        # Select animal
        animal_name = st.selectbox("Choose an Animal", selected_data["MouseName"].unique(), key="slope_animal_select")

    # Get session indices
    session_indices, session_dates = get_sessions_for_animal(selected_data, animal_name)

    # Initialize array for slopes
    slopes = np.full(len(session_indices), np.nan)
    tones_per_class = []
    boundaries = []
    for idx, i in enumerate(session_indices):
        x0, slope_at_midpoint, slope_at_boundary = psychometric_curve(selected_data, i, plot=False)
        # Retrieve session metadata
        tones_per_class.append(selected_data.loc[i, 'Tones_per_class'])
        boundaries.append(selected_data.loc[i, 'N_Boundaries'])

        if slope_at_boundary is not None:
            slopes[idx] = slope_at_boundary  # Store only slope_at_boundary

    # Create DataFrame for plotting
    data = pd.DataFrame({
        'Session Index': np.arange(1, len(session_indices) + 1),
        'Slope at Boundary': slopes,
        'Legend': ['Slope Progression'] * len(session_indices), # Dummy column for legend
        'tones_per_class': tones_per_class,
        'Boundaries': boundaries
    })

    # Plot using Altair
    if plot:

        chart = alt.Chart(data).mark_line().encode(
            x=alt.X('Session Index:Q', title='Session Index'),
            y=alt.Y('Slope at Boundary:Q', title="Slope at Boundary"),
            color=alt.Color('Legend:N', legend=alt.Legend(title="Legend"))  # Adding legend
        )

        # Horizontal reference line at y=0
        horizontal_line = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(
            color='black', strokeDash=[5, 5]
        ).encode(
            y='y:Q',
        )

        # Annotations: empty circle for Boundaries==1, filled circle for Boundaries==2
        annotations = alt.Chart(data).mark_point(size = 50).encode(
            x = 'Session Index:Q',
            y = 'Slope at Boundary:Q',
            fill = alt.condition(alt.datum.Boundaries == 2, alt.value('black'), alt.value('white')),
            stroke = alt.value('black'),
            tooltip = [
                alt.Tooltip('Session Index', title = 'Session Index'),
                alt.Tooltip('tones_per_class', title = 'tones_per_class'),
                alt.Tooltip('Boundaries', title = 'Boundaries'),
                alt.Tooltip('Slope at Boundary', title = "Slope'")
            ]
        )


        st.markdown(f"**Slope Progress for {animal_name}**")
        # Combine plots
        st.altair_chart(chart + horizontal_line + annotations, use_container_width=True)

    return slopes



def multi_animal_psychometric_slope_progression(selected_data, N_Boundaries=1):

    # Filter the rows where the condition is met
    selected_data = selected_data[selected_data["N_Boundaries"] == N_Boundaries]
    selected_data = selected_data.reset_index()

    # Get unique subject names
    subjects = selected_data["MouseName"].unique()

    # Store slope values for each subject
    slope_data = []
    session_counts = []

    for subject in subjects:
        # Compute psychometric slope for each subject
        slope_values = psychometric_curve_multiple_sessions(selected_data, animal_name = subject, plot = False)

        slope_data.append(slope_values)
        session_counts.append(len(slope_values))

    # Determine max number of sessions for alignment
    max_sessions = max(session_counts)

    # Convert list of arrays to DataFrame (aligned by padding with NaN)
    slope_df = pd.DataFrame(
        [np.pad(s, (0, max_sessions - len(s)), constant_values = np.nan) for s in slope_data])

    # Compute average slope across subjects
    avg_slope = slope_df.mean(axis = 0, skipna = True)

    # Prepare DataFrame for Altair
    data_list = []
    for i, subject in enumerate(subjects):
        for session, slope in enumerate(slope_df.iloc[i]):
            if not np.isnan(slope):
                data_list.append({"Session": session + 1, "Slope": slope, "Mouse": subject})

    # Add average line data
    avg_list = []
    for session, slope in enumerate(avg_slope):
        if not np.isnan(slope):
            avg_list.append({"Session": session + 1, "Slope": slope, "Mouse": "Average"})

    df_altair = pd.DataFrame(data_list)
    df_avg = pd.DataFrame(avg_list)

    # 🎨 Altair line chart for each subject (Gray Palette)
    subject_lines = alt.Chart(df_altair).mark_line(opacity = 0.7).encode(
        x = alt.X("Session:Q", title = "Session"),
        y = alt.Y("Slope:Q", title = "Slope"),
        color = alt.Color("Mouse:N", scale = alt.Scale(scheme = "greys"), legend = alt.Legend(title = "Subject")),
        tooltip = ["Mouse", "Session", "Slope"]
    ).properties(width = 700, height = 400)

    # average line
    avg_line = alt.Chart(df_avg).mark_line( strokeWidth = 3).encode(
        x = "Session:Q",
        y = "Slope:Q"
    )

    # ⚫ Dashed black reference line at y = 0
    ref_line = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color = "black", strokeDash = [5, 5]).encode(
        y = "y:Q"
    )

    # 🏆 Combine all charts
    title_name = "One boundary" if N_Boundaries ==1 else "Two boundaries"
    final_chart = (subject_lines + avg_line + ref_line).properties(title = f"Psychometric Slope Progression, {title_name}")

    # Display chart in Streamlit
    st.altair_chart(final_chart, use_container_width = True)



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
    gray_shades = ["#D3D3D3"]

    # Filter the rows where the condition is met
    filtered_df = project_data[project_data["N_Boundaries"] == N_Boundaries]
    filtered_df_reset = filtered_df.reset_index()

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
                mode = 'lines+markers',
                line = dict(width = 1, color = gray_shades[i % len(gray_shades)]),  # Cycle through grayscale colors
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

        # Add the average line in **blue** (width = 5)
        fig.add_trace(go.Scatter(
            x = common_stimuli, y = avg_lick_rate,
            mode = 'lines',
            line = dict(width = 5, color = '#1E90FA'),
            name = "Average Response",
            hovertemplate = "Stimulus: %{x:.2f} kHz<br>Avg Lick Rate: %{y:.2f}%<extra></extra>"
        ))

    if N_Boundaries == 2:
        # Add vertical boundary lines
        for x_val, name in zip([1, 1.5], ["Low Boundary", "High Boundary"]):
            if x_val > 0:
                fig.add_trace(go.Scatter(
                    x = [x_val, x_val], y = [0, 100],
                    mode = "lines", line = dict(dash = "dash", width = 2, color = 'gray'),
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
            mode = "lines", line = dict(dash = "dash", width = 2, color = 'gray'),
            name = "Low Boundary",
            hoverinfo = "skip"
        ))

        # Layout settings
        fig.update_layout(
            title = "Psychometric Curve, One Boundary (Experts)",
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
    st.write("hey")
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
            mode = 'lines', line = dict(width = 3, color = '#9699A7'),
            name = "Fitted Curve",
            hovertemplate = "Stimulus: %{x:.2f} kHz<br>Fitted Lick Rate: %{y:.2f}%<extra></extra>"
        ))

        # Vertical lines at x=1, x=1.5, and x0
        for x_val, name in zip([1, 1.5], ["Boundary Low", "Boundary High"]):
            if x_val > 0:  # Avoid issues with log scale
                fig.add_trace(go.Scatter(
                    x = [x_val, x_val], y = [0, 100],
                    mode = "lines", line = dict(dash = "dash", width = 2, color = 'gray'),
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


