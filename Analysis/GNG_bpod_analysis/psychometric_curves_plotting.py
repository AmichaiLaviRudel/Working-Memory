import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import Analysis.GNG_bpod_analysis.colors as colors

import numpy as np

from Analysis.GNG_bpod_analysis.licking_and_outcome import preprocess_stimuli_outcomes, compute_lick_rate
from Analysis.GNG_bpod_analysis.GNG_bpod_general import get_sessions_for_animal, getNameAndSession
from statsmodels.stats.multitest import multipletests
from scipy.stats import ttest_rel, ttest_ind


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
        import numpy as np
        try:
            return np.fromstring(stim_str.strip("[]"), sep = " ")
        except Exception:
            return np.array([])

    # Handle both possible column names for stimuli data
    if "Unique_Stimuli_Values" in filtered_df_reset.columns:
        filtered_df_reset["Parsed_Stimuli"] = filtered_df_reset["Unique_Stimuli_Values"].apply(parse_stimuli)
    elif "Stimuli" in filtered_df_reset.columns:
        filtered_df_reset["Parsed_Stimuli"] = filtered_df_reset["Stimuli"].apply(parse_stimuli)
    else:
        st.error("Neither 'Unique_Stimuli_Values' nor 'Stimuli' column found in data")
        return

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
    # Initialize common_stimuli with default values to avoid UnboundLocalError
    common_stimuli = [1.0, 1.5]  # Default stimulus values

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

    # Add normalization checkbox (always available)
    normalize_avg = st.checkbox("Normalize average response", value=False, key=f"normalize_avg_{N_Boundaries}_{n_indices}")

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

        # Normalize if requested
        if normalize_avg:
            min_val = np.nanmin(avg_lick_rate)
            max_val = np.nanmax(avg_lick_rate)
            if max_val > min_val:
                avg_lick_rate = 100 * (avg_lick_rate - min_val) / (max_val - min_val)
                st.info("Average response normalized")
            else:
                st.warning("Cannot normalize: all average values are equal.")

        # Add the average line
        fig.add_trace(go.Scatter(
            x = common_stimuli, y = avg_lick_rate,
            mode = 'lines',
            line = dict(width = colors.LINE_WIDTH_THICK, color = colors.COLOR_LOW_BD if N_Boundaries == 1 else colors.COLOR_HIGH_BD),
            name = "Average Response",
            hovertemplate = "Stimulus: %{x:.2f} kHz<br>Avg Normalized Lick Rate: %{y:.2f}%<extra></extra>" if normalize_avg else "Stimulus: %{x:.2f} kHz<br>Avg Lick Rate: %{y:.2f}%<extra></extra>"
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
                tickmode = "array", tickvals = [1, 1.5] + sorted(np.round(common_stimuli, 2).tolist()) if common_stimuli else [1, 1.5],
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
                tickmode = "array", tickvals = [1.5] + sorted(np.round(common_stimuli, 2).tolist()) if common_stimuli else [1.5],
                showgrid = True
            ),
            yaxis = dict(title = "Lick Rate (%)", range = [-5, 110]),
            legend = dict(x = 1.01, y = 0.99, bgcolor = "rgba(255,255,255,0.4)"),
            margin = dict(l = 40, r = 40, t = 60, b = 40),
            hovermode = "x unified"
        )

    if N_Boundaries == 0:
        # Plot both N_Boundaries == 1 and N_Boundaries == 2 cases in the same figure
        avg_responses = {}
        common_stimuli_dict = {}
        for n_bd, color, label in zip([1, 2], [colors.COLOR_LOW_BD, colors.COLOR_HIGH_BD], ["One Boundary", "Two Boundaries"]):
            filtered_df = project_data[project_data["N_Boundaries"] == n_bd].reset_index(drop=True)
            # Handle both possible column names for stimuli data
            if "Unique_Stimuli_Values" in filtered_df.columns:
                filtered_df["Parsed_Stimuli"] = filtered_df["Unique_Stimuli_Values"].apply(parse_stimuli)
            elif "Stimuli" in filtered_df.columns:
                filtered_df["Parsed_Stimuli"] = filtered_df["Stimuli"].apply(parse_stimuli)
            else:
                continue  # Skip this boundary if no stimuli data found
            if n_bd == 1:
                filtered_df = filtered_df[filtered_df["Parsed_Stimuli"].apply(lambda x: np.all(x <= 1.5))].reset_index(drop=True)
            # Get sessions for each mouse
            mouse_sessions = {}
            for index, row in filtered_df.iterrows():
                name, session = getNameAndSession(filtered_df, index)
                if name not in mouse_sessions:
                    mouse_sessions[name] = []
                mouse_sessions[name].append((session, index))
            last_n_indices = []
            for name, sessions in mouse_sessions.items():
                sorted_sessions = sorted(sessions, key=lambda x: x[0])[-n_indices:]
                last_n_indices.extend([idx for _, idx in sorted_sessions])
            filtered_df = filtered_df.iloc[last_n_indices].reset_index()
            all_lick_rates = []
            all_stimuli = []
            for i, (index, row) in enumerate(filtered_df.iterrows()):
                try:
                    name, session = getNameAndSession(filtered_df, index)
                    stimuli, outcomes = preprocess_stimuli_outcomes(filtered_df, index)
                    unique_stimuli, lick_rates = compute_lick_rate(stimuli, outcomes)
                    all_stimuli.append(unique_stimuli)
                    all_lick_rates.append(lick_rates)
                    fig.add_trace(go.Scatter(
                        x=unique_stimuli, y=lick_rates,
                        mode='lines',
                        line=dict(width=colors.LINE_WIDTH_MEDIUM, color=colors.COLOR_GRAY),
                        marker=dict(size=6, color=colors.COLOR_GRAY),
                        name=f"{name}, #{session} ({label})",
                        hovertemplate="Stimulus: %{x:.2f} kHz<br>Lick Rate: %{y:.2f}%<extra></extra>",
                        legendgroup=label,
                        showlegend=False
                    ))
                except Exception as e:
                    print(f"Error processing index {index}: {e}")
            if all_stimuli and all_lick_rates:
                common_stimuli = sorted(set(np.concatenate(all_stimuli)))
                interpolated_lick_rates = np.array([
                    np.interp(common_stimuli, unique_stimuli, lick_rates)
                    for unique_stimuli, lick_rates in zip(all_stimuli, all_lick_rates)
                ])
                avg_lick_rate = np.mean(interpolated_lick_rates, axis=0)
                # Normalize if requested
                if normalize_avg:
                    min_val = np.nanmin(avg_lick_rate)
                    max_val = np.nanmax(avg_lick_rate)
                    if max_val > min_val:
                        avg_lick_rate = 100 * (avg_lick_rate - min_val) / (max_val - min_val)
                        st.info(f"Average response normalized for {label}: lowest point is 0, highest is 100.")
                    else:
                        st.warning(f"Cannot normalize {label}: all average values are equal.")
                fig.add_trace(go.Scatter(
                    x=common_stimuli, y=avg_lick_rate,
                    mode='lines',
                    line=dict(width=colors.LINE_WIDTH_THICK, color=color, dash='solid'),
                    name=f"Average Response ({label})",
                    hovertemplate="Stimulus: %{x:.2f} kHz<br>Avg Lick Rate: %{y:.2f}%<extra></extra>",
                    legendgroup=label
                ))
                avg_responses[n_bd] = interpolated_lick_rates
                common_stimuli_dict[n_bd] = np.array(common_stimuli)
            # Add boundaries
            if n_bd == 2:
                for x_val, bname in zip([1, 1.5], ["Low Boundary", "High Boundary"]):
                    if x_val > 0:
                        fig.add_trace(go.Scatter(
                            x=[x_val, x_val], y=[0, 100],
                            mode="lines", line=dict(dash="dash", width=colors.LINE_WIDTH_MEDIUM, color='gray'),
                            name=f"{bname} (Two Boundaries)",
                            hoverinfo="skip",
                            legendgroup=label
                        ))
            elif n_bd == 1:
                fig.add_trace(go.Scatter(
                    x=[1, 1], y=[0, 100],
                    mode="lines", line=dict(dash="dash", width=colors.LINE_WIDTH_MEDIUM, color='gray'),
                    name="Low Boundary (One Boundary)",
                    hoverinfo="skip",
                    legendgroup=label
                ))
        # --- Statistical comparison and annotation ---
        pvals = None
        corrected_pvals = None
        points_of_interest = []
        shared_x = None
        
        if 1 in avg_responses and 2 in avg_responses:
            x1 = common_stimuli_dict[1]
            x2 = common_stimuli_dict[2]
            shared_x = np.intersect1d(x1, x2)
            low_bd = st.session_state.low_boundary
            high_bd = st.session_state.high_boundary
            points_of_interest = [i for i, x in enumerate(shared_x) if low_bd <= x <= high_bd]
            if points_of_interest:
                idx1 = [np.where(x1 == shared_x[i])[0][0] for i in points_of_interest]
                idx2 = [np.where(x2 == shared_x[i])[0][0] for i in points_of_interest]
                arr1 = avg_responses[1][:, idx1]
                arr2 = avg_responses[2][:, idx2]
                # Robust per-point masking and testing
                pvals = []
                stats = []
                for i in range(len(points_of_interest)):
                    x_vals = arr2[:, i]
                    y_vals = arr1[:, i]
                    # Only keep finite values
                    x_finite = x_vals[np.isfinite(x_vals)]
                    y_finite = y_vals[np.isfinite(y_vals)]
                    # If both have data, test
                    if len(x_finite) == 0 or len(y_finite) == 0:
                        pvals.append(np.nan)
                        stats.append(np.nan)
                        continue
                    # Use paired test if lengths match, else unpaired
                    if len(x_finite) == len(y_finite):
                        stat, p = ttest_rel(x_finite, y_finite, nan_policy='omit')
                    else:
                        stat, p = ttest_ind(x_finite, y_finite, nan_policy='omit')
                    pvals.append(p)
                    stats.append(stat)
                # Multiple comparison correction
                pvals = np.array(pvals)
                stats = np.array(stats)
                if np.all(np.isnan(pvals)):
                    corrected_pvals = pvals
                else:
                    valid = ~np.isnan(pvals)
                    corrected = np.full_like(pvals, np.nan)
                    if np.any(valid):
                        reject, pvals_corr, _, _ = multipletests(pvals[valid], method='fdr_bh')
                        corrected[valid] = pvals_corr
                    corrected_pvals = corrected
                # Annotate the figure
                for i, pi in enumerate(points_of_interest):
                    # pi is the index in shared_x, which is the array of shared stimulus values
                    x_val = np.log10(shared_x[pi])
                    y_max = max(np.nanmax(arr1[:, i]), np.nanmax(arr2[:, i]))
                    pval = corrected_pvals[i]
                    if np.isnan(pval) or pval >= 0.05:
                        annotation = "n.s."
                    elif pval < 0.001:
                        annotation = "***"
                    elif pval < 0.01:
                        annotation = "**"
                    else:
                        annotation = "*"
                    if annotation:
                        # Place annotation at the correct x value, slightly above the max y at this x
                            fig.add_annotation(
                            x=x_val,
                            y=110,  # 7 units above the max y at this x for visibility
                            text=annotation,
                            showarrow=False,
                            font=dict(size=18, color="black", family="Baskerville"),
                            xref="x",
                            yref="y"
                        )
        fig.update_layout(
            title="Psychometric Curves: One vs Two Boundaries",
            xaxis=dict(title="Stimulus Value [kHz] <br> (log scale)", type="log", showgrid=True),
            yaxis=dict(title="Lick Rate (%)", range=[-5, 110]),
            legend=dict(x=1.01, y=0.99, bgcolor="rgba(255,255,255,0.4)"),
            margin=dict(l=40, r=40, t=60, b=40),
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=False)
        
        # Display a table with points of interest and the corrected p-values only if we have data
        if pvals is not None and points_of_interest and shared_x is not None:
            pval_table = pd.DataFrame({
                "Point of Interest": [shared_x[pi] for pi in points_of_interest],
                "p-value": pvals,
                "Corrected p-value": corrected_pvals
            })
            # Use Streamlit's st.dataframe for a more interactive and visually appealing table
            st.markdown("### Statistical Comparison at Points of Interest")
            def highlight_nulls(val):
                import pandas as pd
                return "background-color: lightgray" if pd.isnull(val) else ""

            styled = (
                pval_table.style
                    .applymap(highlight_nulls)
                    .applymap(lambda v: "background-color: #ffe6e6" if isinstance(v, float) and v < 0.05 else "", subset=["Corrected p-value"])
                    .format({"Point of Interest": lambda x: "{:.1f}".format(round(x * 10 * 2) / 2), "p-value": "{:.3g}", "Corrected p-value": "{:.3g}"})
            )
            st.write(styled)
        return

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
