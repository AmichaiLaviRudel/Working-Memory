import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import Analysis.GNG_bpod_analysis.colors as colors

import numpy as np

from Analysis.GNG_bpod_analysis.licking_and_outcome import preprocess_stimuli_outcomes, compute_lick_rate
from Analysis.GNG_bpod_analysis.GNG_bpod_general import get_sessions_for_animal, getNameAndSession
from Analysis.GNG_bpod_analysis.GNG_bpod_general import get_plotly_config
from statsmodels.stats.multitest import multipletests
from scipy.stats import ttest_rel, ttest_ind


def remove_outlier_sessions(project_data, d_prime_threshold=1.0, t=10, show_details=False):
    """
    Remove outlier sessions based on d-prime threshold.
    
    Parameters:
    - project_data: DataFrame containing session data
    - d_prime_threshold: Minimum mean d-prime value to keep a session (default: 1.0)
    - t: Time bin parameter for d-prime calculation (default: 10)
    - show_details: If True, display a table with d-prime values for all sessions (default: False)
    
    Returns:
    - filtered_data: DataFrame with only sessions meeting the d-prime threshold
    - removed_sessions: DataFrame with sessions that were removed
    
    Example usage:
        # Filter sessions with d-prime >= 1.0
        filtered_data, removed = remove_outlier_sessions(project_data, d_prime_threshold=1.0)
        
        # Use filtered data in plotting
        plot_psychometric_curves_with_boundaries(filtered_data, N_Boundaries=2)
    """
    from Analysis.GNG_bpod_analysis.metric import d_prime
    
    valid_indices = []
    removed_indices = []
    session_info = []
    
    for idx in project_data.index:
        try:
            # Calculate d-prime for this session
            d = d_prime(project_data, index=idx, t=t, plot=False)
            
            # Calculate mean d-prime (ignoring NaN values)
            if len(d) == 0 or np.all(np.isnan(d)):
                mean_d_prime = 0.0
            else:
                mean_d_prime = np.nanmean(d)
            
            # Get session metadata for reporting
            mouse_name = project_data.loc[idx, 'MouseName'] if 'MouseName' in project_data.columns else 'Unknown'
            session_date = project_data.loc[idx, 'SessionDate'] if 'SessionDate' in project_data.columns else 'Unknown'
            
            session_info.append({
                'Index': idx,
                'MouseName': mouse_name,
                'SessionDate': session_date,
                "d_prime": mean_d_prime,
                'Kept': mean_d_prime >= d_prime_threshold
            })
            
            # Keep session if d-prime meets threshold
            if mean_d_prime >= d_prime_threshold:
                valid_indices.append(idx)
            else:
                removed_indices.append(idx)
                
        except Exception as e:
            print(f"Error calculating d-prime for session {idx}: {e}")
            removed_indices.append(idx)
            session_info.append({
                'Index': idx,
                'MouseName': 'Error',
                'SessionDate': 'Error',
                "d_prime": np.nan,
                'Kept': False
            })
    
    filtered_data = project_data.loc[valid_indices].reset_index(drop=True)
    removed_sessions = project_data.loc[removed_indices].reset_index(drop=True)
    
    # Log the filtering results
    st.info(f"Sessions kept: {len(valid_indices)} | Sessions removed: {len(removed_indices)} (d' < {d_prime_threshold})")
    
    # Optionally show detailed table
    if show_details:
        st.markdown("### Session Filtering Details")
        details_df = pd.DataFrame(session_info)
        
        # Style the dataframe
        def highlight_kept(row):
            if row['Kept']:
                return ['background-color: #e6ffe6'] * len(row)
            else:
                return ['background-color: #ffe6e6'] * len(row)
        
        styled_df = details_df.style.apply(highlight_kept, axis=1).format({"d_prime": "{:.3f}"})
        st.dataframe(styled_df, use_container_width=True)
    
    return filtered_data, removed_sessions


def parse_stimuli(stim_str):
    """Parse stringified stimulus arrays into numeric NumPy arrays."""
    try:
        return np.fromstring(stim_str.strip("[]"), sep=" ")
    except Exception:
        return np.array([])


def filter_and_prepare_data(project_data, n_bd, n_indices):
    """Filter data by N_Boundaries and keep only last n sessions per mouse."""
    filtered_df = project_data[project_data["N_Boundaries"] == n_bd].reset_index(drop=True)
    
    # Handle both possible column names for stimuli data
    if "Unique_Stimuli_Values" in filtered_df.columns:
        filtered_df["Parsed_Stimuli"] = filtered_df["Unique_Stimuli_Values"].apply(parse_stimuli)
    elif "Stimuli" in filtered_df.columns:
        filtered_df["Parsed_Stimuli"] = filtered_df["Stimuli"].apply(parse_stimuli)
    else:
        return None
    
    # Further filter if N_Boundaries == 1
    if n_bd == 1:
        filtered_df = filtered_df[filtered_df["Parsed_Stimuli"].apply(
            lambda x: np.all(x <= st.session_state.high_boundary))].reset_index(drop=True)
    
    # Get sessions for each mouse and keep only last n_indices
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
    
    return filtered_df.iloc[last_n_indices].reset_index()


def compute_avg_lick_rates(filtered_df):
    """Process filtered dataframe and compute lick rates."""
    all_lick_rates = []
    all_stimuli = []
    individual_traces = []
    
    for i, (index, row) in enumerate(filtered_df.iterrows()):
        try:
            name, session = getNameAndSession(filtered_df, index)
            stimuli, outcomes = preprocess_stimuli_outcomes(filtered_df, index)
            unique_stimuli, lick_rates = compute_lick_rate(stimuli, outcomes)
            
            all_stimuli.append(unique_stimuli)
            all_lick_rates.append(lick_rates)
            individual_traces.append((unique_stimuli, lick_rates, name, session))
        except Exception as e:
            print(f"Error processing index {index}: {e}")
    
    if not all_stimuli or not all_lick_rates:
        return None, None, []
    
    # Find common stimulus values and interpolate
    common_stimuli = sorted(set(np.concatenate(all_stimuli)))
    interpolated_lick_rates = np.array([
        np.interp(common_stimuli, unique_stimuli, lick_rates)
        for unique_stimuli, lick_rates in zip(all_stimuli, all_lick_rates)
    ])
    
    return np.array(common_stimuli), interpolated_lick_rates, individual_traces


def normalize_lick_rate(avg_lick_rate, label=""):
    """Normalize lick rate to 0-100 range."""
    min_val = np.nanmin(avg_lick_rate)
    max_val = np.nanmax(avg_lick_rate)
    if max_val > min_val:
        normalized = 100 * (avg_lick_rate - min_val) / (max_val - min_val)
        msg = f"Average response normalized for {label}" if label else "Average response normalized"
        st.info(msg)
        return normalized
    else:
        msg = f"Cannot normalize {label}: all average values are equal." if label else "Cannot normalize: all average values are equal."
        st.warning(msg)
        return avg_lick_rate


def add_boundary_lines(fig, n_bd, common_stimuli, label=None):
    """Add vertical boundary lines to figure."""
    if n_bd == 2:
        for x_val, bname in zip([1, 1.5], ["Low Boundary", "High Boundary"]):
            if x_val > 0:
                name = f"{bname} (Two Boundaries)" if label else bname
                fig.add_trace(go.Scatter(
                    x=[x_val, x_val], y=[0, 100],
                    mode="lines", line=dict(dash="dash", width=colors.LINE_WIDTH_MEDIUM, color='gray'),
                    name=name,
                    hoverinfo="skip",
                    legendgroup=label
                ))
    elif n_bd == 1:
        name = "Low Boundary (One Boundary)" if label else "Low Boundary"
        fig.add_trace(go.Scatter(
            x=[1, 1], y=[0, 100],
            mode="lines", line=dict(dash="dash", width=colors.LINE_WIDTH_MEDIUM, color='gray'),
            name=name,
            hoverinfo="skip",
            legendgroup=label
        ))


def plot_psychometric_curves_with_boundaries(project_data, N_Boundaries, n_indices=2, 
                                            filter_outliers=False, d_prime_threshold=1.0, t=10):
    """
    Plots psychometric curves for individual trials in grayscale and an average curve in blue.

    Parameters:
    - project_data: DataFrame containing preprocessed data.
    - N_Boundaries: Number of boundaries (0, 1, or 2)
    - n_indices: Number of last sessions to include per mouse
    - filter_outliers: Whether to filter out sessions with low d-prime (default: False)
    - d_prime_threshold: Minimum d-prime threshold for keeping sessions (default: 1.0)
    - t: Time bin parameter for d-prime calculation (default: 10)
    """
    # Apply outlier filtering if requested
    if filter_outliers:
        project_data, removed_sessions = remove_outlier_sessions(
            project_data, 
            d_prime_threshold=d_prime_threshold, 
            t=t,
            show_details=False
        )
    
    fig = go.Figure()
    normalize_avg = st.checkbox("Normalize average response", value=False, 
                                key=f"normalize_avg_{N_Boundaries}_{n_indices}")
    
    # Handle comparison case (N_Boundaries == 0)
    if N_Boundaries == 0:
        avg_responses = {}
        common_stimuli_dict = {}
        
        for n_bd, color, label in zip([1, 2], [colors.COLOR_LOW_BD, colors.COLOR_HIGH_BD], 
                                      ["One Boundary", "Two Boundaries"]):
            filtered_df = filter_and_prepare_data(project_data, n_bd, n_indices)
            if filtered_df is None:
                continue
            
            common_stimuli, interpolated_lick_rates, individual_traces = compute_avg_lick_rates(filtered_df)
            if common_stimuli is None:
                continue
            
            # Add individual traces
            for unique_stimuli, lick_rates, name, session in individual_traces:
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
            
            # Compute and add average trace
            avg_lick_rate = np.mean(interpolated_lick_rates, axis=0)
            if normalize_avg:
                avg_lick_rate = normalize_lick_rate(avg_lick_rate, label)
            
            fig.add_trace(go.Scatter(
                x=common_stimuli, y=avg_lick_rate,
                mode='lines',
                line=dict(width=colors.LINE_WIDTH_THICK, color=color, dash='solid'),
                name=f"Average Response ({label})",
                hovertemplate="Stimulus: %{x:.2f} kHz<br>Avg Lick Rate: %{y:.2f}%<extra></extra>",
                legendgroup=label
            ))
            
            avg_responses[n_bd] = interpolated_lick_rates
            common_stimuli_dict[n_bd] = common_stimuli
            add_boundary_lines(fig, n_bd, common_stimuli, label)
            
        # Statistical comparison
        if 1 in avg_responses and 2 in avg_responses:
            pvals, corrected_pvals, points_of_interest, shared_x = perform_statistical_comparison(
                avg_responses, common_stimuli_dict, fig)
            
            fig.update_layout(
                title="Psychometric Curves: One vs Two Boundaries",
                xaxis=dict(title="Stimulus Value [kHz] <br> (log scale)", type="log", showgrid=True),
                yaxis=dict(title="Lick Rate (%)", range=[-5, 110]),
                legend=dict(x=1.01, y=0.99, bgcolor="rgba(255,255,255,0.4)"),
                margin=dict(l=40, r=40, t=60, b=40),
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=False, config=get_plotly_config('psychometric_curves_comparison'))
            
            if pvals is not None and points_of_interest:
                display_statistical_table(pvals, corrected_pvals, points_of_interest, shared_x)
        return
    
    # Handle single boundary case (N_Boundaries == 1 or 2)
    filtered_df = filter_and_prepare_data(project_data, N_Boundaries, n_indices)
    if filtered_df is None:
        st.error("Neither 'Unique_Stimuli_Values' nor 'Stimuli' column found in data")
        return
    
    common_stimuli, interpolated_lick_rates, individual_traces = compute_avg_lick_rates(filtered_df)
    
    # Default stimulus values
    if common_stimuli is None:
        common_stimuli = np.array([st.session_state.low_boundary, st.session_state.high_boundary])
    
    # Add individual traces
    for unique_stimuli, lick_rates, name, session in individual_traces:
        fig.add_trace(go.Scatter(
            x=unique_stimuli, y=lick_rates,
            mode='lines',
            line=dict(width=colors.LINE_WIDTH_MEDIUM, color=colors.COLOR_GRAY),
            marker=dict(size=6, color=colors.COLOR_GRAY),
            name=f"{name}, #{session}",
            hovertemplate="Stimulus: %{x:.2f} kHz<br>Lick Rate: %{y:.2f}%<extra></extra>"
        ))
    
    # Compute and add average trace
    if interpolated_lick_rates is not None:
        avg_lick_rate = np.mean(interpolated_lick_rates, axis=0)
        if normalize_avg:
            avg_lick_rate = normalize_lick_rate(avg_lick_rate)
        
        color = colors.COLOR_LOW_BD if N_Boundaries == 1 else colors.COLOR_HIGH_BD
        hovertemplate = ("Stimulus: %{x:.2f} kHz<br>Avg Normalized Lick Rate: %{y:.2f}%<extra></extra>" 
                        if normalize_avg else 
                        "Stimulus: %{x:.2f} kHz<br>Avg Lick Rate: %{y:.2f}%<extra></extra>")
        
        fig.add_trace(go.Scatter(
            x=common_stimuli, y=avg_lick_rate,
            mode='lines',
            line=dict(width=colors.LINE_WIDTH_THICK, color=color),
            name="Average Response",
            hovertemplate=hovertemplate
        ))
    
    # Add boundaries and layout
    add_boundary_lines(fig, N_Boundaries, common_stimuli)
    
    title = "Psychometric Curve, Two Boundaries" if N_Boundaries == 2 else "Psychometric Curve, One Boundary"
    default_ticks = [1, 1.5] if N_Boundaries == 2 else [1.5]
    tickvals = default_ticks + sorted(np.round(common_stimuli, 2).tolist())
    
    fig.update_layout(
        title=title,
        xaxis=dict(
            title="Stimulus Value [kHz] <br> (log scale)", 
            type="log",
            tickmode="array", 
            tickvals=tickvals,
            showgrid=True
        ),
        yaxis=dict(title="Lick Rate (%)", range=[-5, 110]),
        legend=dict(x=1.01, y=0.99, bgcolor="rgba(255,255,255,0.4)"),
        margin=dict(l=40, r=40, t=60, b=40),
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=False, config=get_plotly_config(f'psychometric_curve_{N_Boundaries}_boundaries'))


def perform_statistical_comparison(avg_responses, common_stimuli_dict, fig):
    """Perform statistical comparison between two boundary conditions."""
    x1 = common_stimuli_dict[1]
    x2 = common_stimuli_dict[2]
    shared_x = np.intersect1d(x1, x2)
    low_bd = st.session_state.low_boundary
    high_bd = st.session_state.high_boundary
    points_of_interest = [i for i, x in enumerate(shared_x) if low_bd <= x <= high_bd]
    
    if not points_of_interest:
        return None, None, [], None
    
    idx1 = [np.where(x1 == shared_x[i])[0][0] for i in points_of_interest]
    idx2 = [np.where(x2 == shared_x[i])[0][0] for i in points_of_interest]
    arr1 = avg_responses[1][:, idx1]
    arr2 = avg_responses[2][:, idx2]
    
    # Robust per-point testing
    pvals = []
    for i in range(len(points_of_interest)):
        x_vals = arr2[:, i]
        y_vals = arr1[:, i]
        x_finite = x_vals[np.isfinite(x_vals)]
        y_finite = y_vals[np.isfinite(y_vals)]
        
        if len(x_finite) == 0 or len(y_finite) == 0:
            pvals.append(np.nan)
            continue
        
        # Use paired test if lengths match, else unpaired
        if len(x_finite) == len(y_finite):
            stat, p = ttest_rel(x_finite, y_finite, nan_policy='omit')
        else:
            stat, p = ttest_ind(x_finite, y_finite, nan_policy='omit')
        pvals.append(p)
    
    # Multiple comparison correction
    pvals = np.array(pvals)
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
        x_val = np.log10(shared_x[pi])
        pval = corrected_pvals[i]
        
        if np.isnan(pval) or pval >= 0.05:
            annotation = f"({pval:.2g})"
        elif pval < 0.001:
            annotation = "***"
        elif pval < 0.01:
            annotation = "**"
        else:
            annotation = "*"
        
        fig.add_annotation(
            x=x_val,
            y=110,
            text=annotation,
            showarrow=False,
            font=dict(size=18, color="black", family="Baskerville"),
            xref="x",
            yref="y"
        )
    
    return pvals, corrected_pvals, points_of_interest, shared_x


def display_statistical_table(pvals, corrected_pvals, points_of_interest, shared_x):
    """Display statistical comparison table."""
    pval_table = pd.DataFrame({
        "Point of Interest": [shared_x[pi] for pi in points_of_interest],
        "p-value": pvals,
        "Corrected p-value": corrected_pvals
    })
    
    st.markdown("### Statistical Comparison at Points of Interest")
    
    def highlight_nulls(val):
        return "background-color: lightgray" if pd.isnull(val) else ""
    
    styled = (
        pval_table.style
            .applymap(highlight_nulls)
            .applymap(lambda v: "background-color: #ffe6e6" if isinstance(v, float) and v < 0.05 else "", 
                     subset=["Corrected p-value"])
            .format({"Point of Interest": lambda x: "{:.1f}".format(round(x * 10 * 2) / 2), 
                    "p-value": "{:.3g}", 
                    "Corrected p-value": "{:.3g}"})
    )
    st.write(styled)


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
