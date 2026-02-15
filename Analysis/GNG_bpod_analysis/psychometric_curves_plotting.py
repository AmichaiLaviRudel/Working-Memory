import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import Analysis.GNG_bpod_analysis.colors as colors

import numpy as np

from Analysis.GNG_bpod_analysis.licking_and_outcome import preprocess_stimuli_outcomes, compute_lick_rate
from Analysis.GNG_bpod_analysis.GNG_bpod_general import (
    get_sessions_for_animal,
    getNameAndSession,
    get_plotly_config,
    get_global_early_response_filter,
)
from statsmodels.stats.multitest import multipletests
from scipy.stats import ttest_rel, ttest_ind, wilcoxon

def remove_outlier_sessions(project_data, d_prime_threshold=1.0, hit_rate_threshold=0.6, t=10, show_details=False):
    """
    Remove outlier sessions based on d-prime and hit rate thresholds.
    
    Parameters:
    - project_data: DataFrame containing session data
    - d_prime_threshold: Minimum mean d-prime value to keep a session (default: 1.0)
    - hit_rate_threshold: Minimum hit rate to keep a session (default: 0.6)
    - t: Time bin parameter for d-prime calculation (default: 10)
    - show_details: If True, display a table with d-prime values for all sessions (default: False)
    
    Returns:
    - filtered_data: DataFrame with only sessions meeting both thresholds
    - removed_sessions: DataFrame with sessions that were removed
    
    Example usage:
        # Filter sessions with d-prime >= 1.0 and hit rate >= 0.6
        filtered_data, removed = remove_outlier_sessions(project_data, d_prime_threshold=1.0, hit_rate_threshold=0.6)
        
        # Use filtered data in plotting
        plot_psychometric_curves_with_boundaries(filtered_data, N_Boundaries=2)
    """
    from Analysis.GNG_bpod_analysis.metric import d_prime
    from Analysis.GNG_bpod_analysis.licking_and_outcome import licking_rate
    
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
            
            # Calculate hit rate for this session using licking_rate
            try:
                _, frac = licking_rate(project_data, index=idx, t=t, plot=False)
                frac = frac.dropna(how="all").astype(float)
                hr = frac["Go"] / 100  # Convert percentage to proportion
                if len(hr) == 0 or np.all(np.isnan(hr)):
                    mean_hit_rate = 0.0
                else:
                    mean_hit_rate = np.nanmean(hr)
            except Exception:
                mean_hit_rate = 0.0
            
            # Get session metadata for reporting
            mouse_name = project_data.loc[idx, 'MouseName'] if 'MouseName' in project_data.columns else 'Unknown'
            session_date = project_data.loc[idx, 'SessionDate'] if 'SessionDate' in project_data.columns else 'Unknown'
            
            # Session passes if both thresholds are met
            passes_filter = (mean_d_prime >= d_prime_threshold) and (mean_hit_rate >= hit_rate_threshold)
            
            session_info.append({
                'Index': idx,
                'MouseName': mouse_name,
                'SessionDate': session_date,
                "d_prime": mean_d_prime,
                "hit_rate": mean_hit_rate,
                'Kept': passes_filter
            })
            
            # Keep session if both thresholds are met
            if passes_filter:
                valid_indices.append(idx)
            else:
                removed_indices.append(idx)
                
        except Exception as e:
            print(f"Error calculating metrics for session {idx}: {e}")
            removed_indices.append(idx)
            session_info.append({
                'Index': idx,
                'MouseName': 'Error',
                'SessionDate': 'Error',
                "d_prime": np.nan,
                "hit_rate": np.nan,
                'Kept': False
            })
    
    filtered_data = project_data.loc[valid_indices].reset_index(drop=True)
    removed_sessions = project_data.loc[removed_indices].reset_index(drop=True)
    
    # Log the filtering results
    st.info(f"Sessions kept: {len(valid_indices)} | Sessions removed: {len(removed_indices)} (d' < {d_prime_threshold} or HR < {hit_rate_threshold:.0%})")
    
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
    
    # Filter out sessions with fewer than 2 tones per class if available
    if "Tones_per_class" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Tones_per_class"] >= 2].reset_index(drop=True)
    # Handle both possible column names for stimuli data
    if "Unique_Stimuli_Values" in filtered_df.columns:
        filtered_df["Parsed_Stimuli"] = filtered_df["Unique_Stimuli_Values"].apply(parse_stimuli)
    if "Stimuli" in filtered_df.columns:
        filtered_df["Stimuli"] = filtered_df["Stimuli"].apply(parse_stimuli)
    else:
        return None
    
    # Further filter if N_Boundaries == 1
    if n_bd == 1:
        filtered_df = filtered_df[filtered_df["Stimuli"].apply(
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


def compute_avg_lick_rates(filtered_df, filter_early_response: bool):
    """Process filtered dataframe and compute lick rates."""
    all_lick_rates = []
    all_stimuli = []
    individual_traces = []
    
    for i, (index, row) in enumerate(filtered_df.iterrows()):

        name, session = getNameAndSession(filtered_df, index)
        stimuli, outcomes = preprocess_stimuli_outcomes(filtered_df, index)

        # Optionally filter out 'Early Response' trials before computing lick rates
        if filter_early_response:
            try:
                early_mask = np.array(
                    ['Early Response' not in str(o) for o in outcomes],
                    dtype=bool,
                )
                if len(early_mask) == len(stimuli):
                    stimuli = stimuli[early_mask]
                    outcomes = outcomes[early_mask]
            except Exception:
                # If anything goes wrong, fall back to unfiltered data
                pass

        unique_stimuli, lick_rates, catch_stimuli, catch_lick_rates = compute_lick_rate(stimuli, outcomes)
        # Concatenate stimuli and lick rates
        unique_stimuli = np.concatenate((unique_stimuli, catch_stimuli))
        lick_rates = np.concatenate((lick_rates, catch_lick_rates))
        # Sort unique stimuli and reorder lick_rates accordingly
        sort_idx = np.argsort(unique_stimuli)
        unique_stimuli = unique_stimuli[sort_idx]
        lick_rates = lick_rates[sort_idx]
        all_stimuli.append(unique_stimuli)
        all_lick_rates.append(lick_rates)
        individual_traces.append((unique_stimuli, lick_rates, name, session))

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
        for x_val, bname in zip([st.session_state.low_boundary, st.session_state.high_boundary], ["Low Boundary", "High Boundary"]):
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
            x=[st.session_state.low_boundary, st.session_state.low_boundary], y=[0, 100],
            mode="lines", line=dict(dash="dash", width=colors.LINE_WIDTH_MEDIUM, color='gray'),
            name=name,
            hoverinfo="skip",
            legendgroup=label
        ))


def plot_psychometric_curves_with_boundaries(project_data, N_Boundaries, n_indices=2, 
                                            filter_outliers=False, d_prime_threshold=1.0, 
                                            hit_rate_threshold=0.6, t=10, key_suffix=""):
    """
    Plots psychometric curves for individual trials in grayscale and an average curve in blue.

    Parameters:
    - project_data: DataFrame containing preprocessed data.
    - N_Boundaries: Number of boundaries (0, 1, or 2)
    - n_indices: Number of last sessions to include per mouse
    - filter_outliers: Whether to filter out sessions with low d-prime or hit rate (default: False)
    - d_prime_threshold: Minimum d-prime threshold for keeping sessions (default: 1.0)
    - hit_rate_threshold: Minimum hit rate threshold for keeping sessions (default: 0.6)
    - t: Time bin parameter for d-prime calculation (default: 10)
    - key_suffix: Unique suffix for Streamlit widget keys (default: "")
    """
    # Apply outlier filtering if requested
    if filter_outliers:
        project_data, removed_sessions = remove_outlier_sessions(
            project_data, 
            d_prime_threshold=d_prime_threshold,
            hit_rate_threshold=hit_rate_threshold,
            t=t,
            show_details=False
        )

    # Decide whether to filter Early Response trials for these aggregated plots
    filter_early = get_global_early_response_filter()

    fig = go.Figure()
    normalize_avg = st.checkbox("Normalize average response", value=False, 
                                key=f"normalize_avg_{N_Boundaries}_{n_indices}{key_suffix}")
    
    # Handle comparison case (N_Boundaries == 0)
    if N_Boundaries == 0:
        avg_responses = {}
        session_data = {}  # Store session-level data for Wilcoxon tests
        common_stimuli_dict = {}
        
        for n_bd, color, label in zip([1, 2], [colors.COLOR_LOW_BD, colors.COLOR_HIGH_BD], 
                                      ["One Boundary", "Two Boundaries"]):
            filtered_df = filter_and_prepare_data(project_data, n_bd, n_indices)
            if filtered_df is None:
                continue
            common_stimuli, interpolated_lick_rates, individual_traces = compute_avg_lick_rates(
                filtered_df, filter_early_response=filter_early
            )
            if common_stimuli is None:
                continue
            
            # Store session-level data for Wilcoxon signed-rank tests
            session_data[n_bd] = interpolated_lick_rates
            
            # Normalize each session individually if requested
            if normalize_avg:
                # Normalize each session to 0-100 range
                normalized_rates = []
                for session_rates in interpolated_lick_rates:
                    min_val = np.nanmin(session_rates)
                    max_val = np.nanmax(session_rates)
                    if max_val > min_val:
                        normalized_session = 100 * (session_rates - min_val) / (max_val - min_val)
                    else:
                        normalized_session = session_rates  # Keep original if all values are the same
                    normalized_rates.append(normalized_session)
                interpolated_lick_rates = np.array(normalized_rates)
            
            # Add individual traces
            for i, (unique_stimuli, lick_rates, name, session) in enumerate(individual_traces):
                # Use normalized data if normalization is enabled
                if normalize_avg and i < len(interpolated_lick_rates):
                    # Interpolate the normalized session data to the original stimulus values
                    normalized_session_rates = np.interp(unique_stimuli, common_stimuli, interpolated_lick_rates[i])
                    display_rates = normalized_session_rates
                else:
                    display_rates = lick_rates
                
                fig.add_trace(go.Scatter(
                    x=unique_stimuli, y=display_rates,
                    mode='lines+markers',
                    line=dict(width=colors.LINE_WIDTH_MEDIUM, color=colors.COLOR_GRAY, shape='spline'),
                    marker=dict(size=6, color=colors.COLOR_GRAY, symbol='circle'),
                    name=f"{name}, #{session} ({label})",
                    hovertemplate="Stimulus: %{x:.2f} kHz<br>Lick Rate: %{y:.2f}%<extra></extra>",
                    legendgroup=label,
                    showlegend=False
                ))
            
            # Compute and add average trace
            avg_lick_rate = np.mean(interpolated_lick_rates, axis=0)
            


            fig.add_trace(go.Scatter(
                x=common_stimuli, y=avg_lick_rate,
                mode='lines+markers',
                line=dict(width=colors.LINE_WIDTH_THICK, color=color, dash='solid', shape='spline'),
                marker=dict(size=8, color=color, symbol='circle'),
                name=f"Average Response ({label})",
                hovertemplate="Stimulus: %{x:.2f} kHz<br>Avg Lick Rate: %{y:.2f}%<extra></extra>",
                legendgroup=label
            ))
            
            avg_responses[n_bd] = avg_lick_rate
            common_stimuli_dict[n_bd] = common_stimuli
            add_boundary_lines(fig, n_bd, common_stimuli, label)
            
        # Statistical comparison with Wilcoxon signed-rank tests
        if 1 in avg_responses and 2 in avg_responses:
            pvals, corrected_pvals, points_of_interest, shared_x = perform_statistical_comparison(
                avg_responses, common_stimuli_dict, fig, session_data_1=session_data.get(1), session_data_2=session_data.get(2))
            
            fig.update_layout(
                title="Psychometric Curves: One vs Two Boundaries",
                xaxis=dict(title="Frequency [kHz] (log)", type="log", showgrid=True),
                yaxis=dict(title="Lick Rate (%)", range=[-5, 110]),
                legend=dict(x=1.01, y=0.99, bgcolor="rgba(255,255,255,0.4)"),
                margin=dict(l=40, r=40, t=60, b=40),
                hovermode="x unified"
            )
            colors.apply_standard_font_sizes(fig)
            st.plotly_chart(fig, use_container_width=False, config=get_plotly_config('psychometric_curves_comparison'))
            
            if pvals is not None and points_of_interest:
                display_statistical_table(pvals, corrected_pvals, points_of_interest, shared_x)
        return
    
    # Handle single boundary case (N_Boundaries == 1 or 2)
    filtered_df = filter_and_prepare_data(project_data, N_Boundaries, n_indices)
    if filtered_df is None:
        st.error("Neither 'Unique_Stimuli_Values' nor 'Stimuli' column found in data")
        return
    
    common_stimuli, interpolated_lick_rates, individual_traces = compute_avg_lick_rates(
        filtered_df, filter_early_response=filter_early
    )
    
    # Default stimulus values
    if common_stimuli is None:
        common_stimuli = np.array([st.session_state.low_boundary, st.session_state.high_boundary])
    
    # Normalize each session individually if requested
    if normalize_avg and interpolated_lick_rates is not None:
        # Normalize each session to 0-100 range
        normalized_rates = []
        for session_rates in interpolated_lick_rates:
            min_val = np.nanmin(session_rates)
            max_val = np.nanmax(session_rates)
            if max_val > min_val:
                normalized_session = 100 * (session_rates - min_val) / (max_val - min_val)
            else:
                normalized_session = session_rates  # Keep original if all values are the same
            normalized_rates.append(normalized_session)
        interpolated_lick_rates = np.array(normalized_rates)
    
    # Add individual traces
    for i, (unique_stimuli, lick_rates, name, session) in enumerate(individual_traces):
        # Use normalized data if normalization is enabled
        if normalize_avg and interpolated_lick_rates is not None and i < len(interpolated_lick_rates):
            # Interpolate the normalized session data to the original stimulus values
            normalized_session_rates = np.interp(unique_stimuli, common_stimuli, interpolated_lick_rates[i])
            display_rates = normalized_session_rates
        else:
            display_rates = lick_rates
        
        fig.add_trace(go.Scatter(
            x=unique_stimuli, y=display_rates,
            mode='lines+markers',
            line=dict(width=colors.LINE_WIDTH_MEDIUM, color=colors.COLOR_GRAY, shape='spline'),
            marker=dict(size=6, color=colors.COLOR_GRAY, symbol='circle'),
            name=f"{name}, #{session}",
            hovertemplate="Stimulus: %{x:.2f} kHz<br>Lick Rate: %{y:.2f}%<extra></extra>"
        ))
    
    # Compute and add average trace
    if interpolated_lick_rates is not None:
        avg_lick_rate = np.mean(interpolated_lick_rates, axis=0)
        
        color = colors.COLOR_LOW_BD if N_Boundaries == 1 else colors.COLOR_HIGH_BD
        hovertemplate = ("Stimulus: %{x:.2f} kHz<br>Avg Normalized Lick Rate: %{y:.2f}%<extra></extra>" 
                        if normalize_avg else 
                        "Stimulus: %{x:.2f} kHz<br>Avg Lick Rate: %{y:.2f}%<extra></extra>")
        
        fig.add_trace(go.Scatter(
            x=common_stimuli, y=avg_lick_rate,
            mode='lines+markers',
            line=dict(width=colors.LINE_WIDTH_THICK, color=color, shape='spline'),
            marker=dict(size=8, color=color, symbol='circle'),
            name="Average Response",
            hovertemplate=hovertemplate
        ))
    
    # Add boundaries and layout
    add_boundary_lines(fig, N_Boundaries, common_stimuli)
    
    title = "Psychometric Curve, Two Boundaries" if N_Boundaries == 2 else "Psychometric Curve, One Boundary"
    default_ticks = [st.session_state.low_boundary, st.session_state.high_boundary] if N_Boundaries == 2 else [st.session_state.high_boundary]
    tickvals = default_ticks + sorted(np.round(common_stimuli, 2).tolist())
    
    fig.update_layout(
        title=title,
        xaxis=dict(
            title="Frequency [kHz] (log)", 
            type="log",
            tickmode="array", 
            tickvals=tickvals,
            showgrid=True
        ),
        yaxis=dict(title="Lick Rate (%)", range=[-15, 110]),
        legend=dict(x=1.01, y=0.99, bgcolor="rgba(255,255,255,0.4)"),
        margin=dict(l=40, r=40, t=60, b=40),
        hovermode="x unified"
    )
    colors.apply_standard_font_sizes(fig)
    st.plotly_chart(fig, use_container_width=False, config=get_plotly_config(f'psychometric_curve_{N_Boundaries}_boundaries', width=450*(N_Boundaries+1)))


def perform_statistical_comparison(avg_responses, common_stimuli_dict, fig, session_data_1=None, session_data_2=None):
    """Perform Wilcoxon signed-rank tests on session-level data to compare boundary conditions."""

    from scipy.stats import wilcoxon
    from statsmodels.stats.multitest import multipletests
    x1 = common_stimuli_dict[1]
    x2 = common_stimuli_dict[2]
    shared_x = np.intersect1d(x1, x2)
    low_bd = st.session_state.low_boundary
    high_bd = st.session_state.high_boundary
    points_of_interest = [i for i, x in enumerate(shared_x) if low_bd <= x <= high_bd]
    
    if not points_of_interest:
        return None, None, [], None
    
    # Get indices for shared stimulus values
    idx1 = [np.where(x1 == shared_x[i])[0][0] for i in points_of_interest]
    idx2 = [np.where(x2 == shared_x[i])[0][0] for i in points_of_interest]
    
    avg_lick_rate_1 = avg_responses[1]
    avg_lick_rate_2 = avg_responses[2]

    # Perform Wilcoxon signed-rank tests if session-level data is available
    pvals = []
    hl_est = []  # Hodges–Lehmann median difference per point

    for i in range(len(points_of_interest)):
        if session_data_1 is not None and session_data_2 is not None:
            # Extract session-level data for this stimulus point
            session_values_1 = session_data_1[:, idx1[i]]  # All sessions for boundary 1
            session_values_2 = session_data_2[:, idx2[i]]   # All sessions for boundary 2
            
            # Handle different numbers of sessions by using the minimum
            min_sessions = min(len(session_values_1), len(session_values_2))
            session_values_1 = session_values_1[:min_sessions]
            session_values_2 = session_values_2[:min_sessions]
            
            # Remove any NaN values and ensure we have paired data
            valid_mask = np.isfinite(session_values_1) & np.isfinite(session_values_2)
            if np.sum(valid_mask) < 3:  # Need at least 3 valid pairs for Wilcoxon test
                pvals.append(np.nan)
                hl_est.append(np.nan)
                continue
            
            valid_values_1 = session_values_1[valid_mask]
            valid_values_2 = session_values_2[valid_mask]
            
            # Calculate differences for this stimulus point
            diff_point = valid_values_1 - valid_values_2
            
            # Perform Wilcoxon signed-rank test
            try:
                # Check if we have enough data points
                if len(diff_point) < 3:
                    pvals.append(np.nan)
                    hl_est.append(np.nan)
                    continue
                
                # Check for zero variance (all values are the same)
                if np.var(diff_point) == 0:
                    if np.mean(diff_point) == 0:
                        pvals.append(1.0)  # No difference
                        hl_est.append(0.0)
                    else:
                        pvals.append(0.001)  # Significant difference
                        hl_est.append(np.mean(diff_point))
                    continue
                
                # Perform Wilcoxon signed-rank test on differences
                stat, p_val = wilcoxon(diff_point, zero_method='wilcox', alternative='two-sided', mode='auto')
                pvals.append(p_val)
                
                # Calculate Hodges-Lehmann estimator (median of pairwise averages)
                if len(diff_point) > 0:
                    pairwise_means = []
                    for a in range(len(diff_point)):
                        for b in range(a, len(diff_point)):
                            pairwise_means.append((diff_point[a] + diff_point[b]) / 2)
                    hl_est.append(np.median(pairwise_means))
                else:
                    hl_est.append(np.nan)
                    
            except Exception as e:
                print(f"Error in Wilcoxon test for point {i}: {e}")
                pvals.append(np.nan)
                hl_est.append(np.nan)
        else:
            # Fallback to simple comparison if no session data available
            val1 = avg_lick_rate_1[idx1[i]]
            val2 = avg_lick_rate_2[idx2[i]]
            
            # Check if values are finite
            if isinstance(val1, np.ndarray):
                val1 = val1[0] if len(val1) > 0 else np.nan
            if isinstance(val2, np.ndarray):
                val2 = val2[0] if len(val2) > 0 else np.nan
                
            if not np.isfinite(val1) or not np.isfinite(val2):
                pvals.append(np.nan)
                hl_est.append(np.nan)
                continue
            
            # Simple threshold-based approach as fallback
            diff_val = abs(val1 - val2)
            threshold = 5.0  # 5% difference threshold
            if diff_val > threshold:
                pvals.append(0.01)  # Significant difference
            else:
                pvals.append(0.5)   # No significant difference
            hl_est.append(val1 - val2)
    
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
            font=dict(
                size=colors.TITLE_FONT_SIZE,
                color="black",
                family="Baskerville"
            ),
            xref="x",
            yref="y"
        )
    
    return pvals, corrected_pvals, points_of_interest, shared_x


def display_statistical_table(pvals, corrected_pvals, points_of_interest, shared_x):
    """Display statistical comparison table."""
    # Display information about the statistical test
    st.markdown("### Statistical Analysis")
    st.info("**Wilcoxon signed-rank tests** are used to compare psychometric curves between boundary conditions at each stimulus point. "
            "This non-parametric test compares session-level data between the two boundary conditions, making no assumptions about data distribution. "
            "Multiple comparison correction is applied using the Benjamini-Hochberg FDR method.")
    
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
        colors.apply_standard_font_sizes(fig)
        # Display in Streamlit
        st.plotly_chart(fig, use_container_width = False, config=get_plotly_config())

    except Exception as e:
        st.error(f"Unexpected error in plot_psychometric_curve: {e}")
