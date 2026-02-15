# Removed imports to avoid circular dependency

from Analysis.GNG_bpod_analysis.GNG_bpod_general import *
import Analysis.GNG_bpod_analysis.colors as colors

import numpy as np
import pandas as pd
import altair as alt
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import streamlit as st
import plotly.graph_objects as go
import ast
from Analysis.GNG_bpod_analysis.GNG_bpod_general import get_plotly_config

# Function to calculate the d prime
def d_prime(selected_data, index=0, t=10, plot=False, filter_fa_equal_hit=None):
    from Analysis.GNG_bpod_analysis.licking_and_outcome import licking_rate

    # Decide whether to filter Early Response trials at the trial level
    filter_early = get_global_early_response_filter()

    data_for_rates = selected_data
    rate_index = index

    if filter_early:
        try:
            # Work on a single-session copy so we don't mutate the original DataFrame
            session_df = selected_data.loc[[index]].copy().reset_index(drop=True)

            trials_val = session_df.at[0, "TrialTypes"]
            outcomes_val = session_df.at[0, "Outcomes"]

            trialtypes = to_array(trials_val)
            outcomes = to_array(outcomes_val)

            if len(trialtypes) == len(outcomes) and len(outcomes) > 0:
                early_mask = np.array(
                    ['Early Response' not in str(o) for o in outcomes],
                    dtype=bool,
                )
                # Apply mask only if it matches length
                if early_mask.size == len(trialtypes):
                    trialtypes = trialtypes[early_mask]
                    outcomes = outcomes[early_mask]
                    session_df.at[0, "TrialTypes"] = str(trialtypes.tolist())
                    session_df.at[0, "Outcomes"] = str(outcomes.tolist())
                    data_for_rates = session_df
                    rate_index = 0
        except Exception:
            # On any issue, fall back to unfiltered data
            data_for_rates = selected_data
            rate_index = index
    rates, frac = licking_rate(data_for_rates, index=rate_index, t=t, plot=False)
    

    # Ensure the DataFrame has valid numeric data
    frac = frac.dropna(how = "all").astype(float)

    # Convert percentages to proportions (0-1 scale)
    hit_rate = frac["Go"] / 100
    fa_rate = frac["NoGo"] / 100
    # Filter out bins where hit_rate + fa_rate <= 0.3
    valid_bins = (hit_rate + fa_rate) > 0.3
    hit_rate = hit_rate[valid_bins]
    fa_rate = fa_rate[valid_bins]
    
    # Determine and persist global preference for filtering 100%/100% bins
    # Priority: explicit arg (only to initialize) > existing session_state > default False
    if plot:
        # Initialize session state before creating the widget to avoid modification-after-instantiation errors
        if "filter_fa_equal_hit" not in st.session_state:
            st.session_state["filter_fa_equal_hit"] = bool(filter_fa_equal_hit) if filter_fa_equal_hit is not None else False
        # Render the single global checkbox; do not manually overwrite session_state afterwards
        new_pref = st.checkbox(
            "Filter out bins where FA rate = Hit rate = 100%",
            value=st.session_state["filter_fa_equal_hit"],
            key="filter_fa_equal_hit",
            help="When enabled, remove bins where both hit and false alarm rates are 100%"
        )
        filter_fa_equal_hit = bool(new_pref)
    else:
        # Non-plot calls follow persisted preference unless explicitly provided
        if filter_fa_equal_hit is None:
            filter_fa_equal_hit = bool(st.session_state.get("filter_fa_equal_hit", False))
    
    # Filter out bins only when both hit_rate and fa_rate are effectively 100%
    if filter_fa_equal_hit:
        tolerance = 1e-6
        both_hundred_mask = ~((hit_rate >= 1.0 - tolerance) & (fa_rate >= 1.0 - tolerance))
        hit_rate = hit_rate[both_hundred_mask]
        fa_rate = fa_rate[both_hundred_mask]


    # Prevent hit_rate and fa_rate from being exactly 0 or 1
    hit_rate = hit_rate.clip(1e-3, 1 - 1e-3)
    fa_rate = fa_rate.clip(1e-3, 1 - 1e-3)

    # Compute d'
    d = norm.ppf(hit_rate) - norm.ppf(fa_rate)

    # compute criterion
    c = -0.5 * (norm.ppf(hit_rate) + norm.ppf(fa_rate))


    # compute beta
    beta = np.exp(-d*c)

    
    # Create a DataFrame with a safer column name
    df = pd.DataFrame({"index": range(len(d)), "d_prime": d, "criterion": c, "beta": beta})
    if plot:
        st.subheader("d' over trials")
        fig = go.Figure()
        # Helper to add segmented error bands without bridging gaps
        def _add_error_band(center_series, error_series, x_series, fillcolor, upper_name, lower_name):
            try:
                c = np.asarray(center_series, dtype=float)
                e = np.asarray(error_series, dtype=float)
                x = np.asarray(x_series, dtype=float)
            except Exception:
                return
            if not np.isfinite(e).any():
                return
            valid = np.isfinite(c) & np.isfinite(e)
            if not valid.any():
                return
            # Find contiguous segments of True in 'valid'
            indices = np.where(valid)[0]
            if indices.size == 0:
                return
            # Split where gaps are >1 index apart
            splits = np.where(np.diff(indices) > 1)[0] + 1
            segments = np.split(indices, splits)
            for seg in segments:
                if seg.size == 0:
                    continue
                xs = x[seg]
                upper = c[seg] + e[seg]
                lower = c[seg] - e[seg]
                fig.add_trace(go.Scatter(
                    x=xs,
                    y=upper,
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip',
                    name=upper_name
                ))
                fig.add_trace(go.Scatter(
                    x=xs,
                    y=lower,
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor=fillcolor,
                    showlegend=False,
                    hoverinfo='skip',
                    name=lower_name
                ))
        fig.add_trace(go.Scatter(
            x=df['index'], y=df['d_prime'], mode='lines',
            name="Overall",
            line=dict(color=colors.COLOR_ACCENT),
            hovertemplate="Trial: %{x}<br>d': %{y:.3f}<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=[df['index'].min(), df['index'].max()], y=[1, 1],
            mode='lines', name="Learning Threshold",
            line=dict(color=colors.COLOR_GRAY, dash='dash'),
            hoverinfo='skip', showlegend=True
        ))
        fig.add_trace(go.Scatter(
            x=df['index'], y=df['criterion'], mode='lines',
            name="Criterion",
            line=dict(color=colors.COLOR_SUBTLE),
            hovertemplate="Trial: %{x}<br>Criterion: %{y:.3f}<extra></extra>"
        ))

        fig.update_layout(
            xaxis_title="Trial Index",
            yaxis_title="d'",
            title="d' over trials",
            legend=dict(title="Legend"),
            height=400,
            width=700
        )
        colors.apply_standard_font_sizes(fig)
        st.plotly_chart(fig, use_container_width=True, config=get_plotly_config())
    return d

def d_prime_multiple_sessions(selected_data, t=10, animal_name='None', plot = True):
    if animal_name == "None":
        # Step 1: Let the user choose an animal from the data, assign a unique key
        animal_name = st.selectbox("Choose an Animal", selected_data["MouseName"].unique(), key="d_prime_animal_select")
    data = []
    # Step 2: Automatically select all sessions for the chosen animal
    session_indices, session_dates = get_sessions_for_animal(selected_data, animal_name)

    ds = np.zeros([len(session_indices), 3])  # mean, std, max for each session
    tones_per_class = []
    boundaries = []

    # For low/high boundary overlays
    low_boundary_means = []
    high_boundary_means = []
    low_boundary_stds = []
    high_boundary_stds = []

    # Compute d' statistics and collect metadata
    for idx, sess_idx in enumerate(session_indices):
        d = d_prime(selected_data, index=sess_idx, t=t)
        
        # Handle empty arrays safely
        if len(d) == 0 or np.all(np.isnan(d)):
            ds[idx, 0] = np.nan  # mean
            ds[idx, 1] = np.nan  # std
            ds[idx, 2] = np.nan  # max
        else:
            ds[idx, 0] = np.nanmean(d)
            ds[idx, 1] = np.nanstd(d)
            ds[idx, 2] = np.nanmax(d) if len(d[~np.isnan(d)]) > 0 else np.nan
        # Retrieve session metadata
        tones_per_class.append(selected_data.loc[sess_idx, 'Tones_per_class'])
        boundaries.append(selected_data.loc[sess_idx, 'N_Boundaries'])

        # If N_Boundaries==2, get low/high boundary d' for this session
        if selected_data.loc[sess_idx, 'N_Boundaries'] == 2:
            df_low, df_high = d_prime_low_high_boundary_sessions(selected_data.loc[[sess_idx]], sess_idx, t=t, plot=False)
            # Store mean d' for each boundary for this session
            low_boundary_means.append(np.nanmean(df_low['d_prime']))
            high_boundary_means.append(np.nanmean(df_high['d_prime']))
            # Store std d' for each boundary for this session
            low_boundary_stds.append(np.nanstd(df_low['d_prime']))
            high_boundary_stds.append(np.nanstd(df_high['d_prime']))
        else:
            low_boundary_means.append(np.nan)
            high_boundary_means.append(np.nan)
            low_boundary_stds.append(np.nan)
            high_boundary_stds.append(np.nan)

    # Build DataFrame for plotting
    data = pd.DataFrame({
        'Session Index':   np.arange(1, len(session_indices) + 1),
        'SessionDate':     session_dates,
        'd_prime': ds[:, 0],
        'Error': ds[:, 1],
        'Max_d_prime': ds[:, 2],
        'tones_per_class': tones_per_class,
        'Boundaries':      boundaries,
        'Low Boundary d_prime': low_boundary_means,
        'High Boundary d_prime': high_boundary_means,
        'Low Boundary Error': low_boundary_stds,
        'High Boundary Error': high_boundary_stds
    })

    # Clean up isolated std values (replace with NaN if surrounded by NaN)
    for col in ['Low Boundary Error', 'High Boundary Error']:
        for i in range(1, len(data) - 1):
            if pd.isna(data.loc[i, col]):
                continue
            if pd.isna(data.loc[i-1, col]) and pd.isna(data.loc[i+1, col]):
                data.loc[i, col] = np.nan

    if plot:
        import plotly.graph_objects as go
        fig = go.Figure()
        # Helper to add segmented error bands without bridging gaps
        def _add_error_band(center_series, error_series, x_series, fillcolor, upper_name, lower_name):
            try:
                c = np.asarray(center_series, dtype=float)
                e = np.asarray(error_series, dtype=float)
                x = np.asarray(x_series, dtype=float)
            except Exception:
                return
            if not np.isfinite(e).any():
                return
            valid = np.isfinite(c) & np.isfinite(e)
            if not valid.any():
                return
            # Find contiguous segments of True in 'valid'
            indices = np.where(valid)[0]
            if indices.size == 0:
                return
            # Split where gaps are >1 index apart
            splits = np.where(np.diff(indices) > 1)[0] + 1
            segments = np.split(indices, splits)
            for seg in segments:
                if seg.size == 0:
                    continue
                xs = x[seg]
                upper = c[seg] + e[seg]
                lower = c[seg] - e[seg]
                fig.add_trace(go.Scatter(
                    x=xs,
                    y=upper,
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip',
                    name=upper_name
                ))
                fig.add_trace(go.Scatter(
                    x=xs,
                    y=lower,
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor=fillcolor,
                    showlegend=False,
                    hoverinfo='skip',
                    name=lower_name
                ))
        # Prepare marker arrays (shape by boundaries, size by tones per class)
        try:
            marker_symbols = colors.marker_symbols_from_boundaries(data['Boundaries'])
        except Exception:
            marker_symbols = ['circle'] * len(data)
        try:
            marker_sizes = colors.marker_sizes_from_tones(data['tones_per_class'], scale=5.0, default_size=6.0)
        except Exception:
            marker_sizes = [6.0] * len(data)

        # Main overall d' line and markers slightly above the line
        y_vals = np.asarray(data['d_prime'], dtype=float)
        y_level = (np.nanmin(y_vals) if np.isfinite(y_vals).any() else 0.0) - 1.5
        fig.add_trace(go.Scatter(
            x=data['Session Index'], y=data['d_prime'], mode='lines',
            name="Overall d'",
            line=dict(color=colors.COLOR_ACCENT)
        ))
        fig.add_trace(go.Scatter(
            x=data['Session Index'], y=[y_level] * len(y_vals), mode='markers',
            name="Overall d'",
            marker=dict(symbol=marker_symbols, size=marker_sizes, color=colors.COLOR_GRAY),
            showlegend=False
        ))
        # Legend entries for marker shapes (number of boundaries)
        try:
            unique_bounds = sorted(set(int(nb) for nb in data['Boundaries'] if pd.notna(nb)))
        except Exception:
            unique_bounds = [1, 2]
        colors.add_marker_legends(fig, data['Boundaries'], data['tones_per_class'], scale=5.0)
        # Overall d' error band (segmented)
        _add_error_band(
            data['d_prime'],
            data['Error'],
            data['Session Index'],
            colors.COLOR_ACCENT_TRANSPARENT,
            '+1 Std',
            '-1 Std'
        )
        # Low boundary overlay and error band
        y_low = np.asarray(data['Low Boundary d_prime'], dtype=float)
        y_low_level = (np.nanmin(y_low) if np.isfinite(y_low).any() else 0.0) - 1.5
        fig.add_trace(go.Scatter(
            x=data['Session Index'], y=y_low,
            mode='lines', name="Low Boundary d'",
            line=dict(color=colors.COLOR_LOW_BD)
        ))

        _add_error_band(
            data['Low Boundary d_prime'],
            data['Low Boundary Error'],
            data['Session Index'],
            colors.COLOR_LOW_BD_TRANSPARENT,
            '+1 Std Low',
            '-1 Std Low'
        )

        # High boundary overlay and error band
        y_high = np.asarray(data['High Boundary d_prime'], dtype=float)
        y_high_level = (np.nanmin(y_high) if np.isfinite(y_high).any() else 0.0) - 1.5
        fig.add_trace(go.Scatter(
            x=data['Session Index'], y=y_high,
            mode='lines', name="High Boundary d'",
            line=dict(color=colors.COLOR_HIGH_BD)
        ))

        _add_error_band(
            data['High Boundary d_prime'],
            data['High Boundary Error'],
            data['Session Index'],
            colors.COLOR_HIGH_BD_TRANSPARENT,
            '+1 Std High',
            '-1 Std High'
        )

        # Learning threshold
        fig.add_trace(go.Scatter(
            x=[data['Session Index'].min(), data['Session Index'].max()], y=[1, 1],
            mode='lines', name="Learning Threshold",
            line=dict(color=colors.COLOR_GRAY, dash='dash'),
            hoverinfo='skip', showlegend=True
        ))
        fig.update_layout(
            xaxis_title="Session Index",
            yaxis_title="d'",
            title=f"d' Progress for {animal_name}",
            legend=dict(title="Legend"),
            height=400,
            width=700
        )
        colors.apply_standard_font_sizes(fig)
        st.plotly_chart(fig, use_container_width=True, config=get_plotly_config())

    return data

def multi_animal_d_prime_progression(selected_data, N_Boundaries=None):
    """
    Plot d' progression across animals, optionally filtered by number of boundaries.
    This is robust to empty selections, missing keys and NaNs.
    """
    # Optional filtering by number of boundaries
    if N_Boundaries is not None:
        st.title(f"Multi-Animal Progress ({N_Boundaries} boundaries)")
        # Guard against NaN comparison issues
        if pd.isna(N_Boundaries):
            filtered = selected_data[selected_data["N_Boundaries"].isna()]
        else:
            filtered = selected_data[selected_data["N_Boundaries"] == N_Boundaries]
        if filtered.empty:
            st.warning("No sessions match the selected number of boundaries.")
            return
        selected_data = filtered.reset_index(drop=True)
    else:
        st.title("Multi-Animal Progress")

    if selected_data.empty:
        st.warning("No data available for multi-animal d' progression.")
        return

    # Get unique subject names
    subjects = selected_data["MouseName"].unique()
    if len(subjects) == 0:
        st.warning("No animals found in the selected data.")
        return

    # Initialize arrays to store d' values
    d_prime_data = []
    session_counts = []
    low_boundary_data = []
    high_boundary_data = []
    tones_per_class_list = []
    boundaries_list = []

    for subject in subjects:
        # Compute d' for each subject
        d_prime_result = d_prime_multiple_sessions(selected_data, animal_name=subject, plot=False)
        d_prime_values = np.asarray(d_prime_result.get("d_prime", []), dtype=float)
        if d_prime_values.size == 0:
            continue  # skip animals with no valid data
        d_prime_data.append(d_prime_values)
        session_counts.append(len(d_prime_values))

        # Collect tones per class and boundaries for marker encoding
        try:
            tones = np.asarray(d_prime_result.get("tones_per_class", []), dtype=float)
            if tones.size == 0:
                raise ValueError
            tones_per_class_list.append(tones)
        except Exception:
            tones_per_class_list.append(np.array([np.nan] * len(d_prime_values), dtype=float))

        try:
            bounds = np.asarray(d_prime_result.get("Boundaries", []), dtype=float)
            if bounds.size == 0:
                raise ValueError
            boundaries_list.append(bounds)
        except Exception:
            boundaries_list.append(np.array([np.nan] * len(d_prime_values), dtype=float))

        # Collect low/high boundary d' as well (always attempt; fall back to NaNs)
        low_vals = np.asarray(d_prime_result.get("Low Boundary d_prime", []), dtype=float)
        high_vals = np.asarray(d_prime_result.get("High Boundary d_prime", []), dtype=float)
        # Fallback to NaNs if shapes don't match expected session length
        if low_vals.size != len(d_prime_values):
            low_vals = np.full(len(d_prime_values), np.nan)
        if high_vals.size != len(d_prime_values):
            high_vals = np.full(len(d_prime_values), np.nan)
        low_boundary_data.append(low_vals)
        high_boundary_data.append(high_vals)

    # If no valid animals after filtering / cleaning, exit gracefully
    if not d_prime_data:
        st.warning("No valid d' data found for the selected animals.")
        return

    # Determine max number of sessions for alignment
    max_sessions = max(session_counts)

    # Convert list of arrays to DataFrame (aligned by padding with NaN)
    d_prime_df = pd.DataFrame([np.pad(d, (0, max_sessions - len(d)), constant_values=np.nan) for d in d_prime_data])
    low_boundary_df = pd.DataFrame([np.pad(d, (0, max_sessions - len(d)), constant_values=np.nan) for d in low_boundary_data])
    high_boundary_df = pd.DataFrame([np.pad(d, (0, max_sessions - len(d)), constant_values=np.nan) for d in high_boundary_data])
    tones_df = pd.DataFrame([np.pad(arr, (0, max_sessions - len(arr)), constant_values=np.nan) for arr in tones_per_class_list])
    bounds_df = pd.DataFrame([np.pad(arr, (0, max_sessions - len(arr)), constant_values=np.nan) for arr in boundaries_list])

    # Compute average d' across subjects (and std across animals for error bands)
    avg_d_prime = d_prime_df.mean(axis=0, skipna=True)
    std_d_prime = d_prime_df.std(axis=0, skipna=True)

    # Compute representative tones and boundaries per session position
    avg_tones = tones_df.mean(axis=0, skipna=True)
    # Mode for boundaries per session; fallback to nearest non-NaN
    rep_bounds = []
    for col in bounds_df.columns:
        s = bounds_df[col].dropna()
        if len(s) == 0:
            rep_bounds.append(np.nan)
        else:
            try:
                modes = s.mode()
                rep_bounds.append(float(modes.iloc[0]) if len(modes) > 0 else float(s.iloc[0]))
            except Exception:
                rep_bounds.append(float(s.iloc[0]))

    # Compute average low/high boundary d' across animals (and std across animals for error bands)
    low_df = pd.DataFrame([np.pad(d, (0, max_sessions - len(d)), constant_values=np.nan) for d in low_boundary_data])
    high_df = pd.DataFrame([np.pad(d, (0, max_sessions - len(d)), constant_values=np.nan) for d in high_boundary_data])
    avg_low_boundary = low_df.mean(axis=0, skipna=True)
    avg_high_boundary = high_df.mean(axis=0, skipna=True)
    std_low_boundary = low_df.std(axis=0, skipna=True)
    std_high_boundary = high_df.std(axis=0, skipna=True)

    import plotly.graph_objects as go
    fig = go.Figure()

    # Helper to add segmented error bands without bridging gaps
    def _add_error_band(center_series, error_series, x_series, fillcolor, upper_name, lower_name):
        try:
            c = np.asarray(center_series, dtype=float)
            e = np.asarray(error_series, dtype=float)
            x = np.asarray(x_series, dtype=float)
        except Exception:
            return
        if not np.isfinite(e).any():
            return
        valid = np.isfinite(c) & np.isfinite(e)
        if not valid.any():
            return
        indices = np.where(valid)[0]
        if indices.size == 0:
            return
        splits = np.where(np.diff(indices) > 1)[0] + 1
        segments = np.split(indices, splits)
        for seg in segments:
            if seg.size == 0:
                continue
            xs = x[seg]
            upper = c[seg] + e[seg]
            lower = c[seg] - e[seg]
            fig.add_trace(go.Scatter(
                x=xs,
                y=upper,
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip',
                name=upper_name
            ))
            fig.add_trace(go.Scatter(
                x=xs,
                y=lower,
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor=fillcolor,
                showlegend=False,
                hoverinfo='skip',
                name=lower_name
            ))
    # Plot average overall d' with markers slightly above the line
    try:
        marker_sizes = colors.marker_sizes_from_tones(avg_tones, scale=5.0, default_size=6.0)
    except Exception:
        marker_sizes = [6.0] * len(avg_d_prime)
    try:
        marker_symbols = colors.marker_symbols_from_boundaries(rep_bounds)
    except Exception:
        marker_symbols = ['circle'] * len(avg_d_prime)
    y_vals = np.asarray(avg_d_prime, dtype=float)
    y_level = (np.nanmin(y_vals) if np.isfinite(y_vals).any() else 0.0) - 1.5
    fig.add_trace(go.Scatter(
        x=np.arange(1, max_sessions + 1),
        y=avg_d_prime,
        mode='lines',
        name="Average Overall d'",
        line=dict(color=colors.COLOR_ACCENT, width=5)
    ))
    fig.add_trace(go.Scatter(
        x=np.arange(1, max_sessions + 1),
        y=[y_level] * len(y_vals),
        mode='markers',
        name="Average Overall d'",
        marker=dict(symbol=marker_symbols, size=marker_sizes, color=colors.COLOR_GRAY),
        showlegend=False
    ))
    # Overall d' error band (segmented)
    _add_error_band(
        avg_d_prime,
        std_d_prime,
        np.arange(1, max_sessions + 1),
        colors.COLOR_ACCENT_TRANSPARENT,
        "+1 Std Overall",
        "-1 Std Overall"
    )
    # Add legends for marker shapes and sizes
    colors.add_marker_legends(fig, rep_bounds, avg_tones, scale=5.0)
    # Plot average low/high boundary d'
    fig.add_trace(go.Scatter(
        x=np.arange(1, max_sessions + 1),
        y=avg_low_boundary,
        mode='lines',
        name="Average Low Boundary d'",
        line=dict(color=colors.COLOR_LOW_BD, width=2, dash='solid')
    ))
    low_vals = np.asarray(avg_low_boundary, dtype=float)
    low_level = (np.nanmin(low_vals) if np.isfinite(low_vals).any() else 0.0) - 1.5

    # Low boundary error band
    _add_error_band(
        avg_low_boundary,
        std_low_boundary,
        np.arange(1, max_sessions + 1),
        colors.COLOR_LOW_BD_TRANSPARENT,
        "+1 Std Low",
        "-1 Std Low"
    )

    fig.add_trace(go.Scatter(
        x=np.arange(1, max_sessions + 1),
        y=avg_high_boundary,
        mode='lines',
        name="Average High Boundary d'",
        line=dict(color=colors.COLOR_HIGH_BD, width=2, dash='solid')
    ))
    high_vals = np.asarray(avg_high_boundary, dtype=float)
    high_level = (np.nanmin(high_vals) if np.isfinite(high_vals).any() else 0.0) - 1.5

    # High boundary error band
    _add_error_band(
        avg_high_boundary,
        std_high_boundary,
        np.arange(1, max_sessions + 1),
        colors.COLOR_HIGH_BD_TRANSPARENT,
        "+1 Std High",
        "-1 Std High"
    )
    # Learning threshold
    fig.add_trace(go.Scatter(
        x=[1, max_sessions], y=[1, 1],
        mode='lines', name="Learning Threshold",
        line=dict(color=colors.COLOR_GRAY, dash='dash'),
        hoverinfo='skip', showlegend=True
    ))
    fig.update_layout(
        xaxis_title="Session Index",
        yaxis_title="d'",
        title="d' Progression Across Animals",
        legend=dict(title="Legend"),
        height=400,
        width=700
    )
    colors.apply_standard_font_sizes(fig)
    st.plotly_chart(fig, use_container_width=True, config=get_plotly_config())
    return avg_d_prime, avg_low_boundary, avg_high_boundary

def classifier_metric(project_data, index):
    from Analysis.GNG_bpod_analysis.licking_and_outcome import responses, licking_rate

    response = responses(project_data, index)
    hit = np.array(response["Hit"][-1:])[0]
    miss = np.array(response["Miss"][-1:])[0]
    cr = np.array(response["CR"][-1:])[0]
    fa = np.array(response["FA"][-1:])[0]

    # --- Early response rates (Go/NoGo) ---
    # Why: Educage/Bpod exports often include "Early Response" trials which are
    # not part of the confusion-matrix cumulative counts above.
    trialtypes_raw = project_data.iloc[index].get("TrialTypes", None)
    outcomes_raw = project_data.iloc[index].get("Outcomes", None)
    trialtypes_arr = to_array(trialtypes_raw)
    outcomes_arr = to_array(outcomes_raw)

    early_mask = np.array(
        ["early response" in str(o).lower() for o in outcomes_arr],
        dtype=bool,
    )
    trialtypes_norm = np.array([str(t).strip() for t in trialtypes_arr], dtype=object)
    go_mask = np.array([str(t).strip().lower() == "go" for t in trialtypes_norm], dtype=bool)
    nogo_mask = np.array([str(t).strip().lower().replace("-", "").replace(" ", "") == "nogo" for t in trialtypes_norm], dtype=bool)

    n_go = int(np.sum(go_mask))
    n_nogo = int(np.sum(nogo_mask))
    early_go_n = int(np.sum(early_mask & go_mask))
    early_nogo_n = int(np.sum(early_mask & nogo_mask))

    d = d_prime(project_data, index, t=10)
    # Compute rates
    hit_rate = hit / (hit + miss)
    fa_rate = fa / (fa + cr)

    # valid_bins = (hit_rate+fa_rate) > 0.4
    # Avoid extreme values (log(0) issue)
    hit_rate = np.clip(hit_rate, 0.01, 0.99)
    fa_rate = np.clip(fa_rate, 0.01, 0.99)

    # Compute d' and criterion (c)
    d_mean = np.nanmean(d)
    criterion_c = -0.5 * (norm.ppf(hit_rate) + norm.ppf(fa_rate))
    beta = np.exp(-d_mean*criterion_c)
    

    # Accuracy
    accuracy = (hit + cr) / (hit + miss + cr + fa)

    # Precision (PPV)
    precision = hit / (hit + fa) if (hit + fa) > 0 else 0

    # Recall (Sensitivity, TPR)
    recall = hit / (hit + miss) if (hit + miss) > 0 else 0

    # F1-score
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # False Positive Rate (FPR)
    fpr = fa / (fa + cr) if (fa + cr) > 0 else 0

    # Simulating ROC-AUC (Assumes mouse responses as probabilistic)
    y_true = np.array([1] * (hit + miss) + [0] * (cr + fa))  # Actual labels (1=Go, 0=No-Go)
    y_scores = np.array([1] * hit + [0] * miss + [1] * fa + [0] * cr)  # Mouse responses

    roc_auc = roc_auc_score(y_true, y_scores)

    # ---- STREAMLIT APP ----
    # Creating a DataFrame with classification metrics as columns
    class_metric_df = pd.DataFrame({
        "Hit Rate":                  [hit_rate],
        "False Alarm Rate":          [fa_rate],
        "d'":                        [d_mean],
        "Criterion":                 [criterion_c],
        "Beta":                      [beta],
        "Accuracy":                  [accuracy],
        "Precision":                 [precision],
        "F1 Score":                  [f1_score],
        "ROC-AUC Score":             [roc_auc]
    })

    # Convert numbers to 3 decimal places
    class_metric_df = class_metric_df.applymap(lambda x: round(x, 3))

    # Configure explanations for each column
    column_explanations = {
        "Hit Rate":                  st.column_config.NumberColumn(
            "Hit Rate",
            help = "Proportion of correctly detected 'Go' trials (TP / (TP + FN))"
        ),
        "False Alarm Rate":          st.column_config.NumberColumn(
            "False Alarm Rate",
            help = "Proportion of incorrect responses to 'No-Go' trials (FP / (FP + TN))"
        ),
        "d'":                        st.column_config.NumberColumn(
            "d'",
            help = "Sensitivity index (d-prime) measuring signal detection ability: Z(Hit Rate) - Z(False Alarm Rate)"
        ),
        "Accuracy":                  st.column_config.NumberColumn(
            "Accuracy",
            help = "Overall correctness of the classifier: (TP + TN) / Total Trials"
        ),
        "Precision":                 st.column_config.NumberColumn(
            "Precision",
            help = "Proportion of predicted 'Go' responses that were correct: TP / (TP + FP)"
        ),
        "F1 Score":                  st.column_config.NumberColumn(
            "F1 Score",
            help = "Harmonic mean of Precision and Recall: 2 * (Precision * Recall) / (Precision + Recall)"
        ),
        "Criterion":                 st.column_config.NumberColumn(
            "Criterion",
            help = "Criterion value: -0.5 * (Z(Hit Rate) + Z(False Alarm Rate)), liberal is negative, conservative is positive"
        ),
        "ROC-AUC Score":             st.column_config.NumberColumn(
            "ROC-AUC Score",
            help = "Area Under the ROC Curve (AUC), indicating classifier performance (higher is better)"
        )
    }

    # Display the styled table with explanations
    st.dataframe(class_metric_df, column_config = column_explanations)

    # ---- CONFUSION MATRIX ----

    conf_matrix = np.array([[hit, miss], [fa, cr]])
    df_cm = pd.DataFrame(conf_matrix, columns = pd.Index(["Go", "No-Go"]),
                         index = pd.Index(["Go", "No-Go"]))

    # Heatmap with text overlay
    base = alt.Chart(df_cm.reset_index().melt(id_vars = "index")).encode(
        x = alt.X("variable:N", title = "Predicted Label", axis = alt.Axis(labelAngle = 0)),
        y = alt.Y("index:N", title = "True Label")
    )

    # Heatmap
    heatmap = base.mark_rect().encode(
        color = "value:Q",
        tooltip = ["index", "variable", "value"]
    )

    # Text annotations
    text = base.mark_text(size = 20, font="Source Sans Pro", color = "black").encode(
        text = "value:Q"
    )

    confusion_chart = (heatmap + text).properties(
        width = 400,
        height = 300
    )


    # ---- ROC CURVE ----
    fpr_vals, tpr_vals, _ = roc_curve(y_true, y_scores)

    df_roc = pd.DataFrame({
        "False Positive Rate": fpr_vals,
        "True Positive Rate":  tpr_vals
    })

    roc_chart = alt.Chart(df_roc).mark_line().encode(
        x = "False Positive Rate:Q",
        y = "True Positive Rate:Q",
        tooltip = ["False Positive Rate", "True Positive Rate"]
    ).properties(
        width = 300,
        height = 300
    )

    diagonal = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(strokeDash = [5, 5],
                                                                             color = "gray").encode(
        x = "x:Q",
        y = "y:Q"
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        st.altair_chart(confusion_chart, use_container_width = False)
    with col2:
        st.altair_chart(roc_chart + diagonal, use_container_width = False)
    with col3:
        st.subheader("Early response")
        st.metric("Early-Go trials", f"{early_go_n}/{n_go}", help="Count of Early Response outcomes in Go trials")
        st.metric("Early-NoGo trials", f"{early_nogo_n}/{n_nogo}", help="Count of Early Response outcomes in NoGo trials")

    return class_metric_df

def to_array(val):
    if isinstance(val, str):
        try:
            return np.array(ast.literal_eval(val))
        except Exception:
            return np.array([])
    elif isinstance(val, (list, np.ndarray)):
        return np.array(val)
    else:
        return np.array([])

def d_prime_low_high_boundary_sessions(selected_data, idx, t=10, plot=True):
    """
    Calculates d' over trials in bins of t for low and high boundary trials (low: Stimuli < st.session_state.high_boundary, high: Stimuli > st.session_state.low_boundary)
    for a single session (selected_data should be a DataFrame with one row).
    Plots both d' curves on the same Plotly figure and returns the d' arrays/DataFrames for both boundaries.
    """

    # Get the stimuli for this session - use index 0 since selected_data has only one row
    stimuli = parse_stimuli(selected_data.iloc[0, selected_data.columns.get_loc('Stimuli')])
    # Low boundary
    low_mask = stimuli < st.session_state.high_boundary
    # High boundary
    high_mask = stimuli > st.session_state.low_boundary

    # Get the raw values - use index 0 since selected_data has only one row
    trialtypes = selected_data.iloc[0, selected_data.columns.get_loc('TrialTypes')]
    outcomes = selected_data.iloc[0, selected_data.columns.get_loc('Outcomes')]

    # Convert to array if needed
    if isinstance(trialtypes, str):
        trialtypes = to_array(trialtypes)
    if isinstance(outcomes, str):
        outcomes = to_array(outcomes)

    trialtypes = np.array(trialtypes)
    outcomes = np.array(outcomes)

    # Now mask
    filtered_trials_low = trialtypes[low_mask]
    filtered_outcomes_low = outcomes[low_mask]
    filtered_trials_high = trialtypes[high_mask]
    filtered_outcomes_high = outcomes[high_mask]


    selected_data_low = selected_data.copy()
    selected_data_low.iloc[0, selected_data_low.columns.get_loc('TrialTypes')] = str(filtered_trials_low.tolist())
    selected_data_low.iloc[0, selected_data_low.columns.get_loc('Outcomes')] = str(filtered_outcomes_low.tolist())

    selected_data_high = selected_data.copy()
    selected_data_high.iloc[0, selected_data_high.columns.get_loc('TrialTypes')] = str(filtered_trials_high.tolist())
    selected_data_high.iloc[0, selected_data_high.columns.get_loc('Outcomes')] = str(filtered_outcomes_high.tolist())

    # Calculate d' for low and high boundary - use index 0 since we have only one row
    d_low = d_prime(selected_data_low, index=0, t=t)
    d_high = d_prime(selected_data_high, index=0, t=t)



    # Prepare DataFrames for plotting/return
    df_low = pd.DataFrame({
        'Bin': range(len(d_low)),
        "d_prime": d_low,
        'Type': 'Low Boundary'
    })
    df_high = pd.DataFrame({
        'Bin': range(len(d_high)),
        "d_prime": d_high,
        'Type': 'High Boundary'
    })

    if plot:
        import plotly.graph_objects as go
        # Display mean and std for each boundary (as plotly annotation)
        mean_low = np.nanmean(df_low['d_prime'])
        std_low = np.nanstd(df_low['d_prime'])
        mean_high = np.nanmean(df_high['d_prime'])
        std_high = np.nanstd(df_high['d_prime'])
        subtitle = (
            f"<span style='color:black'>"
            f"Low Boundary d': {mean_low:.3f}, ± {std_low:.3f} | "
            f"High Boundary d': {mean_high:.3f} ± {std_high:.3f}"
            f"</span>"
        )
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_low['Bin'], y=df_low['d_prime'], mode='lines',
            name="Low Boundary",
            line=dict(color=colors.COLOR_LOW_BD),
            marker=dict(symbol='circle')
        ))
        fig.add_trace(go.Scatter(
            x=df_high['Bin'], y=df_high['d_prime'], mode='lines',
            name="High Boundary",
            line=dict(color=colors.COLOR_HIGH_BD),
            marker=dict(symbol='square')
        ))
        fig.add_trace(go.Scatter(
            x=[0, max(len(d_low), len(d_high)) - 1], y=[1, 1],
            mode='lines', name="Learning Threshold",
            line=dict(color=colors.COLOR_GRAY, dash='dash'),
            hoverinfo='skip', showlegend=True
        ))
        fig.update_layout(
            xaxis_title="Bin Index",
            yaxis_title="d'",
            title=f"d' Progression (Low vs High Boundary)",
            legend=dict(title="Boundary Type"),
            height=400,
            width=700
        )
        fig.add_annotation(
            text=subtitle,
            xref="paper", yref="paper",
            x=0.01, y=1.08,
            showarrow=False,
            font=dict(size=14),
            xanchor='left',
            yanchor='top'
        )
        colors.apply_standard_font_sizes(fig)
        st.plotly_chart(fig, use_container_width=True, config=get_plotly_config())



    return df_low, df_high

def daily_multi_animal_dprime(project_data, t=10):
    """
    Plot d' over trials (bins of t) for all unique mice on a selected date, overlaid.
    """
    if project_data is None or project_data.empty:
        st.info("No data loaded.")
        return
    dates = sorted(project_data["SessionDate"].astype(str).unique())
    if len(dates) == 0:
        st.info("No dates found in data.")
        return
    selected_date = st.selectbox("Select a date", options=dates,
                                 index=max(0, len(dates) - 1), key="daily_multi_dprime_date")
    date_data = project_data[project_data["SessionDate"].astype(str) == str(selected_date)]
    if date_data.empty:
        st.info(f"No data found for date {selected_date}")
        return
    mice = sorted(date_data["MouseName"].unique())
    if len(mice) == 0:
        st.info("No animals found for selected date.")
        return

    # Use per-mouse color map
    try:
        color_map = st.session_state.get('mouse_color_map', {})
        if not color_map:
            from Analysis.GNG_bpod_analysis.colors import get_subject_color_map
            color_map = get_subject_color_map(date_data['MouseName'])
    except Exception:
        color_map = {}

    fig = go.Figure()
    for mouse in mice:
        rows = date_data.index[date_data["MouseName"] == mouse].tolist()
        if not rows:
            continue
        row_idx = rows[0]
        try:
            d_vals = d_prime(project_data, index=row_idx, t=t, plot=False)
        except Exception:
            d_vals = d_prime(project_data.loc[[row_idx]], index=0, t=t, plot=False)
        if d_vals is None or len(d_vals) == 0:
            continue
        x = np.arange(1, len(d_vals) + 1)
        fig.add_trace(go.Scatter(
            x=x,
            y=np.asarray(d_vals, dtype=float),
            mode='lines',
            name=str(mouse),
            line=dict(width=2, color=color_map.get(str(mouse), colors.COLOR_SUBTLE))
        ))

    if len(fig.data) == 0:
        st.info("No d' data found for any animals on selected date.")
        return

    fig.update_layout(
        title=f"d' by Animal — {selected_date} (bin size={t})",
        xaxis_title="Bin index",
        yaxis_title="d'",
        height=500,
        width=900,
        showlegend=True
    )
    colors.apply_standard_font_sizes(fig)
    st.plotly_chart(fig, use_container_width=True, config=get_plotly_config())

def d_prime_for_stim_pairs(selected_data, index=0, stim_pairs=None, t=10, plot=True, atol=1e-6):
    """
    Compare d' for user-specified pairs of stimuli within a single session.

    Parameters:
    - selected_data: DataFrame with a session row containing 'Stimuli', 'TrialTypes', 'Outcomes'.
    - index: Row index of the session to analyze within selected_data.
    - stim_pairs: List of 2-tuples specifying stimulus value pairs, e.g., [(0.8, 1.2), (1.0, 1.4)].
                  If None, a UI will be shown to pick one pair interactively.
    - t: Bin size used by existing d' computation.
    - plot: If True, shows a Plotly bar chart of mean d' with error bars for each pair.
    - atol: Absolute tolerance for matching float stimulus values.

    Returns:
    - pandas.DataFrame with columns: 'Pair', 'mean_d_prime', 'std_d_prime', 'n_bins'.
    """
    # Extract raw arrays for the session
    try:
        stimuli = selected_data.iloc[index]["Stimuli"]
        if isinstance(stimuli, str):
            stimuli = np.fromstring(stimuli.strip("[]"), sep=" ")
        else:
            stimuli = np.array(stimuli)
    except Exception:
        # Fallback to helper
        from Analysis.GNG_bpod_analysis.licking_and_outcome import preprocess_stimuli_outcomes
        stimuli, _ = preprocess_stimuli_outcomes(selected_data, index=index)

    trials_val = selected_data.iloc[index]["TrialTypes"]
    outcomes_val = selected_data.iloc[index]["Outcomes"]
    trialtypes = to_array(trials_val)
    outcomes = to_array(outcomes_val)

    # Determine default pairs if not provided
    if stim_pairs is None:
        unique_vals = np.unique(np.round(stimuli.astype(float), 6))
        unique_vals = np.sort(unique_vals)
        # Try to read number of boundaries for the session
        try:
            n_boundaries = int(selected_data.iloc[index].get('N_Boundaries', 1))
        except Exception:
            n_boundaries = 1

        default_pairs = []
        if unique_vals.size >= 2:
            if n_boundaries == 1:
                # Pair first vs last, second vs end-1, etc.
                half = unique_vals.size // 2
                for i in range(half):
                    default_pairs.append((float(unique_vals[i]), float(unique_vals[-(i+1)])))
            elif n_boundaries == 2:
                # Use boundaries from session_state if available
                low_bd = getattr(st.session_state, 'low_boundary', None)
                high_bd = getattr(st.session_state, 'high_boundary', None)
                # Fallback: infer mid as the midpoint of middle region
                if low_bd is None or high_bd is None:
                    # Heuristic midpoint between 25% and 75%
                    low_bd = float(np.quantile(unique_vals, 0.33))
                    high_bd = float(np.quantile(unique_vals, 0.66))

                # Middle stimulus ~ closest to midpoint between boundaries
                mid_target = (float(low_bd) + float(high_bd)) / 2.0
                mid_idx = int(np.argmin(np.abs(unique_vals - mid_target)))
                mid_val = float(unique_vals[mid_idx])

                # Extremes vs middle
                default_pairs.append((float(unique_vals[0]), mid_val))
                default_pairs.append((float(unique_vals[-1]), mid_val))

                # Around each boundary: closest below and above
                def around_boundary_pairs(boundary):
                    below = unique_vals[unique_vals < boundary]
                    above = unique_vals[unique_vals > boundary]
                    if below.size == 0 or above.size == 0:
                        return None
                    return (float(below[-1]), float(above[0]))

                p_low = around_boundary_pairs(float(low_bd))
                p_high = around_boundary_pairs(float(high_bd))
                if p_low is not None:
                    default_pairs.append(p_low)
                if p_high is not None:
                    default_pairs.append(p_high)

        # De-duplicate and ensure sorted within pair
        if default_pairs:
            norm_pairs = []
            seen = set()
            for a, b in default_pairs:
                sa, sb = (a, b) if a <= b else (b, a)
                key = (round(sa, 6), round(sb, 6))
                if key not in seen and abs(sa - sb) > atol:
                    seen.add(key)
                    norm_pairs.append((sa, sb))
            stim_pairs = norm_pairs

        # If still empty, fall back to simple UI to pick two stimuli
        if not stim_pairs:
            options = [float(v) for v in unique_vals]
            picked = st.multiselect("Pick exactly two stimuli to compare", options=options, key=f"stim_pair_{index}")
            if len(picked) != 2:
                st.info("Select two stimulus values to compute and compare d'.")
                return pd.DataFrame(columns=["Pair", "mean_d_prime", "std_d_prime", "n_bins"]) 
            stim_pairs = [tuple(sorted([float(picked[0]), float(picked[1])]))]

    results = []
    labels = []
    pair_to_values = {}
    pair_meta = []

    def _format_pair_label(a, b):
        a = float(a)
        b = float(b)
        lo, hi = (a, b) if a <= b else (b, a)
        if lo <= 0 or not np.isfinite(lo) or not np.isfinite(hi):
            return f"{a:g} vs {b:g}"
        ratio = hi / lo
        octaves = np.log2(ratio) if ratio > 0 else np.nan
        # Nicely formatted octave value
        if np.isfinite(octaves):
            # Round to nearest 0.05
            oct_rounded = round(octaves * 20) / 20  # 1/20 = 0.05
            if abs(octaves - oct_rounded) < 0.025:
                oct_str = f"{oct_rounded:.2f}".rstrip('0').rstrip('.')  # Remove trailing zeros
            else:
                oct_str = f"{octaves:.2f}"
            return f"{lo * 10:.1f} vs. {hi * 10:.1f} [~{oct_str} oct]"
        return f"{lo * 10:.1f} vs. {hi * 10:.1f}"

    # Prepare boundary refs if available
    try:
        n_boundaries = int(selected_data.iloc[index].get('N_Boundaries', 1))
    except Exception:
        n_boundaries = 1
    low_bd = getattr(st.session_state, 'low_boundary', None)
    high_bd = getattr(st.session_state, 'high_boundary', None)
    # Derive a middle value if needed
    if n_boundaries == 2:
        if low_bd is None or high_bd is None:
            low_bd = float(np.quantile(stimuli, 0.33))
            high_bd = float(np.quantile(stimuli, 0.66))
        mid_val = (float(low_bd) + float(high_bd)) / 2.0
    else:
        mid_val = None

    for (s1, s2) in stim_pairs:
        # Build mask for the two stimuli with tolerance
        mask = np.isclose(stimuli, s1, atol=atol) | np.isclose(stimuli, s2, atol=atol)
        # Safety checks
        if mask.sum() == 0:
            continue
        # Filter arrays
        filtered_trials = trialtypes[mask]
        filtered_outcomes = outcomes[mask]
        # Prepare a one-row DataFrame for existing d' function
        df_one = selected_data.loc[[index]].copy()
        df_one.iloc[0, df_one.columns.get_loc('TrialTypes')] = str(np.asarray(filtered_trials).tolist())
        df_one.iloc[0, df_one.columns.get_loc('Outcomes')] = str(np.asarray(filtered_outcomes).tolist())
        # Compute d' over bins
        d_vals = d_prime(df_one, index=0, t=t, plot=False)
        d_vals = np.asarray(d_vals, dtype=float)
        d_vals = d_vals[~np.isnan(d_vals)]
        mean_dp = float(np.nanmean(d_vals)) if d_vals.size else np.nan
        std_dp = float(np.nanstd(d_vals)) if d_vals.size else np.nan
        results.append((mean_dp, std_dp, int(d_vals.size)))
        label = _format_pair_label(s1, s2)
        labels.append(label)
        pair_to_values[label] = d_vals
        # Compute metadata: category and distance
        a = float(s1); b = float(s2)
        lo, hi = (a, b) if a <= b else (b, a)
        ratio = (hi / lo) if (lo > 0 and np.isfinite(lo) and np.isfinite(hi)) else np.nan
        octaves = np.log2(ratio) if (isinstance(ratio, (int, float)) and ratio > 0) else np.nan
        if n_boundaries == 2 and low_bd is not None and high_bd is not None and mid_val is not None:
            # Determine category
            around_low = (lo < float(low_bd)) and (hi > float(low_bd))
            around_high = (lo < float(high_bd)) and (hi > float(high_bd))
            near_mid = (abs(a - mid_val) < abs(a - float(low_bd)) and abs(a - mid_val) < abs(a - float(high_bd))) or \
                      (abs(b - mid_val) < abs(b - float(low_bd)) and abs(b - mid_val) < abs(b - float(high_bd)))
            if around_low and not around_high:
                category = 'Around Low Boundary'
                color = colors.COLOR_LOW_BD
            elif around_high and not around_low:
                category = 'Around High Boundary'
                color = colors.COLOR_HIGH_BD
            else:
                category = 'General'
                color = colors.COLOR_SUBTLE
        else:
            category = 'Distance Pair'
            color = colors.COLOR_ACCENT
        pair_meta.append({
            'label': label,
            'category': category,
            'color': color,
            'octaves': float(octaves) if np.isfinite(octaves) else np.nan
        })

    # Build result DataFrame
    out_df = pd.DataFrame([
        {"Pair": labels[i], "mean_d_prime": r[0], "std_d_prime": r[1], "n_bins": r[2]}
        for i, r in enumerate(results)
    ])

    if plot and not out_df.empty:
        fig = go.Figure()
        # Build a lookup for meta
        meta_by_label = {m['label']: m for m in pair_meta}
        categories_present = []
        for pair_label in out_df['Pair']:
            vals = pair_to_values.get(pair_label, np.array([]))
            if vals.size == 0:
                continue
            meta = meta_by_label.get(pair_label, {})
            color = meta.get('color', colors.COLOR_ACCENT)
            octaves = meta.get('octaves', np.nan)
            # Line width encodes distance level
            lw = 1
            if np.isfinite(octaves):
                lw = max(1, min(5, int(round(octaves)) + 1))
            fig.add_trace(go.Box(
                y=vals,
                name=pair_label,
                boxpoints='outliers',
                marker_color=color,
                line=dict(color=color, width=lw),
                showlegend=False
            ))
            cat = meta.get('category')
            if cat and cat not in categories_present:
                categories_present.append(cat)
        # Add legend entries for categories
        for cat in categories_present:
            # Use a dummy invisible scatter to show legend color
            sample_color = None
            for m in pair_meta:
                if m['category'] == cat:
                    sample_color = m['color']
                    break
            if sample_color is None:
                sample_color = colors.COLOR_ACCENT
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode='markers',
                marker=dict(color=sample_color, size=10),
                name=cat
            ))
        ordered_pairs = list(out_df['Pair'])
        if len(ordered_pairs) >= 1:
            x_start = ordered_pairs[0]
            x_end = ordered_pairs[-1]
            fig.add_trace(go.Scatter(
                x=[x_start, x_end],
                y=[1, 1],
                mode='lines',
                name="Learning Threshold",
                line=dict(color=colors.COLOR_GRAY, dash='dash'),
                hoverinfo='skip',
                showlegend=True
            ))
        fig.update_layout(
            title="d' distributions by Boundary Category and Distance",
            xaxis_title="Stimulus Pair",
            yaxis_title="d'",
            height=400,
            width=700
        )
        st.plotly_chart(fig, use_container_width=True, config=get_plotly_config(width=1000))
        

    return out_df


def multi_animal_pairwise_dprime(project_data, t=10, compare_platforms=False, 
                                  filter_successful=False, dprime_threshold=1.0):
    """
    Multi-animal pairwise d' analysis - boxplots of d' by distance from boundary.
    
    Parameters:
    - project_data: DataFrame with session data
    - t: Bin size for d' calculation
    - compare_platforms: If True, show side-by-side comparison between Rig and Educage
    - filter_successful: If True, only include sessions where overall d' >= threshold
    - dprime_threshold: Threshold for filtering successful sessions
    
    Returns:
    - DataFrame with aggregated pairwise d' data
    """
    from Analysis.GNG_bpod_analysis.licking_and_outcome import preprocess_stimuli_outcomes
    
    all_results = []
    skipped_sessions = 0
    
    # Get unique setups for platform comparison
    setups = project_data["Setup"].unique() if "Setup" in project_data.columns else ["Unknown"]
    
    for idx in project_data.index:
        # Filter by overall session d' if requested
        if filter_successful:
            try:
                session_d = d_prime(project_data, index=project_data.index.get_loc(idx), t=t, plot=False)
                session_d = np.asarray(session_d, dtype=float)
                session_d = session_d[~np.isnan(session_d)]
                mean_session_d = float(np.nanmean(session_d)) if len(session_d) > 0 else 0.0
                if mean_session_d < dprime_threshold:
                    skipped_sessions += 1
                    continue
            except Exception:
                skipped_sessions += 1
                continue
        try:
            # Get session metadata
            setup = project_data.loc[idx, "Setup"] if "Setup" in project_data.columns else "Unknown"
            mouse_name = project_data.loc[idx, "MouseName"] if "MouseName" in project_data.columns else "Unknown"
            n_boundaries = int(project_data.loc[idx].get("N_Boundaries", 1))
            
            # Get stimuli
            try:
                stimuli = project_data.loc[idx, "Stimuli"]
                if isinstance(stimuli, str):
                    stimuli = np.fromstring(stimuli.strip("[]"), sep=" ")
                else:
                    stimuli = np.array(stimuli)
            except Exception:
                stimuli, _ = preprocess_stimuli_outcomes(project_data.reset_index(drop=True), 
                                                         index=project_data.index.get_loc(idx))
            
            if len(stimuli) == 0:
                continue
                
            unique_vals = np.unique(np.round(stimuli.astype(float), 6))
            unique_vals = np.sort(unique_vals)
            
            if len(unique_vals) < 2:
                continue
            
            # Generate stimulus pairs based on boundaries
            stim_pairs = []
            if n_boundaries == 1:
                # Pair stimuli symmetrically from extremes
                half = len(unique_vals) // 2
                for i in range(half):
                    stim_pairs.append((float(unique_vals[i]), float(unique_vals[-(i+1)])))
            else:
                # For 2 boundaries, pair across boundaries
                low_bd = getattr(st.session_state, 'low_boundary', float(np.quantile(unique_vals, 0.33)))
                high_bd = getattr(st.session_state, 'high_boundary', float(np.quantile(unique_vals, 0.66)))
                
                # Create pairs at different distances
                for i, v1 in enumerate(unique_vals):
                    for v2 in unique_vals[i+1:]:
                        stim_pairs.append((float(v1), float(v2)))
            
            # Compute d' for each pair
            trials_val = project_data.loc[idx, "TrialTypes"]
            outcomes_val = project_data.loc[idx, "Outcomes"]
            trialtypes = to_array(trials_val)
            outcomes = to_array(outcomes_val)
            
            for (s1, s2) in stim_pairs:
                # Build mask for the two stimuli
                mask = np.isclose(stimuli, s1, atol=1e-6) | np.isclose(stimuli, s2, atol=1e-6)
                if mask.sum() < 5:  # Need minimum trials
                    continue
                    
                # Filter arrays
                filtered_trials = trialtypes[mask]
                filtered_outcomes = outcomes[mask]
                
                # Prepare a one-row DataFrame for d' function
                df_one = project_data.loc[[idx]].copy()
                df_one.iloc[0, df_one.columns.get_loc('TrialTypes')] = str(np.asarray(filtered_trials).tolist())
                df_one.iloc[0, df_one.columns.get_loc('Outcomes')] = str(np.asarray(filtered_outcomes).tolist())
                
                # Compute d'
                d_vals = d_prime(df_one, index=0, t=t, plot=False)
                d_vals = np.asarray(d_vals, dtype=float)
                d_vals = d_vals[~np.isnan(d_vals)]
                
                if len(d_vals) == 0:
                    continue
                    
                mean_dp = float(np.nanmean(d_vals))
                
                # Calculate octave distance
                lo, hi = (s1, s2) if s1 <= s2 else (s2, s1)
                if lo > 0:
                    octaves = np.log2(hi / lo)
                    # Round to nearest 0.25 for grouping
                    octave_bin = round(octaves * 4) / 4
                else:
                    octave_bin = np.nan
                
                all_results.append({
                    'Setup': setup,
                    'MouseName': mouse_name,
                    'N_Boundaries': n_boundaries,
                    'd_prime': mean_dp,
                    'octave_distance': octave_bin,
                    'stim_low': lo,
                    'stim_high': hi
                })
                
        except Exception as e:
            continue
    
    if not all_results:
        st.warning("No valid pairwise d' data could be computed.")
        return pd.DataFrame()
    
    df_results = pd.DataFrame(all_results)
    
    # Create visualization
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Filter to valid octave distances
    df_plot = df_results[df_results['octave_distance'].notna()].copy()
    
    if len(df_plot) == 0:
        st.warning("No valid octave distances computed.")
        return df_results
    
    # Get unique octave bins and sort
    octave_bins = sorted(df_plot['octave_distance'].unique())
    octave_labels = [f"{o:.2f}" for o in octave_bins]
    
    if compare_platforms and len(setups) > 1:
        # Side-by-side comparison
        fig = make_subplots(rows=1, cols=2, 
                           subplot_titles=[str(s) for s in setups[:2]],
                           horizontal_spacing=0.1)
        
        for col_idx, setup in enumerate(setups[:2], 1):
            df_setup = df_plot[df_plot['Setup'] == setup]
            
            for i, oct_bin in enumerate(octave_bins):
                data = df_setup[df_setup['octave_distance'] == oct_bin]['d_prime']
                if len(data) > 0:
                    fig.add_trace(
                        go.Box(
                            y=data,
                            x=[octave_labels[i]] * len(data),
                            name=f"{oct_bin:.2f} oct",
                            marker_color=colors.COLOR_ACCENT if col_idx == 1 else colors.COLOR_LOW_BD,
                            boxmean=True,
                            showlegend=False
                        ),
                        row=1, col=col_idx
                    )
        
        fig.update_layout(
            title=None,
            height=300,
            margin=dict(l=40, r=10, t=30, b=40),
            boxmode='group',
            boxgap=0.05,
            boxgroupgap=0.05
        )
        fig.update_xaxes(title_text="Octave Distance", row=1, col=1)
        fig.update_xaxes(title_text="Octave Distance", row=1, col=2)
        fig.update_yaxes(title_text="d'", row=1, col=1)
        
    else:
        # Single plot with all data or colored by setup
        fig = go.Figure()
        
        if "Setup" in df_plot.columns and df_plot['Setup'].nunique() > 1:
            # Color by setup
            for setup in df_plot['Setup'].unique():
                df_setup = df_plot[df_plot['Setup'] == setup]
                setup_color = colors.get_setup_color(setup)
                
                for i, oct_bin in enumerate(octave_bins):
                    data = df_setup[df_setup['octave_distance'] == oct_bin]['d_prime']
                    if len(data) > 0:
                        fig.add_trace(go.Box(
                            y=data,
                            x=[octave_labels[i]] * len(data),
                            name=setup,
                            marker_color=setup_color,
                            boxmean=True,
                            legendgroup=setup,
                            showlegend=(i == 0)
                        ))
        else:
            # Single color
            for i, oct_bin in enumerate(octave_bins):
                data = df_plot[df_plot['octave_distance'] == oct_bin]['d_prime']
                if len(data) > 0:
                    fig.add_trace(go.Box(
                        y=data,
                        x=[octave_labels[i]] * len(data),
                        name=f"{oct_bin:.2f} oct",
                        marker_color=colors.COLOR_ACCENT,
                        boxmean=True,
                        showlegend=False
                    ))
        
        fig.update_layout(
            title=None,
            xaxis_title="Octave Distance",
            yaxis_title="d'",
            height=300,
            margin=dict(l=40, r=10, t=10, b=40),
            boxmode='group',
            boxgap=0.05,
            boxgroupgap=0.05
        )
    
    # Add threshold line
    fig.add_hline(y=1.0, line_dash="dash", line_color=colors.COLOR_GRAY, 
                  annotation_text="d'=1", annotation_position="right")
    
    colors.apply_standard_font_sizes(fig)
    st.plotly_chart(fig, use_container_width=True, config=get_plotly_config('pairwise_dprime_by_octave'))
    
    # Show filter info
    if filter_successful and skipped_sessions > 0:
        st.caption(f"Filtered: {skipped_sessions} sessions excluded (d' < {dprime_threshold})")
    
    # Show summary stats
    with st.expander("Summary Statistics", expanded=False):
        summary = df_plot.groupby('octave_distance')['d_prime'].agg(['mean', 'std', 'count']).reset_index()
        summary.columns = ['Octave Distance', 'Mean d\'', 'Std d\'', 'N']
        st.dataframe(summary, use_container_width=True, hide_index=True)
    
    return df_results

