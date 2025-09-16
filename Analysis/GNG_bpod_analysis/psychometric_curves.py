from Analysis.GNG_bpod_analysis.licking_and_outcome import preprocess_stimuli_outcomes, compute_lick_rate
from Analysis.GNG_bpod_analysis.metric import *
from Analysis.GNG_bpod_analysis.psychometric_curves_plotting import *
from Analysis.GNG_bpod_analysis.GNG_bpod_general import get_sessions_for_animal
import Analysis.GNG_bpod_analysis.colors as colors
import plotly.graph_objects as go
import numpy as np
import streamlit as st
import pandas as pd
import altair as alt
from scipy.optimize import curve_fit
import hashlib

# -------------------------------------------------------------------
# LOW-LEVEL HELPERS
# -------------------------------------------------------------------
def single_sigmoid_fit(x, y, *, x_boundary: float = 1.0):
    """
    Classic monotone psychometric.  Returns:
        model_boundaries      – np.array([x0])
        slopes_mid      – np.array([k/4])
        slopes_at_model_boundaries  – np.array([slope_at_x_boundary])
        x_fit, y_fit    – smooth curve for plotting
    x_boundary: float
        The x value at which to compute the slope (can be float for log2_x cases).
    """
    def sig(x, L, x0, k):
        return L / (1. + np.exp(-k * (x - x0)))
    # Improved initial guess for slope
    slope_guess = 1.0 if x[-1] > x[0] else -1.0
    p0 = [y.max(), np.median(x), slope_guess]
    try:
        popt, _ = curve_fit(sig, x, y, p0=p0, maxfev=20000)
    except (RuntimeError, ValueError) as e:
        # Handle both RuntimeError and ValueError (including insufficient data points)
        return np.nan, np.nan, np.nan, np.nan, np.nan
    L, x0, k = popt
    slope_mid = (L * k) / 4.0
    slope_at_boundary = (L * k * np.exp(-k * (x_boundary - x0))) / ((1 + np.exp(-k * (x_boundary - x0))) ** 2)
    x_fit = np.linspace(x.min(), x.max(), 200)
    y_fit = sig(x_fit, *popt) / L if L != 0 else np.zeros_like(x_fit)
    return (
        np.array([x0]),
        np.array([slope_mid]),
        np.array([slope_at_boundary]),
        x_fit,
        y_fit,
    )

# -------------------------------------------------------------------
# MAIN FRONT-END FUNCTION
# -------------------------------------------------------------------
@st.cache_data(show_spinner="Fitting psychometric curve...")
def psychometric_fitting_cached(unique_stims_tuple, data_points_tuple, N_Boundaries, log2_x, b_fixed, lapse_fixed, data_hash):
    """Cached version of psychometric fitting."""
    unique_stims = np.array(unique_stims_tuple)
    data_points = np.array(data_points_tuple)
    
    return _psychometric_fitting_core(unique_stims, data_points, N_Boundaries=N_Boundaries, 
                                    log2_x=log2_x, b_fixed=b_fixed, lapse_fixed=lapse_fixed)

def _psychometric_fitting_core(unique_stims, data_points, *, N_Boundaries=1, log2_x=True, b_fixed=None, lapse_fixed=0.0):
    """Core psychometric fitting logic."""
    # ── clean & normalise ─────────────────────────────────────────────
    x = np.asarray(unique_stims, float)
    y = np.asarray(data_points,  float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        raise ValueError(f"Insufficient data for fitting: need at least 3 data points, got {len(x)}")
    y_min, y_max = np.min(data_points), np.max(data_points)
    # Map y to 0…1
    if y.max() > y.min():
        y = (y - y.min()) / (y.max() - y.min())
    # Optional log2 on x
    if log2_x:
        x = np.log2(x)
        b_fixed_log = None if b_fixed is None else tuple(np.log2(b_fixed))
    else:
        b_fixed_log = b_fixed
    
    # ── fit the data to the model───────────────────────────────────────
    if N_Boundaries == 1:
        model_boundaries, slopes_mid, slopes_at_model_boundaries, x_fit, y_fit = single_sigmoid_fit(x, y, x_boundary=x.mean())
        # Rescale y_fit to original y scale
        if y_max > y_min:
            y_fit = y_fit * (y_max - y_min) + y_min
        return model_boundaries, slopes_mid, slopes_at_model_boundaries, x_fit, y_fit
    elif N_Boundaries == 2:
        # Two-boundary fitting logic would go here
        # For now, fallback to single sigmoid
        model_boundaries, slopes_mid, slopes_at_model_boundaries, x_fit, y_fit = single_sigmoid_fit(x, y, x_boundary=x.mean())
        if y_max > y_min:
            y_fit = y_fit * (y_max - y_min) + y_min
        return model_boundaries, slopes_mid, slopes_at_model_boundaries, x_fit, y_fit
    else:
        raise ValueError("N_Boundaries must be 1 or 2")

def psychometric_fitting(unique_stims,
                     data_points,
                     *,
                     N_Boundaries=1,
                     log2_x=True,
                     b_fixed=None,
                     lapse_fixed=0.0):
    """
    Universal psychometric fitter with caching.

    Parameters
    ----------
    unique_stims, data_points: 1-D arrays
        Stimulus axis and lick probability.
    N_Boundaries: 1 or 2
        • 1 → monotone single sigmoid
        • 2 → rise–fall model (two boundaries)
    log2_x : bool
        Log-transform x before fitting (good for frequencies).
    b_fixed : None | (tuple)
        If N_Boundaries==2 you can pin the two boundaries, e.g. (1,1.5).
    lapse_fixed : float
        Symmetric lapse (only in two-boundary model).

    Returns
    -------
    model_boundaries, slopes_mid, slopes_at_model_boundaries, x_fit, y_fit
        Arrays sized according to N_Boundaries.
    """
    # Create hash for caching
    data_hash = hashlib.md5(f"{unique_stims.tobytes()}_{data_points.tobytes()}_{N_Boundaries}_{log2_x}_{b_fixed}_{lapse_fixed}".encode()).hexdigest()
    
    # Use cached version
    return psychometric_fitting_cached(
        tuple(unique_stims), tuple(data_points), N_Boundaries, log2_x, b_fixed, lapse_fixed, data_hash
    )


def psychometric_curve(selected_data, index, plot=True):
    """
    Processes psychometric data, fits a sigmoid curve, and plots the psychometric curve.
    """
    try:
        # Extract and preprocess data
        stimuli, outcomes = preprocess_stimuli_outcomes(selected_data, index)
        unique_stimuli, lick_rates = compute_lick_rate(stimuli, outcomes)
        n_b = selected_data.loc[index, 'N_Boundaries']
        session_type =  selected_data.at[index, "Notes"]
        if "TA" in session_type or "Discrimination" in session_type:
            st.info(f"this is {session_type} session")
            return None, None, None, None, None
        # Check if we have enough data points for fitting
        if len(unique_stimuli) < 3:
            return None, None, None, None, None
            
        # return boundaries, slopes_mid, slopes_at_bnds, x_fit, y_fit
        model_boundaries, slopes_mid, slopes_at_model_boundaries, x_fit, y_fit = psychometric_fitting(unique_stimuli, lick_rates,
                                                   N_Boundaries = n_b,
                                                   log2_x = False)
        # Plot the psychometric curve
        if plot:
            if n_b == 2:
                st.text(f"Double sigmoid fit:\n"
                         f"x0_low: {model_boundaries[0]:.4g}, Slope at Boundary_low: {slopes_mid[0]:.4g}\n"
                         f"x0_high: {model_boundaries[1]:.4g}, Slope at Boundary_high: {slopes_mid[1]:.4g}")
                fig = go.Figure()
                # Data points
                fig.add_trace(go.Scatter(x=unique_stimuli, y=lick_rates, mode='markers', name='Data Points', marker=dict(color='black')))
                # Overlay both fitted sigmoids
                x_fit_A, x_fit_B = x_fit
                y_fit_A, y_fit_B = y_fit
                fig.add_trace(go.Scatter(x=x_fit_A, y=y_fit_A, mode='lines', name='Sigmoid Low', line=dict(color=colors.COLOR_LOW_BD)))
                fig.add_trace(go.Scatter(x=x_fit_B, y=y_fit_B, mode='lines', name='Sigmoid High', line=dict(color=colors.COLOR_HIGH_BD)))
                # Boundaries
                fig.add_trace(go.Scatter(x=[st.session_state.low_boundary, st.session_state.low_boundary], y=[0, 100], mode='lines', name='Low Boundary', line=dict(dash='dash', color=colors.COLOR_GRAY)))
                fig.add_trace(go.Scatter(x=[st.session_state.high_boundary, st.session_state.high_boundary], y=[0, 100], mode='lines', name='High Boundary', line=dict(dash='dash', color=colors.COLOR_GRAY)))
                fig.update_layout(title='Psychometric Curve (Double Sigmoid)', xaxis_title='Stimulus Intensity (log2)', xaxis_type='log', yaxis_title='Lick Rate (normalized)', yaxis_range=[0, 100])
                st.plotly_chart(fig)
            else:
                st.text(f"Single sigmoid fit:\nx0: {model_boundaries}, Slope at Boundary: {slopes_mid}")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=unique_stimuli, y=lick_rates, mode='markers', name='Data Points'))
                fig.add_trace(go.Scatter(x=x_fit, y=y_fit, mode='lines', name='Fitted Curve'))
                # Only plot the relevant boundary line
                boundary = st.session_state.low_boundary
                fig.add_trace(go.Scatter(x=[boundary, boundary], y=[0, 100], mode='lines', name='Boundary', line=dict(dash='dash', color=colors.COLOR_GRAY)))
                # Only use log axis if all x > 0
                x_fit_arr = np.array(x_fit) if isinstance(x_fit, (list, tuple)) else x_fit
                if np.all(unique_stimuli > 0) and np.all(x_fit_arr > 0):
                    fig.update_layout(title="Psychometric Curve", xaxis_title="Stimulus Intensity (log2)", yaxis_title="Lick Rate", xaxis_type='log', yaxis_range=[0, 100])
                else:
                    fig.update_layout(title="Psychometric Curve", xaxis_title="Stimulus Intensity (log2)", yaxis_title="Lick Rate", yaxis_range=[0, 100])
                st.plotly_chart(fig)
        return model_boundaries, slopes_mid, slopes_at_model_boundaries, x_fit, y_fit
    except ValueError as e:
        if "Insufficient data" in str(e):
            st.warning(f"Cannot fit psychometric curve: {e}")
        else:
            st.error(f"Data validation error in psychometric_curve: {e}")
        return None, None, None, None, None
    except NotImplementedError as e:
        st.error(str(e))
        return None, None, None, None, None
    except Exception as e:
        st.error(f"Error in psychometric_curve: {e}")
        return None, None, None, None, None

# -------------------------------------------------------------------
# MULTIPLE SESSIONS
# -------------------------------------------------------------------
def psychometric_curve_multiple_sessions(selected_data, animal_name = "None", plot=False):
    """
    Plots the progression of the slope at the boundary across multiple sessions for a selected animal.
    """
    if animal_name == "None":
        animal_name = st.selectbox("Choose an Animal", selected_data["MouseName"].unique(), key="slope_animal_select")
    session_indices, _ = get_sessions_for_animal(selected_data, animal_name)
    low_slopes, high_slopes, tones, n_bounds = [], [], [], []
    valid_session_indices = []  # Track which sessions are actually processed
    
    for idx in session_indices:
        session_type = selected_data.at[idx, "Notes"]
        if "TA" in session_type or "Discrimination" in session_type:
            continue
            
        valid_session_indices.append(idx)  # Only add sessions that pass the filter
        N_Boundaries = selected_data.at[idx, "N_Boundaries"]
        boundaries, slopes_mid, slopes_at_bnds, x_fit, y_fit = psychometric_curve(selected_data, idx, plot = False)
        # Robustly ensure slopes_mid is always a numpy array of length 2 for safe indexing
        if slopes_mid is None:
            slopes_mid = np.array([np.nan, np.nan])
        elif isinstance(slopes_mid, float):
            slopes_mid = np.array([slopes_mid, np.nan])
        elif isinstance(slopes_mid, (list, np.ndarray)):
            slopes_mid = np.array(slopes_mid, dtype=float)
            if slopes_mid.size == 1:
                slopes_mid = np.array([slopes_mid[0], np.nan])
            elif slopes_mid.size == 0:
                slopes_mid = np.array([np.nan, np.nan])
            elif slopes_mid.size > 2:
                slopes_mid = np.concatenate([slopes_mid[:2], np.full(slopes_mid.size-2, np.nan)])[:2]
        else:
            slopes_mid = np.array([np.nan, np.nan])
        low, high = slopes_mid[0], slopes_mid[1]

        low_slopes.append(low)
        high_slopes.append(high)
        tones.append(selected_data.at[idx, "Tones_per_class"])
        n_bounds.append(selected_data.at[idx, "N_Boundaries"])

    # Check if we have any valid sessions
    if len(valid_session_indices) == 0:
        st.warning(f"No valid sessions found for {animal_name} (all sessions are TA or Discrimination)")
        return np.array([])
    
    # ── tidy dataframe for Altair ────────────────────────────────────────────────
    df = (
        pd.DataFrame(
            dict(
                Session = np.arange(1, len(valid_session_indices) + 1),
                Low = np.abs(low_slopes),
                High = high_slopes,
                tones_per_class = tones,
                Boundaries = n_bounds,
            )
        )
        .melt(
            id_vars = ["Session", "tones_per_class", "Boundaries"],
            value_vars = ["Low", "High"],
            var_name = "Boundary",
            value_name = "Slope",
        )
     )
    # ── plotting ────────────────────────────────────────────────────────────────
    if plot:
        # Blue for single boundary, otherwise use the two-color scale
        if df['Boundary'].nunique() == 1:
            COLOR_BOUNDARY_SCALE = alt.Scale(domain=["Low"], range=["#1E90FA"])
        else:
            COLOR_BOUNDARY_SCALE = alt.Scale(
                domain = ["Low", "High"],
                range = [colors.COLOR_LOW_BD, colors.COLOR_HIGH_BD]
            )
        base = (
            alt.Chart(df)
            .encode(
                x = alt.X("Session:Q", title = "Session index"),
                y = alt.Y("Slope:Q", scale=alt.Scale(type='log'), title = "Slope at boundary"),
                color = alt.Color("Boundary:N", scale=COLOR_BOUNDARY_SCALE, title = "Boundary"),
            )
        )

        lines = base.mark_line(point = True)

        anno = (
            base.mark_point(size = 60, stroke = "black")
            .encode(
                shape = alt.Shape(
                    "Boundaries:N",
                    scale = alt.Scale(domain = [1, 2], range = ["circle", "circle"]),
                ),
                fill = alt.Fill(
                    "Boundaries:N",
                    scale = alt.Scale(domain = [1, 2], range = ["white", "black"]),
                ),
                tooltip = [
                    alt.Tooltip("Session", title = "Session"),
                    alt.Tooltip("tones_per_class", title = "Tones / class"),
                    alt.Tooltip("Boundaries", title = "# boundaries"),
                    alt.Tooltip("Slope", title = "Slope"),
                ],
            )
        )

        st.markdown(f"**Slope progression – {animal_name}**")
        st.altair_chart(lines + anno, use_container_width = True)
    # ── numeric return (n_sessions × 2) ─────────────────────────────────────────
    return np.column_stack([low_slopes, high_slopes])

# -------------------------------------------------------------------
# MULTIPLE ANIMALS
# -------------------------------------------------------------------
def multi_animal_psychometric_slope_progression(selected_data, N_Boundaries=1):
    df = selected_data.copy()  
    if N_Boundaries is not None:
        df = (
            selected_data
            .loc[selected_data["N_Boundaries"] == N_Boundaries]
            .reset_index(drop=True)
        )
    # ─── parse stimuli strings to arrays ───────────────────────────────
    def parse_stimuli(s):
        try:
            return np.fromstring(s.strip("[]"), sep=" ")
        except:
            return np.array([])
    
    # Handle both possible column names for stimuli data
    if "Unique_Stimuli_Values" in df.columns:
        df["Parsed_Stimuli"] = df["Unique_Stimuli_Values"].apply(parse_stimuli)
    elif "Stimuli" in df.columns:
        df["Parsed_Stimuli"] = df["Stimuli"].apply(parse_stimuli)
    else:
        st.error("Neither 'Unique_Stimuli_Values' nor 'Stimuli' column found in data")
        return


    # ─── compute slopes for each subject & session ─────────────────────
    records = []
    for subj in df["MouseName"].unique():
        slopes = psychometric_curve_multiple_sessions(df, animal_name=subj, plot=False)
        # Handle case where no valid sessions are found
        if slopes.size == 0:
            continue
        for sess_idx, (low, high) in enumerate(slopes, start=1):
            records.append({"Mouse": subj, "Session": sess_idx, "Boundary": "Low",  "Slope": np.abs(low)})
            records.append({"Mouse": subj, "Session": sess_idx, "Boundary": "High", "Slope": high})
    
    # Check if we have any records
    if not records:
        st.warning("No valid sessions found for any animal")
        return
        
    long_df = pd.DataFrame(records)

    # ─── compute session‐wise average per boundary ────────────────────
    avg_df = (
        long_df
        .groupby(["Session", "Boundary"], as_index=False)["Slope"]
        .mean()
        .assign(Mouse="Average")
    )

    import plotly.graph_objects as go
    fig = go.Figure()
    # Plot all animals (gray lines, log y)
    for subj in df["MouseName"].unique():
        for boundary, color in zip(["Low", "High"], [colors.COLOR_LOW_BD, colors.COLOR_HIGH_BD]):
            subj_data = long_df[(long_df["Mouse"] == subj) & (long_df["Boundary"] == boundary)]
            if not subj_data.empty:
                fig.add_trace(go.Scatter(
                    x=subj_data["Session"],
                    y=subj_data["Slope"],
                    mode='lines',
                    line=dict(color=colors.COLOR_GRAY, width=2, dash='solid'),
                    name=f'{subj} {boundary}',
                    showlegend=False,
                    opacity=0.3
                ))
    # Plot average lines (bold)
    for boundary, color in zip(["Low", "High"], [colors.COLOR_LOW_BD, colors.COLOR_HIGH_BD]):
        avg_data = avg_df[avg_df["Boundary"] == boundary]
        if not avg_data.empty:
            fig.add_trace(go.Scatter(
                x=avg_data["Session"],
                y=avg_data["Slope"],
                mode='lines',
                line=dict(color=color, width=4),
                name=f'Average {boundary}',
                marker=dict(symbol='circle')
            ))
    # Reference at y=1 (log scale)
    fig.add_trace(go.Scatter(
        x=[long_df["Session"].min(), long_df["Session"].max()],
        y=[1, 1],
        mode='lines',
        name="Learning Threshold",
        line=dict(color=colors.COLOR_GRAY, dash='dash'),
        showlegend=True
    ))
    fig.update_layout(
        xaxis_title="Session Index",
        yaxis_title="log(| Slope at x0 |)",
        yaxis_type='log',
        title="Psychometric Slope Progression - all animals",
        legend=dict(title="Legend"),
        height=400,
        width=700
    )
    fig.update_yaxes(autorange=True)
    st.plotly_chart(fig, use_container_width=True)

    return long_df, avg_df


# TODO:
# - Add a function that measure the distance between X0 and the true boundary