from Analysis.GNG_bpod_analysis.licking_and_outcome import *
from Analysis.GNG_bpod_analysis.metric import *
import Analysis.GNG_bpod_analysis.colors as colors
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

        n_b = selected_data.loc[index, 'N_Boundaries']

        if n_b == 1:
            # Fit the psychometric curve
            x0, slope_at_midpoint, slope_at_boundary, x_fit, y_fit = psychometric_fitting(unique_stimuli, lick_rates)

            # Display results
            if plot:
                st.subheader("Psychometric Curve with Fit")
                st.text(f"x0: {round(x0, 2)}, Slope at Boundary: {round(slope_at_boundary, 2)}")
                plot_psychometric_curve(unique_stimuli, lick_rates, x_fit, y_fit, x0, slope_at_boundary)
            return [x0, np.nan], [slope_at_midpoint, np.nan], [slope_at_boundary, np.nan]

        if n_b == 2:
            unique_stimuli_low = unique_stimuli[unique_stimuli <= 1.5]
            unique_stimuli_high = unique_stimuli[unique_stimuli > 1]
            lick_rates_low = lick_rates[unique_stimuli <= 1.5]
            lick_rates_high = lick_rates[unique_stimuli > 1]
            # Fit the psychometric curve for low and high stimuli
            x0_low, slope_at_midpoint_low, slope_at_boundary_low, x_fit_low, y_fit_low = psychometric_fitting(unique_stimuli_low, lick_rates_low)
            x0_high, slope_at_midpoint_high, slope_at_boundary_high, x_fit_high, y_fit_high = psychometric_fitting(unique_stimuli_high, lick_rates_high)

            # Display results
            if plot:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Psychometric Curve with Low Fit")
                    st.text(f"x0: {round(x0_low, 5)}, Slope at Boundary: {round(slope_at_boundary_low, 5)}")
                    plot_psychometric_curve(unique_stimuli_low, lick_rates_low, x_fit_low, y_fit_low, x0_low, slope_at_boundary_low)
                with col2:
                    st.subheader("Psychometric Curve with hij Fit")
                    st.text(f"x0: {round(x0_high, 5)}, Slope at Boundary: {round(slope_at_boundary_high, 5)}")
                    plot_psychometric_curve(unique_stimuli_high, lick_rates_high, x_fit_high, y_fit_high, x0_high, slope_at_boundary_high)

            x0s = [x0_low, x0_high]
            slopes_at_midpoint = [slope_at_midpoint_low, slope_at_midpoint_high]
            slopes_at_boundary = [slope_at_boundary_low, slope_at_boundary_high]

            return x0s, slopes_at_midpoint, slopes_at_boundary


    except Exception as e:

        return None, None, None

def psychometric_curve_multiple_sessions(selected_data, animal_name = "None", plot=False):
    """
    Plots the progression of the slope at the boundary across multiple sessions for a selected animal.
    """
    if animal_name == "None":
        # Select animal
        animal_name = st.selectbox("Choose an Animal", selected_data["MouseName"].unique(), key="slope_animal_select")

    session_indices, _ = get_sessions_for_animal(selected_data, animal_name)

    low_slopes, high_slopes, tones, n_bounds = [], [], [], []
    for idx in session_indices:
        _, _, slope_bd = psychometric_curve(selected_data, idx, plot = False)

        # accept None, scalar or (low, high) iterable
        if slope_bd is None:
            low, high = np.nan, np.nan
        else:
            try:  # iterable (two‑boundary session)
                low, high = slope_bd
            except TypeError:  # scalar (single‑boundary session)
                low, high = slope_bd, np.nan

        low_slopes.append(low)
        high_slopes.append(high)
        tones.append(selected_data.at[idx, "Tones_per_class"])
        n_bounds.append(selected_data.at[idx, "N_Boundaries"])

    # ── tidy dataframe for Altair ────────────────────────────────────────────────

    df = (
        pd.DataFrame(
            dict(
                Session = np.arange(1, len(session_indices) + 1),
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
        COLOR_BOUNDARY_SCALE = alt.Scale(
            domain = ["Low", "High"],
            range = [colors.COLOR_LOW_BD, colors.COLOR_HIGH_BD]
        )
        base = (
            alt.Chart(df)
            .encode(
                x = alt.X("Session:Q", title = "Session index"),
                y = alt.Y("Slope:Q", title = "Slope at boundary"),
                color = alt.Color("Boundary:N", scale=COLOR_BOUNDARY_SCALE, title = "Boundary"),
            )
        )

        lines = base.mark_line(point = True)


        anno = (
            base.mark_point(size = 30, stroke = "black")
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

        st.markdown(f"**Slope progression – {animal_name}**")
        st.altair_chart(lines + anno, use_container_width = True)
    # ── numeric return (n_sessions × 2) ─────────────────────────────────────────
    return np.column_stack([low_slopes, high_slopes])

def multi_animal_psychometric_slope_progression(selected_data, N_Boundaries=1):

    df = selected_data

    if N_Boundaries != None:
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

    df["Parsed_Stimuli"] = df["Unique_Stimuli_Values"].apply(parse_stimuli)


    # ─── compute slopes for each subject & session ─────────────────────
    records = []
    for subj in df["MouseName"].unique():
        slopes = psychometric_curve_multiple_sessions(df, animal_name=subj, plot=False)
        # slopes shape: (n_sessions, 2) → [low, high]
        for sess_idx, (low, high) in enumerate(slopes, start=1):
            records.append({"Mouse": subj, "Session": sess_idx, "Boundary": "Low",  "Slope": np.abs(low)})
            records.append({"Mouse": subj, "Session": sess_idx, "Boundary": "High", "Slope": high})

    long_df = pd.DataFrame(records)

    # ─── compute session‐wise average per boundary ────────────────────
    avg_df = (
        long_df
        .groupby(["Session", "Boundary"], as_index=False)["Slope"]
        .mean()
        .assign(Mouse="Average")
    )
    # ─── define a two‐color scale for Low vs High ──────────────────────
    two_color = alt.Scale(
        domain=["Low","High"],
        range=[colors.COLOR_LOW_BD,colors.COLOR_HIGH_BD]
    )

    # ─── individual subjects colored by boundary ───────────────────────
    subj_lines = (
        alt.Chart(long_df)
        .mark_line(opacity=0.3)
        .encode(
            x="Session:Q",
            y="Slope:Q",
            color=alt.Color("Boundary:N", scale=two_color, legend=alt.Legend(title="Boundary")),
            detail="Mouse:N",
            tooltip=["Mouse","Session","Boundary","Slope"],
        )
    )

    # ─── average lines, same colors but bolder ─────────────────────────
    avg_line = (
        alt.Chart(avg_df)
        .mark_line(strokeWidth=colors.LINE_WIDTH_THICK)
        .encode(
            x="Session:Q",
            y="Slope:Q",
            color=alt.Color("Boundary:N", scale=two_color, legend=None),
        )
    )

    # ─── reference at y=0 ──────────────────────────────────────────────
    zero_rule = (
        alt.Chart(pd.DataFrame({"y":[0]}))
        .mark_rule(color="black", strokeDash=[5,5])
        .encode(y="y:Q")
    )

    chart = (subj_lines + avg_line + zero_rule).properties(
        width=700, height=400,
        title=f"Psychometric Slope Progression"
    )
    st.altair_chart(chart, use_container_width=True)


    return long_df, avg_df

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
                mode = 'lines+markers',
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

# ------------------------------------------------------------------
# Double‑logistic psychometric fit (three decision boundaries)
# ------------------------------------------------------------------
def double_psychometric_fitting(unique_stims, data_points, *,
                                log2_x=True, lapse_fixed=0.0):
    """
    Fits a non‑monotonic (rise–fall–rise) psychometric function of the form
        P(Go) = σ(f; b1,k1) · [1 − σ(f; b2,k2)] + σ(f; b3,k3)
    with an optional symmetric lapse rate λ.

    Parameters
    ----------
    unique_stims : array‑like
        1‑D stimulus axis (e.g. tone frequency in kHz).
    data_points : array‑like
        1‑D hit‑rate (or lick‑probability) values, same length as `unique_stims`.
    log2_x : bool, default True
        Apply log2 transform to the x‑axis before fitting (recommended for octaves).
    lapse_fixed : float in [0,0.5), default 0
        Fix the lapse rate to this value.  Pass `None` to fit λ as an extra param.

    Returns
    -------
    boundaries : ndarray shape (3,)
        Fitted boundary locations [b1, b2, b3] (on *log2* scale if `log2_x`).
    slopes_mid  : ndarray shape (3,)
        Maximal slopes of each logistic component (L*k/4, here equal to k/4).
    slopes_at_b1b2 : ndarray shape (2,)
        Slope of the composite curve exactly at b1 and b3 (optional diagnostic).
    x_fit, y_fit : ndarrays
        Smooth fitted curve for plotting (100 points, ascending x).

    Notes
    -----
    • If you later want asymmetric lapses (different Go/No‑Go ceiling/floor),
      extend the model with separate upper and lower asymptotes.
    • All preprocessing (NaN removal, normalisation) mirrors the original function.
    """

    # ----- helpers -----
    def σ(x, b, k):
        # logistic with asymptotes 0..1
        return 1.0 / (1.0 + np.exp(-k * (x - b)))

    def model(x, b1, b2, b3, k1, k2, k3, λ=0.0):
        p = σ(x, b1, k1) * (1.0 - σ(x, b2, k2)) + σ(x, b3, k3)
        return λ * 0.5 + (1.0 - λ) * p

    # ----- clean & normalise -----
    unique_stims = np.asarray(unique_stims, dtype=float)
    data_points  = np.asarray(data_points,  dtype=float)
    mask         = np.isfinite(unique_stims) & np.isfinite(data_points)
    x, y         = unique_stims[mask], data_points[mask]

    if len(x) < 5:
        raise ValueError("Insufficient valid data for double‑logistic fitting.")

    # map y to 0..1 (monotonic rescale like your original code)
    y = (y - y.min()) / (y.max() - y.min())

    # optional log2 transform of x (better for frequency tasks)
    if log2_x:
        x = np.log2(x)

    # ----- initial guesses -----
    # crude: boundaries at 25‑, 50‑, 75‑percentiles of x; slopes = 2
    q25, q50, q75 = np.percentile(x, [25, 50, 75])
    p0 = [q25, q50, q75, 2.0, 2.0, 2.0]           # b1, b2, b3, k1..k3
    bounds_lo = [x.min(), x.min(), x.min(), 0.1, 0.1, 0.1]
    bounds_hi = [x.max(), x.max(), x.max(), 20,  20,  20]

    if lapse_fixed is None:
        p0.append(0.02)             # start λ at 2 %
        bounds_lo.append(0.0)
        bounds_hi.append(0.3)

    # ----- curve fit -----
    try:
        popt, _ = curve_fit(
            lambda _x, *pars: model(_x, *pars) if lapse_fixed is None
            else model(_x, *pars, λ=lapse_fixed),
            x, y, p0=p0, bounds=(bounds_lo, bounds_hi), maxfev=20000
        )
    except RuntimeError as e:
        raise RuntimeError(f"Double‑logistic fitting failed: {e}")

    if lapse_fixed is None:
        *popt_core, λ_hat = popt
    else:
        popt_core, λ_hat = popt, lapse_fixed

    b1, b2, b3, k1, k2, k3 = popt_core

    # ----- derived slopes -----
    slopes_mid = np.array([k1, k2, k3]) / 4.0

    # slope of *composite* curve exactly at b1 and b3 (optional check)
    def composite_slope_at(b, which):
        eps  = 1e-6
        if which == "low":
            return (model(b+eps, b1,b2,b3,k1,k2,k3, λ_hat)
                    - model(b-eps, b1,b2,b3,k1,k2,k3, λ_hat)) / (2*eps)
        else:
            return (model(b+eps, b1,b2,b3,k1,k2,k3, λ_hat)
                    - model(b-eps, b1,b2,b3,k1,k2,k3, λ_hat)) / (2*eps)

    slope_b1 = composite_slope_at(1, "low")
    slope_b3 = composite_slope_at(1.5, "high")

    # ----- smooth curve for plotting -----
    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = model(x_fit, b1, b2, b3, k1, k2, k3, λ_hat)

    return (np.array([k1, k2, k3]),
            slopes_mid,
            np.array([slope_b1, slope_b3]),
            x_fit,
            y_fit)

