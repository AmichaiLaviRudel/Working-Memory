from Analysis.GNG_bpod_analysis.licking_and_outcome import *
from Analysis.GNG_bpod_analysis.metric import *
import Analysis.GNG_bpod_analysis.colors as colors
import plotly.graph_objects as go
import numpy as np
import streamlit as st
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d



import numpy as np
from scipy.optimize import curve_fit

# -------------------------------------------------------------------
# LOW-LEVEL HELPERS
# -------------------------------------------------------------------
def _σ(x, b, k):
    """Simple logistic with asymptotes 0…1."""
    return 1.0 / (1.0 + np.exp(-k * (x - b)))


def _single_sigmoid_fit(x, y, *, x_boundary=1):
    """
    Classic monotone psychometric.  Returns:
        boundaries      – np.array([x0])
        slopes_mid      – np.array([k/4])
        slopes_at_bnds  – np.array([slope_at_x_boundary])
        x_fit, y_fit    – smooth curve for plotting
    """
    def sig(x, L, x0, k):
        return L / (1. + np.exp(-k * (x - x0)))

    # initial guess: midpoint = median(x), slope = 2
    p0 = [y.max(), np.median(x), 2.0]
    popt, _ = curve_fit(sig, x, y, p0=p0, maxfev=10000)
    L, x0, k = popt

    slope_mid = (L * k) / 4.0
    slope_at_boundary = (L * k * np.exp(-k * (x_boundary - x0))) / (
        (1 + np.exp(-k * (x_boundary - x0))) ** 2
    )

    x_fit = np.linspace(x.min(), x.max(), 200)
    y_fit = sig(x_fit, *popt) / L  # return on 0‥1 scale

    return (
        np.array([x0]),
        np.array([slope_mid]),
        np.array([slope_at_boundary]),
        x_fit,
        y_fit,
    )


def _double_sigmoid_fit(x, y, *, b_fixed=None, lam_fixed=0.0):
    """
 *Double-boundary* model (phenology style):

            y = p₀ − p₁ · [ 1/(1+e^{p₂(x−p₃)}) + 1/(1+e^{−p₄(x−p₅)}) − 1 ]

      where p₃≈left boundary, p₅≈right boundary.
    """

    # Phenology double sigmoid model
    def dbl_sigmoid(t, p0, p1, p2, p3, p4, p5):
        sigma1 = 1.0 / (1.0 + np.exp(p2 * (t - p3)))
        sigma2 = 1.0 / (1.0 + np.exp(-p4 * (t - p5)))
        return p0 - p1 * (sigma1 + sigma2 - 1.0)

    # ----- bounds & guesses -----
    if b_fixed is None:
        q25, q75 = np.percentile(x, [25, 75])
        p0 = [q25, q75, 2.0, 2.0]
        lo, hi = [x.min(), x.min(), 0.1, 0.1], [x.max(), x.max(), 20.0, 20.0]
    else:
        b1, b2 = b_fixed
        p0 = [2.0, 2.0]
        lo, hi = [0.1, 0.1], [20.0, 20.0]

    # ----- fit -----
    # initial guesses
    p0_guess = [100, 100, 5, 1, 5, 1.5]

    popt, _ = curve_fit(dbl_sigmoid, x, y, p0 = p0_guess, maxfev = 20000)
    x_fit = np.linspace(x.min(), x.max(), 300)
    y_fit = dbl_sigmoid(x_fit, *popt)

    p0, p1, p2, p3, p4, p5 = popt

    boundaries = np.array([p3, p5])
    slopes_midpoint = np.array([p2, p4]) / 4.0


    return boundaries, slopes_midpoint, x_fit, y_fit


# -------------------------------------------------------------------
# MAIN FRONT-END FUNCTION
# -------------------------------------------------------------------
def psychometric_fitting(unique_stims,
                     data_points,
                     *,
                     N_Boundaries=1,
                     log2_x=True,
                     b_fixed=None,
                     lapse_fixed=0.0):
    """
    Universal psychometric fitter.

    Parameters
    ----------
    unique_stims, data_points : 1-D arrays
        Stimulus axis and lick probability.
    N_Boundaries : 1 or 2
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
    boundaries, slopes_mid, slopes_at_bnds, x_fit, y_fit
        Arrays sized according to N_Boundaries.
    """

    # ── clean & normalise ─────────────────────────────────────────────
    x = np.asarray(unique_stims, float)
    y = np.asarray(data_points,  float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    if len(x) < 4:
        raise ValueError("Insufficient data for fitting.")

    # Map y to 0…1
    y = (y - y.min()) / (y.max() - y.min())

    # Optional log2 on x
    if log2_x:
        x = np.log2(x)
        b_fixed_log = None if b_fixed is None else tuple(np.log2(b_fixed))
    else:
        b_fixed_log = b_fixed

    # ── choose model ─────────────────────────────────────────────────
    if N_Boundaries == 1:
        b, sm, sb, x_fit, y_fit = _single_sigmoid_fit(x, y)
    elif N_Boundaries == 2:
        b, sm, x_fit, y_fit = _double_sigmoid_fit(
            x, y, b_fixed=b_fixed_log, lam_fixed=lapse_fixed
        )
    else:
        raise ValueError("N_Boundaries must be 1 or 2")

    # transform boundaries & x_fit back if log-scaled earlier
    if log2_x:
        b = np.power(2.0, b)
        x_fit = np.power(2.0, x_fit)

    return b, sm, x_fit, y_fit

def psychometric_curve(selected_data, index, plot=True):
    """
    Processes psychometric data, fits a sigmoid curve, and plots the psychometric curve.
    """
    try:
        # Extract and preprocess data
        stimuli, outcomes = preprocess_stimuli_outcomes(selected_data, index)
        unique_stimuli, lick_rates = compute_lick_rate(stimuli, outcomes)

        n_b = selected_data.loc[index, 'N_Boundaries']
        b, sm, x_fit, y_fit = psychometric_fitting(unique_stimuli, lick_rates,
                                                   N_Boundaries = n_b,
                                                   log2_x = True)

        # st.text(f"x0: {b}, Slope at Boundary: {sm}")

        boudaries = [1, 1.5]

        # Plot the psychometric curve
        if plot:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=unique_stimuli, y=lick_rates/100, mode='markers', name='Data Points'))
            fig.add_trace(go.Scatter(x=x_fit, y=y_fit, mode='lines', name='Fitted Curve'))
            fig.update_layout(title="Psychometric Curve", xaxis_title="Stimulus Intensity", yaxis_title="Lick Rate")
            for bs in boudaries:
                fig.add_trace(go.Scatter(x=[bs, bs], y=[0, 1], mode='lines', name='Boundary', line=dict(dash='dash', color=colors.COLOR_GRAY), fillcolor = colors.COLOR_GRAY))
            st.plotly_chart(fig)

        return None, None, None

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
        N_Boundaries =selected_data.at[idx, "N_Boundaries"]
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



