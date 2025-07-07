from Analysis.GNG_bpod_analysis.licking_and_outcome import *
from Analysis.GNG_bpod_analysis.metric import *
from Analysis.GNG_bpod_analysis.psychometric_curves_plotting import *


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

    # ── fit the data to the model───────────────────────────────────────
    if N_Boundaries == 1:
        b, sm, sb, x_fit, y_fit = _single_sigmoid_fit(x, y)

    elif N_Boundaries == 2:
        st.error("Double sigmoid fit is not implemented yet")
        raise NotImplementedError("Double sigmoid fit is not implemented yet")
        # TODO: implement double sigmoid fit

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
        # return boundaries, slopes_mid, slopes_at_bnds, x_fit, y_fit
        b, sm, x_fit, y_fit = psychometric_fitting(unique_stimuli, lick_rates,
                                                   N_Boundaries = n_b,
                                                   log2_x = True)

        st.text(f"x0: {b}, Slope at Boundary: {sm}")

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

        return b, sm, x_fit, y_fit

    except Exception as e:

        return None, None, None




# -------------------------------------------------------------------
# MULTIPLE SESSIONS
# -------------------------------------------------------------------


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
        boundaries, slopes_mid, x_fit, y_fit = psychometric_curve(selected_data, idx, plot = False)

        # accept None, scalar or (low, high) iterable
        if slopes_at_bnds is None:
            low, high = np.nan, np.nan
        else:
            try:  # iterable (two‑boundary session)
                low, high = slopes_at_bnds
            except TypeError:  # scalar (single‑boundary session)
                low, high = slopes_at_bnds, np.nan

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
