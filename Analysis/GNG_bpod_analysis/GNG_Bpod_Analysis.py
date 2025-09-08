from Analysis.GNG_bpod_analysis.psychometric_curves import *
from Analysis.GNG_bpod_analysis.metric import *
from Analysis.GNG_bpod_analysis.GNG_bpod_general import *
from Analysis.GNG_bpod_analysis.licking_and_outcome import *
from Analysis.GNG_bpod_analysis.biases import plot_bias_analysis, bias_multiple_sessions

import traceback
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np


def gng_bpod_analysis(project_data, index):
    name, session = getNameAndSession(project_data, index)
    st.header(f"{name}  ___#{session}___")
    bin = st.slider("Choose bin size", 5, 50, 15, 5)


    tab1, tab2, tab3, tab4, tab5 = st.tabs([ "👨‍🎓Matrices", "👅 Lick Rate", "📈 Learning Curve", "👂 Psychometric Curve", "🎯 Bias Analysis"])

    with tab1:
        try:
            classifier_metric(project_data, index)
            d_prime(project_data, index, t=bin, plot=True)
            d_prime_low_high_boundary_sessions(project_data, index, t=bin, plot=True)
        except Exception as e:
            st.warning(f"something went wrong with this graph :|\n\n{e}")
            st.text(traceback.format_exc())


    with tab2:
        try:
            licking_rate(project_data, index, t=bin)
        except  Exception as e:
            st.warning(f"something went wrong with this graph :|\n\n{e}")
            st.text(traceback.format_exc())

        try:
            df_go_first_licks, df_no_go_first_licks = process_and_plot_lick_data(project_data, index)
        except  Exception as e:
            st.warning(f"something went wrong with this graph :|\n\n{e}")
            st.text(traceback.format_exc())
            df_first_licks = None
            
        try:
            st.subheader("First Lick Latency Analysis")
            plot_first_lick_latency(project_data, index, df_go_first_licks, df_no_go_first_licks)
        except Exception as e:
            st.warning(f"something went wrong with latency analysis :|\n\n{e}")
            st.text(traceback.format_exc())

    with tab3:
        try:
            learning_curve(project_data, index)
        except  Exception as e:
            st.warning(f"something went wrong with this graph :|\n\n{e}")
            st.text(traceback.format_exc())

    with tab4:
        try:
            psychometric_curve(project_data, index)
        except  Exception as e:
            st.warning(f"something went wrong with this graph :|\n\n{e}")
            st.text(traceback.format_exc())

    with tab5:
        try:
            n_previous_trials = st.slider("Number of previous trials to consider", 1, 10, 3, 1)
            plot_bias_analysis(project_data, index, n_previous_trials=n_previous_trials, plot=True)
        except Exception as e:
            st.warning(f"something went wrong with bias analysis :|\n\n{e}")
            st.text(traceback.format_exc())


def gng_bpod_analysis_multipule(project_data, index):
    bin = st.slider("Choose bin size", 5, 50, 30, 5)
    animal_name = st.selectbox("Choose an Animal",
        sorted(project_data["MouseName"].unique()),  # Convert to list and sort
        key = "animal_select")
    tab1, tab2, tab3, tab4 = st.tabs(["👅 Lick Rate", "👨‍🎓 D Prime", "👂 Psychometric Slope", "🎯 Bias Analysis"])


    with tab1:
        lick_rate_multipule_sessions(project_data, t=bin, plot=True, animal_name = animal_name)
        
        try:
            st.subheader("First Lick Latency Progression")
            plot_first_lick_latency_multiple_sessions(project_data, animal_name=animal_name, plot=True)
        except Exception as e:
            st.warning(f"Something went wrong with first lick latency analysis :|\n\n{e}")
            st.text(traceback.format_exc())

        try:
            st.subheader("Daily Multi-Animal Lick Rate Comparison")
            daily_multi_animal_lick_rate(project_data, t=bin)
        except Exception as e:
            st.warning(f"Something went wrong with daily multi-animal lick rate :|\n\n{e}")
            st.text(traceback.format_exc())

    with tab2:
        d_prime_multiple_sessions(project_data, t=bin, animal_name = animal_name)
        # d_prime_multiple_sessions_divde_oneNtwo(project_data, t = 10, animal_name = 'None', plot = True)

        multi_animal_d_prime_progression(project_data)
        # multi_animal_d_prime_progression(project_data, N_Boundaries = 2)
        try:
            st.subheader("Daily Multi-Animal d' Comparison")
            daily_multi_animal_dprime(project_data, t=bin)
        except Exception as e:
            st.warning(f"Something went wrong with daily multi-animal d' :|\n\n{e}")
            st.text(traceback.format_exc())


    with tab3:
        try:
            psychometric_curve_multiple_sessions(project_data, animal_name = animal_name, plot=True)
        except Exception as e:
            st.warning(f"Something went wrong with psychometric curve analysis :|\n\n{e}")
            st.text(traceback.format_exc())
            
        try:
            multi_animal_psychometric_slope_progression(project_data,  N_Boundaries = None)
        except Exception as e:
            st.warning(f"Something went wrong with multi-animal psychometric analysis :|\n\n{e}")
            st.text(traceback.format_exc())
            
        # multi_animal_psychometric_slope_progression(project_data,  N_Boundaries = 2)
        try:
            plot_psychometric_curves_with_boundaries(project_data, N_Boundaries = 1, n_indices = 2)
            plot_psychometric_curves_with_boundaries(project_data,  N_Boundaries = 2, n_indices = 2)
            plot_psychometric_curves_with_boundaries(project_data,  N_Boundaries = 0, n_indices = 2)
        except Exception as e:
            st.warning(f"Something went wrong with psychometric curves plotting :|\n\n{e}")
            st.text(traceback.format_exc())


    with tab4:
        try:
            n_previous_trials = st.slider("Number of previous trials to consider", 1, 10, 3, 1, key="bias_prev_trials")
            bias_multiple_sessions(project_data, animal_name=animal_name, n_previous_trials=n_previous_trials)
        except Exception as e:
            st.warning(f"Something went wrong with bias analysis :|\n\n{e}")
            st.text(traceback.format_exc())

def object_to_array(obj_array, pad_value=np.nan):
    """
    Convert a 1D object array of 1D arrays/lists into
    a 2D numeric array with NaN padding.
    """
    # lengths of each sub-array
    lengths = [len(x) for x in obj_array]
    max_len = max(lengths)

    out = np.full((len(obj_array), max_len), pad_value, dtype=float)
    for i, arr in enumerate(obj_array):
        arr = np.asarray(arr, dtype=float)
        out[i, :len(arr)] = arr
    return out

def daily_multi_animal_lick_rate(project_data, t=15):
    """
    Plot lick rate data for all unique mice on a selected date, overlaid on the same plot.
    Uses the same logic as the licking_rate function but for multiple animals.
    """
    if project_data is None or project_data.empty:
        st.info("No data loaded.")
        return

    # Get unique dates
    dates = sorted(project_data["SessionDate"].astype(str).unique())
    if len(dates) == 0:
        st.info("No dates found in data.")
        return

    selected_date = st.selectbox("Select a date", options=dates, 
                                index=max(0, len(dates) - 1), 
                                key="daily_multi_lick_date")

    # Filter data for selected date
    date_data = project_data[project_data["SessionDate"].astype(str) == str(selected_date)]
    
    if date_data.empty:
        st.info(f"No data found for date {selected_date}")
        return
    # Get unique mice for this date
    mice = sorted(date_data["MouseName"].unique())
    if len(mice) == 0:
        st.info("No animals found for selected date.")
        return
    
    fig = go.Figure()

    for mouse in mice:
        mouse_data = date_data[date_data["MouseName"] == mouse]
        if len(mouse_data) == 0:
            continue
        # Compute Go hit-rate series using existing function and selected bin size t
        _, frac = licking_rate(mouse_data, index=0, t=t, plot=False)
        go_series = frac["Go"].dropna()
        if len(go_series) == 0:
            continue
        x = np.arange(1, len(go_series) + 1)
        fig.add_trace(go.Scatter(
            x=x,
            y=go_series.values,
            mode='lines',
            name=str(mouse),
            line=dict(width=2)
        ))

    if len(fig.data) == 0:
        st.info("No lick data found for any animals on selected date.")
        return

    fig.update_layout(
        title=f"Go Hit Rate by Animal — {selected_date} (rolling window={t})",
        xaxis_title="Trial index",
        yaxis_title="Hit rate (%)",
        yaxis=dict(range=[0, 100]),
        height=500,
        width=900,
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)


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
            line=dict(width=2)
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
    st.plotly_chart(fig, use_container_width=True)
