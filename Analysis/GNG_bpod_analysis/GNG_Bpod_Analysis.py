from Analysis.GNG_bpod_analysis.psychometric_curves import *
from Analysis.GNG_bpod_analysis.metric import *
from Analysis.GNG_bpod_analysis.GNG_bpod_general import *
from Analysis.GNG_bpod_analysis.licking_and_outcome import *
from Analysis.GNG_bpod_analysis.biases import plot_bias_analysis, bias_multiple_sessions
from Analysis.GNG_bpod_analysis.colors import *

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
        try:
            daily_activity_single_animal(project_data, index)
        except Exception as e:
            st.warning(f"something went wrong with daily activity analysis :|\n\n{e}")
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
        try:
            daily_activity_multi_animal(project_data)
        except Exception as e:
            st.warning(f"Something went wrong with daily activity analysis :|\n\n{e}")
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

