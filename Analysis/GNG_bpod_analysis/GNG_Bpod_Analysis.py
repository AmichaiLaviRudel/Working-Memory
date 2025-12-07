from Analysis.GNG_bpod_analysis.psychometric_curves import *
from Analysis.GNG_bpod_analysis.psychometric_curves_plotting import remove_outlier_sessions, plot_psychometric_curves_with_boundaries
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
    # Performance info
    with st.expander("ℹ️ Performance Info"):
        st.info("🚀 Analysis results are cached for faster performance. Change parameters to trigger recomputation.")
        st.caption(f"📊 Session: {name} #{session}")
    bin = st.slider("Choose bin size", 5, 50, 30, 5, help="⚡ Cached - only recomputes when changed")


    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([ "👨‍🎓Matrices", "👅 Lick Rate", "📈 Learning Curve", "👂 Psychometric Curve", "🎯 Bias Analysis", "🧠 GLM Analysis"])

    with tab1:
        try:
            classifier_metric(project_data, index)
            d_prime(project_data, index, t=bin, plot=True)
            d_prime_for_stim_pairs(project_data, index, stim_pairs=None, t=bin, plot=True)
        except Exception as e:
            st.warning(f"something went wrong with this graph :|\n\n{e}")
            st.text(traceback.format_exc())

    with tab2:
        try:
            licking_rate(project_data, index, t=bin)
        except  Exception as e:
            st.warning(f"something went wrong with this graph :|\n\n{e}")
            st.text(traceback.format_exc())
        st.write(st.session_state.analysis_type)

        try:
            df_go_first_licks, df_no_go_first_licks, lick_by_stimulus = process_and_plot_lick_data(
                project_data,
                index,
                plot=True,
            )
            if st.session_state.analysis_type == 'Behavior-Bpod GUI':
                plot_first_lick_by_stimulus(
                    project_data,
                    index,
                    plot=True,
                )
                try:
                    st.subheader("First Lick Latency Analysis")
                    plot_first_lick_latency(
                        project_data,
                        index,
                        df_go_first_licks,
                        df_no_go_first_licks,
                        plot=True,
                    )
                except Exception as e:
                    st.warning(f"something went wrong with latency analysis :|\n\n{e}")
                    st.text(traceback.format_exc())
            elif st.session_state.analysis_type == 'Educage':
                plot_n_lick_by_stimulus(
                    project_data,
                    index,
                    plot=True,
                )

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
            n_previous_trials = st.slider("Number of previous trials to consider", 1, 10, 3, 1, help="⚡ Bias computation cached")
            plot_bias_analysis(project_data, index, n_previous_trials=n_previous_trials, plot=True)
        except Exception as e:
            st.warning(f"something went wrong with bias analysis :|\n\n{e}")
            st.text(traceback.format_exc())

    with tab6:
        try:
            from Analysis.GNG_bpod_analysis.licking_and_outcome import glm_licking_analysis
            glm_licking_analysis(project_data, index, plot=True)
        except Exception as e:
            st.warning(f"something went wrong with GLM analysis :|\n\n{e}")
            st.text(traceback.format_exc())


def gng_bpod_analysis_multi_session(project_data, index):
    """
    Multi-session analysis for a single animal.
    Analyzes one animal's performance across multiple sessions.
    """
    # Performance info
    with st.expander("ℹ️ Performance Info"):
        st.info("🚀 Multi-session analysis uses caching for faster performance.")
        st.caption(f"📊 Dataset: {len(project_data)} sessions across {len(project_data['MouseName'].unique())} animals")
        # Add cache management
        if st.button("🗑️ Clear GNG Analysis Cache", key="clear_cache_single"):
            st.cache_data.clear()
            st.toast("GNG analysis cache cleared - next computation will be fresh")

    bin = st.slider("Choose bin size", 5, 50, 30, 5, help="⚡ Cached computation", key="bin_single")
    animal_name = st.selectbox("Choose an Animal",
        sorted(project_data["MouseName"].unique()),  # Convert to list and sort
        key = "animal_select_single", help="⚡ Results cached per animal")
    
    st.header(f"🐭 Multi-Session Analysis: {animal_name}")
    
    tab1, tab2, tab3, tab4 = st.tabs(["👅 Lick Rate", "👨‍🎓 D Prime", "👂 Psychometric Slope", "🎯 Bias Analysis"])

    with tab1:
        try:
            st.subheader(f"Lick Rate Progression - {animal_name}")
            lick_rate_multipule_sessions(project_data, t=bin, plot=True, animal_name = animal_name)
        except Exception as e:
            st.warning(f"Something went wrong with lick rate analysis :|\n\n{e}")
            st.text(traceback.format_exc())
        
        try:
            st.subheader("First Lick Latency Progression")
            plot_first_lick_latency_multiple_sessions(project_data, animal_name=animal_name, plot=True)
        except Exception as e:
            st.warning(f"Something went wrong with first lick latency analysis :|\n\n{e}")
            st.text(traceback.format_exc())

    with tab2:
        try:
            st.subheader(f"Daily Multi-Session d' Comparison - {animal_name}")
            d_prime_multiple_sessions(project_data, t=bin, animal_name = animal_name, plot = True)
        except Exception as e:
            st.warning(f"Something went wrong with daily multi-session d' :|\n\n{e}")
            st.text(traceback.format_exc())

    with tab3:
        # Session filtering options
        with st.expander("🔍 Filter Outlier Sessions", expanded=False):
            filter_outliers = st.checkbox("Remove sessions with low d-prime", value=False, 
                                         help="Filter out sessions where mean d-prime is below threshold",
                                         key="filter_outliers_single")
            d_prime_threshold = 1.0
            if filter_outliers:
                d_prime_threshold = st.slider("d-prime threshold", 0.0, 3.0, 1.0, 0.1, 
                                             help="Sessions with d' below this value will be removed",
                                             key="d_prime_threshold_single")
        
        # Filter project_data for the selected animal
        animal_data = project_data[project_data['MouseName'] == animal_name].copy()
        
        n_indices = st.slider("Number of indices to include", 1, 10, 2, 1, key="n_indices_single")
        
        try:
            st.subheader(f"Psychometric Curves - {animal_name} (1 Boundary)")
            # Show Early Response filter checkbox only once; subsequent calls reuse its value.
            plot_psychometric_curves_with_boundaries(animal_data, N_Boundaries = 1, n_indices = n_indices, 
                                                    filter_outliers=filter_outliers, 
                                                    d_prime_threshold=d_prime_threshold, 
                                                    t=bin)
        except Exception as e:
            st.warning(f"Something went wrong with psychometric curve analysis :|\n\n{e}")
            st.text(traceback.format_exc())

        try:
            st.subheader(f"Psychometric Curves - {animal_name} (2 Boundaries)")
            plot_psychometric_curves_with_boundaries(animal_data, N_Boundaries = 2, n_indices = n_indices,
                                                    filter_outliers=filter_outliers, 
                                                    d_prime_threshold=d_prime_threshold, 
                                                    t=bin)
        except Exception as e:
            st.warning(f"Something went wrong with psychometric curve analysis :|\n\n{e}")
            st.text(traceback.format_exc())
        
        try:
            st.subheader(f"Psychometric Curves - {animal_name} (0 Boundaries)")
            plot_psychometric_curves_with_boundaries(animal_data, N_Boundaries = 0, n_indices = n_indices,
                                                    filter_outliers=filter_outliers, 
                                                    d_prime_threshold=d_prime_threshold, 
                                                    t=bin)
        except Exception as e:
            st.warning(f"Something went wrong with psychometric curve analysis :|\n\n{e}")
            st.text(traceback.format_exc())

    with tab4:
        try:
            st.subheader(f"Bias Analysis - {animal_name}")
            n_previous_trials = st.slider("Number of previous trials to consider", 1, 10, 3, 1, key="bias_prev_trials_single")
            bias_multiple_sessions(project_data, animal_name=animal_name, n_previous_trials=n_previous_trials)
        except Exception as e:
            st.warning(f"Something went wrong with bias analysis :|\n\n{e}")
            st.text(traceback.format_exc())


def gng_bpod_analysis_multi_animal(project_data, index):
    """
    Multi-animal comparison analysis.
    Compares performance across multiple animals.
    """
    # Performance info
    with st.expander("ℹ️ Performance Info"):
        st.info("🚀 Multi-animal analysis uses caching for faster performance.")
        st.caption(f"📊 Dataset: {len(project_data)} sessions across {len(project_data['MouseName'].unique())} animals")
        # Add cache management
        if st.button("🗑️ Clear GNG Analysis Cache", key="clear_cache_multi"):
            st.cache_data.clear()
            st.toast("GNG analysis cache cleared - next computation will be fresh")

    bin = st.slider("Choose bin size", 5, 50, 30, 5, help="⚡ Cached computation", key="bin_multi")
    
    st.header(f"🐭🐭 Multi-Animal Comparison")
    
    tab1, tab2, tab3 = st.tabs(["👨‍🎓 D Prime", "👅 Lick Rate", "👂 Psychometric Slope"])

    with tab1:
        try:
            st.subheader("Multi-Animal d' Progression")
            multi_animal_d_prime_progression(project_data)
        except Exception as e:
            st.warning(f"Something went wrong with multi-animal d' progression :|\n\n{e}")
            st.text(traceback.format_exc())

        try:
            st.subheader("Cumulative Number of Trials vs Daily d' Progression")
            cumulative_number_of_trials_vs_daily_dprime(project_data, t=bin)
        except Exception as e:
            st.warning(f"Something went wrong with cumulative number of trials vs daily d' progression :|\n\n{e}")
            st.text(traceback.format_exc())

    with tab2:

        try:
            st.subheader("Daily Multi-Animal Lick Rate Comparison")
            daily_multi_animal_lick_rate(project_data, t=bin)
        except Exception as e:
            st.warning(f"Something went wrong with daily multi-animal lick rate :|\n\n{e}")
            st.text(traceback.format_exc())
        
        try:
            st.subheader("Daily Activity - Multi-Animal")
            daily_activity_multi_animal(project_data)
        except Exception as e:
            st.warning(f"Something went wrong with daily activity analysis :|\n\n{e}")
            st.text(traceback.format_exc())
        
    with tab3:
        # Session filtering options
        with st.expander("🔍 Filter Outlier Sessions", expanded=False):
            filter_outliers = st.checkbox("Remove sessions with low d-prime", value=False, 
                                         help="Filter out sessions where mean d-prime is below threshold",
                                         key="filter_outliers_multi")
            d_prime_threshold = 1.0
            if filter_outliers:
                d_prime_threshold = st.slider("d-prime threshold", 0.0, 3.0, 1.0, 0.1, 
                                             help="Sessions with d' below this value will be removed",
                                             key="d_prime_threshold_multi")
        
        n_indices_multi = st.slider("Number of indices to include", 1, 10, 2, 1, key="n_indices_multi")
        
        try:
            # Show Early Response filter checkbox only for the first call in this section.
            plot_psychometric_curves_with_boundaries(project_data, N_Boundaries = 1, n_indices = n_indices_multi, 
                                                    filter_outliers=filter_outliers, 
                                                    d_prime_threshold=d_prime_threshold, 
                                                    t=bin)
        except Exception as e:
            st.warning(f"Something went wrong with multi-animal psychometric curve analysis :|\n\n{e}")
            st.text(traceback.format_exc())

        try:
            plot_psychometric_curves_with_boundaries(project_data, N_Boundaries = 2, n_indices = n_indices_multi,
                                                    filter_outliers=filter_outliers, 
                                                    d_prime_threshold=d_prime_threshold, 
                                                    t=bin)
        except Exception as e:
            st.warning(f"Something went wrong with multi-animal psychometric curve analysis :|\n\n{e}")
            st.text(traceback.format_exc())
        
        try:
            plot_psychometric_curves_with_boundaries(project_data, N_Boundaries = 0, n_indices = n_indices_multi,
                                                    filter_outliers=filter_outliers, 
                                                    d_prime_threshold=d_prime_threshold, 
                                                    t=bin)
        except Exception as e:
            st.warning(f"Something went wrong with multi-animal psychometric curve analysis :|\n\n{e}")
            st.text(traceback.format_exc())

