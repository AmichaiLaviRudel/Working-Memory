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
from scipy import stats



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
        with st.expander("👨‍🎓 Matrices & d'", expanded=False):
            if st.button("Run analysis", key="run_matrices_single"):
                try:
                    classifier_metric(project_data, index)
                    d_prime(project_data, index, t=bin, plot=True)
                    d_prime_for_stim_pairs(project_data, index, stim_pairs=None, t=bin, plot=True)
                except Exception as e:
                    st.warning(f"something went wrong with this graph :|\n\n{e}")
                    st.text(traceback.format_exc())

    with tab2:
        with st.expander("👅 Lick Rate", expanded=False):
            if st.button("Run analysis", key="run_lick_rate_single"):
                try:
                    licking_rate(project_data, index, t=bin)
                except Exception as e:
                    st.warning(f"something went wrong with this graph :|\n\n{e}")
                    st.text(traceback.format_exc())
        st.write(st.session_state.analysis_type)

        with st.expander("👅 Lick Data & First Lick", expanded=False):
            normalize_ftl = True
            if st.session_state.analysis_type == 'Behavior-Bpod GUI':
                normalize_ftl = st.checkbox(
                    "Normalize (Z-score per session)",
                    value=True,
                    key="first_lick_normalize_single",
                    help="Z-score first-lick latency within this session for comparability.",
                )
            if st.button("Run analysis", key="run_lick_data_single"):
                try:
                    df_go_first_licks, df_no_go_first_licks, lick_by_stimulus = process_and_plot_lick_data(
                        project_data, index, plot=True,
                    )
                    if st.session_state.analysis_type == 'Behavior-Bpod GUI':
                        plot_first_lick_by_stimulus(
                            project_data, index, plot=True, normalize=normalize_ftl
                        )
                        try:
                            st.subheader("First Lick Latency Analysis")
                            plot_first_lick_latency(
                                project_data, index, df_go_first_licks, df_no_go_first_licks, plot=True,
                            )
                        except Exception as e:
                            st.warning(f"something went wrong with latency analysis :|\n\n{e}")
                            st.text(traceback.format_exc())
                    elif st.session_state.analysis_type == 'Educage':
                        plot_n_lick_by_stimulus(project_data, index, plot=True)
                except Exception as e:
                    st.warning(f"something went wrong with latency analysis :|\n\n{e}")
                    st.text(traceback.format_exc())

        with st.expander("📊 Daily Activity", expanded=False):
            if st.button("Run analysis", key="run_daily_activity_single"):
                try:
                    daily_activity_single_animal(project_data, index)
                except Exception as e:
                    st.warning(f"something went wrong with daily activity analysis :|\n\n{e}")
                    st.text(traceback.format_exc())

    with tab3:
        with st.expander("📈 Learning Curve", expanded=False):
            if st.button("Run analysis", key="run_learning_curve_single"):
                try:
                    learning_curve(project_data, index)
                except Exception as e:
                    st.warning(f"something went wrong with this graph :|\n\n{e}")
                    st.text(traceback.format_exc())

    with tab4:
        with st.expander("👂 Psychometric Curve", expanded=False):
            if st.button("Run analysis", key="run_psychometric_single"):
                try:
                    psychometric_curve(project_data, index)
                except Exception as e:
                    st.warning(f"something went wrong with this graph :|\n\n{e}")
                    st.text(traceback.format_exc())

    with tab5:
        with st.expander("🎯 Bias Analysis", expanded=False):
            n_previous_trials = st.slider("Number of previous trials to consider", 1, 10, 3, 1, help="⚡ Bias computation cached")
            if st.button("Run analysis", key="run_bias_single"):
                try:
                    plot_bias_analysis(project_data, index, n_previous_trials=n_previous_trials, plot=True)
                except Exception as e:
                    st.warning(f"something went wrong with bias analysis :|\n\n{e}")
                    st.text(traceback.format_exc())

    with tab6:
        with st.expander("🧠 GLM Analysis", expanded=False):
            if st.button("Run analysis", key="run_glm_single"):
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
        with st.expander("👅 Lick Rate Progression", expanded=False):
            if st.button("Run analysis", key="run_lick_rate_prog"):
                try:
                    lick_rate_multipule_sessions(project_data, t=bin, plot=True, animal_name=animal_name)
                except Exception as e:
                    st.warning(f"Something went wrong with lick rate analysis :|\n\n{e}")
                    st.text(traceback.format_exc())

        with st.expander("👅 First Lick Latency Progression", expanded=False):
            if st.button("Run analysis", key="run_first_lick_lat_prog"):
                try:
                    plot_first_lick_latency_multiple_sessions(project_data, animal_name=animal_name, plot=True)
                except Exception as e:
                    st.warning(f"Something went wrong with first lick latency analysis :|\n\n{e}")
                    st.text(traceback.format_exc())

    with tab2:
        with st.expander("👨‍🎓 Daily Multi-Session d' Comparison", expanded=False):
            if st.button("Run analysis", key="run_dprime_multi_sess"):
                try:
                    d_prime_multiple_sessions(project_data, t=bin, animal_name=animal_name, plot=True)
                except Exception as e:
                    st.warning(f"Something went wrong with daily multi-session d' :|\n\n{e}")
                    st.text(traceback.format_exc())

    with tab3:
        # Session filtering options
        with st.expander("🔍 Filter Outlier Sessions", expanded=False):
            filter_outliers = st.checkbox("Remove sessions with low d-prime or hit rate", value=False, 
                                         help="Filter out sessions where d-prime or hit rate is below threshold",
                                         key="filter_outliers_single")
            d_prime_threshold = 1.0
            hit_rate_threshold = 0.6
            if filter_outliers:
                d_prime_threshold = st.slider("d-prime threshold", 0.0, 3.0, 1.0, 0.1, 
                                             help="Sessions with d' below this value will be removed",
                                             key="d_prime_threshold_single")
                hit_rate_threshold = st.slider("Hit rate threshold", 0.0, 1.0, 0.6, 0.05, 
                                              help="Sessions with hit rate below this value will be removed",
                                              key="hit_rate_threshold_single")
        
        # Filter project_data for the selected animal
        animal_data = project_data[project_data['MouseName'] == animal_name].copy()
        
        n_indices = st.slider("Number of indices to include", 1, 10, 2, 1, key="n_indices_single")

        with st.expander("👂 Psychometric Curves (1 Boundary)", expanded=False):
            if st.button("Run analysis", key="run_psych_1b_single"):
                try:
                    plot_psychometric_curves_with_boundaries(animal_data, N_Boundaries=1, n_indices=n_indices,
                                                        filter_outliers=filter_outliers,
                                                        d_prime_threshold=d_prime_threshold,
                                                        hit_rate_threshold=hit_rate_threshold,
                                                        t=bin, key_suffix="_single")
                except Exception as e:
                    st.warning(f"Something went wrong with psychometric curve analysis :|\n\n{e}")
                    st.text(traceback.format_exc())

        with st.expander("👂 Psychometric Curves (2 Boundaries)", expanded=False):
            if st.button("Run analysis", key="run_psych_2b_single"):
                try:
                    plot_psychometric_curves_with_boundaries(animal_data, N_Boundaries=2, n_indices=n_indices,
                                                        filter_outliers=filter_outliers,
                                                        d_prime_threshold=d_prime_threshold,
                                                        hit_rate_threshold=hit_rate_threshold,
                                                        t=bin, key_suffix="_single")
                except Exception as e:
                    st.warning(f"Something went wrong with psychometric curve analysis :|\n\n{e}")
                    st.text(traceback.format_exc())

        with st.expander("👂 Psychometric Curves (0 Boundaries)", expanded=False):
            if st.button("Run analysis", key="run_psych_0b_single"):
                try:
                    plot_psychometric_curves_with_boundaries(animal_data, N_Boundaries=0, n_indices=n_indices,
                                                        filter_outliers=filter_outliers,
                                                        d_prime_threshold=d_prime_threshold,
                                                        hit_rate_threshold=hit_rate_threshold,
                                                        t=bin, key_suffix="_single")
                except Exception as e:
                    st.warning(f"Something went wrong with psychometric curve analysis :|\n\n{e}")
                    st.text(traceback.format_exc())

        with st.expander("📈 Slope Progression", expanded=False):
            if st.button("Run analysis", key="run_slope_prog_single"):
                try:
                    st.caption("Slope at boundary across sessions (extracted from fitted psychometric curves)")
                    psychometric_curve_multiple_sessions(animal_data, animal_name=animal_name, plot=True)
                except Exception as e:
                    st.warning(f"Something went wrong with slope progression analysis :|\n\n{e}")
                    st.text(traceback.format_exc())

        with st.expander("📈 Correlation log(|slope|) vs d'", expanded=False):
            if st.button("Run analysis", key="run_corr_slope_dprime_single"):
                try:
                    correlation_log_slope_vs_dprime_multi_sessions(project_data, animal_name=animal_name)
                except Exception as e:
                    st.warning(f"Something went wrong with correlation log(|slope|) vs d' analysis :|\n\n{e}")
                    st.text(traceback.format_exc())

    with tab4:
        with st.expander("🎯 Bias Analysis", expanded=False):
            n_previous_trials = st.slider("Number of previous trials to consider", 1, 10, 3, 1, key="bias_prev_trials_single")
            if st.button("Run analysis", key="run_bias_multi_single"):
                try:
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
        with st.expander("📈 Multi-Animal d' Progression", expanded=False):
            if st.button("Run analysis", key="run_dprime_prog_multi"):
                try:
                    multi_animal_d_prime_progression(project_data)
                except Exception as e:
                    st.warning(f"Something went wrong with multi-animal d' progression :|\n\n{e}")
                    st.text(traceback.format_exc())

        with st.expander("📈 Cumulative Number of Trials vs Daily d'", expanded=False):
            if st.button("Run analysis", key="run_cumulative_dprime_multi"):
                try:
                    cumulative_number_of_trials_vs_daily_dprime(project_data, t=bin)
                except Exception as e:
                    st.warning(f"Something went wrong with cumulative number of trials vs daily d' progression :|\n\n{e}")
                    st.text(traceback.format_exc())

        with st.expander("📈 Pairwise d' by Octave Distance", expanded=False):
            st.caption("d' distributions for stimulus pairs at different octave distances from boundary")
            col_opt1, col_opt2 = st.columns(2)
            with col_opt1:
                compare_platforms = st.checkbox("Compare platforms (Rig vs Educage)", 
                                               value=False, key="compare_platforms_dprime")
            with col_opt2:
                filter_successful = st.checkbox("Only successful sessions (d' ≥ threshold)", 
                                               value=False, key="filter_successful_dprime")
            dprime_thresh = 1.0
            if filter_successful:
                dprime_thresh = st.slider("d' threshold", 0.5, 3.0, 1.0, 0.1, key="dprime_thresh_pairwise")
            if st.button("Run analysis", key="run_pairwise_dprime_multi"):
                try:
                    multi_animal_pairwise_dprime(project_data, t=bin, compare_platforms=compare_platforms,
                                                filter_successful=filter_successful, dprime_threshold=dprime_thresh)
                except Exception as e:
                    st.warning(f"Something went wrong with pairwise d' analysis :|\n\n{e}")
                    st.text(traceback.format_exc())

    with tab2:
        with st.expander("👅 Daily Multi-Animal Lick Rate Comparison", expanded=False):
            if st.button("Run analysis", key="run_daily_lick_multi"):
                try:
                    daily_multi_animal_lick_rate(project_data, t=bin)
                except Exception as e:
                    st.warning(f"Something went wrong with daily multi-animal lick rate :|\n\n{e}")
                    st.text(traceback.format_exc())

        with st.expander("📊 Daily Activity - Multi-Animal", expanded=False):
            if st.button("Run analysis", key="run_daily_activity_multi"):
                try:
                    daily_activity_multi_animal(project_data)
                except Exception as e:
                    st.warning(f"Something went wrong with daily activity analysis :|\n\n{e}")
                    st.text(traceback.format_exc())

        with st.expander("📏 First Lick Hellinger: First vs Last Day", expanded=False):
            if st.button("Run analysis", key="run_hellinger_multi"):
                try:
                    plot_first_lick_hellinger_first_vs_last_day(project_data)
                except Exception as e:
                    st.warning(f"Something went wrong with first-lick Hellinger comparison :|\n\n{e}")
                    st.text(traceback.format_exc())

        with st.expander("📏 Go first-lick distance to reinforcement delay: First vs Last Day", expanded=False):
            if st.button("Run analysis", key="run_go_dist_reinf_multi"):
                try:
                    plot_go_first_lick_distance_to_reinforcement_first_vs_last_day(project_data)
                except Exception as e:
                    st.warning(f"Something went wrong with Go first-lick distance to reinforcement delay :|\n\n{e}")
                    st.text(traceback.format_exc())

        with st.expander("👅 First Lick by Distance (by learning phase)", expanded=False):
            scope_phase = st.radio(
                "Scope",
                ["All animals", "Selected animal only"],
                key="first_lick_phase_scope",
                horizontal=True,
            )
            if scope_phase == "Selected animal only":
                animal_for_phase = st.selectbox("Animal (if Selected only)", sorted(project_data["MouseName"].unique()), key="animal_phase_multi")
                indices_phase = project_data[project_data["MouseName"] == animal_for_phase].index.tolist()
            else:
                indices_phase = None
            show_2b_phase = st.checkbox("Include 2B Expert", value=True, key="show_2b_phase_multi")
            min_n_phase = st.slider("Min trials per bin", min_value=1, max_value=50, value=5, step=1, key="min_n_phase_multi",
                                    help="Exclude distance bins with fewer than this many trials")
            st.caption("Novice (first 2 of 1B, Tones>1), 1B Expert (last 2 of 1B, Tones≥3), 2B Expert (last 2 of 2B, Tones≥3). X = distance to closest boundary (oct).")
            if st.button("Run analysis", key="run_first_lick_phase_multi"):
                try:
                    plot_first_lick_by_distance_by_phase(project_data, index=indices_phase, plot=True, show_2b=show_2b_phase, min_n=min_n_phase)
                except Exception as e:
                    st.warning(f"Something went wrong with first-lick-by-phase plot :|\n\n{e}")
                    st.text(traceback.format_exc())
        
    with tab3:
        # Session filtering options
        with st.expander("🔍 Filter Outlier Sessions", expanded=False):
            filter_outliers = st.checkbox("Remove sessions with low d-prime or hit rate", value=False, 
                                         help="Filter out sessions where d-prime or hit rate is below threshold",
                                         key="filter_outliers_multi")
            d_prime_threshold = 1.0
            hit_rate_threshold = 0.6
            if filter_outliers:
                d_prime_threshold = st.slider("d-prime threshold", 0.0, 3.0, 1.0, 0.1, 
                                             help="Sessions with d' below this value will be removed",
                                             key="d_prime_threshold_multi")
                hit_rate_threshold = st.slider("Hit rate threshold", 0.0, 1.0, 0.6, 0.05, 
                                              help="Sessions with hit rate below this value will be removed",
                                              key="hit_rate_threshold_multi")
        
        n_indices_multi = st.slider("Number of indices to include", 1, 10, 2, 1, key="n_indices_multi")

        with st.expander("👂 Psychometric Curves (1 Boundary)", expanded=False):
            if st.button("Run analysis", key="run_psych_1b_multi"):
                try:
                    plot_psychometric_curves_with_boundaries(project_data, N_Boundaries=1, n_indices=n_indices_multi,
                                                        filter_outliers=filter_outliers,
                                                        d_prime_threshold=d_prime_threshold,
                                                        hit_rate_threshold=hit_rate_threshold,
                                                        t=bin, key_suffix="_multi")
                except Exception as e:
                    st.warning(f"Something went wrong with multi-animal psychometric curve analysis :|\n\n{e}")
                    st.text(traceback.format_exc())

        with st.expander("👂 Psychometric Curves (2 Boundaries)", expanded=False):
            if st.button("Run analysis", key="run_psych_2b_multi"):
                try:
                    plot_psychometric_curves_with_boundaries(project_data, N_Boundaries=2, n_indices=n_indices_multi,
                                                        filter_outliers=filter_outliers,
                                                        d_prime_threshold=d_prime_threshold,
                                                        hit_rate_threshold=hit_rate_threshold,
                                                        t=bin, key_suffix="_multi")
                except Exception as e:
                    st.warning(f"Something went wrong with multi-animal psychometric curve analysis :|\n\n{e}")
                    st.text(traceback.format_exc())

        with st.expander("👂 Psychometric Curves (0 Boundaries)", expanded=False):
            if st.button("Run analysis", key="run_psych_0b_multi"):
                try:
                    plot_psychometric_curves_with_boundaries(project_data, N_Boundaries=0, n_indices=n_indices_multi,
                                                        filter_outliers=filter_outliers,
                                                        d_prime_threshold=d_prime_threshold,
                                                        hit_rate_threshold=hit_rate_threshold,
                                                        t=bin, key_suffix="_multi")
                except Exception as e:
                    st.warning(f"Something went wrong with multi-animal psychometric curve analysis :|\n\n{e}")
                    st.text(traceback.format_exc())

        with st.expander("📈 Multi-Animal Slope Progression", expanded=False):
            n_boundaries_slope = st.radio("Number of Boundaries", [1, 2], horizontal=True, key="n_boundaries_slope_multi")
            if st.button("Run analysis", key="run_slope_prog_multi"):
                try:
                    st.caption("Slope at boundary progression across animals and sessions")
                    multi_animal_psychometric_slope_progression(project_data, N_Boundaries=n_boundaries_slope)
                except Exception as e:
                    st.warning(f"Something went wrong with multi-animal slope progression :|\n\n{e}")
                    st.text(traceback.format_exc())

        with st.expander("📈 Correlation log(|slope|) vs d' - Multi-Animal", expanded=False):
            if st.button("Run analysis", key="run_corr_slope_dprime_multi"):
                try:
                    correlation_log_slope_vs_dprime_multi_animal(project_data)
                except Exception as e:
                    st.warning(f"Something went wrong with correlation log(|slope|) vs d' analysis :|\n\n{e}")
                    st.text(traceback.format_exc())

        with st.expander("📏 Distance between x0 and task boundary", expanded=False):
            st.caption("Per-session distance of fitted threshold (x0) to true boundary; In_between flags for 2B.")
            if st.button("Run analysis", key="run_dist_x0_boundary_multi"):
                try:
                    df_dist = distance_between_x0_and_boundary(project_data)
                    cols = [c for c in ["MouseName", "SessionDate", "N_Boundaries", "Distance_x0", "Distance_x0_low", "Distance_x0_high", "In_between_boundaries_low", "In_between_boundaries_high", "In_between_boundaries"] if c in df_dist.columns]
                    if cols:
                        st.dataframe(df_dist[cols], use_container_width=True, hide_index=True)
                    else:
                        st.info("Need Psychometric_x0 (or x0_low/x0_high) columns. Run 'Compute Metrics' on the global dataset.")
                except Exception as e:
                    st.warning(f"Something went wrong :|\n\n{e}")
                    st.text(traceback.format_exc())

        with st.expander("📊 Slope & distance by boundary (+ hit rate by region)", expanded=False):
            st.caption("Long-format: slope, distance to boundary, and hit rate (overall / below low / above high) per boundary.")
            if st.button("Run analysis", key="run_slope_dist_boundary_multi"):
                try:
                    df_compare = compare_slope_and_distance_by_boundary(project_data, plot=True)
                    if not df_compare.empty:
                        st.dataframe(df_compare, use_container_width=True, hide_index=True)
                    else:
                        st.info("No data. Need Psychometric_slope_* and distance columns (run 'Compute Metrics').")
                except Exception as e:
                    st.warning(f"Something went wrong :|\n\n{e}")
                    st.text(traceback.format_exc())