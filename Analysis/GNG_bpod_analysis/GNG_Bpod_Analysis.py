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
import hashlib
from functools import lru_cache


def gng_bpod_analysis(project_data, index):
    name, session = getNameAndSession(project_data, index)
    st.header(f"{name}  ___#{session}___")
    
    # Performance info
    with st.expander("ℹ️ Performance Info"):
        st.info("🚀 Analysis results are cached for faster performance. Change parameters to trigger recomputation.")
        st.caption(f"📊 Session: {name} #{session}")
    
    bin = st.slider("Choose bin size", 5, 50, 15, 5, help="⚡ Cached - only recomputes when changed")


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
            n_previous_trials = st.slider("Number of previous trials to consider", 1, 10, 3, 1, help="⚡ Bias computation cached")
            plot_bias_analysis(project_data, index, n_previous_trials=n_previous_trials, plot=True)
        except Exception as e:
            st.warning(f"something went wrong with bias analysis :|\n\n{e}")
            st.text(traceback.format_exc())
            
        # Add cache management
        if st.button("🗑️ Clear GNG Analysis Cache"):
            st.cache_data.clear()
            st.toast("GNG analysis cache cleared - next computation will be fresh")





def gng_bpod_analysis_multipule(project_data, index):
    # Performance info
    with st.expander("ℹ️ Performance Info"):
        st.info("🚀 Multi-session analysis uses caching for faster performance.")
        st.caption(f"📊 Dataset: {len(project_data)} sessions across {len(project_data['MouseName'].unique())} animals")
    
    bin = st.slider("Choose bin size", 5, 50, 30, 5, help="⚡ Cached computation")
    animal_name = st.selectbox("Choose an Animal",
        sorted(project_data["MouseName"].unique()),  # Convert to list and sort
        key = "animal_select", help="⚡ Results cached per animal")
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
        try:
            st.subheader("Cumulative Number of Trials vs Daily d' Progression")

            # Prepare data: for each mouse, sum up number of trials over days, and plot daily d' vs cumulative trials
            mice = sorted(project_data["MouseName"].unique())
            fig = go.Figure()

            for mouse in mice:
                mouse_data = project_data[project_data["MouseName"] == mouse].sort_values("SessionDate")
                session_dates = mouse_data["SessionDate"].values
                n_trials_per_day = []
                d_prime_per_day = []
                colors = []
                for i in range(len(mouse_data)):
                    color = mouse_data.iloc[i]["Color"] if "Color" in mouse_data.columns else "gray"
                    colors.append(color)
                    stimuli, outcomes = preprocess_stimuli_outcomes(mouse_data, i)
                    n_trials_per_day.append(len(stimuli))
                # Cumulative number of trials
                cumulative_trials = np.cumsum(n_trials_per_day)
                data = d_prime_multiple_sessions(project_data, t=bin, animal_name = mouse, plot = False)
                d_prime_per_day = data["d_prime"]
                n_t = data["tones_per_class"]
                n_b = data["Boundaries"]

                # Determine marker symbols: 'circle' if n_b == 1, 'square' if n_b == 2
                marker_symbols = ['circle' if nb == 1 else 'square' for nb in n_b]
                # Marker line width is n_t value for each point
                marker_line_widths = n_t

                fig.add_trace(go.Scatter(
                    x=cumulative_trials,
                    y=d_prime_per_day,
                    mode='lines+markers',
                    marker=dict(
                        color=colors[0],
                        size=marker_line_widths*5,
                        symbol=marker_symbols,
                        line=dict(
                            width=2,
                            color=colors[0]
                        )
                    ),
                    name=str(mouse),
                    text=[f"{n_t[i]}T_ {n_b[i]}B" for i, date in enumerate(session_dates)],
                    textposition="top center",
                    showlegend=True
                ))

                # Add legend entries for marker shape (number of boundaries)
                if mouse == mice[0]:
                    # Only add these once to avoid duplicate legend entries
                    fig.add_trace(go.Scatter(
                        x=[None], y=[None],
                        mode='markers',
                        marker=dict(symbol='circle', color='gray', size=8),
                        name="1 Boundary",
                        showlegend=True,
                        hoverinfo='skip'
                    ))
                    fig.add_trace(go.Scatter(
                        x=[None], y=[None],
                        mode='markers',
                        marker=dict(symbol='square', color='gray', size=8),
                        name="2 Boundaries",
                        showlegend=True,
                        hoverinfo='skip'
                    ))
                    # Add legend entries for marker size (number of tones per class)
                    # We'll show a few representative sizes
                    for sizes in sorted(set(n_t)):
                        fig.add_trace(go.Scatter(
                            x=[None], y=[None],
                            mode='markers',
                            marker=dict(symbol='circle', color='white', size=sizes*5, line=dict(width=3, color='gray')),
                            name=f"{sizes} Tones",
                            showlegend=True,
                            hoverinfo='skip'
                        ))

                fig.update_layout(
                    xaxis_title="Cumulative Number of Trials",
                    yaxis_title="Daily d'",
                    title="Daily d' vs Cumulative Number of Trials per Mouse",
                    plot_bgcolor="white",
                    legend_title_text="Legend"
                )

            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Something went wrong with cumulative number of trials vs daily d' progression :|\n\n{e}")
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

