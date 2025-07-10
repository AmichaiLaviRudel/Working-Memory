from Analysis.GNG_bpod_analysis.psychometric_curves import *
from Analysis.GNG_bpod_analysis.metric import *
from Analysis.GNG_bpod_analysis.GNG_bpod_general import *
from Analysis.GNG_bpod_analysis.licking_and_outcome import *

import traceback
import streamlit as st


def gng_bpod_analysis(project_data, index):
    name, session = getNameAndSession(project_data, index)
    st.header(f"{name}  ___#{session}___")
    bin = st.slider("Choose bin size", 5, 50, 15, 5)


    tab1, tab2, tab3, tab4 = st.tabs([ "👨‍🎓Matrices", "👅 Lick Rate", "📈 Learning Curve", "👂 Psychometric Curve"])

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
            process_and_plot_lick_data(project_data, index)
        except  Exception as e:
            st.warning(f"something went wrong with this graph :|\n\n{e}")
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


def gng_bpod_analysis_multipule(project_data, index):
    bin = st.slider("Choose bin size", 5, 50, 30, 5)
    animal_name = st.selectbox("Choose an Animal",
        sorted(project_data["MouseName"].unique()),  # Convert to list and sort
        key = "animal_select")
    tab1, tab2, tab3 = st.tabs(["👅 Lick Rate", "👨‍🎓 D Prime", "👂 Psychometric Slope"])


    with tab1:
        lick_rate_multipule_sessions(project_data, t=bin, plot=True, animal_name = animal_name)
    with tab2:
        d_prime_multiple_sessions(project_data, t=bin, animal_name = animal_name)
        # d_prime_multiple_sessions_divde_oneNtwo(project_data, t = 10, animal_name = 'None', plot = True)

        multi_animal_d_prime_progression(project_data)
        multi_animal_d_prime_progression(project_data, N_Boundaries = 2)


    with tab3:
        psychometric_curve_multiple_sessions(project_data, animal_name = animal_name, plot=True)
        multi_animal_psychometric_slope_progression(project_data,  N_Boundaries = None)
        # multi_animal_psychometric_slope_progression(project_data,  N_Boundaries = 2)
        plot_psychometric_curves_with_boundaries(project_data, N_Boundaries = 1, n_indices = 2)
        plot_psychometric_curves_with_boundaries(project_data,  N_Boundaries = 2, n_indices = 2)




