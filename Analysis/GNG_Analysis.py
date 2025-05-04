from functions import *

import streamlit as st
import scipy.io as spio
import numpy as np
import pandas as pd
from scipy.stats import norm
import os
import altair as alt


def gng_analysis(project_data, index):
    bin = st.slider("Choose bin size", 5, 50, 30, 5)

    tab1, tab2, tab3 = st.tabs(["📈 Learning Curve", "👅 Lick Rate", "👨‍🎓 D Prime"])

    with tab1:
        try:
            learning_curve(project_data, index)
        except:
            st.warning("something went wrong with this graphs :|")

    with tab2:
        try:
            licking_rate(project_data, index, t=bin)
        except:
            st.warning("something went wrong with this graphs :|")
    with tab3:
        try:
            d_prime(project_data, index, t=bin)
        except:
            st.warning("something went wrong with this graph :|")


def gng_analysis_multipule(project_data, index):
    bin = st.slider("Choose bin size", 5, 50, 30, 5)
    tab1, tab2 = st.tabs(["👅 Lick Rate", "👨‍🎓 D Prime"])
    with tab1:
        lick_rate_multipule_sessions(project_data, t=bin, plot=True, indexs=index)
    with tab2:
        d_prime_multiple_sessions(project_data, t=bin, indexs=index)

def read_results_file(file):
    try:
        data = pd.read_csv(file)
    except:
        st.warning("Can't read uploaded file")

    return data


def read_gng_file(file_path):
    data = spio.loadmat(file_path, squeeze_me=True)
    # run over data and force to be length 1
    for key in data:
        data[key] = [data[key]]

    # remove '__header__', '__version__', '__globals__'
    data.pop('__header__')
    data.pop('__version__')
    data.pop('__globals__')

    # dict to dataframe
    data = pd.DataFrame.from_dict(data)

    path, name = os.path.dirname(file_path), os.path.basename(file_path).split(".")[0]
    data.to_csv(f"{os.path.join(path, name)}.csv", index=False)

    return data


def responses(selected_data, index=0):
    stim = selected_data.iloc[index]["stim_types"]
    trail = selected_data.iloc[index]["trial_responses"]

    stim = np.fromstring(stim[1:-1], sep=" ")
    trail = np.fromstring(trail[1:-1], sep=" ")
    comper = stim - trail
    "---"
    responses = pd.DataFrame({"Hit":  np.cumsum(comper == 0),
                              "CR":   np.cumsum(comper == -1),
                              "FA":   np.cumsum(comper == -2),
                              "Miss": np.cumsum(comper == 1)})
    return responses


def learning_curve(selected_data, index=0):
    st.subheader("Mouse performance")
    st.line_chart(responses(selected_data, index))


def licking_rate(selected_data, index=0, t=10, plot=True):
    # selected_data = pd.read_csv("G:\My Drive\Study\Lab\Code_temp\Rec_OFC_2_formmated_for_db.csv")

    data = responses(selected_data, index)

    hit_bin = data["Hit"].diff().rolling(t, step=t).sum()
    miss_bin = data["Miss"].diff().rolling(t, step=t).sum()
    cr_bin = data["CR"].diff().rolling(t, step=t).sum()
    fa_bin = data["FA"].diff().rolling(t, step=t).sum()

    rates = pd.DataFrame({"Hit":  hit_bin,
                          "Miss": miss_bin,
                          "CR":   cr_bin,
                          "FA":   fa_bin})

    hit_rate = 100 * hit_bin / (hit_bin + miss_bin)
    fa_rate = 100 * fa_bin / (cr_bin + fa_bin)

    frac = pd.DataFrame({"Go": hit_rate, "NoGo": fa_rate})
    if plot:
        st.subheader("Licking rate")
        st.line_chart(frac)

    return rates


# Function to calculate the d prime
def d_prime(selected_data, index=0, t=10):
    rates = licking_rate(selected_data, index, t=t, plot=False)
    # Calculate hits, misses, false alarms, and correct rejections

    hit = rates["Hit"]
    miss = rates["Miss"]
    fa = rates["FA"]
    cr = rates["CR"]

    # Calculate hit rate and avoid dividing by 0
    hit_rate = hit / (hit + miss)
    fa_rate = fa / (cr + fa)

    # Replace 0 and 1 with 0+ and 1- to avoid d' infinity
    hit_rate[hit_rate == 1] = 1 - 1 / (2 * (hit + miss))
    hit_rate[hit_rate == 0] = 1 / (2 * (hit + miss))
    fa_rate[fa_rate == 1] = 1 - 1 / (2 * (cr + fa))
    fa_rate[fa_rate == 0] = 1 / (2 * (cr + fa))


    # Calculate d'
    d = norm.ppf(hit_rate) - norm.ppf(fa_rate)
    st.subheader("d'")
    df = pd.DataFrame({"d'": d})
    st.line_chart(df)

    return d

