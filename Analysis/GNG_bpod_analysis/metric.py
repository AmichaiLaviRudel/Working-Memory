from Analysis.GNG_bpod_analysis.psychometric_curves import *
from Analysis.GNG_bpod_analysis.GNG_bpod_general import *

import re
import ast
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import altair as alt
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import streamlit as st

# Function to calculate the d prime
def d_prime(selected_data, index=0, t=10, plot=False):
    from Analysis.GNG_bpod_analysis.licking_and_outcome import licking_rate
    rates, frac = licking_rate(selected_data, index, t=t, plot=False)
    # Calculate hits, misses, false alarms, and correct rejections
    hit = rates["Hit"]
    miss = rates["Miss"]
    fa = rates["FA"]
    cr = rates["CR"]

    # Ensure the DataFrame has valid numeric data
    frac = frac.dropna(how = "all").astype(float)

    # Convert percentages to proportions (0-1 scale)
    hit_rate = frac["Go"] / 100
    fa_rate = frac["NoGo"] / 100

    # Filter out bins where hit_rate + fa_rate <= 0.4
    valid_bins = (hit_rate + fa_rate) > 0.4
    # hit_rate = hit_rate[valid_bins]
    # fa_rate = fa_rate[valid_bins]

    # Prevent hit_rate and fa_rate from being exactly 0 or 1
    hit_rate = hit_rate.clip(1e-3, 1 - 1e-3)
    fa_rate = fa_rate.clip(1e-3, 1 - 1e-3)

    # Compute d'
    d = norm.ppf(hit_rate) - norm.ppf(fa_rate)
    # valid_bins = (fa_rate+hit_rate) >= 0.5

    # Create a DataFrame with a safer column name
    df = pd.DataFrame({"index": range(len(d)), "d_prime": d})

    if plot:
        st.subheader("d' over trials")

        # Create an Altair line chart
        line_chart = alt.Chart(df).mark_line().encode(
            x = alt.X("index:Q", title = "Trial Index"),
            y = alt.Y("d_prime:Q", title = "d'"),
            tooltip = ["index", "d_prime"]
        ).interactive().properties(

        )

        # Add a horizontal line at y=1
        horizontal_line = alt.Chart(pd.DataFrame({"y": [1]})).mark_rule(
            color = "gray",
            strokeDash = [4, 4]
        ).encode(
            y = "y:Q"
        )

        # Combine the charts
        final_chart = line_chart + horizontal_line

        # Display the Altair chart
        st.altair_chart(final_chart, use_container_width = True)
    return d

def d_prime_multiple_sessions(selected_data, t=10, animal_name='None', plot = True):
    if animal_name == "None":
        # Step 1: Let the user choose an animal from the data, assign a unique key
        animal_name = st.selectbox("Choose an Animal", selected_data["MouseName"].unique(), key="d_prime_animal_select")
    data = []
    # Step 2: Automatically select all sessions for the chosen animal
    session_indices, session_dates = get_sessions_for_animal(selected_data, animal_name)

    ds = np.zeros([len(session_indices), 3])  # mean, std, max for each session
    tones_per_class = []
    boundaries = []


    # Compute d' statistics and collect metadata
    for idx, sess_idx in enumerate(session_indices):
        d = d_prime(selected_data, index=sess_idx, t=t)
        ds[idx, 0] = np.nanmean(d)
        ds[idx, 1] = np.nanstd(d)
        ds[idx, 2] = np.nanmax(d)
        # Retrieve session metadata
        tones_per_class.append(selected_data.loc[sess_idx, 'Tones_per_class'])
        boundaries.append(selected_data.loc[sess_idx, 'N_Boundaries'])

    # Build DataFrame for plotting
    data = pd.DataFrame({
        'Session Index':   np.arange(1, len(session_indices) + 1),
        'SessionDate':     session_dates,
        'd_prime': ds[:, 0],
        'Error': ds[:, 1],
        'Max_d_prime': ds[:, 2],
        'tones_per_class': tones_per_class,
        'Boundaries':      boundaries
    })



    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('Session Index:Q', title='Session Index', axis=alt.Axis(format='.0f', tickCount=len(session_indices))),  # Use session index
        y=alt.Y('d_prime:Q', title="Mean d'", scale=alt.Scale(domain=[-2, 6])),  # Set y-limit between -2 and 6
        tooltip=['Session Index:Q', 'd_prime', 'Error']
    ).properties(
        width=600,
        height=300
    )

    # Adding error bars
    error_bars = chart.mark_errorbar().encode(
        x='Session Index:Q',
        y='d_prime:Q',
        yError='Error:Q'
    )
    # Adding a horizontal line at y = 1
    horizontal_line = alt.Chart(pd.DataFrame({'y': [1]})).mark_rule(color='gray').encode(
        y='y:Q'
    )

    # Annotations: empty circle for Boundaries==1, filled circle for Boundaries==2
    annotations = alt.Chart(data).mark_point(size = 50).encode(
        x = 'Session Index:Q',
        y = 'd_prime:Q',
        fill = alt.condition(alt.datum.Boundaries == 2, alt.value('black'), alt.value('white')),
        stroke = alt.value('black'),
        tooltip = [
            alt.Tooltip('Session Index', title = 'Session Index'),
            alt.Tooltip('tones_per_class', title = 'tones_per_class'),
            alt.Tooltip('Boundaries', title = 'Boundaries'),
            alt.Tooltip('d_prime', title = "d'")
        ]
    )

    if plot:
        # Plot using Altair with session index
        st.title(f"d' Progress for {animal_name}")
        # Combine the line chart, error bars, and the horizontal line
        st.altair_chart(chart + error_bars + horizontal_line+annotations, use_container_width=True)

    return data

def multi_animal_d_prime_progression(selected_data):
    st.title("Multi-Animal Progress")

    # Get unique subject names
    subjects = selected_data["MouseName"].unique()

    # Initialize arrays to store d' values
    d_prime_data = []
    session_counts = []

    for subject in subjects:
        # Compute d' for each subject
        d_prime_result = d_prime_multiple_sessions(selected_data, animal_name=subject, plot=False)
        d_prime_values = d_prime_result["d_prime"]

        # Store d' progression
        d_prime_data.append(d_prime_values)
        session_counts.append(len(d_prime_values))

    # Determine max number of sessions for alignment
    max_sessions = max(session_counts)

    # Convert list of arrays to DataFrame (aligned by padding with NaN)
    d_prime_df = pd.DataFrame([np.pad(d, (0, max_sessions - len(d)), constant_values=np.nan) for d in d_prime_data])

    # Compute average d' across subjects
    avg_d_prime = d_prime_df.mean(axis=0, skipna=True)

    # Prepare DataFrame for Altair
    data_list = []
    for i, subject in enumerate(subjects):
        for session, d_prime in enumerate(d_prime_df.iloc[i]):
            if not np.isnan(d_prime):
                data_list.append({"Session": session + 1, "d_prime": d_prime, "Mouse": subject})

    # Add average line data
    avg_list = []
    for session, d_prime in enumerate(avg_d_prime):
        if not np.isnan(d_prime):
            avg_list.append({"Session": session + 1, "d_prime": d_prime, "Mouse": "Average"})

    df_altair = pd.DataFrame(data_list)
    df_avg = pd.DataFrame(avg_list)

    line_chart = alt.Chart(df_altair).mark_line(opacity = 0.7, strokeWidth=1).encode(
        x = alt.X("Session:Q", title = "Session"),
        y = alt.Y("d_prime:Q", title = "d'"),
        color=alt.Color("Mouse:N", scale=alt.Scale(scheme="greys"), legend=alt.Legend(title="Subject")),  # ✅ Gray palette
        tooltip = ["Mouse", "Session", "d_prime"]
    ).properties(title = "d' Progression Across Animals", width = 700, height = 400)

    # Create Altair line chart for the average line
    avg_line = alt.Chart(df_avg).mark_line(strokeWidth=3).encode(
        x="Session:Q",
        y="d_prime:Q"
    )

    # Add horizontal reference line at y = 1 (dashed black)
    ref_line = alt.Chart(pd.DataFrame({"y": [1]})).mark_rule(color="black", strokeDash=[5, 5]).encode(
        y="y:Q"
    )

    # Combine charts
    final_chart = (line_chart + avg_line + ref_line).properties(title="d' Progression Across Animals")

    # Display chart in Streamlit
    st.altair_chart(final_chart, use_container_width=True)

def classifier_metric(project_data, index):
    from Analysis.GNG_bpod_analysis.licking_and_outcome import responses, licking_rate

    response = responses(project_data, index)
    hit = np.array(response["Hit"][-1:])[0]
    miss = np.array(response["Miss"][-1:])[0]
    cr = np.array(response["CR"][-1:])[0]
    fa = np.array(response["FA"][-1:])[0]

    d = d_prime(project_data, index, t=10)
    # Compute rates
    hit_rate = hit / (hit + miss)
    fa_rate = fa / (fa + cr)

    # valid_bins = (hit_rate+fa_rate) > 0.4
    # Avoid extreme values (log(0) issue)
    hit_rate = np.clip(hit_rate, 0.01, 0.99)
    fa_rate = np.clip(fa_rate, 0.01, 0.99)

    # Compute d' and criterion (c)
    d_mean = np.nanmean(d)
    criterion_c = -0.5 * (norm.ppf(hit_rate) + norm.ppf(fa_rate))

    # Accuracy
    accuracy = (hit + cr) / (hit + miss + cr + fa)

    # Precision (PPV)
    precision = hit / (hit + fa) if (hit + fa) > 0 else 0

    # Recall (Sensitivity, TPR)
    recall = hit / (hit + miss) if (hit + miss) > 0 else 0

    # F1-score
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # False Positive Rate (FPR)
    fpr = fa / (fa + cr) if (fa + cr) > 0 else 0

    # Simulating ROC-AUC (Assumes mouse responses as probabilistic)
    y_true = np.array([1] * (hit + miss) + [0] * (cr + fa))  # Actual labels (1=Go, 0=No-Go)
    y_scores = np.array([1] * hit + [0] * miss + [1] * fa + [0] * cr)  # Mouse responses

    roc_auc = roc_auc_score(y_true, y_scores)

    # ---- STREAMLIT APP ----
    # Creating a DataFrame with classification metrics as columns
    class_metric_df = pd.DataFrame({
        "Hit Rate":                  [hit_rate],
        "False Alarm Rate":          [fa_rate],
        "d'":                        [d_mean],
        "Accuracy":                  [accuracy],
        "Precision":                 [precision],
        "F1 Score":                  [f1_score],
        "False Positive Rate (FPR)": [fpr],
        "ROC-AUC Score":             [roc_auc]
    })

    # Convert numbers to 3 decimal places
    class_metric_df = class_metric_df.applymap(lambda x: round(x, 3))

    # Configure explanations for each column
    column_explanations = {
        "Hit Rate":                  st.column_config.NumberColumn(
            "Hit Rate",
            help = "Proportion of correctly detected 'Go' trials (TP / (TP + FN))"
        ),
        "False Alarm Rate":          st.column_config.NumberColumn(
            "False Alarm Rate",
            help = "Proportion of incorrect responses to 'No-Go' trials (FP / (FP + TN))"
        ),
        "d'":                        st.column_config.NumberColumn(
            "d'",
            help = "Sensitivity index (d-prime) measuring signal detection ability: Z(Hit Rate) - Z(False Alarm Rate)"
        ),
        "Accuracy":                  st.column_config.NumberColumn(
            "Accuracy",
            help = "Overall correctness of the classifier: (TP + TN) / Total Trials"
        ),
        "Precision":                 st.column_config.NumberColumn(
            "Precision",
            help = "Proportion of predicted 'Go' responses that were correct: TP / (TP + FP)"
        ),
        "F1 Score":                  st.column_config.NumberColumn(
            "F1 Score",
            help = "Harmonic mean of Precision and Recall: 2 * (Precision * Recall) / (Precision + Recall)"
        ),
        "False Positive Rate (FPR)": st.column_config.NumberColumn(
            "False Positive Rate (FPR)",
            help = "Proportion of No-Go trials incorrectly classified as 'Go': FP / (FP + TN)"
        ),
        "ROC-AUC Score":             st.column_config.NumberColumn(
            "ROC-AUC Score",
            help = "Area Under the ROC Curve (AUC), indicating classifier performance (higher is better)"
        )
    }

    # Display the styled table with explanations
    st.dataframe(class_metric_df, column_config = column_explanations)

    # ---- CONFUSION MATRIX ----

    conf_matrix = np.array([[hit, miss], [fa, cr]])
    df_cm = pd.DataFrame(conf_matrix, columns = ["Go", "No-Go"],
                         index = ["Go", "No-Go"])

    # Heatmap with text overlay
    base = alt.Chart(df_cm.reset_index().melt(id_vars = "index")).encode(
        x = alt.X("variable:N", title = "Predicted Label", axis = alt.Axis(labelAngle = 0)),
        y = alt.Y("index:N", title = "True Label")
    )

    # Heatmap
    heatmap = base.mark_rect().encode(
        color = "value:Q",
        tooltip = ["index", "variable", "value"]
    )

    # Text annotations
    text = base.mark_text(size = 20, font="Source Sans Pro", color = "black").encode(
        text = "value:Q"
    )

    confusion_chart = (heatmap + text).properties(
        width = 400,
        height = 300
    )


    # ---- ROC CURVE ----
    fpr_vals, tpr_vals, _ = roc_curve(y_true, y_scores)

    df_roc = pd.DataFrame({
        "False Positive Rate": fpr_vals,
        "True Positive Rate":  tpr_vals
    })

    roc_chart = alt.Chart(df_roc).mark_line().encode(
        x = "False Positive Rate:Q",
        y = "True Positive Rate:Q",
        tooltip = ["False Positive Rate", "True Positive Rate"]
    ).properties(
        width = 300,
        height = 300
    )

    diagonal = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(strokeDash = [5, 5],
                                                                             color = "gray").encode(
        x = "x:Q",
        y = "y:Q"
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        st.altair_chart(confusion_chart, use_container_width = False)
    with col2:
        st.altair_chart(roc_chart + diagonal, use_container_width = False)

    return class_metric_df