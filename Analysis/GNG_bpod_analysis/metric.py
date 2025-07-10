from Analysis.GNG_bpod_analysis.psychometric_curves import *
from Analysis.GNG_bpod_analysis.GNG_bpod_general import *
import Analysis.GNG_bpod_analysis.colors as colors

import numpy as np
import pandas as pd
import altair as alt
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import streamlit as st
import plotly.graph_objects as go
import ast

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
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['index'], y=df['d_prime'], mode='lines',
            name="Overall",
            line=dict(color=colors.COLOR_ACCENT),
            hovertemplate="Trial: %{x}<br>d': %{y:.3f}<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=[df['index'].min(), df['index'].max()], y=[1, 1],
            mode='lines', name="Learning Threshold",
            line=dict(color=colors.COLOR_GRAY, dash='dash'),
            hoverinfo='skip', showlegend=True
        ))
        fig.update_layout(
            xaxis_title="Trial Index",
            yaxis_title="d'",
            title="d' over trials",
            legend=dict(title="Legend"),
            height=400,
            width=700
        )
        st.plotly_chart(fig, use_container_width=True)
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

    # For low/high boundary overlays
    low_boundary_means = []
    high_boundary_means = []
    low_boundary_stds = []
    high_boundary_stds = []

    # Compute d' statistics and collect metadata
    for idx, sess_idx in enumerate(session_indices):
        d = d_prime(selected_data, index=sess_idx, t=t)
        ds[idx, 0] = np.nanmean(d)
        ds[idx, 1] = np.nanstd(d)
        ds[idx, 2] = np.nanmax(d)
        # Retrieve session metadata
        tones_per_class.append(selected_data.loc[sess_idx, 'Tones_per_class'])
        boundaries.append(selected_data.loc[sess_idx, 'N_Boundaries'])

        # If N_Boundaries==2, get low/high boundary d' for this session
        if selected_data.loc[sess_idx, 'N_Boundaries'] == 2:
            df_low, df_high = d_prime_low_high_boundary_sessions(selected_data.loc[[sess_idx]], sess_idx, t=t, plot=False)
            # Store mean d' for each boundary for this session
            low_boundary_means.append(np.nanmean(df_low['d_prime']))
            high_boundary_means.append(np.nanmean(df_high['d_prime']))
            # Store std d' for each boundary for this session
            low_boundary_stds.append(np.nanstd(df_low['d_prime']))
            high_boundary_stds.append(np.nanstd(df_high['d_prime']))
        else:
            low_boundary_means.append(np.nan)
            high_boundary_means.append(np.nan)
            low_boundary_stds.append(np.nan)
            high_boundary_stds.append(np.nan)

    # Build DataFrame for plotting
    data = pd.DataFrame({
        'Session Index':   np.arange(1, len(session_indices) + 1),
        'SessionDate':     session_dates,
        'd_prime': ds[:, 0],
        'Error': ds[:, 1],
        'Max_d_prime': ds[:, 2],
        'tones_per_class': tones_per_class,
        'Boundaries':      boundaries,
        'Low Boundary d_prime': low_boundary_means,
        'High Boundary d_prime': high_boundary_means,
        'Low Boundary Error': low_boundary_stds,
        'High Boundary Error': high_boundary_stds
    })

    # Clean up isolated std values (replace with NaN if surrounded by NaN)
    for col in ['Low Boundary Error', 'High Boundary Error']:
        for i in range(1, len(data) - 1):
            if pd.isna(data.loc[i, col]):
                continue
            if pd.isna(data.loc[i-1, col]) and pd.isna(data.loc[i+1, col]):
                data.loc[i, col] = np.nan

    if plot:
        import plotly.graph_objects as go
        fig = go.Figure()
        # Main overall d' line
        fig.add_trace(go.Scatter(
            x=data['Session Index'], y=data['d_prime'], mode='lines',
            name="Overall d'",
            line=dict(color=colors.COLOR_ACCENT),
            marker=dict(symbol='circle')
        ))
        # Overall d' error band
        fig.add_trace(go.Scatter(
            x=data['Session Index'], y=data['d_prime'] + data['Error'],
            mode='lines', line=dict(width=0),  # invisible line
            showlegend=False,
            hoverinfo='skip',
            name='+1 Std'
        ))
        fig.add_trace(go.Scatter(
            x=data['Session Index'], y=data['d_prime'] - data['Error'],
            mode='lines', line=dict(width=0),  # invisible line
            fill='tonexty',
            fillcolor=colors.COLOR_ACCENT_TRANSPARENT,  # transparent version of COLOR_SUBTLE
            showlegend=False,
            hoverinfo='skip',
            name='-1 Std'
        ))
        # Low boundary overlay and error band
        fig.add_trace(go.Scatter(
            x=data['Session Index'], y=data['Low Boundary d_prime'],
            mode='lines+markers', name="Low Boundary d'",
            line=dict(color=colors.COLOR_LOW_BD),
            marker=dict(symbol='triangle-up')
        ))
        fig.add_trace(go.Scatter(
            x=data['Session Index'], y=data['Low Boundary d_prime'] + data['Low Boundary Error'],
            mode='lines', line=dict(width=0),
            showlegend=False,
            hoverinfo='skip',
            name='+1 Std Low',
        ))
        fig.add_trace(go.Scatter(
            x=data['Session Index'], y=data['Low Boundary d_prime'] - data['Low Boundary Error'],
            mode='lines', line=dict(width=0),
            fill='tonexty',
            fillcolor=colors.COLOR_LOW_BD_TRANSPARENT,  
            showlegend=False,
            hoverinfo='skip',
            name='-1 Std Low',
        ))

        # High boundary overlay and error band
        fig.add_trace(go.Scatter(
            x=data['Session Index'], y=data['High Boundary d_prime'],
            mode='lines+markers', name="High Boundary d'",
            line=dict(color=colors.COLOR_HIGH_BD),
            marker=dict(symbol='triangle-down')
        ))
        fig.add_trace(go.Scatter(
            x=data['Session Index'], y=data['High Boundary d_prime'] + data['High Boundary Error'],
            mode='lines', line=dict(width=0),
            showlegend=False,
            hoverinfo='skip',
            name='+1 Std High',
        ))
        fig.add_trace(go.Scatter(
            x=data['Session Index'], y=data['High Boundary d_prime'] - data['High Boundary Error'],
            mode='lines', line=dict(width=0),
            fill='tonexty',
            fillcolor=colors.COLOR_HIGH_BD_TRANSPARENT,
            showlegend=False,
            hoverinfo='skip',
            name='-1 Std High',
        ))

        # Learning threshold
        fig.add_trace(go.Scatter(
            x=[data['Session Index'].min(), data['Session Index'].max()], y=[1, 1],
            mode='lines', name="Learning Threshold",
            line=dict(color=colors.COLOR_GRAY, dash='dash'),
            hoverinfo='skip', showlegend=True
        ))
        fig.update_layout(
            xaxis_title="Session Index",
            yaxis_title="d'",
            title=f"d' Progress for {animal_name}",
            legend=dict(title="Legend"),
            height=400,
            width=700
        )
        st.plotly_chart(fig, use_container_width=True)

    return data

def multi_animal_d_prime_progression(selected_data, N_Boundaries = None):

    if N_Boundaries != None:
        st.title(f"Multi-Animal Progress {N_Boundaries} boudaries")
        selected_data = selected_data[selected_data["N_Boundaries"] == N_Boundaries].reset_index(drop = True)
    else:
        st.title("Multi-Animal Progress")

    # Get unique subject names
    subjects = selected_data["MouseName"].unique()

    # Initialize arrays to store d' values
    d_prime_data = []
    session_counts = []
    low_boundary_data = []
    high_boundary_data = []

    for subject in subjects:
        # Compute d' for each subject
        d_prime_result = d_prime_multiple_sessions(selected_data, animal_name=subject, plot=False)
        d_prime_values = d_prime_result["d_prime"]
        d_prime_data.append(d_prime_values)
        session_counts.append(len(d_prime_values))
        # If N_Boundaries == 2, collect low/high boundary d' as well
        if N_Boundaries == 2:
            low_vals = d_prime_result['Low Boundary d_prime']
            high_vals = d_prime_result['High Boundary d_prime']
            low_boundary_data.append(np.array(low_vals))
            high_boundary_data.append(np.array(high_vals))

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
    for session, d_prime in enumerate(np.asarray(avg_d_prime)):
        if not np.isnan(d_prime):
            avg_list.append({"Session": session + 1, "d_prime": d_prime, "Mouse": "Average"})

    df_altair = pd.DataFrame(data_list)
    df_avg = pd.DataFrame(avg_list)

    # If N_Boundaries == 2, compute average low/high boundary d' across animals
    if N_Boundaries == 2 and low_boundary_data and high_boundary_data:
        low_df = pd.DataFrame([np.pad(d, (0, max_sessions - len(d)), constant_values=np.nan) for d in low_boundary_data])
        high_df = pd.DataFrame([np.pad(d, (0, max_sessions - len(d)), constant_values=np.nan) for d in high_boundary_data])
        avg_low = low_df.mean(axis=0, skipna=True)
        avg_high = high_df.mean(axis=0, skipna=True)

    import plotly.graph_objects as go
    fig = go.Figure()
    # Plot all animals (gray lines)
    for i, subject in enumerate(subjects):
        fig.add_trace(go.Scatter(
            x=np.arange(1, max_sessions + 1),
            y=d_prime_df.iloc[i],
            mode='lines',
            line=dict(color=colors.COLOR_VERY_SUBTLE, width=1),
            name=f'{subject}',
            showlegend=False
        ))
    # Plot average overall d'
    fig.add_trace(go.Scatter(
        x=np.arange(1, max_sessions + 1),
        y=avg_d_prime,
        mode='lines',
        name="Average Overall d'",
        line=dict(color=colors.COLOR_ACCENT, width=7),
        marker=dict(symbol='circle')
    ))
    # Plot average low/high boundary d' if N_Boundaries == 2
    if N_Boundaries == 2 and low_boundary_data and high_boundary_data:
        fig.add_trace(go.Scatter(
            x=np.arange(1, max_sessions + 1),
            y=avg_low,
            mode='lines+markers',
            name="Average Low Boundary d'",
            line=dict(color=colors.COLOR_LOW_BD, width=2, dash='solid'),
            marker=dict(symbol='triangle-up')
        ))
        fig.add_trace(go.Scatter(
            x=np.arange(1, max_sessions + 1),
            y=avg_high,
            mode='lines+markers',
            name="Average High Boundary d'",
            line=dict(color=colors.COLOR_HIGH_BD, width=2, dash='solid'),
            marker=dict(symbol='triangle-down')
        ))
    # Learning threshold
    fig.add_trace(go.Scatter(
        x=[1, max_sessions], y=[1, 1],
        mode='lines', name="Learning Threshold",
        line=dict(color=colors.COLOR_GRAY, dash='dash'),
        hoverinfo='skip', showlegend=True
    ))
    fig.update_layout(
        xaxis_title="Session Index",
        yaxis_title="d'",
        title="d' Progression Across Animals",
        legend=dict(title="Legend"),
        height=400,
        width=700
    )
    st.plotly_chart(fig, use_container_width=True)




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
    df_cm = pd.DataFrame(conf_matrix, columns = pd.Index(["Go", "No-Go"]),
                         index = pd.Index(["Go", "No-Go"]))

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



def d_prime_multiple_sessions_divde_oneNtwo(selected_data, t=10, animal_name='None', plot = True):
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
        stimuli = parse_stimuli(selected_data.loc[sess_idx, 'Stimuli'])
        stimuli_mask = stimuli < st.session_state.high_boundary

        # Filter relevant columns using the same mask
        filtered_trials = selected_data.loc[sess_idx, 'TrialTypes'][stimuli_mask]
        filtered_outcomes = selected_data.loc[sess_idx, 'Outcomes'][stimuli_mask]
        selected_data_filtered = selected_data.copy()
        selected_data_filtered.at[sess_idx, 'TrialTypes'] = filtered_trials
        selected_data_filtered.at[sess_idx, 'Outcomes'] = filtered_outcomes

        d = d_prime(selected_data, index=sess_idx, t=t)

    st.table(selected_data)


    return data


def d_prime_low_high_boundary_sessions(selected_data, idx, t=10, plot=True):
    """
    Calculates d' over trials in bins of t for low and high boundary trials (low: Stimuli < st.session_state.high_boundary, high: Stimuli > st.session_state.low_boundary)
    for a single session (selected_data should be a DataFrame with one row).
    Plots both d' curves on the same Plotly figure and returns the d' arrays/DataFrames for both boundaries.
    """

    # Get the stimuli for this session
    stimuli = parse_stimuli(selected_data.loc[idx, 'Stimuli'])
    # Low boundary
    low_mask = stimuli < st.session_state.high_boundary
    # High boundary
    high_mask = stimuli > st.session_state.low_boundary


    # Get the raw values
    trialtypes = selected_data.loc[idx, 'TrialTypes']
    outcomes = selected_data.loc[idx, 'Outcomes']

    # Convert to array if needed
    if isinstance(trialtypes, str):
        trialtypes = to_array(trialtypes)
    if isinstance(outcomes, str):
        outcomes = to_array(outcomes)

    trialtypes = np.array(trialtypes)
    outcomes = np.array(outcomes)

    # Now mask
    filtered_trials_low = trialtypes[low_mask]
    filtered_outcomes_low = outcomes[low_mask]
    filtered_trials_high = trialtypes[high_mask]
    filtered_outcomes_high = outcomes[high_mask]

    selected_data_low = selected_data.copy()
    selected_data_low.at[idx, 'TrialTypes'] = str(filtered_trials_low.tolist())
    selected_data_low.at[idx, 'Outcomes'] = str(filtered_outcomes_low.tolist())

    selected_data_high = selected_data.copy()
    selected_data_high.at[idx, 'TrialTypes'] = str(filtered_trials_high.tolist())
    selected_data_high.at[idx, 'Outcomes'] = str(filtered_outcomes_high.tolist())


    # Calculate d' for low and high boundary
    d_low = d_prime(selected_data_low, index=0, t=t, plot=False)
    d_high = d_prime(selected_data_high, index=0, t=t, plot=False)

    # Prepare DataFrames for plotting/return
    df_low = pd.DataFrame({
        'Bin': range(len(d_low)),
        "d_prime": d_low,
        'Type': 'Low Boundary'
    })
    df_high = pd.DataFrame({
        'Bin': range(len(d_high)),
        "d_prime": d_high,
        'Type': 'High Boundary'
    })

    if plot:
        import plotly.graph_objects as go
        # Display mean and std for each boundary (as plotly annotation)
        mean_low = np.nanmean(df_low['d_prime'])
        std_low = np.nanstd(df_low['d_prime'])
        mean_high = np.nanmean(df_high['d_prime'])
        std_high = np.nanstd(df_high['d_prime'])
        subtitle = (
            f"<span style='color:black'>"
            f"Low Boundary d': {mean_low:.3f}, ± {std_low:.3f} | "
            f"High Boundary d': {mean_high:.3f} ± {std_high:.3f}"
            f"</span>"
        )
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_low['Bin'], y=df_low['d_prime'], mode='lines',
            name="Low Boundary",
            line=dict(color=colors.COLOR_LOW_BD),
            marker=dict(symbol='circle')
        ))
        fig.add_trace(go.Scatter(
            x=df_high['Bin'], y=df_high['d_prime'], mode='lines',
            name="High Boundary",
            line=dict(color=colors.COLOR_HIGH_BD),
            marker=dict(symbol='square')
        ))
        fig.add_trace(go.Scatter(
            x=[0, max(len(d_low), len(d_high)) - 1], y=[1, 1],
            mode='lines', name="Learning Threshold",
            line=dict(color=colors.COLOR_GRAY, dash='dash'),
            hoverinfo='skip', showlegend=True
        ))
        fig.update_layout(
            xaxis_title="Bin Index",
            yaxis_title="d'",
            title=f"d' Progression (Low vs High Boundary)",
            legend=dict(title="Boundary Type"),
            height=400,
            width=700
        )
        fig.add_annotation(
            text=subtitle,
            xref="paper", yref="paper",
            x=0.01, y=1.08,
            showarrow=False,
            font=dict(size=14),
            xanchor='left',
            yanchor='top'
        )
        st.plotly_chart(fig, use_container_width=True)



    return df_low, df_high

def to_array(val):
    if isinstance(val, str):
        try:
            return np.array(ast.literal_eval(val))
        except Exception:
            return np.array([])
    elif isinstance(val, (list, np.ndarray)):
        return np.array(val)
    else:
        return np.array([])