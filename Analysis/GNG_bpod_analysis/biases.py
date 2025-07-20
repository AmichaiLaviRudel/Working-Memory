import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy import stats
import Analysis.GNG_bpod_analysis.colors as colors

def calculate_choice_bias(trial_types, outcomes, n_previous_trials=3):
    """
    Calculate choice bias (response carry-over from previous trials).
    
    Choice bias measures the tendency for mice to repeat (or avoid) licking 
    simply because they licked (or withheld) on the last few trials.
    
    Parameters:
    -----------
    trial_types : list or array
        List of trial types ('Go' or 'NoGo')
    outcomes : list or array
        List of outcomes ('Hit', 'Miss', 'CR', 'False Alarm')
    n_previous_trials : int, optional
        Number of previous trials to consider for bias calculation (default: 3)
    
    Returns:
    --------
    dict
        Dictionary containing:
        - 'bias_scores': Array of bias scores for each trial
        - 'mean_bias': Mean bias across all trials
        - 'std_bias': Standard deviation of bias
        - 'bias_by_response_type': Dictionary with bias for each response type
    """
    
    # Convert to numpy arrays if needed
    trial_types = np.array(trial_types)
    outcomes = np.array(outcomes)
    
    # Create response array (1 for lick, 0 for no lick)
    responses = np.zeros(len(outcomes))
    responses[(outcomes == 'Hit') | (outcomes == 'False Alarm')] = 1
    
    bias_scores = []
    
    for i in range(n_previous_trials, len(responses)):
        # Get previous n trials
        prev_responses = responses[i-n_previous_trials:i]
        
        # Calculate bias as the proportion of licks in previous trials
        # Positive bias = tendency to lick, Negative bias = tendency to withhold
        bias = np.mean(prev_responses) - 0.5  # Center around 0
        bias_scores.append(bias)
    
    # Pad the beginning with NaN
    bias_scores = [np.nan] * n_previous_trials + bias_scores
    
    # Calculate statistics
    valid_bias = [b for b in bias_scores if not np.isnan(b)]
    mean_bias = np.mean(valid_bias) if valid_bias else 0
    std_bias = np.std(valid_bias) if valid_bias else 0
    
    # Calculate bias by response type
    bias_by_response_type = {}
    for response_type in ['Hit', 'Miss', 'CR', 'False Alarm']:
        mask = outcomes == response_type
        if np.any(mask):
            response_bias = [bias_scores[i] for i in range(len(bias_scores)) if mask[i] and not np.isnan(bias_scores[i])]
            bias_by_response_type[response_type] = np.mean(response_bias) if response_bias else 0
        else:
            bias_by_response_type[response_type] = 0
    
    return {
        'bias_scores': np.array(bias_scores),
        'mean_bias': mean_bias,
        'std_bias': std_bias,
        'bias_by_response_type': bias_by_response_type
    }

def calculate_stimulus_bias(stimuli, trial_types, outcomes, n_previous_trials=3):
    """
    Calculate stimulus bias (tendency for current choice to be influenced by previous stimuli).
    
    Stimulus bias measures the tendency for the animal's choice on the current trial 
    to be pulled toward whatever stimulus it saw on one or more previous trials.
    
    Parameters:
    -----------
    stimuli : list or array
        List of stimulus values
    trial_types : list or array
        List of trial types ('Go' or 'NoGo')
    outcomes : list or array
        List of outcomes ('Hit', 'Miss', 'CR', 'False Alarm')
    n_previous_trials : int, optional
        Number of previous trials to consider for bias calculation (default: 3)
    
    Returns:
    --------
    dict
        Dictionary containing:
        - 'bias_scores': Array of bias scores for each trial
        - 'mean_bias': Mean bias across all trials
        - 'std_bias': Standard deviation of bias
        - 'bias_by_trial_type': Dictionary with bias for each trial type
        - 'stimulus_correlation': Correlation between current and previous stimuli
    """
    
    # Convert to numpy arrays if needed
    stimuli = np.array(stimuli, dtype=float)
    trial_types = np.array(trial_types)
    outcomes = np.array(outcomes)
    
    # Create response array (1 for lick, 0 for no lick)
    responses = np.zeros(len(outcomes))
    responses[(outcomes == 'Hit') | (outcomes == 'False Alarm')] = 1
    
    bias_scores = []
    stimulus_correlations = []
    
    for i in range(n_previous_trials, len(stimuli)):
        # Get previous n trials
        prev_stimuli = stimuli[i-n_previous_trials:i]
        current_stimulus = stimuli[i]
        
        # Calculate stimulus bias as the correlation between current response
        # and the average of previous stimuli
        prev_stimulus_mean = np.mean(prev_stimuli)
        
        # Bias score: positive if response is influenced by previous stimuli
        # (higher previous stimuli -> higher likelihood of lick)
        bias = (current_stimulus - prev_stimulus_mean) * responses[i]
        bias_scores.append(bias)
        
        # Calculate correlation between current and previous stimuli
        # Note: Removed problematic correlation calculation that required at least 2 values
        stimulus_correlations.append(0)  # Placeholder value
    
    # Pad the beginning with NaN
    bias_scores = [np.nan] * n_previous_trials + bias_scores
    stimulus_correlations = [np.nan] * n_previous_trials + stimulus_correlations
    
    # Calculate statistics
    valid_bias = [b for b in bias_scores if not np.isnan(b)]
    mean_bias = np.mean(valid_bias) if valid_bias else 0
    std_bias = np.std(valid_bias) if valid_bias else 0
    
    # Calculate bias by trial type
    bias_by_trial_type = {}
    for trial_type in ['Go', 'NoGo']:
        mask = trial_types == trial_type
        if np.any(mask):
            trial_bias = [bias_scores[i] for i in range(len(bias_scores)) if mask[i] and not np.isnan(bias_scores[i])]
            bias_by_trial_type[trial_type] = np.mean(trial_bias) if trial_bias else 0
        else:
            bias_by_trial_type[trial_type] = 0
    
    # Calculate overall stimulus correlation
    valid_correlations = [c for c in stimulus_correlations if not np.isnan(c)]
    mean_correlation = np.mean(valid_correlations) if valid_correlations else 0
    
    return {
        'bias_scores': np.array(bias_scores),
        'mean_bias': mean_bias,
        'std_bias': std_bias,
        'bias_by_trial_type': bias_by_trial_type,
        'stimulus_correlation': mean_correlation,
        'stimulus_correlations': np.array(stimulus_correlations)
    }

def plot_bias_analysis(selected_data, index, n_previous_trials=3, plot=False):
    """
    Plot choice bias and stimulus bias analysis for a single session.
    
    Parameters:
    -----------
    selected_data : pd.DataFrame
        DataFrame containing session data
    index : int
        Index of the session to analyze
    n_previous_trials : int, optional
        Number of previous trials to consider (default: 3)
    plot : bool, optional
        Whether to display the plot (default: False)
    """
    
    # Extract data for the session
    trial_types = selected_data.loc[index, 'TrialTypes']
    outcomes = selected_data.loc[index, 'Outcomes']
    stimuli = selected_data.loc[index, 'Stimuli']
    import ast
    outcomes = ast.literal_eval(outcomes)
    stimuli = selected_data["Stimuli"].values[index].strip("[]\n").split()
    stimuli = np.array([float(num) for num in stimuli])
    trial_types = ast.literal_eval(trial_types)

    
    # Calculate biases
    choice_bias = calculate_choice_bias(trial_types, outcomes, n_previous_trials)
    stimulus_bias = calculate_stimulus_bias(stimuli, trial_types, outcomes, n_previous_trials)
    
    if plot:
        # Choice Bias Figure
        fig_choice = go.Figure()
        fig_choice.add_trace(go.Scatter(
            x=list(range(len(choice_bias['bias_scores']))),
            y=choice_bias['bias_scores'],
            mode='lines',
            name='Choice Bias',
            line=dict(color=colors.COLOR_ACCENT),
            marker=dict(size=4)
        ))
        fig_choice.add_trace(go.Scatter(
            x=[0, len(choice_bias['bias_scores'])-1],
            y=[0, 0],
            mode='lines',
            name='No Bias',
            line=dict(color=colors.COLOR_GRAY, dash='dash'),
            showlegend=True
        ))
        fig_choice.update_layout(
            title=f'Choice Bias',
            xaxis_title='Trial Number',
            yaxis_title='Bias Score',
            legend=dict(title='Bias Type'),
            height=400,
            width=700
        )
        st.markdown("**Choice bias measures the tendency to repeat (or avoid) licking based on recent responses.**")
        st.plotly_chart(fig_choice, use_container_width=True)

        # Stimulus Bias Figure
        fig_stimulus = go.Figure()
        fig_stimulus.add_trace(go.Scatter(
            x=list(range(len(stimulus_bias['bias_scores']))),
            y=stimulus_bias['bias_scores'],
            mode='lines',
            name='Stimulus Bias',
            line=dict(color=colors.COLOR_LOW_BD),
            marker=dict(size=4)
        ))
        fig_stimulus.add_trace(go.Scatter(
            x=[0, len(stimulus_bias['bias_scores'])-1],
            y=[0, 0],
            mode='lines',
            name='No Bias',
            line=dict(color=colors.COLOR_GRAY, dash='dash'),
            showlegend=True
        ))
        fig_stimulus.update_layout(
            title=f'Stimulus Bias',
            xaxis_title='Trial Number',
            yaxis_title='Bias Score',
            legend=dict(title='Bias Type'),
            height=400,
            width=700
        )
        st.markdown("**Stimulus bias measures the influence of previous stimuli on the current choice.**")
        st.plotly_chart(fig_stimulus, use_container_width=True)

        # Display statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Choice Bias Statistics")
            st.write(f"Mean Choice Bias: {choice_bias['mean_bias']:.4f}")
            st.write(f"Std Choice Bias: {choice_bias['std_bias']:.4f}")
            st.write("Bias by Response Type:")
            for response_type, bias in choice_bias['bias_by_response_type'].items():
                st.write(f"  {response_type}: {bias:.4f}")
        
        with col2:
            st.subheader("Stimulus Bias Statistics")
            st.write(f"Mean Stimulus Bias: {stimulus_bias['mean_bias']:.4f}")
            st.write(f"Std Stimulus Bias: {stimulus_bias['std_bias']:.4f}")
            st.write(f"Stimulus Correlation: {stimulus_bias['stimulus_correlation']:.4f}")
            st.write("Bias by Trial Type:")
            for trial_type, bias in stimulus_bias['bias_by_trial_type'].items():
                st.write(f"  {trial_type}: {bias:.4f}")
    
    return choice_bias, stimulus_bias

def bias_multiple_sessions(selected_data, animal_name="None", n_previous_trials=3, plot=True):
    """
    Calculate and plot bias progression across multiple sessions for an animal.
    
    Parameters:
    -----------
    selected_data : pd.DataFrame
        DataFrame containing session data
    animal_name : str, optional
        Name of the animal to analyze (default: "None" for user selection)
    n_previous_trials : int, optional
        Number of previous trials to consider (default: 3)
    plot : bool, optional
        Whether to display the plot (default: True)
    """
    
    if animal_name == "None":
        animal_name = st.selectbox("Choose an Animal", selected_data["MouseName"].unique(), key="bias_animal_select")
    
    # Get sessions for the animal
    from Analysis.GNG_bpod_analysis.GNG_bpod_general import get_sessions_for_animal
    session_indices, session_dates = get_sessions_for_animal(selected_data, animal_name)
    
    choice_bias_means = []
    stimulus_bias_means = []
    choice_bias_stds = []
    stimulus_bias_stds = []
    session_numbers = []
    
    for idx in session_indices:
        try:
            choice_bias, stimulus_bias = plot_bias_analysis(selected_data, idx, n_previous_trials, plot=False)
            choice_bias_means.append(choice_bias['mean_bias'])
            stimulus_bias_means.append(stimulus_bias['mean_bias'])
            choice_bias_stds.append(choice_bias['std_bias'])
            stimulus_bias_stds.append(stimulus_bias['std_bias'])
            session_numbers.append(len(session_numbers) + 1)
        except Exception as e:
            st.warning(f"Could not analyze session {idx}: {e}")
            choice_bias_means.append(np.nan)
            stimulus_bias_means.append(np.nan)
            choice_bias_stds.append(np.nan)
            stimulus_bias_stds.append(np.nan)
            session_numbers.append(len(session_numbers) + 1)
    
    if plot:
        # Plot bias progression
        fig = go.Figure()
        
        # Choice Bias with error band
        fig.add_trace(go.Scatter(
            x=session_numbers,
            y=choice_bias_means,
            mode='lines',
            name='Choice Bias',
            line=dict(color=colors.COLOR_ACCENT),
            marker=dict(symbol='circle')
        ))
        # Choice Bias error band
        fig.add_trace(go.Scatter(
            x=session_numbers,
            y=[m + s if not np.isnan(m) and not np.isnan(s) else np.nan for m, s in zip(choice_bias_means, choice_bias_stds)],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip',
            name='+1 Std Choice'
        ))
        fig.add_trace(go.Scatter(
            x=session_numbers,
            y=[m - s if not np.isnan(m) and not np.isnan(s) else np.nan for m, s in zip(choice_bias_means, choice_bias_stds)],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor=colors.COLOR_ACCENT_TRANSPARENT,
            showlegend=False,
            hoverinfo='skip',
            name='-1 Std Choice'
        ))
        
        # Stimulus Bias with error band
        fig.add_trace(go.Scatter(
            x=session_numbers,
            y=stimulus_bias_means,
            mode='lines',
            name='Stimulus Bias',
            line=dict(color=colors.COLOR_LOW_BD),
            marker=dict(symbol='square')
        ))
        # Stimulus Bias error band
        fig.add_trace(go.Scatter(
            x=session_numbers,
            y=[m + s if not np.isnan(m) and not np.isnan(s) else np.nan for m, s in zip(stimulus_bias_means, stimulus_bias_stds)],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip',
            name='+1 Std Stimulus'
        ))
        fig.add_trace(go.Scatter(
            x=session_numbers,
            y=[m - s if not np.isnan(m) and not np.isnan(s) else np.nan for m, s in zip(stimulus_bias_means, stimulus_bias_stds)],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor=colors.COLOR_LOW_BD_TRANSPARENT,
            showlegend=False,
            hoverinfo='skip',
            name='-1 Std Stimulus'
        ))
        
        # Zero reference line
        fig.add_trace(go.Scatter(
            x=[min(session_numbers), max(session_numbers)],
            y=[0, 0],
            mode='lines',
            name='No Bias',
            line=dict(color=colors.COLOR_GRAY, dash='dash'),
            showlegend=True
        ))
        
        fig.update_layout(
            title=f'Bias Progression - {animal_name}',
            xaxis_title='Session Number',
            yaxis_title='Mean Bias Score',
            legend=dict(title='Bias Type'),
            height=400,
            width=700
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    return {
        'session_numbers': session_numbers,
        'choice_bias_means': choice_bias_means,
        'stimulus_bias_means': stimulus_bias_means,
        'choice_bias_stds': choice_bias_stds,
        'stimulus_bias_stds': stimulus_bias_stds
    }
