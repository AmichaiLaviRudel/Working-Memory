import plotly.graph_objects as go
from Analysis.GNG_bpod_analysis.colors import COLOR_HIT, COLOR_MISS, COLOR_FA, COLOR_CR, GO_COLORS, NOGO_COLORS
import streamlit as st
import numpy as np
import pandas as pd
from Analysis.NPXL_analysis.population_analysis_advanced import (
    PopulationAnalyzer, StimulusDecoder, ChoiceDecoder, 
    DimensionalityReducer, RepresentationalSimilarityAnalyzer,
    plot_decoding_results, plot_time_resolved_decoding,
    plot_pca_results, plot_umap_results, plot_rsa_results
)

def plot_population_heatmap(event_windows_matrix, stimuli_outcome_df, metadata):
    """
    Plot population heatmap that reconstructs events back into a time stream with navigation.
    
    Args:
        event_windows_matrix: 3D array [units × time_bins_per_window × n_events]
        stimuli_outcome_df: DataFrame with event information including 'time' column
        metadata: Dictionary with window_duration, bin_size, etc.
    """
    
    if len(event_windows_matrix.shape) != 3:
        st.error("Expected 3D event windows matrix [units × time × events]")
        return
    
    n_units, n_time_per_event, n_events = event_windows_matrix.shape
    
    # Get parameters from metadata
    window_duration = float(metadata.get('window_duration', 4.0))
    bin_size = float(metadata.get('bin_size', 0.1))
    
    # Navigation controls
    st.subheader("Heatmap Navigation")
    display_window_duration = st.slider(
        "Display Window Duration (seconds)", 
        min_value=5.0, 
        max_value=60.0, 
        value=20.0, 
        step=1.0,
        help="Duration of the time window to display"
    )
    
    # Get original event times (in bins) from stimuli_outcome_df
    original_event_times = stimuli_outcome_df['time'].values
    
    # Calculate the range of time we need to cover
    min_event_time = np.min(original_event_times)
    max_event_time = np.max(original_event_times)
    
    # Add buffer around events to show context
    buffer_duration = window_duration  # Same as window duration
    buffer_bins = int(buffer_duration / bin_size)
    
    # Calculate the total time range needed
    # Add extra buffer to ensure all events fit completely
    extra_buffer = int(window_duration / bin_size)  # Full window duration as extra buffer
    start_time_bin = int(min_event_time - buffer_bins - extra_buffer)
    end_time_bin = int(max_event_time + buffer_bins + extra_buffer)
    
    total_time_bins = end_time_bin - start_time_bin
    total_duration_seconds = total_time_bins * bin_size
    
    # Navigation slider for time position
    max_start_time = max(0, total_duration_seconds - display_window_duration)
    
    if max_start_time > 0:
        # Add navigation buttons
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            if st.button("⏮️ Start", help="Jump to beginning"):
                st.session_state.heatmap_start_time = 0.0
        
        with col2:
            if st.button("⏪ Back", help="Go back 10 seconds"):
                current_time = st.session_state.get('heatmap_start_time', 0.0)
                st.session_state.heatmap_start_time = max(0.0, current_time - 10.0)
        
        with col3:
            if st.button("⏩ Forward", help="Go forward 10 seconds"):
                current_time = st.session_state.get('heatmap_start_time', 0.0)
                st.session_state.heatmap_start_time = min(max_start_time, current_time + 10.0)
        
        with col4:
            if st.button("⏭️ End", help="Jump to end"):
                st.session_state.heatmap_start_time = max_start_time
        
        with col5:
            if st.button("🔄 Reset", help="Reset to beginning"):
                st.session_state.heatmap_start_time = 0.0
        
        # Initialize session state if not exists
        if 'heatmap_start_time' not in st.session_state:
            st.session_state.heatmap_start_time = 0.0
        
        start_time_position = st.slider(
            "Start Time Position (seconds)",
            min_value=0.0,
            max_value=max_start_time,
            value=st.session_state.heatmap_start_time,
            step=1.0,
            help="Starting position of the display window",
            key="heatmap_time_slider"
        )
        
        # Update session state from slider
        st.session_state.heatmap_start_time = start_time_position
        
    else:
        start_time_position = 0.0
        st.info(f"Recording duration ({total_duration_seconds:.1f}s) is shorter than display window. Showing entire recording.")
    
    # Create the reconstructed matrix
    reconstructed_matrix = np.zeros((n_units, total_time_bins))
    
    # Place each event window in the correct position in the time stream
    # Use the actual window size from the event windows matrix
    actual_window_size = n_time_per_event
    window_half_duration_bins = actual_window_size // 2
    
    for event_idx, original_time in enumerate(original_event_times):
        if event_idx >= n_events:  # Safety check
            break
            
        # Calculate where this event should be placed in the reconstructed matrix
        event_center_in_reconstructed = int(original_time - start_time_bin)
        
        # Calculate start and end positions for this event window
        window_start = event_center_in_reconstructed - window_half_duration_bins
        window_end = window_start + actual_window_size  # Use actual window size
        
        # Ensure we don't go out of bounds and that the slice size matches the data
        if window_start >= 0 and window_end <= total_time_bins:
            # Verify that the slice size matches the event window size
            slice_size = window_end - window_start
            if slice_size == actual_window_size:
                try:
                    # Place the event window in the reconstructed matrix
                    reconstructed_matrix[:, window_start:window_end] = event_windows_matrix[:, :, event_idx]
                except ValueError as e:
                    st.warning(f"Error placing event {event_idx}: {e}. Shapes: reconstructed slice {reconstructed_matrix[:, window_start:window_end].shape}, event window {event_windows_matrix[:, :, event_idx].shape}")
            else:
                st.warning(f"Slice size mismatch for event {event_idx}: expected {actual_window_size}, got {slice_size}")
        else:
            # Try to clip the window to fit within bounds
            clipped_start = max(0, window_start)
            clipped_end = min(total_time_bins, window_end)
            
            # Calculate corresponding indices in the event window
            event_start_offset = clipped_start - window_start
            event_end_offset = event_start_offset + (clipped_end - clipped_start)
            
            if event_start_offset >= 0 and event_end_offset <= actual_window_size:
                try:
                    reconstructed_matrix[:, clipped_start:clipped_end] = event_windows_matrix[:, event_start_offset:event_end_offset, event_idx]
                    st.info(f"Event {event_idx} partially placed (clipped to fit bounds)")
                except ValueError as e:
                    st.warning(f"Error placing clipped event {event_idx}: {e}")
            else:
                st.warning(f"Event {event_idx} completely outside bounds or clipping failed")
    
    # Create time axis for the reconstructed matrix (in seconds relative to recording start)
    time_axis_bins = np.arange(start_time_bin, end_time_bin)
    time_axis_seconds = time_axis_bins * bin_size
    
    # Calculate display window boundaries
    display_start_bin = int(start_time_position / bin_size)
    display_end_bin = int((start_time_position + display_window_duration) / bin_size)
    display_end_bin = min(display_end_bin, total_time_bins)
    
    # Extract the subset of data for the display window
    matrix_subset = reconstructed_matrix[:, display_start_bin:display_end_bin]
    time_subset = time_axis_seconds[display_start_bin:display_end_bin]
    
    # Display information about the current view
    st.info(f"Showing {display_window_duration}s window from {start_time_position:.1f}s to {start_time_position + display_window_duration:.1f}s | Total duration: {total_duration_seconds:.1f}s")
    
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix_subset,
            x=time_subset,
            colorscale='Greys',
            colorbar=dict(title="Spikes/sec"),
        )
    )
    outcome_color_map = {
        "Hit": COLOR_HIT,
        "Miss": COLOR_MISS,
        "False Alarm": COLOR_FA,
        "CR": COLOR_CR,
    }
    
    # Track which outcomes we've already added to legend
    legend_added = set()
    
    # Add event markers at their actual times (only those within the display window)
    if 'time' in stimuli_outcome_df.columns and 'outcome' in stimuli_outcome_df.columns:
        # Calculate display window boundaries in seconds
        display_start_seconds = start_time_position
        display_end_seconds = start_time_position + display_window_duration
        
        for t, outcome in zip(stimuli_outcome_df['time'], stimuli_outcome_df['outcome']):
            color = outcome_color_map.get(outcome, "white")
            
            # Convert event time from bins to seconds for plotting
            event_time_seconds = t * bin_size
            
            # Only show events within the current display window
            if display_start_seconds <= event_time_seconds <= display_end_seconds:
                # Show legend only once per outcome type
                showlegend = outcome not in legend_added
                if showlegend:
                    legend_added.add(outcome)
                
                fig.add_vline(
                    x=event_time_seconds, 
                    line_width=2, 
                    line_dash="solid", 
                    line_color=color,
                    name=outcome,
                    showlegend=showlegend
                )
    
    fig.update_layout(
        xaxis=dict(
            title="Time in Recording (s)",
            constrain='domain'
        ),
        yaxis_title="Unit",
        title=f"Population Activity Heatmap | {start_time_position:.1f}s - {start_time_position + display_window_duration:.1f}s",
        legend=dict(
            title="Events by Outcome",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    fig.update_layout(height=800)
    st.plotly_chart(fig, use_container_width=True)


def advanced_population_analysis_panel(event_windows_matrix, stimuli_outcome_df, metadata):
    """
    Advanced population analysis panel with decoding and other analyses.
    
    Args:
        event_windows_matrix: 3D array [units × time_bins × trials]
        stimuli_outcome_df: DataFrame with trial information
        metadata: Dictionary with recording parameters
    """
    st.header("🧠 Advanced Population Analysis")
    
    # Create analyzer instances
    try:
        analyzer = PopulationAnalyzer(event_windows_matrix, stimuli_outcome_df, metadata)
        stimulus_decoder = StimulusDecoder(event_windows_matrix, stimuli_outcome_df, metadata)
        choice_decoder = ChoiceDecoder(event_windows_matrix, stimuli_outcome_df, metadata)
        
        # Analysis type selection
        analysis_type = st.selectbox(
            "Select Analysis Type",
            ["Stimulus Decoding", "Choice Decoding", "Time-Resolved Decoding", 
             "Dimensionality Reduction", "Representational Similarity", "Population Summary"]
        )
        
        if analysis_type == "Stimulus Decoding":
            _stimulus_decoding_analysis(stimulus_decoder)
        elif analysis_type == "Choice Decoding":
            _choice_decoding_analysis(choice_decoder)
        elif analysis_type == "Time-Resolved Decoding":
            _time_resolved_analysis(stimulus_decoder, choice_decoder)
        elif analysis_type == "Dimensionality Reduction":
            dimensionality_reducer = DimensionalityReducer(event_windows_matrix, stimuli_outcome_df, metadata)
            _dimensionality_reduction_analysis(dimensionality_reducer)
        elif analysis_type == "Representational Similarity":
            rsa_analyzer = RepresentationalSimilarityAnalyzer(event_windows_matrix, stimuli_outcome_df, metadata)
            _rsa_analysis(rsa_analyzer)
        elif analysis_type == "Population Summary":
            _population_summary(analyzer)
            
    except Exception as e:
        st.error(f"Error initializing advanced analysis: {e}")
        st.info("Please ensure your data has the required columns: 'stimulus', 'outcome'")


def _stimulus_decoding_analysis(stimulus_decoder):
    """Stimulus decoding analysis panel."""
    st.subheader("🎯 Stimulus Decoding Analysis")
    
    if stimulus_decoder.stimulus_labels is None:
        st.warning("No stimulus labels found in the data.")
        return
    
    # Parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        start_time = st.number_input("Start Time (s)", value=-0.1, step=0.1)
    with col2:
        end_time = st.number_input("End Time (s)", value=0.5, step=0.1)
    with col3:
        classifier = st.selectbox("Classifier", ["logistic", "svm", "lda"])
    
    time_window = (start_time, end_time)
    
    if st.button("Run Stimulus Decoding"):
        with st.spinner("Running stimulus decoding analysis..."):
            try:
                results = stimulus_decoder.decode_stimuli(
                    time_window=time_window,
                    classifier=classifier,
                    cv_folds=5
                )
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Mean Accuracy", f"{results['mean_accuracy']:.3f}")
                with col2:
                    st.metric("Std Accuracy", f"{results['std_accuracy']:.3f}")
                with col3:
                    st.metric("N Stimuli", results['n_stimuli'])
                
                # Plot results
                actual_values = results.get('actual_stimulus_values', None)
                fig = plot_decoding_results(results, "Stimulus Decoding Results", actual_values)
                st.plotly_chart(fig, use_container_width=True)
                
                # Show classification report
                st.subheader("Classification Report")
                report_df = pd.DataFrame(results['classification_report']).transpose()
                st.dataframe(report_df)
                
            except Exception as e:
                st.error(f"Error in stimulus decoding: {e}")


def _choice_decoding_analysis(choice_decoder):
    """Choice decoding analysis panel."""
    st.subheader("🔄 Choice Decoding Analysis")
    
    if choice_decoder.choice_labels is None:
        st.warning("No choice labels found in the data.")
        return
    
    # Parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        start_time = st.number_input("Start Time (s)", value=-0.1, step=0.1, key="choice_start")
    with col2:
        end_time = st.number_input("End Time (s)", value=0.5, step=0.1, key="choice_end")
    with col3:
        classifier = st.selectbox("Classifier", ["logistic", "svm", "lda"], key="choice_classifier")
    
    time_window = (start_time, end_time)
    
    if st.button("Run Choice Decoding"):
        with st.spinner("Running choice decoding analysis..."):
            try:
                results = choice_decoder.decode_choice(
                    time_window=time_window,
                    classifier=classifier,
                    cv_folds=5
                )
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Mean Accuracy", f"{results['mean_accuracy']:.3f}")
                with col2:
                    st.metric("Std Accuracy", f"{results['std_accuracy']:.3f}")
                with col3:
                    chance_level = 1.0 / len(results['choice_types'])
                    st.metric("Chance Level", f"{chance_level:.3f}")
                
                # Plot results
                choice_values = results.get('choice_types', None)
                fig = plot_decoding_results(results, "Choice Decoding Results", choice_values)
                st.plotly_chart(fig, use_container_width=True)
                
                # Show classification report
                st.subheader("Classification Report")
                report_df = pd.DataFrame(results['classification_report']).transpose()
                st.dataframe(report_df)
                
            except Exception as e:
                st.error(f"Error in choice decoding: {e}")


def _time_resolved_analysis(stimulus_decoder, choice_decoder):
    """Time-resolved decoding analysis panel."""
    st.subheader("⏱️ Time-Resolved Decoding")
    
    # Parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        window_size = st.number_input("Window Size (s)", value=0.2, step=0.05)
    with col2:
        step_size = st.number_input("Step Size (s)", value=0.05, step=0.01)
    with col3:
        classifier = st.selectbox("Classifier", ["logistic", "svm", "lda"], key="time_classifier")
    
    analysis_options = []
    if stimulus_decoder.stimulus_labels is not None:
        analysis_options.append("Stimulus")
    if choice_decoder.choice_labels is not None:
        analysis_options.append("Choice")
    
    if not analysis_options:
        st.warning("No valid labels found for time-resolved analysis.")
        return
    
    selected_analysis = st.multiselect("Analysis Type", analysis_options, default=analysis_options)
    
    if st.button("Run Time-Resolved Analysis"):
        with st.spinner("Running time-resolved analysis..."):
            try:
                fig = go.Figure()
                
                if "Stimulus" in selected_analysis and stimulus_decoder.stimulus_labels is not None:
                    stimulus_results = stimulus_decoder.time_resolved_stimulus_decoding(
                        window_size=window_size,
                        step_size=step_size,
                        classifier=classifier
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=stimulus_results['time_centers'],
                            y=stimulus_results['accuracies'],
                            mode='lines+markers',
                            name='Stimulus Decoding',
                            line=dict(color='blue', width=2)
                        )
                    )
                
                if "Choice" in selected_analysis and choice_decoder.choice_labels is not None:
                    choice_results = choice_decoder.time_resolved_choice_decoding(
                        window_size=window_size,
                        step_size=step_size,
                        classifier=classifier
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=choice_results['time_centers'],
                            y=choice_results['accuracies'],
                            mode='lines+markers',
                            name='Choice Decoding',
                            line=dict(color='red', width=2)
                        )
                    )
                
                # Add reference lines with proper chance levels
                if "Stimulus" in selected_analysis and stimulus_decoder.stimulus_labels is not None:
                    stimulus_chance = 1.0 / len(stimulus_decoder.unique_stimuli) if stimulus_decoder.unique_stimuli is not None else 0.5
                elif "Choice" in selected_analysis and choice_decoder.choice_labels is not None:
                    choice_chance = 0.5  # Binary Go/NoGo
                else:
                    chance_level = 0.5
                
                # Use the appropriate chance level based on the first analysis
                if "Stimulus" in selected_analysis and stimulus_decoder.stimulus_labels is not None:
                    chance_level = 1.0 / len(stimulus_decoder.unique_stimuli)
                else:
                    chance_level = 0.5
                    
                fig.add_hline(y=chance_level, line_dash="dash", line_color="gray", 
                             annotation_text=f"Chance Level ({chance_level:.1%})")
                fig.add_vline(x=0, line_dash="dash", line_color="black", 
                             annotation_text="Event Onset")
                
                fig.update_layout(
                    title="Time-Resolved Decoding Analysis",
                    xaxis_title="Time from Event (s)",
                    yaxis_title="Decoding Accuracy",
                    yaxis=dict(range=[0, 1]),
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error in time-resolved analysis: {e}")


def _population_summary(analyzer):
    """Population summary panel."""
    st.subheader("📊 Population Summary")
    
    # Basic statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Units", analyzer.n_units)
    with col2:
        st.metric("Total Trials", analyzer.n_trials)
    with col3:
        st.metric("Time Bins", analyzer.n_time_bins)
    with col4:
        st.metric("Duration (s)", f"{analyzer.window_duration * 2:.1f}")
    
    # Data availability
    st.subheader("Available Data")
    
    data_status = {
        "Stimulus Labels": "✅" if analyzer.stimulus_labels is not None else "❌",
        "Choice Labels": "✅" if analyzer.choice_labels is not None else "❌", 
        "Reward Labels": "✅" if analyzer.reward_labels is not None else "❌"
    }
    
    for label, status in data_status.items():
        st.write(f"{status} {label}")
    
    # Sample distribution
    if analyzer.stimulus_labels is not None:
        st.subheader("Stimulus Distribution")
        stimulus_counts = pd.Series(analyzer.stimulus_labels).value_counts()
        fig = go.Figure(data=[go.Bar(x=stimulus_counts.index, y=stimulus_counts.values)])
        fig.update_layout(title="Trials per Stimulus", xaxis_title="Stimulus", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
    
    if analyzer.choice_labels is not None:
        st.subheader("Choice Distribution")
        choice_counts = pd.Series(analyzer.choice_labels).value_counts()
        fig = go.Figure(data=[go.Bar(x=choice_counts.index, y=choice_counts.values)])
        fig.update_layout(title="Trials per Choice Type", xaxis_title="Choice", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)


def _dimensionality_reduction_analysis(dimensionality_reducer):
    """Dimensionality reduction analysis panel."""
    st.subheader("🔍 Dimensionality Reduction Analysis")
    
    # Method selection
    method = st.selectbox("Reduction Method", ["PCA", "UMAP"])
    
    # Parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        start_time = st.number_input("Start Time (s)", value=-0.5, step=0.1, key="dimred_start")
    with col2:
        end_time = st.number_input("End Time (s)", value=1.0, step=0.1, key="dimred_end")
    with col3:
        n_components = st.number_input("N Components", value=10 if method == "PCA" else 2, 
                                     min_value=2, max_value=20, key="dimred_components")
    
    time_window = (start_time, end_time)
    
    if method == "PCA":
        if st.button("Run PCA Analysis"):
            with st.spinner("Running PCA analysis..."):
                try:
                    results = dimensionality_reducer.compute_pca(
                        time_window=time_window,
                        n_components=n_components
                    )
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Components", len(results['explained_variance_ratio']))
                    with col2:
                        var_explained = results['explained_variance_ratio'][:3].sum()
                        st.metric("Variance (PC1-3)", f"{var_explained:.1%}")
                    with col3:
                        st.metric("Total Variance", f"{results['cumulative_variance'][-1]:.1%}")
                    
                    # Plot results
                    fig = plot_pca_results(results, "PCA Analysis Results")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show explained variance details
                    st.subheader("Explained Variance by Component")
                    variance_df = pd.DataFrame({
                        'Component': range(1, len(results['explained_variance_ratio']) + 1),
                        'Variance Explained': results['explained_variance_ratio'],
                        'Cumulative Variance': results['cumulative_variance']
                    })
                    st.dataframe(variance_df)
                    
                except Exception as e:
                    st.error(f"Error in PCA analysis: {e}")
    
    elif method == "UMAP":
        # Additional UMAP parameters
        col1, col2 = st.columns(2)
        with col1:
            n_neighbors = st.number_input("N Neighbors", value=15, min_value=2, max_value=100)
        with col2:
            min_dist = st.number_input("Min Distance", value=0.1, min_value=0.0, max_value=1.0, step=0.05)
        
        if st.button("Run UMAP Analysis"):
            with st.spinner("Running UMAP analysis..."):
                try:
                    results = dimensionality_reducer.compute_umap(
                        time_window=time_window,
                        n_components=n_components,
                        n_neighbors=n_neighbors,
                        min_dist=min_dist
                    )
                    
                    if results is not None:
                        # Display basic info
                        st.metric("Embedding Dimensions", results['transformed_data'].shape[1])
                        
                        # Plot results
                        fig = plot_umap_results(results, "UMAP Analysis Results")
                        if fig is not None:
                            st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error in UMAP analysis: {e}")


def _rsa_analysis(rsa_analyzer):
    """RSA analysis panel."""
    st.subheader("🧩 Representational Similarity Analysis")
    
    # Parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        start_time = st.number_input("Start Time (s)", value=-0.5, step=0.1, key="rsa_start")
    with col2:
        end_time = st.number_input("End Time (s)", value=1.0, step=0.1, key="rsa_end")
    with col3:
        metric = st.selectbox("Distance Metric", ["correlation", "euclidean", "cosine"])
    
    time_window = (start_time, end_time)
    
    if st.button("Run RSA Analysis"):
        with st.spinner("Running RSA analysis..."):
            try:
                results = rsa_analyzer.compute_rsa(
                    time_window=time_window,
                    metric=metric
                )
                
                if results is not None:
                    # Display basic info
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("N Conditions", len(results['condition_labels']))
                    with col2:
                        st.metric("Metric Used", results['metric'])
                    with col3:
                        avg_dissimilarity = np.mean(results['rdm'][np.triu_indices_from(results['rdm'], k=1)])
                        st.metric("Avg Dissimilarity", f"{avg_dissimilarity:.3f}")
                    
                    # Plot results
                    fig = plot_rsa_results(results, "RSA Analysis Results")
                    if fig is not None:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Show condition labels
                    st.subheader("Conditions Analyzed")
                    conditions_df = pd.DataFrame({
                        'Condition': results['condition_labels']
                    })
                    st.dataframe(conditions_df)
                    
                    # Show RDM values
                    st.subheader("Representational Dissimilarity Matrix")
                    rdm_df = pd.DataFrame(
                        results['rdm'], 
                        index=results['condition_labels'],
                        columns=results['condition_labels']
                    )
                    st.dataframe(rdm_df.round(3))
                
            except Exception as e:
                st.error(f"Error in RSA analysis: {e}") 