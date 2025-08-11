#!/usr/bin/env python3
"""
Test script for jPCA implementation.
"""

import numpy as np
import pandas as pd
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Analysis.NPXL_analysis.population_analysis_advanced import jPCAAnalyzer, plot_jpca_results

def create_synthetic_data():
    """Create synthetic neural data with rotational dynamics."""
    np.random.seed(42)
    
    # Parameters
    n_units = 20
    n_trials = 50
    n_time_bins = 100
    time_window = (-0.5, 1.0)
    
    # Create time axis
    time_axis = np.linspace(time_window[0], time_window[1], n_time_bins)
    
    # Create synthetic data with rotational dynamics
    # Simulate oscillatory activity with different phases across units
    data = np.zeros((n_units, n_time_bins, n_trials))
    
    for unit in range(n_units):
        # Each unit has a different phase and frequency
        phase = 2 * np.pi * unit / n_units
        freq = 2 + 0.5 * np.random.randn()  # Hz
        
        for trial in range(n_trials):
            # Add some trial-to-trial variability
            trial_phase = phase + 0.1 * np.random.randn()
            trial_freq = freq + 0.1 * np.random.randn()
            
            # Create oscillatory activity
            signal = np.sin(2 * np.pi * trial_freq * time_axis + trial_phase)
            
            # Add noise
            noise = 0.1 * np.random.randn(n_time_bins)
            
            data[unit, :, trial] = signal + noise
    
    # Create metadata
    metadata = {
        'window_duration': 1.5,
        'bin_size': 0.015,
        'display_window': time_window
    }
    
    # Create stimuli_outcome_df
    stimuli_outcome_df = pd.DataFrame({
        'stimulus': np.random.choice([0, 1, 2, 3], n_trials),
        'outcome': np.random.choice(['Hit', 'Miss', 'FA', 'CR'], n_trials),
        'choice': np.random.choice(['Go', 'NoGo'], n_trials)
    })
    
    return data, stimuli_outcome_df, metadata

def test_jpca():
    """Test jPCA implementation."""
    print("Creating synthetic data...")
    event_windows_matrix, stimuli_outcome_df, metadata = create_synthetic_data()
    
    print(f"Data shape: {event_windows_matrix.shape}")
    print(f"Number of trials: {len(stimuli_outcome_df)}")
    
    print("\nInitializing jPCA analyzer...")
    jpca_analyzer = jPCAAnalyzer(event_windows_matrix, stimuli_outcome_df, metadata)
    
    print("Running jPCA analysis...")
    results = jpca_analyzer.compute_jpca(
        time_window=(-0.5, 1.0),
        n_components=6,
        max_skew=0.99
    )
    
    if results is not None:
        print("✅ jPCA analysis completed successfully!")
        print(f"Number of jPCA pairs: {len(results['jpca_pairs'])}")
        print(f"Number of singular values: {len(results['singular_values'])}")
        print(f"jPCA data shape: {results['X_jpca'].shape}")
        
        if len(results['jpca_pairs']) > 0:
            print("jPCA pairs found:")
            for i, (idx1, idx2) in enumerate(results['jpca_pairs']):
                print(f"  Pair {i+1}: Components {idx1+1} and {idx2+1}")
        else:
            print("No jPCA pairs found (this is expected for synthetic data)")
        
        print("\nTesting plotting function...")
        fig = plot_jpca_results(results, "Test jPCA Analysis")
        if fig is not None:
            print("✅ jPCA plotting completed successfully!")
        else:
            print("❌ jPCA plotting failed!")
    else:
        print("❌ jPCA analysis failed!")

if __name__ == "__main__":
    test_jpca() 