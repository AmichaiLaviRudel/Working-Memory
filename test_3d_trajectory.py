import numpy as np
import pandas as pd
from Analysis.NPXL_analysis.population_analysis_advanced import (
    DimensionalityReducer, plot_pca_results
)

def test_3d_trajectory():
    """Test the 3D trajectory functionality."""
    print("Testing 3D trajectory functionality...")
    
    # Create synthetic data
    n_units = 15
    n_trials = 40
    n_time_bins = 30
    
    # Create synthetic event windows matrix with some temporal structure
    event_windows_matrix = np.random.randn(n_units, n_time_bins, n_trials)
    
    # Add some temporal structure (simulate stimulus response)
    time_axis = np.linspace(-1.5, 1.5, n_time_bins)
    for unit in range(n_units):
        # Add a temporal response pattern
        response_pattern = np.exp(-(time_axis - 0.2)**2 / 0.1) * (unit % 3 + 1)
        event_windows_matrix[unit, :, :] += response_pattern[:, np.newaxis]
    
    # Create synthetic stimuli_outcome_df
    stimulus_values = np.random.choice([4.5, 8.2, 16.7, 32.1], n_trials)
    
    stimuli_outcome_df = pd.DataFrame({
        'stimulus': stimulus_values,
        'outcome': np.random.choice(['Hit', 'Miss', 'CR', 'False Alarm'], n_trials),
        'choice': np.random.choice(['Go', 'NoGo'], n_trials)
    })
    
    # Create metadata
    metadata = {
        'window_duration': 3.0,
        'bin_size': 0.1
    }
    
    try:
        # Test PCA with 3D trajectory
        print("Running PCA analysis...")
        dimensionality_reducer = DimensionalityReducer(event_windows_matrix, stimuli_outcome_df, metadata)
        pca_results = dimensionality_reducer.compute_pca(time_window=(-0.5, 1.0), n_components=5)
        
        print("Creating plots...")
        fig, fig_3d = plot_pca_results(pca_results, title="Test PCA with 3D Trajectory")
        
        print("✅ PCA plotting successful!")
        print(f"   - Main PCA figure: {type(fig)}")
        print(f"   - 3D trajectory figure: {type(fig_3d)}")
        
        # Check if 3D figure has traces
        if fig_3d is not None:
            n_traces = len(fig_3d.data)
            print(f"   - 3D figure has {n_traces} traces")
            
            # Check if we have the trajectory trace
            has_trajectory = any('Neural Trajectory' in trace.name for trace in fig_3d.data if hasattr(trace, 'name'))
            print(f"   - Has neural trajectory: {has_trajectory}")
        
        print("\n🎉 3D trajectory test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_3d_trajectory() 