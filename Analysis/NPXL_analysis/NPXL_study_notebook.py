"""
NPXL Analysis Study Script

This script is for studying and exploring the NPXL analysis modules and their functionality.
Refactored from Jupyter notebook to a standalone Python script.
"""

# Standard imports
import sys
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import inspect

# Add the workspace root to Python path
# The script is in Analysis/NPXL_analysis, so we go up 2 levels to reach workspace root
current_dir = os.path.dirname(os.path.abspath(__file__))
# If we're in the NPXL_analysis folder, go up 2 levels
if 'NPXL_analysis' in current_dir:
    workspace_root = os.path.dirname(os.path.dirname(current_dir))
elif os.path.basename(current_dir) == 'NPXL_analysis':
    # If current directory is NPXL_analysis, go up 2 levels
    workspace_root = os.path.dirname(os.path.dirname(current_dir))
else:
    # Fallback: try to find the workspace root by going up directories
    # Look for a directory structure that suggests we're in the workspace
    test_dir = current_dir
    for _ in range(3):  # Try going up at most 3 levels
        if os.path.exists(os.path.join(test_dir, 'Analysis', 'NPXL_analysis')):
            workspace_root = test_dir
            break
        test_dir = os.path.dirname(test_dir)
    else:
        # Last fallback: use current directory
        workspace_root = current_dir

if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

# Import NPXL analysis modules
from Analysis.NPXL_analysis.NPXL_Preprocessing import (
    find_ks_folders,
    read_sample_rate,
    read_bin_size_from_metadata,
    load_cluster_info,
    load_event_windows_data
)
from Analysis.NPXL_analysis import npxl_single_unit_analysis
from Analysis.NPXL_analysis import population_analysis

print("Imports successful!")
print(f"Workspace root: {workspace_root}")


# ============================================================================
# 1. Exploring NPXL_Preprocessing Functions
# ============================================================================

def explore_preprocessing_functions():
    """Study the preprocessing functions."""
    import Analysis.NPXL_analysis.NPXL_Preprocessing as npxl_preproc
    
    # Get all functions from the module
    preprocessing_functions = [name for name, obj in inspect.getmembers(npxl_preproc) 
                              if inspect.isfunction(obj) and not name.startswith('_')]
    
    print("\nAvailable preprocessing functions:")
    for func_name in preprocessing_functions:
        print(f"  - {func_name}")


# ============================================================================
# 2. Exploring Single Unit Analysis Functions
# ============================================================================

def explore_single_unit_functions():
    """Study single unit analysis functions."""
    single_unit_functions = [name for name, obj in inspect.getmembers(npxl_single_unit_analysis) 
                            if inspect.isfunction(obj) and not name.startswith('_')]
    
    print("\nAvailable single unit analysis functions:")
    for func_name in single_unit_functions:
        print(f"  - {func_name}")


# ============================================================================
# 3. Exploring Population Analysis Functions
# ============================================================================

def explore_population_functions():
    """Study population analysis functions."""
    population_functions = [name for name, obj in inspect.getmembers(population_analysis) 
                           if inspect.isfunction(obj) and not name.startswith('_')]
    
    print("\nAvailable population analysis functions:")
    for func_name in population_functions:
        print(f"  - {func_name}")


# ============================================================================
# 4. Function Documentation and Signatures
# ============================================================================

def study_function(function_name='find_ks_folders', module_name='NPXL_Preprocessing'):
    """
    Get detailed information about a specific function.
    
    Parameters:
    -----------
    function_name : str
        Name of the function to study
    module_name : str
        Name of the module (default: 'NPXL_Preprocessing')
    """
    import Analysis.NPXL_analysis.NPXL_Preprocessing as npxl_preproc
    
    try:
        func = getattr(npxl_preproc, function_name)
        print(f"\nFunction: {function_name}")
        print(f"\nSignature:")
        print(inspect.signature(func))
        print(f"\nDocstring:")
        print(inspect.getdoc(func))
    except AttributeError:
        print(f"Function '{function_name}' not found in {module_name}")


# ============================================================================
# 5. Data Structure Exploration
# ============================================================================

def load_data(data_dir_parent=None, data_dir_OFC=None, data_dir_ACx=None):
    """
    Load event windows data for OFC and ACx.
    
    Parameters:
    -----------
    data_dir_parent : str, optional
        Parent directory path
    data_dir_OFC : str, optional
        OFC data directory path
    data_dir_ACx : str, optional
        ACx data directory path
    
    Returns:
    --------
    tuple : (OFC_all, ACx_all) numpy arrays
    """
    # Default paths if not provided
    if data_dir_parent is None:
        data_dir_parent = r"Z:\Shared\Amichai\NPXL\Recs\group7\catgt_G7A2_novice_2b_4t_g1"
    if data_dir_OFC is None:
        data_dir_OFC = os.path.join(data_dir_parent, "G7A2_novice_2b_4t_g1_imec1")
    if data_dir_ACx is None:
        data_dir_ACx = os.path.join(data_dir_parent, "G7A2_novice_2b_4t_g1_imec0")
    
    # Load good clusters OFC and ACx
    ofc_path = os.path.join(data_dir_OFC, "analysis_output", "event_windows_matrix.npy")
    acx_path = os.path.join(data_dir_ACx, "analysis_output", "event_windows_matrix.npy")
    
    if not os.path.exists(ofc_path):
        raise FileNotFoundError(f"OFC data file not found: {ofc_path}")
    if not os.path.exists(acx_path):
        raise FileNotFoundError(f"ACx data file not found: {acx_path}")
    
    OFC_all = np.load(ofc_path)  # [units, time, trials]
    ACx_all = np.load(acx_path)  # [units, time, trials]
    
    print(f"\nLoaded data:")
    print(f"  OFC_all shape: {OFC_all.shape}")
    print(f"  ACx_all shape: {ACx_all.shape}")
    
    return OFC_all, ACx_all, data_dir_OFC, data_dir_ACx


def load_unit_labels(data_dir_OFC, data_dir_ACx):
    """
    Load unit labels for OFC and ACx.
    
    Parameters:
    -----------
    data_dir_OFC : str
        OFC data directory path
    data_dir_ACx : str
        ACx data directory path
    
    Returns:
    --------
    tuple : (ofc_g_index, acx_g_index) DataFrames
    """
    ofc_labels = os.path.join(data_dir_OFC, "bombcell", "unit_labels.tsv")
    acx_labels = os.path.join(data_dir_ACx, "bombcell", "unit_labels.tsv")
    
    # Check if files exist before reading
    if not os.path.exists(ofc_labels):
        raise FileNotFoundError(f"File not found: {ofc_labels}")
    if not os.path.exists(acx_labels):
        raise FileNotFoundError(f"File not found: {acx_labels}")
    
    ofc_g_index = pd.read_csv(ofc_labels, header=0, sep="\t")
    acx_g_index = pd.read_csv(acx_labels, header=0, sep="\t")
    
    print("\nSuccessfully loaded unit labels for OFC and ACx")
    print(f"  OFC labels shape: {ofc_g_index.shape}")
    print(f"  ACx labels shape: {acx_g_index.shape}")
    
    # Get good units (UnitType == 1)
    OFC_g = ofc_g_index.index[ofc_g_index["UnitType"] == 1].tolist()
    ACx_g = acx_g_index.index[acx_g_index["UnitType"] == 1].tolist()
    
    print(f"\nGood units:")
    print(f"  OFC good units: {len(OFC_g)}")
    print(f"  ACx good units: {len(ACx_g)}")
    
    return ofc_g_index, acx_g_index, OFC_g, ACx_g


def explore_sample_data(OFC_all, unit_idx=1):
    """
    Explore sample data from a specific unit.
    
    Parameters:
    -----------
    OFC_all : numpy.ndarray
        OFC event windows matrix [units, time, trials]
    unit_idx : int
        Index of the unit to explore (default: 1)
    """
    sample = OFC_all[unit_idx, :, :]
    print(f"\nSample unit {unit_idx} shape: {sample.shape}")
    return sample


# ============================================================================
# Main execution
# ============================================================================

def main():
    """Main execution function."""
    print("=" * 70)
    print("NPXL Analysis Study Script")
    print("=" * 70)
    
    # Load and explore data

    OFC_all, ACx_all, data_dir_OFC, data_dir_ACx = load_data()
    ofc_g_index, acx_g_index, OFC_g, ACx_g = load_unit_labels(data_dir_OFC, data_dir_ACx)
    
    # Display unit type information
    print("\n" + "=" * 70)
    print("ACx UnitType == 1 (Good units):")
    print("=" * 70)
    print(acx_g_index["UnitType"] == 1)
    print("\nACx unit labels DataFrame:")
    print(acx_g_index)
    
    # Explore sample data
    sample = OFC_all[:,:,:].mean(axis=2)
    lables = pd.read_csv(os.path.join(data_dir_OFC, "analysis_output", "event_windows_stimuli_outcome.csv"))
    stim_labels = lables["outcome"].astype(str)  # use string type for clearer labeling
    # PCA analysis of the sample unit data
    from sklearn.decomposition import PCA
    import plotly.express as px

    # sample.shape: [time, trials], so transpose to [trials, time] for PCA (if necessary)
    X = sample.T  # shape: [trials, time]

    print(f"Running PCA on shape {X.shape} (trials x time bins)")

    pca = PCA(n_components=30)
    X_pca = pca.fit_transform(X)  # shape: [trials, 2]

    # Use a different color palette, e.g., Plotly's 'Viridis' palette or any other preferred palette
    fig = px.line(
        x=X_pca[:, 1],
        y=X_pca[:, 2],
        color=stim_labels,  # color by stim_labels for grouping
        labels={'x': 'PC1', 'y': 'PC2', 'color': 'Stimulus'},
        title="PCA of Sample Unit Trials",
        color_continuous_scale='Viridis' if stim_labels.dtype.kind in 'fc' else None,  # only applies if numeric
        color_discrete_sequence=px.colors.qualitative.Pastel  # more gentle color palette
    )
    fig.show()

    X_pca.shape


if __name__ == "__main__":
    main()
