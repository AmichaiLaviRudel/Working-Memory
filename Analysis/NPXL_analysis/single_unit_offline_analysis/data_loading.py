"""
Data loading functions for NPXL offline analysis.
"""
import os
import numpy as np
import pandas as pd
from Analysis.NPXL_analysis.NPXL_Preprocessing import load_event_windows_data


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
    tuple : (OFC_all, ACx_all, data_dir_OFC, data_dir_ACx) numpy arrays
    """
    # Default paths if not provided
    if data_dir_parent is None:
        data_dir_parent = r"Z:/Shared/Amichai/NPXL/Recs/group5/catgt_G5A3_naive_true_g0"
    
    # Auto-detect OFC and ACx directories based on parent directory
    if data_dir_OFC is None or data_dir_ACx is None:
        if not os.path.exists(data_dir_parent):
            raise FileNotFoundError(f"Parent directory not found: {data_dir_parent}")
        
        # Find directories ending with imec1 (OFC) and imec0 (ACx)
        subdirs = [d for d in os.listdir(data_dir_parent) 
                   if os.path.isdir(os.path.join(data_dir_parent, d))]
        
        if data_dir_OFC is None:
            ofc_dirs = [d for d in subdirs if d.endswith('imec1')]
            if not ofc_dirs:
                raise FileNotFoundError(f"No directory ending with 'imec1' found in {data_dir_parent}")
            if len(ofc_dirs) > 1:
                raise ValueError(f"Multiple directories ending with 'imec1' found: {ofc_dirs}")
            data_dir_OFC = os.path.join(data_dir_parent, ofc_dirs[0])
        
        if data_dir_ACx is None:
            acx_dirs = [d for d in subdirs if d.endswith('imec0')]
            if not acx_dirs:
                raise FileNotFoundError(f"No directory ending with 'imec0' found in {data_dir_parent}")
            if len(acx_dirs) > 1:
                raise ValueError(f"Multiple directories ending with 'imec0' found: {acx_dirs}")
            data_dir_ACx = os.path.join(data_dir_parent, acx_dirs[0])
    
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
    tuple : (ofc_g_index, acx_g_index, OFC_g, ACx_g) DataFrames and lists
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


def read_event_windows_metadata(data_dir_x: str) -> dict:
    """
    Read event windows metadata from the analysis_output folder of a given probe directory.
    """
    metadata_path = os.path.join(data_dir_x, "analysis_output", "event_windows_metadata.txt")
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    metadata: dict[str, str] = {}
    with open(metadata_path, "r") as f:
        for line in f:
            if ":" not in line:
                continue
            key, value = line.strip().split(": ", 1)
            metadata[key] = value

    # Parse and cast to numeric types
    n_units = int(float(metadata.get("n_units", 0)))
    n_time_bins = int(float(metadata.get("n_time_bins", 0)))
    n_events = int(float(metadata.get("n_events", 0)))
    window_duration = float(metadata.get("window_duration", 0.0))
    bin_size = float(metadata.get("bin_size", 0.0))

    print(f"n_units: {n_units}")
    print(f"n_time_bins: {n_time_bins}")
    print(f"n_events: {n_events}")
    print(f"window_duration: {window_duration}")
    print(f"bin_size: {bin_size}")

    return {
        "n_units": n_units,
        "n_time_bins": n_time_bins,
        "n_events": n_events,
        "window_duration": window_duration,
        "bin_size": bin_size,
    }


def load_full_event_windows_data(data_dir_x: str):
    """
    Load full event windows data including stimuli/outcome information.
    
    Returns:
        tuple: (event_windows_matrix, time_axis, valid_event_indices, 
                stimuli_outcome_df, metadata, lick_event_windows_matrix)
    """
    analysis_output_dir = os.path.join(data_dir_x, "analysis_output")
    return load_event_windows_data(analysis_output_dir)


def load_behavioral_data(folder: str):
    analysis_output_dir = os.path.join(folder, "analysis_output")
     # Load the licking event windows matrix if it exists
    lick_file_path = os.path.join(analysis_output_dir, "lick_event_windows_matrix.npy")
    if os.path.exists(lick_file_path):
        lick_event_windows_matrix = np.load(lick_file_path)
    else:
        lick_event_windows_matrix = None
    
    # Load the time axis
    time_axis = np.load(os.path.join(analysis_output_dir, "event_window_time_axis.npy"))
    
    # Load the valid event indices
    valid_event_indices = np.load(os.path.join(analysis_output_dir, "valid_event_indices.npy"))
    
    # Load the filtered stimuli_outcome DataFrame
    stimuli_outcome_df = pd.read_csv(os.path.join(analysis_output_dir, "event_windows_stimuli_outcome.csv"))
    
    # Load metadata
    metadata = {}
    metadata_file = os.path.join(analysis_output_dir, "event_windows_metadata.txt")
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            for line in f:
                key, value = line.strip().split(': ')
                metadata[key] = value

    return lick_event_windows_matrix, time_axis, valid_event_indices, stimuli_outcome_df, metadata