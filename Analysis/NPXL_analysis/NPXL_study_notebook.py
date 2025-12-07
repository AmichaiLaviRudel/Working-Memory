

"""
NPXL Analysis Study Script

This script is for studying and exploring the NPXL analysis modules and their functionality.
Refactored from Jupyter notebook to a standalone Python script.
"""
# %%
# Standard imports
import sys
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio  # Controls how Plotly renders figures (needed for VS Code / interactive window)
import inspect
from scipy.stats import ttest_rel, f_oneway  # paired t-test for before/after activity per unit, ANOVA for category sensitivity
from sklearn.metrics import roc_auc_score  # ROC AUC for go/nogo selectivity

# %%

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

# %%
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
from Analysis.GNG_bpod_analysis.colors import (
    OUTCOME_COLOR_MAP,
    SUBJECT_COLORS,
    GO_COLORS,
    NOGO_COLORS,
    COLOR_ACCENT,
    COLOR_ACCENT_TRANSPARENT,
    COLOR_GRAY,
)

print("Imports successful!")
print(f"Workspace root: {workspace_root}")

# Configure Plotly renderer
# Try inline-friendly renderers first; fall back to browser if needed.
# Note: Removed "plotly_mimetype" as it can cause warnings in some environments
preferred_renderers = ["vscode", "notebook", "notebook_connected"]
available = list(pio.renderers)

# Filter to only use renderers that actually exist
valid_renderers = [r for r in preferred_renderers if r in available]

if valid_renderers:
    pio.renderers.default = valid_renderers[0]
else:
    # Fallback to browser or any available renderer
    if "browser" in available:
        pio.renderers.default = "browser"
    elif available:
        pio.renderers.default = available[0]
    else:
        # Last resort: use 'json' renderer which just returns the figure object
        pio.renderers.default = "json"

# Suppress renderer warnings by setting renderer mode
import warnings
warnings.filterwarnings('ignore', message='.*renderer.*', category=UserWarning)

print(f"Using Plotly renderer: {pio.renderers.default}")
print(f"Available Plotly renderers: {available}")


# %% load data for OFC and ACx
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

# %% Load unit labels for OFC and ACx
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

# %% Read event window metadata (n_units, n_time_bins, n_events, window_duration, bin_size)
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


# %% # Example unit time course: find "active" units (differential before vs after midpoint)
def find_active_units_by_midpoint(
    data: np.ndarray,
    alpha: float = 0.05,
    before_range: tuple[int, int] | None = None,
    after_range: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find units that are differentially active before vs after in specified time windows
    relative to the temporal midpoint.

    `before_range` and `after_range` are (start_offset, end_offset) in time‑bins
    **relative to the midpoint** (negative = before mid, positive = after mid).
    Example: before_range = (-200, 0), after_range = (0, 200).

    Uses a paired t‑test across trials for mean firing rate before/after and keeps only
    units where AFTER > BEFORE (directional effect).
    """
    # data shape: [units, time, trials]
    n_units, n_time, n_trials = data.shape

    mid = n_time // 2

    # Default: symmetric windows around midpoint if explicit ranges are not given
    if before_range is None or after_range is None:
        before_range = (-mid, 0)
        after_range = (0, mid)

    b_off_start, b_off_end = before_range
    a_off_start, a_off_end = after_range

    # Convert relative offsets to absolute indices
    b_start = mid + b_off_start
    b_end = mid + b_off_end
    a_start = mid + a_off_start
    a_end = mid + a_off_end

    # Clamp indices to valid range
    b_start = max(0, min(b_start, n_time))
    b_end = max(b_start + 1, min(b_end, n_time))
    a_start = max(0, min(a_start, n_time))
    a_end = max(a_start + 1, min(a_end, n_time))

    # Mean over time within each window, keep per‑trial values
    before = data[:, b_start:b_end, :].mean(axis=1)  # shape: [units, trials]
    after = data[:, a_start:a_end, :].mean(axis=1)   # shape: [units, trials]

    t_vals = np.empty(n_units)
    p_vals = np.empty(n_units)
    mean_before = before.mean(axis=1)
    mean_after = after.mean(axis=1)

    for u in range(n_units):
        t_stat, p_val = ttest_rel(before[u, :], after[u, :])
        t_vals[u] = t_stat
        p_vals[u] = p_val

    # Require statistical difference AND after > before
    active_mask = (p_vals < alpha) & (mean_after > mean_before)
    active_units = np.where(active_mask)[0]

    print(
        f"Found {active_units.size} active units (after > before, p < {alpha}) "
        f"out of {n_units}"
    )
    # Sort active_units by their smallest p_val (ascending)
    sorted_indices = np.argsort(p_vals[active_units])
    active_units_sorted = active_units[sorted_indices]
    t_vals_sorted = t_vals[active_units_sorted]
    p_vals_sorted = p_vals[active_units_sorted]
    return active_units_sorted, t_vals_sorted, p_vals_sorted


def plot_active_units_timecourses(
    data: np.ndarray,
    active_units: np.ndarray,
    metadata: dict,
    *,
    max_units: int = 10,
    target_bin_size_sec: float = 0.01,
    window_ms: tuple[float, float] = (-500.0, 1000.0),
    region_name: str = "Unit",
) -> None:
    """
    Plot mean time‑courses for a subset of active units around the recording midpoint.
    """
    if active_units.size == 0:
        print("No active units to plot.")
        return

    n_units, n_time, _ = data.shape
    orig_bin_size_sec = float(metadata["bin_size"])  # seconds per bin from preprocessing

    # Limit number of plotted units
    n_to_plot = int(min(max_units, active_units.size))

    for unit_idx in active_units[:n_to_plot]:
        if unit_idx < 0 or unit_idx >= n_units:
            continue  # safety check

        unit_trace = data[unit_idx, :, :].mean(axis=1)  # mean over trials, keep time

        # Re‑bin to desired bin size if needed
        if not np.isclose(orig_bin_size_sec, target_bin_size_sec):
            factor = int(round(target_bin_size_sec / orig_bin_size_sec))
            if factor <= 0:
                raise ValueError(
                    f"Invalid binning factor computed from orig_bin_size={orig_bin_size_sec}, "
                    f"target={target_bin_size_sec}"
                )
            n_bins = len(unit_trace) // factor
            unit_binned = (
                unit_trace[: n_bins * factor]
                .reshape(n_bins, factor)
                .mean(axis=1)
            )
            effective_bin_size_sec = orig_bin_size_sec * factor
        else:
            unit_binned = unit_trace
            effective_bin_size_sec = orig_bin_size_sec

        # Time axis (ms) relative to recording midpoint
        mid_bin = n_time // 2
        mid_time_sec = mid_bin * orig_bin_size_sec
        time_axis_ms = (
            np.arange(len(unit_binned)) * effective_bin_size_sec * 1000.0
            - mid_time_sec * 1000.0
        )

        # Window mask in ms
        w_start, w_end = window_ms
        mask = (time_axis_ms >= w_start) & (time_axis_ms <= w_end)
        if not np.any(mask):
            continue

        time_to_plot = time_axis_ms[mask]
        unit_to_plot = unit_binned[mask]

        trace = go.Scatter(
            x=time_to_plot,
            y=unit_to_plot,
            mode="lines",
            name=f"{region_name} Unit {unit_idx} Mean (binned)",
        )
        layout = go.Layout(
            title=f"Example Active {region_name} Unit Avg Response",
            xaxis=dict(
                title=f"Time relative to midpoint (ms, {effective_bin_size_sec * 1000.0:.1f} ms/bin)"
            ),
            yaxis=dict(title="Mean Response (Hz)"),
        )
        fig = go.Figure(data=[trace], layout=layout)
        # fig  # noqa: E305
        # fig.show()


# %% main execution

# ============================================================================
# Main execution
# ============================================================================

# Load and explore data
parent_dir = r"Z:\Shared\Amichai\NPXL\Recs\group5\catGTGroup5\catgt_G5A3_2b_4t_new2_g0"

OFC_all, ACx_all, data_dir_OFC, data_dir_ACx = load_data(data_dir_parent=parent_dir, data_dir_OFC=None, data_dir_ACx=None)

# %%
ofc_g_index, acx_g_index, OFC_g, ACx_g = load_unit_labels(data_dir_OFC, data_dir_ACx)

# Example usage for OFC and ACx probes
ofc_metadata = read_event_windows_metadata(data_dir_OFC)
acx_metadata = read_event_windows_metadata(data_dir_ACx)
orig_bin_size_sec = float(acx_metadata["bin_size"])  # seconds per bin from preprocessing
bin_to_sec = 1 / orig_bin_size_sec

# %% 



# %% find active units by midpoint of ACx
active_units_acx, t_vals_acx, p_vals_acx = find_active_units_by_midpoint(
    ACx_all,
    alpha=0.05,
    # Example: 200 bins before mid vs 200 bins after mid (relative to midpoint)
    before_range=(int(-1 * bin_to_sec), 0),
    after_range=(0, int(1 * bin_to_sec)),
)
# %% plot active units timecourses

# Plot ACx active units (top 10 by significance) using the general function
plot_active_units_timecourses(
    ACx_all,
    active_units_acx,
    acx_metadata,
    max_units=10,
    target_bin_size_sec=0.01,
    window_ms=(-500.0, 1000.0),
    region_name="ACx",
)

# %% Setup output directories for saving results
def setup_results_directory(analysis_output_dir: str, subfolder: str = "") -> str:
    """
    Create organized directory structure for saving analysis results.
    
    Parameters:
    -----------
    analysis_output_dir : str
        Base analysis output directory (parent directory)
    subfolder : str
        Name of subfolder for this analysis (default: "" for parent directory)
    
    Returns:
    --------
    str : Path to the results directory
    """
    if subfolder:
        results_dir = os.path.join(analysis_output_dir, subfolder)
    else:
        results_dir = analysis_output_dir  # Save directly in parent directory
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Create subdirectories for organization
    os.makedirs(os.path.join(results_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "plots", "acx"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "plots", "ofc"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "plots", "comparison"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "plots", "psth_by_stimulus"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "plots", "psth_by_outcome"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "plots", "psth_by_category"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "plots", "raw_psth"), exist_ok=True)
    
    print(f"Results will be saved to: {results_dir}")
    return results_dir

def save_dataframe_to_csv(df: pd.DataFrame, filepath: str, description: str = ""):
    """Save DataFrame to CSV file."""
    df.to_csv(filepath, index=False)
    if description:
        print(f"  Saved {description} to: {filepath}")
    else:
        print(f"  Saved DataFrame to: {filepath}")

def save_plot_to_html(fig: go.Figure, filepath: str, description: str = ""):
    """Save Plotly figure to HTML file."""
    fig.write_html(filepath)
    if description:
        print(f"  Saved {description} to: {filepath}")
    else:
        print(f"  Saved plot to: {filepath}")

# %% Load full event windows data with stimuli/outcome information
def load_full_event_windows_data(data_dir_x: str):
    """
    Load full event windows data including stimuli/outcome information.
    
    Returns:
        tuple: (event_windows_matrix, time_axis, valid_event_indices, 
                stimuli_outcome_df, metadata, lick_event_windows_matrix)
    """
    analysis_output_dir = os.path.join(data_dir_x, "analysis_output")
    return load_event_windows_data(analysis_output_dir)

# %%
# Load full event windows data for ACx and OFC
acx_event_windows_data = load_full_event_windows_data(data_dir_ACx)
ofc_event_windows_data = load_full_event_windows_data(data_dir_OFC)

# Extract components
acx_event_matrix, acx_time_axis, acx_valid_indices, acx_stimuli_outcome_df, acx_metadata_full, acx_lick_data = acx_event_windows_data
ofc_event_matrix, ofc_time_axis, ofc_valid_indices, ofc_stimuli_outcome_df, ofc_metadata_full, ofc_lick_data = ofc_event_windows_data

print(f"\nLoaded event windows data:")
print(f"  ACx: {acx_event_matrix.shape} units × time × events")
print(f"  OFC: {ofc_event_matrix.shape} units × time × events")
print(f"\nACx stimuli/outcome columns: {list(acx_stimuli_outcome_df.columns)}")
print(f"ACx unique outcomes: {acx_stimuli_outcome_df['outcome'].unique() if 'outcome' in acx_stimuli_outcome_df.columns else 'N/A'}")
if 'stimulus' in acx_stimuli_outcome_df.columns:
    print(f"ACx unique stimuli: {sorted(acx_stimuli_outcome_df['stimulus'].unique())}")

# %% Setup results directory
# Use ACx analysis_output_dir as base (save directly in parent directory)
analysis_output_dir = os.path.join(parent_dir, "analysis_output")
results_dir = setup_results_directory(analysis_output_dir, subfolder="")

# %%
def compute_selectivity_metrics_for_active_units(
    event_windows_data: tuple,
    active_units: np.ndarray,
    window: tuple[float, float] = (-0.1, 0.5),
) -> pd.DataFrame:
    """
    Compute stimulus selectivity, outcome modulation, go/nogo coding, and choice probability
    for all active units.
    
    Returns:
        pd.DataFrame with columns: unit_idx, stimulus_selective, outcome_p_value, 
        go_nogo_dprime, go_nogo_roc_auc, choice_probability, choice_probability_corr
    """
    # Unpack event_windows_data (may be 5 or 6 elements)
    if len(event_windows_data) == 6:
        event_windows_matrix, time_axis, valid_event_indices, stimuli_outcome_df, metadata, _ = event_windows_data
        # Create 5-tuple for analysis functions that expect it
        event_windows_data_5 = (event_windows_matrix, time_axis, valid_event_indices, stimuli_outcome_df, metadata)
    else:
        event_windows_data_5 = event_windows_data
        event_windows_matrix, time_axis, valid_event_indices, stimuli_outcome_df, metadata = event_windows_data
    
    results = []
    
    for unit_idx in active_units:
        unit_results = {"unit_idx": int(unit_idx)}
        
        # Stimulus selectivity
        if 'stimulus' in stimuli_outcome_df.columns:
            unique_stimuli, tuning_curve, tuning_sem, best_stimulus = npxl_single_unit_analysis.compute_stimulus_selectivity(
                event_windows_data_5, stimuli_outcome_df, int(unit_idx), window=window
            )
            if tuning_curve is not None and len(tuning_curve) > 1:
                # Test if there's significant variation across stimuli (ANOVA-like)
                # Simple test: check if max - min > 2*SEM
                max_response = np.max(tuning_curve)
                min_response = np.min(tuning_curve)
                max_sem = np.max(tuning_sem) if len(tuning_sem) > 0 else 0
                stimulus_selective = (max_response - min_response) > (2 * max_sem)
                unit_results["stimulus_selective"] = stimulus_selective
                unit_results["best_stimulus"] = best_stimulus
                unit_results["max_stimulus_response"] = max_response
                unit_results["min_stimulus_response"] = min_response
            else:
                unit_results["stimulus_selective"] = False
                unit_results["best_stimulus"] = None
        else:
            unit_results["stimulus_selective"] = False
            unit_results["best_stimulus"] = None
        
        # Outcome modulation
        if 'outcome' in stimuli_outcome_df.columns:
            outcome_p, outcome_rates, outcome_means = npxl_single_unit_analysis.compute_outcome_modulation(
                event_windows_data_5, stimuli_outcome_df, int(unit_idx), window=window
            )
            if outcome_p is not None:
                unit_results["outcome_p_value"] = float(outcome_p)
                unit_results["outcome_modulated"] = outcome_p < 0.05
                if outcome_means is not None:
                    unit_results["rewarded_mean_rate"] = float(outcome_means[0])
                    unit_results["non_rewarded_mean_rate"] = float(outcome_means[1])
            else:
                unit_results["outcome_p_value"] = np.nan
                unit_results["outcome_modulated"] = False
        else:
            unit_results["outcome_p_value"] = np.nan
            unit_results["outcome_modulated"] = False
        
        # Go/NoGo coding
        if 'outcome' in stimuli_outcome_df.columns:
            go_nogo_dprime, go_nogo_roc_auc, go_nogo_rates = npxl_single_unit_analysis.compute_go_nogo_coding(
                event_windows_data_5, stimuli_outcome_df, int(unit_idx), window=window
            )
            if go_nogo_dprime is not None:
                unit_results["go_nogo_dprime"] = float(go_nogo_dprime)
                unit_results["go_nogo_roc_auc"] = float(go_nogo_roc_auc)
                unit_results["go_nogo_selective"] = abs(go_nogo_dprime) > 0.5  # Threshold for selectivity
            else:
                unit_results["go_nogo_dprime"] = np.nan
                unit_results["go_nogo_roc_auc"] = np.nan
                unit_results["go_nogo_selective"] = False
        else:
            unit_results["go_nogo_dprime"] = np.nan
            unit_results["go_nogo_roc_auc"] = np.nan
            unit_results["go_nogo_selective"] = False
        
        # Choice probability
        if 'outcome' in stimuli_outcome_df.columns:
            cp, cp_corr = npxl_single_unit_analysis.compute_choice_probability(
                event_windows_data_5, stimuli_outcome_df, int(unit_idx), window=window
            )
            if cp is not None:
                unit_results["choice_probability"] = float(cp)
                unit_results["choice_probability_corr"] = float(cp_corr)
                unit_results["choice_coding"] = abs(cp_corr) > 0.1  # Threshold for choice coding
            else:
                unit_results["choice_probability"] = np.nan
                unit_results["choice_probability_corr"] = np.nan
                unit_results["choice_coding"] = False
        else:
            unit_results["choice_probability"] = np.nan
            unit_results["choice_probability_corr"] = np.nan
            unit_results["choice_coding"] = False
        
        results.append(unit_results)
    
    return pd.DataFrame(results)

# %%
# Compute selectivity metrics for ACx active units
acx_selectivity_df = compute_selectivity_metrics_for_active_units(
    acx_event_windows_data,
    active_units_acx,
    window=(-0.1, 1),
)

print(f"\nSelectivity metrics for {len(acx_selectivity_df)} ACx active units:")
if len(acx_selectivity_df) > 0 and 'stimulus_selective' in acx_selectivity_df.columns:
    print(f"  Stimulus selective: {acx_selectivity_df['stimulus_selective'].sum()}")
    print(f"  Outcome modulated (p<0.05): {acx_selectivity_df['outcome_modulated'].sum()}")
    print(f"  Go/NoGo selective (|d'|>0.5): {acx_selectivity_df['go_nogo_selective'].sum()}")
    print(f"  Choice coding (|CP_corr|>0.1): {acx_selectivity_df['choice_coding'].sum()}")
    print(f"\nSummary statistics:")
    print(acx_selectivity_df.describe())
else:
    print("  No active units found - skipping selectivity metrics")

# Save ACx selectivity results
print("\n=== Saving ACx selectivity results ===")
save_dataframe_to_csv(
    acx_selectivity_df,
    os.path.join(results_dir, "tables", "acx_selectivity_metrics.csv"),
    "ACx selectivity metrics table"
)



# %% Visualization functions for PSTH by condition
def _hex_to_rgba(hex_color: str, alpha: float = 0.2) -> str:
    """
    Convert hex color to rgba string for transparency.
    """
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'

def plot_psth_by_stimulus(
    event_windows_data: tuple,
    unit_idx: int,
    display_window: tuple[float, float] = (-0.5, 1.0),
    region_name: str = "Unit",
) -> go.Figure:
    """
    Plot PSTH separated by stimulus type for a single unit.
    """
    # Handle both 5-tuple and 6-tuple formats
    if len(event_windows_data) == 6:
        event_windows_matrix, time_axis, valid_event_indices, stimuli_outcome_df, metadata, _ = event_windows_data
    else:
        event_windows_matrix, time_axis, valid_event_indices, stimuli_outcome_df, metadata = event_windows_data
    
    if 'stimulus' not in stimuli_outcome_df.columns:
        print("No stimulus information available")
        return go.Figure()
    
    # Get unit data
    unit_data = event_windows_matrix[unit_idx, :, :]  # [time × events]
    
    # Find time indices for display window
    start_idx = np.argmin(np.abs(time_axis - display_window[0]))
    end_idx = np.argmin(np.abs(time_axis - display_window[1]))
    unit_data_windowed = unit_data[start_idx:end_idx, :]
    time_axis_windowed = time_axis[start_idx:end_idx]
    
    # Get unique stimuli
    unique_stimuli = sorted(stimuli_outcome_df['stimulus'].unique())
    
    # Create figure
    fig = go.Figure()
    
    # Use SUBJECT_COLORS from colors.py (already in hex format)
    for stim_idx, stim in enumerate(unique_stimuli):
        stim_mask = (stimuli_outcome_df['stimulus'] == stim).values
        stim_trials = unit_data_windowed[:, stim_mask]
        
        if stim_trials.shape[1] > 0:
            psth_mean = np.mean(stim_trials, axis=1)
            psth_sem = np.std(stim_trials, axis=1) / np.sqrt(stim_trials.shape[1])
            
            # Get color from SUBJECT_COLORS palette
            color = SUBJECT_COLORS[stim_idx % len(SUBJECT_COLORS)]
            
            # Main trace
            fig.add_trace(go.Scatter(
                x=time_axis_windowed,
                y=psth_mean,
                mode='lines',
                name=f'Stim {stim} (n={stim_trials.shape[1]})',
                line=dict(color=color, width=2)
            ))
            
            # SEM shading using helper function
            fig.add_trace(go.Scatter(
                x=np.concatenate([time_axis_windowed, time_axis_windowed[::-1]]),
                y=np.concatenate([psth_mean + psth_sem, (psth_mean - psth_sem)[::-1]]),
                fill='toself',
                fillcolor=_hex_to_rgba(color, alpha=0.2),
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # Add vertical line at event onset
    fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)
    
    fig.update_layout(
        title=f"{region_name} Unit {unit_idx} - PSTH by Stimulus",
        xaxis_title="Time (s)",
        yaxis_title="Firing Rate (Hz)",
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig

def plot_psth_by_outcome(
    event_windows_data: tuple,
    unit_idx: int,
    display_window: tuple[float, float] = (-0.5, 1.0),
    region_name: str = "Unit",
) -> go.Figure:
    """
    Plot PSTH separated by behavioral outcome (Hit/Miss/FA/CR) for a single unit.
    """
    # Handle both 5-tuple and 6-tuple formats
    if len(event_windows_data) == 6:
        event_windows_matrix, time_axis, valid_event_indices, stimuli_outcome_df, metadata, _ = event_windows_data
    else:
        event_windows_matrix, time_axis, valid_event_indices, stimuli_outcome_df, metadata = event_windows_data
    
    if 'outcome' not in stimuli_outcome_df.columns:
        print("No outcome information available")
        return go.Figure()
    
    # Get unit data
    unit_data = event_windows_matrix[unit_idx, :, :]  # [time × events]
    
    # Find time indices for display window
    start_idx = np.argmin(np.abs(time_axis - display_window[0]))
    end_idx = np.argmin(np.abs(time_axis - display_window[1]))
    unit_data_windowed = unit_data[start_idx:end_idx, :]
    time_axis_windowed = time_axis[start_idx:end_idx]
    
    # Define outcomes - use OUTCOME_COLOR_MAP from colors.py
    outcomes = ['Hit', 'Miss', 'False Alarm', 'CR']
    
    # Create figure
    fig = go.Figure()
    
    for outcome in outcomes:
        outcome_mask = (stimuli_outcome_df['outcome'] == outcome).values
        outcome_trials = unit_data_windowed[:, outcome_mask]
        
        if outcome_trials.shape[1] > 0:
            psth_mean = np.mean(outcome_trials, axis=1)
            psth_sem = np.std(outcome_trials, axis=1) / np.sqrt(outcome_trials.shape[1])
            
            # Get color from OUTCOME_COLOR_MAP
            color = OUTCOME_COLOR_MAP.get(outcome, '#808080')  # Default gray if not found
            
            # Main trace
            fig.add_trace(go.Scatter(
                x=time_axis_windowed,
                y=psth_mean,
                mode='lines',
                name=f'{outcome} (n={outcome_trials.shape[1]})',
                line=dict(color=color, width=2)
            ))
            
            # SEM shading using helper function
            fig.add_trace(go.Scatter(
                x=np.concatenate([time_axis_windowed, time_axis_windowed[::-1]]),
                y=np.concatenate([psth_mean + psth_sem, (psth_mean - psth_sem)[::-1]]),
                fill='toself',
                fillcolor=_hex_to_rgba(color, alpha=0.2),
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # Add vertical line at event onset
    fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)
    
    fig.update_layout(
        title=f"{region_name} Unit {unit_idx} - PSTH by Outcome",
        xaxis_title="Time (s)",
        yaxis_title="Firing Rate (Hz)",
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig

def plot_raw_psth(
    event_windows_data: tuple,
    unit_idx: int,
    display_window: tuple[float, float] = (-0.5, 1.0),
    region_name: str = "Unit",
) -> go.Figure:
    """
    Plot raw PSTH (all trials averaged together) for a single unit.
    """
    # Handle both 5-tuple and 6-tuple formats
    if len(event_windows_data) == 6:
        event_windows_matrix, time_axis, valid_event_indices, stimuli_outcome_df, metadata, _ = event_windows_data
    else:
        event_windows_matrix, time_axis, valid_event_indices, stimuli_outcome_df, metadata = event_windows_data
    
    # Get unit data
    unit_data = event_windows_matrix[unit_idx, :, :]  # [time × events]
    
    # Find time indices for display window
    start_idx = np.argmin(np.abs(time_axis - display_window[0]))
    end_idx = np.argmin(np.abs(time_axis - display_window[1]))
    unit_data_windowed = unit_data[start_idx:end_idx, :]
    time_axis_windowed = time_axis[start_idx:end_idx]
    
    # Calculate PSTH (mean and SEM across all trials)
    psth_mean = np.mean(unit_data_windowed, axis=1)
    psth_sem = np.std(unit_data_windowed, axis=1) / np.sqrt(unit_data_windowed.shape[1])
    
    # Create figure
    fig = go.Figure()
    
    # Main PSTH trace
    fig.add_trace(go.Scatter(
        x=time_axis_windowed,
        y=psth_mean,
        mode='lines',
        name='Mean Firing Rate',
        line=dict(color=COLOR_ACCENT, width=2)
    ))
    
    # SEM shading
    fig.add_trace(go.Scatter(
        x=np.concatenate([time_axis_windowed, time_axis_windowed[::-1]]),
        y=np.concatenate([psth_mean + psth_sem, (psth_mean - psth_sem)[::-1]]),
        fill='toself',
        fillcolor=COLOR_ACCENT_TRANSPARENT,
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Add vertical line at event onset
    fig.add_vline(x=0, line_dash="dash", line_color=COLOR_GRAY, line_width=1)
    
    fig.update_layout(
        title=f"{region_name} Unit {unit_idx} - Raw PSTH (n={unit_data_windowed.shape[1]} trials)",
        xaxis_title="Time (s)",
        yaxis_title="Firing Rate (Hz)",
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig

def save_raw_psth_for_active_units(
    event_windows_data: tuple,
    active_units: np.ndarray,
    p_vals: np.ndarray,
    region_name: str,
    results_dir: str,
    display_window: tuple[float, float] = (-0.5, 1.0),
) -> None:
    """
    Save raw PSTH plots for all active units in a region, sorted by p-values (most significant first).
    
    Parameters:
    -----------
    event_windows_data : tuple
        Event windows data tuple
    active_units : np.ndarray
        Array of active unit indices
    p_vals : np.ndarray
        Array of p-values corresponding to active_units (for sorting)
    region_name : str
        Name of the region (e.g., "ACx", "OFC")
    results_dir : str
        Base results directory path
    display_window : tuple[float, float]
        Time window for display (start, end) in seconds
    """
    if len(active_units) == 0:
        print(f"\n=== No {region_name} active units found - skipping raw PSTH plots ===")
        return
    
    # Sort units by p-value (most significant first)
    sorted_indices = np.argsort(p_vals)
    sorted_units = active_units[sorted_indices]
    sorted_p_vals = p_vals[sorted_indices]
    
    print(f"\n=== Saving raw PSTH plots for all {region_name} responsive units (sorted by p-value) ===")
    for rank, (unit_idx, p_val) in enumerate(zip(sorted_units, sorted_p_vals), start=1):
        fig_raw_psth = plot_raw_psth(
            event_windows_data,
            int(unit_idx),
            display_window=display_window,
            region_name=region_name
        )
        # Include rank in filename for easy identification
        save_plot_to_html(
            fig_raw_psth,
            os.path.join(results_dir, "plots", "raw_psth", f"{region_name.lower()}_unit_{unit_idx}_rank{rank:03d}_p{p_val:.4f}_raw_psth.html"),
            f"{region_name} Unit {unit_idx} (Rank {rank}, p={p_val:.4f}) Raw PSTH"
        )
    print(f"  Saved {len(active_units)} {region_name} raw PSTH plots (sorted by significance)")

# %% # Plot and save raw PSTH for all responsive units
save_raw_psth_for_active_units(
    acx_event_windows_data,
    active_units_acx,
    p_vals_acx,
    "ACx",
    results_dir,
    display_window=(-0.5, 1.0),
)

# %% # Plot PSTH by stimulus and outcome for top active units
n_units_to_plot = min(5, len(active_units_acx))
for i, unit_idx in enumerate(active_units_acx[:n_units_to_plot]):
    print(f"\n=== ACx Unit {unit_idx} (Rank {i+1} by significance) ===")
    
    # Plot by stimulus
    if 'stimulus' in acx_stimuli_outcome_df.columns:
        fig_stim = plot_psth_by_stimulus(
            acx_event_windows_data,
            int(unit_idx),
            display_window=(-0.5, 1.0),
            region_name="ACx"
        )
        # fig_stim.show()  # Disabled inline plotting; figures are saved to disk instead
    
    # Plot by outcome
    if 'outcome' in acx_stimuli_outcome_df.columns:
        fig_outcome = plot_psth_by_outcome(
            acx_event_windows_data,
            int(unit_idx),
            display_window=(-0.5, 1.0),
            region_name="ACx"
        )
        # fig_outcome.show()  # Disabled inline plotting; figures are saved to disk instead
        
        # Save PSTH plots
        if 'stimulus' in acx_stimuli_outcome_df.columns:
            save_plot_to_html(
                fig_stim,
                os.path.join(results_dir, "plots", "psth_by_stimulus", f"acx_unit_{unit_idx}_psth_by_stimulus.html"),
                f"ACx Unit {unit_idx} PSTH by Stimulus"
            )
        save_plot_to_html(
            fig_outcome,
            os.path.join(results_dir, "plots", "psth_by_outcome", f"acx_unit_{unit_idx}_psth_by_outcome.html"),
            f"ACx Unit {unit_idx} PSTH by Outcome"
        )

# %% Create selectivity summary plots
def plot_selectivity_summary(
    selectivity_df: pd.DataFrame,
    region_name: str = "Region",
) -> tuple[go.Figure, go.Figure]:
    """
    Create summary plots showing selectivity metrics for all active units.
    Returns two figures: (metrics_plot, classification_plot)
    """
    # Figure 1: Scatter plot of selectivity metrics
    fig_metrics = go.Figure()

    # If there are no active units or required columns are missing, return placeholder figures
    required_cols = [
        "unit_idx",
        "outcome_p_value",
        "go_nogo_dprime",
        "choice_probability_corr",
        "stimulus_selective",
        "outcome_modulated",
        "go_nogo_selective",
        "choice_coding",
    ]
    if (
        selectivity_df is None
        or len(selectivity_df) == 0
        or not all(col in selectivity_df.columns for col in required_cols)
    ):
        # Explicit placeholders avoid KeyError when there are no active units
        fig_metrics.update_layout(
            title=f"{region_name} - No selectivity metrics (no active units)",
            template="plotly_white",
            height=400,
        )
        fig_class = go.Figure()
        fig_class.update_layout(
            title=f"{region_name} - No unit classification (no active units)",
            template="plotly_white",
            height=300,
        )
        return fig_metrics, fig_class
    
    # Outcome modulation (p-value)
    outcome_p_vals = selectivity_df['outcome_p_value'].values
    valid_outcome = ~np.isnan(outcome_p_vals)
    
    if np.any(valid_outcome):
        fig_metrics.add_trace(go.Scatter(
            x=selectivity_df.loc[valid_outcome, 'unit_idx'],
            y=-np.log10(outcome_p_vals[valid_outcome] + 1e-10),  # -log10(p) with small offset
            mode='markers',
            name='Outcome Modulation (-log10 p)',
            marker=dict(color='blue', size=8, symbol='circle'),
            hovertemplate='Unit %{x}<br>-log10(p) = %{y:.2f}<extra></extra>'
        ))
    
    # Go/NoGo d'
    go_nogo_dprime = selectivity_df['go_nogo_dprime'].values
    valid_go_nogo = ~np.isnan(go_nogo_dprime)
    
    if np.any(valid_go_nogo):
        fig_metrics.add_trace(go.Scatter(
            x=selectivity_df.loc[valid_go_nogo, 'unit_idx'],
            y=go_nogo_dprime[valid_go_nogo],
            mode='markers',
            name="Go/NoGo d'",
            marker=dict(color='green', size=8, symbol='square'),
            yaxis='y2',
            hovertemplate='Unit %{x}<br>d\' = %{y:.2f}<extra></extra>'
        ))
    
    # Choice probability correlation
    cp_corr = selectivity_df['choice_probability_corr'].values
    valid_cp = ~np.isnan(cp_corr)
    
    if np.any(valid_cp):
        fig_metrics.add_trace(go.Scatter(
            x=selectivity_df.loc[valid_cp, 'unit_idx'],
            y=cp_corr[valid_cp],
            mode='markers',
            name='Choice Probability (corr)',
            marker=dict(color='red', size=8, symbol='diamond'),
            yaxis='y3',
            hovertemplate='Unit %{x}<br>CP corr = %{y:.2f}<extra></extra>'
        ))
    
    # Add significance thresholds
    fig_metrics.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="blue", 
                         annotation_text="p=0.05", annotation_position="right")
    
    fig_metrics.update_layout(
        title=f"{region_name} - Selectivity Metrics Summary",
        xaxis_title="Unit Index",
        yaxis=dict(title="-log10(Outcome p-value)", side="left"),
        yaxis2=dict(title="Go/NoGo d'", side="right", overlaying="y"),
        yaxis3=dict(title="Choice Probability (corr)", side="right", overlaying="y", position=0.95),
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    # Figure 2: Classification summary (bar chart)
    categories = {
        'Stimulus Selective': selectivity_df['stimulus_selective'].sum(),
        'Outcome Modulated': selectivity_df['outcome_modulated'].sum(),
        'Go/NoGo Selective': selectivity_df['go_nogo_selective'].sum(),
        'Choice Coding': selectivity_df['choice_coding'].sum(),
    }
    
    # Count units in each category combination
    both_stim_outcome = ((selectivity_df['stimulus_selective']) & 
                         (selectivity_df['outcome_modulated'])).sum()
    all_selective = ((selectivity_df['stimulus_selective']) | 
                     (selectivity_df['outcome_modulated']) | 
                     (selectivity_df['go_nogo_selective']) | 
                     (selectivity_df['choice_coding'])).sum()
    
    fig_class = go.Figure()
    
    fig_class.add_trace(go.Bar(
        x=list(categories.keys()),
        y=list(categories.values()),
        marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
        text=list(categories.values()),
        textposition='outside',
        hovertemplate='%{x}<br>Count: %{y}<extra></extra>'
    ))
    
    fig_class.update_layout(
        title=f"{region_name} - Unit Classification Summary (n={len(selectivity_df)} active units)",
        xaxis_title="Selectivity Type",
        yaxis_title="Number of Units",
        template='plotly_white',
        height=400
    )
    
    # Add annotation for combined categories
    fig_class.add_annotation(
        x=0.5, y=0.95,
        xref='paper', yref='paper',
        text=f"Units with any selectivity: {all_selective}<br>Units with both stimulus & outcome: {both_stim_outcome}",
        showarrow=False,
        font=dict(size=12),
        bgcolor='rgba(255,255,255,0.8)',
        bordercolor='black',
        borderwidth=1
    )
    
    return fig_metrics, fig_class

# %% # Create selectivity summary for ACx
acx_fig_metrics, acx_fig_class = plot_selectivity_summary(acx_selectivity_df, region_name="ACx")

# Save ACx summary plots
print("\n=== Saving ACx summary plots ===")
save_plot_to_html(
    acx_fig_metrics,
    os.path.join(results_dir, "plots", "acx", "acx_selectivity_metrics_summary.html"),
    "ACx selectivity metrics summary"
)
save_plot_to_html(
    acx_fig_class,
    os.path.join(results_dir, "plots", "acx", "acx_unit_classification_summary.html"),
    "ACx unit classification summary"
)

# Display detailed summary table
print("\n=== ACx Selectivity Summary Table ===")
if len(acx_selectivity_df) > 0 and 'stimulus_selective' in acx_selectivity_df.columns:
    print(acx_selectivity_df[['unit_idx', 'stimulus_selective', 'outcome_modulated', 
                               'go_nogo_selective', 'choice_coding']].to_string(index=False))
else:
    print("No active units found - no summary table to display")

# %% Compare OFC vs ACx # Find active units in OFC
active_units_ofc, t_vals_ofc, p_vals_ofc = find_active_units_by_midpoint(
    OFC_all,
    alpha=0.05,
    before_range=(int(-1*bin_to_sec), 0),
    after_range=(0, int(1*bin_to_sec)),
)

print(f"\n=== OFC Active Units ===")
print(f"Found {len(active_units_ofc)} active OFC units")

# %% # Plot and save raw PSTH for all OFC responsive units
save_raw_psth_for_active_units(
    ofc_event_windows_data,
    active_units_ofc,
    p_vals_ofc,
    "OFC",
    results_dir,
    display_window=(-0.5, 1.0),
)

# %% # Compute selectivity metrics for OFC
if len(active_units_ofc) > 0:
    ofc_selectivity_df = compute_selectivity_metrics_for_active_units(
        ofc_event_windows_data,
        active_units_ofc,
        window=(-0.1, 0.5),
    )
    
    print(f"\nSelectivity metrics for {len(ofc_selectivity_df)} OFC active units:")
    if len(ofc_selectivity_df) > 0 and 'stimulus_selective' in ofc_selectivity_df.columns:
        print(f"  Stimulus selective: {ofc_selectivity_df['stimulus_selective'].sum()}")
        print(f"  Outcome modulated (p<0.05): {ofc_selectivity_df['outcome_modulated'].sum()}")
        print(f"  Go/NoGo selective (|d'|>0.5): {ofc_selectivity_df['go_nogo_selective'].sum()}")
        print(f"  Choice coding (|CP_corr|>0.1): {ofc_selectivity_df['choice_coding'].sum()}")
    else:
        print("  No active units found - skipping selectivity metrics")
    
    # Save OFC selectivity results
    print("\n=== Saving OFC selectivity results ===")
    save_dataframe_to_csv(
        ofc_selectivity_df,
        os.path.join(results_dir, "tables", "ofc_selectivity_metrics.csv"),
        "OFC selectivity metrics table"
    )
    
    # Create OFC summary plots
    ofc_fig_metrics, ofc_fig_class = plot_selectivity_summary(ofc_selectivity_df, region_name="OFC")
    
    # Save OFC summary plots
    print("\n=== Saving OFC summary plots ===")
    save_plot_to_html(
        ofc_fig_metrics,
        os.path.join(results_dir, "plots", "ofc", "ofc_selectivity_metrics_summary.html"),
        "OFC selectivity metrics summary"
    )
    save_plot_to_html(
        ofc_fig_class,
        os.path.join(results_dir, "plots", "ofc", "ofc_unit_classification_summary.html"),
        "OFC unit classification summary"
    )
    
    # Comparison plot: OFC vs ACx
    # Check if we have data for comparison
    has_acx_data = len(acx_selectivity_df) > 0 and 'stimulus_selective' in acx_selectivity_df.columns
    has_ofc_data = len(ofc_selectivity_df) > 0 and 'stimulus_selective' in ofc_selectivity_df.columns
    
    if has_acx_data or has_ofc_data:
        fig_comparison = go.Figure()
        
        regions = ['ACx', 'OFC']
        metrics = ['Stimulus Selective', 'Outcome Modulated', 'Go/NoGo Selective', 'Choice Coding']
        
        if has_acx_data:
            acx_counts = [
                acx_selectivity_df['stimulus_selective'].sum(),
                acx_selectivity_df['outcome_modulated'].sum(),
                acx_selectivity_df['go_nogo_selective'].sum(),
                acx_selectivity_df['choice_coding'].sum(),
            ]
            # Normalize by number of active units
            acx_proportions = [c / len(acx_selectivity_df) * 100 for c in acx_counts]
        else:
            acx_counts = [0, 0, 0, 0]
            acx_proportions = [0, 0, 0, 0]
        
        if has_ofc_data:
            ofc_counts = [
                ofc_selectivity_df['stimulus_selective'].sum(),
                ofc_selectivity_df['outcome_modulated'].sum(),
                ofc_selectivity_df['go_nogo_selective'].sum(),
                ofc_selectivity_df['choice_coding'].sum(),
            ]
            # Normalize by number of active units
            ofc_proportions = [c / len(ofc_selectivity_df) * 100 for c in ofc_counts]
        else:
            ofc_counts = [0, 0, 0, 0]
            ofc_proportions = [0, 0, 0, 0]
        
        x_pos = np.arange(len(metrics))
        width = 0.35
        
        fig_comparison.add_trace(go.Bar(
            x=metrics,
            y=acx_proportions,
            name='ACx',
            marker_color='#1f77b4',
            text=[f'{c} ({p:.1f}%)' for c, p in zip(acx_counts, acx_proportions)],
            textposition='outside',
            hovertemplate='ACx<br>%{x}<br>%{y:.1f}% (%{text})<extra></extra>'
        ))
        
        fig_comparison.add_trace(go.Bar(
            x=metrics,
            y=ofc_proportions,
            name='OFC',
            marker_color='#ff7f0e',
            text=[f'{c} ({p:.1f}%)' for c, p in zip(ofc_counts, ofc_proportions)],
            textposition='outside',
            hovertemplate='OFC<br>%{x}<br>%{y:.1f}% (%{text})<extra></extra>'
        ))
        
        fig_comparison.update_layout(
            title='OFC vs ACx - Selectivity Comparison (Proportion of Active Units)',
            xaxis_title='Selectivity Type',
            yaxis_title='Percentage of Active Units',
            barmode='group',
            template='plotly_white',
            height=500,
            legend=dict(x=0.7, y=0.95)
        )
        
        # Save comparison plot
        print("\n=== Saving comparison plots ===")
        save_plot_to_html(
            fig_comparison,
            os.path.join(results_dir, "plots", "comparison", "ofc_vs_acx_selectivity_comparison.html"),
            "OFC vs ACx selectivity comparison"
        )
        
        # Summary statistics comparison
        print("\n=== OFC vs ACx Comparison ===")
        comparison_data = {
            'Region': ['ACx', 'OFC'],
            'Total Active Units': [len(acx_selectivity_df), len(ofc_selectivity_df)],
            'Stimulus Selective': [acx_counts[0], ofc_counts[0]],
            'Outcome Modulated': [acx_counts[1], ofc_counts[1]],
            'Go/NoGo Selective': [acx_counts[2], ofc_counts[2]],
            'Choice Coding': [acx_counts[3], ofc_counts[3]],
        }
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        
        # Save comparison table
        save_dataframe_to_csv(
            comparison_df,
            os.path.join(results_dir, "tables", "ofc_vs_acx_comparison.csv"),
            "OFC vs ACx comparison table"
        )
    else:
        print("\n=== Skipping OFC vs ACx comparison (no active units in either region) ===")
    
    # Plot example units from OFC
    print("\n=== Plotting example OFC active units ===")
    n_units_to_plot_ofc = min(3, len(active_units_ofc))
    for i, unit_idx in enumerate(active_units_ofc[:n_units_to_plot_ofc]):
        print(f"\n=== OFC Unit {unit_idx} (Rank {i+1}) ===")
        
        if 'stimulus' in ofc_stimuli_outcome_df.columns:
            fig_stim_ofc = plot_psth_by_stimulus(
                ofc_event_windows_data,
                int(unit_idx),
                display_window=(-0.5, 1.0),
                region_name="OFC"
            )
        
        if 'outcome' in ofc_stimuli_outcome_df.columns:
            fig_outcome_ofc = plot_psth_by_outcome(
                ofc_event_windows_data,
                int(unit_idx),
                display_window=(-0.5, 1.0),
                region_name="OFC"
            )
            
            # Save OFC PSTH plots
            if 'stimulus' in ofc_stimuli_outcome_df.columns:
                save_plot_to_html(
                    fig_stim_ofc,
                    os.path.join(results_dir, "plots", "psth_by_stimulus", f"ofc_unit_{unit_idx}_psth_by_stimulus.html"),
                    f"OFC Unit {unit_idx} PSTH by Stimulus"
                )
            save_plot_to_html(
                fig_outcome_ofc,
                os.path.join(results_dir, "plots", "psth_by_outcome", f"ofc_unit_{unit_idx}_psth_by_outcome.html"),
                f"OFC Unit {unit_idx} PSTH by Outcome"
            )
else:
    print("No active OFC units found for comparison")

# %% Category sensitivity analysis 
def assign_stimulus_categories(
    stimuli: np.ndarray,
    low_boundary: float = 0.983,
    high_boundary: float = 1.525,
) -> np.ndarray:
    """
    Assign each stimulus to a category based on boundaries.
    
    Categories:
    - 'Low': stimulus < low_boundary
    - 'Middle': low_boundary <= stimulus <= high_boundary
    - 'High': stimulus > high_boundary
    
    Parameters:
    -----------
    stimuli : np.ndarray
        Array of stimulus values
    low_boundary : float
        Lower category boundary (default: 0.983)
    high_boundary : float
        Upper category boundary (default: 1.525)
    
    Returns:
    --------
    np.ndarray
        Array of category labels ('Low', 'Middle', 'High')
    """
    categories = np.full(len(stimuli), 'Middle', dtype=object)
    categories[stimuli < low_boundary] = 'Low'
    categories[stimuli > high_boundary] = 'High'
    return categories

def compute_category_sensitivity(
    event_windows_data: tuple,
    unit_idx: int,
    low_boundary: float = 0.983,
    high_boundary: float = 1.525,
    window: tuple[float, float] = (-0.2, 1),
) -> dict:
    """
    Test if a unit is sensitive to stimulus categories defined by boundaries.
    
    Uses one-way ANOVA to test for differences in firing rates across categories.
    
    Parameters:
    -----------
    event_windows_data : tuple
        Event windows data tuple
    unit_idx : int
        Unit index to test
    low_boundary : float
        Lower category boundary
    high_boundary : float
        Upper category boundary
    window : tuple[float, float]
        Time window for analysis (start, end) in seconds
    
    Returns:
    --------
    dict
        Dictionary with category sensitivity metrics
    """
    # Handle both 5-tuple and 6-tuple formats
    if len(event_windows_data) == 6:
        event_windows_matrix, time_axis, valid_event_indices, stimuli_outcome_df, metadata, _ = event_windows_data
    else:
        event_windows_matrix, time_axis, valid_event_indices, stimuli_outcome_df, metadata = event_windows_data
    
    if 'stimulus' not in stimuli_outcome_df.columns:
        return {
            'category_sensitive': False,
            'category_anova_p': np.nan,
            'category_anova_f': np.nan,
            'low_mean_rate': np.nan,
            'middle_mean_rate': np.nan,
            'high_mean_rate': np.nan,
            'best_category': None,
            'go_nogo_dprime': np.nan,
            'go_nogo_roc_auc': np.nan,
            'go_nogo_selective': False,
            'go_mean_rate': np.nan,
            'nogo_mean_rate': np.nan,
        }
    
    # Get unit data
    unit_data = event_windows_matrix[unit_idx, :, :]  # [time × events]
    
    # Find time indices for analysis window
    start_idx = np.argmin(np.abs(time_axis - window[0]))
    end_idx = np.argmin(np.abs(time_axis - window[1]))
    unit_data_windowed = unit_data[start_idx:end_idx, :]
    
    # Mean firing rate per trial (across time window)
    mean_rates = np.mean(unit_data_windowed, axis=0)  # [trials]
    
    # Get stimuli for each trial
    stimuli = stimuli_outcome_df['stimulus'].values
    
    # Assign categories
    categories = assign_stimulus_categories(stimuli, low_boundary, high_boundary)
    
    # Group firing rates by category
    low_rates = mean_rates[categories == 'Low']
    middle_rates = mean_rates[categories == 'Middle']
    high_rates = mean_rates[categories == 'High']
    
    # Check if we have enough data in each category (at least 2 trials)
    has_low = len(low_rates) >= 2
    has_middle = len(middle_rates) >= 2
    has_high = len(high_rates) >= 2
    
    # Perform ANOVA if we have at least 2 categories with sufficient data
    category_groups = []
    category_names = []
    if has_low:
        category_groups.append(low_rates)
        category_names.append('Low')
    if has_middle:
        category_groups.append(middle_rates)
        category_names.append('Middle')
    if has_high:
        category_groups.append(high_rates)
        category_names.append('High')
    
    if len(category_groups) < 2:
        # Not enough categories for ANOVA, but still compute go/nogo if possible
        go_rates = np.concatenate([high_rates, low_rates]) if (has_high and has_low) else (
            high_rates if has_high else (low_rates if has_low else np.array([]))
        )
        nogo_rates = middle_rates if has_middle else np.array([])
        
        go_nogo_dprime = np.nan
        go_nogo_roc_auc = np.nan
        go_nogo_selective = False
        
        if len(go_rates) >= 2 and len(nogo_rates) >= 2:
            go_mean, go_std = np.mean(go_rates), np.std(go_rates)
            nogo_mean, nogo_std = np.mean(nogo_rates), np.std(nogo_rates)
            pooled_std = np.sqrt((go_std**2 + nogo_std**2) / 2)
            go_nogo_dprime = (go_mean - nogo_mean) / pooled_std if pooled_std > 0 else 0.0
            
            try:
                labels = np.concatenate([np.ones(len(go_rates)), np.zeros(len(nogo_rates))])
                scores = np.concatenate([go_rates, nogo_rates])
                go_nogo_roc_auc = roc_auc_score(labels, scores)
            except Exception:
                go_nogo_roc_auc = 0.5
            
            go_nogo_selective = abs(go_nogo_dprime) > 0.5
        
        return {
            'category_sensitive': False,
            'category_anova_p': np.nan,
            'category_anova_f': np.nan,
            'low_mean_rate': float(np.mean(low_rates)) if has_low else np.nan,
            'middle_mean_rate': float(np.mean(middle_rates)) if has_middle else np.nan,
            'high_mean_rate': float(np.mean(high_rates)) if has_high else np.nan,
            'best_category': None,
            'go_nogo_dprime': float(go_nogo_dprime) if not np.isnan(go_nogo_dprime) else np.nan,
            'go_nogo_roc_auc': float(go_nogo_roc_auc) if not np.isnan(go_nogo_roc_auc) else np.nan,
            'go_nogo_selective': go_nogo_selective,
            'go_mean_rate': float(np.mean(go_rates)) if len(go_rates) > 0 else np.nan,
            'nogo_mean_rate': float(np.mean(nogo_rates)) if len(nogo_rates) > 0 else np.nan,
        }
    
    # Perform one-way ANOVA
    f_stat, p_val = f_oneway(*category_groups)
    
    # Determine best category (highest mean rate)
    mean_rates_by_category = {}
    if has_low:
        mean_rates_by_category['Low'] = np.mean(low_rates)
    if has_middle:
        mean_rates_by_category['Middle'] = np.mean(middle_rates)
    if has_high:
        mean_rates_by_category['High'] = np.mean(high_rates)
    
    best_category = max(mean_rates_by_category, key=mean_rates_by_category.get) if mean_rates_by_category else None
    
    # Compute Go/NoGo selectivity: Go = High + Low, NoGo = Middle
    go_rates = np.concatenate([high_rates, low_rates]) if (has_high and has_low) else (
        high_rates if has_high else (low_rates if has_low else np.array([]))
    )
    nogo_rates = middle_rates if has_middle else np.array([])
    
    # Compute d' and ROC AUC for Go vs NoGo
    go_nogo_dprime = np.nan
    go_nogo_roc_auc = np.nan
    go_nogo_selective = False
    
    if len(go_rates) >= 2 and len(nogo_rates) >= 2:
        # Compute d'
        go_mean, go_std = np.mean(go_rates), np.std(go_rates)
        nogo_mean, nogo_std = np.mean(nogo_rates), np.std(nogo_rates)
        
        # Pooled standard deviation
        pooled_std = np.sqrt((go_std**2 + nogo_std**2) / 2)
        go_nogo_dprime = (go_mean - nogo_mean) / pooled_std if pooled_std > 0 else 0.0
        
        # Compute ROC AUC
        try:
            # Create labels: 1 for Go, 0 for NoGo
            labels = np.concatenate([np.ones(len(go_rates)), np.zeros(len(nogo_rates))])
            scores = np.concatenate([go_rates, nogo_rates])
            go_nogo_roc_auc = roc_auc_score(labels, scores)
        except Exception:
            go_nogo_roc_auc = 0.5
        
        # Consider selective if |d'| > 0.5 (same threshold as elsewhere in codebase)
        go_nogo_selective = abs(go_nogo_dprime) > 0.5
    
    return {
        'category_sensitive': p_val < 0.05,
        'category_anova_p': float(p_val),
        'category_anova_f': float(f_stat),
        'low_mean_rate': float(np.mean(low_rates)) if has_low else np.nan,
        'middle_mean_rate': float(np.mean(middle_rates)) if has_middle else np.nan,
        'high_mean_rate': float(np.mean(high_rates)) if has_high else np.nan,
        'low_n_trials': len(low_rates),
        'middle_n_trials': len(middle_rates),
        'high_n_trials': len(high_rates),
        'best_category': best_category,
        'go_nogo_dprime': float(go_nogo_dprime) if not np.isnan(go_nogo_dprime) else np.nan,
        'go_nogo_roc_auc': float(go_nogo_roc_auc) if not np.isnan(go_nogo_roc_auc) else np.nan,
        'go_nogo_selective': go_nogo_selective,
        'go_mean_rate': float(np.mean(go_rates)) if len(go_rates) > 0 else np.nan,
        'nogo_mean_rate': float(np.mean(nogo_rates)) if len(nogo_rates) > 0 else np.nan,
    }

def compute_category_sensitivity_for_all_units(
    event_windows_data: tuple,
    active_units: np.ndarray,
    low_boundary: float = 0.983,
    high_boundary: float = 1.525,
    window: tuple[float, float] = (-0.1, 0.5),
) -> pd.DataFrame:
    """
    Compute category sensitivity for all active units.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with category sensitivity metrics for each unit
    """
    results = []
    
    for unit_idx in active_units:
        unit_results = compute_category_sensitivity(
            event_windows_data,
            int(unit_idx),
            low_boundary=low_boundary,
            high_boundary=high_boundary,
            window=window,
        )
        unit_results['unit_idx'] = int(unit_idx)
        results.append(unit_results)
    
    return pd.DataFrame(results)

def plot_psth_by_category(
    event_windows_data: tuple,
    unit_idx: int,
    low_boundary: float = 0.983,
    high_boundary: float = 1.525,
    display_window: tuple[float, float] = (-0.5, 1.0),
    region_name: str = "Unit",
) -> go.Figure:
    """
    Plot PSTH separated by category (Low/Middle/High) for a single unit.
    """
    # Handle both 5-tuple and 6-tuple formats
    if len(event_windows_data) == 6:
        event_windows_matrix, time_axis, valid_event_indices, stimuli_outcome_df, metadata, _ = event_windows_data
    else:
        event_windows_matrix, time_axis, valid_event_indices, stimuli_outcome_df, metadata = event_windows_data
    
    if 'stimulus' not in stimuli_outcome_df.columns:
        print("No stimulus information available")
        return go.Figure()
    
    # Get unit data
    unit_data = event_windows_matrix[unit_idx, :, :]  # [time × events]
    
    # Find time indices for display window
    start_idx = np.argmin(np.abs(time_axis - display_window[0]))
    end_idx = np.argmin(np.abs(time_axis - display_window[1]))
    unit_data_windowed = unit_data[start_idx:end_idx, :]
    time_axis_windowed = time_axis[start_idx:end_idx]
    
    # Get stimuli and assign categories
    stimuli = stimuli_outcome_df['stimulus'].values
    categories = assign_stimulus_categories(stimuli, low_boundary, high_boundary)
    
    # Create figure
    fig = go.Figure()
    
    # Category colors
    category_colors = {
        'Low': '#2ca02c',      # Green
        'Middle': '#ff7f0e',   # Orange
        'High': '#d62728',     # Red
    }
    
    category_order = ['Low', 'Middle', 'High']
    
    for category in category_order:
        cat_mask = (categories == category)
        cat_trials = unit_data_windowed[:, cat_mask]
        
        if cat_trials.shape[1] > 0:
            psth_mean = np.mean(cat_trials, axis=1)
            psth_sem = np.std(cat_trials, axis=1) / np.sqrt(cat_trials.shape[1])
            
            color = category_colors[category]
            
            # Main trace
            fig.add_trace(go.Scatter(
                x=time_axis_windowed,
                y=psth_mean,
                mode='lines',
                name=f'{category} (n={cat_trials.shape[1]})',
                line=dict(color=color, width=2)
            ))
            
            # SEM shading
            fig.add_trace(go.Scatter(
                x=np.concatenate([time_axis_windowed, time_axis_windowed[::-1]]),
                y=np.concatenate([psth_mean + psth_sem, (psth_mean - psth_sem)[::-1]]),
                fill='toself',
                fillcolor=_hex_to_rgba(color, alpha=0.2),
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # Add vertical line at event onset
    fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)
    
    # Add boundary lines annotation
    fig.add_annotation(
        x=0.02, y=0.98,
        xref='paper', yref='paper',
        text=f'Boundaries: Low={low_boundary:.3f}, High={high_boundary:.3f}',
        showarrow=False,
        font=dict(size=10),
        bgcolor='rgba(255,255,255,0.8)',
        bordercolor='black',
        borderwidth=1,
        align='left'
    )
    
    fig.update_layout(
        title=f"{region_name} Unit {unit_idx} - PSTH by Category",
        xaxis_title="Time (s)",
        yaxis_title="Firing Rate (Hz)",
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig

def plot_category_sensitivity_summary(
    category_df: pd.DataFrame,
    region_name: str = "Region",
    low_boundary: float = 0.983,
    high_boundary: float = 1.525,
) -> go.Figure:
    """
    Create summary plot showing category sensitivity across units.
    """
    if category_df is None or len(category_df) == 0:
        fig = go.Figure()
        fig.update_layout(
            title=f"{region_name} - No category sensitivity data",
            template="plotly_white",
            height=400,
        )
        return fig
    
    fig = go.Figure()
    
    # Scatter plot: -log10(p-value) vs unit index
    valid_p = ~np.isnan(category_df['category_anova_p'])
    if np.any(valid_p):
        p_vals = category_df.loc[valid_p, 'category_anova_p']
        unit_indices = category_df.loc[valid_p, 'unit_idx']
        neg_log10_p = -np.log10(p_vals + 1e-10)
        
        # Color by significance
        colors = ['red' if p < 0.05 else 'gray' for p in p_vals]
        
        fig.add_trace(go.Scatter(
            x=unit_indices,
            y=neg_log10_p,
            mode='markers',
            name='Category Sensitivity (-log10 p)',
            marker=dict(
                color=colors,
                size=8,
                symbol='circle',
                line=dict(width=1, color='black')
            ),
            hovertemplate='Unit %{x}<br>-log10(p) = %{y:.2f}<br>p = %{customdata:.4f}<extra></extra>',
            customdata=p_vals
        ))
        
        # Add significance threshold
        fig.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="red", 
                     annotation_text="p=0.05", annotation_position="right")
    
    fig.update_layout(
        title=f"{region_name} - Category Sensitivity Summary (Boundaries: {low_boundary:.3f}, {high_boundary:.3f})",
        xaxis_title="Unit Index",
        yaxis_title="-log10(ANOVA p-value)",
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    return fig

# %% # Run category sensitivity analysis for ACx
print("\n=== Category Sensitivity Analysis ===")
print(f"Using boundaries: Low={0.983:.3f}, High={1.525:.3f}")

# Compute category sensitivity for ACx active units
acx_category_df = compute_category_sensitivity_for_all_units(
    acx_event_windows_data,
    active_units_acx,
    low_boundary=0.983,
    high_boundary=1.525,
    window=(-0.1, 0.5),
)

print(f"\nACx Category Sensitivity Results:")
if len(acx_category_df) > 0:
    n_sensitive = acx_category_df['category_sensitive'].sum()
    print(f"  Category-sensitive units: {n_sensitive} / {len(acx_category_df)}")
    print(f"  Significant units (p<0.05): {n_sensitive}")
    
    # Show go/nogo selectivity (Go = High+Low, NoGo = Middle)
    if 'go_nogo_selective' in acx_category_df.columns:
        n_go_nogo = acx_category_df['go_nogo_selective'].sum()
        print(f"  Go/NoGo selective units (|d'|>0.5): {n_go_nogo} / {len(acx_category_df)}")
    
    # Show best category distribution
    if 'best_category' in acx_category_df.columns:
        best_cat_counts = acx_category_df['best_category'].value_counts()
        print(f"  Best category distribution:")
        for cat, count in best_cat_counts.items():
            print(f"    {cat}: {count}")
    
    # Save ACx category sensitivity results
    save_dataframe_to_csv(
        acx_category_df,
        os.path.join(results_dir, "tables", "acx_category_sensitivity.csv"),
        "ACx category sensitivity table"
    )
    
    # Create and save summary plot
    acx_category_fig = plot_category_sensitivity_summary(
        acx_category_df,
        region_name="ACx",
        low_boundary=0.983,
        high_boundary=1.525,
    )
    save_plot_to_html(
        acx_category_fig,
        os.path.join(results_dir, "plots", "acx", "acx_category_sensitivity_summary.html"),
        "ACx category sensitivity summary"
    )
    
    # Plot PSTH by category for top category-sensitive units
    if n_sensitive > 0:
        # Sort by p-value (most significant first)
        sensitive_units = acx_category_df[acx_category_df['category_sensitive']].copy()
        sensitive_units = sensitive_units.sort_values('category_anova_p')
        
        n_to_plot = min(5, len(sensitive_units))
        print(f"\n  Plotting PSTH by category for top {n_to_plot} category-sensitive units:")
        
        for i, row in sensitive_units.head(n_to_plot).iterrows():
            unit_idx = int(row['unit_idx'])
            p_val = row['category_anova_p']
            print(f"    Unit {unit_idx} (p={p_val:.4f})")
            
            fig_cat = plot_psth_by_category(
                acx_event_windows_data,
                unit_idx,
                low_boundary=0.983,
                high_boundary=1.525,
                display_window=(-0.5, 1.0),
                region_name="ACx"
            )
            save_plot_to_html(
                fig_cat,
                os.path.join(results_dir, "plots", "psth_by_category", f"acx_unit_{unit_idx}_psth_by_category.html"),
                f"ACx Unit {unit_idx} PSTH by Category"
            )
else:
    print("  No active units to analyze")

# %% # Run category sensitivity analysis for OFC (using all good units, not just active units)
# Filter good units to only include those that exist in the event_windows_matrix
ofc_event_matrix, _, _, _, _, _ = ofc_event_windows_data
n_units_in_matrix = ofc_event_matrix.shape[0]
ofc_good_units_filtered = [idx for idx in OFC_g if idx < n_units_in_matrix]
ofc_good_units_array = np.array(ofc_good_units_filtered)  # Convert list to numpy array

if len(ofc_good_units_array) > 0:
    print(f"\n=== OFC Category Sensitivity Analysis (using {len(ofc_good_units_array)} good units out of {len(OFC_g)} total good units) ===")
    print(f"  Event matrix has {n_units_in_matrix} units, filtered {len(OFC_g) - len(ofc_good_units_array)} units that are out of bounds")
    ofc_category_df = compute_category_sensitivity_for_all_units(
        ofc_event_windows_data,
        ofc_good_units_array,
        low_boundary=0.983,
        high_boundary=1.525,
        window=(-0.1, 0.5),
    )
    
    print(f"\nOFC Category Sensitivity Results:")
    if len(ofc_category_df) > 0:
        n_sensitive_ofc = ofc_category_df['category_sensitive'].sum()
        print(f"  Category-sensitive units: {n_sensitive_ofc} / {len(ofc_category_df)}")
        
        # Show go/nogo selectivity (Go = High+Low, NoGo = Middle)
        if 'go_nogo_selective' in ofc_category_df.columns:
            n_go_nogo_ofc = ofc_category_df['go_nogo_selective'].sum()
            print(f"  Go/NoGo selective units (|d'|>0.5): {n_go_nogo_ofc} / {len(ofc_category_df)}")
        
        # Show best category distribution
        if 'best_category' in ofc_category_df.columns:
            best_cat_counts = ofc_category_df['best_category'].value_counts()
            print(f"  Best category distribution:")
            for cat, count in best_cat_counts.items():
                print(f"    {cat}: {count}")
        
        # Save OFC category sensitivity results
        save_dataframe_to_csv(
            ofc_category_df,
            os.path.join(results_dir, "tables", "ofc_category_sensitivity.csv"),
            "OFC category sensitivity table"
        )
        
        # Create and save summary plot
        ofc_category_fig = plot_category_sensitivity_summary(
            ofc_category_df,
            region_name="OFC",
            low_boundary=0.983,
            high_boundary=1.525,
        )
        save_plot_to_html(
            ofc_category_fig,
            os.path.join(results_dir, "plots", "ofc", "ofc_category_sensitivity_summary.html"),
            "OFC category sensitivity summary"
        )
        
        # Plot PSTH by category for top category-sensitive units
        if n_sensitive_ofc > 0:
            sensitive_units_ofc = ofc_category_df[ofc_category_df['category_sensitive']].copy()
            sensitive_units_ofc = sensitive_units_ofc.sort_values('category_anova_p')
            
            n_to_plot_ofc = min(5, len(sensitive_units_ofc))
            print(f"\n  Plotting PSTH by category for top {n_to_plot_ofc} category-sensitive units:")
            
            for i, row in sensitive_units_ofc.head(n_to_plot_ofc).iterrows():
                unit_idx = int(row['unit_idx'])
                p_val = row['category_anova_p']
                print(f"    Unit {unit_idx} (p={p_val:.4f})")
                
                fig_cat_ofc = plot_psth_by_category(
                    ofc_event_windows_data,
                    unit_idx,
                    low_boundary=0.983,
                    high_boundary=1.525,
                    display_window=(-0.5, 1.0),
                    region_name="OFC"
                )
                save_plot_to_html(
                    fig_cat_ofc,
                    os.path.join(results_dir, "plots", "psth_by_category", f"ofc_unit_{unit_idx}_psth_by_category.html"),
                    f"OFC Unit {unit_idx} PSTH by Category"
                )
    else:
        print("  No good units to analyze")
else:
    print("\nOFC: No good units found for category sensitivity analysis")

# %%
print(f"\n=== Analysis complete! All results saved to: {results_dir} ===")

# %%
