"""
Configuration and setup for NPXL offline analysis.

Handles imports, path setup, and Plotly configuration.
"""
import sys
import os
import warnings
import plotly.io as pio

# Add the workspace root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
# If we're in the single_unit_offline_analysis folder, go up 3 levels to reach workspace root
if 'single_unit_offline_analysis' in current_dir or 'NPXL_offline_analysis' in current_dir:
    workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
elif 'NPXL_analysis' in current_dir:
    workspace_root = os.path.dirname(os.path.dirname(current_dir))
else:
    # Fallback: try to find the workspace root by going up directories
    test_dir = current_dir
    for _ in range(4):  # Try going up at most 4 levels
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

# Import single unit metrics functions and add them to npxl_single_unit_analysis module
# This allows existing code to continue using npxl_single_unit_analysis.compute_* functions
from Analysis.NPXL_analysis.single_unit_offline_analysis.single_unit_metrics import (
    compute_stimulus_selectivity,
    compute_go_nogo_coding,
    compute_outcome_modulation,
    compute_choice_probability,
    compute_d_prime,
    compute_peri_event_rate,
    compute_peri_event_rate_from_event_windows,
    fit_glm_single_unit,
    calculate_psth_metrics,
)

# Add functions to npxl_single_unit_analysis module for backward compatibility
npxl_single_unit_analysis.compute_stimulus_selectivity = compute_stimulus_selectivity
npxl_single_unit_analysis.compute_go_nogo_coding = compute_go_nogo_coding
npxl_single_unit_analysis.compute_outcome_modulation = compute_outcome_modulation
npxl_single_unit_analysis.compute_choice_probability = compute_choice_probability
npxl_single_unit_analysis.compute_d_prime = compute_d_prime
npxl_single_unit_analysis.compute_peri_event_rate = compute_peri_event_rate
npxl_single_unit_analysis.compute_peri_event_rate_from_event_windows = compute_peri_event_rate_from_event_windows
npxl_single_unit_analysis.fit_glm_single_unit = fit_glm_single_unit
npxl_single_unit_analysis.calculate_psth_metrics = calculate_psth_metrics
from Analysis.GNG_bpod_analysis.colors import (
    OUTCOME_COLOR_MAP,
    SUBJECT_COLORS,
    GO_COLORS,
    NOGO_COLORS,
    COLOR_ACCENT,
    COLOR_ACCENT_TRANSPARENT,
    COLOR_GRAY,
    COLOR_HIT,
    COLOR_FA,
    COLOR_CR,
    COLOR_MISS,
)

# Configure Plotly renderer
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

# Suppress renderer warnings
warnings.filterwarnings('ignore', message='.*renderer.*', category=UserWarning)

def print_config():
    """Print configuration information."""
    print("Imports successful!")
    print(f"Workspace root: {workspace_root}")


