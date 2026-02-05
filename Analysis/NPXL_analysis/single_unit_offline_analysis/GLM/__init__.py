"""
GLM Analysis Package for Neural Data

This package provides tools for fitting Generalized Linear Models (GLMs)
to neural spike data with various predictors including:
- Temporal events (tone onset, licks, outcome)
- Categorical features (stimulus ID, category, outcome type)
- Spike history (self and cross-neuron coupling)

Modules:
--------
config : Configuration parameters and constants
colors : Color schemes and font styling
loading : Data loading functions
design_matrix : Design matrix building functions
plotting : Visualization functions
main_glm : Main analysis script

Usage:
------
Run the main analysis script:
    python main_glm.py

Or import individual modules:
    from loading import load_all_probes
    from design_matrix import build_design_matrix
    from plotting import plot_actual_vs_predicted
"""

from . import config
from . import colors
from . import loading
from . import design_matrix

# NOTE:
# `plotting.py` currently relies on non-package (script-style) imports.
# Importing it unconditionally breaks `import Analysis....GLM` which is needed
# by the Streamlit app. Keep it best-effort here to avoid hard failures.
try:
    from . import plotting  # type: ignore
except Exception:  # pragma: no cover
    plotting = None  # type: ignore

__version__ = "1.0.0"
__all__ = ['config', 'colors', 'loading', 'design_matrix', 'plotting']
