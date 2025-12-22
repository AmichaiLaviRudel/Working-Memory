"""
NPXL Offline Analysis Package

This package provides modules for offline analysis of NPXL (Neuropixels) data,
including active unit detection, selectivity metrics, visualization,
category sensitivity analysis, and GLM fitting.
"""

__version__ = "1.0.0"

# Main exports
from Analysis.NPXL_analysis.single_unit_offline_analysis.unit import Unit, create_units_from_event_data
from Analysis.NPXL_analysis.single_unit_offline_analysis.GLM.glm_fitting import (
    fit_glm_for_unit,
    fit_glm_for_all_units,
    build_design_matrix,
    fit_glm_poisson,
)

__all__ = [
    "Unit",
    "create_units_from_event_data",
    "fit_glm_for_unit",
    "fit_glm_for_all_units",
    "build_design_matrix",
    "fit_glm_poisson",
]

