"""
NPXL Offline Analysis Package

This package provides modules for offline analysis of NPXL (Neuropixels) data,
including active unit detection, selectivity metrics, visualization,
category sensitivity analysis, and GLM fitting.
"""

__version__ = "1.0.0"

# Main exports
from Analysis.NPXL_analysis.single_unit_offline_analysis.unit import Unit, create_units_from_event_data

# GLM functions - import only if available to avoid errors
try:
    from Analysis.NPXL_analysis.single_unit_offline_analysis.GLM.glm_fitting import (
        fit_glm_for_unit_nemos,
        fit_glm_for_all_units_nemos,
        build_nemos_time_series_inputs,
        make_nemos_bases,
        build_nemos_feature_matrix,
    )
    from Analysis.NPXL_analysis.single_unit_offline_analysis.GLM.design_matrix import (
        build_trial_design_matrix,
    )
    __all__ = [
        "Unit",
        "create_units_from_event_data",
        "fit_glm_for_unit_nemos",
        "fit_glm_for_all_units_nemos",
        "build_nemos_time_series_inputs",
        "make_nemos_bases",
        "build_nemos_feature_matrix",
        "build_trial_design_matrix",
    ]
except ImportError:
    # If GLM module has import issues, just export basic unit functions
    __all__ = [
        "Unit",
        "create_units_from_event_data",
    ]

