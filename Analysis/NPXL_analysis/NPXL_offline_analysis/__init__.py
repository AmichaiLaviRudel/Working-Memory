"""
NPXL Offline Analysis Package

This package provides modules for offline analysis of NPXL (Neuropixels) data,
including active unit detection, selectivity metrics, visualization, and
category sensitivity analysis.
"""

__version__ = "1.0.0"

# Main exports
from Analysis.NPXL_analysis.single_unit_offline_analysis.unit import Unit, create_units_from_event_data

__all__ = [
    "Unit",
    "create_units_from_event_data",
]

