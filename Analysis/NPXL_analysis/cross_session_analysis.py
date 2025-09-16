import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from Analysis.NPXL_analysis.npxl_single_unit_analysis import (
    compute_psth_pvalues_from_event_windows,
    compute_stimulus_selectivity,
    compute_go_nogo_coding,
    compute_choice_probability,
)
from Analysis.GNG_bpod_analysis.colors import COLOR_ACCENT, COLOR_ORANGE
from load_data.load_bpod_data import find_mat_files_in_session_data, load_mat_file
from plotly.subplots import make_subplots

