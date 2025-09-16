#!/usr/bin/env python3
print("DEBUG: Script started")

import sys
print(f"DEBUG: Python version {sys.version}")

try:
    import numpy as np
    print("DEBUG: NumPy imported")
    
    from sklearn.svm import SVC
    print("DEBUG: SVM imported")
    
    from Analysis.NPXL_analysis.NPXL_Preprocessing import load_event_windows_data
    print("DEBUG: Data loader imported")
    
    print("DEBUG: About to load data...")
    folder = r"Z:\Shared\Amichai\NPXL\Recs\group5\catgt_G5A3_1b_4t_new2_g0\G5A3_1b_4t_new2_g0_imec1\analysis_output"
    
    # Check if folder exists
    import os
    print(f"DEBUG: Folder exists: {os.path.exists(folder)}")
    
    # Try loading data
    event_windows_matrix, time_axis, valid_event_indices, stimuli_outcome_df, metadata, lick_event_windows_matrix = load_event_windows_data(folder)
    print("DEBUG: Data loaded successfully!")
    
    print(f"DEBUG: Matrix shape: {event_windows_matrix.shape}")
    print(f"DEBUG: Outcomes: {stimuli_outcome_df['outcome'].value_counts()}")
    
except Exception as e:
    print(f"DEBUG: Error occurred: {e}")
    import traceback
    traceback.print_exc()

print("DEBUG: Script completed")

