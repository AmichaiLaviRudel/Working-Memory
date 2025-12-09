"""
Utility functions for NPXL offline analysis.
"""
import os
import json
import time
import pandas as pd
import plotly.graph_objects as go


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
    os.makedirs(os.path.join(results_dir, "plots", "heatmap"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "plots", "heatmap", "choice_aligned"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "plots", "heatmap", "outcome_aligned"), exist_ok=True)
    
    print(f"Results will be saved to: {results_dir}")
    return results_dir


def save_dataframe_to_csv(df: pd.DataFrame, filepath: str, description: str = ""):
    """
    Save DataFrame to CSV file.
    Serializes list columns as JSON strings for proper CSV storage.
    """
    # Create a copy to avoid modifying the original DataFrame
    df_copy = df.copy()
    
    # Convert list columns to JSON strings for CSV compatibility
    for col in df_copy.columns:
        # Check if column contains lists
        if df_copy[col].dtype == 'object':
            # Sample non-null values to check if they're lists
            non_null_values = df_copy[col].dropna()
            if len(non_null_values) > 0:
                # Check if any values are lists
                has_lists = any(isinstance(val, list) for val in non_null_values.head(10))
                if has_lists:
                    # Convert lists to JSON strings, handle empty lists and None/NaN
                    df_copy[col] = df_copy[col].apply(
                        lambda x: json.dumps(x) if isinstance(x, list) else (json.dumps([]) if pd.isna(x) else x)
                    )
    
    df_copy.to_csv(filepath, index=False)
    if description:
        print(f"  Saved {description} to: {filepath}")
        # Debug: print columns that contain lists
        list_cols = [col for col in df.columns if df[col].dtype == 'object' and 
                     any(isinstance(val, list) for val in df[col].dropna().head(5) if len(df[col].dropna()) > 0)]
        if list_cols:
            print(f"    Serialized list columns: {list_cols}")
    else:
        print(f"  Saved DataFrame to: {filepath}")


def save_plot_to_html(fig: go.Figure, filepath: str, description: str = ""):
    """
    Save Plotly figure to HTML file with error handling for permission issues.
    
    Handles common Windows permission errors by:
    - Ensuring directory exists
    - Attempting to remove locked files
    - Retrying with delay if initial write fails
    """
    # Ensure directory exists
    dir_path = os.path.dirname(filepath)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    
    # Try to remove existing file if it exists (may be locked by browser)
    max_retries = 3
    retry_delay = 0.5  # seconds
    
    for attempt in range(max_retries):
        try:
            # If file exists, try to remove it first (may be locked by browser)
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    if attempt > 0:
                        time.sleep(retry_delay)  # Brief pause after removal on retry
                except (OSError, PermissionError):
                    # File is locked, wait and retry
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
            
            # Write the file
            fig.write_html(filepath)
            
            if description:
                print(f"  Saved {description} to: {filepath}")
            else:
                print(f"  Saved plot to: {filepath}")
            return  # Success, exit function
            
        except (OSError, PermissionError) as e:
            if attempt < max_retries - 1:
                print(f"  Warning: Permission denied writing to {filepath} (attempt {attempt + 1}/{max_retries}). Retrying...")
                time.sleep(retry_delay)
            else:
                # Final attempt failed
                error_msg = (
                    f"Failed to save plot after {max_retries} attempts: {filepath}\n"
                    f"Error: {e}\n"
                    f"Possible causes:\n"
                    f"  - File is open in a browser or another program\n"
                    f"  - Insufficient write permissions\n"
                    f"  - File path too long (Windows 260 char limit)\n"
                    f"Please close any programs using this file and try again."
                )
                raise PermissionError(error_msg) from e


def units_to_dataframe(
    units: list,
    compute_all_metrics: bool = True,
    selectivity_window: tuple[float, float] = (-0.1, 0.5),
    category_window: tuple[float, float] = (-0.1, 0.5),
    category_boundaries: tuple[float, float] = (0.983, 1.525),
    psth_baseline_window: tuple[float, float] = (-0.5, 0),
) -> pd.DataFrame:
    """
    Convert a list of Unit objects to a comprehensive DataFrame with all metrics.
    
    Parameters:
    -----------
    units : list[Unit]
        List of Unit objects to process
    compute_all_metrics : bool
        If True, compute all metrics for units that haven't been computed yet
    selectivity_window : tuple[float, float]
        Time window for selectivity analysis
    category_window : tuple[float, float]
        Time window for category sensitivity analysis
    category_boundaries : tuple[float, float]
        Category boundaries (low, high)
    psth_baseline_window : tuple[float, float]
        Baseline window for PSTH metrics
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with one row per unit and all computed metrics as columns
    """
    # Lazy import to avoid circular dependencies
    from Analysis.NPXL_analysis.single_unit_offline_analysis.unit import Unit
    
    rows = []
    
    for i, unit in enumerate(units):
        if not isinstance(unit, Unit):
            continue
        
        # Start with basic unit information
        row = {
            "unit_idx": unit.unit_idx,
            "region_name": unit.region_name,
            "n_trials": unit.n_trials,
            "n_time_bins": unit.n_time_bins,
            "bin_size": unit.bin_size,
            "window_duration": unit.window_duration,
            "mean_firing_rate": unit.get_mean_firing_rate(),
        }
        
        # Compute or get selectivity metrics
        if compute_all_metrics or unit._selectivity_metrics is None:
            selectivity = unit.compute_selectivity(window=selectivity_window)
        else:
            selectivity = unit._selectivity_metrics
        
        if selectivity:
            # Flatten selectivity metrics
            row.update({
                "stimulus_selective": selectivity.get("stimulus_selective", False),
                "best_stimulus": selectivity.get("best_stimulus"),
                "max_stimulus_response": selectivity.get("max_stimulus_response"),
                "min_stimulus_response": selectivity.get("min_stimulus_response"),
                "outcome_p_value": selectivity.get("outcome_p_value"),
                "outcome_modulated": selectivity.get("outcome_modulated", False),
                "rewarded_mean_rate": selectivity.get("rewarded_mean_rate"),
                "non_rewarded_mean_rate": selectivity.get("non_rewarded_mean_rate"),
                "go_nogo_dprime": selectivity.get("go_nogo_dprime"),
                "go_nogo_roc_auc": selectivity.get("go_nogo_roc_auc"),
                "go_nogo_selective": selectivity.get("go_nogo_selective", False),
                "choice_probability": selectivity.get("choice_probability"),
                "choice_probability_corr": selectivity.get("choice_probability_corr"),
                "choice_coding": selectivity.get("choice_coding", False),
            })
        
        # Compute or get category sensitivity
        if compute_all_metrics or unit._category_sensitivity is None:
            category_sens = unit.compute_category_sensitivity(
                low_boundary=category_boundaries[0],
                high_boundary=category_boundaries[1],
                window=category_window
            )
        else:
            category_sens = unit._category_sensitivity
        
        if category_sens:
            # Flatten category sensitivity metrics
            row.update({
                "category_sensitive": category_sens.get("category_sensitive", False),
                "category_anova_p": category_sens.get("category_anova_p"),
                "category_anova_f": category_sens.get("category_anova_f"),
                "go_mean_rate": category_sens.get("go_mean_rate"),
                "nogo_mean_rate": category_sens.get("nogo_mean_rate"),
                "go_n_trials": category_sens.get("go_n_trials"),
                "nogo_n_trials": category_sens.get("nogo_n_trials"),
                "best_category": category_sens.get("best_category"),
            })
        
        # Compute or get PSTH metrics
        if compute_all_metrics or unit._psth_metrics is None:
            psth_metrics = unit.compute_psth_metrics(baseline_window=psth_baseline_window)
        else:
            psth_metrics = unit._psth_metrics
        
        if psth_metrics:
            # Flatten PSTH metrics
            suppression_metrics = psth_metrics.get("suppression_metrics", {})
            if isinstance(suppression_metrics, dict):
                row.update({
                    "psth_onset_latency": psth_metrics.get("onset_latency"),
                    "psth_peak_latency": psth_metrics.get("peak_latency"),
                    "psth_response_magnitude": psth_metrics.get("response_magnitude"),
                    "psth_response_type": psth_metrics.get("response_type"),
                    "psth_fwhm": psth_metrics.get("fwhm"),
                    "psth_rise_time": psth_metrics.get("rise_time"),
                    "psth_decay_time": psth_metrics.get("decay_time"),
                    "psth_trial_variability": psth_metrics.get("trial_variability"),
                    "psth_signal_to_noise": psth_metrics.get("signal_to_noise"),
                    "psth_baseline_rate": psth_metrics.get("baseline_rate"),
                    "psth_peak_rate": psth_metrics.get("peak_rate"),
                    "psth_suppression_magnitude": suppression_metrics.get("magnitude"),
                    "psth_suppression_duration": suppression_metrics.get("duration"),
                    "psth_fraction_suppressed": suppression_metrics.get("fraction_suppressed"),
                })
            else:
                row.update({
                    "psth_onset_latency": psth_metrics.get("onset_latency"),
                    "psth_peak_latency": psth_metrics.get("peak_latency"),
                    "psth_response_magnitude": psth_metrics.get("response_magnitude"),
                    "psth_response_type": psth_metrics.get("response_type"),
                    "psth_fwhm": psth_metrics.get("fwhm"),
                    "psth_rise_time": psth_metrics.get("rise_time"),
                    "psth_decay_time": psth_metrics.get("decay_time"),
                    "psth_trial_variability": psth_metrics.get("trial_variability"),
                    "psth_signal_to_noise": psth_metrics.get("signal_to_noise"),
                    "psth_baseline_rate": psth_metrics.get("baseline_rate"),
                    "psth_peak_rate": psth_metrics.get("peak_rate"),
                })
        
        # Get trial statistics
        trial_stats = unit.get_trial_stats()
        if trial_stats:
            total_trials = (
                trial_stats.get("Hit", 0) + 
                trial_stats.get("Miss", 0) + 
                trial_stats.get("FA", 0) + 
                trial_stats.get("CR", 0)
            )
            row.update({
                "trial_count_hit": trial_stats.get("Hit", 0),
                "trial_count_miss": trial_stats.get("Miss", 0),
                "trial_count_fa": trial_stats.get("FA", 0),
                "trial_count_cr": trial_stats.get("CR", 0),
                "trial_count_total": trial_stats.get("Total", total_trials),
            })
        
        # Get GLM results if computed
        if unit._glm_results is not None:
            glm = unit._glm_results
            row.update({
                "glm_r_squared": glm.get("r_squared"),
                "glm_intercept": glm.get("intercept"),
                "glm_coef_stimulus": glm.get("coefficients", [None, None, None])[0] if len(glm.get("coefficients", [])) > 0 else None,
                "glm_coef_trial_type": glm.get("coefficients", [None, None, None])[1] if len(glm.get("coefficients", [])) > 1 else None,
                "glm_coef_outcome": glm.get("coefficients", [None, None, None])[2] if len(glm.get("coefficients", [])) > 2 else None,
            })
        
        # Add unit labels if available
        if unit._unit_labels is not None and unit.unit_idx < len(unit._unit_labels.index):
            try:
                unit_label_row = unit._unit_labels.iloc[unit.unit_idx]
                # Add unit label columns (avoid conflicts by prefixing)
                for col in unit_label_row.index:
                    row[f"label_{col}"] = unit_label_row[col]
            except (IndexError, KeyError):
                # Unit index not in labels DataFrame, skip
                pass
        
        # Add plot paths if available
        plot_paths = unit.plot_paths
        row["heatmap_tone_path"] = plot_paths.get("heatmap", "")
        row["plot_path_psth_by_category"] = plot_paths.get("psth_by_category", "")
        row["plot_path_raw_psth"] = plot_paths.get("raw_psth", "")
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def save_units_to_csv(
    units: list,
    filepath: str,
    compute_all_metrics: bool = True,
    selectivity_window: tuple[float, float] = (-0.1, 0.5),
    category_window: tuple[float, float] = (-0.1, 0.5),
    category_boundaries: tuple[float, float] = (0.983, 1.525),
    psth_baseline_window: tuple[float, float] = (-0.5, 0),
    description: str = "",
) -> pd.DataFrame:
    """
    Convert Unit objects to DataFrame and save to CSV.
    
    Parameters:
    -----------
    units : list[Unit]
        List of Unit objects to process
    filepath : str
        Path to save CSV file
    compute_all_metrics : bool
        If True, compute all metrics for units that haven't been computed yet
    selectivity_window : tuple[float, float]
        Time window for selectivity analysis
    category_window : tuple[float, float]
        Time window for category sensitivity analysis
    category_boundaries : tuple[float, float]
        Category boundaries (low, high)
    psth_baseline_window : tuple[float, float]
        Baseline window for PSTH metrics
    description : str
        Description for logging
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with all unit metrics
    """
    df = units_to_dataframe(
        units,
        compute_all_metrics=compute_all_metrics,
        selectivity_window=selectivity_window,
        category_window=category_window,
        category_boundaries=category_boundaries,
        psth_baseline_window=psth_baseline_window,
    )
    
    save_dataframe_to_csv(df, filepath, description=description or f"Unit metrics table ({len(units)} units)")
    
    return df

