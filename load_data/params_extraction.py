"""
Training Parameters and Performance Dataset Builder

Extracts training parameters from bpod .mat files and combines them with
performance metrics to create a comprehensive training-performance dataset.
"""

import scipy.io
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Tuple, Optional, Callable
from collections import Counter
import os


def extract_training_parameters(trial_settings: Any) -> Dict[str, List[Any]]:
    """
    Extract all GUI parameters from TrialSettings array.
    
    Args:
        trial_settings: TrialSettings array from bpod .mat file (SessionData.TrialSettings[0])
        
    Returns:
        Dictionary with parameter names as keys and lists of values per trial as values
    """
    param_dict: Dict[str, List[Any]] = {}
    n_trials = len(trial_settings)
    
    # Collect all parameter names from all trials (some parameters might not be in all trials)
    all_param_names = set()
    for i in range(n_trials):
        try:
            gui_struct = trial_settings[i]["GUI"][0, 0]
            if hasattr(gui_struct, 'dtype') and gui_struct.dtype.names:
                all_param_names.update(gui_struct.dtype.names)
        except (KeyError, IndexError, TypeError):
            continue
    
    # Extract parameter values for each trial
    for param_name in all_param_names:
        param_dict[param_name] = []
        for i in range(n_trials):
            try:
                gui_struct = trial_settings[i]["GUI"][0, 0]
                if hasattr(gui_struct, 'dtype') and gui_struct.dtype.names and param_name in gui_struct.dtype.names:
                    value = gui_struct[param_name]
                    # Handle nested arrays/structures
                    if isinstance(value, np.ndarray):
                        if value.size == 1:
                            value = value.item()
                        elif value.size == 0:
                            value = None
                        else:
                            # For multi-element arrays, convert to list
                            value = value.tolist()
                    elif isinstance(value, (np.void, np.generic)):
                        # Try to extract scalar value
                        try:
                            if hasattr(value, 'item'):
                                value = value.item()
                            elif hasattr(value, 'tolist'):
                                value = value.tolist()
                            else:
                                value = str(value)
                        except (ValueError, TypeError):
                            value = str(value)
                    # Convert numpy types to Python native types for better serialization
                    if isinstance(value, (np.integer, np.floating)):
                        value = value.item()
                    elif isinstance(value, np.bool_):
                        value = bool(value)
                    param_dict[param_name].append(value)
                else:
                    param_dict[param_name].append(None)
            except (KeyError, IndexError, TypeError, AttributeError) as e:
                # Parameter not available for this trial
                param_dict[param_name].append(None)
    
    return param_dict


def aggregate_parameters_per_session(param_dict: Dict[str, List[Any]]) -> Tuple[Dict[str, Any], str]:
    """
    Aggregate parameters per session using mode (most common value) and track changes.
    
    Args:
        param_dict: Dictionary with parameter names as keys and lists of values per trial
        
    Returns:
        Tuple of (aggregated_params_dict, parameter_changes_note)
        - aggregated_params_dict: Dictionary with mode values for each parameter
        - parameter_changes_note: String describing any parameter changes during the session
    """
    aggregated_params = {}
    changed_params = []
    
    for param_name, values in param_dict.items():
        # Filter out None values for mode calculation
        non_none_values = [v for v in values if v is not None]
        
        if not non_none_values:
            aggregated_params[param_name] = None
            continue
        
        # Calculate mode (most common value)
        # For numeric values, use Counter
        # For complex types, convert to string for comparison
        try:
            # Try to use Counter directly for hashable types
            value_counts = Counter(non_none_values)
            mode_value, mode_count = value_counts.most_common(1)[0]
            aggregated_params[param_name] = mode_value
        except TypeError:
            # For unhashable types (lists, arrays), convert to string
            str_values = [str(v) for v in non_none_values]
            str_value_counts = Counter(str_values)
            mode_str, mode_count = str_value_counts.most_common(1)[0]
            # Find the original value corresponding to the mode string
            mode_value = next(v for v in non_none_values if str(v) == mode_str)
            aggregated_params[param_name] = mode_value
        
        # Check if parameter changed during session
        unique_values = set(str(v) for v in non_none_values)
        if len(unique_values) > 1:
            # Parameter changed - track the changes
            # Group consecutive trials with same value
            value_ranges = []
            current_value = None
            start_trial = 0
            
            for i, val in enumerate(values):
                val_str = str(val) if val is not None else "None"
                if val_str != current_value:
                    if current_value is not None:
                        # End of previous range
                        end_trial = i - 1
                        if start_trial == end_trial:
                            value_ranges.append(f"{current_value} (trial {start_trial + 1})")
                        else:
                            value_ranges.append(f"{current_value} (trials {start_trial + 1}-{end_trial + 1})")
                    current_value = val_str
                    start_trial = i
            
            # Add final range
            if current_value is not None:
                end_trial = len(values) - 1
                if start_trial == end_trial:
                    value_ranges.append(f"{current_value} (trial {start_trial + 1})")
                else:
                    value_ranges.append(f"{current_value} (trials {start_trial + 1}-{end_trial + 1})")
            
            changed_params.append(f"{param_name}: [{', '.join(value_ranges)}]")
    
    # Create parameter changes note
    if changed_params:
        parameter_changes_note = "Parameter changes: " + "; ".join(changed_params)
    else:
        parameter_changes_note = "No parameter changes during session"
    
    return aggregated_params, parameter_changes_note


def extract_performance_metrics(project_data: pd.DataFrame, index: int) -> Dict[str, Any]:
    """
    Extract only essential performance metrics: hit rate, FA rate, and d' prime.
    
    Args:
        project_data: DataFrame with session data
        index: Index of the session row
        
    Returns:
        Dictionary with hit rate, FA rate, d' prime, and early response rates
    """
    from Analysis.GNG_bpod_analysis.metric import d_prime
    from Analysis.GNG_bpod_analysis.GNG_bpod_general import to_array
    
    # Initialize defaults
    hit = miss = cr = fa = 0
    early_go_n = early_nogo_n = n_go = n_nogo = 0
    early_go_rate = early_nogo_rate = 0.0
    hit_rate = fa_rate = 0.0
    d_mean = 0.0
    
    # Extract outcomes and trial types directly
    try:
        trialtypes_raw = project_data.iloc[index].get("TrialTypes", None)
        outcomes_raw = project_data.iloc[index].get("Outcomes", None)
        trialtypes_arr = to_array(trialtypes_raw)
        outcomes_arr = to_array(outcomes_raw)
        
        # Count responses directly from outcomes (more robust than using responses function)
        # Convert to string for comparison to handle different data types
        outcomes_str = [str(o).strip() for o in outcomes_arr]
        
        hit = int(np.sum([o == "Hit" for o in outcomes_str]))
        miss = int(np.sum([o == "Miss" for o in outcomes_str]))
        cr = int(np.sum([o == "CR" or o == "Correct Reject" for o in outcomes_str]))
        fa = int(np.sum([o == "False Alarm" for o in outcomes_str]))
        
        # Compute rates
        if (hit + miss) > 0:
            hit_rate = hit / (hit + miss)
        if (fa + cr) > 0:
            fa_rate = fa / (fa + cr)
        
        # Extract early response rates
        early_mask = np.array(
            ["early response" in str(o).lower() for o in outcomes_arr],
            dtype=bool,
        )
        trialtypes_norm = np.array([str(t).strip() for t in trialtypes_arr], dtype=object)
        go_mask = np.array([str(t).strip().lower() == "go" for t in trialtypes_norm], dtype=bool)
        nogo_mask = np.array([str(t).strip().lower().replace("-", "").replace(" ", "") == "nogo" for t in trialtypes_norm], dtype=bool)
        
        n_go = int(np.sum(go_mask))
        n_nogo = int(np.sum(nogo_mask))
        early_go_n = int(np.sum(early_mask & go_mask))
        early_nogo_n = int(np.sum(early_mask & nogo_mask))
        
        # Calculate early response rates
        early_go_rate = early_go_n / n_go if n_go > 0 else 0.0
        early_nogo_rate = early_nogo_n / n_nogo if n_nogo > 0 else 0.0
    except Exception as e:
        print(f"Error extracting response counts and early rates for index {index}: {e}")
        hit = miss = cr = fa = 0
    
    # Calculate d' (same as classifier_metric)
    try:
        d = d_prime(project_data, index, t=10)
        d_mean = float(np.nanmean(d))
    except Exception as e:
        print(f"Error calculating d' for index {index}: {e}")
        d_mean = 0.0
    
    # Validate rates
    if not np.isfinite(hit_rate):
        hit_rate = 0.0
    if not np.isfinite(fa_rate):
        fa_rate = 0.0
    
    return {
        # Response counts
        "Hit": int(hit),
        "Miss": int(miss),
        "CR": int(cr),
        "FA": int(fa),
        # Early response counts and rates
        "Early_Go_N": early_go_n,
        "Early_NoGo_N": early_nogo_n,
        "Early_Go_Rate": float(early_go_rate),
        "Early_NoGo_Rate": float(early_nogo_rate),
        "N_Go": n_go,
        "N_NoGo": n_nogo,
        # Essential performance metrics only
        "Hit_Rate": float(hit_rate),
        "False_Alarm_Rate": float(fa_rate),
        "d_prime": d_mean,
    }


def compute_metrics_for_loaded_data(
    df: pd.DataFrame,
    output_path: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> pd.DataFrame:
    """
    Compute performance metrics for each session in a pre-loaded DataFrame.
    
    This function is designed for DataFrames loaded from CSV (e.g., global_training.csv)
    where trial data columns (TrialTypes, Outcomes, Stimuli) are string-serialized.
    
    Args:
        df: DataFrame with session data (must have TrialTypes, Outcomes columns)
        output_path: If provided, save enriched DataFrame to this CSV path
        progress_callback: Optional callback(done, total, mouse_name) for progress updates
        
    Returns:
        DataFrame with added performance metric columns:
        - Hit, Miss, CR, FA (counts)
        - Hit_Rate, False_Alarm_Rate, d_prime
        - Early_Go_N, Early_NoGo_N, Early_Go_Rate, Early_NoGo_Rate
        - N_Go, N_NoGo
    """
    # Columns that will be added/updated
    metric_columns = [
        "Hit", "Miss", "CR", "FA",
        "Hit_Rate", "False_Alarm_Rate", "d_prime",
        "Early_Go_N", "Early_NoGo_N", "Early_Go_Rate", "Early_NoGo_Rate",
        "N_Go", "N_NoGo",
    ]
    
    # Initialize metric columns with NaN if they don't exist
    for col in metric_columns:
        if col not in df.columns:
            df[col] = np.nan
    
    total = len(df)
    for idx in range(total):
        mouse_name = df.iloc[idx].get("MouseName", f"row_{idx}")
        
        if progress_callback:
            progress_callback(idx + 1, total, str(mouse_name))
        
        try:
            metrics = extract_performance_metrics(df, idx)
            # Update row with computed metrics
            for key, value in metrics.items():
                if key in df.columns:
                    df.at[df.index[idx], key] = value
        except Exception as e:
            print(f"[WARN] Failed to compute metrics for row {idx} ({mouse_name}): {e}")
            # Leave NaN values for failed rows
            continue
    
    # Save to CSV if output path provided
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Saved enriched dataset with metrics to: {output_path}")
    
    return df


def create_training_performance_dataset(project_data: pd.DataFrame) -> pd.DataFrame:
    """
    Create a dataset combining training parameters with performance metrics.
    
    Args:
        project_data: DataFrame with session data (must have FilePath column)
        
    Returns:
        DataFrame with one row per session containing:
        - Session identifiers (MouseName, SessionDate, SessionTime, FilePath)
        - Training parameters (all GUI parameters as mode values)
        - Performance metrics (from classifier_metric)
        - Early response rates
        - ParameterChanges note field
    """
    from load_data.load_bpod_data import load_mat_file
    
    results = []
    
    for index, row in project_data.iterrows():
        file_path = row.get("FilePath", None)
        if file_path is None or not os.path.exists(file_path):
            print(f"Warning: FilePath not found or invalid for index {index}: {file_path}")
            continue
        
        try:
            # Load the .mat file to get TrialSettings
            (
                trial_types_df,
                raw_events_df,
                session_date,
                session_time,
                trial_settings,
                notes,
                licks,
                states,
                stimuli,
                Unique_Stimuli_Values,
                tones_per_class,
                boundaries,
                recs,
                outcome_names,
            ) = load_mat_file(file_path)
            
            # Extract training parameters
            param_dict = extract_training_parameters(trial_settings)
            aggregated_params, param_changes_note = aggregate_parameters_per_session(param_dict)
            
            # Extract performance metrics
            performance_metrics = extract_performance_metrics(project_data, index)
            
            # Combine all data
            session_data = {
                # Session identifiers
                "MouseName": row.get("MouseName", None),
                "SessionDate": row.get("SessionDate", None),
                "SessionTime": row.get("SessionTime", None),
                "FilePath": file_path,
                # Training parameters (with prefix to avoid conflicts)
                **{f"Param_{k}": v for k, v in aggregated_params.items()},
                # Performance metrics
                **performance_metrics,
                # Parameter changes note
                "ParameterChanges": param_changes_note,
            }
            
            results.append(session_data)
            
        except Exception as e:
            print(f"Error processing session at index {index} (file: {file_path}): {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not results:
        return pd.DataFrame()
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Sort by MouseName, SessionDate, SessionTime
    if "SessionDate" in df.columns and "SessionTime" in df.columns:
        df["SessionDate"] = pd.to_datetime(df["SessionDate"], errors="coerce")
        df = df.sort_values(by=["MouseName", "SessionDate", "SessionTime"]).reset_index(drop=True)
    
    return df


def save_training_performance_dataset(project_data: pd.DataFrame, output_path: str) -> str:
    """
    Create and save the training-performance dataset to a CSV file.
    
    Args:
        project_data: DataFrame with session data (must have FilePath column)
        output_path: Path where to save the CSV file
        
    Returns:
        Path to the saved CSV file
    """
    df = create_training_performance_dataset(project_data)
    df.to_csv(output_path, index=False)
    return output_path


def find_bpod_mat_files_in_data_root(
    data_root: str = r"Z:\Shared\Amichai\Data",
    *,
    exclude_default_settings: bool = True,
    group_filter: Optional[str] = None,
) -> List[str]:
    """
    Find all Bpod `.mat` files under the organized data root.

    Args:
        data_root: Root directory to search
        exclude_default_settings: Skip DefaultSettings.mat files
        group_filter: If provided (e.g., "8"), only include files from that group (G8A*)
    
    Why: user wants a single dataset across all mice/sessions without grouping.
    """
    import re
    mat_files: List[str] = []
    group_pattern = None
    if group_filter:
        group_pattern = re.compile(rf"G{group_filter}A\d+", re.IGNORECASE)
    
    for root, _dirs, files in os.walk(data_root):
        for fname in files:
            if not fname.lower().endswith(".mat"):
                continue
            if exclude_default_settings and fname.lower() == "defaultsettings.mat":
                continue
            
            file_path = os.path.join(root, fname)
            
            # Filter by group if specified
            if group_pattern:
                # Check file name and path for group pattern
                if not group_pattern.search(fname) and not group_pattern.search(root):
                    continue
            
            mat_files.append(file_path)
    return mat_files


def find_educage_txt_files(
    data_root: str = r"Z:\Shared\Amichai\Data",
    *,
    group_filter: Optional[str] = None,
    projects_list_path: Optional[str] = None,
) -> List[str]:
    """
    Find all Educage `.txt` data files.
    
    If projects_list_path is provided, reads the projects_list.csv and searches
    DataDir paths for Educage-type projects. Otherwise falls back to searching data_root.
    
    Educage files are CSV-like .txt files containing columns like mouse_id, date, 
    go_no_go, score, stim_name, etc.
    
    Args:
        data_root: Fallback root directory to search (used if projects_list_path not provided)
        group_filter: If provided (e.g., "8"), only include files from that group
        projects_list_path: Path to projects_list.csv to find Educage project DataDir paths
    
    Returns:
        List of paths to Educage .txt files
    """
    import re
    txt_files: List[str] = []
    group_pattern = None
    if group_filter:
        group_pattern = re.compile(rf"G{group_filter}A\d+", re.IGNORECASE)
    
    # Determine which directories to search
    search_dirs: List[str] = []
    
    if projects_list_path and os.path.exists(projects_list_path):
        # Read projects_list.csv and find Educage projects' DataDir paths
        try:
            projects_df = pd.read_csv(projects_list_path)
            for _, row in projects_df.iterrows():
                project_type = str(row.get("Project Type", ""))
                data_dir = row.get("DataDir", None)
                
                # Check if this is an Educage project
                if "educage" in project_type.lower() and data_dir and pd.notna(data_dir):
                    data_dir = str(data_dir).strip()
                    if os.path.exists(data_dir):
                        search_dirs.append(data_dir)
        except Exception:
            # Fall back to data_root if projects_list can't be read
            pass
    
    # If no Educage project dirs found, fall back to data_root
    if not search_dirs:
        search_dirs = [data_root]
    
    # Search each directory for Educage .txt files
    for search_dir in search_dirs:
        for root, _dirs, files in os.walk(search_dir):
            for fname in files:
                if not fname.lower().endswith(".txt"):
                    continue
                
                file_path = os.path.join(root, fname)
                
                # Filter by group if specified
                if group_pattern:
                    if not group_pattern.search(fname) and not group_pattern.search(root):
                        continue
                
                # Verify it's an Educage file by checking for required columns
                try:
                    # Read first few lines to check structure
                    df_check = pd.read_csv(file_path, nrows=5)
                    # Normalize column names for checking
                    cols_lower = [str(c).lower().replace(" ", "_").replace("-", "_") for c in df_check.columns]
                    # Educage files have mouse_id and go_no_go columns
                    if "mouse_id" in cols_lower and ("go_no_go" in cols_lower or "score" in cols_lower):
                        txt_files.append(file_path)
                except Exception:
                    # Not a valid Educage file, skip
                    continue
    
    return txt_files


def process_educage_file(
    file_path: str,
    *,
    min_trials: int = 10,
) -> List[Dict[str, Any]]:
    """
    Process a single Educage .txt file and return session rows with performance metrics.
    
    Reuses logic from educage_data_formmater.py but returns dicts for DataFrame creation.
    
    Args:
        file_path: Path to the Educage .txt file
        min_trials: Minimum trials per session to include
        
    Returns:
        List of dicts, one per session (grouped by mouse_id, date, level)
    """
    import ast
    import re
    
    results: List[Dict[str, Any]] = []
    
    try:
        df = pd.read_csv(file_path)
    except Exception:
        return results
    
    # Normalize column names
    def _normalize(col: Any) -> str:
        c = str(col).strip().lower()
        for ch in [" ", "\\", "/", "-"]:
            c = c.replace(ch, "_")
        while "__" in c:
            c = c.replace("__", "_")
        return c
    
    df.columns = [_normalize(c) for c in df.columns]
    
    # Exclude dummy probe mouse_id
    if "mouse_id" in df.columns:
        df = df[df["mouse_id"] != "000799EB9B"]
    
    if df.empty:
        return results
    
    # Parse licks
    def _parse_licks_list(x: Any) -> list:
        try:
            return ast.literal_eval(x) if isinstance(x, str) else ([] if pd.isna(x) else x)
        except Exception:
            return []
    
    # Build start datetime and relative licks
    if all(col in df.columns for col in ["date", "start_time"]):
        df["start_dt"] = pd.to_datetime(df["date"] + " " + df["start_time"], errors="coerce")
    
    if "licks_time" in df.columns:
        df["licks_time_list"] = df["licks_time"].apply(_parse_licks_list)
        if "licks_time_rd" in df.columns:
            licks_time_rd_parsed = df["licks_time_rd"].apply(_parse_licks_list)
            df["licks_time_list"] = licks_time_rd_parsed + df["licks_time_list"]
        
        if "start_dt" in df.columns:
            trial_id = df.index
            licks_exploded = (
                df[["date", "start_dt", "licks_time_list"]]
                .assign(trial_id=trial_id)
                .explode("licks_time_list")
            )
            licks_exploded = licks_exploded.dropna(subset=["licks_time_list"]) if not licks_exploded.empty else licks_exploded
            
            if not licks_exploded.empty:
                licks_exploded["lick_dt"] = pd.to_datetime(
                    licks_exploded["date"] + " " + licks_exploded["licks_time_list"].astype(str),
                    errors="coerce",
                )
                rel_sec = (licks_exploded["lick_dt"] - licks_exploded["start_dt"]).dt.total_seconds()
                licks_exploded["rel_sec"] = rel_sec
                licks_exploded = licks_exploded[np.isfinite(licks_exploded["rel_sec"])]
                licks_exploded = licks_exploded[licks_exploded["rel_sec"] < 4]
                
                licks_list = licks_exploded.groupby("trial_id")["rel_sec"].apply(list)
                df["licks_rel"] = licks_list.reindex(trial_id)
            else:
                df["licks_rel"] = None
    
    # Clean stimulus names
    if "stim_name" in df.columns:
        df["stim_name_clean"] = (
            df["stim_name"].astype(str)
            .str.replace(".npz", "", regex=False)
            .str.replace(r"[A-Za-z]", "", regex=True)
            .str.replace(r"(?<=\d)-(?=\d)", ".", regex=True)
            .str.replace("-", "", regex=False)
        )
        df["stim_value"] = pd.to_numeric(df["stim_name_clean"], errors="coerce").astype(float) * 0.1
    
    # Normalize trial types and outcomes
    if "go_no_go" in df.columns:
        _tt = df["go_no_go"].astype(str).str.replace("_", " ").str.replace("-", " ").str.strip().str.lower()
        df["go_no_go_norm"] = np.where(
            _tt.isin(["no go", "nogo", "no  go"]),
            "NoGo",
            np.where(_tt.eq("go"), "Go", _tt.str.title()),
        )
    
    if "score" in df.columns:
        _sc = df["score"].astype(str).str.replace("_", " ").str.strip().str.lower()
        _map = {
            "hit": "Hit", "miss": "Miss", "fa": "False Alarm", "false alarm": "False Alarm",
            "false_alarm": "False Alarm", "cr": "CR", "correct rejection": "CR",
            "correct_rejection": "CR", "catch - no response": "Catch - No Response",
            "catch - response": "Catch - Response",
        }
        df["score_norm"] = _sc.map(_map).fillna(_sc.str.title())
    
    # Group by mouse, date, level
    if "level" not in df.columns:
        df["level"] = "Unknown"
    
    group_cols = ["mouse_id", "date", "level"]
    if not all(c in df.columns for c in ["mouse_id", "date"]):
        return results
    
    # Sort and group
    sort_cols = [c for c in ["mouse_id", "date", "level", "start_time"] if c in df.columns]
    df_sorted = df.sort_values(sort_cols)
    
    for (mouse_id, date, level), group_df in df_sorted.groupby(group_cols, dropna=False):
        n_trials = len(group_df)
        if n_trials < min_trials:
            continue
        
        # Extract trial types
        trial_types = group_df["go_no_go_norm"].tolist() if "go_no_go_norm" in group_df.columns else []
        outcomes = group_df["score_norm"].tolist() if "score_norm" in group_df.columns else []
        stimuli = group_df["stim_value"].tolist() if "stim_value" in group_df.columns else []
        licks = group_df["licks_rel"].tolist() if "licks_rel" in group_df.columns else []
        start_times = group_df["start_time"].tolist() if "start_time" in group_df.columns else []
        
        # Parse tones and boundaries from level
        def _parse_level_tones(level_str: Any) -> int:
            try:
                m = re.search(r"(\d+)\s*[tT]", str(level_str))
                return int(m.group(1)) if m else 1
            except Exception:
                return 1
        
        def _parse_level_boundaries(level_str: Any) -> int:
            try:
                m = re.search(r"(\d+)\s*[bB]", str(level_str))
                return int(m.group(1)) if m else 1
            except Exception:
                return 1
        
        tones_per_class = _parse_level_tones(level)
        n_boundaries = _parse_level_boundaries(level)
        
        # Compute performance metrics from outcomes
        hit = sum(1 for o in outcomes if o == "Hit")
        miss = sum(1 for o in outcomes if o == "Miss")
        cr = sum(1 for o in outcomes if o == "CR")
        fa = sum(1 for o in outcomes if o == "False Alarm")
        
        hit_rate = hit / (hit + miss) if (hit + miss) > 0 else 0.0
        fa_rate = fa / (fa + cr) if (fa + cr) > 0 else 0.0
        
        # Compute d' using standard formula
        from scipy.stats import norm
        # Avoid extreme values (0 or 1)
        hr_adj = min(max(hit_rate, 0.01), 0.99)
        far_adj = min(max(fa_rate, 0.01), 0.99)
        d_prime = norm.ppf(hr_adj) - norm.ppf(far_adj)
        
        # Early response counts (if available in outcomes)
        early_go_n = sum(1 for i, o in enumerate(outcomes) if "early" in str(o).lower() and i < len(trial_types) and trial_types[i] == "Go")
        early_nogo_n = sum(1 for i, o in enumerate(outcomes) if "early" in str(o).lower() and i < len(trial_types) and trial_types[i] == "NoGo")
        n_go = sum(1 for t in trial_types if t == "Go")
        n_nogo = sum(1 for t in trial_types if t == "NoGo")
        
        # Format stimuli as space-separated list string
        def _format_space_list(lst: list) -> str:
            try:
                vals = [f"{float(v):.3f}" for v in lst if v is not None and not (isinstance(v, float) and pd.isna(v))]
                return "[" + " ".join(vals) + "]"
            except Exception:
                return str(lst)
        
        session_row = {
            "MouseName": mouse_id,
            "SessionDate": date,
            "TrialTypes": trial_types,
            "Outcomes": outcomes,
            "Stimuli": _format_space_list(stimuli),
            "Licks": licks,
            "StartTime": start_times,
            "Notes": level,
            "FilePath": file_path,
            "Tones_per_class": tones_per_class,
            "N_Boundaries": n_boundaries,
            # Performance metrics
            "Hit": hit,
            "Miss": miss,
            "CR": cr,
            "FA": fa,
            "Hit_Rate": hit_rate,
            "False_Alarm_Rate": fa_rate,
            "d_prime": d_prime,
            "Early_Go_N": early_go_n,
            "Early_NoGo_N": early_nogo_n,
            "N_Go": n_go,
            "N_NoGo": n_nogo,
            "Early_Go_Rate": early_go_n / n_go if n_go > 0 else 0.0,
            "Early_NoGo_Rate": early_nogo_n / n_nogo if n_nogo > 0 else 0.0,
            # Source identifier
            "Source": "Educage",
        }
        results.append(session_row)
    
    return results


def create_global_training_performance_dataset(
    data_root: str = r"Z:\Shared\Amichai\Data",
    *,
    min_trials: int = 50,
    skip_fake: bool = True,
    group_filter: Optional[str] = None,
    source_types: Optional[List[str]] = None,
    user_path: Optional[str] = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> pd.DataFrame:
    """
    Build one combined dataset across Bpod and/or Educage sessions found under `data_root`.

    Output: one row per session with:
    - session identifiers (MouseName, SessionDate, SessionTime, FilePath, Notes, Recording, etc.)
    - training GUI params (mode per session, prefixed as Param_*) - Bpod only
    - performance metrics + early response rates (from extract_performance_metrics)
    - ParameterChanges note - Bpod only
    - Source column ("Bpod" or "Educage")

    Args:
        data_root: Root directory to search for Bpod data files
        min_trials: Minimum trials per session to include
        skip_fake: Skip files with 'fake' in path
        group_filter: Filter by group number (e.g., "8" for G8A*)
        source_types: List of sources to include ["Bpod", "Educage"]. Default: both.
        user_path: Path to user data directory containing projects_list.csv (for Educage DataDir lookup)
        progress_callback: Callback for progress updates (done, total, current_path)

    Performance: single-pass per file (loads each file once).
    """
    from load_data.load_bpod_data import load_mat_file, create_single_row_with_outcome

    # Default to both sources if not specified
    if source_types is None:
        source_types = ["Bpod", "Educage"]
    
    results: List[Dict[str, Any]] = []
    
    # Build projects_list_path for Educage DataDir lookup
    projects_list_path = None
    if user_path:
        projects_list_path = os.path.join(user_path, "projects_list.csv")
    
    # Collect all files to process
    files_to_process: List[tuple] = []  # (file_path, source_type)
    
    if "Bpod" in source_types:
        mat_files = find_bpod_mat_files_in_data_root(data_root, group_filter=group_filter)
        files_to_process.extend((f, "Bpod") for f in mat_files)
    
    if "Educage" in source_types:
        txt_files = find_educage_txt_files(
            data_root, 
            group_filter=group_filter, 
            projects_list_path=projects_list_path,
        )
        files_to_process.extend((f, "Educage") for f in txt_files)

    total = len(files_to_process)
    processed = 0
    
    for file_path, source_type in files_to_process:
        processed += 1
        if progress_callback is not None:
            try:
                progress_callback(processed, total, f"[{source_type}] {file_path}")
            except Exception:
                # Never let UI callbacks break data processing.
                pass

        if skip_fake and "fake" in str(file_path).lower():
            continue

        if source_type == "Bpod":
            # Process Bpod .mat file
            try:
                (
                    trial_types_df,
                    raw_events_df,
                    session_date,
                    session_time,
                    trial_settings,
                    notes,
                    licks,
                    states,
                    stimuli,
                    Unique_Stimuli_Values,
                    tones_per_class,
                    boundaries,
                    recs,
                    outcome_names,
                ) = load_mat_file(file_path)
            except Exception:
                continue

            if len(trial_types_df) < int(min_trials):
                continue

            try:
                combined_row_df = create_single_row_with_outcome(
                    file_path,
                    trial_types_df,
                    raw_events_df,
                    session_date,
                    session_time,
                    trial_settings,
                    notes,
                    licks,
                    states,
                    Unique_Stimuli_Values,
                    tones_per_class,
                    boundaries,
                    recs,
                    outcome_names,
                )
            except Exception:
                continue

            # Training params (mode + change notes)
            try:
                param_dict = extract_training_parameters(trial_settings)
                aggregated_params, param_changes_note = aggregate_parameters_per_session(param_dict)
            except Exception:
                aggregated_params = {}
                param_changes_note = "Parameter extraction failed"

            # Performance metrics
            try:
                performance_metrics = extract_performance_metrics(combined_row_df, 0)
            except Exception:
                performance_metrics = {}

            base_row: Dict[str, Any] = {}
            try:
                base_row = combined_row_df.iloc[0].to_dict()
            except Exception:
                base_row = {"FilePath": file_path}

            out_row: Dict[str, Any] = {
                **base_row,
                **{f"Param_{k}": v for k, v in aggregated_params.items()},
                **performance_metrics,
                "ParameterChanges": param_changes_note,
                "Source": "Bpod",
            }
            results.append(out_row)
        
        elif source_type == "Educage":
            # Process Educage .txt file - returns multiple sessions per file
            educage_sessions = process_educage_file(file_path, min_trials=min_trials)
            results.extend(educage_sessions)

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Normalize and sort (best-effort)
    if "SessionDate" in df.columns:
        df["SessionDate"] = pd.to_datetime(df["SessionDate"].apply(lambda x: str(x)), errors="coerce")

    sort_cols = [c for c in ["Source", "MouseName", "SessionDate", "SessionTime"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(by=sort_cols).reset_index(drop=True)

    return df


def save_global_training_performance_dataset(
    output_path: str,
    data_root: str = r"Z:\Shared\Amichai\Data",
    *,
    min_trials: int = 50,
    skip_fake: bool = True,
) -> str:
    """Create the global dataset and save it to CSV."""
    df = create_global_training_performance_dataset(
        data_root,
        min_trials=min_trials,
        skip_fake=skip_fake,
    )
    df.to_csv(output_path, index=False)
    return output_path
