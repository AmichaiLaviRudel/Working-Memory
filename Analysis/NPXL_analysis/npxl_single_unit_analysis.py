import plotly.graph_objects as go
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
import pandas as pd
import os
import streamlit as st
import streamlit.components.v1 as components
from functools import lru_cache
            
# Import single unit metrics functions from single_unit_offline_analysis
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
    compute_psth_pvalues_from_event_windows,
    compute_psth_pvalues_from_event_windows_cached,
    compute_all_unit_metrics_cached,
)

# Import load_event_windows_data from NPXL_Preprocessing
from Analysis.NPXL_analysis.NPXL_Preprocessing import load_event_windows_data as load_event_windows_data_base
from Analysis.NPXL_analysis.single_unit_offline_analysis.visualization import (
    plot_unit_heatmap,
    get_trial_statistics,
)
from Analysis.GNG_bpod_analysis.colors import COLOR_GO, COLOR_GRAY, COLOR_NOGO, COLOR_HIT, COLOR_FA, COLOR_CR, COLOR_MISS, COLOR_BLUE, COLOR_BLUE_TRANSPARENT, COLOR_ACCENT, COLOR_ACCENT_TRANSPARENT

def save_pvalues_to_folder(pvals, selected_folder, window=(-1, 2), bin_size=0.01):
    """
    Save p-values to the analysis output folder.
    
    Args:
        pvals: numpy array of p-values for each unit
        selected_folder: path to the analysis output folder
        window: time window used for p-value calculation
        bin_size: bin size used for p-value calculation
    """
    try:
        # Create a DataFrame with unit indices and p-values
        pvals_df = pd.DataFrame({
            'unit_index': range(len(pvals)),
            'p_value': pvals,
            'significant': pvals < 0.05
        })
        
        # Add metadata about the analysis
        metadata = {
            'window_start': window[0],
            'window_end': window[1],
            'bin_size': bin_size,
            'total_units': len(pvals),
            'significant_units': np.sum(pvals < 0.05),
            'significance_threshold': 0.05
        }
        
        # Save p-values DataFrame
        pvals_file = os.path.join(selected_folder, "pvalues_analysis.csv")
        pvals_df.to_csv(pvals_file, index=False)
        
        # Save metadata
        metadata_file = os.path.join(selected_folder, "pvalues_metadata.txt")
        with open(metadata_file, 'w') as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
        
        return True
    except Exception as e:
        print(f"Error saving p-values: {e}")
        return False

def save_all_psth_metrics(event_windows_data, selected_folder, display_window, pvals=None, baseline_window=(-0.5, 0)):
    """
    Save PSTH metrics for all units to the analysis output folder.
    
    Args:
        event_windows_data: tuple containing (event_windows_matrix, time_axis, valid_event_indices, event_stimuli_outcome_df, metadata)
        selected_folder: path to the analysis output folder
        display_window: tuple of (start, end) time for the display window
        pvals: numpy array of p-values for each unit (optional)
        baseline_window: tuple of (start, end) time for baseline calculation
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Extract data from event_windows_data
        event_windows_matrix, time_axis, valid_event_indices, event_stimuli_outcome_df, metadata = event_windows_data
        
        # Create single_unit subfolder
        single_unit_folder = os.path.join(selected_folder, "single_unit")
        os.makedirs(single_unit_folder, exist_ok=True)
        
        # Initialize list to store all metrics
        all_metrics = []
        
        # Calculate metrics for each unit
        num_units = event_windows_matrix.shape[0]
        for unit_idx in range(num_units):
            # Get unit data
            unit_data = event_windows_matrix[unit_idx, :, :]  # [time × events]
            
            # Calculate PSTH metrics
            metrics = calculate_psth_metrics(unit_data, time_axis, baseline_window)
            
            # Add unit index to metrics
            metrics['unit_index'] = unit_idx
            
            # Add p-value if available
            if pvals is not None and unit_idx < len(pvals):
                metrics['p_value'] = pvals[unit_idx]
                metrics['significant'] = pvals[unit_idx] < 0.05
            else:
                metrics['p_value'] = np.nan
                metrics['significant'] = False
            
            # Flatten suppression metrics for CSV storage
            if isinstance(metrics['suppression_metrics'], dict):
                metrics['suppression_magnitude'] = metrics['suppression_metrics']['magnitude']
                metrics['suppression_duration'] = metrics['suppression_metrics']['duration']
                metrics['fraction_suppressed'] = metrics['suppression_metrics']['fraction_suppressed']
                del metrics['suppression_metrics']  # Remove the nested dict
            
            all_metrics.append(metrics)
        
        # Convert to DataFrame
        metrics_df = pd.DataFrame(all_metrics)
        
        # Reorder columns to put unit_index first
        cols = ['unit_index'] + [col for col in metrics_df.columns if col != 'unit_index']
        metrics_df = metrics_df[cols]
        
        # Save to CSV
        metrics_file = os.path.join(single_unit_folder, "psth_metrics.csv")
        metrics_df.to_csv(metrics_file, index=False)
        
        # Save metadata
        metadata_file = os.path.join(single_unit_folder, "psth_metrics_metadata.txt")
        with open(metadata_file, 'w') as f:
            f.write(f"Analysis Date: {pd.Timestamp.now()}\n")
            f.write(f"Total Units: {num_units}\n")
            f.write(f"Display Window: {display_window[0]} to {display_window[1]} seconds\n")
            f.write(f"Baseline Window: {baseline_window[0]} to {baseline_window[1]} seconds\n")
            f.write(f"Time Axis Range: {time_axis[0]:.3f} to {time_axis[-1]:.3f} seconds\n")
            f.write(f"Number of Events: {event_windows_matrix.shape[2]}\n")
            f.write(f"Time Bins: {len(time_axis)}\n")
            if pvals is not None:
                f.write(f"P-values Available: Yes\n")
                f.write(f"Significant Units (p < 0.05): {np.sum(pvals < 0.05)}\n")
            else:
                f.write(f"P-values Available: No\n")
            f.write("\nMetrics Description and Mathematical Formulas:\n")
            f.write("- response_type: Type of response ('excitation' or 'suppression')\n")
            f.write("  Formula: excitation if max_deviation >= min_deviation, else suppression\n")
            f.write("- onset_latency: Time from event onset to first significant response (seconds)\n")
            f.write("  Formula: min(t | rate(t) > baseline_rate + 2*std) for excitation\n")
            f.write("  Formula: min(t | rate(t) < baseline_rate - 2*std) for suppression\n")
            f.write("- peak_latency: Time from event onset to peak response (seconds)\n")
            f.write("  Formula: argmax(rate(t)) for excitation, argmin(rate(t)) for suppression\n")
            f.write("- response_magnitude: Peak response magnitude relative to baseline (spikes/s)\n")
            f.write("  Formula: max(rate(t)) - baseline_rate for excitation\n")
            f.write("  Formula: baseline_rate - min(rate(t)) for suppression\n")
            f.write("- fwhm: Full-width at half-maximum of the response (seconds)\n")
            f.write("  Formula: t2 - t1 where rate(t1) = rate(t2) = baseline + (peak - baseline)/2 for excitation\n")
            f.write("  Formula: t2 - t1 where rate(t1) = rate(t2) = baseline - (baseline - peak)/2 for suppression\n")
            f.write("- rise_time: Time from onset to peak (seconds)\n")
            f.write("  Formula: peak_latency - onset_latency\n")
            f.write("- decay_time: Time from peak to return to baseline (seconds)\n")
            f.write("  Formula: t_return - peak_latency where rate(t_return) ≈ baseline_rate + std for excitation\n")
            f.write("  Formula: t_return - peak_latency where rate(t_return) ≈ baseline_rate - std for suppression\n")
            f.write("- trial_variability: Coefficient of variation across trials\n")
            f.write("  Formula: std(trial_rates) / mean(trial_rates)\n")
            f.write("- signal_to_noise: Response magnitude divided by baseline standard deviation\n")
            f.write("  Formula: response_magnitude / std(baseline_rate)\n")
            f.write("- baseline_rate: Average firing rate during baseline period (spikes/s)\n")
            f.write("  Formula: mean(rate(t)) where t is in baseline window\n")
            f.write("- peak_rate: Peak firing rate during response period (spikes/s)\n")
            f.write("  Formula: max(rate(t)) for excitation, min(rate(t)) for suppression\n")
            f.write("- suppression_magnitude: Magnitude of suppression below baseline (spikes/s)\n")
            f.write("  Formula: baseline_rate - min(rate(t)) where rate(t) < baseline_rate\n")
            f.write("- suppression_duration: Duration of suppression period (seconds)\n")
            f.write("  Formula: sum(dt) where rate(t) < baseline_rate\n")
            f.write("- fraction_suppressed: Fraction of response period that was suppressed\n")
            f.write("  Formula: suppression_duration / total_response_duration\n")
            f.write("- p_value: Statistical significance of response modulation\n")
            f.write("  Formula: Mann-Whitney U test between baseline and response periods\n")
            f.write("- significant: Boolean indicating p < 0.05\n")
            f.write("  Formula: p_value < 0.05\n")
        
        return True
        
    except Exception as e:
        print(f"Error saving PSTH metrics: {e}")
        return False

# All computation functions have been moved to Analysis.NPXL_analysis.NPXL_offline_analysis.single_unit_metrics


def plot_unit_psth(
    event_windows_data,
    display_window,
    unit_idx,
    sorted_pvals,
    unit_rank,
    bin_size=0.005,
    analysis_output_dir=None,
):
    """Plot a single-unit PSTH aligned to events and compute significance metrics.

    If analysis_output_dir is provided, attempts to load a precomputed PSTH HTML
    from the offline analysis output (plots folder). Falls back to computing from
    event_windows_data if no saved plot is found.
    """
   
    
    # Try to load precomputed PSTH from analysis output (HTML)
    if analysis_output_dir:
        import glob
        plot_type = "raw_psth"
        plots_dir = os.path.join(analysis_output_dir, "plots")
        html_path = None
        if os.path.isdir(plots_dir):
            patterns = [
                os.path.join(plots_dir, plot_type, f"*unit_{unit_idx}_{plot_type}.html"),
                os.path.join(plots_dir, f"*unit_{unit_idx}_{plot_type}.html"),
                os.path.join(plots_dir, f"*unit_{unit_idx}_*.html"),
            ]
            for pattern in patterns:
                matches = glob.glob(pattern)
                if matches:
                    html_path = matches[0]
                    break
        if html_path and os.path.exists(html_path):
            with open(html_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            return html_content, None, True

    # Fallback: compute PSTH from matrices
    event_windows_matrix, time_axis, valid_event_indices, event_stimuli_outcome_df, metadata = event_windows_data
    # Get the unit's data
    unit_data = event_windows_matrix[unit_idx, :, :]  # Shape: [time × events]
    n_time_bins = event_windows_matrix.shape[1]

    # Create time axis for the window
    peri_event_window = np.linspace(display_window[0], display_window[1], n_time_bins)
    # Find the index corresponding to time 0
    zero_idx = np.argmin(np.abs(peri_event_window))
    sec2bin = 1/bin_size
    unit_data = unit_data[zero_idx+int(display_window[0]*sec2bin):zero_idx+int(display_window[1]*sec2bin),:]

    # Calculate PSTH statistics (fallback to computing from matrices)
    if unit_data.shape[1] > 0:
        psth_mean = np.mean(unit_data, axis=1)  # Average across events
        psth_sem = np.std(unit_data, axis=1) / np.sqrt(unit_data.shape[1])  # SEM across events
    else:
        # If no data, create empty arrays
        psth_mean = np.zeros(unit_data.shape[0])
        psth_sem = np.zeros(unit_data.shape[0])
    
    # Create time axis for PSTH that matches the display_window
    num_time_bins = unit_data.shape[0]
    psth_time_axis = np.linspace(display_window[0], display_window[1], num_time_bins)
    
    # Calculate comprehensive PSTH metrics
    psth_metrics = calculate_psth_metrics(unit_data, psth_time_axis)
    
    # Create PSTH plot with proper time axis
    psth_fig = go.Figure()
    
    # Add main PSTH line
    psth_fig.add_trace(go.Scatter(
        x=psth_time_axis,
        y=psth_mean,
        mode='lines',
        name='Mean Firing Rate',
        line=dict(color=COLOR_ACCENT, width=3)
    ))
    
    # Add shaded area for SEM
    psth_fig.add_trace(go.Scatter(
        x=np.concatenate([psth_time_axis, psth_time_axis[::-1]]),
        y=np.concatenate([psth_mean + psth_sem, (psth_mean - psth_sem)[::-1]]),
        fill='toself',
        fillcolor=f'rgba(0,0,255,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        name='SEM'
    ))
    
    # Add vertical line at x = 0
    psth_fig.add_vline(
        x=0, 
        line_dash="dash", 
        line_color=COLOR_GRAY, 
        line_width=2
    )
    
    # Add markers for key metrics if they exist
    if not np.isnan(psth_metrics['onset_latency']):
        psth_fig.add_vline(
            x=psth_metrics['onset_latency'],
            line_dash="dot",
            line_color="orange",
            line_width=2,
            annotation_text=f"Onset",
            annotation_position="top left",
            annotation=dict(textangle=90)  # Rotate annotation 90 degrees
        )
    
    if not np.isnan(psth_metrics['peak_latency']):
        psth_fig.add_vline(
            x=psth_metrics['peak_latency'],
            line_dash="dot",
            line_color="red",
            line_width=2,
            annotation_text=f"Peak",
            annotation_position="top left",
            annotation=dict(textangle=90)  # Rotate annotation 90 degrees
        )
        
        # Add horizontal line at half-maximum for FWHM visualization
        if not np.isnan(psth_metrics['fwhm']):
            half_max = psth_metrics['baseline_rate'] + (psth_metrics['peak_rate'] - psth_metrics['baseline_rate']) / 2
            psth_fig.add_hline(
                y=half_max,
                line_dash="dot",
                line_color="purple",
                line_width=1,
                annotation_text=f"Half-Max: {half_max:.2f}",
                annotation_position="right",
                annotation=dict(textangle=90)  # Rotate annotation 90 degrees
            )
    
    # Create title with outcome filter information
    title_text = f"PSTH - Unit {unit_idx} (p={sorted_pvals[unit_rank]:.3g})"
    if outcome_filter != "All":
        title_text += f" - {outcome_filter} (n={unit_data.shape[1]})"
    else:
        title_text += f" - All trials (n={unit_data.shape[1]})"
    
    psth_fig.update_layout(
        title=title_text,
        xaxis_title="Time (s)",
        yaxis_title="Firing Rate (spikes/s)",
        xaxis=dict(constrain='domain'),
        margin=dict(r=80),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    return psth_fig, psth_metrics, False


def check_analysis_output_exists(selected_folder):
    """
    Check if analysis_output folder exists and contains data.
    
    Args:
        selected_folder: Path to the analysis_output folder (or parent directory)
        
    Returns:
        tuple: (exists: bool, analysis_output_dir: str, parent_dir: str)
    """
    if selected_folder is None:
        return False, None, None
    
    # If selected_folder is already analysis_output, use it
    if os.path.basename(selected_folder) == "analysis_output":
        analysis_output_dir = selected_folder
        parent_dir = os.path.dirname(selected_folder)
    else:
        # Otherwise, look for analysis_output in parent directory
        parent_dir = selected_folder
        analysis_output_dir = os.path.join(parent_dir, "analysis_output")
    
    if not os.path.exists(analysis_output_dir):
        return False, analysis_output_dir, parent_dir
    
    # Check if it has data (tables or plots folders with files)
    tables_dir = os.path.join(analysis_output_dir, "tables")
    plots_dir = os.path.join(analysis_output_dir, "plots")
    
    has_data = False
    if os.path.exists(tables_dir):
        has_data = len([f for f in os.listdir(tables_dir) if f.endswith('.csv')]) > 0
    if not has_data and os.path.exists(plots_dir):
        has_data = len([f for f in os.listdir(plots_dir) if f.endswith('.html')]) > 0
    
    return has_data, analysis_output_dir, parent_dir

def run_offline_analysis(parent_dir):
    """
    Run the offline analysis pipeline.
    
    Args:
        parent_dir: Path to the parent directory (catgt folder)
    """
    try:
        import streamlit as st
    except Exception:
        st = None

    progress_bar = None
    status_placeholder = None
    if st is not None:
        progress_bar = st.progress(0, text="Starting offline analysis...")
        status_placeholder = st.empty()
        status_placeholder.info(f"Preparing to run analysis in: {parent_dir}")

    def progress_fn(pct: int, msg: str = ""):
        if progress_bar is not None:
            progress_bar.progress(pct, text=msg or "Running offline analysis...")

    def status_fn(msg: str):
        if status_placeholder is not None:
            status_placeholder.info(msg)

    try:
        from Analysis.NPXL_analysis.single_unit_offline_analysis.main import main
        main(parent_dir=parent_dir, progress_fn=progress_fn, status_fn=status_fn)
        if progress_bar:
            progress_bar.progress(100, text="Offline analysis completed")
            status_placeholder.success("Offline analysis completed successfully.")
        return True, None
    except Exception as e:
        if progress_bar:
            progress_bar.progress(100, text="Offline analysis failed")
            status_placeholder.error(f"Offline analysis failed: {e}")
        return False, str(e)

def single_unit_analysis_panel(selected_recording_dir=None, selected_area=None, raw_folder=None):
    import streamlit as st
    import numpy as np
    # plot_unit_psth is defined in this same file, no need to import
    from Analysis.NPXL_analysis.population_analysis import plot_population_heatmap

    # Check if analysis output exists
    has_analysis_data, analysis_output_dir, parent_dir = check_analysis_output_exists(selected_recording_dir)
    if parent_dir is not None:
        if st.button("🚀 Run single Unit Analysis", type="primary", key="run_analysis_btn"):
            with st.spinner("Running offline analysis... This may take several minutes."):
                success, error = run_offline_analysis(parent_dir)
                if success:
                    st.success("✅ Analysis completed successfully! Please refresh the page to see results.")
                    st.rerun()
                else:
                    st.error(f"❌ Analysis failed: {error}")
    # Create tabs for different analysis types
    tab1, tab2, tab3, qa_tab = st.tabs(["Basic PSTH", "Selectivity Analysis", "GLM Analysis", "QA"])
    
    # Show analysis status and run button if needed

    
    with tab1:
        st.header("Basic Analysis")
        units_metrics_df = None
        units_metrics_df_sorted = None
        pvals = None
        sorted_indices = None
        sorted_pvals = None


        # Load analysis output data if available
        if has_analysis_data and analysis_output_dir is not None:
            # Try to load unit metrics from analysis output (preferred)

            if "OFC" in selected_area:
                primary = "ofc_all_units_metrics.csv"
            else:
                primary = "acx_all_units_metrics.csv"

            units_metrics_path = os.path.join(analysis_output_dir, "tables", primary)
           
            
            if os.path.exists(units_metrics_path):
                units_metrics_df = pd.read_csv(units_metrics_path)
                st.info(f"📊 Loaded {len(units_metrics_df)} units from analysis output")
                
                # If p-values are present, use them instead of recomputing
                if 'tone_active_p_val' in units_metrics_df.columns:
                    pvals = units_metrics_df['tone_active_p_val'].values
                    sorted_indices = np.argsort(pvals)
                    sorted_pvals = pvals[sorted_indices]
                    # sort dataframe by p-value order for consistent UI
                    units_metrics_df_sorted = units_metrics_df.iloc[sorted_indices].reset_index(drop=True)
                else:
                    units_metrics_df_sorted = units_metrics_df.copy()
        
        # Display summary statistics
        st.subheader("Analysis Summary")
        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
        with summary_col1:
            st.metric("Total Units", len(units_metrics_df))
        if 'tone_active_p_val' in units_metrics_df.columns:
            significant = units_metrics_df['tone_active_p_val'] < 0.05
            with summary_col2:
                st.metric("Tone Active Units", f"{significant.sum()}/{len(units_metrics_df)}")
            with summary_col3:
                significance_rate = (significant.sum() / len(units_metrics_df)) * 100 if len(units_metrics_df) > 0 else 0
                st.metric("Tone Active Rate", f"{significance_rate:.1f}%")
        if 'psth_response_type' in units_metrics_df.columns:
            with summary_col4:
                exc_count = (units_metrics_df['psth_response_type'] == 'excitation').sum()
                st.metric("Excitatory Units", exc_count)


        # # Load event windows data for accurate p-value calculation (cached)
        # if event_windows_data is None:
        #     data_folder = analysis_output_dir if analysis_output_dir is not None else selected_folder
        #     if data_folder is not None:
        #         event_windows_data = load_event_windows_data_base(data_folder)
        
        # # Recompute p-values only if not loaded from metrics table
        # if pvals is None and event_windows_data is not None:
        #     event_windows_matrix, time_axis, valid_event_indices, event_stimuli_outcome_df, metadata = event_windows_data
        #     pvals = compute_psth_pvalues_from_event_windows(
        #         event_windows_matrix,
        #         event_times,
        #         bin_size=bin_size_display,
        #         window=[-0.1, 0.3],
        #     )
        #     sorted_indices = np.argsort(pvals)
        #     sorted_pvals = pvals[sorted_indices]
        
        # # Display statistics
        # stats_col1, stats_col2, stats_col3 = st.columns(3)
        # with stats_col1:
        #     st.metric("Total Units", len(pvals))
        # with stats_col2:
        #     significant_units = np.sum(pvals < 0.05)
        #     st.metric("Significant Units", f"{significant_units}/{len(pvals)}")
        # with stats_col3:
        #     significance_rate = (significant_units / len(pvals)) * 100 if len(pvals) > 0 else 0
        #     st.metric("Significance Rate", f"{significance_rate:.1f}%")


        # # Cache management
        #     cache_col1, cache_col2 = st.columns(2)
        #     with cache_col1:
        #         if st.button("🗑️ Clear Analysis Cache"):
        #             st.cache_data.clear()
        #             st.toast("Analysis cache cleared - next computation will be fresh")
        #     with cache_col2:
        #         st.caption("💡 Metrics are cached for faster performance")


        # Unit Selection Section
        st.subheader("Unit Selection")
        unit_col1, unit_col2 = st.columns(2)
        
        with unit_col1:
            if units_metrics_df_sorted is not None and sorted_pvals is not None:
                unit_rank = st.slider(
                    "Unit Rank (by p-value)", 
                    0, len(sorted_pvals) - 1, 0,
                    help="Select unit by statistical significance rank"
                )
                row = units_metrics_df_sorted.iloc[unit_rank]
                unit_idx = int(row["unit_idx"]) if "unit_idx" in row else unit_rank
                current_pval = sorted_pvals[unit_rank]
            else:
                unit_rank = 0
                unit_idx = 0
                current_pval = np.nan
        
        with unit_col2:
            st.metric("Selected Unit", unit_idx)
            st.metric("P-value", f"{current_pval:.3g}" if not np.isnan(current_pval) else "N/A")
        # Visualization Section
        st.subheader("Visualizations")
        
        if units_metrics_df is not None:
            # Create two columns: PSTH plot (from saved HTML) and metrics table from units_metrics_df
            viz_col1, viz_col2 = st.columns(2)
            
            # --- Column 1: PSTH plot from saved HTML path ---
            with viz_col1:
                html_path = None
                if "plot_path_raw_psth" in units_metrics_df_sorted.columns:
                    if pd.notna(row.get("plot_path_raw_psth", None)):
                        html_path = row["plot_path_raw_psth"]
                if html_path and os.path.exists(html_path):
                    try:
                        with open(html_path, "r", encoding="utf-8") as f:
                            psth_render = f.read()
                        components.html(psth_render, height=500, scrolling=False)
                    except Exception as e:
                        st.warning(f"Error loading PSTH plot: {e}")
                else:
                    st.info("No saved PSTH plot found for this unit.")
            
                psth_cols = [
                    ("psth_onset_latency", "Onset Latency", "s"),
                    ("psth_peak_latency", "Peak Latency", "s"),
                    ("psth_response_magnitude", "Response Magnitude", "spikes/s"),
                    ("psth_response_type", "Response Type", ""),
                    ("psth_fwhm", "FWHM", "s"),
                    ("psth_rise_time", "Rise Time", "s"),
                    ("psth_decay_time", "Decay Time", "s"),
                    ("psth_trial_variability", "Trial Variability", ""),
                    ("psth_signal_to_noise", "Signal-to-Noise", ""),
                    ("psth_baseline_rate", "Baseline Rate", "spikes/s"),
                    ("psth_peak_rate", "Peak Rate", "spikes/s"),
                    ("psth_suppression_magnitude", "Suppression Magnitude", "spikes/s"),
                    ("psth_suppression_duration", "Suppression Duration", "s"),
                    ("psth_fraction_suppressed", "Fraction Suppressed", ""),
                ]
                mcol1, mcol2, mcol3 = st.columns(3)
                cols_cycle = [mcol1, mcol2, mcol3]
                for idx, (key, label, unit) in enumerate(psth_cols):
                    if key in row and pd.notna(row[key]):
                        col = cols_cycle[idx % 3]
                        val = row[key]
                        suffix = f" {unit}".strip()
                        col.metric(label, f"{val:.3f}{(' ' + unit) if unit else ''}" if isinstance(val, (int, float, float)) else str(val))

            with viz_col2:
                heatmap_html_path = None
                if "plot_path_heatmap" in units_metrics_df.columns:
                    # Locate row by unit_idx column if exists
                    if pd.notna(row.get("plot_path_heatmap", None)):
                        heatmap_html_path = row["plot_path_heatmap"]
                if heatmap_html_path and os.path.exists(heatmap_html_path):
                    try:
                        with open(heatmap_html_path, "r", encoding="utf-8") as f:
                            heatmap_render = f.read()
                        components.html(heatmap_render, height=500, scrolling=False)
                    except Exception as e:
                        st.warning(f"Error loading heatmap plot: {e}")
                else:
                    st.info("No saved heatmap plot found for this unit.")
    #             # Create heatmap using the new function
    #             heatmap_fig = plot_unit_heatmap(event_windows_data, display_window, unit_idx)
    #             st.plotly_chart(heatmap_fig, use_container_width=True)
                
                # Heatmap statistics
                heatmap_stats_col1, heatmap_stats_col2, heatmap_stats_col3, heatmap_stats_col4 = st.columns(4)
                
                # Prefer metrics from units_metrics_df if available for this unit
                cr_val = row["trial_count_cr"] 
                fa_val = row["trial_count_fa"] 
                miss_val = row["trial_count_miss"] 
                hit_val = row["trial_count_hit"] 
                with heatmap_stats_col1:
                    st.metric("CR", f"{cr_val:.0f}" if not pd.isna(cr_val) else "N/A")
                with heatmap_stats_col2:
                    st.metric("FA", f"{fa_val:.0f}" if not pd.isna(fa_val) else "N/A")
                with heatmap_stats_col3:
                    st.metric("Miss", f"{miss_val:.0f}" if not pd.isna(miss_val) else "N/A")
                with heatmap_stats_col4:
                    st.metric("Hit", f"{hit_val:.0f}" if not pd.isna(hit_val) else "N/A")
        else:
            st.warning("No event windows data available. Please ensure data is loaded.")
            
    
    with tab2:
        st.header("Advanced Single Unit Analysis")
        
        if not has_analysis_data:
            st.warning("⚠️ Analysis output not found. Please run the offline analysis first.")
        else:
            
            if "OFC" in selected_area:
                area_selectivity = "ofc_selectivity_metrics.csv"
            else:
                area_selectivity = "acx_selectivity_metrics.csv"
            # Load selectivity metrics
            selectivity_path = os.path.join(analysis_output_dir, "tables", area_selectivity)

            if os.path.exists(selectivity_path):
                selectivity_df = pd.read_csv(selectivity_path)
                st.success(f"✅ Loaded selectivity metrics for {len(selectivity_df)} units")
                
                # Display summary
                st.subheader("Selectivity Summary")
                if len(selectivity_df) > 0:
                    summary_cols = st.columns(4)
                    with summary_cols[0]:
                        if 'stimulus_selective' in selectivity_df.columns:
                            st.metric("Stimulus Selective", selectivity_df['stimulus_selective'].sum())
                    with summary_cols[1]:
                        if 'outcome_modulated' in selectivity_df.columns:
                            st.metric("Outcome Modulated", selectivity_df['outcome_modulated'].sum())
                    with summary_cols[2]:
                        if 'go_nogo_selective' in selectivity_df.columns:
                            st.metric("Go/NoGo Selective", selectivity_df['go_nogo_selective'].sum())
                    with summary_cols[3]:
                        if 'choice_coding' in selectivity_df.columns:
                            st.metric("Choice Coding", selectivity_df['choice_coding'].sum())
                    
                    # Unit selection
                    st.subheader("Unit Selection")
                    if 'unit_idx' in selectivity_df.columns:
                        unit_options = selectivity_df['unit_idx'].tolist()
                        selected_unit_idx = st.selectbox("Select Unit", unit_options, key="adv_unit_select")
                        
                        # Display metrics for selected unit
                        unit_data = selectivity_df[selectivity_df['unit_idx'] == selected_unit_idx].iloc[0]
                        
                        st.subheader(f"Metrics for Unit {selected_unit_idx}")
                        metrics_col1, metrics_col2 = st.columns(2)
                        
                        with metrics_col1:
                            if 'go_nogo_dprime' in unit_data:
                                st.metric("d' (Go/NoGo)", f"{unit_data['go_nogo_dprime']:.3f}")
                            if 'go_nogo_roc_auc' in unit_data:
                                st.metric("ROC AUC", f"{unit_data['go_nogo_roc_auc']:.3f}")
                            if 'choice_probability' in unit_data:
                                st.metric("Choice Probability", f"{unit_data['choice_probability']:.3f}")
                        
                        with metrics_col2:
                            if 'outcome_p_value' in unit_data:
                                st.metric("Outcome p-value", f"{unit_data['outcome_p_value']:.3g}")
                            if 'best_stimulus' in unit_data:
                                st.metric("Best Stimulus", f"{unit_data['best_stimulus']:.2f}")
                        
                        # Load and display plots if available
                        plots_dir = os.path.join(analysis_output_dir, "plots")
                        if os.path.exists(plots_dir):
                            st.subheader("Saved Plots")
                            # Check if tuning curve data exists
                            if 'tuning_curve_stimuli' in unit_data and 'tuning_curve' in unit_data:
                                try:
                                    from Analysis.NPXL_analysis.NPXL_offline_analysis.visualization import plot_tuning_curve
                                    
                                    # Plot tuning curve (boundaries are read from session state inside the function)
                                    region_name = selected_area if selected_area else "Unit"
                                    fig = plot_tuning_curve(
                                        tuning_curve_stimuli=unit_data['tuning_curve_stimuli'],
                                        tuning_curve=unit_data['tuning_curve'],
                                        tuning_curve_sem=unit_data.get('tuning_curve_sem'),
                                        unit_idx=int(selected_unit_idx),
                                        region_name=region_name,
                                        use_log_scale=True
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                except Exception as e:
                                    st.warning(f"Could not display tuning curve: {e}")
                            
                            # Check for PSTH by stimulus plot
                            stim_plot_path = os.path.join(plots_dir, "psth_by_stimulus", f"acx_unit_{selected_unit_idx}_psth_by_stimulus.html")
                            if not os.path.exists(stim_plot_path):
                                stim_plot_path = os.path.join(plots_dir, "psth_by_stimulus", f"ofc_unit_{selected_unit_idx}_psth_by_stimulus.html")
                            
                            if os.path.exists(stim_plot_path):
                                with open(stim_plot_path, 'r', encoding='utf-8') as f:
                                    st.components.v1.html(f.read(), height=600)
                            
                            # Check for PSTH by outcome plot
                            outcome_plot_path = os.path.join(plots_dir, "psth_by_outcome", f"acx_unit_{selected_unit_idx}_psth_by_outcome.html")
                            if not os.path.exists(outcome_plot_path):
                                outcome_plot_path = os.path.join(plots_dir, "psth_by_outcome", f"ofc_unit_{selected_unit_idx}_psth_by_outcome.html")
                            
                            if os.path.exists(outcome_plot_path):
                                with open(outcome_plot_path, 'r', encoding='utf-8') as f:
                                    st.components.v1.html(f.read(), height=600)
            else:
                st.warning("Selectivity metrics file not found in analysis output.")
        
    # with tab3:
    #     st.header("Generalized Linear Model Analysis")
        
    #     if not has_analysis_data:
    #         st.warning("⚠️ Analysis output not found. Please run the offline analysis first.")
    #     else:
    #         # Load comprehensive unit metrics
    #         units_metrics_path = os.path.join(analysis_output_dir, "tables", "acx_all_units_metrics.csv")
    #         if not os.path.exists(units_metrics_path):
    #             units_metrics_path = os.path.join(analysis_output_dir, "tables", "ofc_all_units_metrics.csv")
            
    #         if os.path.exists(units_metrics_path):
    #             units_metrics_df = pd.read_csv(units_metrics_path)
    #             st.success(f"✅ Loaded comprehensive metrics for {len(units_metrics_df)} units")
                
    #             # Unit selection
    #             st.subheader("Unit Selection")
    #             if 'unit_idx' in units_metrics_df.columns:
    #                 unit_options = units_metrics_df['unit_idx'].tolist()
    #                 selected_unit_idx = st.selectbox("Select Unit", unit_options, key="glm_unit_select")
                    
    #                 # Display comprehensive metrics for selected unit
    #                 unit_data = units_metrics_df[units_metrics_df['unit_idx'] == selected_unit_idx].iloc[0]
                    
    #                 st.subheader(f"Comprehensive Metrics for Unit {selected_unit_idx}")
                    
    #                 # Display key metrics in columns
    #                 metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                    
    #                 with metrics_col1:
    #                     st.write("**Response Properties**")
    #                     if 'response_type' in unit_data:
    #                         st.metric("Response Type", str(unit_data['response_type']))
    #                     if 'onset_latency' in unit_data and not pd.isna(unit_data['onset_latency']):
    #                         st.metric("Onset Latency", f"{unit_data['onset_latency']:.3f}s")
    #                     if 'peak_latency' in unit_data and not pd.isna(unit_data['peak_latency']):
    #                         st.metric("Peak Latency", f"{unit_data['peak_latency']:.3f}s")
    #                     if 'response_magnitude' in unit_data and not pd.isna(unit_data['response_magnitude']):
    #                         st.metric("Response Magnitude", f"{unit_data['response_magnitude']:.2f} spikes/s")
                    
    #                 with metrics_col2:
    #                     st.write("**Selectivity Metrics**")
    #                     if 'd_prime' in unit_data and not pd.isna(unit_data['d_prime']):
    #                         st.metric("d' (Go/NoGo)", f"{unit_data['d_prime']:.3f}")
    #                     if 'choice_probability' in unit_data and not pd.isna(unit_data['choice_probability']):
    #                         st.metric("Choice Probability", f"{unit_data['choice_probability']:.3f}")
    #                     if 'outcome_p_value' in unit_data and not pd.isna(unit_data['outcome_p_value']):
    #                         st.metric("Outcome p-value", f"{unit_data['outcome_p_value']:.3g}")
                    
    #                 with metrics_col3:
    #                     st.write("**PSTH Metrics**")
    #                     if 'baseline_rate' in unit_data and not pd.isna(unit_data['baseline_rate']):
    #                         st.metric("Baseline Rate", f"{unit_data['baseline_rate']:.2f} spikes/s")
    #                     if 'peak_rate' in unit_data and not pd.isna(unit_data['peak_rate']):
    #                         st.metric("Peak Rate", f"{unit_data['peak_rate']:.2f} spikes/s")
    #                     if 'signal_to_noise' in unit_data and not pd.isna(unit_data['signal_to_noise']):
    #                         st.metric("Signal-to-Noise", f"{unit_data['signal_to_noise']:.2f}")
                    
    #                 # Display full metrics table
    #                 st.subheader("All Metrics")
    #                 st.dataframe(unit_data.to_frame().T, use_container_width=True)
                    
    #                 # Load and display saved plots
    #                 plots_dir = os.path.join(analysis_output_dir, "plots")
    #                 if os.path.exists(plots_dir):
    #                     # Check for heatmap
    #                     heatmap_plot_path = os.path.join(plots_dir, "raw_psth", f"acx_unit_{selected_unit_idx}_heatmap.html")
    #                     if not os.path.exists(heatmap_plot_path):
    #                         heatmap_plot_path = os.path.join(plots_dir, "raw_psth", f"ofc_unit_{selected_unit_idx}_heatmap.html")
                        
    #                     if os.path.exists(heatmap_plot_path):
    #                         st.subheader("Unit Heatmap")
    #                         with open(heatmap_plot_path, 'r', encoding='utf-8') as f:
    #                             st.components.v1.html(f.read(), height=600)
    #         else:
    #             st.warning("Comprehensive unit metrics file not found in analysis output.")
        
  
    
    # with qa_tab:
    #     st.header("Quality Assurance")
    #     import PIL.Image
    #     import base64
    #     from io import BytesIO

    #     if selected_folder is not None:
    #         # Determine parent directory
    #         if os.path.basename(selected_folder) == "analysis_output":
    #             parent_dir = os.path.dirname(selected_folder)
    #         else:
    #             parent_dir = selected_folder
            
    #         # Check for bombcell folder
    #         qa_folder = os.path.join(parent_dir, "bombcell")
    #         has_bombcell = os.path.exists(qa_folder)
            
    #         # Check for analysis output summary plots
    #         if has_analysis_data and analysis_output_dir is not None:
    #             st.subheader("Analysis Output Summary")
    #             plots_dir = os.path.join(analysis_output_dir, "plots")
                
    #             # Try to load summary plots
    #             summary_plots = []
    #             if os.path.exists(plots_dir):
    #                 # Check for ACx summary plots
    #                 acx_dir = os.path.join(plots_dir, "acx")
    #                 if os.path.exists(acx_dir):
    #                     acx_metrics_plot = os.path.join(acx_dir, "acx_selectivity_metrics_summary.html")
    #                     acx_class_plot = os.path.join(acx_dir, "acx_unit_classification_summary.html")
    #                     if os.path.exists(acx_metrics_plot):
    #                         st.write("**ACx Selectivity Metrics Summary**")
    #                         with open(acx_metrics_plot, 'r', encoding='utf-8') as f:
    #                             st.components.v1.html(f.read(), height=500)
    #                     if os.path.exists(acx_class_plot):
    #                         st.write("**ACx Unit Classification Summary**")
    #                         with open(acx_class_plot, 'r', encoding='utf-8') as f:
    #                             st.components.v1.html(f.read(), height=500)
                    
    #                 # Check for OFC summary plots
    #                 ofc_dir = os.path.join(plots_dir, "ofc")
    #                 if os.path.exists(ofc_dir):
    #                     ofc_metrics_plot = os.path.join(ofc_dir, "ofc_selectivity_metrics_summary.html")
    #                     ofc_class_plot = os.path.join(ofc_dir, "ofc_unit_classification_summary.html")
    #                     if os.path.exists(ofc_metrics_plot):
    #                         st.write("**OFC Selectivity Metrics Summary**")
    #                         with open(ofc_metrics_plot, 'r', encoding='utf-8') as f:
    #                             st.components.v1.html(f.read(), height=500)
    #                     if os.path.exists(ofc_class_plot):
    #                         st.write("**OFC Unit Classification Summary**")
    #                         with open(ofc_class_plot, 'r', encoding='utf-8') as f:
    #                             st.components.v1.html(f.read(), height=500)
                    
    #                 # Check for comparison plots
    #                 comparison_dir = os.path.join(plots_dir, "comparison")
    #                 if os.path.exists(comparison_dir):
    #                     comparison_plot = os.path.join(comparison_dir, "ofc_vs_acx_selectivity_comparison.html")
    #                     if os.path.exists(comparison_plot):
    #                         st.write("**OFC vs ACx Comparison**")
    #                         with open(comparison_plot, 'r', encoding='utf-8') as f:
    #                             st.components.v1.html(f.read(), height=500)
            
    #         # Bombcell QA section
    #         if has_bombcell:
    #             st.subheader("Bombcell Quality Metrics")
    #             st.write(f"QA folder: {qa_folder}")
                
    #         # Try the first path; if not found, try the second path
    #         distribution_img_path = os.path.join(qa_folder, "quality_metrics_histograms.png")
    #         classification_img_path = os.path.join(qa_folder, "waveforms_overlay.png")
    #         if not (os.path.exists(distribution_img_path) and os.path.exists(classification_img_path)):
    #             distribution_img_path = os.path.join(qa_folder, "quality_metrics_distribution.png")
    #             classification_img_path = os.path.join(qa_folder, "waveform_classification.png")

    #         def pil_image_to_data_uri(image):
    #             """Convert a PIL Image to a data URI for plotly."""
    #             buffered = BytesIO()
    #             image.save(buffered, format="PNG")
    #             img_bytes = buffered.getvalue()
    #             img_b64 = base64.b64encode(img_bytes).decode()
    #             return f"data:image/png;base64,{img_b64}"

    #         def plot_image(img_path, title="Image"):
    #             if os.path.exists(img_path):
    #                 image = PIL.Image.open(img_path)
    #                 data_uri = pil_image_to_data_uri(image)
    #                 width, height = image.size
    #                 fig_img = go.Figure()
    #                 fig_img.add_layout_image(
    #                     dict(
    #                         source=data_uri,
    #                         xref="x",
    #                         yref="y",
    #                         x=0,
    #                         y=0,
    #                         sizex=width,
    #                         sizey=height,
    #                         layer="below"
    #                     )
    #                 )
    #                 fig_img.update_xaxes(visible=False, range=[0, width])
    #                 fig_img.update_yaxes(visible=False, range=[height, 0])
    #                 fig_img.update_layout(
    #                     title=title,
    #                     margin=dict(l=0, r=0, t=40, b=0),
    #                     width=width,
    #                     height=height
    #                 )
    #                 st.plotly_chart(fig_img, use_container_width=True)
    #             else:
    #                 st.warning(f"Image not found: {img_path}")
                
    #         col1, col2 = st.columns(2)
    #         with col1:      
    #             plot_image(distribution_img_path, title="Quality Metrics Distribution")
    #         with col2:
    #             plot_image(classification_img_path, title="Waveform Classification")

    #         # Load TSV file as a numpy array of unit labels
    #         units_labels_file = os.path.join(qa_folder, "unit_labels.tsv")
    #         if os.path.exists(units_labels_file):
    #             good_units = pd.read_csv(units_labels_file, sep="\t")
    #             good_units = good_units[good_units["UnitType"] == 1]
    #             st.write("Indices of units with UnitType == 1:", good_units.index.tolist())
    #             good_idxs = good_units.index.tolist()

    #             waveforms_file = os.path.join(qa_folder, "templates._bc_rawWaveforms.npy")
    #             if os.path.exists(waveforms_file):
    #                 waveforms = np.load(waveforms_file)

    #                 def plot_waveforms(waveforms, idxs):
    #                     # Plot the waveforms[:,0,:] as a line plot using plotly.graph_objects
    #                     if waveforms.ndim == 3:
    #                         # waveforms shape: (n_units, n_channels, n_samples)
    #                         # waveforms[:,0,:] shape: (n_units, n_samples)
    #                         n_units, n_samples = waveforms[idxs,:,:].shape
    #                         fig_wave = go.Figure()
    #                         # Plot each unit's waveform in transparent gray
    #                         for i in range(n_units):
    #                             fig_wave.add_trace(
    #                                 go.Scatter(
    #                                     y=waveforms[idxs, i, :],
    #                                     mode='lines',
    #                                     line=dict(color='rgba(100,100,100,0.2)', width=1),
    #                                     name=f'Unit {i}',
    #                                     showlegend=False
    #                                 )
    #                             )
    #                         # Add black average line
    #                         avg_waveform = np.mean(waveforms[idxs, :, :], axis=0)
    #                         fig_wave.add_trace(
    #                             go.Scatter(
    #                                 y=avg_waveform,
    #                                 mode='lines',
    #                                 line=dict(color='black', width=2),
    #                                 name='Average',
    #                                 showlegend=True
    #                             )
    #                         )
    #                         fig_wave.update_layout(
    #                             title=f"Raw Waveforms (Channel {idxs})",
    #                             xaxis_title="Sample",
    #                             yaxis_title="Amplitude",
    #                             showlegend=True,
    #                             margin=dict(l=40, r=20, t=40, b=40)
    #                         )
    #                         st.plotly_chart(fig_wave, use_container_width=True)
    #                     else:
    #                         st.warning("Waveforms array does not have expected 3D shape.")
       
    #                 for idx in good_idxs[:10]:
    #                     plot_waveforms(waveforms, idx)
    #         else:
    #             st.info("Bombcell folder not found. Only analysis output summary is available.")
    #     else:
    #         st.warning("No folder selected. Please select an analysis output folder.")