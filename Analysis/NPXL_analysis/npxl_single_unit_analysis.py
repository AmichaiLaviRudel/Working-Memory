import re
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score, roc_curve
import pandas as pd
import os
import streamlit as st
import streamlit.components.v1 as components
from functools import lru_cache
import json

# Pre-compile once; matches <script src="https://cdn.plot.ly/..."></script>
_CDN_PLOTLY_RE = re.compile(
    r'<script[^>]+src=["\']https://cdn\.plot\.ly/[^"\']*["\'][^>]*>\s*</script>',
    re.IGNORECASE | re.DOTALL,
)

@lru_cache(maxsize=1)
def _plotly_js_inline() -> str:
    """Return the minified plotly.js wrapped in a <script> tag (cached once per session)."""
    from plotly.offline import get_plotlyjs
    return f'<script type="text/javascript">{get_plotlyjs()}</script>'

def ensure_plotlyjs_inline(html: str) -> str:
    """
    Replace a CDN plotly.js <script> tag with an inline bundled version.
    No-op if the file already has inline JS or no CDN reference.
    """
    if not _CDN_PLOTLY_RE.search(html):
        return html
    return _CDN_PLOTLY_RE.sub(_plotly_js_inline(), html, count=1)
            
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
from Analysis.GNG_bpod_analysis.colors import COLOR_GO, COLOR_GRAY, COLOR_NOGO, COLOR_HIT, COLOR_FA, COLOR_CR, COLOR_MISS, COLOR_BLUE, COLOR_BLUE_TRANSPARENT, COLOR_ACCENT, COLOR_ACCENT_TRANSPARENT, LEARNING_STAGE_COLORS
from Analysis.GNG_bpod_analysis.GNG_bpod_general import normalize_workspace_path, resolve_analysis_path
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


def _category_boundary_khz() -> tuple[float, float]:
    """Low/high category boundaries (kHz); match ``session_states.initialize_session_state`` defaults."""
    _default_lo, _default_hi = 0.983, 1.525
    try:
        import streamlit as st

        lo = float(st.session_state.get("low_boundary", _default_lo))
        hi = float(st.session_state.get("high_boundary", _default_hi))
        return lo, hi
    except Exception:
        return _default_lo, _default_hi


def plot_tuning_curves_heatmap(
    selectivity_df,
    use_log_scale=True,
    normalize_per_unit=True,
    *,
    fixed_x_range: tuple[float, float] | None = None,
    shade_x_range: tuple[float, float] | None = None,
):
    """
    Create a heatmap of all units' tuning curves, sorted by best frequency (lowest first).
    
    Parameters:
    -----------
    selectivity_df : pd.DataFrame
        DataFrame containing tuning curve data with columns:
        - tuning_curve_stimuli: list or JSON string of stimulus frequencies
        - tuning_curve: list or JSON string of firing rates
        - best_stimulus: best frequency for each unit
        - unit_idx: unit index
    use_log_scale : bool, optional
        If True, use log scale for x-axis (default: True)
    normalize_per_unit : bool, optional
        If True, normalize each unit's curve to [0, 1] (default: True)
        If False, use raw firing rates
    fixed_x_range : (lo, hi) kHz, optional
        If set, frequency grid and axis limits use this span instead of the data-driven median range
        (used for FRA-style sweeps e.g. 5–40 kHz).
    shade_x_range : (lo, hi) kHz, optional
        If set, a semi-transparent vertical band is drawn between these frequencies (same units as stimuli).

    Returns:
    --------
    go.Figure
        Plotly figure with heatmap of all tuning curves
    """
    def _parse_value(val):
        """Parse value from JSON string, Python list string, or return as-is."""
        if isinstance(val, (list, np.ndarray)):
            return val if len(val) > 0 else None
        if pd.isna(val) or val == '' or val == '[]':
            return None
        if isinstance(val, str):
            try:
                return json.loads(val)
            except json.JSONDecodeError:
                try:
                    return eval(val)
                except Exception:
                    return None
        return val
    
    # Collect all valid tuning curves
    tuning_data = []
    for idx, row in selectivity_df.iterrows():
        if 'tuning_curve_stimuli' not in row or 'tuning_curve' not in row:
            continue
        
        stimuli = _parse_value(row['tuning_curve_stimuli'])
        curve = _parse_value(row['tuning_curve'])
        best_stim = row.get('best_stimulus', None)
        unit_idx = row.get('unit_idx', idx)
        
        if stimuli is None or curve is None:
            continue
        if not isinstance(stimuli, (list, tuple, np.ndarray)) or not isinstance(curve, (list, tuple, np.ndarray)):
            continue
        if len(stimuli) == 0 or len(curve) == 0 or len(stimuli) != len(curve):
            continue
        
        # Convert to numpy arrays
        stimuli_arr = np.array(stimuli)
        curve_arr = np.array(curve)
        
        # Filter out invalid values
        valid_mask = np.isfinite(stimuli_arr) & np.isfinite(curve_arr) & (stimuli_arr > 0 if use_log_scale else True)
        if np.sum(valid_mask) == 0:
            continue
        
        tuning_data.append({
            'unit_idx': unit_idx,
            'stimuli': stimuli_arr[valid_mask],
            'curve': curve_arr[valid_mask],
            'best_stimulus': best_stim if pd.notna(best_stim) else np.max(stimuli_arr[valid_mask])
        })
    
    if len(tuning_data) == 0:
        return None
    
    # Sort by best_stimulus (ascending - lowest first)
    tuning_data.sort(key=lambda x: x['best_stimulus'] if x['best_stimulus'] is not None else float('inf'))
    
    # Find common stimulus range for interpolation.
    if fixed_x_range is not None:
        min_stim, max_stim = float(fixed_x_range[0]), float(fixed_x_range[1])
        if min_stim > max_stim:
            min_stim, max_stim = max_stim, min_stim
    else:
        # Use per-unit min/max and take the median across units to avoid outlier
        # units (e.g. Hz vs kHz mismatch) from stretching the shared x-axis.
        per_unit_min = np.array([np.min(d["stimuli"]) for d in tuning_data])
        per_unit_max = np.array([np.max(d["stimuli"]) for d in tuning_data])
        min_stim = float(np.median(per_unit_min))
        max_stim = float(np.median(per_unit_max))

    # Create common x-axis (150 points for better resolution)
    if use_log_scale:
        if min_stim <= 0 or max_stim <= 0:
            return None
        x_common = np.logspace(np.log10(min_stim), np.log10(max_stim), 150)
    else:
        x_common = np.linspace(min_stim, max_stim, 150)
    
    # Interpolate all curves to common x-axis and optionally normalize
    heatmap_matrix = []
    unit_indices = []
    best_frequencies = []
    
    for d in tuning_data:
        # Interpolate to common x-axis; NaN outside the unit's own stimulus range
        # so extrapolated regions don't show as flat boundary values.
        curve_interp = np.interp(x_common, d['stimuli'], d['curve'],
                                 left=np.nan, right=np.nan)
        # Smooth only finite values; preserve NaN for out-of-range regions
        nan_mask = np.isnan(curve_interp)
        curve_smooth = curve_interp.copy()
        if not nan_mask.all():
            curve_smooth[~nan_mask] = gaussian_filter1d(curve_interp[~nan_mask], sigma=1.0)

        if normalize_per_unit:
            finite_vals = curve_smooth[~nan_mask]
            curve_min = np.min(finite_vals) if len(finite_vals) else 0.0
            curve_max = np.max(finite_vals) if len(finite_vals) else 0.0
            if curve_max > curve_min:
                curve_norm = (curve_smooth - curve_min) / (curve_max - curve_min)
            else:
                curve_norm = np.where(nan_mask, np.nan, 0.0)
            curve_norm = np.where(nan_mask, np.nan, np.maximum(curve_norm, 0))
            heatmap_matrix.append(curve_norm)
        else:
            curve_smooth = np.where(nan_mask, np.nan, np.maximum(curve_smooth, 0))
            heatmap_matrix.append(curve_smooth)
        
        unit_indices.append(d['unit_idx'])
        best_frequencies.append(d['best_stimulus'])
    
    # Convert to numpy array (rows = units, columns = frequencies)
    heatmap_matrix = np.array(heatmap_matrix)

    low_boundary, high_boundary = _category_boundary_khz()

    from plotly.subplots import make_subplots

    # KDE of best frequencies projected onto the shared x-axis
    bf_arr = np.array(best_frequencies, dtype=float)
    bf_finite = bf_arr[np.isfinite(bf_arr)]
    # bw_method as a scalar directly sets the bandwidth factor;
    # 0.15 is ~2x narrower than Scott's rule for sharper resolution.
    KDE_BW = 0.04
    if use_log_scale and len(bf_finite):
        # KDE in log space so it looks natural on a log x-axis
        log_bf = np.log10(bf_finite)
        kde = stats.gaussian_kde(log_bf, bw_method=KDE_BW)
        kde_y = kde(np.log10(x_common))
    else:
        kde = stats.gaussian_kde(bf_finite, bw_method=KDE_BW)
        kde_y = kde(x_common)
    kde_y = kde_y / kde_y.max()  # normalize to [0, 1] for display

    # Build combined figure: heatmap (80%) + KDE strip (20%)
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.82, 0.18],
        shared_xaxes=True,
        vertical_spacing=0.02,
    )

    # --- Heatmap ---
    customdata_matrix = np.tile(bf_arr[:, np.newaxis], (1, len(x_common)))
    fig.add_trace(go.Heatmap(
        z=heatmap_matrix,
        x=x_common,
        y=[f"Unit {uid}" for uid in unit_indices],
        colorscale='Viridis',
        colorbar=dict(
            title="Norm. Response" if normalize_per_unit else "Firing Rate (sp/s)",
            len=0.82, y=0.59, yanchor='middle',
        ),
        hovertemplate='Unit: %{y}<br>Frequency: %{x:.2f} kHz<br>Response: %{z:.3f}<br>Best Freq: %{customdata:.2f} kHz<extra></extra>',
        customdata=customdata_matrix,
    ), row=1, col=1)

    # --- KDE projection ---
    fig.add_trace(go.Scatter(
        x=x_common,
        y=kde_y,
        mode='lines',
        fill='tozeroy',
        fillcolor='rgba(100,180,255,0.25)',
        line=dict(color='rgba(80,160,240,0.9)', width=1.5),
        hovertemplate='Frequency: %{x:.2f} kHz<br>Density: %{y:.3f}<extra></extra>',
        name='Best-freq density',
        showlegend=False,
    ), row=2, col=1)

    # --- Optional background band (e.g. passband / behaviourally relevant band) ---
    if shade_x_range is not None:
        sx0, sx1 = float(shade_x_range[0]), float(shade_x_range[1])
        if sx0 > sx1:
            sx0, sx1 = sx1, sx0
        _shade = dict(
            x0=sx0,
            x1=sx1,
            fillcolor="rgba(120, 140, 200, 0.22)",
            layer="below",
            line_width=0,
        )
        fig.add_vrect(**_shade, row=1, col=1)
        fig.add_vrect(**_shade, row=2, col=1)

    # --- Dashed vertical lines at session category boundaries (heatmap + KDE strip) ---
    xc_lo, xc_hi = float(x_common[0]), float(x_common[-1])
    if xc_lo > xc_hi:
        xc_lo, xc_hi = xc_hi, xc_lo
    for boundary in (low_boundary, high_boundary):
        if not np.isfinite(boundary):
            continue
        if boundary < xc_lo or boundary > xc_hi:
            continue
        for row in (1, 2):
            fig.add_vline(
                x=boundary,
                row=row,
                col=1,
                line_dash="dash",
                line_color="rgba(255, 255, 255, 0.92)",
                line_width=2,
                opacity=0.9,
            )

    x_axis_cfg: dict = dict(type='log' if use_log_scale else 'linear', showgrid=False)
    if fixed_x_range is not None:
        lo, hi = float(fixed_x_range[0]), float(fixed_x_range[1])
        if lo > hi:
            lo, hi = hi, lo
        if use_log_scale and lo > 0 and hi > 0:
            x_axis_cfg["range"] = [np.log10(lo), np.log10(hi)]
        elif not use_log_scale:
            x_axis_cfg["range"] = [lo, hi]
    heatmap_height = min(700, max(300, 300 + len(tuning_data) * 3))
    fig.update_layout(
        title="Tuning Curves Heatmap (Sorted by Best Frequency - Lowest First)",
        xaxis2=dict(title="Frequency (kHz)" + (" [log scale]" if use_log_scale else ""), **x_axis_cfg),
        xaxis=dict(**x_axis_cfg),
        yaxis=dict(showgrid=False, autorange='reversed', showticklabels=False),
        yaxis2=dict(showgrid=False, showticklabels=False, title='Density'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=60, r=40, t=60, b=60),
        height=heatmap_height + 120,
    )

    return fig


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


def _nonempty_sorted_values(df: pd.DataFrame, column: str) -> list:
    if column not in df.columns:
        return []

    values = df[column].dropna().astype(str).str.strip()
    return sorted(value for value in values.unique().tolist() if value)


def _numeric_metric_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "mean_firing_rate",
        "psth_response_magnitude",
        "psth_peak_latency",
        "psth_onset_latency",
        "psth_signal_to_noise",
        "go_nogo_dprime",
        "go_nogo_roc_auc",
        "choice_probability",
        "tone_p_value",
        "choice_p_value",
        "outcome_p_value",
        "category_p_value",
    ]
    return [
        column
        for column in preferred
        if column in df.columns and pd.to_numeric(df[column], errors="coerce").notna().any()
    ]


def _apply_multi_session_unit_filters(units_df: pd.DataFrame) -> pd.DataFrame:
    filtered_df = units_df.copy()

    with st.expander("Smart Filters", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            session_types = _nonempty_sorted_values(units_df, "session_type")
            selected_session_types = st.multiselect(
                "Session type",
                session_types,
                default=session_types,
                key="multi_unit_session_type_filter",
            )
            brain_areas = _nonempty_sorted_values(units_df, "brain_area")
            selected_brain_areas = st.multiselect(
                "Recording brain area",
                brain_areas,
                default=brain_areas,
                key="multi_unit_brain_area_filter",
            )

        with col2:
            histology_regions = _nonempty_sorted_values(units_df, "histology_region")
            selected_histology_regions = st.multiselect(
                "Aligned histological area",
                histology_regions,
                default=histology_regions,
                key="multi_unit_histology_filter",
            )
            cortex_groups = _nonempty_sorted_values(units_df, "cortex_group")
            selected_cortex_groups = st.multiselect(
                "Cortex group",
                cortex_groups,
                default=cortex_groups,
                key="multi_unit_cortex_group_filter",
            )

        with col3:
            unit_types = _nonempty_sorted_values(units_df, "unit_type")
            default_unit_types = ["good"] if "good" in unit_types else unit_types
            selected_unit_types = st.multiselect(
                "Unit type",
                unit_types,
                default=default_unit_types,
                key="multi_unit_type_filter",
            )
            search_text = st.text_input(
                "Search animal/session/unit",
                key="multi_unit_search_text",
                placeholder="e.g. G7A3, novice, 250",
            ).strip()

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            if "mean_firing_rate" in units_df.columns:
                rates = pd.to_numeric(units_df["mean_firing_rate"], errors="coerce").dropna()
                if not rates.empty and float(rates.min()) < float(rates.max()):
                    min_rate, max_rate = st.slider(
                        "Mean firing rate",
                        min_value=float(rates.min()),
                        max_value=float(rates.max()),
                        value=(float(rates.min()), float(rates.max())),
                        key="multi_unit_firing_rate_range",
                    )
                    filtered_df = filtered_df[
                        pd.to_numeric(filtered_df["mean_firing_rate"], errors="coerce").between(min_rate, max_rate)
                    ]

        with metric_col2:
            pvalue_columns = [
                column
                for column in ["tone_p_value", "choice_p_value", "outcome_p_value", "category_p_value"]
                if column in units_df.columns
            ]
            if pvalue_columns:
                selected_pvalue_column = st.selectbox(
                    "P-value filter",
                    pvalue_columns,
                    key="multi_unit_pvalue_column",
                )
                max_pvalue = st.slider(
                    "Max p-value",
                    min_value=0.0,
                    max_value=1.0,
                    value=1.0,
                    step=0.01,
                    key="multi_unit_max_pvalue",
                )
                pvalues = pd.to_numeric(filtered_df[selected_pvalue_column], errors="coerce")
                filtered_df = filtered_df[pvalues.isna() | (pvalues <= max_pvalue)]

        with metric_col3:
            if "psth_response_magnitude" in units_df.columns:
                response_magnitudes = pd.to_numeric(units_df["psth_response_magnitude"], errors="coerce").dropna()
                if not response_magnitudes.empty and float(response_magnitudes.min()) < float(response_magnitudes.max()):
                    min_response = st.slider(
                        "Min response magnitude",
                        min_value=float(response_magnitudes.min()),
                        max_value=float(response_magnitudes.max()),
                        value=float(response_magnitudes.min()),
                        key="multi_unit_min_response_magnitude",
                    )
                    filtered_df = filtered_df[
                        pd.to_numeric(filtered_df["psth_response_magnitude"], errors="coerce") >= min_response
                    ]

    if selected_session_types and set(selected_session_types) != set(session_types):
        filtered_df = filtered_df[filtered_df["session_type"].astype(str).isin(selected_session_types)]
    if selected_brain_areas and set(selected_brain_areas) != set(brain_areas):
        filtered_df = filtered_df[filtered_df["brain_area"].astype(str).isin(selected_brain_areas)]
    if (
        selected_histology_regions
        and set(selected_histology_regions) != set(histology_regions)
        and "histology_region" in filtered_df.columns
    ):
        filtered_df = filtered_df[filtered_df["histology_region"].astype(str).isin(selected_histology_regions)]
    if (
        selected_cortex_groups
        and set(selected_cortex_groups) != set(cortex_groups)
        and "cortex_group" in filtered_df.columns
    ):
        filtered_df = filtered_df[filtered_df["cortex_group"].astype(str).isin(selected_cortex_groups)]
    if selected_unit_types and set(selected_unit_types) != set(unit_types) and "unit_type" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["unit_type"].astype(str).isin(selected_unit_types)]

    if search_text:
        search_columns = [
            column
            for column in ["animal", "date", "session_type", "brain_area", "histology_region", "unit_idx", "label_unitID"]
            if column in filtered_df.columns
        ]
        search_blob = filtered_df[search_columns].astype(str).agg(" ".join, axis=1).str.lower()
        filtered_df = filtered_df[search_blob.str.contains(search_text.lower(), regex=False, na=False)]

    return filtered_df


def _plot_multi_session_unit_counts(filtered_df: pd.DataFrame, compare_column: str):
    count_df = (
        filtered_df.groupby(["session_type", compare_column], dropna=False)
        .size()
        .reset_index(name="unit_count")
    )
    return px.bar(
        count_df,
        x="session_type",
        y="unit_count",
        color=compare_column,
        barmode="group",
        title=f"Unit Counts by Session Type and {compare_column.replace('_', ' ').title()}",
        labels={"session_type": "Session Type", "unit_count": "Units"},
    )


def _render_selected_unit_preview(filtered_df: pd.DataFrame) -> None:
    if filtered_df.empty:
        return

    st.subheader("Single Unit Drilldown")
    id_columns = [column for column in ["animal", "date", "session_type", "brain_area", "histology_region", "unit_idx"] if column in filtered_df.columns]
    option_df = filtered_df[id_columns + ["unit_global_id"]].copy()
    option_df["label"] = option_df[id_columns].astype(str).agg(" | ".join, axis=1)
    selected_label = st.selectbox(
        "Select filtered unit",
        option_df["label"].tolist(),
        key="multi_unit_drilldown_select",
    )
    selected_unit_id = option_df.loc[option_df["label"] == selected_label, "unit_global_id"].iloc[0]
    row = filtered_df.loc[filtered_df["unit_global_id"] == selected_unit_id].iloc[0]

    metric_columns = [
        "mean_firing_rate",
        "tone_p_value",
        "choice_p_value",
        "psth_response_magnitude",
        "psth_peak_latency",
        "psth_signal_to_noise",
        "go_nogo_dprime",
    ]
    metric_columns = [column for column in metric_columns if column in row and pd.notna(row[column])]
    if metric_columns:
        cols = st.columns(min(4, len(metric_columns)))
        for idx, column in enumerate(metric_columns):
            value = row[column]
            label = column.replace("_", " ").title()
            cols[idx % len(cols)].metric(label, f"{float(value):.3g}" if isinstance(value, (int, float, np.integer, np.floating)) else str(value))

    plot_options = {
        "Tone PSTH": "psth_tone_path",
        "Choice PSTH": "psth_choice_path",
        "Outcome PSTH": "psth_outcome_path",
        "Tone Heatmap": "heatmap_tone_path",
        "Choice Heatmap": "heatmap_choice_path",
        "Outcome Heatmap": "heatmap_outcome_path",
    }
    available_plots = {
        label: column
        for label, column in plot_options.items()
        if column in row and pd.notna(row[column]) and os.path.exists(resolve_analysis_path(row[column]))
    }

    if not available_plots:
        st.info("No saved PSTH or heatmap HTML exists for this filtered unit.")
        return

    selected_plot = st.selectbox(
        "Saved plot",
        list(available_plots.keys()),
        key="multi_unit_saved_plot_select",
    )
    plot_path = resolve_analysis_path(row[available_plots[selected_plot]])
    try:
        with open(plot_path, "r", encoding="utf-8") as f:
            components.html(ensure_plotlyjs_inline(f.read()), height=520, scrolling=False)
    except Exception as e:
        st.warning(f"Could not render saved plot: {e}")


# Category boundary constants (kHz) – match session_states.py defaults
_BOUNDARY_LOW_KHZ: float = 0.983
_BOUNDARY_HIGH_KHZ: float = 1.525


def _render_tuning_curves_panel(selectivity_df: pd.DataFrame) -> None:
    """Panel B – Mean population tuning curve per session type / brain area."""
    if selectivity_df.empty:
        st.info("No selectivity data available. Run offline single-unit analysis first.")
        return

    # Filters
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        brain_areas = _nonempty_sorted_values(selectivity_df, "brain_area")
        selected_areas = st.multiselect(
            "Brain area",
            brain_areas,
            default=brain_areas,
            key="tc_brain_area",
        )
    with filter_col2:
        selective_only = st.checkbox(
            "Stimulus-selective units only",
            value=False,
            key="tc_selective_only",
        )
    with filter_col3:
        log_x = st.checkbox("Log frequency axis", value=True, key="tc_log_x")

    plot_df = selectivity_df.copy()
    if selected_areas:
        plot_df = plot_df[plot_df["brain_area"].isin(selected_areas)]
    if selective_only and "stimulus_selective" in plot_df.columns:
        plot_df = plot_df[plot_df["stimulus_selective"].astype(str).str.lower() == "true"]

    # Drop rows without parsed tuning data
    plot_df = plot_df.dropna(subset=["tuning_curve_stimuli", "tuning_curve"])
    if plot_df.empty:
        st.info("No tuning curve data available after filtering.")
        return

    # Build a list of (session_type, brain_area) groups; produce one line per group
    group_cols = ["session_type", "brain_area"]
    fig = go.Figure()

    palette = px.colors.qualitative.Plotly
    color_idx = 0

    for (session_type, brain_area), grp in plot_df.groupby(group_cols, dropna=False):
        # Collect all (stimuli, rates) pairs and interpolate onto a common grid
        all_stimuli: list[list[float]] = []
        all_rates: list[list[float]] = []

        for _, row in grp.iterrows():
            stimuli = row["tuning_curve_stimuli"]
            rates = row["tuning_curve"]
            if isinstance(stimuli, list) and isinstance(rates, list) and len(stimuli) == len(rates) and len(stimuli) > 1:
                all_stimuli.append(stimuli)
                all_rates.append(rates)

        if not all_stimuli:
            continue

        # Use the most common stimulus grid as the reference grid
        ref_stimuli = sorted(set(s for stim_list in all_stimuli for s in stim_list))
        aligned_rates = np.full((len(all_rates), len(ref_stimuli)), np.nan)
        for i, (stimuli, rates) in enumerate(zip(all_stimuli, all_rates)):
            stim_map = dict(zip(stimuli, rates))
            for j, s in enumerate(ref_stimuli):
                if s in stim_map:
                    aligned_rates[i, j] = stim_map[s]

        mean_rates = np.nanmean(aligned_rates, axis=0)
        n_units = np.sum(~np.isnan(aligned_rates), axis=0)
        # SEM: std / sqrt(n), nan where n < 2
        with np.errstate(invalid="ignore"):
            sem_rates = np.nanstd(aligned_rates, axis=0, ddof=1) / np.sqrt(np.maximum(n_units, 1))

        color = palette[color_idx % len(palette)]
        color_idx += 1
        label = f"{session_type} | {brain_area} (n={len(all_rates)})"
        x_vals = [float(s) for s in ref_stimuli]

        fig.add_trace(go.Scatter(
            x=x_vals + x_vals[::-1],
            y=(mean_rates + sem_rates).tolist() + (mean_rates - sem_rates).tolist()[::-1],
            fill="toself",
            fillcolor=color.replace("rgb", "rgba").replace(")", ",0.15)") if color.startswith("rgb") else color,
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            showlegend=False,
            name=label + " (SEM)",
        ))
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=mean_rates.tolist(),
            mode="lines+markers",
            name=label,
            line=dict(color=color, width=2),
            marker=dict(size=5),
        ))

    # Shade the category boundary region
    fig.add_vrect(
        x0=_BOUNDARY_LOW_KHZ,
        x1=_BOUNDARY_HIGH_KHZ,
        fillcolor="gray",
        opacity=0.12,
        layer="below",
        line_width=0,
        annotation_text="Boundary",
        annotation_position="top left",
    )

    x_type = "log" if log_x else "linear"
    fig.update_layout(
        title="Population Tuning Curves by Session Type",
        xaxis=dict(title="Frequency (kHz)", type=x_type),
        yaxis=dict(title="Mean firing rate (spikes/s)"),
        legend=dict(orientation="v"),
        height=480,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Shaded band: category boundary ({_BOUNDARY_LOW_KHZ}–{_BOUNDARY_HIGH_KHZ} kHz). "
        "Ribbons = ± 1 SEM across units."
    )


def _render_responsive_units_panel(filtered_df: pd.DataFrame) -> None:
    """Panel C – % responsive units per brain area and session type."""
    pvalue_candidates = [
        col for col in ["tone_p_value", "choice_p_value", "outcome_p_value", "category_p_value"]
        if col in filtered_df.columns
    ]
    if not pvalue_candidates:
        st.info("No p-value columns found in the filtered data.")
        return

    ctrl_col1, ctrl_col2 = st.columns(2)
    with ctrl_col1:
        pval_col = st.selectbox(
            "Responsiveness criterion (p-value)",
            pvalue_candidates,
            key="responsive_pval_col",
        )
    with ctrl_col2:
        threshold = st.slider(
            "Significance threshold (α)",
            min_value=0.001,
            max_value=0.20,
            value=0.05,
            step=0.001,
            format="%.3f",
            key="responsive_threshold",
        )

    work_df = filtered_df.copy()
    work_df["_pval"] = pd.to_numeric(work_df[pval_col], errors="coerce")
    work_df["_responsive"] = work_df["_pval"] < threshold

    # Compute %responsive per (session_index, session_type, brain_area) to get per-session rates
    # then average those to get mean ± SEM per group
    per_session = (
        work_df.groupby(["session_index", "session_type", "brain_area"], dropna=False)
        .agg(
            n_responsive=("_responsive", "sum"),
            n_total=("_responsive", "count"),
        )
        .reset_index()
    )
    per_session = per_session[per_session["n_total"] > 0].copy()
    per_session["pct_responsive"] = 100.0 * per_session["n_responsive"] / per_session["n_total"]

    summary = (
        per_session.groupby(["session_type", "brain_area"], dropna=False)["pct_responsive"]
        .agg(mean="mean", sem=lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0)
        .reset_index()
    )

    fig = go.Figure()
    brain_areas = sorted(summary["brain_area"].dropna().unique().tolist())
    palette = px.colors.qualitative.Plotly

    for idx, area in enumerate(brain_areas):
        area_df = summary[summary["brain_area"] == area].sort_values("session_type")
        fig.add_trace(go.Bar(
            x=area_df["session_type"],
            y=area_df["mean"],
            error_y=dict(type="data", array=area_df["sem"].tolist(), visible=True),
            name=area,
            marker_color=palette[idx % len(palette)],
        ))

    fig.update_layout(
        title=f"% Responsive Units by Session Type ({pval_col} < {threshold})",
        xaxis_title="Session Type",
        yaxis_title="% Responsive Units",
        barmode="group",
        height=440,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Error bars = ± 1 SEM across recording sessions.")


def _render_best_frequency_panel(filtered_df: pd.DataFrame) -> None:
    """Panel D – Population distribution of best frequencies near the category boundary."""
    if "best_stimulus" not in filtered_df.columns:
        st.info("best_stimulus column not found in the filtered data.")
        return

    work_df = filtered_df.copy()
    work_df["best_stimulus"] = pd.to_numeric(work_df["best_stimulus"], errors="coerce")
    work_df = work_df.dropna(subset=["best_stimulus"])
    if work_df.empty:
        st.info("No best_stimulus data available after filtering.")
        return

    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns(3)
    with ctrl_col1:
        brain_areas = _nonempty_sorted_values(work_df, "brain_area")
        selected_areas = st.multiselect(
            "Brain area",
            brain_areas,
            default=brain_areas,
            key="bf_brain_area",
        )
    with ctrl_col2:
        n_bins = st.slider("Histogram bins", min_value=5, max_value=40, value=15, key="bf_nbins")
    with ctrl_col3:
        log_x = st.checkbox("Log frequency axis", value=True, key="bf_log_x")

    if selected_areas:
        work_df = work_df[work_df["brain_area"].isin(selected_areas)]

    session_types = sorted(work_df["session_type"].dropna().unique().tolist())
    areas_in_data = sorted(work_df["brain_area"].dropna().unique().tolist())
    n_areas = len(areas_in_data)

    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=1,
        cols=max(n_areas, 1),
        subplot_titles=[str(a) for a in areas_in_data] if n_areas > 1 else None,
        shared_yaxes=True,
    )

    palette = px.colors.qualitative.Plotly

    # Determine common bin edges in log space so subplots are aligned
    bf_min = float(work_df["best_stimulus"].min())
    bf_max = float(work_df["best_stimulus"].max())
    if log_x and bf_min > 0:
        bin_edges = np.logspace(np.log10(bf_min), np.log10(bf_max), n_bins + 1)
    else:
        bin_edges = np.linspace(bf_min, bf_max, n_bins + 1)

    for col_idx, area in enumerate(areas_in_data, start=1):
        area_df = work_df[work_df["brain_area"] == area]
        for st_idx, session_type in enumerate(session_types):
            st_df = area_df[area_df["session_type"] == session_type]["best_stimulus"].dropna()
            if st_df.empty:
                continue

            counts, _ = np.histogram(st_df.values, bins=bin_edges)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            fig.add_trace(
                go.Bar(
                    x=bin_centers.tolist(),
                    y=counts.tolist(),
                    name=session_type,
                    marker_color=palette[st_idx % len(palette)],
                    opacity=0.7,
                    showlegend=(col_idx == 1),
                    legendgroup=session_type,
                ),
                row=1,
                col=col_idx,
            )

        # Shade boundary zone on every subplot
        for vrect_col in [col_idx]:
            fig.add_vrect(
                x0=_BOUNDARY_LOW_KHZ,
                x1=_BOUNDARY_HIGH_KHZ,
                fillcolor="gray",
                opacity=0.15,
                layer="below",
                line_width=0,
                row=1,
                col=vrect_col,
            )

    x_type = "log" if log_x else "linear"
    fig.update_xaxes(title_text="Best frequency (kHz)", type=x_type)
    fig.update_yaxes(title_text="Unit count", col=1)
    fig.update_layout(
        title="Best Frequency Distribution by Session Type",
        barmode="overlay",
        height=440,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Shaded band: category boundary ({_BOUNDARY_LOW_KHZ}–{_BOUNDARY_HIGH_KHZ} kHz). "
        "Includes stimulus-selective units only where best_stimulus is defined."
    )


def _classify_learning_stage(session_type: str) -> str:
    """Map a session_type string to a canonical learning stage label.

    Matches the naming convention in recording folder names, e.g.
    'catgt_G4A2_novice2_2b_4t_g0' → session_type contains 'novice', '1b', or '2b'.
    Returns 'Novice', '1b Expert', '2b Expert', or 'Other'.
    """
    s = str(session_type).lower()
    if "novice" in s:
        return "Novice"
    if "2b" in s:
        return "2b Expert"
    if "1b" in s:
        return "1b Expert"
    return "Other"


# Alias imported palette so rest of this module uses the same name
_STAGE_COLORS = LEARNING_STAGE_COLORS

KDE_BW_DENSITY = 0.08  # bandwidth factor for overlay density KDE (log space)


_BF_EXCLUDE_CENTER_KHZ = 1.5   # category boundary to exclude from density KDE
_BF_EXCLUDE_MARGIN_KHZ = 0.05  # ±margin around the boundary


def _plot_best_freq_density_overlay(
    plot_df: pd.DataFrame,
    stage_labels: list[str],
    use_log_scale: bool,
    title: str,
    x_range: tuple[float, float] | None = None,
    kde_bw: float = KDE_BW_DENSITY,
    *,
    reference_vlines_khz: tuple[float, ...] | None = None,
) -> go.Figure | None:
    """Overlay best-frequency KDE density curves for the given learning stages.

    BF values within ±_BF_EXCLUDE_MARGIN_KHZ of _BF_EXCLUDE_CENTER_KHZ are
    excluded before KDE to avoid the category boundary dominating the density.

    Args:
        plot_df:      DataFrame that already has a 'learning_stage' column and
                      a numeric 'best_stimulus' column.
        stage_labels: Ordered list of stage names to include (e.g. ["Novice", "1b Expert"]).
        use_log_scale: Whether to compute/display on a log frequency axis.
        title:        Plot title.
        reference_vlines_khz: Optional extra dashed vlines at these frequencies (kHz)
            when they fall inside the plot x-range (e.g. FRA markers at 10 and 15 kHz).
    """
    _bf_lo = _BF_EXCLUDE_CENTER_KHZ - _BF_EXCLUDE_MARGIN_KHZ
    _bf_hi = _BF_EXCLUDE_CENTER_KHZ + _BF_EXCLUDE_MARGIN_KHZ

    all_bf = pd.to_numeric(plot_df["best_stimulus"], errors="coerce").dropna()
    all_bf = all_bf[all_bf > 0] if use_log_scale else all_bf
    # apply boundary exclusion to the shared range estimate too
    all_bf = all_bf[~all_bf.between(_bf_lo, _bf_hi)]
    if all_bf.empty:
        return None

    # Evaluation grid: use explicit x_range when given so the KDE spans the same axis as the heatmaps
    if x_range is not None:
        xr_lo, xr_hi = float(x_range[0]), float(x_range[1])
        if xr_lo > xr_hi:
            xr_lo, xr_hi = xr_hi, xr_lo
        if use_log_scale:
            if xr_lo <= 0 or xr_hi <= 0:
                return None
            x_common = np.logspace(np.log10(xr_lo), np.log10(xr_hi), 300)
        else:
            x_common = np.linspace(xr_lo, xr_hi, 300)
    elif use_log_scale:
        x_common = np.logspace(np.log10(all_bf.min()), np.log10(all_bf.max()), 300)
    else:
        x_common = np.linspace(all_bf.min(), all_bf.max(), 300)

    fig = go.Figure()
    for stage in stage_labels:
        mask = plot_df["learning_stage"] == stage
        bf = pd.to_numeric(plot_df.loc[mask, "best_stimulus"], errors="coerce").dropna()
        bf = bf[bf > 0] if use_log_scale else bf
        # exclude boundary zone
        bf = bf[~bf.between(_bf_lo, _bf_hi)].values
        if len(bf) < 3:
            continue

        line_color, fill_color = _STAGE_COLORS.get(stage, ("#888888", "rgba(136,136,136,0.15)"))
        if use_log_scale:
            kde = stats.gaussian_kde(np.log10(bf), bw_method=kde_bw)
            kde_y = kde(np.log10(x_common))
        else:
            kde = stats.gaussian_kde(bf, bw_method=kde_bw)
            kde_y = kde(x_common)
        kde_y = kde_y / kde_y.max()

        fig.add_trace(go.Scatter(
            x=x_common,
            y=kde_y,
            mode="lines",
            fill="tozeroy",
            fillcolor=fill_color,
            line=dict(color=line_color, width=2),
            name=f"{stage} (n={len(bf)})",
            hovertemplate="Freq: %{x:.2f} kHz<br>Density: %{y:.3f}<extra></extra>",
        ))

    if not fig.data:
        return None

    x_min, x_max = x_range if x_range is not None else (0.6, 2.2)
    if x_min > x_max:
        x_min, x_max = x_max, x_min

    lo_b, hi_b = _category_boundary_khz()
    for xv in (lo_b, hi_b):
        if x_min <= xv <= x_max:
            fig.add_vline(
                x=xv,
                line_width=2,
                line_dash="dash",
                line_color="rgba(45, 45, 55, 0.88)",
            )

    # Optional fixed reference frequencies (e.g. FRA at 10 and 15 kHz)
    if reference_vlines_khz:
        for xv in reference_vlines_khz:
            if not np.isfinite(xv):
                continue
            if x_min <= float(xv) <= x_max:
                fig.add_vline(
                    x=float(xv),
                    line_width=2,
                    line_dash="dash",
                    line_color="rgba(200, 95, 40, 0.92)",
                )

    fig.update_layout(
        title=title,
        xaxis=dict(
            type="log" if use_log_scale else "linear",
            title="Frequency (kHz)" + (" [log scale]" if use_log_scale else ""),
            range=[np.log10(x_min), np.log10(x_max)] if use_log_scale else [x_min, x_max],
            showgrid=True,
        ),
        yaxis=dict(title="Normalised density", showgrid=True, range=[0, 1.1]),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def _render_selectivity_heatmap_panel(
    selectivity_df: pd.DataFrame,
    *,
    widget_key_prefix: str = "sh",
    fixed_tuning_x_range: tuple[float, float] | None = None,
    tuning_shade_x_range: tuple[float, float] | None = None,
) -> None:
    """Panel E – Per-session-type tuning-curve heatmaps sorted by best frequency."""
    if selectivity_df.empty:
        st.info("No selectivity data available. Run offline single-unit analysis first.")
        return

    def _wk(suffix: str) -> str:
        return f"{widget_key_prefix}_{suffix}"

    # --- Shared filter controls ---
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4, ctrl_col5 = st.columns(5)
    with ctrl_col1:
        brain_areas = _nonempty_sorted_values(selectivity_df, "brain_area")
        selected_areas = st.multiselect(
            "Brain area",
            brain_areas,
            default=brain_areas,
            key=_wk("brain_area"),
        )
    with ctrl_col2:
        selective_only = st.checkbox(
            "Stimulus-selective units only",
            value=True,
            key=_wk("selective_only"),
        )
    with ctrl_col3:
        good_units_only = st.checkbox(
            "Good units only",
            value=False,
            key=_wk("good_units_only"),
        )
    with ctrl_col4:
        log_x = st.checkbox("Log frequency axis", value=True, key=_wk("log_x"))
    with ctrl_col5:
        normalize = st.checkbox("Normalize per unit", value=True, key=_wk("normalize"))

    # Apply shared filters (brain area, quality flags) — keep ALL units for density
    filtered_df = selectivity_df.copy()
    # Always drop noise units regardless of other filters
    if "unit_type" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["unit_type"].astype(str).str.lower() != "noise"]
    if selected_areas:
        filtered_df = filtered_df[filtered_df["brain_area"].isin(selected_areas)]
    if selective_only and "stimulus_selective" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["stimulus_selective"].astype(str).str.lower() == "true"]
    if good_units_only and "unit_type" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["unit_type"].astype(str).str.lower() == "good"]

    # Attach learning stage before splitting so both views share the same column
    if "session_type" in filtered_df.columns:
        filtered_df["learning_stage"] = filtered_df["session_type"].apply(_classify_learning_stage)
    else:
        filtered_df["learning_stage"] = "Other"

    # density_df: only needs best_stimulus — do NOT drop rows missing tuning curves
    density_df = filtered_df.dropna(subset=["best_stimulus"]).copy()

    # plot_df (heatmap): requires both tuning-curve columns to be present
    plot_df = filtered_df.dropna(subset=["tuning_curve_stimuli", "tuning_curve"]).copy()

    if plot_df.empty:
        st.info("No tuning curve data available after filtering.")
        return

    session_types = sorted(plot_df["session_type"].dropna().unique().tolist())
    if not session_types:
        st.info("No session types found in the data.")
        return

    # --- Density overlay option ---
    show_density = st.checkbox(
        "Show best-frequency density comparison by learning stage",
        value=True,
        key=_wk("density_overlay"),
    )
    if show_density:
        kde_bw = st.slider(
            "KDE smoothing (bandwidth)",
            min_value=0.01,
            max_value=0.30,
            value=KDE_BW_DENSITY,
            step=0.01,
            format="%.2f",
            key=_wk("kde_bw"),
            help="Controls how smooth the density curves are. Lower = sharper peaks, higher = broader.",
        )
        density_areas = sorted(density_df["brain_area"].dropna().unique().tolist()) if "brain_area" in density_df.columns else ["All"]
        for area in density_areas:
            st.markdown(f"**{area}**")
            area_df = density_df[density_df["brain_area"] == area] if area != "All" else density_df

            if fixed_tuning_x_range is not None:
                # FRA: one density figure on 6–22 kHz, learning stages in a single overlay
                fig_fra = _plot_best_freq_density_overlay(
                    area_df,
                    stage_labels=["Novice", "1b Expert", "2b Expert"],
                    use_log_scale=log_x,
                    title=f"{area} — FRA: Novice / 1b Expert / 2b Expert",
                    x_range=(6.0, 22.0),
                    kde_bw=kde_bw,
                    reference_vlines_khz=(10.0, 15.0),
                )
                if fig_fra is not None:
                    st.plotly_chart(fig_fra, use_container_width=True)
                    st.caption(
                        "BF density on **6–22 kHz** axis (FRA). "
                        "Orange dashed lines at **10** and **15 kHz**. "
                        "Dark dashed lines: session low/high boundaries only if inside the axis. "
                        "Tuning heatmaps below use the broadband axis."
                    )
                else:
                    st.info(f"{area}: insufficient data for FRA stage comparison (need ≥3 units/stage).")
            else:
                dcol1, dcol2 = st.columns(2)
                with dcol1:
                    fig_1b = _plot_best_freq_density_overlay(
                        area_df,
                        stage_labels=["Novice", "1b Expert"],
                        use_log_scale=log_x,
                        title=f"{area} — 1b: Novice vs Expert",
                        x_range=(0.68, 1.5),
                        kde_bw=kde_bw,
                    )
                    if fig_1b is not None:
                        st.plotly_chart(fig_1b, use_container_width=True)
                        st.caption("1b context · 'novice' or '1b' sessions · BF ≠ 1.5±0.05 kHz")
                    else:
                        st.info(f"{area}: insufficient data for 1b comparison (need ≥3 units/stage).")
                with dcol2:
                    fig_2b = _plot_best_freq_density_overlay(
                        area_df,
                        stage_labels=["Novice", "1b Expert", "2b Expert"],
                        use_log_scale=log_x,
                        title=f"{area} — 2b: Novice / 1b Expert / 2b Expert",
                        x_range=None,
                        kde_bw=kde_bw,
                    )
                    if fig_2b is not None:
                        st.plotly_chart(fig_2b, use_container_width=True)
                        st.caption("2b context · 'novice', '1b', '2b' sessions · BF ≠ 1.5±0.05 kHz")
                    else:
                        st.info(f"{area}: insufficient data for 2b comparison (need ≥3 units/stage).")
            st.divider()

    # One sub-tab per session type so heatmaps don't stack vertically
    sub_tabs = st.tabs(session_types)
    for tab, session_type in zip(sub_tabs, session_types):
        with tab:
            st_df = plot_df[plot_df["session_type"] == session_type]
            if st_df.empty:
                st.info(f"No tuning curve data for session type '{session_type}'.")
                continue

            fig = plot_tuning_curves_heatmap(
                st_df,
                use_log_scale=log_x,
                normalize_per_unit=normalize,
                fixed_x_range=fixed_tuning_x_range,
                shade_x_range=tuning_shade_x_range,
            )
            if fig is None:
                st.info(f"Could not build heatmap for session type '{session_type}' (insufficient data).")
                continue

            fig.update_layout(
                title=f"Tuning Curves Heatmap — {session_type} (n={len(st_df)} units, sorted by best frequency)"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                f"Each row = one unit. Columns = frequency (kHz). "
                f"Color = {'normalised [0,1]' if normalize else 'firing rate (spikes/s)'}. "
                f"Sorted top → bottom by ascending best frequency."
            )


def multi_session_single_unit_analysis_panel(
    selected_sessions_df: pd.DataFrame,
    sessions_table_df: pd.DataFrame | None = None,
) -> None:
    """Render an aggregated unit-analysis panel across checked NPXL sessions.

    Args:
        selected_sessions_df: Rows the user checked in the session editor (main panel).
        sessions_table_df: Full monitoring table used to discover FRA sessions for the
            bottom tuning section. When omitted, FRA rows are taken only from
            ``selected_sessions_df`` (same source as the checkboxes).
    """
    from Analysis.NPXL_analysis.single_unit_dataset import (
        filter_sessions_by_session_type_contains,
        load_multi_session_unit_metrics,
        load_selectivity_data,
    )

    st.write("### Multi-Session Single Unit Analysis")
    if selected_sessions_df is None or selected_sessions_df.empty:
        st.info("Select one or more sessions to compare units across sessions.")
        return

    with st.spinner("Loading unit metrics and histology mappings..."):
        units_df = load_multi_session_unit_metrics(selected_sessions_df)

    if units_df.empty:
        st.warning("No unit metrics were found for the selected sessions. Run offline single-unit analysis first.")
        return

    filtered_df = _apply_multi_session_unit_filters(units_df)
    matched_mapping = 0
    if "mapping_join_status" in units_df.columns:
        matched_mapping = int((units_df["mapping_join_status"] == "matched").sum())

    metric_cols = st.columns(4)
    metric_cols[0].metric("Filtered Units", f"{len(filtered_df):,}/{len(units_df):,}")
    metric_cols[1].metric("Sessions", units_df["session_index"].nunique())
    metric_cols[2].metric("Histology Matches", f"{matched_mapping:,}/{len(units_df):,}")
    metric_cols[3].metric("Brain Areas", units_df["brain_area"].nunique())


    if filtered_df.empty:
        st.info("No units match the current filters.")
        return

    compare_options = [
        column
        for column in ["brain_area", "histology_region", "cortex_group", "unit_type"]
        if column in filtered_df.columns and filtered_df[column].notna().any()
    ]
    metric_options = _numeric_metric_columns(filtered_df)

    plot_col1, plot_col2 = st.columns(2)
    with plot_col1:
        compare_column = st.selectbox(
            "Compare by",
            compare_options,
            key="multi_unit_compare_column",
        )
        st.plotly_chart(
            _plot_multi_session_unit_counts(filtered_df, compare_column),
            use_container_width=True,
        )

    with plot_col2:
        if metric_options:
            trait_metric = st.selectbox(
                "Trait distribution",
                metric_options,
                key="multi_unit_trait_metric",
            )
            trait_df = filtered_df.copy()
            trait_df[trait_metric] = pd.to_numeric(trait_df[trait_metric], errors="coerce")
            trait_df = trait_df.dropna(subset=[trait_metric])
            if not trait_df.empty:
                fig = px.box(
                    trait_df,
                    x="session_type",
                    y=trait_metric,
                    color=compare_column,
                    points="outliers",
                    title=f"{trait_metric.replace('_', ' ').title()} by Session Type",
                    labels={"session_type": "Session Type", trait_metric: trait_metric.replace("_", " ").title()},
                )
                st.plotly_chart(fig, use_container_width=True)

    if len(metric_options) >= 2:
        scatter_col1, scatter_col2, scatter_col3, scatter_col4 = st.columns(4)
        with scatter_col1:
            x_metric = st.selectbox("Scatter X", metric_options, index=0, key="multi_unit_scatter_x")
        with scatter_col2:
            y_metric = st.selectbox("Scatter Y", metric_options, index=1, key="multi_unit_scatter_y")
        with scatter_col3:
            color_column = st.selectbox(
                "Color by",
                compare_options,
                key="multi_unit_scatter_color",
            )
        with scatter_col4:
            scatter_session_types = _nonempty_sorted_values(filtered_df, "session_type")
            selected_scatter_session_types = st.multiselect(
                "Scatter session type",
                scatter_session_types,
                default=scatter_session_types,
                key="multi_unit_scatter_session_type",
            )

        scatter_df = filtered_df.copy()
        if selected_scatter_session_types and set(selected_scatter_session_types) != set(scatter_session_types):
            scatter_df = scatter_df[scatter_df["session_type"].astype(str).isin(selected_scatter_session_types)]

        scatter_df[x_metric] = pd.to_numeric(scatter_df[x_metric], errors="coerce")
        scatter_df[y_metric] = pd.to_numeric(scatter_df[y_metric], errors="coerce")
        scatter_df = scatter_df.dropna(subset=[x_metric, y_metric])
        if not scatter_df.empty:
            hover_columns = [
                column
                for column in ["animal", "date", "session_type", "brain_area", "histology_region", "unit_idx", "unit_type"]
                if column in scatter_df.columns
            ]
            fig = px.scatter(
                scatter_df,
                x=x_metric,
                y=y_metric,
                color=color_column,
                hover_data=hover_columns,
                title=f"{x_metric.replace('_', ' ').title()} vs {y_metric.replace('_', ' ').title()}",
            )
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Filtered Unit Table")
    display_columns = [
        column
        for column in [
            # identity
            "animal",
            "date",
            "session_type",
            "brain_area",
            "histology_region",
            "cortex_group",
            "unit_type",
            "unit_idx",
            "label_unitID",
            "peak_channel",
            "y_pos",
            # basic activity
            "mean_firing_rate",
            "psth_response_type",
            "psth_baseline_rate",
            "psth_peak_rate",
            "psth_onset_latency",
            "psth_peak_latency",
            "psth_response_magnitude",
            "psth_signal_to_noise",
            "psth_fwhm",
            # tone / p-values
            "tone_p_value",
            "outcome_p_value",
            "choice_p_value",
            "category_p_value",
            # stimulus selectivity
            "stimulus_selective",
            "best_stimulus",
            "max_stimulus_response",
            "min_stimulus_response",
            # outcome modulation
            "outcome_modulated",
            "rewarded_mean_rate",
            "non_rewarded_mean_rate",
            # go/nogo discrimination
            "go_nogo_dprime",
            "go_nogo_roc_auc",
            "go_nogo_selective",
            # choice coding
            "choice_probability",
            "choice_probability_corr",
            "choice_coding",
            # category sensitivity
            "category_sensitive",
            "category_anova_p",
            "best_category",
            # trial counts
            "trial_count_hit",
            "trial_count_miss",
            "trial_count_fa",
            "trial_count_cr",
            # mapping
            "mapping_join_status",
        ]
        if column in filtered_df.columns
    ]
    st.dataframe(filtered_df[display_columns], use_container_width=True, height=420)
    st.download_button(
        "Download filtered units CSV",
        filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_multi_session_units.csv",
        mime="text/csv",
        key="download_multi_session_units",
    )

    _render_selected_unit_preview(filtered_df)

    # ------------------------------------------------------------------
    # Figure 5: Single-Unit Tuning & Plasticity
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("Figure 5: Single-Unit Tuning & Plasticity")
    with st.expander("Methods", expanded=False):
        st.markdown("""
**Tuning curve estimation.**
For each unit, mean firing rate was computed in a post-stimulus window (−0.1 to +0.5 s relative to tone onset)
across all valid trials, averaged per frequency. The **best frequency (BF)** was defined as the tone frequency
that elicited the highest mean firing rate. Units were classified as **stimulus-selective** if the peak-to-trough
range of the tuning curve exceeded twice the maximum SEM across frequencies.

**Stimulus selectivity & coding metrics.**
Four metrics were computed per unit per session:
- *Outcome modulation* — Wilcoxon rank-sum test comparing spike rates on rewarded vs. unrewarded trials (p < 0.05 threshold).
- *Go/NoGo coding* — d′ and ROC-AUC computed from firing rates on Go vs. NoGo trials (selective if |d′| > 0.5).
- *Choice probability* — area under the ROC curve computed from Hit vs. Miss trials, corrected for firing-rate bias (|CP\_corr| > 0.1 threshold).
- *Category sensitivity* — one-way ANOVA across tone-frequency categories; units passing p < 0.05 were flagged as category-sensitive.

**Unit quality.**
Only Kilosort-labelled **"good"** units (single units passing manual or automated curation) are included by default.
MUA and noise labels are available for comparison via the filter controls.

**Best-frequency density comparison.**
Population BF distributions are estimated with a **Gaussian KDE** (bandwidth adjustable via slider) applied
separately for each brain area (ACx, OFC) and each learning stage:
- *1b context* — Novice vs. 1b Expert (x-axis 0.68–1.5 kHz; the 1-boundary category boundary).
- *2b context* — Novice, 1b Expert, and 2b Expert (x-axis 0.6–2.2 kHz; the 2-boundary category range).

Units with BF within ±0.05 kHz of the 1.5 kHz boundary are excluded from density estimation to prevent the
boundary itself from artificially inflating density in that region.

**Heatmaps.**
Tuning curves are interpolated to a common 150-point log-spaced frequency axis, optionally smoothed (σ = 1 bin,
Gaussian) and normalised per unit to \[0, 1\]. Rows are sorted by ascending BF. A KDE strip below each heatmap
projects the BF distribution of that session type.
        """)

    with st.spinner("Loading selectivity / tuning-curve data..."):
        selectivity_df = load_selectivity_data(selected_sessions_df)

    _render_selectivity_heatmap_panel(selectivity_df, widget_key_prefix="sh")

    # ------------------------------------------------------------------
    # FRA sessions: same tuning landscape (all table rows with "FRA" in session type)
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("FRA sessions: Tuning landscape")
    st.caption(
        "Sessions whose session type contains “FRA”, taken from the full monitoring table "
        "when available so they are included even if the multi-session checkbox defaults exclude FRA. "
        "Tuning heatmaps use a fixed 5–40 kHz frequency axis with a shaded 7–21 kHz band."
    )
    with st.expander("Methods (same as Figure 5)", expanded=False):
        st.markdown("""
Same tuning-curve pipeline as **Figure 5** (mean rates per frequency, BF, heatmaps, optional
best-frequency KDE by learning stage). Only units from sessions whose **Session Type** contains
`FRA` are loaded here.
        """)

    fra_source_df = sessions_table_df if sessions_table_df is not None else selected_sessions_df
    fra_sessions_df = filter_sessions_by_session_type_contains(fra_source_df, "FRA")
    if fra_sessions_df.empty:
        st.info("No sessions with “FRA” in the session type were found in the monitoring table.")
    else:
        preview_cols = [
            c
            for c in [
                "Animal",
                "Date",
                "Session Type",
                "SessionType",
                "session_type",
                "current_dir",
                "Current Dir",
                "RecordingDir",
            ]
            if c in fra_sessions_df.columns
        ]
        with st.expander(f"FRA sessions included ({len(fra_sessions_df)})", expanded=False):
            st.dataframe(
                fra_sessions_df[preview_cols] if preview_cols else fra_sessions_df,
                use_container_width=True,
                height=min(360, 60 + 28 * len(fra_sessions_df)),
            )

        with st.spinner("Loading FRA selectivity / tuning-curve data..."):
            fra_selectivity_df = load_selectivity_data(fra_sessions_df)

        if fra_selectivity_df.empty:
            st.info(
                "No selectivity CSVs found for these FRA sessions (run offline single-unit analysis "
                "on each recording, or verify analysis_output/tables/*_selectivity_metrics.csv)."
            )
        else:
            _render_selectivity_heatmap_panel(
                fra_selectivity_df,
                widget_key_prefix="fra_sh",
                fixed_tuning_x_range=(5.0, 40.0),
                tuning_shade_x_range=(7.0, 21.0),
            )


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


        # Unit Selection Section
        st.subheader("Unit Selection")
        unit_col1, unit_col2 = st.columns(2)
        
        row = None  # Initialize row to avoid UnboundLocalError
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
            elif units_metrics_df_sorted is not None:
                # If we have sorted df but no pvals, use first row
                unit_rank = st.slider(
                    "Unit Rank", 
                    0, len(units_metrics_df_sorted) - 1, 0,
                    help="Select unit by rank"
                )
                row = units_metrics_df_sorted.iloc[unit_rank]
                unit_idx = int(row["unit_idx"]) if "unit_idx" in row else unit_rank
                current_pval = np.nan
            elif units_metrics_df is not None:
                # Fallback to unsorted dataframe
                unit_rank = st.slider(
                    "Unit Rank", 
                    0, len(units_metrics_df) - 1, 0,
                    help="Select unit by rank"
                )
                row = units_metrics_df.iloc[unit_rank]
                unit_idx = int(row["unit_idx"]) if "unit_idx" in row else unit_rank
                current_pval = np.nan
            else:
                unit_rank = 0
                unit_idx = 0
                current_pval = np.nan
        
        with unit_col2:
            st.metric("Selected Unit", unit_idx)
            st.metric("P-value", f"{current_pval:.3g}" if not np.isnan(current_pval) else "N/A")
        # Visualization Section
        st.subheader("Visualizations")
        
        if units_metrics_df is not None and row is not None:
            # Create two columns: PSTH plot (from saved HTML) and metrics table from units_metrics_df
            viz_col1, viz_col2 = st.columns(2)
            
            # --- Column 1: PSTH plot from saved HTML path ---
            with viz_col1:
                html_tone_path = None
                html_outcome_path = None
                html_choice_path = None
                # plot psth by tone
                df_to_check = units_metrics_df_sorted if units_metrics_df_sorted is not None else units_metrics_df
                if "psth_tone_path" in df_to_check.columns:
                    if pd.notna(row.get("psth_tone_path", None)):
                        html_tone_path = resolve_analysis_path(row["psth_tone_path"], analysis_output_dir)
                if html_tone_path and os.path.exists(html_tone_path):
                    try:
                        with open(html_tone_path, "r", encoding="utf-8") as f:
                            psth_render = f.read()
                        st.markdown(f"### Tone PSTH")
                        components.html(ensure_plotlyjs_inline(psth_render), height=500, scrolling=False)
                    except Exception as e:
                        st.warning(f"Error loading PSTH plot: {e}")
                else:
                    st.info("No saved tone PSTH plot found for this unit.")
                # plot psth by choice
                if "psth_choice_path" in df_to_check.columns:
                    if pd.notna(row.get("psth_choice_path", None)):
                        html_choice_path = resolve_analysis_path(row["psth_choice_path"], analysis_output_dir)
                if html_choice_path and os.path.exists(html_choice_path):
                    try:
                        with open(html_choice_path, "r", encoding="utf-8") as f:
                            psth_render = f.read()
                        st.markdown(f"### Choice PSTH")
                        components.html(ensure_plotlyjs_inline(psth_render), height=500, scrolling=False)
                    except Exception as e:
                        st.warning(f"Error loading PSTH plot: {e}")
                else:
                    st.info("No saved choice PSTH plot found for this unit.")
                # plot psth by outcome
                if "psth_outcome_path" in df_to_check.columns:
                    if pd.notna(row.get("psth_outcome_path", None)):
                        html_outcome_path = resolve_analysis_path(row["psth_outcome_path"], analysis_output_dir)
                if html_outcome_path and os.path.exists(html_outcome_path):
                    try:
                        with open(html_outcome_path, "r", encoding="utf-8") as f:
                            psth_render = f.read()
                        st.markdown(f"### Outcome PSTH")
                        components.html(ensure_plotlyjs_inline(psth_render), height=500, scrolling=False)
                    except Exception as e:
                        st.warning(f"Error loading PSTH plot: {e}")
                else:
                    st.info("No saved outcome PSTH plot found for this unit.")

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
                    if pd.notna(row.get("plot_path_heatmap", None)):
                        heatmap_html_path = resolve_analysis_path(row["plot_path_heatmap"], analysis_output_dir)
                if heatmap_html_path and os.path.exists(heatmap_html_path):
                    try:
                        with open(heatmap_html_path, "r", encoding="utf-8") as f:
                            heatmap_render = f.read()
                        components.html(ensure_plotlyjs_inline(heatmap_render), height=500, scrolling=False)
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
        st.header("Selectivity Analysis")
        
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
                    
                    # Heatmap of all tuning curves
                    st.subheader("Tuning Curves Heatmap")
                    if 'tuning_curve_stimuli' in selectivity_df.columns and 'tuning_curve' in selectivity_df.columns:
                        try:
                            normalize_heatmap = st.checkbox("Normalize per unit", value=True, key="normalize_heatmap")
                            heatmap_fig = plot_tuning_curves_heatmap(selectivity_df, use_log_scale=True, normalize_per_unit=normalize_heatmap)
                            if heatmap_fig is not None:
                                st.plotly_chart(heatmap_fig, use_container_width=True)
                            else:
                                st.info("No valid tuning curve data available for heatmap.")
                        except Exception as e:
                            st.warning(f"Could not create heatmap: {e}")
                    else:
                        st.info("Tuning curve data not available in selectivity metrics.")
                    
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
        
    with tab3:
        st.header("Generalized Linear Model Analysis")
        if not selected_recording_dir or not isinstance(selected_recording_dir, str):
            st.warning("No recording directory selected.")
        else:
            # Streamlit GLM panel (PopulationGLM + per-unit visualizations)
            from Analysis.NPXL_analysis.single_unit_offline_analysis.GLM.glm_streamlit_panel import (
                glm_analysis_panel,
            )

            glm_analysis_panel(
                base_path=selected_recording_dir,
                selected_area=selected_area or "",
            )
