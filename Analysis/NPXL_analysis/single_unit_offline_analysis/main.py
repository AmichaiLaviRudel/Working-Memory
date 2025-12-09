"""
Main execution script for NPXL offline analysis.

This script orchestrates the complete analysis pipeline:
1. Data loading
2. Active unit detection
3. Selectivity metrics computation
4. Visualization
5. Category sensitivity analysis
"""
# %%
import sys
import os

# Add the workspace root to Python path before any Analysis imports
current_dir = os.path.dirname(os.path.abspath(__file__))
# If we're in the single_unit_offline_analysis folder, go up 3 levels to reach workspace root
if 'single_unit_offline_analysis' in current_dir or 'NPXL_offline_analysis' in current_dir:
    workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
elif 'NPXL_analysis' in current_dir:
    workspace_root = os.path.dirname(os.path.dirname(current_dir))
else:
    # Fallback: try to find the workspace root by going up directories
    test_dir = current_dir
    for _ in range(4):  # Try going up at most 4 levels
        if os.path.exists(os.path.join(test_dir, 'Analysis', 'NPXL_analysis')):
            workspace_root = test_dir
            break
        test_dir = os.path.dirname(test_dir)
    else:
        # Last fallback: use current directory
        workspace_root = current_dir

if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from Analysis.NPXL_analysis.single_unit_offline_analysis.config import print_config
from Analysis.NPXL_analysis.single_unit_offline_analysis.data_loading import (
    load_data,
    load_unit_labels,
    read_event_windows_metadata,
    load_full_event_windows_data,
)
from Analysis.NPXL_analysis.single_unit_offline_analysis.active_units import (
    find_active_units_by_midpoint,
    find_action_modulated_units,
    find_outcome_modulated_units,
    plot_active_units_timecourses,
)
from Analysis.NPXL_analysis.single_unit_offline_analysis.utils import (
    setup_results_directory,
    save_dataframe_to_csv,
    save_plot_to_html,
    save_units_to_csv,
)
from Analysis.NPXL_analysis.single_unit_offline_analysis.selectivity import (
    compute_selectivity_metrics_for_active_units,
)
from Analysis.NPXL_analysis.single_unit_offline_analysis.visualization import (
    plot_psth_by_stimulus,
    plot_psth_by_outcome,
    plot_raw_psth,
    plot_unit_heatmap,
    save_raw_psth_for_active_units,
    plot_selectivity_summary,
)
from Analysis.NPXL_analysis.single_unit_offline_analysis.category_analysis import (
    compute_category_sensitivity_for_all_units,
    plot_psth_by_category,
    plot_category_sensitivity_summary,
)
from Analysis.NPXL_analysis.single_unit_offline_analysis.unit import (
    Unit,
    create_units_from_event_data,
)

# %%
def main(parent_dir: str = None, progress_fn=None, status_fn=None):
    """
    Main execution function.
    
    Parameters:
    -----------
    parent_dir : str, optional
        Path to the parent directory containing the data. If None, uses default path.
    progress_fn : callable, optional
        Callable taking (percent:int, message:str) to report progress.
    status_fn : callable, optional
        Callable taking (message:str) to emit status updates.
    """
    # %%
    def report(msg: str, status_fn=True):
        if status_fn:
            try:
                status_fn(msg)
            except Exception:
                pass
        print(msg)

    def set_progress(pct: int, msg: str = "", progress_fn=True):
        if progress_fn:
            try:
                progress_fn(pct, msg)
            except Exception:
                pass
        if msg:
            report(msg)

    def save_raw_psth_with_path(
        event_windows_data: tuple,
        unit_idx: int,
        region_label: str,
        version_label: str,
        base_results_dir: str,
        display_window: tuple[float, float],
    ) -> str:
        """Save a raw PSTH and return its file path."""
        fig_raw = plot_raw_psth(
            event_windows_data,
            int(unit_idx),
            display_window=display_window,
            region_name=region_label,
        )
        subdir = os.path.join(base_results_dir, "plots", "raw_psth", region_label.lower())
        os.makedirs(subdir, exist_ok=True)
        fname = f"unit_{unit_idx}_{version_label}_raw_psth.html"
        fpath = os.path.join(subdir, fname)
        save_plot_to_html(fig_raw, fpath, f"{region_label} Unit {unit_idx} Raw PSTH ({version_label})")
        return fpath

    def compute_event_offsets(df: pd.DataFrame, column: str, bin_size: float) -> np.ndarray | None:
        """
        Return per-trial offsets (seconds) from tone onset to the target column.
        """
        if df is None or column not in df.columns or "time" not in df.columns:
            return None
        tone_bins = df["time"].to_numpy(dtype=float)
        event_bins = df[column].to_numpy(dtype=float)
        offsets_bins = event_bins - tone_bins
        if offsets_bins.size == 0 or np.all(np.isnan(offsets_bins)):
            return None
        return offsets_bins * bin_size
    # Print configuration
    print_config()
    set_progress(5, "Configuration loaded")
    # %%
    # ============================================================================
    # Load and explore data
    # ============================================================================
    if parent_dir is None:
        # %%
        # Use a representative default recording so the script runs without args
        parent_dir = r"Z:\Shared\Amichai\NPXL\Recs\group7\catgt_G7A1_1BExpert _2B_3T_g0"
    # %%
    OFC_all, ACx_all, data_dir_OFC, data_dir_ACx = load_data(
        data_dir_parent=parent_dir, data_dir_OFC=None, data_dir_ACx=None
    )  # Returns both matrices plus resolved region-specific paths
    set_progress(15, "Data loaded")
    
    # Load unit labels
    ofc_g_index, acx_g_index, OFC_g, ACx_g = load_unit_labels(data_dir_OFC, data_dir_ACx)
    
    # Read metadata
    ofc_metadata = read_event_windows_metadata(data_dir_OFC)
    acx_metadata = read_event_windows_metadata(data_dir_ACx)
    acx_bin_size_sec = float(acx_metadata["bin_size"])  # seconds per bin from preprocessing
    ofc_bin_size_sec = float(ofc_metadata["bin_size"])
    bin_to_sec = 1 / acx_bin_size_sec  # Convert bin counts to seconds for window math
    set_progress(20, "Metadata loaded")
    # %%
    # Find active units by midpoint of ACx
    active_units_acx, t_vals_acx, p_vals_acx = find_active_units_by_midpoint(
        ACx_all,
        alpha=0.05,
        before_range=(int(-0.1 * bin_to_sec), 0),  # 100 ms pre-tone baseline
        after_range=(0, int(0.5 * bin_to_sec)))  # 500 ms post-tone response
    set_progress(30, f"Found {len(active_units_acx)} active ACx units")

    # %%
    # Load full event windows data with stimuli/outcome information
    acx_event_windows_data = load_full_event_windows_data(data_dir_ACx)  # matrix, time axis, indices, trial table, meta, licks
    ofc_event_windows_data = load_full_event_windows_data(data_dir_OFC)
    
    # Extract components
    (acx_event_matrix, acx_time_axis, acx_valid_indices, acx_stimuli_outcome_df, 
     acx_metadata_full, acx_lick_data) = acx_event_windows_data
    (ofc_event_matrix, ofc_time_axis, ofc_valid_indices, ofc_stimuli_outcome_df, 
     ofc_metadata_full, ofc_lick_data) = ofc_event_windows_data
    acx_all_unit_indices = np.arange(acx_event_matrix.shape[0])
    
    print(f"\nLoaded event windows data:")
    print(f"  ACx: {acx_event_matrix.shape} units × time × events")
    print(f"  OFC: {ofc_event_matrix.shape} units × time × events")
# %%
    print(f"\nACx stimuli/outcome columns: {list(acx_stimuli_outcome_df.columns)}")
    print(f"ACx unique outcomes: {acx_stimuli_outcome_df['outcome'].unique() if 'outcome' in acx_stimuli_outcome_df.columns else 'N/A'}")
    if 'stimulus' in acx_stimuli_outcome_df.columns:
        print(f"ACx unique stimuli: {sorted(acx_stimuli_outcome_df['stimulus'].unique())}")
# %%
    
    # Lick/outcome offsets relative to tone onset (tone is at window midpoint)
    lick_offsets_acx = compute_event_offsets(acx_stimuli_outcome_df, "first_lick_time", acx_bin_size_sec)
    outcome_offsets_acx = compute_event_offsets(acx_stimuli_outcome_df, "outcome_time", acx_bin_size_sec)
# %%
    # Baseline before window spans from window start to tone onset
    acx_before_window = (-0.2, 0.0)  # 200 ms baseline before lick/outcome alignment
    acx_after_window = (0.0, 0.2)  # 600 ms response window after alignment

    choice_units_acx, choice_t_acx, choice_p_acx, acx_choice_aligned = find_action_modulated_units(
        acx_event_windows_data,
        lick_offsets_acx,
        bin_size_sec=acx_bin_size_sec,
        alpha=0.15,
        before_window=acx_before_window,
        after_window=acx_after_window,
    )
    outcome_units_acx, outcome_t_acx, outcome_p_acx, acx_outcome_aligned = find_outcome_modulated_units(
        acx_event_windows_data,
        outcome_offsets_acx,
        bin_size_sec=acx_bin_size_sec,
        alpha=0.15,
        before_window=acx_before_window,
        after_window=acx_after_window,
    )

    print(f"Found {len(choice_units_acx)} choice-modulated ACx units (lick aligned)")
    print(f"Found {len(outcome_units_acx)} outcome-modulated ACx units (reward/punish aligned)")
    # %%
    # Setup results directory
    analysis_output_dir = os.path.join(parent_dir, "analysis_output")
    results_dir = setup_results_directory(analysis_output_dir, subfolder="")  # Creates tables/plots layout if missing
    
    # ============================================================================
    # Compute selectivity metrics for ACx active units
    # ============================================================================

    # Compute selectivity for all units (for complete matrix)
    acx_selectivity_all_df = compute_selectivity_metrics_for_active_units(
        acx_event_windows_data,
        acx_all_unit_indices,
        window=(-0.1, 1.5),
        region_name="ACx",
        use_unit_class=True,
        aligned_action_data=acx_choice_aligned,
        aligned_outcome_data=acx_outcome_aligned,
    )
    
    # Alternative: Create Unit objects for interactive analysis
    # acx_units = create_units_from_event_data(
    #     acx_event_windows_data,
    #     active_units_acx,
    #     region_name="ACx",
    #     unit_labels=acx_g_index,
    # )
    # # Then you can do: unit = acx_units[0]; unit.compute_selectivity(); unit.plot_psth_by_stimulus()
    
    print(f"\nSelectivity metrics for {len(acx_selectivity_all_df)} ACx active units:")
    if len(acx_selectivity_all_df) > 0 and 'stimulus_selective' in acx_selectivity_all_df.columns:
        print(f"  Stimulus selective: {acx_selectivity_all_df['stimulus_selective'].sum()}")
        print(f"  Outcome modulated (p<0.05): {acx_selectivity_all_df['outcome_modulated'].sum()}")
        print(f"  Go/NoGo selective (|d'|>0.5): {acx_selectivity_all_df['go_nogo_selective'].sum()}")
        print(f"  Choice coding (|CP_corr|>0.1): {acx_selectivity_all_df['choice_coding'].sum()}")
        print(f"\nSummary statistics:")
        print(acx_selectivity_all_df.describe())
    else:
        print("  No active units found - skipping selectivity metrics")
    
    # Save ACx selectivity results
    print("\n=== Saving ACx selectivity results ===")
    save_dataframe_to_csv(
        acx_selectivity_all_df,
        os.path.join(results_dir, "tables", "acx_selectivity_metrics.csv"),
        "ACx selectivity metrics table"
    )
    set_progress(40, "ACx selectivity saved")
    
    # ============================================================================
    # Create Unit objects and save comprehensive metrics table
    # ============================================================================
    print("\n=== Creating Unit objects and computing all metrics ===")
    acx_all_units = create_units_from_event_data(
        acx_event_windows_data,
        acx_all_unit_indices,
        region_name="ACx",
        unit_labels=acx_g_index,
        aligned_action_data=acx_choice_aligned,
        aligned_outcome_data=acx_outcome_aligned,
    )
    acx_unit_map = {unit.unit_idx: unit for unit in acx_all_units}  # Reuse same Unit instances so plot paths persist
    acx_units = [
        acx_unit_map[int(idx)]
        for idx in active_units_acx
        if int(idx) in acx_unit_map
    ]
    
    # Set plots directory for units
    plots_dir = os.path.join(results_dir, "plots")
    for unit in acx_units:
        unit.set_plots_directory(plots_dir)  # Keep plots under the run-specific results folder
    
    # Generate and save plots for all units (heatmap and PSTHs)
    print("\n=== Generating plots for all ACx units ===")
    for i, unit in enumerate(acx_units):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Processing unit {i+1}/{len(acx_units)}: Unit {unit.unit_idx}")
        # Generate and save heatmap
        unit.plot_heatmap(display_window=(-0.5, 2.0), cache_plot=True)
        
        # Generate and save PSTH plots
        # unit.plot_psth_by_stimulus(display_window=(-0.5, 2.0), cache_plot=True)
        # unit.plot_psth_by_outcome(display_window=(-0.5, 2.0), cache_plot=True)
    
    # Save comprehensive metrics table for all ACx units
    print("\n=== Saving comprehensive ACx unit metrics table ===")
    acx_units_df = save_units_to_csv(
        acx_all_units,
        os.path.join(results_dir, "tables", "acx_all_units_metrics.csv"),
        compute_all_metrics=True,
        selectivity_window=(-0.1, 0.2),  # Early+mid response window for selectivity
        category_window=(-0.1, 0.5),  # Narrower window for category ANOVA
        category_boundaries=(0.983, 1.525),  # Empirical Go/NoGo split points
        psth_baseline_window=(-0.2, 0),  # Pre-tone baseline for PSTH normalization
        description="ACx comprehensive unit metrics (all units)"
    )
    # Attach active-unit stats (midpoint t/p values) to the saved table
    acx_pval_map = {int(u): float(p_vals_acx[i]) for i, u in enumerate(active_units_acx)}
    acx_tval_map = {int(u): float(t_vals_acx[i]) for i, u in enumerate(active_units_acx)}
    choice_pval_map = {int(u): float(choice_p_acx[i]) for i, u in enumerate(choice_units_acx)}
    outcome_pval_map = {int(u): float(outcome_p_acx[i]) for i, u in enumerate(outcome_units_acx)}
    acx_units_df["tone_active_p_val"] = acx_units_df["unit_idx"].map(acx_pval_map)
    acx_units_df["tone_active_t_val"] = acx_units_df["unit_idx"].map(acx_tval_map)
    acx_units_df["choice_p_val"] = acx_units_df["unit_idx"].map(choice_pval_map)
    acx_units_df["outcome_p_val"] = acx_units_df["unit_idx"].map(outcome_pval_map)
    acx_units_df["tone_active"] = acx_units_df["unit_idx"].isin(set(int(u) for u in active_units_acx))
    save_dataframe_to_csv(
        acx_units_df,
        os.path.join(results_dir, "tables", "acx_all_units_metrics.csv"),
        "ACx comprehensive unit metrics (all units)",
    )
    set_progress(60, "ACx unit metrics saved")
    print(f"  Saved metrics for {len(acx_units)} ACx units")
    
    # ============================================================================
    # Plot and save raw PSTH for all responsive units
    # ============================================================================
    # Determine significant units (stimulus, category, choice, or outcome)
    acx_category_all_df = compute_category_sensitivity_for_all_units(
        acx_event_windows_data,
        acx_all_unit_indices,
        low_boundary=0.983,
        high_boundary=1.525,  # Empirical Go/NoGo split points in odor ratio space
        window=(-0.1, 1),
    )
    acx_sig_df = acx_selectivity_all_df.merge(
        acx_category_all_df[["unit_idx", "category_sensitive"]],
        on="unit_idx",
        how="left",
    )  # Merge category flags so any significance type can mark a unit as "interesting"
    acx_sig_mask = (
        acx_sig_df["stimulus_selective"].fillna(False)
        | acx_sig_df["category_sensitive"].fillna(False)
        | acx_sig_df["choice_coding"].fillna(False)
        | acx_sig_df["outcome_modulated"].fillna(False)
        | acx_sig_df["unit_idx"].isin(active_units_acx)  # Relax criterion: keep all active units
    )
    acx_sig_units = acx_sig_df.loc[acx_sig_mask, "unit_idx"].astype(int).to_numpy()  # Collapse to ndarray for indexing

    # Save PSTHs (tone/choice/outcome) for significant ACx units and store paths
    acx_psth_paths_tone = {}
    acx_psth_paths_choice = {}
    acx_psth_paths_outcome = {}
    acx_psth_paths_category = {}
    acx_heatmap_paths_choice = {}
    acx_heatmap_paths_outcome = {}

    # # Save tone-aligned PSTH for significant ACx units
    # if len(acx_sig_units) > 0:
    #     save_raw_psth_for_active_units(
    #         acx_event_windows_data,
    #         acx_sig_units,
    #         np.full(len(acx_sig_units), 1.0),
    #         "ACx_Tone",
    #         results_dir,
    #         display_window=(-0.5, 1.0),
    #     )
    #     for u in acx_sig_units:
    #         acx_psth_paths_tone[int(u)] = save_raw_psth_with_path(
    #             acx_event_windows_data,
    #             int(u),
    #             "ACx",
    #             "tone",
    #             results_dir,
    #             display_window=(-0.5, 1.0),
    #         )

    # Choice- and outcome-aligned PSTH (lick/outcome) for significant or modulated units
    if acx_choice_aligned is not None and len(acx_sig_mask) > 0:
        choice_sig_units = np.union1d(acx_sig_units, choice_units_acx)  # OR: keep any significant or choice-modulated unit
        if len(choice_sig_units) > 0:
            save_raw_psth_for_active_units(
                acx_choice_aligned,
                choice_sig_units,
                np.full(len(choice_sig_units), 1.0),
                "ACx_Choice",
                results_dir,
                display_window=(-0.6, 2),
            )
            for u in choice_sig_units:
                acx_psth_paths_choice[int(u)] = save_raw_psth_with_path(
                    acx_choice_aligned,
                    int(u),
                    "ACx",
                    "choice",
                    results_dir,
                    display_window=(-0.6, 2),
                )
                # Generate and save choice-aligned heatmap
                heatmap_choice_path = os.path.join(results_dir, "plots", "heatmap", "choice_aligned", f"acx_unit_{u}_choice_heatmap.html")
                fig_heatmap_choice = plot_unit_heatmap(
                    acx_choice_aligned,
                    int(u),
                    display_window=(-0.6, 2),
                    region_name="ACx"
                )
                save_plot_to_html(
                    fig_heatmap_choice,
                    heatmap_choice_path,
                    f"ACx Unit {u} Heatmap (Choice-Aligned)"
                )
                acx_heatmap_paths_choice[int(u)] = heatmap_choice_path
    if acx_outcome_aligned is not None and len(outcome_units_acx) > 0:
        outcome_sig_units = np.union1d(acx_sig_units, outcome_units_acx)  # OR: keep any significant or outcome-modulated unit
        if len(outcome_sig_units) > 0:
            save_raw_psth_for_active_units(
                acx_outcome_aligned,
                outcome_sig_units,
                np.full(len(outcome_sig_units), 1.0),
                "ACx_Outcome",
                results_dir,
                display_window=(-0.6, 1.2),
            )
            for u in outcome_sig_units:
                acx_psth_paths_outcome[int(u)] = save_raw_psth_with_path(
                    acx_outcome_aligned,
                    int(u),
                    "ACx",
                    "outcome",
                    results_dir,
                    display_window=(-0.6, 1.2),
                )
                # Generate and save outcome-aligned heatmap
                heatmap_outcome_path = os.path.join(results_dir, "plots", "heatmap", "outcome_aligned", f"acx_unit_{u}_outcome_heatmap.html")
                fig_heatmap_outcome = plot_unit_heatmap(
                    acx_outcome_aligned,
                    int(u),
                    display_window=(-0.6, 1.2),
                    region_name="ACx"
                )
                save_plot_to_html(
                    fig_heatmap_outcome,
                    heatmap_outcome_path,
                    f"ACx Unit {u} Heatmap (Outcome-Aligned)"
                )
                acx_heatmap_paths_outcome[int(u)] = heatmap_outcome_path

    # Plot PSTH by category for category-sensitive ACx units (part of significant units)
    if len(acx_sig_units) > 0 and acx_category_all_df is not None:
        category_sensitive_acx = acx_category_all_df[
            (acx_category_all_df['category_sensitive']) & 
            (acx_category_all_df['unit_idx'].isin(acx_sig_units))
        ]
        if len(category_sensitive_acx) > 0:
            print(f"\n  Plotting PSTH by category for {len(category_sensitive_acx)} category-sensitive ACx units:")
            for _, row in category_sensitive_acx.iterrows():
                unit_idx = int(row['unit_idx'])
                p_val = row['category_anova_p']
                print(f"    Unit {unit_idx} (p={p_val:.4f})")
                
                fig_cat = plot_psth_by_category(
                    acx_event_windows_data,
                    unit_idx,
                    low_boundary=0.983,
                    high_boundary=1.525,
                    display_window=(-0.5, 1.0),
                    region_name="ACx"
                )
                category_psth_path = os.path.join(results_dir, "plots", "psth_by_category", f"acx_unit_{unit_idx}_psth_by_category.html")
                save_plot_to_html(
                    fig_cat,
                    category_psth_path,
                    f"ACx Unit {unit_idx} PSTH by Category"
                )
                acx_psth_paths_category[unit_idx] = category_psth_path

    # Attach PSTH and heatmap paths to ACx metrics table
    acx_units_df["psth_tone_path"] = acx_units_df["unit_idx"].map(acx_psth_paths_tone)
    acx_units_df["psth_choice_path"] = acx_units_df["unit_idx"].map(acx_psth_paths_choice)
    acx_units_df["psth_outcome_path"] = acx_units_df["unit_idx"].map(acx_psth_paths_outcome)
    acx_units_df["psth_category_path"] = acx_units_df["unit_idx"].map(acx_psth_paths_category)
    acx_units_df["heatmap_choice_path"] = acx_units_df["unit_idx"].map(acx_heatmap_paths_choice)
    acx_units_df["heatmap_outcome_path"] = acx_units_df["unit_idx"].map(acx_heatmap_paths_outcome)
    save_dataframe_to_csv(
        acx_units_df,
        os.path.join(results_dir, "tables", "acx_all_units_metrics.csv"),
        "ACx comprehensive unit metrics (all units, with PSTH paths)",
    )
    
    # ============================================================================
    # Create selectivity summary plots
    # ============================================================================
    # acx_fig_metrics, acx_fig_class = plot_selectivity_summary(acx_selectivity_all_df, region_name="ACx")
    
    # # Save ACx summary plots
    # print("\n=== Saving ACx summary plots ===")
    # save_plot_to_html(
    #     acx_fig_metrics,
    #     os.path.join(results_dir, "plots", "acx", "acx_selectivity_metrics_summary.html"),
    #     "ACx selectivity metrics summary"
    # )
    # save_plot_to_html(
    #     acx_fig_class,
    #     os.path.join(results_dir, "plots", "acx", "acx_unit_classification_summary.html"),
    #     "ACx unit classification summary"
    # )
    
    # Display detailed summary table
    print("\n=== ACx Selectivity Summary Table ===")
    if len(acx_selectivity_all_df) > 0 and 'stimulus_selective' in acx_selectivity_all_df.columns:
        print(acx_selectivity_all_df[['unit_idx', 'stimulus_selective', 'outcome_modulated', 
                                   'go_nogo_selective', 'choice_coding']].to_string(index=False))
    else:
        print("No active units found - no summary table to display")
    
    # ============================================================================
    # Compare OFC vs ACx
    # ============================================================================
    # Find active units in OFC
    active_units_ofc, t_vals_ofc, p_vals_ofc = find_active_units_by_midpoint(
        OFC_all,
        alpha=0.15,
        before_range=(int(-0.5*bin_to_sec), 0),  # Wider baseline to match longer OFC window
        after_range=(0, int(2 * bin_to_sec)),  # Capture delayed OFC responses
    )

    # OFC action/outcome modulation (aligned to lick/outcome times)
    lick_offsets_ofc = compute_event_offsets(ofc_stimuli_outcome_df, "first_lick_time", ofc_bin_size_sec)
    outcome_offsets_ofc = compute_event_offsets(ofc_stimuli_outcome_df, "outcome_time", ofc_bin_size_sec)
    ofc_before_window = (-0.2, 0.0)  # Match ACx lick/outcome baseline window
    ofc_after_window = (0.0, 1.5)  # Post-alignment response window

    choice_units_ofc, choice_t_ofc, choice_p_ofc, ofc_choice_aligned = find_action_modulated_units(
        ofc_event_windows_data,
        lick_offsets_ofc,
        bin_size_sec=ofc_bin_size_sec,
        alpha=0.15,
        before_window=ofc_before_window,
        after_window=ofc_after_window,
    )
    outcome_units_ofc, outcome_t_ofc, outcome_p_ofc, ofc_outcome_aligned = find_outcome_modulated_units(
        ofc_event_windows_data,
        outcome_offsets_ofc,
        bin_size_sec=ofc_bin_size_sec,
        alpha=0.15,
        before_window=ofc_before_window,
        after_window=ofc_after_window,
    )
    
    print(f"\n=== OFC Active Units ===")
    print(f"Found {len(active_units_ofc)} active OFC units")
    
    # Compute selectivity metrics (active + all) for OFC
    ofc_selectivity_df = compute_selectivity_metrics_for_active_units(
        ofc_event_windows_data,
        active_units_ofc,
        window=(-0.1, 1.5),
        region_name="OFC",
        use_unit_class=True,
        aligned_action_data=ofc_choice_aligned,
        aligned_outcome_data=ofc_outcome_aligned,
    )
    ofc_selectivity_all_df = compute_selectivity_metrics_for_active_units(
        ofc_event_windows_data,
        np.arange(ofc_event_matrix.shape[0]),
        window=(-0.1, 1.5),
        region_name="OFC",
        use_unit_class=True,
        aligned_action_data=ofc_choice_aligned,
        aligned_outcome_data=ofc_outcome_aligned,
    )

    # Plot and save raw PSTH for all OFC responsive units
    # Save tone-aligned PSTH for all OFC units
    # Determine significant units (stimulus, category, choice, or outcome) for OFC
    ofc_category_all_df = compute_category_sensitivity_for_all_units(
        ofc_event_windows_data,
        np.arange(ofc_event_matrix.shape[0]),
        low_boundary=0.983,
        high_boundary=1.525,
        window=(-0.1, 1.5),
    )
    ofc_sig_df = ofc_selectivity_all_df.merge(
        ofc_category_all_df[["unit_idx", "category_sensitive"]],
        on="unit_idx",
        how="left",
    )
    ofc_sig_mask = (
        ofc_sig_df["stimulus_selective"].fillna(False)
        | ofc_sig_df["category_sensitive"].fillna(False)
        | ofc_sig_df["choice_coding"].fillna(False)
        | ofc_sig_df["outcome_modulated"].fillna(False)
        | ofc_sig_df["unit_idx"].isin(active_units_ofc)  # Relax criterion: keep all active units
    )
    ofc_sig_units = ofc_sig_df.loc[ofc_sig_mask, "unit_idx"].astype(int).to_numpy()

    # Save PSTHs (tone/choice/outcome) for significant OFC units and store paths
    ofc_psth_paths_tone = {}
    ofc_psth_paths_choice = {}
    ofc_psth_paths_outcome = {}
    ofc_psth_paths_category = {}
    ofc_heatmap_paths_choice = {}
    ofc_heatmap_paths_outcome = {}

    # if len(ofc_sig_units) > 0:
    #     # Tone-aligned
    #     for u in ofc_sig_units:
    #         ofc_psth_paths_tone[int(u)] = save_raw_psth_with_path(
    #             ofc_event_windows_data,
    #             int(u),
    #             "OFC",
    #             "tone",
    #             results_dir,
    #             display_window=(-0.5, 2),
    #         )

    # Choice- and outcome-aligned PSTH for significant or modulated OFC units
    if ofc_choice_aligned is not None and len(choice_units_ofc) > 0:
        choice_sig_units_ofc = np.union1d(ofc_sig_units, choice_units_ofc)  # OR: keep any significant or choice-modulated unit
        for u in choice_sig_units_ofc:
            ofc_psth_paths_choice[int(u)] = save_raw_psth_with_path(
                ofc_choice_aligned,
                int(u),
                "OFC",
                "choice",
                results_dir,
                display_window=(-0.6, 2),
            )
            # Generate and save choice-aligned heatmap
            heatmap_choice_path_ofc = os.path.join(results_dir, "plots", "heatmap", "choice_aligned", f"ofc_unit_{u}_choice_heatmap.html")
            fig_heatmap_choice_ofc = plot_unit_heatmap(
                ofc_choice_aligned,
                int(u),
                display_window=(-0.6, 2),
                region_name="OFC"
            )
            save_plot_to_html(
                fig_heatmap_choice_ofc,
                heatmap_choice_path_ofc,
                f"OFC Unit {u} Heatmap (Choice-Aligned)"
            )
            ofc_heatmap_paths_choice[int(u)] = heatmap_choice_path_ofc

    if ofc_outcome_aligned is not None and len(outcome_units_ofc) > 0:
        outcome_sig_units_ofc = np.union1d(ofc_sig_units, outcome_units_ofc)  # OR: keep any significant or outcome-modulated unit
        for u in outcome_sig_units_ofc:
            ofc_psth_paths_outcome[int(u)] = save_raw_psth_with_path(
                ofc_outcome_aligned,
                int(u),
                "OFC",
                "outcome",
                results_dir,
                display_window=(-0.6, 2),
            )
            # Generate and save outcome-aligned heatmap
            heatmap_outcome_path_ofc = os.path.join(results_dir, "plots", "heatmap", "outcome_aligned", f"ofc_unit_{u}_outcome_heatmap.html")
            fig_heatmap_outcome_ofc = plot_unit_heatmap(
                ofc_outcome_aligned,
                int(u),
                display_window=(-0.6, 2),
                region_name="OFC"
            )
            save_plot_to_html(
                fig_heatmap_outcome_ofc,
                heatmap_outcome_path_ofc,
                f"OFC Unit {u} Heatmap (Outcome-Aligned)"
            )
            ofc_heatmap_paths_outcome[int(u)] = heatmap_outcome_path_ofc

    # Plot PSTH by category for category-sensitive OFC units (part of significant units)
    if len(ofc_sig_units) > 0 and ofc_category_all_df is not None:
        category_sensitive_ofc = ofc_category_all_df[
            (ofc_category_all_df['category_sensitive']) & 
            (ofc_category_all_df['unit_idx'].isin(ofc_sig_units))
        ]
        if len(category_sensitive_ofc) > 0:
            print(f"\n  Plotting PSTH by category for {len(category_sensitive_ofc)} category-sensitive OFC units:")
            for _, row in category_sensitive_ofc.iterrows():
                unit_idx = int(row['unit_idx'])
                p_val = row['category_anova_p']
                print(f"    Unit {unit_idx} (p={p_val:.4f})")
                
                fig_cat_ofc = plot_psth_by_category(
                    ofc_event_windows_data,
                    unit_idx,
                    low_boundary=0.983,
                    high_boundary=1.525,
                    display_window=(-0.5, 1.0),
                    region_name="OFC"
                )
                category_psth_path_ofc = os.path.join(results_dir, "plots", "psth_by_category", f"ofc_unit_{unit_idx}_psth_by_category.html")
                save_plot_to_html(
                    fig_cat_ofc,
                    category_psth_path_ofc,
                    f"OFC Unit {unit_idx} PSTH by Category"
                )
                ofc_psth_paths_category[unit_idx] = category_psth_path_ofc

    print(f"\nSelectivity metrics for {len(ofc_selectivity_df)} OFC active units:")
    if len(ofc_selectivity_df) > 0 and 'stimulus_selective' in ofc_selectivity_df.columns:
        print(f"  Stimulus selective: {ofc_selectivity_df['stimulus_selective'].sum()}")
        print(f"  Outcome modulated (p<0.05): {ofc_selectivity_df['outcome_modulated'].sum()}")
        print(f"  Go/NoGo selective (|d'|>0.5): {ofc_selectivity_df['go_nogo_selective'].sum()}")
        print(f"  Choice coding (|CP_corr|>0.1): {ofc_selectivity_df['choice_coding'].sum()}")
    else:
        print("  No active units found - skipping selectivity metrics")
    
    # Save OFC selectivity results
    print("\n=== Saving OFC selectivity results ===")
    save_dataframe_to_csv(
        ofc_selectivity_df,
        os.path.join(results_dir, "tables", "ofc_selectivity_metrics.csv"),
        "OFC selectivity metrics table"
    )
    set_progress(70, "OFC selectivity saved")
    
    # ============================================================================
    # Create Unit objects and save comprehensive metrics table for OFC
    # ============================================================================
    print("\n=== Creating Unit objects and computing all metrics for OFC ===")
    ofc_all_units = create_units_from_event_data(
        ofc_event_windows_data,
        np.arange(ofc_event_matrix.shape[0]),
        region_name="OFC",
        unit_labels=ofc_g_index,
        aligned_action_data=ofc_choice_aligned,
        aligned_outcome_data=ofc_outcome_aligned,
    )
    ofc_unit_map = {unit.unit_idx: unit for unit in ofc_all_units}  # Keep single instance per unit to retain plot paths
    ofc_units = [
        ofc_unit_map[int(idx)]
        for idx in active_units_ofc
        if int(idx) in ofc_unit_map
    ]
    
    # Set plots directory for units
    for unit in ofc_units:
        unit.set_plots_directory(plots_dir)  # Reuse same plot root as ACx for consistency
    
    # Generate and save plots for all units (heatmap and PSTHs)
    print("\n=== Generating plots for all OFC units ===")
    for i, unit in enumerate(ofc_units):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Processing unit {i+1}/{len(ofc_units)}: Unit {unit.unit_idx}")
        # Generate and save heatmap
        unit.plot_heatmap(display_window=(-0.5, 2.0), cache_plot=True)
        # Generate and save PSTH plots
        # unit.plot_psth_by_stimulus(display_window=(-0.5, 2.0), cache_plot=True)
        # unit.plot_psth_by_outcome(display_window=(-0.5, 2.0), cache_plot=True)
    
    # Save comprehensive metrics table for all OFC units
    print("\n=== Saving comprehensive OFC unit metrics table ===")
    ofc_units_df = save_units_to_csv(
        ofc_all_units,
        os.path.join(results_dir, "tables", "ofc_all_units_metrics.csv"),
        compute_all_metrics=True,
        selectivity_window=(-0.1, 1.5),  # Capture early + sustained tone response
        category_window=(-0.1, 1.5),  # Align with selectivity window for category tests
        category_boundaries=(0.983, 1.525),  # Same odor ratio split as ACx for consistency
        psth_baseline_window=(-0.5, 0),  # Baseline relative to tone onset
        description="OFC comprehensive unit metrics (all units)"
    )
    ofc_pval_map = {int(u): float(p_vals_ofc[i]) for i, u in enumerate(active_units_ofc)}
    ofc_tval_map = {int(u): float(t_vals_ofc[i]) for i, u in enumerate(active_units_ofc)}
    choice_pval_map_ofc = {int(u): float(choice_p_ofc[i]) for i, u in enumerate(choice_units_ofc)}
    outcome_pval_map_ofc = {int(u): float(outcome_p_ofc[i]) for i, u in enumerate(outcome_units_ofc)}
    ofc_units_df["tone_active_p_val"] = ofc_units_df["unit_idx"].map(ofc_pval_map)
    ofc_units_df["tone_active_t_val"] = ofc_units_df["unit_idx"].map(ofc_tval_map)
    ofc_units_df["choice_p_val"] = ofc_units_df["unit_idx"].map(choice_pval_map_ofc)
    ofc_units_df["outcome_p_val"] = ofc_units_df["unit_idx"].map(outcome_pval_map_ofc)
    ofc_units_df["tone_active"] = ofc_units_df["unit_idx"].isin(set(int(u) for u in active_units_ofc))
    ofc_units_df["psth_tone_path"] = ofc_units_df["unit_idx"].map(ofc_psth_paths_tone)
    ofc_units_df["psth_choice_path"] = ofc_units_df["unit_idx"].map(ofc_psth_paths_choice)
    ofc_units_df["psth_outcome_path"] = ofc_units_df["unit_idx"].map(ofc_psth_paths_outcome)
    ofc_units_df["psth_category_path"] = ofc_units_df["unit_idx"].map(ofc_psth_paths_category)
    ofc_units_df["heatmap_choice_path"] = ofc_units_df["unit_idx"].map(ofc_heatmap_paths_choice)
    ofc_units_df["heatmap_outcome_path"] = ofc_units_df["unit_idx"].map(ofc_heatmap_paths_outcome)
    save_dataframe_to_csv(
        ofc_units_df,
        os.path.join(results_dir, "tables", "ofc_all_units_metrics.csv"),
        "OFC comprehensive unit metrics (all units)",
    )
    set_progress(85, "OFC unit metrics saved")
    print(f"Saved metrics for {len(ofc_units)} OFC units")
    
    # # Create OFC summary plots
    # ofc_fig_metrics, ofc_fig_class = plot_selectivity_summary(ofc_selectivity_df, region_name="OFC")
    
    # # Save OFC summary plots
    # print("\n=== Saving OFC summary plots ===")
    # save_plot_to_html(
    #     ofc_fig_metrics,
    #     os.path.join(results_dir, "plots", "ofc", "ofc_selectivity_metrics_summary.html"),
    #     "OFC selectivity metrics summary"
    # )
    # save_plot_to_html(
    #     ofc_fig_class,
    #     os.path.join(results_dir, "plots", "ofc", "ofc_unit_classification_summary.html"),
    #     "OFC unit classification summary"
    # )
    
    # # Comparison plot: OFC vs ACx
    # has_acx_data = len(acx_selectivity_all_df) > 0 and 'stimulus_selective' in acx_selectivity_all_df.columns
    # has_ofc_data = len(ofc_selectivity_df) > 0 and 'stimulus_selective' in ofc_selectivity_df.columns
    
    # if has_acx_data or has_ofc_data:
    #     fig_comparison = go.Figure()
        
    #     metrics = ['Stimulus Selective', 'Outcome Modulated', 'Go/NoGo Selective', 'Choice Coding']  # Keep label order stable across regions
        
    #     if has_acx_data:
    #         acx_counts = [
    #             acx_selectivity_all_df['stimulus_selective'].sum(),
    #             acx_selectivity_all_df['outcome_modulated'].sum(),
    #             acx_selectivity_all_df['go_nogo_selective'].sum(),
    #             acx_selectivity_all_df['choice_coding'].sum(),
    #         ]
    #         acx_proportions = [c / len(acx_selectivity_all_df) * 100 for c in acx_counts]
    #     else:
    #         acx_counts = [0, 0, 0, 0]
    #         acx_proportions = [0, 0, 0, 0]
        
    #     if has_ofc_data:
    #         ofc_counts = [
    #             ofc_selectivity_df['stimulus_selective'].sum(),
    #             ofc_selectivity_df['outcome_modulated'].sum(),
    #             ofc_selectivity_df['go_nogo_selective'].sum(),
    #             ofc_selectivity_df['choice_coding'].sum(),
    #         ]
    #         ofc_proportions = [c / len(ofc_selectivity_df) * 100 for c in ofc_counts]
    #     else:
    #         ofc_counts = [0, 0, 0, 0]
    #         ofc_proportions = [0, 0, 0, 0]
        
    #     fig_comparison.add_trace(go.Bar(
    #         x=metrics,
    #         y=acx_proportions,
    #         name='ACx',
    #         marker_color='#1f77b4',
    #         text=[f'{c} ({p:.1f}%)' for c, p in zip(acx_counts, acx_proportions)],
    #         textposition='outside',
    #         hovertemplate='ACx<br>%{x}<br>%{y:.1f}% (%{text})<extra></extra>'
    #     ))
        
    #     fig_comparison.add_trace(go.Bar(
    #         x=metrics,
    #         y=ofc_proportions,
    #         name='OFC',
    #         marker_color='#ff7f0e',
    #         text=[f'{c} ({p:.1f}%)' for c, p in zip(ofc_counts, ofc_proportions)],
    #         textposition='outside',
    #         hovertemplate='OFC<br>%{x}<br>%{y:.1f}% (%{text})<extra></extra>'
    #     ))
        
    #     fig_comparison.update_layout(
    #         title='OFC vs ACx - Selectivity Comparison (Proportion of Active Units)',
    #         xaxis_title='Selectivity Type',
    #         yaxis_title='Percentage of Active Units',
    #         barmode='group',
    #         template='plotly_white',
    #         height=500,
    #         legend=dict(x=0.7, y=0.95)
    #     )
        
    #     # Save comparison plot
    #     print("\n=== Saving comparison plots ===")
    #     save_plot_to_html(
    #         fig_comparison,
    #         os.path.join(results_dir, "plots", "comparison", "ofc_vs_acx_selectivity_comparison.html"),
    #         "OFC vs ACx selectivity comparison"
    #     )
        
    #     # Summary statistics comparison
    #     print("\n=== OFC vs ACx Comparison ===")
    #     comparison_data = {
    #         'Region': ['ACx', 'OFC'],
    #         'Total Active Units': [len(acx_selectivity_all_df), len(ofc_selectivity_df)],
    #         'Stimulus Selective': [acx_counts[0], ofc_counts[0]],
    #         'Outcome Modulated': [acx_counts[1], ofc_counts[1]],
    #         'Go/NoGo Selective': [acx_counts[2], ofc_counts[2]],
    #         'Choice Coding': [acx_counts[3], ofc_counts[3]],
    #     }
    #     comparison_df = pd.DataFrame(comparison_data)
    #     print(comparison_df.to_string(index=False))
        
    #     # Save comparison table
    #     save_dataframe_to_csv(
    #         comparison_df,
    #         os.path.join(results_dir, "tables", "ofc_vs_acx_comparison.csv"),
    #         "OFC vs ACx comparison table"
    #     )
    # else:
    #     print("\n=== Skipping OFC vs ACx comparison (no active units in either region) ===")
        
    # ============================================================================
    # Category sensitivity analysis
    # ============================================================================
    print("\n=== Category Sensitivity Analysis ===")
    print(f"Using boundaries: Low={0.983:.3f}, High={1.525:.3f}")
    
    # Compute category sensitivity for all ACx units
    acx_category_df = compute_category_sensitivity_for_all_units(
        acx_event_windows_data,
        acx_all_unit_indices,
        low_boundary=0.983,
        high_boundary=1.525,
        window=(-0.1, 0.5),
    )
    
    print(f"\nACx Category Sensitivity Results:")
    if len(acx_category_df) > 0:
        n_sensitive = acx_category_df['category_sensitive'].sum()  # Boolean column; sum is count
        print(f"  Category-sensitive units: {n_sensitive} / {len(acx_category_df)}")
        print(f"  Significant units (p<0.05): {n_sensitive}")
        
        # Show go/nogo selectivity (Go = High+Low, NoGo = Middle)
        if 'go_nogo_selective' in acx_category_df.columns:
            n_go_nogo = acx_category_df['go_nogo_selective'].sum()
            print(f"  Go/NoGo selective units (|d'|>0.5): {n_go_nogo} / {len(acx_category_df)}")
        
        # Show best category distribution
        if 'best_category' in acx_category_df.columns:
            best_cat_counts = acx_category_df['best_category'].value_counts()
            print(f"  Best category distribution:")
            for cat, count in best_cat_counts.items():
                print(f"    {cat}: {count}")
        
        # Save ACx category sensitivity results
        save_dataframe_to_csv(
            acx_category_df,
            os.path.join(results_dir, "tables", "acx_category_sensitivity.csv"),
            "ACx category sensitivity table"
        )
        
        # Create and save summary plot
        acx_category_fig = plot_category_sensitivity_summary(
            acx_category_df,
            region_name="ACx",
            low_boundary=0.983,
            high_boundary=1.525,
        )
        save_plot_to_html(
            acx_category_fig,
            os.path.join(results_dir, "plots", "acx", "acx_category_sensitivity_summary.html"),
            "ACx category sensitivity summary"
        )
        
    else:
        print("  No active units to analyze")
    
    # Run category sensitivity analysis for OFC (all units in matrix)
    ofc_event_matrix, _, _, _, _, _ = ofc_event_windows_data
    n_units_in_matrix = ofc_event_matrix.shape[0]
    ofc_all_unit_indices = np.arange(n_units_in_matrix)  # Include every unit in the matrix
    
    if len(ofc_all_unit_indices) > 0:
        print(f"\n=== OFC Category Sensitivity Analysis (all units, n={len(ofc_all_unit_indices)}) ===")
        ofc_category_df = compute_category_sensitivity_for_all_units(
            ofc_event_windows_data,
            ofc_all_unit_indices,
            low_boundary=0.983,
            high_boundary=1.525,
            window=(-0.1, 0.5),
        )
        
        print(f"\nOFC Category Sensitivity Results:")
        if len(ofc_category_df) > 0:
            n_sensitive_ofc = ofc_category_df['category_sensitive'].sum()  # Count significant OFC units
            print(f"  Category-sensitive units: {n_sensitive_ofc} / {len(ofc_category_df)}")
            
            # Show go/nogo selectivity (Go = High+Low, NoGo = Middle)
            if 'go_nogo_selective' in ofc_category_df.columns:
                n_go_nogo_ofc = ofc_category_df['go_nogo_selective'].sum()
                print(f"  Go/NoGo selective units (|d'|>0.5): {n_go_nogo_ofc} / {len(ofc_category_df)}")
            
            # Show best category distribution
            if 'best_category' in ofc_category_df.columns:
                best_cat_counts = ofc_category_df['best_category'].value_counts()
                print(f"  Best category distribution:")
                for cat, count in best_cat_counts.items():
                    print(f"    {cat}: {count}")
            
            # Save OFC category sensitivity results
            save_dataframe_to_csv(
                ofc_category_df,
                os.path.join(results_dir, "tables", "ofc_category_sensitivity.csv"),
                "OFC category sensitivity table"
            )
            
            # Create and save summary plot
            ofc_category_fig = plot_category_sensitivity_summary(
                ofc_category_df,
                region_name="OFC",
                low_boundary=0.983,
                high_boundary=1.525,
            )
            save_plot_to_html(
                ofc_category_fig,
                os.path.join(results_dir, "plots", "ofc", "ofc_category_sensitivity_summary.html"),
                "OFC category sensitivity summary"
            )
            
        else:
            print("  No good units to analyze")
    else:
        print("\nOFC: No good units found for category sensitivity analysis")
    
    set_progress(100, f"Analysis complete! Results saved to: {results_dir}")  # Final status for UI hooks


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # Allow parent_dir to be passed as command-line argument
        main(parent_dir=sys.argv[1])
    else:
        main()

