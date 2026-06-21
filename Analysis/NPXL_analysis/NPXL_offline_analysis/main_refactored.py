"""
Refactored main execution script for NPXL offline analysis.

This script uses nested loops to eliminate code duplication and improve maintainability:
1. Data loading (done once, outside loops)
2. Loop over brain regions (ACx, OFC)
3. Loop over alignment types (tone, choice, outcome)
4. Generate all plots and metrics systematically
"""
import sys
import os
from typing import Any

# Add workspace root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if 'single_unit_offline_analysis' in current_dir or 'NPXL_offline_analysis' in current_dir:
    workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
elif 'NPXL_analysis' in current_dir:
    workspace_root = os.path.dirname(os.path.dirname(current_dir))
else:
    test_dir = current_dir
    for _ in range(4):
        if os.path.exists(os.path.join(test_dir, 'Analysis', 'NPXL_analysis')):
            workspace_root = test_dir
            break
        test_dir = os.path.dirname(test_dir)
    else:
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
    save_raw_psth_for_active_units,
    plot_unit_heatmap,
    plot_psth_by_choice,
)
from Analysis.NPXL_analysis.single_unit_offline_analysis.category_analysis import (
    compute_category_sensitivity_for_all_units,
)
from Analysis.NPXL_analysis.single_unit_offline_analysis.unit import (
    create_units_from_event_data,
)


def compute_event_offsets(df: pd.DataFrame, column: str, bin_size: float) -> np.ndarray:
    """
    Compute per-trial offsets (seconds) from tone onset to the target event.
    
    Returns None if column doesn't exist or contains only NaN values.
    """
    if df is None or column not in df.columns or "time" not in df.columns:
        return None
    tone_bins = df["time"].to_numpy(dtype=float)
    event_bins = df[column].to_numpy(dtype=float)
    offsets_bins = event_bins - tone_bins
    if offsets_bins.size == 0 or np.all(np.isnan(offsets_bins)):
        return None
    return offsets_bins * bin_size


def main(parent_dir: str | None = None, save_plots: bool = True) -> None:
    """
    Main execution function with nested loops for clean, DRY code.
    
    Parameters:
    -----------
    parent_dir : str, optional
        Path to the parent directory containing the data
    save_plots : bool, optional
        If True, write PSTH/heatmap HTML under results_dir/plots. Default True.
        Set False for batch runs when only tables/metrics are needed.
    """
    print_config()
    
    # ============================================================================
    # STEP 1: Load all data (done once, outside loops)
    # ============================================================================
    if parent_dir is None:
        parent_dir = r"Z:\Shared\Amichai\NPXL\Recs\group7\catgt_G7A1_Expert_1B_3T_g0"
    
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    # Load spike data matrices (either side may be missing event windows → single-probe run)
    OFC_all, ACx_all, data_dir_OFC, data_dir_ACx = load_data(
        data_dir_parent=parent_dir, data_dir_OFC=None, data_dir_ACx=None
    )
    
    # Load unit labels for probes that have data
    ofc_g_index, acx_g_index, OFC_g, ACx_g = load_unit_labels(data_dir_OFC, data_dir_ACx)
    
    if data_dir_ACx is not None:
        acx_metadata = read_event_windows_metadata(data_dir_ACx)
    else:
        acx_metadata = None
    if data_dir_OFC is not None:
        ofc_metadata = read_event_windows_metadata(data_dir_OFC)
    else:
        ofc_metadata = None

    meta_for_bins = acx_metadata if acx_metadata is not None else ofc_metadata
    if meta_for_bins is None:
        raise RuntimeError("No event-window metadata loaded (both probes missing?)")
    orig_bin_size_sec = float(meta_for_bins["bin_size"])
    bin_to_sec = 1 / orig_bin_size_sec
    
    acx_event_windows_data = (
        load_full_event_windows_data(data_dir_ACx) if data_dir_ACx is not None else None
    )
    ofc_event_windows_data = (
        load_full_event_windows_data(data_dir_OFC) if data_dir_OFC is not None else None
    )
    
    print(f"\nLoaded event windows data:")
    if acx_event_windows_data is not None:
        acx_event_matrix = acx_event_windows_data[0]
        print(f"  ACx: {acx_event_matrix.shape} units × time × events")
    else:
        print("  ACx: (skipped — no event windows on imec0)")
    if ofc_event_windows_data is not None:
        ofc_event_matrix = ofc_event_windows_data[0]
        print(f"  OFC: {ofc_event_matrix.shape} units × time × events")
    else:
        print("  OFC: (skipped — no event windows on imec1)")
    
    # Setup results directory
    analysis_output_dir = os.path.join(parent_dir, "analysis_output")
    results_dir = setup_results_directory(analysis_output_dir, subfolder="")
    plots_dir = os.path.join(results_dir, "plots")
    
    # ============================================================================
    # STEP 2: Define region configurations (to iterate over)
    # ============================================================================
    regions_config: dict[str, dict[str, Any]] = {}
    if data_dir_ACx is not None and ACx_all is not None and acx_event_windows_data is not None:
        _, _, _, acx_stimuli_outcome_df, _, _ = acx_event_windows_data
        regions_config["acx"] = {
            "name": "ACx",
            "data_matrix": ACx_all,
            "event_windows_data": acx_event_windows_data,
            "stimuli_outcome_df": acx_stimuli_outcome_df,
            "unit_labels": acx_g_index,
            "good_units": ACx_g,
            "data_dir": data_dir_ACx,
            "tone_before": (int(-0.1 * bin_to_sec), 0),
            "tone_after": (0, int(0.3 * bin_to_sec)),
            "selectivity_window": (-0.1, 0.3),
            "category_window": (-0.1, 0.3),
            "tone_display_window": (-0.1, 0.3),
            "choice_outcome_display_window": (-0.1, 0.3),
            "alpha": 0.005,
        }
    if data_dir_OFC is not None and OFC_all is not None and ofc_event_windows_data is not None:
        _, _, _, ofc_stimuli_outcome_df, _, _ = ofc_event_windows_data
        regions_config["ofc"] = {
            "name": "OFC",
            "data_matrix": OFC_all,
            "event_windows_data": ofc_event_windows_data,
            "stimuli_outcome_df": ofc_stimuli_outcome_df,
            "unit_labels": ofc_g_index,
            "good_units": OFC_g,
            "data_dir": data_dir_OFC,
            "tone_before": (int(-0.1 * bin_to_sec), 0),
            "tone_after": (0, int(0.3 * bin_to_sec)),
            "selectivity_window": (-0.1, 0.3),
            "category_window": (-0.1, 0.3),
            "tone_display_window": (-0.1, 0.3),
            "choice_outcome_display_window": (-0.1, 0.3),
            "alpha": 0.05,
        }

    if not regions_config:
        raise RuntimeError(
            "No brain regions to analyze. Check that at least one probe has "
            "analysis_output/event_windows_matrix.npy under the catgt folder."
        )
    
    # Alignment configurations
    alignment_configs = {
        "tone": {
            "name": "Tone",
            "offset_column": None,  # Use original data
            "align_function": None,
            "is_category": False,
        },
        "choice": {
            "name": "Choice",
            "offset_column": "first_lick_time",
            "align_function": find_action_modulated_units,
            "is_category": False,
        },
        "outcome": {
            "name": "Outcome",
            "offset_column": "outcome_time",
            "align_function": find_outcome_modulated_units,
            "is_category": False,
        },
        "category": {
            "name": "Category",
            "offset_column": None,  # Uses tone-aligned data
            "align_function": None,
            "is_category": True,  # Special flag for category analysis
        },
    }
    
    # ============================================================================
    # STEP 3: Loop over brain regions
    # ============================================================================
    for region_key, region_config in regions_config.items():
        region_name = region_config["name"]
        
        print("\n" + "="*80)
        print(f"PROCESSING {region_name}")
        print("="*80)
        
        # Extract region-specific data
        data_matrix = region_config["data_matrix"]
        event_windows_data = region_config["event_windows_data"]
        stimuli_outcome_df = region_config["stimuli_outcome_df"]
        unit_labels = region_config["unit_labels"]
        
        # ========================================================================
        # STEP 3.1: Find active units (tone-aligned)
        # ========================================================================
        print(f"\n=== Finding active {region_name} units (tone-aligned) ===")
        active_units, t_vals, p_vals = find_active_units_by_midpoint(
            data_matrix,
            alpha=1,
            before_range=region_config["tone_before"],
            after_range=region_config["tone_after"],
        )
        print(f"  Found {len(active_units)} active {region_name} units")
        
        if len(active_units) == 0:
            print(f"  No active {region_name} units found - skipping region")
            continue
        
        # ========================================================================
        # STEP 3.2: Compute selectivity metrics
        # ========================================================================
        print(f"\n=== Computing selectivity metrics for {region_name} ===")
        selectivity_df = compute_selectivity_metrics_for_active_units(
            event_windows_data,
            active_units,
            window=region_config["selectivity_window"],
            region_name=region_name,
            use_unit_class=True,
        )
        
        if len(selectivity_df) > 0 and 'stimulus_selective' in selectivity_df.columns:
            print(f"  Stimulus selective: {selectivity_df['stimulus_selective'].sum()}")
            print(f"  Outcome modulated: {selectivity_df['outcome_modulated'].sum()}")
            print(f"  Go/NoGo selective: {selectivity_df['go_nogo_selective'].sum()}")
            print(f"  Choice coding: {selectivity_df['choice_coding'].sum()}")
        
        # Save selectivity results
        save_dataframe_to_csv(
            selectivity_df,
            os.path.join(results_dir, "tables", f"{region_key}_selectivity_metrics.csv"),
            f"{region_name} selectivity metrics"
        )
        
        # ========================================================================
        # STEP 3.3: Create Unit objects
        # ========================================================================
        print(f"\n=== Creating Unit objects for {region_name} ===")
        units = create_units_from_event_data(
            event_windows_data,
            active_units,
            region_name=region_name,
            unit_labels=unit_labels,
        )
        
        if save_plots:
            for unit in units:
                unit.set_plots_directory(plots_dir)
        
        # ========================================================================
        # STEP 3.4: Compute aligned data for all alignment types
        # ========================================================================
        print(f"\n=== Computing aligned data for {region_name} ===")
        aligned_data = {
            "tone": event_windows_data,  # Original data is tone-aligned
            "choice": None,
            "outcome": None,
            "category": event_windows_data,  # Category uses tone-aligned data
        }
        
        # Store p-values for each alignment type (for filtering plots)
        alignment_pvals = {
            "tone": p_vals,  # From tone-aligned active unit detection
            "choice": None,
            "outcome": None,
            "category": None,  # Will be computed from category ANOVA
        }
        
        # Store units for each alignment type
        alignment_units = {
            "tone": active_units,
            "choice": None,
            "outcome": None,
            "category": None,  # Will be computed from category sensitivity
        }
        
        # Compute choice-aligned data
        lick_offsets = compute_event_offsets(stimuli_outcome_df, "first_lick_time", orig_bin_size_sec)
        if lick_offsets is not None:
            print(f"  Computing choice-aligned data...")
            choice_units, _, choice_pvals, aligned_data["choice"] = find_action_modulated_units(
                event_windows_data,
                lick_offsets,
                bin_size_sec=orig_bin_size_sec,
                alpha=1.0,
                before_window=(-0.1, 0.0),
                after_window=(0.0, 0.3),
            )
            alignment_units["choice"] = choice_units
            alignment_pvals["choice"] = choice_pvals
            print(f"  Found {len(choice_units)} choice-modulated units")
        
        # Compute outcome-aligned data
        outcome_offsets = compute_event_offsets(stimuli_outcome_df, "outcome_time", orig_bin_size_sec)
        if outcome_offsets is not None:
            print(f"  Computing outcome-aligned data...")
            outcome_units, _, outcome_pvals, aligned_data["outcome"] = find_outcome_modulated_units(
                event_windows_data,
                outcome_offsets,
                bin_size_sec=orig_bin_size_sec,
                alpha=1.0,
                before_window=(-0.1, 0.0),
                after_window=(0.0, 0.3),
            )
            alignment_units["outcome"] = outcome_units
            alignment_pvals["outcome"] = outcome_pvals
            print(f"  Found {len(outcome_units)} outcome-modulated units")
        
        # Compute category sensitivity (uses tone-aligned data)
        print(f"  Computing category sensitivity...")
        category_df = compute_category_sensitivity_for_all_units(
            event_windows_data,
            active_units,
            low_boundary=0.983,
            high_boundary=1.525,
            window=region_config["category_window"],
        )
        
        if len(category_df) > 0 and 'category_anova_p' in category_df.columns:
            # Extract category-sensitive units and their p-values
            category_units_list = category_df['unit_idx'].values
            category_pvals_list = category_df['category_anova_p'].values
            alignment_units["category"] = np.array([int(u) for u in category_units_list])
            alignment_pvals["category"] = np.array(category_pvals_list)
            
            n_sensitive = category_df['category_sensitive'].sum()
            print(f"  Found {n_sensitive} category-sensitive units (p<0.05)")
            print(f"  Total units analyzed for category: {len(category_df)}")
        
        # ========================================================================
        # STEP 3.5: Loop over alignment types and generate plots
        # ========================================================================
        # Store plot paths for each alignment type
        psth_paths = {
            "tone": {},
            "choice": {},
            "outcome": {},
            "category": {},
        }
        heatmap_paths = {
            "tone": {},
            "choice": {},
            "outcome": {},
            "category": {},
        }
        
        if not save_plots:
            print(
                f"\n=== Skipping HTML plot generation for {region_name} "
                f"(save_plots=False); metrics CSVs still include empty plot path columns ==="
            )
        
        if save_plots:
            for align_type, align_config in alignment_configs.items():
                align_data = aligned_data[align_type]
                
                if align_data is None:
                    print(f"\n  Skipping {align_type}-aligned plots (no data available)")
                    continue
                
                # Get alignment-specific units and p-values
                align_units = alignment_units[align_type]
                align_pvals = alignment_pvals[align_type]
                
                if align_units is None or len(align_units) == 0:
                    print(f"\n  Skipping {align_type}-aligned plots (no significant units)")
                    continue
                
                # Create p-value lookup for this alignment type
                align_pval_lookup = {int(u): float(p) for u, p in zip(align_units, align_pvals)}
                
                print(f"\n=== Generating {align_type}-aligned plots for {region_name} ===")
                print(f"  Total {align_type}-modulated units: {len(align_units)}")
                print(f"  Units with p<{region_config['alpha']}: {sum(1 for p in align_pvals if p < region_config['alpha'])}")
                
                # Special handling for category plots
                if align_config.get("is_category", False):
                    # Category plots are different - use plot_psth_by_category
                    # Filter units with p < alpha for category sensitivity
                    significant_mask = align_pvals < region_config['alpha']
                    significant_units = align_units[significant_mask]
                    significant_pvals = align_pvals[significant_mask]
                    
                    if len(significant_units) == 0:
                        print(f"  No units with p<{region_config['alpha']} for category plots")
                        continue
                    
                    # Sort by p-value and take top units
                    sorted_idx = np.argsort(significant_pvals)
                    sorted_units = significant_units[sorted_idx]
                    sorted_p = significant_pvals[sorted_idx]
                    
                    n_to_plot = min(10, len(sorted_units))  # Plot top 10 or fewer
                    print(f"  Plotting PSTH by category for top {n_to_plot} units with p<{region_config['alpha']}:")
                    
                    for rank, (unit_idx, p_val) in enumerate(zip(sorted_units[:n_to_plot], sorted_p[:n_to_plot]), start=1):
                        unit_idx_int = int(unit_idx)
                        print(f"    Unit {unit_idx_int} (p={p_val:.4f}, rank={rank})")
                        
                        # Find the unit object
                        unit_obj = next((u for u in units if u.unit_idx == unit_idx_int), None)
                        if unit_obj is not None:
                            unit_obj.plot_psth_by_category(
                                low_boundary=0.983,
                                high_boundary=1.525,
                                display_window=region_config["category_window"],
                                cache_plot=True
                            )
                            # Store category PSTH path
                            region_lower = region_name.lower()
                            psth_category_path = os.path.join(
                                plots_dir, "psth", f"{region_key}_category",
                                f"{region_key}_unit_{unit_idx_int}_psth_by_category.html"
                            )
                            psth_paths["category"][unit_idx_int] = psth_category_path
                    
                    # Update metrics with category paths (will be saved in final step)
                    print(f"\n  Stored category plot paths for {len(psth_paths['category'])} units")
                    
                    continue  # Skip regular PSTH and heatmap generation for category
                
                # Regular alignment types (tone, choice, outcome)
                # Determine display window
                if align_type == "tone":
                    display_window = region_config["tone_display_window"]
                else:
                    display_window = region_config["choice_outcome_display_window"]
                
                # Generate region-specific name for PSTH saving
                # Directory structure: plots/psth/{region_key}_{align_type}/
                psth_region_name = f"{region_name}_{align_config['name']}"
                
                # Filter units with p < alpha for this alignment type
                significant_mask = align_pvals < region_config['alpha']
                significant_units = align_units[significant_mask]
                significant_pvals = align_pvals[significant_mask]
                
                if len(significant_units) == 0:
                    print(f"  No units with p<{region_config['alpha']} for {align_type}-aligned plots")
                    continue

                sorted_idx = np.argsort(significant_pvals)
                sorted_units = significant_units[sorted_idx]
                sorted_p = significant_pvals[sorted_idx]
                psth_dir = os.path.join(results_dir, "plots", "psth", f"{region_key}_{align_type}")

                if align_type == "choice":
                    # Choice PSTH: Lick vs Withhold (like category PSTH split)
                    os.makedirs(psth_dir, exist_ok=True)
                    print(
                        f"  Plotting PSTH by choice (Lick vs Withhold) for "
                        f"{len(significant_units)} units with p<{region_config['alpha']}..."
                    )
                    for rank, (unit_idx, p_val) in enumerate(zip(sorted_units, sorted_p), start=1):
                        unit_idx_int = int(unit_idx)
                        fig_choice_psth = plot_psth_by_choice(
                            align_data,
                            unit_idx_int,
                            display_window=display_window,
                            region_name=region_name,
                        )
                        fname = (
                            f"unit_{unit_idx_int}_rank{rank:03d}_p{float(p_val):.4f}"
                            f"_psth_by_choice.html"
                        )
                        choice_psth_path = os.path.join(psth_dir, fname)
                        save_plot_to_html(
                            fig_choice_psth,
                            choice_psth_path,
                            f"{region_name} Unit {unit_idx_int} PSTH by Choice (Lick vs Withhold)",
                        )
                        psth_paths["choice"][unit_idx_int] = choice_psth_path
                else:
                    # Tone / outcome: raw PSTH (all trials averaged)
                    save_raw_psth_for_active_units(
                        align_data,
                        significant_units,
                        significant_pvals,
                        psth_region_name,
                        results_dir,
                        display_window=display_window,
                    )
                    for rank, (u, p) in enumerate(zip(sorted_units, sorted_p), start=1):
                        fname = f"unit_{int(u)}_rank{rank:03d}_p{float(p):.4f}_raw_psth.html"
                        psth_paths[align_type][int(u)] = os.path.join(psth_dir, fname)
                
                # Generate heatmaps for significant units (p < alpha for this alignment)
                print(f"  Generating {align_type}-aligned heatmaps for {len(significant_units)} units with p<{region_config['alpha']}...")
                heatmap_count = 0
                for unit_idx in significant_units:
                    unit_idx_int = int(unit_idx)
                    
                    # Find the unit object
                    unit_obj = next((u for u in units if u.unit_idx == unit_idx_int), None)
                    if unit_obj is None:
                        continue
                    
                    heatmap_count += 1
                    if heatmap_count % 10 == 0 or heatmap_count == 1:
                        print(f"    Processing heatmap {heatmap_count}/{len(significant_units)}: Unit {unit_idx_int}")
                    
                    fig_heatmap = plot_unit_heatmap(
                        align_data,
                        unit_idx_int,
                        display_window=display_window,
                        region_name=region_name
                    )
                    
                    # Update the figure title to include alignment target
                    fig_heatmap.update_layout(
                        title=f"{region_name} Unit {unit_idx_int} - {align_config['name']} Aligned"
                    )
                    
                    heatmap_path = os.path.join(
                        results_dir, "plots", "heatmap", f"{align_type}_aligned",
                        f"{region_key}_unit_{unit_idx_int}_{align_type}_heatmap.html"
                    )
                    save_plot_to_html(
                        fig_heatmap,
                        heatmap_path,
                        f"{region_name} Unit {unit_idx_int} Heatmap ({align_config['name']}-Aligned)"
                    )
                    
                    # Store heatmap path
                    heatmap_paths[align_type][unit_idx_int] = heatmap_path
                
                print(f"  Generated {heatmap_count} {align_type}-aligned heatmaps")
                print(f"  Stored {len(psth_paths[align_type])} PSTH paths and {len(heatmap_paths[align_type])} heatmap paths for {align_type}")
        
        # ========================================================================
        # STEP 3.6: Final save of comprehensive metrics table with all paths
        # ========================================================================
        print(f"\n=== Final save: comprehensive {region_name} unit metrics with all plot paths ===")
        units_df = save_units_to_csv(
            units,
            os.path.join(results_dir, "tables", f"{region_key}_all_units_metrics.csv"),
            compute_all_metrics=True,
            selectivity_window=region_config["selectivity_window"],
            category_window=region_config["category_window"],
            category_boundaries=(0.983, 1.525),
            psth_baseline_window=(-0.5, 0),
            description=f"{region_name} comprehensive unit metrics"
        )
        
        # Add p-values for all alignment types
        for align_type in ["tone", "choice", "outcome", "category"]:
            align_units_arr = alignment_units[align_type]
            align_pvals_arr = alignment_pvals[align_type]
            
            if align_units_arr is not None and align_pvals_arr is not None:
                pval_map = {int(u): float(p) for u, p in zip(align_units_arr, align_pvals_arr)}
                units_df[f"{align_type}_p_value"] = units_df["unit_idx"].map(pval_map)
        
        # Add all plot paths from all alignment types
        for align_type in ["tone", "choice", "outcome", "category"]:
            units_df[f"psth_{align_type}_path"] = units_df["unit_idx"].map(psth_paths[align_type])
            if align_type != "category":  # Category doesn't have heatmaps
                units_df[f"heatmap_{align_type}_path"] = units_df["unit_idx"].map(heatmap_paths[align_type])
        
        # Resave with all added columns
        save_dataframe_to_csv(
            units_df,
            os.path.join(results_dir, "tables", f"{region_key}_all_units_metrics.csv"),
            f"{region_name} comprehensive metrics (with all p-values and plot paths)"
        )
        print(f"  Saved metrics for {len(units)} {region_name} units")
        print(f"  P-value columns: tone_p_value, choice_p_value, outcome_p_value, category_p_value")
        print(f"  Plot path columns: psth_tone_path, psth_choice_path, psth_outcome_path, psth_category_path")
        print(f"                     heatmap_tone_path, heatmap_choice_path, heatmap_outcome_path")
        
        # ========================================================================
        # STEP 3.7: Save category sensitivity results to CSV
        # ========================================================================
        # Category HTML PSTHs are produced in STEP 3.5 when save_plots is True.
        print(f"\n=== Saving category sensitivity results for {region_name} ===")
        if category_df is not None and len(category_df) > 0:
            # Save category sensitivity results
            save_dataframe_to_csv(
                category_df,
                os.path.join(results_dir, "tables", f"{region_key}_category_sensitivity.csv"),
                f"{region_name} category sensitivity"
            )
    
    # ============================================================================
    # FINAL SUMMARY
    # ============================================================================
    print("\n" + "="*80)
    print(f"ANALYSIS COMPLETE!")
    print(f"All results saved to: {results_dir}")
    print("="*80)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main(parent_dir=sys.argv[1])
    else:
        main()
