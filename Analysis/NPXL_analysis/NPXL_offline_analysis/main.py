"""
Main execution script for NPXL offline analysis.

This script orchestrates the complete analysis pipeline:
1. Data loading
2. Active unit detection
3. Selectivity metrics computation
4. Visualization
5. Category sensitivity analysis
"""
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


def main(parent_dir: str = None):
    """
    Main execution function.
    
    Parameters:
    -----------
    parent_dir : str, optional
        Path to the parent directory containing the data. If None, uses default path.
    """
    # Print configuration
    print_config()
    
    # ============================================================================
    # Load and explore data
    # ============================================================================
    if parent_dir is None:
        parent_dir = r"Z:\Shared\Amichai\NPXL\Recs\group5\catGTGroup5\catgt_G5A3_2b_4t_new2_g0"
    
    OFC_all, ACx_all, data_dir_OFC, data_dir_ACx = load_data(
        data_dir_parent=parent_dir, data_dir_OFC=None, data_dir_ACx=None
    )
    
    # Load unit labels
    ofc_g_index, acx_g_index, OFC_g, ACx_g = load_unit_labels(data_dir_OFC, data_dir_ACx)
    
    # Read metadata
    ofc_metadata = read_event_windows_metadata(data_dir_OFC)
    acx_metadata = read_event_windows_metadata(data_dir_ACx)
    orig_bin_size_sec = float(acx_metadata["bin_size"])  # seconds per bin from preprocessing
    bin_to_sec = 1 / orig_bin_size_sec
    
    # Find active units by midpoint of ACx
    active_units_acx, t_vals_acx, p_vals_acx = find_active_units_by_midpoint(
        ACx_all,
        alpha=1,
        before_range=(int(-0.1 * bin_to_sec), 0),
        after_range=(0, int(0.5 * bin_to_sec)))
    
    # Load full event windows data with stimuli/outcome information
    acx_event_windows_data = load_full_event_windows_data(data_dir_ACx)
    ofc_event_windows_data = load_full_event_windows_data(data_dir_OFC)
    
    # Extract components
    (acx_event_matrix, acx_time_axis, acx_valid_indices, acx_stimuli_outcome_df, 
     acx_metadata_full, acx_lick_data) = acx_event_windows_data
    (ofc_event_matrix, ofc_time_axis, ofc_valid_indices, ofc_stimuli_outcome_df, 
     ofc_metadata_full, ofc_lick_data) = ofc_event_windows_data
    
    print(f"\nLoaded event windows data:")
    print(f"  ACx: {acx_event_matrix.shape} units × time × events")
    print(f"  OFC: {ofc_event_matrix.shape} units × time × events")
    print(f"\nACx stimuli/outcome columns: {list(acx_stimuli_outcome_df.columns)}")
    print(f"ACx unique outcomes: {acx_stimuli_outcome_df['outcome'].unique() if 'outcome' in acx_stimuli_outcome_df.columns else 'N/A'}")
    if 'stimulus' in acx_stimuli_outcome_df.columns:
        print(f"ACx unique stimuli: {sorted(acx_stimuli_outcome_df['stimulus'].unique())}")
    
    # Setup results directory
    analysis_output_dir = os.path.join(parent_dir, "analysis_output")
    results_dir = setup_results_directory(analysis_output_dir, subfolder="")
    
    # ============================================================================
    # Compute selectivity metrics for ACx active units
    # ============================================================================
    # Using batch processing function (efficient for many units)
    acx_selectivity_df = compute_selectivity_metrics_for_active_units(
        acx_event_windows_data,
        active_units_acx,
        window=(-0.1, 1),
        region_name="ACx",
        use_unit_class=True,  # Use Unit class internally for cleaner code
    )
    
    # Alternative: Create Unit objects for interactive analysis
    # acx_units = create_units_from_event_data(
    #     acx_event_windows_data,
    #     active_units_acx,
    #     region_name="ACx",
    #     unit_labels=acx_g_index,
    # )
    # # Then you can do: unit = acx_units[0]; unit.compute_selectivity(); unit.plot_psth_by_stimulus()
    
    print(f"\nSelectivity metrics for {len(acx_selectivity_df)} ACx active units:")
    if len(acx_selectivity_df) > 0 and 'stimulus_selective' in acx_selectivity_df.columns:
        print(f"  Stimulus selective: {acx_selectivity_df['stimulus_selective'].sum()}")
        print(f"  Outcome modulated (p<0.05): {acx_selectivity_df['outcome_modulated'].sum()}")
        print(f"  Go/NoGo selective (|d'|>0.5): {acx_selectivity_df['go_nogo_selective'].sum()}")
        print(f"  Choice coding (|CP_corr|>0.1): {acx_selectivity_df['choice_coding'].sum()}")
        print(f"\nSummary statistics:")
        print(acx_selectivity_df.describe())
    else:
        print("  No active units found - skipping selectivity metrics")
    
    # Save ACx selectivity results
    print("\n=== Saving ACx selectivity results ===")
    save_dataframe_to_csv(
        acx_selectivity_df,
        os.path.join(results_dir, "tables", "acx_selectivity_metrics.csv"),
        "ACx selectivity metrics table"
    )
    
    # ============================================================================
    # Create Unit objects and save comprehensive metrics table
    # ============================================================================
    print("\n=== Creating Unit objects and computing all metrics ===")
    acx_units = create_units_from_event_data(
        acx_event_windows_data,
        active_units_acx,
        region_name="ACx",
        unit_labels=acx_g_index,
    )
    
    # Set plots directory for units
    plots_dir = os.path.join(results_dir, "plots")
    for unit in acx_units:
        unit.set_plots_directory(plots_dir)
    
    # Generate and save plots for all units (heatmap and PSTHs)
    print("\n=== Generating plots for all ACx units ===")
    for i, unit in enumerate(acx_units):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Processing unit {i+1}/{len(acx_units)}: Unit {unit.unit_idx}")
        # Generate and save heatmap
        unit.plot_heatmap(display_window=(-0.5, 1.0), cache_plot=True)
        # Generate and save PSTH plots
        unit.plot_psth_by_stimulus(display_window=(-0.5, 1.0), cache_plot=True)
        unit.plot_psth_by_outcome(display_window=(-0.5, 1.0), cache_plot=True)
        unit.plot_raw_psth(display_window=(-0.5, 1.0), cache_plot=True)
    
    # Save comprehensive metrics table for all ACx units
    print("\n=== Saving comprehensive ACx unit metrics table ===")
    acx_units_df = save_units_to_csv(
        acx_units,
        os.path.join(results_dir, "tables", "acx_all_units_metrics.csv"),
        compute_all_metrics=True,
        selectivity_window=(-0.1, 1),
        category_window=(-0.1, 0.5),
        category_boundaries=(0.983, 1.525),
        psth_baseline_window=(-0.5, 0),
        description="ACx comprehensive unit metrics"
    )
    print(f"  Saved metrics for {len(acx_units)} ACx units")
    
    # ============================================================================
    # Plot and save raw PSTH for all responsive units
    # ============================================================================
    save_raw_psth_for_active_units(
        acx_event_windows_data,
        active_units_acx,
        p_vals_acx,
        "ACx",
        results_dir,
        display_window=(-0.5, 1.0),
    )
    
    # ============================================================================
    # Plot PSTH by stimulus and outcome for top active units
    # ============================================================================
    n_units_to_plot = min(5, len(active_units_acx))
    for i, unit_idx in enumerate(active_units_acx[:n_units_to_plot]):
        print(f"\n=== ACx Unit {unit_idx} (Rank {i+1} by significance) ===")
        
        # Plot by stimulus
        if 'stimulus' in acx_stimuli_outcome_df.columns:
            fig_stim = plot_psth_by_stimulus(
                acx_event_windows_data,
                int(unit_idx),
                display_window=(-0.5, 1.0),
                region_name="ACx"
            )
        
        # Plot by outcome
        if 'outcome' in acx_stimuli_outcome_df.columns:
            fig_outcome = plot_psth_by_outcome(
                acx_event_windows_data,
                int(unit_idx),
                display_window=(-0.5, 1.0),
                region_name="ACx"
            )
            
            # Save PSTH plots
            if 'stimulus' in acx_stimuli_outcome_df.columns:
                save_plot_to_html(
                    fig_stim,
                    os.path.join(results_dir, "plots", "psth_by_stimulus", f"acx_unit_{unit_idx}_psth_by_stimulus.html"),
                    f"ACx Unit {unit_idx} PSTH by Stimulus"
                )
            save_plot_to_html(
                fig_outcome,
                os.path.join(results_dir, "plots", "psth_by_outcome", f"acx_unit_{unit_idx}_psth_by_outcome.html"),
                f"ACx Unit {unit_idx} PSTH by Outcome"
            )
    
    # ============================================================================
    # Create selectivity summary plots
    # ============================================================================
    acx_fig_metrics, acx_fig_class = plot_selectivity_summary(acx_selectivity_df, region_name="ACx")
    
    # Save ACx summary plots
    print("\n=== Saving ACx summary plots ===")
    save_plot_to_html(
        acx_fig_metrics,
        os.path.join(results_dir, "plots", "acx", "acx_selectivity_metrics_summary.html"),
        "ACx selectivity metrics summary"
    )
    save_plot_to_html(
        acx_fig_class,
        os.path.join(results_dir, "plots", "acx", "acx_unit_classification_summary.html"),
        "ACx unit classification summary"
    )
    
    # Display detailed summary table
    print("\n=== ACx Selectivity Summary Table ===")
    if len(acx_selectivity_df) > 0 and 'stimulus_selective' in acx_selectivity_df.columns:
        print(acx_selectivity_df[['unit_idx', 'stimulus_selective', 'outcome_modulated', 
                                   'go_nogo_selective', 'choice_coding']].to_string(index=False))
    else:
        print("No active units found - no summary table to display")
    
    # ============================================================================
    # Compare OFC vs ACx
    # ============================================================================
    # Find active units in OFC
    active_units_ofc, t_vals_ofc, p_vals_ofc = find_active_units_by_midpoint(
        OFC_all,
        alpha=1,
        before_range=(int(-0.5*bin_to_sec), 0),
        after_range=(0, int(2 * bin_to_sec)),
    )
    
    print(f"\n=== OFC Active Units ===")
    print(f"Found {len(active_units_ofc)} active OFC units")
    
    # Plot and save raw PSTH for all OFC responsive units
    save_raw_psth_for_active_units(
        ofc_event_windows_data,
        active_units_ofc,
        p_vals_ofc,
        "OFC",
        results_dir,
        display_window=(-0.5, 1.0),
    )
    
    # Compute selectivity metrics for OFC
    if len(active_units_ofc) > 0:
        ofc_selectivity_df = compute_selectivity_metrics_for_active_units(
            ofc_event_windows_data,
            active_units_ofc,
            window=(-0.1, 0.5),
            region_name="OFC",
            use_unit_class=True,
        )
    
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
        
        # ============================================================================
        # Create Unit objects and save comprehensive metrics table for OFC
        # ============================================================================
        print("\n=== Creating Unit objects and computing all metrics for OFC ===")
        ofc_units = create_units_from_event_data(
            ofc_event_windows_data,
            active_units_ofc,
            region_name="OFC",
            unit_labels=ofc_g_index,
        )
        
        # Set plots directory for units
        for unit in ofc_units:
            unit.set_plots_directory(plots_dir)
        
        # Generate and save plots for all units (heatmap and PSTHs)
        print("\n=== Generating plots for all OFC units ===")
        for i, unit in enumerate(ofc_units):
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  Processing unit {i+1}/{len(ofc_units)}: Unit {unit.unit_idx}")
            # Generate and save heatmap
            unit.plot_heatmap(display_window=(-0.5, 2.0), cache_plot=True)
            # Generate and save PSTH plots
            unit.plot_psth_by_stimulus(display_window=(-0.5, 1.0), cache_plot=True)
            unit.plot_psth_by_outcome(display_window=(-0.5, 1.0), cache_plot=True)
            unit.plot_raw_psth(display_window=(-0.5, 2.0), cache_plot=True)
        
        # Save comprehensive metrics table for all OFC units
        print("\n=== Saving comprehensive OFC unit metrics table ===")
        ofc_units_df = save_units_to_csv(
            ofc_units,
            os.path.join(results_dir, "tables", "ofc_all_units_metrics.csv"),
            compute_all_metrics=True,
            selectivity_window=(-0.1, 1.5),
            category_window=(-0.1, 1.5),
            category_boundaries=(0.983, 1.525),
            psth_baseline_window=(-0.5, 0),
            description="OFC comprehensive unit metrics"
        )
        print(f"  Saved metrics for {len(ofc_units)} OFC units")
        
        # Create OFC summary plots
        ofc_fig_metrics, ofc_fig_class = plot_selectivity_summary(ofc_selectivity_df, region_name="OFC")
        
        # Save OFC summary plots
        print("\n=== Saving OFC summary plots ===")
        save_plot_to_html(
            ofc_fig_metrics,
            os.path.join(results_dir, "plots", "ofc", "ofc_selectivity_metrics_summary.html"),
            "OFC selectivity metrics summary"
        )
        save_plot_to_html(
            ofc_fig_class,
            os.path.join(results_dir, "plots", "ofc", "ofc_unit_classification_summary.html"),
            "OFC unit classification summary"
        )
        
        # Comparison plot: OFC vs ACx
        has_acx_data = len(acx_selectivity_df) > 0 and 'stimulus_selective' in acx_selectivity_df.columns
        has_ofc_data = len(ofc_selectivity_df) > 0 and 'stimulus_selective' in ofc_selectivity_df.columns
        
        if has_acx_data or has_ofc_data:
            fig_comparison = go.Figure()
            
            metrics = ['Stimulus Selective', 'Outcome Modulated', 'Go/NoGo Selective', 'Choice Coding']
            
            if has_acx_data:
                acx_counts = [
                    acx_selectivity_df['stimulus_selective'].sum(),
                    acx_selectivity_df['outcome_modulated'].sum(),
                    acx_selectivity_df['go_nogo_selective'].sum(),
                    acx_selectivity_df['choice_coding'].sum(),
                ]
                acx_proportions = [c / len(acx_selectivity_df) * 100 for c in acx_counts]
            else:
                acx_counts = [0, 0, 0, 0]
                acx_proportions = [0, 0, 0, 0]
            
            if has_ofc_data:
                ofc_counts = [
                    ofc_selectivity_df['stimulus_selective'].sum(),
                    ofc_selectivity_df['outcome_modulated'].sum(),
                    ofc_selectivity_df['go_nogo_selective'].sum(),
                    ofc_selectivity_df['choice_coding'].sum(),
                ]
                ofc_proportions = [c / len(ofc_selectivity_df) * 100 for c in ofc_counts]
            else:
                ofc_counts = [0, 0, 0, 0]
                ofc_proportions = [0, 0, 0, 0]
            
            fig_comparison.add_trace(go.Bar(
                x=metrics,
                y=acx_proportions,
                name='ACx',
                marker_color='#1f77b4',
                text=[f'{c} ({p:.1f}%)' for c, p in zip(acx_counts, acx_proportions)],
                textposition='outside',
                hovertemplate='ACx<br>%{x}<br>%{y:.1f}% (%{text})<extra></extra>'
            ))
            
            fig_comparison.add_trace(go.Bar(
                x=metrics,
                y=ofc_proportions,
                name='OFC',
                marker_color='#ff7f0e',
                text=[f'{c} ({p:.1f}%)' for c, p in zip(ofc_counts, ofc_proportions)],
                textposition='outside',
                hovertemplate='OFC<br>%{x}<br>%{y:.1f}% (%{text})<extra></extra>'
            ))
            
            fig_comparison.update_layout(
                title='OFC vs ACx - Selectivity Comparison (Proportion of Active Units)',
                xaxis_title='Selectivity Type',
                yaxis_title='Percentage of Active Units',
                barmode='group',
                template='plotly_white',
                height=500,
                legend=dict(x=0.7, y=0.95)
            )
            
            # Save comparison plot
            print("\n=== Saving comparison plots ===")
            save_plot_to_html(
                fig_comparison,
                os.path.join(results_dir, "plots", "comparison", "ofc_vs_acx_selectivity_comparison.html"),
                "OFC vs ACx selectivity comparison"
            )
            
            # Summary statistics comparison
            print("\n=== OFC vs ACx Comparison ===")
            comparison_data = {
                'Region': ['ACx', 'OFC'],
                'Total Active Units': [len(acx_selectivity_df), len(ofc_selectivity_df)],
                'Stimulus Selective': [acx_counts[0], ofc_counts[0]],
                'Outcome Modulated': [acx_counts[1], ofc_counts[1]],
                'Go/NoGo Selective': [acx_counts[2], ofc_counts[2]],
                'Choice Coding': [acx_counts[3], ofc_counts[3]],
            }
            comparison_df = pd.DataFrame(comparison_data)
            print(comparison_df.to_string(index=False))
            
            # Save comparison table
            save_dataframe_to_csv(
                comparison_df,
                os.path.join(results_dir, "tables", "ofc_vs_acx_comparison.csv"),
                "OFC vs ACx comparison table"
            )
        else:
            print("\n=== Skipping OFC vs ACx comparison (no active units in either region) ===")
        
        # Plot example units from OFC
        print("\n=== Plotting example OFC active units ===")
        n_units_to_plot_ofc = min(3, len(active_units_ofc))
        for i, unit_idx in enumerate(active_units_ofc[:n_units_to_plot_ofc]):
            print(f"\n=== OFC Unit {unit_idx} (Rank {i+1}) ===")
            
            if 'stimulus' in ofc_stimuli_outcome_df.columns:
                fig_stim_ofc = plot_psth_by_stimulus(
                    ofc_event_windows_data,
                    int(unit_idx),
                    display_window=(-0.5, 1.0),
                    region_name="OFC"
                )
            
            if 'outcome' in ofc_stimuli_outcome_df.columns:
                fig_outcome_ofc = plot_psth_by_outcome(
                    ofc_event_windows_data,
                    int(unit_idx),
                    display_window=(-0.5, 1.0),
                    region_name="OFC"
                )
                
                # Save OFC PSTH plots
                if 'stimulus' in ofc_stimuli_outcome_df.columns:
                    save_plot_to_html(
                        fig_stim_ofc,
                        os.path.join(results_dir, "plots", "psth_by_stimulus", f"ofc_unit_{unit_idx}_psth_by_stimulus.html"),
                        f"OFC Unit {unit_idx} PSTH by Stimulus"
                    )
                save_plot_to_html(
                    fig_outcome_ofc,
                    os.path.join(results_dir, "plots", "psth_by_outcome", f"ofc_unit_{unit_idx}_psth_by_outcome.html"),
                    f"OFC Unit {unit_idx} PSTH by Outcome"
                )
    else:
        print("No active OFC units found for comparison")
    
    # ============================================================================
    # Category sensitivity analysis
    # ============================================================================
    print("\n=== Category Sensitivity Analysis ===")
    print(f"Using boundaries: Low={0.983:.3f}, High={1.525:.3f}")
    
    # Compute category sensitivity for ACx active units
    acx_category_df = compute_category_sensitivity_for_all_units(
        acx_event_windows_data,
        active_units_acx,
        low_boundary=0.983,
        high_boundary=1.525,
        window=(-0.1, 0.5),
    )
    
    print(f"\nACx Category Sensitivity Results:")
    if len(acx_category_df) > 0:
        n_sensitive = acx_category_df['category_sensitive'].sum()
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
        
        # Plot PSTH by category for top category-sensitive units
        if n_sensitive > 0:
            sensitive_units = acx_category_df[acx_category_df['category_sensitive']].copy()
            sensitive_units = sensitive_units.sort_values('category_anova_p')
            
            n_to_plot = min(5, len(sensitive_units))
            print(f"\n  Plotting PSTH by category for top {n_to_plot} category-sensitive units:")
            
            for i, row in sensitive_units.head(n_to_plot).iterrows():
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
                save_plot_to_html(
                    fig_cat,
                    os.path.join(results_dir, "plots", "psth_by_category", f"acx_unit_{unit_idx}_psth_by_category.html"),
                    f"ACx Unit {unit_idx} PSTH by Category"
                )
    else:
        print("  No active units to analyze")
    
    # Run category sensitivity analysis for OFC (using all good units, not just active units)
    ofc_event_matrix, _, _, _, _, _ = ofc_event_windows_data
    n_units_in_matrix = ofc_event_matrix.shape[0]
    ofc_good_units_filtered = [idx for idx in OFC_g if idx < n_units_in_matrix]
    ofc_good_units_array = np.array(ofc_good_units_filtered)
    
    if len(ofc_good_units_array) > 0:
        print(f"\n=== OFC Category Sensitivity Analysis (using {len(ofc_good_units_array)} good units out of {len(OFC_g)} total good units) ===")
        print(f"  Event matrix has {n_units_in_matrix} units, filtered {len(OFC_g) - len(ofc_good_units_array)} units that are out of bounds")
        ofc_category_df = compute_category_sensitivity_for_all_units(
            ofc_event_windows_data,
            ofc_good_units_array,
            low_boundary=0.983,
            high_boundary=1.525,
            window=(-0.1, 0.5),
        )
        
        print(f"\nOFC Category Sensitivity Results:")
        if len(ofc_category_df) > 0:
            n_sensitive_ofc = ofc_category_df['category_sensitive'].sum()
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
            
            # Plot PSTH by category for top category-sensitive units
            if n_sensitive_ofc > 0:
                sensitive_units_ofc = ofc_category_df[ofc_category_df['category_sensitive']].copy()
                sensitive_units_ofc = sensitive_units_ofc.sort_values('category_anova_p')
                
                n_to_plot_ofc = min(5, len(sensitive_units_ofc))
                print(f"\n  Plotting PSTH by category for top {n_to_plot_ofc} category-sensitive units:")
                
                for i, row in sensitive_units_ofc.head(n_to_plot_ofc).iterrows():
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
                    save_plot_to_html(
                        fig_cat_ofc,
                        os.path.join(results_dir, "plots", "psth_by_category", f"ofc_unit_{unit_idx}_psth_by_category.html"),
                        f"OFC Unit {unit_idx} PSTH by Category"
                    )
        else:
            print("  No good units to analyze")
    else:
        print("\nOFC: No good units found for category sensitivity analysis")
    
    print(f"\n=== Analysis complete! All results saved to: {results_dir} ===")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # Allow parent_dir to be passed as command-line argument
        main(parent_dir=sys.argv[1])
    else:
        main()

