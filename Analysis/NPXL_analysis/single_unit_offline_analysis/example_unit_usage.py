"""
Example script demonstrating Unit class usage for interactive analysis.

This shows how to use the Unit class for single-unit analysis,
which provides a cleaner API than the batch processing functions.
"""
import numpy as np
from Analysis.NPXL_analysis.single_unit_offline_analysis.data_loading import (
    load_data,
    load_unit_labels,
    load_full_event_windows_data,
)
from Analysis.NPXL_analysis.single_unit_offline_analysis.active_units import (
    find_active_units_by_midpoint,
)
from Analysis.NPXL_analysis.single_unit_offline_analysis.unit import (
    Unit,
    create_units_from_event_data,
)


def example_single_unit_analysis():
    """Example of analyzing a single unit using the Unit class."""
    # Load data (same as main.py)
    parent_dir = r"Z:\Shared\Amichai\NPXL\Recs\group5\catGTGroup5\catgt_G5A3_2b_4t_new2_g0"
    OFC_all, ACx_all, data_dir_OFC, data_dir_ACx = load_data(data_dir_parent=parent_dir)
    ofc_g_index, acx_g_index, OFC_g, ACx_g = load_unit_labels(data_dir_OFC, data_dir_ACx)
    
    # Load event windows data
    acx_event_windows_data = load_full_event_windows_data(data_dir_ACx)
    
    # Find active units
    active_units_acx, _, _ = find_active_units_by_midpoint(ACx_all, alpha=0.05)
    
    # ============================================================================
    # Example 1: Create a single Unit object
    # ============================================================================
    print("=== Example 1: Single Unit Analysis ===")
    unit_idx = active_units_acx[0]  # Get first active unit
    unit = Unit(
        unit_idx,
        acx_event_windows_data,
        region_name="ACx",
        unit_labels=acx_g_index,
    )
    
    print(f"Created: {unit}")
    print(f"  Number of trials: {unit.n_trials}")
    print(f"  Number of time bins: {unit.n_time_bins}")
    print(f"  Bin size: {unit.bin_size} seconds")
    print(f"  Mean firing rate: {unit.get_mean_firing_rate():.2f} Hz")
    
    # Compute selectivity metrics
    selectivity = unit.compute_selectivity(window=(-0.1, 1.0))
    print(f"\nSelectivity metrics:")
    print(f"  Stimulus selective: {selectivity['stimulus_selective']}")
    print(f"  Outcome modulated: {selectivity['outcome_modulated']}")
    print(f"  Go/NoGo selective: {selectivity['go_nogo_selective']}")
    print(f"  Choice coding: {selectivity['choice_coding']}")
    
    # Compute category sensitivity
    category_sens = unit.compute_category_sensitivity(
        low_boundary=0.983,
        high_boundary=1.525,
        window=(-0.1, 0.5),
    )
    print(f"\nCategory sensitivity:")
    print(f"  Category sensitive: {category_sens['category_sensitive']}")
    print(f"  ANOVA p-value: {category_sens['category_anova_p']:.4f}")
    print(f"  Best category: {category_sens['best_category']}")
    
    # Compute PSTH metrics
    psth_metrics = unit.compute_psth_metrics(baseline_window=(-0.5, 0))
    print(f"\nPSTH Metrics:")
    print(f"  Response type: {psth_metrics['response_type']}")
    print(f"  Onset latency: {psth_metrics['onset_latency']:.3f}s")
    print(f"  Peak latency: {psth_metrics['peak_latency']:.3f}s")
    print(f"  Response magnitude: {psth_metrics['response_magnitude']:.2f} spikes/s")
    print(f"  FWHM: {psth_metrics['fwhm']:.3f}s")
    print(f"  Signal-to-noise: {psth_metrics['signal_to_noise']:.2f}")
    print(f"  Baseline rate: {psth_metrics['baseline_rate']:.2f} spikes/s")
    print(f"  Peak rate: {psth_metrics['peak_rate']:.2f} spikes/s")
    
    # Compute d' between conditions
    d_hit_miss = unit.compute_d_prime("Hit", "Miss", window=(-0.1, 0.5))
    d_fa_cr = unit.compute_d_prime("False Alarm", "CR", window=(-0.1, 0.5))
    print(f"\nd' Metrics:")
    if d_hit_miss is not None:
        print(f"  d' (Hit vs Miss): {d_hit_miss:.3f}")
    if d_fa_cr is not None:
        print(f"  d' (FA vs CR): {d_fa_cr:.3f}")
    
    # Fit GLM
    glm_results = unit.fit_glm(window=(-0.1, 0.5))
    if glm_results is not None:
        print(f"\nGLM Results:")
        print(f"  R²: {glm_results['r_squared']:.3f}")
        print(f"  Intercept: {glm_results['intercept']:.3f}")
        print(f"  Coefficients:")
        for name, coef in zip(glm_results['feature_names'], glm_results['coefficients']):
            print(f"    {name}: {coef:.3f}")
    
    # Generate plots (automatically cached)
    print(f"\nGenerating plots...")
    fig_stim = unit.plot_psth_by_stimulus(display_window=(-0.5, 1.0))
    fig_outcome = unit.plot_psth_by_outcome(display_window=(-0.5, 1.0))
    fig_raw = unit.plot_raw_psth(display_window=(-0.5, 1.0))
    fig_category = unit.plot_psth_by_category(
        low_boundary=0.983,
        high_boundary=1.525,
        display_window=(-0.5, 1.0),
    )
    fig_heatmap = unit.plot_heatmap(display_window=(-0.5, 1.0))
    
    # Access cached plots
    print(f"\nCached plots available:")
    print(f"  Available plot types: {list(unit.plots.keys())}")
    
    # Get a specific plot
    cached_heatmap = unit.get_plot('heatmap')
    if cached_heatmap is not None:
        print(f"  Retrieved heatmap plot: {type(cached_heatmap)}")
    
    # Get trial statistics
    trial_stats = unit.get_trial_stats()
    print(f"\nTrial Statistics:")
    print(f"  Hit: {trial_stats['Hit']}")
    print(f"  Miss: {trial_stats['Miss']}")
    print(f"  False Alarm: {trial_stats['FA']}")
    print(f"  CR: {trial_stats['CR']}")
    print(f"  Total: {trial_stats['Total']}")
    
    # Note: In interactive use, you might call fig_stim.show() here
    
    # ============================================================================
    # Example 2: Create multiple Unit objects for batch analysis
    # ============================================================================
    print("\n=== Example 2: Batch Unit Analysis ===")
    # Create Unit objects for top 5 active units
    top_units = create_units_from_event_data(
        acx_event_windows_data,
        active_units_acx[:5],
        region_name="ACx",
        unit_labels=acx_g_index,
    )
    
    print(f"Created {len(top_units)} Unit objects")
    
    # Analyze each unit
    for i, unit in enumerate(top_units, 1):
        selectivity = unit.compute_selectivity(window=(-0.1, 1.0))
        print(f"\nUnit {i} (idx={unit.unit_idx}):")
        print(f"  Stimulus selective: {selectivity['stimulus_selective']}")
        print(f"  Outcome modulated: {selectivity['outcome_modulated']}")
        print(f"  Go/NoGo d': {selectivity['go_nogo_dprime']:.3f}")
    
    # ============================================================================
    # Example 3: Caching benefits
    # ============================================================================
    print("\n=== Example 3: Caching Benefits ===")
    unit = top_units[0]
    
    # First computation (will compute)
    import time
    start = time.time()
    selectivity1 = unit.compute_selectivity(window=(-0.1, 1.0))
    time1 = time.time() - start
    print(f"First computation: {time1:.4f} seconds")
    
    # Second computation (will use cache)
    start = time.time()
    selectivity2 = unit.compute_selectivity(window=(-0.1, 1.0))
    time2 = time.time() - start
    print(f"Second computation (cached): {time2:.4f} seconds")
    print(f"Speedup: {time1/time2:.1f}x faster")
    
    # Force recompute
    start = time.time()
    selectivity3 = unit.compute_selectivity(window=(-0.1, 1.0), force_recompute=True)
    time3 = time.time() - start
    print(f"Force recompute: {time3:.4f} seconds")
    
    # ============================================================================
    # Example 4: Convert to dictionary for serialization
    # ============================================================================
    print("\n=== Example 4: Unit Dictionary Representation ===")
    unit_dict = unit.to_dict()
    print(f"Unit dictionary keys: {list(unit_dict.keys())}")
    print(f"Selectivity keys: {list(unit_dict.get('selectivity', {}).keys())}")
    
    return unit, top_units


if __name__ == "__main__":
    example_single_unit_analysis()

