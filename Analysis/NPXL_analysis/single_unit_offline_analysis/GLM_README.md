# Generalized Linear Model (GLM) Analysis for Neural Data

This module provides comprehensive GLM fitting functionality for analyzing single-unit neural responses with behavioral predictors.

## Overview

The GLM framework models neural firing rates as a function of multiple behavioral and task-related predictors using a Poisson regression model. This approach allows you to:

1. Identify which behavioral variables drive neural activity
2. Quantify the contribution of each predictor
3. Test for statistical significance of effects
4. Compare predictive power across units and brain regions

## Predictors

The module supports the following predictors:

### 1. **Cue Onset**
- Binary indicator for stimulus presentation timing
- Always present since data is aligned to cue onset

### 2. **Stimulus**
- Continuous variable representing stimulus frequency
- Captures tuning to stimulus properties

### 3. **Category (Go vs NoGo)**
- Binary indicator (1 = Go trial, 0 = NoGo trial)
- Determined by stimulus boundaries or trial outcomes
- Captures task demand differences

### 4. **First Lick**
- Timing of the first lick in each trial (in time bins)
- Captures motor preparation and execution timing

### 5. **In-Trial Lick Count**
- Total number of licks during each trial
- Captures overall licking behavior/effort

### 6. **Reward**
- Binary indicator (1 = Hit trial, 0 = otherwise)
- Captures reward delivery events

### 7. **Punishment**
- Binary indicator (1 = False Alarm trial, 0 = otherwise)
- Captures punishment/error feedback

### 8. **Previous Trial Outcome**
- Binary indicator for reward on previous trial
- Captures history effects and learning

### 9. **Trial States**
- Duration of different trial phases:
  - Wait duration
  - Sound presentation duration
  - Reinforcement delay duration
  - Response window duration
- Captures temporal structure of task

## Installation

No additional installation required. The module uses standard scientific Python packages:
- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`

## Quick Start

### Fit GLM for a Single Unit

```python
from Analysis.NPXL_analysis.single_unit_offline_analysis import fit_glm_for_unit
from Analysis.NPXL_analysis.single_unit_offline_analysis.data_loading import load_full_event_windows_data
from Analysis.NPXL_analysis.single_unit_offline_analysis.unit import create_units_from_event_data

# Load data
event_windows_data = load_full_event_windows_data(data_dir)
units = create_units_from_event_data(event_windows_data, region_name="ACx")

# Select a unit
unit = units[0]

# Fit GLM
glm_results = fit_glm_for_unit(
    unit_data=unit.unit_data,
    time_bins=unit.time_axis,
    stimuli_outcome_df=unit.stimuli_outcome_df,
    bin_size=unit.bin_size,
    time_window=(0.0, 0.5),  # Response window in seconds
    alpha=1.0,  # Regularization strength
    category_boundaries=(0.983, 1.525),  # Boundaries for Go/NoGo classification
)

# View results
print(f"Pseudo R²: {glm_results['pseudo_r2']:.4f}")
for name, coef, pval in zip(
    glm_results['feature_names'],
    glm_results['coefficients'],
    glm_results['p_values']
):
    print(f"{name}: {coef:.4f} (p={pval:.4f})")
```

### Fit GLM for All Units

```python
from Analysis.NPXL_analysis.single_unit_offline_analysis import fit_glm_for_all_units

# Fit GLM for all units
glm_df = fit_glm_for_all_units(
    units=units,
    time_window=(0.0, 0.5),
    alpha=1.0,
    category_boundaries=(0.983, 1.525),  # Boundaries for Go/NoGo classification
    save_to_csv=True,
    output_path="glm_results.csv"
)

# Analyze results
print(f"Mean Pseudo R²: {glm_df['glm_pseudo_r2'].mean():.4f}")
print(f"Units with R² > 0.1: {(glm_df['glm_pseudo_r2'] > 0.1).sum()}")
```

## Advanced Usage

### Custom Predictor Selection

You can selectively include/exclude predictors:

```python
from Analysis.NPXL_analysis.single_unit_offline_analysis.glm_fitting import build_design_matrix

X, y, feature_names = build_design_matrix(
    unit_data=unit.unit_data,
    time_bins=unit.time_axis,
    stimuli=stimuli,
    outcomes=outcomes,
    bin_size=unit.bin_size,
    time_window=(0.0, 0.5),
    category_boundaries=(0.983, 1.525),
    include_cue_onset=True,
    include_stimulus=True,
    include_category=True,  # Include Go/NoGo category
    include_licks=True,
    include_reward_punishment=True,
    include_prev_outcome=True,
    include_trial_states=False,  # Exclude trial states
)
```

### Different Time Windows

Analyze different epochs of the trial:

```python
# Early response (0-200ms)
glm_early = fit_glm_for_unit(..., time_window=(0.0, 0.2))

# Late response (200-500ms)
glm_late = fit_glm_for_unit(..., time_window=(0.2, 0.5))

# Pre-stimulus baseline
glm_baseline = fit_glm_for_unit(..., time_window=(-0.5, 0.0))
```

### Regularization Tuning

Adjust regularization strength to prevent overfitting:

```python
# Weak regularization (more flexible model)
glm_weak = fit_glm_for_unit(..., alpha=0.1)

# Strong regularization (simpler model)
glm_strong = fit_glm_for_unit(..., alpha=10.0)
```

## Output Format

### GLM Results Dictionary

```python
{
    'coefficients': [0.12, -0.05, 0.08, ...],  # Feature coefficients
    'intercept': 2.34,  # Model intercept
    'feature_names': ['cue_onset', 'stimulus', ...],  # Feature names
    'p_values': [0.001, 0.234, 0.012, ...],  # Statistical significance
    'std_errors': [0.03, 0.04, 0.03, ...],  # Standard errors
    'deviance': 123.45,  # Model deviance
    'pseudo_r2': 0.234,  # McFadden's pseudo R²
    'cv_score': 145.67,  # Cross-validation score (lower is better)
    'cv_std': 12.34,  # Cross-validation standard deviation
    'n_features': 8,  # Number of predictors
    'n_trials': 150,  # Number of trials
    'time_window': (0.0, 0.5),  # Time window used
}
```

### DataFrame Output (All Units)

When using `fit_glm_for_all_units()`, the output DataFrame contains:

- `unit_idx`: Unit index
- `region_name`: Brain region
- `glm_pseudo_r2`: Model fit quality (0-1, higher is better)
- `glm_deviance`: Model deviance (lower is better)
- `glm_cv_score`: Cross-validation score
- `glm_intercept`: Model intercept
- `glm_coef_*`: Coefficient for each predictor
- `glm_pval_*`: P-value for each predictor

## Interpretation

### Coefficients
- **Positive**: Predictor increases firing rate
- **Negative**: Predictor decreases firing rate
- **Magnitude**: Strength of the effect

### Pseudo R²
- **0.0**: Model explains no variance (predictors have no effect)
- **0.1**: Weak explanatory power
- **0.2-0.3**: Moderate explanatory power
- **>0.4**: Strong explanatory power

### P-values
- **p < 0.001**: Highly significant (***)
- **p < 0.01**: Very significant (**)
- **p < 0.05**: Significant (*)
- **p ≥ 0.05**: Not significant

### Cross-Validation Score
- Lower scores indicate better generalization
- Compare across models to identify overfitting

## Examples

See `example_glm_usage.py` for complete working examples:

1. **Single Unit Analysis**: Fit and visualize GLM for one unit
2. **Population Analysis**: Fit GLM for all units and generate summary statistics
3. **Visualization**: Create coefficient plots and summary figures

Run the examples:

```bash
python example_glm_usage.py
```

## Technical Details

### Model Specification

The GLM uses a **Poisson regression** model appropriate for spike count data:

```
log(firing_rate) = β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ
```

Where:
- `firing_rate`: Mean firing rate in the specified time window
- `β₀`: Intercept (baseline firing rate)
- `βᵢ`: Coefficient for predictor i
- `Xᵢ`: Predictor i value

### Feature Standardization

Features are standardized (zero mean, unit variance) before fitting to:
- Ensure coefficients are on comparable scales
- Improve numerical stability
- Enable fair comparison of predictor importance

### Regularization

L2 regularization (Ridge) is applied to prevent overfitting:
- Shrinks coefficients toward zero
- Parameter `alpha` controls strength (higher = more regularization)
- Default `alpha=1.0` provides balanced regularization

### Statistical Testing

P-values are computed using the **Wald test**:
- Assumes asymptotic normality of coefficient estimates
- Standard errors derived from Fisher information matrix
- Valid for large sample sizes (typically >30 trials)

### Cross-Validation

5-fold cross-validation is used to assess generalization:
- Data split into 5 folds
- Model trained on 4 folds, tested on 1 fold
- Repeated for all combinations
- Reports mean Poisson deviance across folds

## Troubleshooting

### "Insufficient trials for GLM fitting"
**Solution**: Ensure you have at least 10 trials per unit. GLM requires sufficient data for stable parameter estimation.

### "GLM fitting failed: Singular matrix"
**Solution**: 
- Increase regularization (`alpha` parameter)
- Remove redundant predictors
- Check for constant predictors (no variance)

### Poor model fit (low pseudo R²)
**Possible causes**:
- Unit is not task-modulated
- Wrong time window selected
- Missing important predictors
- High trial-to-trial variability

**Solutions**:
- Try different time windows
- Check unit is responsive (use PSTH analysis first)
- Increase regularization to reduce overfitting

### Very high coefficients
**Cause**: Features may not be standardized properly
**Solution**: Ensure `standardize=True` (default) in `fit_glm_poisson()`

## Integration with Existing Workflow

The GLM module integrates seamlessly with existing analysis:

```python
# Standard analysis workflow
from Analysis.NPXL_analysis.single_unit_offline_analysis import (
    Unit,
    create_units_from_event_data,
    fit_glm_for_all_units,
)
from Analysis.NPXL_analysis.single_unit_offline_analysis.utils import (
    units_to_dataframe,
    save_units_to_csv,
)

# Create units
units = create_units_from_event_data(event_windows_data, region_name="ACx")

# Compute standard metrics
for unit in units:
    unit.compute_selectivity()
    unit.compute_category_sensitivity()
    unit.compute_psth_metrics()

# Add GLM results
glm_df = fit_glm_for_all_units(units)

# Combine with other metrics
all_metrics_df = units_to_dataframe(units)
combined_df = all_metrics_df.merge(glm_df, on=['unit_idx', 'region_name'])

# Save
save_units_to_csv(combined_df, "complete_analysis.csv")
```

## Citation

If you use this GLM module in your research, please cite:

```
[Your publication or lab reference]
```

## References

1. Pillow, J. W., et al. (2008). "Spatio-temporal correlations and visual signalling in a complete neuronal population." Nature, 454(7207), 995-999.

2. Truccolo, W., et al. (2005). "A point process framework for relating neural spiking activity to spiking history, neural ensemble, and extrinsic covariate effects." Journal of Neurophysiology, 93(2), 1074-1089.

3. Park, I. M., et al. (2014). "Encoding and decoding in parietal cortex during sensorimotor decision-making." Nature Neuroscience, 17(10), 1395-1403.
