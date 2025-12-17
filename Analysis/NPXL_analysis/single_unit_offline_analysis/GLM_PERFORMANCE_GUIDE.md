# GLM Performance Interpretation Guide

## Your Current Results

**Pseudo R² = 0.0084** (0.84%)

This is **poor performance** - the model explains less than 1% of the variance in neural activity.

## What Do These Numbers Mean?

### Pseudo R² (McFadden's)
**Range:** 0 to 1 (though rarely exceeds 0.5 for neural data)

| Pseudo R² | Interpretation | What it means |
|-----------|----------------|---------------|
| **< 0.02** | **Very Poor** | Unit not driven by behavioral variables |
| 0.02 - 0.05 | Poor | Weak relationship to behavior |
| 0.05 - 0.10 | Fair | Modest behavioral modulation |
| 0.10 - 0.20 | Good | Clear behavioral encoding |
| 0.20 - 0.30 | Very Good | Strong behavioral encoding |
| **> 0.30** | **Excellent** | Highly task-modulated unit |

### Your Unit (R² = 0.0084)
- Falls in the "Very Poor" category
- The behavioral predictors don't explain this unit's firing
- Possible reasons:
  1. Unit is not task-modulated (responds to other factors)
  2. Wrong time window selected
  3. Missing important predictors
  4. High intrinsic variability/noise

## What Should You Look For?

### In Well-Modulated Units:
- **Pseudo R² > 0.10** (at minimum)
- **Significant predictors** (p < 0.05) for relevant variables
- **Good correlation** between predicted and actual (r > 0.3)
- **Low CV score** (cross-validation deviance)

### Example of Good Performance:
```
Pseudo R²: 0.25
CV Score: 0.85
Significant predictors:
  - stimulus: coef=0.45, p=0.001 ***
  - category_go: coef=0.32, p=0.008 **
  - reward: coef=0.28, p=0.015 *
```

## Why Is Your Unit Performing Poorly?

### Likely Reasons:

1. **Not Task-Modulated**
   - Check the unit's PSTH - does it show any response?
   - Mean firing rate might be too low or high
   - Unit might be noise/artifact

2. **Wrong Time Window**
   - Currently using (0.0, 0.5s) after cue onset
   - Try different windows:
     - Early: (0.0, 0.2s) - sensory response
     - Late: (0.3, 0.6s) - motor/decision
     - Pre-stimulus: (-0.5, 0.0s) - anticipation

3. **Missing Predictors**
   - Lick data might not be loaded properly
   - Trial state information missing
   - Need to extract behavioral data from metadata

## How to Find Well-Modulated Units

### Strategy 1: Screen All Units
Run GLM on all units and filter:
```python
glm_df = fit_glm_for_all_units(units)

# Find well-modulated units
good_units = glm_df[glm_df['glm_pseudo_r2'] > 0.10]
print(f"Found {len(good_units)} / {len(glm_df)} well-modulated units")

# Sort by performance
top_units = glm_df.nlargest(10, 'glm_pseudo_r2')
```

### Strategy 2: Pre-filter by PSTH Response
Only test units that show significant responses:
```python
# Filter for stimulus-selective units
selective_units = [u for u in units if u.compute_selectivity().get('stimulus_selective')]

# Then run GLM on filtered units
glm_df = fit_glm_for_all_units(selective_units)
```

## Typical Performance Distributions

In a typical neural recording session:

### Auditory Cortex (ACx)
- **~20-40%** of units show R² > 0.10
- **~5-15%** show R² > 0.20 (strong encoding)
- Best predictors: **stimulus**, **category_go**

### Orbitofrontal Cortex (OFC)
- **~15-30%** of units show R² > 0.10
- **~10-20%** show R² > 0.20
- Best predictors: **reward**, **prev_trial_reward**, **category_go**

### Expected Range Across Population
```
R² Distribution (example):
  0.00 - 0.05:  60% of units (not task-modulated)
  0.05 - 0.10:  20% of units (weakly modulated)
  0.10 - 0.20:  15% of units (well-modulated)
  0.20 - 0.30:   4% of units (strongly modulated)
  > 0.30:        1% of units (highly task-specific)
```

## Next Steps for Your Analysis

### 1. Check Unit Quality
```python
# Is this unit even responsive?
unit.compute_selectivity()
unit.compute_psth_metrics()

# Plot PSTH to visually inspect
unit.plot_psth_by_stimulus()
```

### 2. Try Different Time Windows
```python
# Test multiple epochs
windows = [
    (0.0, 0.2),    # Early sensory
    (0.2, 0.5),    # Late/decision
    (0.5, 1.0),    # Post-stimulus
    (-0.5, 0.0),   # Pre-stimulus
]

for window in windows:
    glm = fit_glm_for_unit(unit, time_window=window)
    print(f"Window {window}: R² = {glm['pseudo_r2']:.4f}")
```

### 3. Analyze Population
```python
# Process all units
glm_df = fit_glm_for_all_units(units)

# Summary statistics
print(f"Median R²: {glm_df['glm_pseudo_r2'].median():.4f}")
print(f"Units with R² > 0.10: {(glm_df['glm_pseudo_r2'] > 0.10).sum()}")

# Find best units
best_units = glm_df.nlargest(5, 'glm_pseudo_r2')
print("\nTop 5 units:")
print(best_units[['unit_idx', 'glm_pseudo_r2']])
```

## Visualization Guide

### Prediction Plots
Now available with `plot_predicted_vs_actual()`:
- **Top panel**: Predicted (red dashed) vs Actual (blue) firing rates
- **Bottom panel**: Scatter plot with identity line

**What to look for:**
- Good fit: Points cluster near identity line
- Poor fit: Random scatter (like your current unit)
- Correlation value should be > 0.3 for decent fits

### Running the Prediction Plot
The example will now automatically generate:
- `glm_predictions_unit_X.html` - Shows predicted vs actual traces

Look at these plots to visually assess model quality!

## Literature Benchmarks

From published studies on GLM for neural encoding:

- **Pillow et al. (2008), Nature**: R² = 0.15-0.40 for V1 neurons
- **Park et al. (2014), Nature Neuroscience**: R² = 0.10-0.25 for parietal cortex
- **Typical task-encoding studies**: R² = 0.05-0.20 for modulated units

**Key Point:** R² < 0.05 generally indicates the unit is not well-explained by your behavioral model.

## Summary

Your current unit (R² = 0.0084) is **not well-modulated** by the behavioral predictors. This is **normal** - most units in a recording are not strongly task-related.

**Recommended actions:**
1. ✓ Run GLM on ALL units to find well-modulated ones
2. ✓ Look for units with R² > 0.10
3. ✓ Try different time windows
4. ✓ Check which predictors are significant across the population
5. ✓ Use prediction plots to visualize model quality

**Remember:** Finding that most units have low R² is informative - it tells you which neurons ARE encoding task variables!
