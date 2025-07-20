import numpy as np
from scipy.stats import ttest_rel, ttest_ind, wilcoxon, mannwhitneyu
from statsmodels.stats.multitest import multipletests

def compare_conditions_at_points(avg_resp_2bd, avg_resp_1bd, points_of_interest, paired=True, test_type='auto', correction='fdr_bh'):
    """
    Compare average responses between 2-boundary and 1-boundary conditions at specified points.

    Parameters:
    - avg_resp_2bd: np.ndarray, shape (n_subjects, n_points)
    - avg_resp_1bd: np.ndarray, shape (n_subjects, n_points)
    - points_of_interest: list or array of indices to compare
    - paired: bool, whether to use paired tests (default True)
    - test_type: 'auto', 't', or 'nonparametric' (default 'auto')
    - correction: str, multiple comparison correction method (default 'fdr_bh')

    Returns:
    - corrected_pvals: np.ndarray, corrected p-values for each point
    - test_stats: np.ndarray, test statistic for each point
    """
    pvals = []
    stats = []
    for idx in points_of_interest:
        x = avg_resp_2bd[:, idx]
        y = avg_resp_1bd[:, idx]
        # Remove NaNs
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if len(x) == 0 or len(y) == 0:
            pvals.append(np.nan)
            stats.append(np.nan)
            continue
        # Choose test
        if test_type == 'auto':
            if paired and len(x) == len(y):
                # Use paired t-test
                stat, p = ttest_rel(x, y, nan_policy='omit')
            else:
                # Use unpaired t-test
                stat, p = ttest_ind(x, y, nan_policy='omit')
        elif test_type == 't':
            if paired and len(x) == len(y):
                stat, p = ttest_rel(x, y, nan_policy='omit')
            else:
                stat, p = ttest_ind(x, y, nan_policy='omit')
        elif test_type == 'nonparametric':
            if paired and len(x) == len(y):
                try:
                    stat, p = wilcoxon(x, y)
                except ValueError:
                    stat, p = np.nan, np.nan
            else:
                try:
                    stat, p = mannwhitneyu(x, y, alternative='two-sided')
                except ValueError:
                    stat, p = np.nan, np.nan
        else:
            raise ValueError(f"Unknown test_type: {test_type}")
        pvals.append(p)
        stats.append(stat)
    # Multiple comparison correction
    pvals = np.array(pvals)
    stats = np.array(stats)
    if np.all(np.isnan(pvals)):
        corrected_pvals = pvals
    else:
        # Only correct non-nan pvals
        valid = ~np.isnan(pvals)
        corrected = np.full_like(pvals, np.nan)
        if np.any(valid):
            reject, pvals_corr, _, _ = multipletests(pvals[valid], method=correction)
            corrected[valid] = pvals_corr
        corrected_pvals = corrected
    return corrected_pvals, stats
