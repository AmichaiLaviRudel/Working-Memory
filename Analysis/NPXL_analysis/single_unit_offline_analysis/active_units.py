"""
Active unit detection functions for NPXL offline analysis.
"""
from typing import Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from scipy.stats import ttest_rel


def find_active_units_by_midpoint(
    data: np.ndarray,
    alpha: float = 0.05,
    before_range: tuple[int, int] | None = None,
    after_range: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find units that are differentially active before vs after in specified time windows
    relative to the temporal midpoint.

    `before_range` and `after_range` are (start_offset, end_offset) in time‑bins
    **relative to the midpoint** (negative = before mid, positive = after mid).
    Example: before_range = (-200, 0), after_range = (0, 200).

    Uses a paired t‑test across trials for mean firing rate before/after and keeps only
    units where AFTER > BEFORE (directional effect).
    """
    # data shape: [units, time, trials]
    n_units, n_time, n_trials = data.shape

    mid = n_time // 2

    # Default: symmetric windows around midpoint if explicit ranges are not given
    if before_range is None or after_range is None:
        before_range = (-mid, 0)
        after_range = (0, mid)

    b_off_start, b_off_end = before_range
    a_off_start, a_off_end = after_range

    # Convert relative offsets to absolute indices
    b_start = mid + b_off_start
    b_end = mid + b_off_end
    a_start = mid + a_off_start
    a_end = mid + a_off_end

    # Clamp indices to valid range
    b_start = max(0, min(b_start, n_time))
    b_end = max(b_start + 1, min(b_end, n_time))
    a_start = max(0, min(a_start, n_time))
    a_end = max(a_start + 1, min(a_end, n_time))

    # Mean over time within each window, keep per‑trial values
    before = data[:, b_start:b_end, :].mean(axis=1)  # shape: [units, trials]
    after = data[:, a_start:a_end, :].mean(axis=1)   # shape: [units, trials]

    t_vals = np.empty(n_units)
    p_vals = np.empty(n_units)
    mean_before = before.mean(axis=1)
    mean_after = after.mean(axis=1)

    for u in range(n_units):
        t_stat, p_val = ttest_rel(before[u, :], after[u, :])
        t_vals[u] = t_stat
        p_vals[u] = p_val

    # Require statistical difference AND after > before
    active_mask = (p_vals < alpha) & (mean_after > mean_before)
    active_units = np.where(active_mask)[0]

    print(
        f"Found {active_units.size} active units (after > before, p < {alpha}) "
        f"out of {n_units}"
    )
    # Sort active_units by their smallest p_val (ascending)
    sorted_indices = np.argsort(p_vals[active_units])
    active_units_sorted = active_units[sorted_indices]
    t_vals_sorted = t_vals[active_units_sorted]
    p_vals_sorted = p_vals[active_units_sorted]
    return active_units_sorted, t_vals_sorted, p_vals_sorted


def align_event_windows_to_offsets(
    event_windows_data: tuple,
    event_offsets_sec: np.ndarray,
    bin_size_sec: float,
    *,
    fill_value: float = np.nan,
    event_label: str = "event",
    margin_bins: Optional[Tuple[int, int]] = None,
) -> tuple[tuple, np.ndarray]:
    """
    Align per-trial event windows so that a secondary event (e.g., lick/outcome)
    occurs at time zero (the midpoint).

    Parameters
    ----------
    event_windows_data : tuple
        Event windows tuple as returned by load_event_windows_data (5- or 6-tuple).
    event_offsets_sec : np.ndarray
        Per-trial offsets (seconds) from tone onset to the target event.
    bin_size_sec : float
        Duration of a single bin in seconds.
    fill_value : float
        Value used to pad when shifting (np.nan keeps averages unbiased).
    event_label : str
        Label recorded in metadata to mark the alignment source.
    margin_bins : (int, int), optional
        Minimum pre/post bins required to keep a trial (pre, post).

    Returns
    -------
    tuple : aligned event_windows_data (same shape as input)
    np.ndarray : boolean mask indicating which events were retained during alignment
    """
    if len(event_windows_data) == 6:
        (
            event_windows_matrix,
            time_axis,
            valid_event_indices,
            stimuli_outcome_df,
            metadata,
            lick_event_windows_matrix,
        ) = event_windows_data
    else:
        (
            event_windows_matrix,
            time_axis,
            valid_event_indices,
            stimuli_outcome_df,
            metadata,
        ) = event_windows_data
        lick_event_windows_matrix = None

    n_units, n_time, n_events = event_windows_matrix.shape
    aligned_matrix = np.full((n_units, n_time, n_events), fill_value, dtype=float)
    aligned_lick = (
        np.full_like(lick_event_windows_matrix, fill_value, dtype=float)
        if lick_event_windows_matrix is not None
        else None
    )

    mid = n_time // 2
    pre_needed, post_needed = margin_bins or (0, 0)
    valid_events_mask = np.zeros(n_events, dtype=bool)

    for idx, offset_sec in enumerate(event_offsets_sec):
        if np.isnan(offset_sec):
            continue

        shift_bins = int(np.round(offset_sec / bin_size_sec))
        center_idx = mid + shift_bins

        if (
            abs(shift_bins) >= n_time
            or center_idx - pre_needed < 0
            or center_idx + post_needed >= n_time
        ):
            continue  # insufficient data around the event to align

        src = event_windows_matrix[:, :, idx]

        if shift_bins > 0:
            aligned_matrix[:, :-shift_bins, idx] = src[:, shift_bins:]
            if aligned_lick is not None:
                aligned_lick[:, :-shift_bins, idx] = lick_event_windows_matrix[:, shift_bins:, idx]
        elif shift_bins < 0:
            aligned_matrix[:, -shift_bins:, idx] = src[:, :shift_bins]
            if aligned_lick is not None:
                aligned_lick[:, -shift_bins:, idx] = lick_event_windows_matrix[:, :shift_bins, idx]
        else:
            aligned_matrix[:, :, idx] = src
            if aligned_lick is not None:
                aligned_lick[:, :, idx] = lick_event_windows_matrix[:, :, idx]

        valid_events_mask[idx] = True

    aligned_metadata = dict(metadata)
    aligned_metadata["aligned_to"] = event_label
    aligned_metadata["bin_size"] = str(bin_size_sec)

    if aligned_lick is not None:
        aligned_tuple = (
            aligned_matrix,
            time_axis,
            valid_event_indices,
            stimuli_outcome_df,
            aligned_metadata,
            aligned_lick,
        )
    else:
        aligned_tuple = (
            aligned_matrix,
            time_axis,
            valid_event_indices,
            stimuli_outcome_df,
            aligned_metadata,
        )

    return aligned_tuple, valid_events_mask


def find_event_modulated_units(
    aligned_data: np.ndarray,
    *,
    bin_size_sec: float,
    alpha: float = 0.05,
    before_window: tuple[float, float] = (-0.2, 0.0),
    after_window: tuple[float, float] = (0.0, 0.5),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Identify units modulated around an aligned event (e.g., lick or outcome).

    Parameters
    ----------
    aligned_data : np.ndarray
        Aligned event windows [units, time, trials] where the target event is at the midpoint.
    bin_size_sec : float
        Duration of a single bin in seconds.
    alpha : float
        Significance threshold for the paired t-test.
    before_window : (float, float)
        Window (seconds) before the aligned event.
    after_window : (float, float)
        Window (seconds) after the aligned event.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        active_units, t_values, p_values sorted by significance.
    """
    n_units, n_time, _ = aligned_data.shape
    mid = n_time // 2

    def _to_range(win: tuple[float, float]) -> tuple[int, int]:
        start = int(np.floor(win[0] / bin_size_sec))
        end = int(np.ceil(win[1] / bin_size_sec))
        return start, end

    b_off_start, b_off_end = _to_range(before_window)
    a_off_start, a_off_end = _to_range(after_window)

    b_start = max(0, min(mid + b_off_start, n_time - 1))
    b_end = max(b_start + 1, min(mid + b_off_end, n_time))
    a_start = max(0, min(mid + a_off_start, n_time - 1))
    a_end = max(a_start + 1, min(mid + a_off_end, n_time))

    before = np.nanmean(aligned_data[:, b_start:b_end, :], axis=1)
    after = np.nanmean(aligned_data[:, a_start:a_end, :], axis=1)

    t_vals = np.empty(n_units)
    p_vals = np.empty(n_units)
    mean_before = np.nanmean(before, axis=1)
    mean_after = np.nanmean(after, axis=1)

    for u in range(n_units):
        valid_mask = ~np.isnan(before[u, :]) & ~np.isnan(after[u, :])
        if valid_mask.sum() < 2:
            t_vals[u] = np.nan
            p_vals[u] = 1.0
            continue
        t_stat, p_val = ttest_rel(
            before[u, valid_mask],
            after[u, valid_mask],
            nan_policy="omit",
        )
        t_vals[u] = t_stat
        p_vals[u] = p_val

    active_mask = np.isfinite(p_vals) & (p_vals < alpha) & (mean_after > mean_before)
    active_units = np.where(active_mask)[0]

    sorted_indices = np.argsort(p_vals[active_units])
    active_units_sorted = active_units[sorted_indices]
    t_vals_sorted = t_vals[active_units_sorted]
    p_vals_sorted = p_vals[active_units_sorted]
    return active_units_sorted, t_vals_sorted, p_vals_sorted


def find_action_modulated_units(
    event_windows_data: tuple,
    action_offsets_sec: Optional[np.ndarray],
    *,
    bin_size_sec: float,
    alpha: float = 0.05,
    before_window: tuple[float, float] = (-0.2, 0.0),
    after_window: tuple[float, float] = (0.0, 0.6),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[tuple]]:
    """
    Align to first-lick times and find action-modulated units.
    """
    if action_offsets_sec is None:
        return np.array([]), np.array([]), np.array([]), None

    pre_bins = abs(int(np.floor(before_window[0] / bin_size_sec)))
    post_bins = int(np.ceil(after_window[1] / bin_size_sec))

    aligned_tuple, _ = align_event_windows_to_offsets(
        event_windows_data,
        action_offsets_sec,
        bin_size_sec,
        event_label="first_lick_time",
        margin_bins=(pre_bins, post_bins),
    )
    aligned_matrix = aligned_tuple[0]
    action_units, t_vals, p_vals = find_event_modulated_units(
        aligned_matrix,
        bin_size_sec=bin_size_sec,
        alpha=alpha,
        before_window=before_window,
        after_window=after_window,
    )
    return action_units, t_vals, p_vals, aligned_tuple


def find_outcome_modulated_units(
    event_windows_data: tuple,
    outcome_offsets_sec: Optional[np.ndarray],
    *,
    bin_size_sec: float,
    alpha: float = 0.05,
    before_window: tuple[float, float] = (-0.2, 0.0),
    after_window: tuple[float, float] = (0.0, 0.6),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[tuple]]:
    """
    Align to outcome times and find outcome-modulated units.
    """
    if outcome_offsets_sec is None:
        return np.array([]), np.array([]), np.array([]), None

    pre_bins = abs(int(np.floor(before_window[0] / bin_size_sec)))
    post_bins = int(np.ceil(after_window[1] / bin_size_sec))

    aligned_tuple, _ = align_event_windows_to_offsets(
        event_windows_data,
        outcome_offsets_sec,
        bin_size_sec,
        event_label="outcome_time",
        margin_bins=(pre_bins, post_bins),
    )
    aligned_matrix = aligned_tuple[0]
    outcome_units, t_vals, p_vals = find_event_modulated_units(
        aligned_matrix,
        bin_size_sec=bin_size_sec,
        alpha=alpha,
        before_window=before_window,
        after_window=after_window,
    )
    return outcome_units, t_vals, p_vals, aligned_tuple


def plot_active_units_timecourses(
    data: np.ndarray,
    active_units: np.ndarray,
    metadata: dict,
    *,
    max_units: int = 10,
    target_bin_size_sec: float = 0.01,
    window_ms: tuple[float, float] = (-500.0, 1000.0),
    region_name: str = "Unit",
) -> None:
    """
    Plot mean time‑courses for a subset of active units around the recording midpoint.
    """
    if active_units.size == 0:
        print("No active units to plot.")
        return

    n_units, n_time, _ = data.shape
    orig_bin_size_sec = float(metadata["bin_size"])  # seconds per bin from preprocessing

    # Limit number of plotted units
    n_to_plot = int(min(max_units, active_units.size))

    for unit_idx in active_units[:n_to_plot]:
        if unit_idx < 0 or unit_idx >= n_units:
            continue  # safety check

        unit_trace = data[unit_idx, :, :].mean(axis=1)  # mean over trials, keep time

        # Re‑bin to desired bin size if needed
        if not np.isclose(orig_bin_size_sec, target_bin_size_sec):
            factor = int(round(target_bin_size_sec / orig_bin_size_sec))
            if factor <= 0:
                raise ValueError(
                    f"Invalid binning factor computed from orig_bin_size={orig_bin_size_sec}, "
                    f"target={target_bin_size_sec}"
                )
            n_bins = len(unit_trace) // factor
            unit_binned = (
                unit_trace[: n_bins * factor]
                .reshape(n_bins, factor)
                .mean(axis=1)
            )
            effective_bin_size_sec = orig_bin_size_sec * factor
        else:
            unit_binned = unit_trace
            effective_bin_size_sec = orig_bin_size_sec

        # Time axis (ms) relative to recording midpoint
        mid_bin = n_time // 2
        mid_time_sec = mid_bin * orig_bin_size_sec
        time_axis_ms = (
            np.arange(len(unit_binned)) * effective_bin_size_sec * 1000.0
            - mid_time_sec * 1000.0
        )

        # Window mask in ms
        w_start, w_end = window_ms
        mask = (time_axis_ms >= w_start) & (time_axis_ms <= w_end)
        if not np.any(mask):
            continue

        time_to_plot = time_axis_ms[mask]
        unit_to_plot = unit_binned[mask]

        trace = go.Scatter(
            x=time_to_plot,
            y=unit_to_plot,
            mode="lines",
            name=f"{region_name} Unit {unit_idx} Mean (binned)",
        )
        layout = go.Layout(
            title=f"Example Active {region_name} Unit Avg Response",
            xaxis=dict(
                title=f"Time relative to midpoint (ms, {effective_bin_size_sec * 1000.0:.1f} ms/bin)"
            ),
            yaxis=dict(title="Mean Response (Hz)"),
        )
        fig = go.Figure(data=[trace], layout=layout)
        # fig  # noqa: E305
        # fig.show()

