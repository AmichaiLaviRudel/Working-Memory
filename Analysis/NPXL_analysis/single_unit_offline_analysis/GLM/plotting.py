"""
Plotting functions for GLM visualizations.
"""
import re
import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pynapple as nap
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from colors import (
    REGION_COLORS, FEATURE_COLORS, GRADIENT_COLORS, CONNECTIVITY_CMAP,
    get_feature_color, get_coefficient_colors, PLOTLY_REGION_COLORS, PLOTLY_FEATURE_LABELS
)


# ============================================================================
# Gradient Area Plot Helpers
# ============================================================================

def plot_gradient_area(
    ax, 
    x: np.ndarray, 
    y: np.ndarray, 
    color: str = 'tab:blue', 
    n_layers: int = 15,
    alpha_max: float = 0.4
):
    """
    Plot a gradient-filled area chart.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on
    x : np.ndarray
        X values
    y : np.ndarray
        Y values
    color : str
        Color for the gradient
    n_layers : int
        Number of gradient layers
    alpha_max : float
        Maximum alpha value
    """
    for i in range(n_layers):
        y_bottom = y * (i / n_layers)
        y_top = y * ((i + 1) / n_layers)
        alpha = alpha_max * ((i + 1) / n_layers)
        ax.fill_between(x, y_bottom, y_top, alpha=alpha, color=color, linewidth=0)


def smooth_and_fill(
    t_full: np.ndarray,
    t_raw: np.ndarray,
    values_raw: np.ndarray,
    bin_size: float,
    smooth_std: float = 0.05,
    smooth_window: float = 0.25
) -> np.ndarray:
    """
    Smooth time series and fill to regular grid.
    
    Parameters
    ----------
    t_full : np.ndarray
        Full time grid
    t_raw : np.ndarray
        Raw time points
    values_raw : np.ndarray
        Raw values
    bin_size : float
        Bin size for proximity threshold
    smooth_std : float
        Smoothing standard deviation
    smooth_window : float
        Smoothing window size
        
    Returns
    -------
    np.ndarray
        Filled and smoothed values
    """
    tsd = nap.Tsd(t=t_raw, d=values_raw)
    smooth_tsd = tsd.smooth(std=smooth_std, windowsize=smooth_window)
    
    filled = np.zeros_like(t_full, dtype=float)
    smooth_times = smooth_tsd.t
    smooth_values = smooth_tsd.d
    
    for i, t_val in enumerate(t_full):
        dists = np.abs(smooth_times - t_val)
        if len(dists) > 0 and np.min(dists) < bin_size:
            filled[i] = smooth_values[np.argmin(dists)]
    
    return filled


# ============================================================================
# Single Neuron Plots
# ============================================================================

def plot_actual_vs_predicted(
    ax,
    t_plot: np.ndarray,
    actual: np.ndarray,
    predicted: np.ndarray,
    title: str = "",
    actual_color: str = 'orange',
    pred_color: str = 'green',
    n_gradient_layers: int = 15
):
    """
    Plot actual vs predicted firing rates with gradient fills.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on
    t_plot : np.ndarray
        Time values
    actual : np.ndarray
        Actual firing rates
    predicted : np.ndarray
        Predicted firing rates
    title : str
        Plot title
    actual_color : str
        Color for actual
    pred_color : str
        Color for predicted
    n_gradient_layers : int
        Number of gradient layers
    """
    # Actual gradient
    plot_gradient_area(ax, t_plot, actual, actual_color, n_gradient_layers)
    ax.plot(t_plot, actual, color=actual_color, linewidth=2, alpha=0.7, label='Actual FR')
    
    # Predicted gradient
    plot_gradient_area(ax, t_plot, predicted, pred_color, n_gradient_layers)
    ax.plot(t_plot, predicted, color=pred_color, linewidth=2, alpha=0.7, label='Predicted FR')
    
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Firing rate (Hz)")
    ax.set_title(title, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_spike_count_with_events(
    axes: List,
    neuron_count: nap.Tsd,
    temporal_features: nap.TsdFrame,
    epoch: nap.IntervalSet,
    bin_size: float
):
    """
    Plot spike count and event features.
    
    Parameters
    ----------
    axes : list
        List of 4 axes
    neuron_count : nap.Tsd
        Neuron spike count
    temporal_features : nap.TsdFrame
        Temporal features
    epoch : nap.IntervalSet
        Epoch to plot
    bin_size : float
        Bin size
    """
    # Convert spike count to firing rate and smooth
    neuron_count_restricted = neuron_count.restrict(epoch)
    firing_rate_tsd = nap.Tsd(t=neuron_count_restricted.t, d=neuron_count_restricted.d / bin_size)
    firing_rate_smooth = firing_rate_tsd.smooth(std=0.05, windowsize=0.25)
    
    # Create regular time grid
    t_start = epoch.start[0]
    t_end = epoch.end[0]
    t_full = np.arange(t_start, t_end + bin_size, bin_size)
    
    fr_filled = smooth_and_fill(t_full, firing_rate_smooth.t, firing_rate_smooth.d, bin_size)
    
    # Firing rate plot
    plot_gradient_area(axes[0], t_full, fr_filled, 'tab:blue')
    axes[0].plot(t_full, fr_filled, color='tab:blue', linewidth=2, alpha=0.7, label="Firing Rate")
    axes[0].set_ylabel("Firing Rate (Hz)")
    axes[0].set_title("Spike Count Time Series (Smoothed)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Event plots
    events = [('tone_onset', 'red', 'Tone Onset'),
              ('licks', 'tab:blue', 'Licks'),
              ('outcome_onset', 'green', 'Outcome Onset')]
    
    for i, (col, color, title) in enumerate(events):
        data = temporal_features[col].restrict(epoch)
        axes[i+1].step(data.t, data.d, where="post", color=color, alpha=0.7, linewidth=1.5, label=title)
        axes[i+1].set_ylabel("Event")
        axes[i+1].set_title(title)
        axes[i+1].legend()
        axes[i+1].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel("Time (sec)")


# ============================================================================
# Partial Contribution Plot
# ============================================================================

def plot_partial_contributions(
    coefs: np.ndarray,
    X_columns: List[str],
    temporal_features_columns: List[str],
    categorical_features_columns: List[str],
    figsize: Tuple[int, int] = (8, 10)
) -> plt.Figure:
    """
    Plot partial contribution of each predictor as stacked bar.
    
    Parameters
    ----------
    coefs : np.ndarray
        GLM coefficients
    X_columns : list
        Column names from design matrix
    temporal_features_columns : list
        Temporal feature column names
    categorical_features_columns : list
        Categorical feature column names
    figsize : tuple
        Figure size
        
    Returns
    -------
    plt.Figure
        Figure object
    """
    feature_contributions = {}
    
    # Temporal features
    for temp_feat in temporal_features_columns:
        feature_indices = [i for i, col in enumerate(X_columns) if col.startswith(f"{temp_feat}_basis")]
        if len(feature_indices) > 0:
            feature_contributions[temp_feat] = np.sum(np.abs(coefs[feature_indices]))
    
    # Categorical features
    for cat_feat in categorical_features_columns:
        feature_indices = [i for i, col in enumerate(X_columns) if cat_feat in col and 'basis' in col]
        if len(feature_indices) > 0:
            short_name = cat_feat
            if 'stimulus' in cat_feat.lower():
                stim_match = re.search(r'\[T?\.?([\d.]+)\]', cat_feat)
                if not stim_match:
                    stim_match = re.search(r'([\d.]+)', cat_feat)
                if stim_match:
                    short_name = f'stim:{stim_match.group(1)}'
                else:
                    short_name = 'stimulus'
            else:
                short_name = cat_feat.replace('category_ID[T.', 'cat:').replace('outcome_ID[T.', 'out:').replace('previous_outcome[T.', 'prev:').replace(']', '')
            
            feature_contributions[short_name] = np.sum(np.abs(coefs[feature_indices]))
    
    # Sort and normalize
    sorted_features = sorted(feature_contributions.items(), key=lambda x: x[1], reverse=True)
    feature_names = [name for name, _ in sorted_features]
    contributions = [val for _, val in sorted_features]
    
    total_contribution = sum(contributions) if contributions else 1
    percentages = [(c / total_contribution) * 100 for c in contributions]
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    colors = [get_feature_color(name) for name in feature_names]
    
    bottom = 0
    x_pos = 0
    bar_width = 0.6
    
    for i, (name, pct, color) in enumerate(zip(feature_names, percentages, colors)):
        ax.bar(x_pos, pct, bottom=bottom, width=bar_width, color=color, 
               edgecolor='white', linewidth=1.5)
        
        segment_center = bottom + pct / 2
        
        if pct >= 3:
            ax.text(x_pos, segment_center, f'{pct:.1f}%',
                   ha='center', va='center', fontsize=10, fontweight='bold', color='white')
            ax.text(x_pos + bar_width/2 + 0.15, segment_center, name,
                   ha='left', va='center', fontsize=9, color='black')
        else:
            ax.text(x_pos - bar_width/2 - 0.05, segment_center, f'{pct:.1f}%',
                   ha='right', va='center', fontsize=9, fontweight='bold', color='black')
            ax.text(x_pos + bar_width/2 + 0.15, segment_center, name,
                   ha='left', va='center', fontsize=9, color='black')
        
        bottom += pct
    
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.5, 1.2)
    ax.set_ylabel('Relative Contribution (%)', fontsize=11)
    ax.set_title('Partial Contribution of Each Predictor to GLM\n(Sum of Absolute Coefficients)', 
                fontsize=12, fontweight='bold', pad=20)
    ax.set_xticks([])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=9)
    ax.grid(False, axis='x')
    ax.grid(True, axis='y', alpha=0.2, linestyle='--', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_visible(False)
    
    return fig


# ============================================================================
# Basis Functions and Kernels Plot
# ============================================================================

def plot_basis_and_kernels(
    glm_coefs: np.ndarray,
    X_columns: List[str],
    temporal_features_columns: List[str],
    categorical_features_columns: List[str],
    basis_events,
    basis_categorical,
    basis_history,
    event_window_bins: int,
    event_window_sec: float,
    acausal_window_bins: int,
    acausal_before_sec: float,
    acausal_total_sec: float,
    history_window_bins: int,
    history_window_sec: float,
    n_basis_funcs: int
) -> plt.Figure:
    """
    Plot basis functions and reconstructed kernels for all features.
    
    Returns a matplotlib Figure.
    """
    # Evaluate basis functions
    time_event, basis_kernels_event = basis_events.evaluate_on_grid(event_window_bins)
    time_event_sec = time_event * event_window_sec
    
    time_cat, basis_kernels_cat = basis_categorical.evaluate_on_grid(acausal_window_bins)
    time_cat_sec = time_cat * acausal_total_sec - acausal_before_sec
    
    time_hist, basis_kernels_hist = basis_history.evaluate_on_grid(history_window_bins)
    time_hist_sec = time_hist * history_window_sec
    
    n_temporal = len(temporal_features_columns)
    n_categorical = len(categorical_features_columns)
    n_total_features = n_temporal + n_categorical + 1
    
    fig = plt.figure(figsize=(18, 3 * n_total_features))
    gs = fig.add_gridspec(n_total_features + 2, 3, hspace=0.4, wspace=0.3)
    
    # Row 0: Causal basis functions
    ax_basis_temp = fig.add_subplot(gs[0, :])
    for i in range(n_basis_funcs):
        ax_basis_temp.plot(time_event_sec, basis_kernels_event[:, i], alpha=0.7, linewidth=2, label=f"Basis {i}")
    ax_basis_temp.set_xlabel("Time from event (s)")
    ax_basis_temp.set_ylabel("Basis function value")
    ax_basis_temp.set_title("Temporal Event Basis Functions (Causal - RaisedCosineLog)", fontsize=12, fontweight='bold')
    ax_basis_temp.legend(fontsize=8, ncol=4, loc='upper right')
    ax_basis_temp.axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax_basis_temp.axvline(0, color='red', linewidth=1, linestyle='--', alpha=0.5)
    ax_basis_temp.grid(True, alpha=0.3)
    
    # Row 1: Acausal basis functions
    ax_basis_cat = fig.add_subplot(gs[1, :])
    for i in range(n_basis_funcs):
        ax_basis_cat.plot(time_cat_sec, basis_kernels_cat[:, i], alpha=0.7, linewidth=2, label=f"Basis {i}")
    ax_basis_cat.set_xlabel("Time from event (s)")
    ax_basis_cat.set_ylabel("Basis function value")
    ax_basis_cat.set_title("Categorical Basis Functions (Acausal - RaisedCosineLinear)", fontsize=12, fontweight='bold')
    ax_basis_cat.legend(fontsize=8, ncol=4, loc='upper right')
    ax_basis_cat.axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax_basis_cat.axvline(0, color='red', linewidth=1, linestyle='--', alpha=0.5)
    ax_basis_cat.grid(True, alpha=0.3)
    
    row_idx = 2
    coefs = glm_coefs.flatten()
    
    # Temporal features
    for feature_name in temporal_features_columns:
        feature_idx = [i for i, col in enumerate(X_columns) if feature_name in col and 'basis' in col]
        
        if len(feature_idx) == n_basis_funcs:
            feature_coefs = coefs[np.array(feature_idx)]
            reconstructed_kernel = np.dot(basis_kernels_event, feature_coefs)
            
            _plot_kernel_row(fig, gs, row_idx, time_event_sec, basis_kernels_event,
                           feature_coefs, reconstructed_kernel, feature_name,
                           n_basis_funcs, 'tab:blue', is_temporal=True)
        row_idx += 1
    
    # Categorical features
    for feature_name in categorical_features_columns:
        feature_idx = [i for i, col in enumerate(X_columns) if feature_name in col and 'basis' in col]
        
        if len(feature_idx) == n_basis_funcs:
            feature_coefs = coefs[np.array(feature_idx)]
            reconstructed_kernel = np.dot(basis_kernels_cat, feature_coefs)
            
            short_name = feature_name.replace('category_ID[T.', 'cat:').replace('outcome_ID[T.', 'out:').replace('previous_outcome[T.', 'prev:').replace(']', '').replace('stimulus_ID', 'stimulus')
            
            _plot_kernel_row(fig, gs, row_idx, time_cat_sec, basis_kernels_cat,
                           feature_coefs, reconstructed_kernel, short_name,
                           n_basis_funcs, 'tab:green', is_temporal=False)
        row_idx += 1
    
    # Spike history
    hist_idx = [i for i, col in enumerate(X_columns) if 'spike_history' in col and 'basis' in col]
    if len(hist_idx) == basis_history.n_basis_funcs:
        hist_coefs = coefs[np.array(hist_idx)]
        reconstructed_hist = np.dot(basis_kernels_hist, hist_coefs)
        
        _plot_kernel_row(fig, gs, row_idx, time_hist_sec, basis_kernels_hist,
                       hist_coefs, reconstructed_hist, "Spike History",
                       basis_history.n_basis_funcs, 'tab:red', is_temporal=True)
    
    fig.suptitle("GLM Temporal Kernels: Causal (Temporal) + Acausal (Categorical) + Spike History", 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def _plot_kernel_row(fig, gs, row_idx, time_sec, basis_kernels, feature_coefs, 
                     reconstructed_kernel, feature_name, n_basis, color, is_temporal=True):
    """Helper to plot a single row of kernel visualizations."""
    # Column 1: Weighted basis
    ax1 = fig.add_subplot(gs[row_idx, 0])
    for i in range(n_basis):
        ax1.plot(time_sec, basis_kernels[:, i] * feature_coefs[i], alpha=0.6, linewidth=1.5, label=f"B{i}")
    ax1.set_xlabel("Time (s)", fontsize=9)
    ax1.set_ylabel("Weighted basis", fontsize=9)
    title_prefix = "Temporal" if is_temporal else "Categorical"
    ax1.set_title(f"{title_prefix}: {feature_name}\nWeighted Basis", fontsize=10)
    ax1.axhline(0, color='k', linewidth=0.5, linestyle='--')
    if not is_temporal:
        ax1.axvline(0, color='red', linewidth=0.5, linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=6, ncol=2, loc='best')
    
    # Column 2: Reconstructed kernel
    ax2 = fig.add_subplot(gs[row_idx, 1])
    plot_gradient_area(ax2, time_sec, reconstructed_kernel, color)
    ax2.plot(time_sec, reconstructed_kernel, color=color, linewidth=2, alpha=0.7)
    ax2.set_xlabel("Time (s)", fontsize=9)
    ax2.set_ylabel("Kernel weight", fontsize=9)
    ax2.set_title("Reconstructed Kernel", fontsize=10)
    ax2.axhline(0, color='k', linewidth=0.5, linestyle='--')
    if not is_temporal:
        ax2.axvline(0, color='red', linewidth=0.5, linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    # Column 3: Coefficients
    ax3 = fig.add_subplot(gs[row_idx, 2])
    colors = get_coefficient_colors(feature_coefs, color, 'tab:red' if color != 'tab:red' else 'tab:purple')
    ax3.bar(range(n_basis), feature_coefs, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_xlabel("Basis index", fontsize=9)
    ax3.set_ylabel("Coefficient", fontsize=9)
    ax3.set_title("Basis Coefficients", fontsize=10)
    ax3.set_xticks(range(n_basis))
    ax3.axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax3.grid(True, alpha=0.3, axis='y')


# ============================================================================
# Connectivity Matrix Plots
# ============================================================================

def plot_connectivity_matrix(
    coupling_strength: np.ndarray,
    acx_indices: List[int],
    ofc_indices: List[int],
    region_order: List[int],
    figsize: Tuple[int, int] = (8, 8)
) -> plt.Figure:
    """
    Plot full connectivity matrix.
    
    Parameters
    ----------
    coupling_strength : np.ndarray
        Coupling strength matrix
    acx_indices : list
        Indices of ACx neurons
    ofc_indices : list
        Indices of OFC neurons
    region_order : list
        Order of neurons by region
    figsize : tuple
        Figure size
        
    Returns
    -------
    plt.Figure
        Figure object
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    
    coupling_ordered = coupling_strength[np.ix_(region_order, region_order)]
    aspect_ratio = coupling_ordered.shape[0] / coupling_ordered.shape[1] if coupling_ordered.shape[1] > 0 else 1
    
    im = ax.imshow(coupling_ordered, aspect=aspect_ratio, cmap=CONNECTIVITY_CMAP, vmin=-1, vmax=1)
    ax.set_xlabel('Target Neuron', fontsize=11)
    ax.set_ylabel('Source Neuron', fontsize=11)
    ax.set_title('Full Connectivity Matrix (Grouped by Region, Sorted by Coupling Strength)', 
                fontsize=12, fontweight='bold')
    
    # Add region boundaries
    if len(acx_indices) > 0 and len(ofc_indices) > 0:
        ax.axvline(len(acx_indices) - 0.5, color='black', linewidth=2, linestyle='--')
        ax.axhline(len(acx_indices) - 0.5, color='black', linewidth=2, linestyle='--')
        ax.text(len(acx_indices)/2, -1, 'ACx', ha='center', va='top', fontsize=10, fontweight='bold', color='black')
        ax.text(len(acx_indices) + len(ofc_indices)/2, -1, 'OFC', ha='center', va='top', fontsize=10, fontweight='bold', color='black')
        ax.text(-1, len(acx_indices)/2, 'ACx', ha='right', va='center', fontsize=10, fontweight='bold', color='black', rotation=90)
        ax.text(-1, len(acx_indices) + len(ofc_indices)/2, 'OFC', ha='right', va='center', fontsize=10, fontweight='bold', color='black', rotation=90)
    
    # Colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="1%", pad=0.45)
    cb = plt.colorbar(im, cax=cax, orientation='horizontal')
    cb.set_label('Coupling Strength (Normalized -1 to 1)')
    
    # Center colorbar
    box = ax.get_position()
    cbox = cax.get_position()
    new_width = box.width * 0.5
    cax.set_position([box.x0 + (box.width - new_width) / 2, cbox.y0, new_width, cbox.height])
    
    plt.suptitle('All-to-All Connectivity Matrix', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    return fig


def plot_connectivity_breakdown(
    acx_to_acx: np.ndarray,
    acx_to_ofc: np.ndarray,
    ofc_to_acx: np.ndarray,
    ofc_to_ofc: np.ndarray,
    acx_indices: List[int],
    ofc_indices: List[int],
    figsize: Tuple[int, int] = (16, 8)
) -> plt.Figure:
    """
    Plot connectivity breakdown by region with histograms.
    
    Returns
    -------
    plt.Figure
        Figure object
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3, height_ratios=[3, 1])
    
    # ACx → ACx
    if acx_to_acx.size > 0:
        ax = fig.add_subplot(gs[0, 0])
        im = ax.imshow(acx_to_acx, aspect=1, cmap=CONNECTIVITY_CMAP, vmin=-1, vmax=1)
        ax.set_title('ACx → ACx', fontweight='bold')
        ax.set_xlabel('Target ACx')
        ax.set_ylabel('Source ACx')
        ax.plot([-0.5, len(acx_indices)-0.5], [-0.5, len(acx_indices)-0.5], 'r--', linewidth=1, alpha=0.5)
    
    # ACx → OFC
    if acx_to_ofc.size > 0:
        ax = fig.add_subplot(gs[0, 1])
        im = ax.imshow(acx_to_ofc, aspect=1, cmap=CONNECTIVITY_CMAP, vmin=-1, vmax=1)
        ax.set_title('ACx → OFC', fontweight='bold')
        ax.set_xlabel('Target OFC')
        ax.set_ylabel('Source ACx')
    
    # OFC → ACx
    if ofc_to_acx.size > 0:
        ax = fig.add_subplot(gs[0, 2])
        im = ax.imshow(ofc_to_acx, aspect=1, cmap=CONNECTIVITY_CMAP, vmin=-1, vmax=1)
        ax.set_title('OFC → ACx', fontweight='bold')
        ax.set_xlabel('Target ACx')
        ax.set_ylabel('Source OFC')
    
    # OFC → OFC
    if ofc_to_ofc.size > 0:
        ax = fig.add_subplot(gs[0, 3])
        im = ax.imshow(ofc_to_ofc, aspect=1, cmap=CONNECTIVITY_CMAP, vmin=-1, vmax=1)
        ax.set_title('OFC → OFC', fontweight='bold')
        ax.set_xlabel('Target OFC')
        ax.set_ylabel('Source OFC')
        plt.colorbar(im, ax=ax)
        ax.plot([-0.5, len(ofc_indices)-0.5], [-0.5, len(ofc_indices)-0.5], 'r--', linewidth=1, alpha=0.5)
    
    # Histogram
    ax = fig.add_subplot(gs[1, :])
    within_region = []
    cross_region = []
    
    if acx_to_acx.size > 0:
        within_region.extend(acx_to_acx.flatten())
    if ofc_to_ofc.size > 0:
        within_region.extend(ofc_to_ofc.flatten())
    if acx_to_ofc.size > 0:
        cross_region.extend(acx_to_ofc.flatten())
    if ofc_to_acx.size > 0:
        cross_region.extend(ofc_to_acx.flatten())
    
    if len(within_region) > 0 or len(cross_region) > 0:
        all_values = within_region + cross_region
        bins = 50
        bin_range = (np.min(all_values), np.max(all_values))
        
        if len(within_region) > 0:
            within_counts, within_bins = np.histogram(within_region, bins=bins, range=bin_range, density=True)
            within_centers = (within_bins[:-1] + within_bins[1:]) / 2
            plot_gradient_area(ax, within_centers, within_counts, 'tab:blue')
            ax.plot(within_centers, within_counts, color='tab:blue', linewidth=2, alpha=0.8, label='Within Region')
        
        if len(cross_region) > 0:
            cross_counts, cross_bins = np.histogram(cross_region, bins=bins, range=bin_range, density=True)
            cross_centers = (cross_bins[:-1] + cross_bins[1:]) / 2
            plot_gradient_area(ax, cross_centers, cross_counts, 'tab:orange')
            ax.plot(cross_centers, cross_counts, color='tab:orange', linewidth=2, alpha=0.8, label='Cross Region')
        
        ax.set_xlabel('Coupling Strength')
        ax.set_ylabel('Density')
        ax.set_title('Coupling Strength Distribution', fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        stats_text = []
        if len(within_region) > 0:
            stats_text.append(f'Within: μ={np.mean(within_region):.4f}, σ={np.std(within_region):.4f}')
        if len(cross_region) > 0:
            stats_text.append(f'Cross: μ={np.mean(cross_region):.4f}, σ={np.std(cross_region):.4f}')
        
        if stats_text:
            ax.text(0.05, 0.95, '\n'.join(stats_text), transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Connectivity Breakdown by Region', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    return fig


# ============================================================================
# Plotly Visualization Functions
# ============================================================================

def plot_contributions_by_region_plotly(
    grouped_coefs_pct: np.ndarray,
    grouped_feature_names: List[str],
    acx_indices: List[int],
    ofc_indices: List[int],
    output_path: str
):
    """
    Create Plotly half violin plot for contributions by region.
    
    Parameters
    ----------
    grouped_coefs_pct : np.ndarray
        Grouped coefficients as percentages
    grouped_feature_names : list
        Names of grouped features
    acx_indices : list
        Indices of ACx neurons
    ofc_indices : list
        Indices of OFC neurons
    output_path : str
        Path to save HTML file
    """
    features_to_plot = ['stimulus', 'category', 'licks', 'outcome', 'previous_outcome', 'tone_onset', 'outcome_onset']
    features_to_plot = [f for f in features_to_plot if f in grouped_feature_names]
    
    fig = make_subplots(
        rows=1, 
        cols=len(features_to_plot),
        subplot_titles=[PLOTLY_FEATURE_LABELS.get(f, f) for f in features_to_plot],
        shared_yaxes=True,
        horizontal_spacing=0.05
    )
    
    for col_idx, feature in enumerate(features_to_plot):
        feat_idx = grouped_feature_names.index(feature)
        
        acx_data = grouped_coefs_pct[feat_idx, acx_indices].flatten()
        ofc_data = grouped_coefs_pct[feat_idx, ofc_indices].flatten()
        
        if len(acx_data) > 0:
            fig.add_trace(
                go.Violin(
                    y=acx_data,
                    x=[col_idx] * len(acx_data),
                    name='ACx',
                    side='negative',
                    box_visible=True,
                    meanline_visible=True,
                    fillcolor=PLOTLY_REGION_COLORS['ACx'],
                    line_color=PLOTLY_REGION_COLORS['ACx'],
                    opacity=0.6,
                    showlegend=(col_idx == 0),
                    legendgroup='ACx'
                ),
                row=1, col=col_idx+1
            )
        
        if len(ofc_data) > 0:
            fig.add_trace(
                go.Violin(
                    y=ofc_data,
                    x=[col_idx] * len(ofc_data),
                    name='OFC',
                    side='positive',
                    box_visible=True,
                    meanline_visible=True,
                    fillcolor=PLOTLY_REGION_COLORS['OFC'],
                    line_color=PLOTLY_REGION_COLORS['OFC'],
                    opacity=0.6,
                    showlegend=(col_idx == 0),
                    legendgroup='OFC'
                ),
                row=1, col=col_idx+1
            )
    
    fig.update_layout(
        title={
            'text': 'Distribution of Relative Contributions by Region<br><sub>Excluding Spike History</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16, 'family': 'sans-serif'}
        },
        height=600,
        showlegend=True,
        violinmode='overlay',
        violingroupgap=0,
        violingap=0,
        font=dict(family='sans-serif', size=11)
    )
    
    fig.update_yaxes(title_text='Relative Contribution (%)', row=1, col=1)
    
    for col_idx in range(1, len(features_to_plot) + 1):
        fig.update_xaxes(showticklabels=False, row=1, col=col_idx)
    
    fig.write_html(output_path)
    print(f"Plot saved to: {output_path}")


def plot_time_to_react_boxplot(
    time_to_react: Dict[str, np.ndarray],
    acx_indices: List[int],
    ofc_indices: List[int],
    output_path: str
):
    """
    Create matplotlib boxplot for time to react by region.
    
    Parameters
    ----------
    time_to_react : dict
        Dictionary mapping feature names to time arrays
    acx_indices : list
        Indices of ACx neurons
    ofc_indices : list
        Indices of OFC neurons
    output_path : str
        Path to save figure
    """
    features_to_plot = ['stimulus', 'category', 'licks', 'outcome', 'previous_outcome', 'tone_onset', 'outcome_onset']
    features_to_plot = [f for f in features_to_plot if f in time_to_react]
    
    fig, axes = plt.subplots(1, len(features_to_plot), figsize=(5 * len(features_to_plot), 6), sharey=True)
    if len(features_to_plot) == 1:
        axes = [axes]
    
    for col_idx, feature in enumerate(features_to_plot):
        ax = axes[col_idx]
        times_all = time_to_react[feature]
        
        acx_times = times_all[acx_indices]
        ofc_times = times_all[ofc_indices]
        
        acx_times_clean = acx_times[~np.isnan(acx_times)]
        ofc_times_clean = ofc_times[~np.isnan(ofc_times)]
        
        box_data = []
        labels = []
        
        if len(acx_times_clean) > 0:
            box_data.append(acx_times_clean)
            labels.append('ACx')
        if len(ofc_times_clean) > 0:
            box_data.append(ofc_times_clean)
            labels.append('OFC')
        
        if len(box_data) > 0:
            bp = ax.boxplot(box_data, positions=list(range(1, len(box_data) + 1)),
                           widths=0.6, patch_artist=True, showmeans=True, meanline=True)
            
            box_colors = [REGION_COLORS.get(l, '#95A5A6') for l in labels]
            
            for patch, color in zip(bp['boxes'], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
                patch.set_edgecolor(color)
                patch.set_linewidth(2)
            
            for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
                if element in bp:
                    for item in bp[element]:
                        item.set_linewidth(2)
        
        ax.set_title(PLOTLY_FEATURE_LABELS.get(feature, feature), fontsize=12, fontweight='bold')
        ax.set_xticklabels(labels)
        ax.set_ylabel('Time to React (s)' if col_idx == 0 else '')
        ax.grid(True, alpha=0.3, axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.suptitle('Time to React by Feature and Region\n(Time at which maximum absolute coefficient occurs)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Time to react plot saved to: {output_path}")
    
    return fig

