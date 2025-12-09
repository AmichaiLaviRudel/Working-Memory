"""
Unit class for representing individual neural units.

This class encapsulates unit data, metadata, and provides methods for computing
unit-specific metrics and visualizations.
"""
import os
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any
from Analysis.NPXL_analysis.single_unit_offline_analysis.config import npxl_single_unit_analysis
# Note: compute_selectivity_metrics_for_active_units is not imported here to avoid circular import
# The Unit class implements its own compute_selectivity method
from Analysis.NPXL_analysis.single_unit_offline_analysis.category_analysis import (
    compute_category_sensitivity,
    assign_stimulus_categories,
    plot_psth_by_category,
)
from Analysis.NPXL_analysis.single_unit_offline_analysis.visualization import (
    plot_psth_by_stimulus,
    plot_psth_by_outcome,
    plot_raw_psth,
    plot_unit_heatmap,
    get_trial_statistics,
)


class Unit:
    """
    Represents a single neural unit with its data and computed metrics.
    
    Encapsulates unit-specific data access and provides methods for computing
    selectivity metrics, category sensitivity, and generating visualizations.
    """
    
    def __init__(
        self, 
        unit_idx: int, 
        event_windows_data: tuple,
        region_name: str = "Unknown",
        unit_labels: Optional[pd.DataFrame] = None,
        *,
        aligned_action_data: Optional[tuple] = None,
        aligned_outcome_data: Optional[tuple] = None,
    ):
        """
        Initialize a Unit object.
        
        Parameters:
        -----------
        unit_idx : int
            Index of the unit in the event_windows_matrix
        event_windows_data : tuple
            Event windows data tuple (5 or 6 elements)
        region_name : str
            Name of the brain region (e.g., "ACx", "OFC")
        unit_labels : pd.DataFrame, optional
            DataFrame containing unit labels/metadata
        """
        self.unit_idx = int(unit_idx)
        self.region_name = region_name
        self._event_windows_data = event_windows_data
        self._aligned_action_data = aligned_action_data
        self._aligned_outcome_data = aligned_outcome_data
        self._unit_labels = unit_labels
        
        # Unpack data once
        self._unpack_data()
        
        # Cache for computed metrics
        self._selectivity_metrics: Optional[Dict[str, Any]] = None
        self._category_sensitivity: Optional[Dict[str, Any]] = None
        self._psth_metrics: Optional[Dict[str, Any]] = None
        self._glm_results: Optional[Dict[str, Any]] = None
        self._selectivity_window: Optional[Tuple[float, float]] = None
        self._category_window: Optional[Tuple[float, float]] = None
        self._category_boundaries: Optional[Tuple[float, float]] = None
        self._psth_baseline_window: Optional[Tuple[float, float]] = None
        
        # Store references to generated plots
        # Hybrid approach: store file paths by default, with optional in-memory cache
        # File paths are preferred for memory efficiency and persistence
        self._plot_paths: Dict[str, str] = {}  # Store file paths for saved plots (preferred)
        self._plot_cache: Dict[str, Any] = {}  # Optional in-memory cache for frequently accessed plots
        self._plots_dir: Optional[str] = None  # Base directory for saving plots
    
    def _unpack_data(self):
        """Unpack event windows data once and extract unit-specific data."""
        if len(self._event_windows_data) == 6:
            (self.data, self.time_axis, self.valid_indices, 
             self.stimuli_outcome_df, self.metadata, _) = self._event_windows_data
        else:
            (self.data, self.time_axis, self.valid_indices,
             self.stimuli_outcome_df, self.metadata) = self._event_windows_data
        
        # Extract unit-specific data: [time × events]
        if self.unit_idx >= self.data.shape[0]:
            raise IndexError(
                f"Unit index {self.unit_idx} out of range. "
                f"Data has {self.data.shape[0]} units."
            )
        self.unit_data = self.data[self.unit_idx, :, :]
    
    @property
    def n_trials(self) -> int:
        """Number of trials/events for this unit."""
        return self.unit_data.shape[1]
    
    @property
    def n_time_bins(self) -> int:
        """Number of time bins for this unit."""
        return self.unit_data.shape[0]
    
    @property
    def bin_size(self) -> float:
        """Bin size in seconds."""
        return float(self.metadata.get("bin_size", 0.0))
    
    @property
    def window_duration(self) -> float:
        """Window duration in seconds."""
        return float(self.metadata.get("window_duration", 0.0))
    
    def get_mean_firing_rate(self, window: Optional[Tuple[float, float]] = None) -> float:
        """
        Get mean firing rate for this unit.
        
        Parameters:
        -----------
        window : tuple[float, float], optional
            Time window (start, end) in seconds. If None, uses entire window.
        
        Returns:
        --------
        float
            Mean firing rate in Hz
        """
        if window is None:
            return float(np.mean(self.unit_data))
        
        start_idx = np.argmin(np.abs(self.time_axis - window[0]))
        end_idx = np.argmin(np.abs(self.time_axis - window[1]))
        windowed_data = self.unit_data[start_idx:end_idx, :]
        return float(np.mean(windowed_data))
    
    def compute_selectivity(
        self, 
        window: Tuple[float, float] = (-0.1, 0.5),
        force_recompute: bool = False,
        *,
        aligned_action_data: Optional[tuple] = None,
        aligned_outcome_data: Optional[tuple] = None,
    ) -> Dict[str, Any]:
        """
        Compute selectivity metrics for this unit.
        
        Parameters:
        -----------
        window : tuple[float, float]
            Time window for analysis (start, end) in seconds
        force_recompute : bool
            If True, recompute even if cached
        
        Returns:
        --------
        dict
            Dictionary containing selectivity metrics
        """
        # Check cache
        if (self._selectivity_metrics is not None and 
            not force_recompute and 
            self._selectivity_window == window):
            return self._selectivity_metrics
        
        # Create 5-tuple for analysis functions
        def _to_five_tuple(data_tuple: tuple) -> tuple:
            if len(data_tuple) == 6:
                return (data_tuple[0], data_tuple[1], data_tuple[2], data_tuple[3], data_tuple[4])
            return data_tuple

        if len(self._event_windows_data) == 6:
            event_windows_data_5 = (
                self.data, self.time_axis, self.valid_indices,
                self.stimuli_outcome_df, self.metadata
            )
        else:
            event_windows_data_5 = self._event_windows_data

        action_data = aligned_action_data or self._aligned_action_data
        outcome_data = aligned_outcome_data or self._aligned_outcome_data
        
        results = {}
        
        # Stimulus selectivity
        if 'stimulus' in self.stimuli_outcome_df.columns:
            unique_stimuli, tuning_curve, tuning_sem, best_stimulus = (
                npxl_single_unit_analysis.compute_stimulus_selectivity(
                    event_windows_data_5, self.stimuli_outcome_df, 
                    self.unit_idx, window=window
                )
            )
            # Save the full tuning curve data (stimuli, curve, and SEM)
            if unique_stimuli is not None:
                results["tuning_curve_stimuli"] = list(unique_stimuli)
            else:
                results["tuning_curve_stimuli"] = []
            
            if tuning_curve is not None and len(tuning_curve) > 1:
                max_response = np.max(tuning_curve)
                min_response = np.min(tuning_curve)
                max_sem = np.max(tuning_sem) if len(tuning_sem) > 0 else 0
                stimulus_selective = (max_response - min_response) > (2 * max_sem)
                results["stimulus_selective"] = stimulus_selective
                results["best_stimulus"] = best_stimulus
                results["max_stimulus_response"] = float(max_response)
                results["min_stimulus_response"] = float(min_response)
                results["tuning_curve"] = list(tuning_curve) if tuning_curve is not None else []
                results["tuning_curve_sem"] = list(tuning_sem) if tuning_sem is not None else []
            else:
                results["stimulus_selective"] = False
                results["best_stimulus"] = None
                results["tuning_curve"] = []
                results["tuning_curve_sem"] = []
        else:
            results["stimulus_selective"] = False
            results["best_stimulus"] = None
            results["tuning_curve_stimuli"] = []
            results["tuning_curve"] = []
            results["tuning_curve_sem"] = []
        
        # Outcome modulation (use outcome-aligned data if provided)
        if 'outcome' in self.stimuli_outcome_df.columns:
            outcome_tuple = _to_five_tuple(outcome_data) if outcome_data is not None else event_windows_data_5
            outcome_p, outcome_rates, outcome_means = (
                npxl_single_unit_analysis.compute_outcome_modulation(
                    outcome_tuple, self.stimuli_outcome_df,
                    self.unit_idx, window=window
                )
            )
            if outcome_p is not None:
                results["outcome_p_value"] = float(outcome_p)
                results["outcome_modulated"] = outcome_p < 0.05
                if outcome_means is not None:
                    results["rewarded_mean_rate"] = float(outcome_means[0])
                    results["non_rewarded_mean_rate"] = float(outcome_means[1])
            else:
                results["outcome_p_value"] = np.nan
                results["outcome_modulated"] = False
        else:
            results["outcome_p_value"] = np.nan
            results["outcome_modulated"] = False
        
        # Go/NoGo coding
        if 'outcome' in self.stimuli_outcome_df.columns:
            go_nogo_dprime, go_nogo_roc_auc, go_nogo_rates = (
                npxl_single_unit_analysis.compute_go_nogo_coding(
                    event_windows_data_5, self.stimuli_outcome_df,
                    self.unit_idx, window=window
                )
            )
            if go_nogo_dprime is not None:
                results["go_nogo_dprime"] = float(go_nogo_dprime)
                results["go_nogo_roc_auc"] = float(go_nogo_roc_auc)
                results["go_nogo_selective"] = abs(go_nogo_dprime) > 0.5
            else:
                results["go_nogo_dprime"] = np.nan
                results["go_nogo_roc_auc"] = np.nan
                results["go_nogo_selective"] = False
        else:
            results["go_nogo_dprime"] = np.nan
            results["go_nogo_roc_auc"] = np.nan
            results["go_nogo_selective"] = False
        
        # Choice probability (use action-aligned data if provided)
        if 'outcome' in self.stimuli_outcome_df.columns:
            action_tuple = _to_five_tuple(action_data) if action_data is not None else event_windows_data_5
            cp, cp_corr = npxl_single_unit_analysis.compute_choice_probability(
                action_tuple, self.stimuli_outcome_df,
                self.unit_idx, window=window
            )
            if cp is not None:
                results["choice_probability"] = float(cp)
                results["choice_probability_corr"] = float(cp_corr)
                results["choice_coding"] = abs(cp_corr) > 0.1
            else:
                results["choice_probability"] = np.nan
                results["choice_probability_corr"] = np.nan
                results["choice_coding"] = False
        else:
            results["choice_probability"] = np.nan
            results["choice_probability_corr"] = np.nan
            results["choice_coding"] = False
        
        # Cache results
        self._selectivity_metrics = results
        self._selectivity_window = window
        
        return results
    
    def compute_category_sensitivity(
        self,
        low_boundary: float = 0.983,
        high_boundary: float = 1.525,
        window: Tuple[float, float] = (-0.1, 0.5),
        force_recompute: bool = False
    ) -> Dict[str, Any]:
        """
        Compute category sensitivity for this unit.
        
        Parameters:
        -----------
        low_boundary : float
            Lower category boundary
        high_boundary : float
            Upper category boundary
        window : tuple[float, float]
            Time window for analysis (start, end) in seconds
        force_recompute : bool
            If True, recompute even if cached
        
        Returns:
        --------
        dict
            Dictionary containing category sensitivity metrics
        """
        # Check cache
        if (self._category_sensitivity is not None and 
            not force_recompute and
            self._category_window == window and
            self._category_boundaries == (low_boundary, high_boundary)):
            return self._category_sensitivity
        
        results = compute_category_sensitivity(
            self._event_windows_data,
            self.unit_idx,
            low_boundary=low_boundary,
            high_boundary=high_boundary,
            window=window,
        )
        
        # Cache results
        self._category_sensitivity = results
        self._category_window = window
        self._category_boundaries = (low_boundary, high_boundary)
        
        return results
    
    def plot_psth_by_stimulus(
        self, 
        display_window: Tuple[float, float] = (-0.5, 1.0),
        cache_plot: bool = True
    ):
        """
        Plot PSTH separated by stimulus type for this unit.
        
        Parameters:
        -----------
        display_window : tuple[float, float]
            Time window for display (start, end) in seconds
        cache_plot : bool
            If True, store the plot reference in self._plots
        
        Returns:
        --------
        go.Figure
            Plotly figure
        """
        fig = plot_psth_by_stimulus(
            self._event_windows_data,
            self.unit_idx,
            display_window=display_window,
            region_name=self.region_name
        )
        if cache_plot:
            # Store in memory cache
            self._plot_cache['psth_by_stimulus'] = fig
            # Auto-save if plots directory is set
            if self._plots_dir is not None:
                self.save_plot('psth_by_stimulus', fig, subfolder='psth_by_stimulus', cache_in_memory=True)
        return fig
    
    def plot_psth_by_outcome(
        self,
        display_window: Tuple[float, float] = (-0.5, 1.0),
        cache_plot: bool = True
    ):
        """
        Plot PSTH separated by behavioral outcome for this unit.
        
        Parameters:
        -----------
        display_window : tuple[float, float]
            Time window for display (start, end) in seconds
        cache_plot : bool
            If True, store the plot reference in self._plots
        
        Returns:
        --------
        go.Figure
            Plotly figure
        """
        fig = plot_psth_by_outcome(
            self._event_windows_data,
            self.unit_idx,
            display_window=display_window,
            region_name=self.region_name
        )
        if cache_plot:
            self._plot_cache['psth_by_outcome'] = fig
            if self._plots_dir is not None:
                self.save_plot('psth_by_outcome', fig, subfolder='psth_by_outcome', cache_in_memory=True)
        return fig
    
    def plot_raw_psth(
        self,
        display_window: Tuple[float, float] = (-0.5, 1.0),
        cache_plot: bool = True
    ):
        """
        Plot raw PSTH (all trials averaged) for this unit.
        
        Parameters:
        -----------
        display_window : tuple[float, float]
            Time window for display (start, end) in seconds
        cache_plot : bool
            If True, store the plot reference in self._plots
        
        Returns:
        --------
        go.Figure
            Plotly figure
        """
        fig = plot_raw_psth(
            self._event_windows_data,
            self.unit_idx,
            display_window=display_window,
            region_name=self.region_name
        )
        if cache_plot:
            self._plot_cache['raw_psth'] = fig
            if self._plots_dir is not None:
                self.save_plot('raw_psth', fig, subfolder='raw_psth', cache_in_memory=True)
        return fig
    
    def plot_psth_by_category(
        self,
        low_boundary: float = 0.983,
        high_boundary: float = 1.525,
        display_window: Tuple[float, float] = (-0.5, 1.0),
        cache_plot: bool = True
    ):
        """
        Plot PSTH separated by category for this unit.
        
        Parameters:
        -----------
        low_boundary : float
            Lower category boundary
        high_boundary : float
            Upper category boundary
        display_window : tuple[float, float]
            Time window for display (start, end) in seconds
        cache_plot : bool
            If True, store the plot reference in self._plots
        
        Returns:
        --------
        go.Figure
            Plotly figure
        """
        fig = plot_psth_by_category(
            self._event_windows_data,
            self.unit_idx,
            low_boundary=low_boundary,
            high_boundary=high_boundary,
            display_window=display_window,
            region_name=self.region_name
        )
        if cache_plot:
            self._plot_cache['psth_by_category'] = fig
            if self._plots_dir is not None:
                self.save_plot('psth_by_category', fig, subfolder='psth_by_category', cache_in_memory=True)
        return fig
    
    def plot_heatmap(
        self,
        display_window: Tuple[float, float] = (-0.5, 1.0),
        target: str = "tone",
        cache_plot: bool = True
    ):
        """
        Plot heatmap visualization for this unit.
        
        Parameters:
        -----------
        display_window : tuple[float, float]
            Time window for display (start, end) in seconds
        target : str
            Alignment target name (e.g., "tone", "choice", "outcome")
        cache_plot : bool
            If True, store the plot reference in self._plots
        
        Returns:
        --------
        go.Figure
            Plotly figure with heatmap
        """
        fig = plot_unit_heatmap(
            self._event_windows_data,
            self.unit_idx,
            display_window=display_window,
            region_name=self.region_name
        )
        if cache_plot:
            self._plot_cache['heatmap'] = fig
            if self._plots_dir is not None:
                # Generate filename with target name: {region}_unit_{unit_idx}_{target}_heatmap.html
                filename = f"{self.region_name.lower()}_unit_{self.unit_idx}_{target}_heatmap.html"
                
                # Use target-specific subfolder (tone_align for tone, {target}_aligned for others)
                subfolder = f"heatmap/{target}_aligned"
                self.save_plot('heatmap', fig, subfolder=subfolder, filename=filename, cache_in_memory=True)
        return fig
    
    def get_trial_stats(self) -> Dict[str, int]:
        """
        Get trial statistics for this unit.
        
        Returns:
        --------
        dict
            Dictionary with trial counts for each outcome type
        """
        return get_trial_statistics(self._event_windows_data, self.unit_idx)
    
    def set_plots_directory(self, plots_dir: str):
        """
        Set the base directory for saving plots.
        
        Parameters:
        -----------
        plots_dir : str
            Base directory path for saving plots
        """
        self._plots_dir = plots_dir
        os.makedirs(plots_dir, exist_ok=True)
    
    def save_plot(
        self,
        plot_type: str,
        fig: Any,
        subfolder: str = "",
        filename: Optional[str] = None,
        cache_in_memory: bool = False
    ) -> str:
        """
        Save a plot to disk and store the file path.
        
        Parameters:
        -----------
        plot_type : str
            Type of plot ('psth_by_stimulus', 'psth_by_outcome', etc.)
        fig : go.Figure
            Plotly figure to save
        subfolder : str
            Subfolder within plots directory (e.g., 'psth_by_stimulus')
        filename : str, optional
            Custom filename. If None, auto-generates based on unit_idx and plot_type
        cache_in_memory : bool
            If True, also keep figure in memory cache for fast access
        
        Returns:
        --------
        str
            Path to saved plot file
        """
        if self._plots_dir is None:
            raise ValueError("Plots directory not set. Call set_plots_directory() first.")
        
        # Generate filename if not provided
        if filename is None:
            filename = f"{self.region_name.lower()}_unit_{self.unit_idx}_{plot_type}.html"
        
        # Create subfolder if specified
        if subfolder:
            save_dir = os.path.join(self._plots_dir, subfolder)
            os.makedirs(save_dir, exist_ok=True)
            filepath = os.path.join(save_dir, filename)
        else:
            filepath = os.path.join(self._plots_dir, filename)
        
        # Save plot to disk
        from Analysis.NPXL_analysis.single_unit_offline_analysis.utils import save_plot_to_html
        save_plot_to_html(fig, filepath, description=f"{plot_type} for unit {self.unit_idx}")
        
        # Store file path
        self._plot_paths[plot_type] = filepath
        
        # Optionally cache in memory for fast access
        if cache_in_memory:
            self._plot_cache[plot_type] = fig
        
        return filepath
    
    def load_plot(self, plot_type: str, use_cache: bool = True) -> Optional[Any]:
        """
        Load a plot from disk or memory cache.
        
        Parameters:
        -----------
        plot_type : str
            Type of plot to load
        use_cache : bool
            If True, check memory cache first before loading from disk
        
        Returns:
        --------
        go.Figure or None
            Plotly figure, or None if not found
        """
        # Check memory cache first
        if use_cache and plot_type in self._plot_cache:
            return self._plot_cache[plot_type]
        
        # Load from disk if path exists
        if plot_type in self._plot_paths:
            filepath = self._plot_paths[plot_type]
            if os.path.exists(filepath):
                # Note: Plotly HTML files can be loaded, but for interactive use,
                # you typically want to open them in a browser or use plotly.io.read_json
                # For now, return the filepath - the user can decide how to load it
                return filepath
        
        return None
    
    @property
    def plot_paths(self) -> Dict[str, str]:
        """
        Get dictionary of plot file paths.
        
        Returns:
        --------
        dict
            Dictionary mapping plot types to file paths
        """
        return self._plot_paths.copy()
    
    @property
    def plots(self) -> Dict[str, Any]:
        """
        Get dictionary of cached plots (in-memory cache).
        
        Returns:
        --------
        dict
            Dictionary containing cached plot figures
        """
        return self._plot_cache.copy()
    
    def get_plot(self, plot_type: str, load_from_disk: bool = False) -> Optional[Any]:
        """
        Get a specific plot by type.
        
        Parameters:
        -----------
        plot_type : str
            Type of plot ('psth_by_stimulus', 'psth_by_outcome', 
                         'raw_psth', 'psth_by_category', 'heatmap')
        load_from_disk : bool
            If True and plot not in cache, attempt to load from disk
        
        Returns:
        --------
        go.Figure, str, or None
            Cached plot figure, file path, or None if not found
        """
        # Check memory cache first
        if plot_type in self._plot_cache:
            return self._plot_cache[plot_type]
        
        # Return file path if available
        if plot_type in self._plot_paths:
            if load_from_disk:
                return self.load_plot(plot_type, use_cache=False)
            return self._plot_paths[plot_type]
        
        return None
    
    def compute_psth_metrics(
        self,
        baseline_window: Tuple[float, float] = (-0.5, 0),
        force_recompute: bool = False
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive PSTH metrics for this unit.
        
        Parameters:
        -----------
        baseline_window : tuple[float, float]
            Time window for baseline calculation (start, end) in seconds
        force_recompute : bool
            If True, recompute even if cached
        
        Returns:
        --------
        dict
            Dictionary containing PSTH metrics:
            - onset_latency: Time from event onset to first significant response
            - peak_latency: Time from event onset to peak response
            - response_magnitude: Peak response magnitude relative to baseline
            - response_type: "excitation" or "suppression"
            - fwhm: Full-width at half-maximum of the response
            - rise_time: Time from onset to peak
            - decay_time: Time from peak to return to baseline
            - suppression_metrics: Dict with magnitude, duration, fraction_suppressed
            - trial_variability: Coefficient of variation across trials
            - signal_to_noise: Response magnitude divided by baseline std
            - baseline_rate: Average firing rate during baseline period
            - peak_rate: Peak firing rate during response period
        """
        # Check cache
        if (self._psth_metrics is not None and 
            not force_recompute and 
            self._psth_baseline_window == baseline_window):
            return self._psth_metrics
        
        # Import the function from single_unit_metrics
        from Analysis.NPXL_analysis.single_unit_offline_analysis.single_unit_metrics import calculate_psth_metrics
        
        # Compute metrics
        metrics = calculate_psth_metrics(
            self.unit_data,
            self.time_axis,
            baseline_window=baseline_window
        )
        
        # Cache results
        self._psth_metrics = metrics
        self._psth_baseline_window = baseline_window
        
        return metrics
    
    def compute_d_prime(
        self,
        condition1: str,
        condition2: str,
        window: Tuple[float, float] = (-0.1, 0.5)
    ) -> Optional[float]:
        """
        Compute d' between two outcome conditions.
        
        Parameters:
        -----------
        condition1 : str
            First condition (e.g., "Hit", "Miss", "False Alarm", "CR")
        condition2 : str
            Second condition (e.g., "Hit", "Miss", "False Alarm", "CR")
        window : tuple[float, float]
            Time window for analysis (start, end) in seconds
        
        Returns:
        --------
        float or None
            d' value, or None if insufficient data
        """
        from Analysis.NPXL_analysis.single_unit_offline_analysis.single_unit_metrics import compute_d_prime
        
        return compute_d_prime(
            self._event_windows_data,
            self.stimuli_outcome_df,
            self.unit_idx,
            condition1,
            condition2,
            window=window
        )
    
    def fit_glm(
        self,
        window: Tuple[float, float] = (-0.1, 0.5),
        force_recompute: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Fit Generalized Linear Model (Poisson regression) to this unit's spike trains.
        
        Parameters:
        -----------
        window : tuple[float, float]
            Time window for analysis (start, end) in seconds
        force_recompute : bool
            If True, recompute even if cached
        
        Returns:
        --------
        dict or None
            Dictionary containing:
            - coefficients: Array of GLM coefficients
            - intercept: Intercept term
            - r_squared: R-squared value
            - y_pred: Predicted firing rates
            - y_actual: Actual firing rates
            Or None if GLM fitting failed
        """
        # Check cache (simplified - could add window to cache key)
        if self._glm_results is not None and not force_recompute:
            return self._glm_results
        
        from Analysis.NPXL_analysis.single_unit_offline_analysis.single_unit_metrics import fit_glm_single_unit
        
        result = fit_glm_single_unit(
            self._event_windows_data,
            self.stimuli_outcome_df,
            self.unit_idx,
            window=window
        )
        
        if result is None or result[0] is None:
            return None
        
        coefficients, (intercept, r_squared), y_pred, y_actual = result
        
        glm_results = {
            'coefficients': coefficients,
            'intercept': intercept,
            'r_squared': r_squared,
            'y_pred': y_pred,
            'y_actual': y_actual,
            'feature_names': ['Stimulus', 'Trial Type (Go=1)', 'Outcome (Hit=1)']
        }
        
        # Cache results
        self._glm_results = glm_results
        
        return glm_results
    
    def compute_peri_event_rate(
        self,
        window: Tuple[float, float] = (-0.1, 0.5)
    ) -> float:
        """
        Compute average firing rate around event times for this unit.
        
        Parameters:
        -----------
        window : tuple[float, float]
            Time window around event (start, end) in seconds
        
        Returns:
        --------
        float
            Average firing rate in Hz
        """
        from Analysis.NPXL_analysis.single_unit_offline_analysis.single_unit_metrics import compute_peri_event_rate_from_event_windows
        
        return compute_peri_event_rate_from_event_windows(
            self._event_windows_data,
            self.unit_idx,
            window=window,
            bin_size=self.bin_size
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert unit to dictionary representation.
        
        Returns:
        --------
        dict
            Dictionary containing unit information and metrics
        """
        result = {
            "unit_idx": self.unit_idx,
            "region_name": self.region_name,
            "n_trials": self.n_trials,
            "n_time_bins": self.n_time_bins,
            "bin_size": self.bin_size,
            "window_duration": self.window_duration,
        }
        
        # Add selectivity metrics if computed
        if self._selectivity_metrics is not None:
            result["selectivity"] = self._selectivity_metrics.copy()
            # Remove tuning curves from dict (too large)
            result["selectivity"].pop("tuning_curve", None)
            result["selectivity"].pop("tuning_sem", None)
        
        # Add category sensitivity if computed
        if self._category_sensitivity is not None:
            result["category_sensitivity"] = self._category_sensitivity.copy()
        
        # Add PSTH metrics if computed
        if self._psth_metrics is not None:
            psth_metrics_copy = self._psth_metrics.copy()
            # Flatten suppression_metrics for serialization
            if isinstance(psth_metrics_copy.get('suppression_metrics'), dict):
                supp = psth_metrics_copy['suppression_metrics']
                psth_metrics_copy['suppression_magnitude'] = supp.get('magnitude', 0)
                psth_metrics_copy['suppression_duration'] = supp.get('duration', 0)
                psth_metrics_copy['fraction_suppressed'] = supp.get('fraction_suppressed', 0)
                del psth_metrics_copy['suppression_metrics']
            result["psth_metrics"] = psth_metrics_copy
        
        # Add GLM results if computed (exclude large arrays)
        if self._glm_results is not None:
            glm_copy = self._glm_results.copy()
            # Remove large arrays for serialization
            glm_copy.pop('y_pred', None)
            glm_copy.pop('y_actual', None)
            if 'coefficients' in glm_copy:
                glm_copy['coefficients'] = glm_copy['coefficients'].tolist() if isinstance(glm_copy['coefficients'], np.ndarray) else glm_copy['coefficients']
            result["glm"] = glm_copy
        
        # Add unit labels if available
        if self._unit_labels is not None and self.unit_idx < len(self._unit_labels):
            unit_label_row = self._unit_labels.iloc[self.unit_idx]
            result["unit_labels"] = unit_label_row.to_dict()
        
        return result
    
    def __repr__(self) -> str:
        """String representation of the unit."""
        return (
            f"Unit(unit_idx={self.unit_idx}, region='{self.region_name}', "
            f"n_trials={self.n_trials}, n_time_bins={self.n_time_bins})"
        )


def create_units_from_event_data(
    event_windows_data: tuple,
    unit_indices: np.ndarray,
    region_name: str = "Unknown",
    unit_labels: Optional[pd.DataFrame] = None,
    *,
    aligned_action_data: Optional[tuple] = None,
    aligned_outcome_data: Optional[tuple] = None,
) -> list[Unit]:
    """
    Create a list of Unit objects from event windows data.
    
    Parameters:
    -----------
    event_windows_data : tuple
        Event windows data tuple
    unit_indices : np.ndarray
        Array of unit indices to create
    region_name : str
        Name of the brain region
    unit_labels : pd.DataFrame, optional
        DataFrame containing unit labels/metadata
    
    Returns:
    --------
    list[Unit]
        List of Unit objects
    """
    return [
        Unit(
            idx,
            event_windows_data,
            region_name=region_name,
            unit_labels=unit_labels,
            aligned_action_data=aligned_action_data,
            aligned_outcome_data=aligned_outcome_data,
        )
        for idx in unit_indices
    ]

