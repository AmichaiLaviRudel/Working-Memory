"""
Advanced Population Analysis Module for Neuropixels Data
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
try:
    from umap import UMAP
except ImportError:
    UMAP = None
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import warnings
warnings.filterwarnings('ignore')

# Import color palettes
from Analysis.GNG_bpod_analysis.colors import GO_COLORS, NOGO_COLORS


class PopulationAnalyzer:
    """
    Main class for advanced population analysis of neural data.
    """
    
    def __init__(self, event_windows_matrix, stimuli_outcome_df, metadata):
        """
        Initialize the analyzer with neural data and metadata.
        
        Args:
            event_windows_matrix: 3D array [units × time_bins × trials]
            stimuli_outcome_df: DataFrame with trial information
            metadata: Dictionary with recording parameters
        """
        self.event_windows_matrix = event_windows_matrix
        self.stimuli_outcome_df = stimuli_outcome_df
        self.metadata = metadata
        
        # Extract key parameters
        self.n_units, self.n_time_bins, self.n_trials = event_windows_matrix.shape
        self.bin_size = float(metadata.get('bin_size', 0.1))
        self.window_duration = float(metadata.get('window_duration', 3.0))
        
        # Create time axis
        self.time_axis = np.arange(-self.window_duration, self.window_duration, self.bin_size)
        
        # Prepare trial labels
        self._prepare_trial_labels()
        
    def _prepare_trial_labels(self):
        """Prepare trial labels for classification analyses."""
        # Stimulus labels
        if 'stimulus' in self.stimuli_outcome_df.columns:
            raw_stimuli = self.stimuli_outcome_df['stimulus'].values
            
            # Store original stimulus values for plotting
            self._original_stimulus_values = np.unique(raw_stimuli)
            
            # Create mapping from label index to actual stimulus value
            self._label_to_stimulus_map = {}
            self._stimulus_to_label_map = {}
            
            # Check if stimuli are continuous (numerical) and convert to categorical
            if np.issubdtype(raw_stimuli.dtype, np.number):
                # For continuous values, create categorical labels
                self.unique_stimuli = np.unique(raw_stimuli)
                
                # If there are too many unique values, bin them
                if len(self.unique_stimuli) > 10:
                    # Prefer binning by stimulus identity (Go/NoGo) to avoid mixing identities
                    if 'outcome' in self.stimuli_outcome_df.columns:
                        outcomes = self.stimuli_outcome_df['outcome'].astype(str).values
                        is_go = np.isin(outcomes, ['Hit', 'Miss'])
                        stim_vals = raw_stimuli
                        unique_vals = np.unique(stim_vals)
                        # Determine identity per stimulus value by majority vote
                        val_to_label = {}
                        for val in unique_vals:
                            mask = stim_vals == val
                            if np.sum(mask) == 0:
                                continue
                            go_fraction = np.mean(is_go[mask])
                            val_to_label[val] = 1 if go_fraction >= 0.5 else 0  # Go=1, NoGo=0
                        # Map trials to identity labels
                        self.stimulus_labels = np.array([val_to_label.get(val, -1) for val in stim_vals])
                        valid_mask = self.stimulus_labels >= 0
                        if not np.any(valid_mask):
                            # Fallback to unique mapping if identity failed
                            label_map = {val: i for i, val in enumerate(unique_vals)}
                            self.stimulus_labels = np.array([label_map[val] for val in raw_stimuli])
                            self.unique_stimuli = np.arange(len(unique_vals))
                            for i, val in enumerate(unique_vals):
                                self._label_to_stimulus_map[i] = val
                                self._stimulus_to_label_map[val] = i
                        else:
                            # Keep only valid trials for downstream decoders that might use labels
                            # Note: we still store full labels; decoders filter trials as needed
                            present_labels = np.unique(self.stimulus_labels[self.stimulus_labels >= 0])
                            self.unique_stimuli = present_labels
                            # Map label indices to identity strings for readability
                            self._label_to_stimulus_map = {0: 'NoGo', 1: 'Go'}
                            # Stimulus to label map cannot be strictly defined here due to grouping many values
                            st.info(f"Binned {len(np.unique(raw_stimuli))} continuous stimulus values by identity into {len(present_labels)} categories (Go/NoGo)")
                    else:
                        # Fallback to quantile-based binning if outcomes are not available
                        n_bins = min(10, len(self.unique_stimuli) // 2)
                        quantiles = np.linspace(0, 1, n_bins + 1)
                        bins = np.quantile(raw_stimuli, quantiles)
                        bins = np.unique(bins)
                        n_bins = len(bins) - 1
                        if n_bins < 2:
                            unique_vals = np.unique(raw_stimuli)
                            label_map = {val: i for i, val in enumerate(unique_vals)}
                            self.stimulus_labels = np.array([label_map[val] for val in raw_stimuli])
                            self.unique_stimuli = np.arange(len(unique_vals))
                            for i, val in enumerate(unique_vals):
                                self._label_to_stimulus_map[i] = val
                                self._stimulus_to_label_map[val] = i
                        else:
                            self.stimulus_labels = np.digitize(raw_stimuli, bins) - 1
                            unique_labels, counts = np.unique(self.stimulus_labels, return_counts=True)
                            min_count = np.min(counts)
                            while min_count < 5 and n_bins > 2:
                                n_bins = n_bins - 1
                                quantiles = np.linspace(0, 1, n_bins + 1)
                                bins = np.unique(np.quantile(raw_stimuli, quantiles))
                                self.stimulus_labels = np.digitize(raw_stimuli, bins) - 1
                                unique_labels, counts = np.unique(self.stimulus_labels, return_counts=True)
                                min_count = np.min(counts)
                            self.unique_stimuli = np.arange(n_bins)
                            for i in range(n_bins):
                                if i < len(bins) - 1:
                                    bin_center = (bins[i] + bins[i+1]) / 2
                                    self._label_to_stimulus_map[i] = bin_center
                        st.info(f"Binned {len(np.unique(raw_stimuli))} continuous stimulus values into {len(self.unique_stimuli)} categories")
                else:
                    # Convert to integer labels for discrete classification
                    unique_vals = np.unique(raw_stimuli)
                    label_map = {val: i for i, val in enumerate(unique_vals)}
                    self.stimulus_labels = np.array([label_map[val] for val in raw_stimuli])
                    self.unique_stimuli = np.arange(len(unique_vals))
                    
                    # Create mappings
                    for i, val in enumerate(unique_vals):
                        self._label_to_stimulus_map[i] = val
                        self._stimulus_to_label_map[val] = i
            else:
                # Already categorical
                unique_vals = np.unique(raw_stimuli)
                label_map = {val: i for i, val in enumerate(unique_vals)}
                self.stimulus_labels = np.array([label_map[val] for val in raw_stimuli])
                self.unique_stimuli = np.arange(len(unique_vals))
                
                # Create mappings
                for i, val in enumerate(unique_vals):
                    self._label_to_stimulus_map[i] = val
                    self._stimulus_to_label_map[val] = i
        else:
            self.stimulus_labels = None
            self.unique_stimuli = None
            self._label_to_stimulus_map = {}
            self._stimulus_to_label_map = {}
            
        # Outcome labels (Go/NoGo based on trial types)
        if 'outcome' in self.stimuli_outcome_df.columns:
            outcomes = self.stimuli_outcome_df['outcome'].values
            # Convert to Go/NoGo
            self.choice_labels = np.array(['Go' if outcome in ['Hit', 'Miss'] else 'NoGo' 
                                         for outcome in outcomes])
        else:
            self.choice_labels = None
            
        # Reward labels
        if 'outcome' in self.stimuli_outcome_df.columns:
            outcomes = self.stimuli_outcome_df['outcome'].values
            self.reward_labels = np.array([outcome in ['Hit'] for outcome in outcomes])
        else:
            self.reward_labels = None
        
        # Create Go/NoGo stimulus mapping
        self._create_go_nogo_stimulus_mapping()
    
    def _create_go_nogo_stimulus_mapping(self):
        """Create mapping between stimuli and their Go/NoGo status."""
        if self.stimulus_labels is not None and self.choice_labels is not None:
            # Create mapping from stimulus to Go/NoGo status
            self.stimulus_go_nogo_map = {}
            for stim in np.unique(self.stimulus_labels):
                stim_mask = self.stimulus_labels == stim
                stim_choices = self.choice_labels[stim_mask]
                
                # Determine if this stimulus is primarily Go or NoGo
                go_count = np.sum(stim_choices == 'Go')
                nogo_count = np.sum(stim_choices == 'NoGo')
                
                if go_count > nogo_count:
                    self.stimulus_go_nogo_map[stim] = 'Go'
                elif nogo_count > go_count:
                    self.stimulus_go_nogo_map[stim] = 'NoGo'
                else:
                    # If equal, use the first choice as tiebreaker
                    self.stimulus_go_nogo_map[stim] = stim_choices[0] if len(stim_choices) > 0 else 'Unknown'
        else:
            self.stimulus_go_nogo_map = {}
    
    def extract_population_activity(self, time_window=None, average_trials=True):
        """
        Extract population activity matrix for analysis.
        
        Args:
            time_window: tuple (start, end) in seconds relative to event
            average_trials: whether to average across trials
            
        Returns:
            activity_matrix: [trials × units] or [trials × units × time] depending on average_trials
        """
        if time_window is None:
            # Use entire window
            start_idx, end_idx = 0, self.n_time_bins
        else:
            # Convert time window to indices
            start_time, end_time = time_window
            start_idx = np.argmin(np.abs(self.time_axis - start_time))
            end_idx = np.argmin(np.abs(self.time_axis - end_time))
        
        # Extract the specified time window
        windowed_data = self.event_windows_matrix[:, start_idx:end_idx, :]
        
        if average_trials:
            # Average across time bins: [units × time × trials] -> [trials × units]
            activity_matrix = np.mean(windowed_data, axis=1).T  # [trials × units]
        else:
            # Keep time dimension: [units × time × trials] -> [trials × units × time]
            activity_matrix = windowed_data.transpose(2, 0, 1)  # [trials × units × time]
            
        return activity_matrix
    
    def cross_validate_decoder(self, X, y, classifier='svm', cv_folds=5, random_state=42):
        """
        Perform cross-validated decoding analysis.
        
        Args:
            X: feature matrix [trials × features]
            y: labels [trials]
            classifier: 'logistic', 'svm', or 'lda'
            cv_folds: number of cross-validation folds
            random_state: random seed
            
        Returns:
            dict with scores, confusion matrix, and detailed results
        """
        # Initialize classifier
        if classifier == 'logistic':
            clf = LogisticRegression(random_state=random_state, max_iter=1000)
        elif classifier == 'svm':
            clf = SVC(random_state=random_state, kernel='linear')
        elif classifier == 'lda':
            clf = LinearDiscriminantAnalysis()
        else:
            raise ValueError(f"Unknown classifier: {classifier}")
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Cross-validation
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        cv_scores = cross_val_score(clf, X_scaled, y, cv=skf, scoring='accuracy')
        
        # Detailed analysis on train/test split
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=random_state, stratify=y
            )
        except ValueError as e:
            # If stratified split fails due to insufficient samples, use regular split
            st.warning(f"Stratified split failed, using regular split: {e}")
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=random_state
            )
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        results = {
            'cv_scores': cv_scores,
            'mean_accuracy': np.mean(cv_scores),
            'std_accuracy': np.std(cv_scores),
            'confusion_matrix': cm,
            'classification_report': class_report,
            'test_accuracy': accuracy_score(y_test, y_pred),
            'feature_importance': getattr(clf, 'coef_', None),
            'classifier': clf,
            'scaler': scaler
        }
        
        return results


class StimulusDecoder(PopulationAnalyzer):
    """Specialized class for stimulus decoding analysis."""
    
    def decode_stimuli(self, time_window=(-0.1, 0.5), classifier='logistic', cv_folds=5, group_by_identity=False):
        """
        Decode stimulus identity from population activity.
        
        Args:
            time_window: time window for analysis (seconds)
            classifier: classifier type
            cv_folds: cross-validation folds
            
        Returns:
            decoding results dictionary
        """
        if self.stimulus_labels is None:
            raise ValueError("No stimulus labels available for decoding")
        
        # Extract activity matrix
        X = self.extract_population_activity(time_window=time_window, average_trials=True)
        y = self.stimulus_labels

        # Optionally group stimuli by identity (Go/NoGo)
        actual_values = None
        if group_by_identity:
            # Build mapping from stimulus label index -> 'Go'/'NoGo'
            stim_to_identity = getattr(self, 'stimulus_go_nogo_map', {})
            if stim_to_identity is None or len(stim_to_identity) == 0:
                raise ValueError("Cannot group by identity: stimulus-to-identity map is missing")
            identities = np.array([stim_to_identity.get(lbl, 'Unknown') for lbl in y])
            # Keep only Go/NoGo and drop Unknown
            valid_mask = (identities == 'Go') | (identities == 'NoGo')
            X = X[valid_mask]
            y = identities[valid_mask]
            # Convert to binary labels
            label_map = {'NoGo': 0, 'Go': 1}
            y = np.array([label_map[val] for val in y], dtype=int)
            actual_values = np.array(['NoGo', 'Go'])
        
        # Remove trials with missing labels
        valid_trials = ~pd.isna(y)
        X = X[valid_trials]
        y = y[valid_trials]
        
        # Validate that we have enough samples for each class
        unique_labels, counts = np.unique(y, return_counts=True)
        min_samples = np.min(counts)
        
        if len(unique_labels) < 2:
            raise ValueError(f"Need at least 2 stimulus classes for decoding, found {len(unique_labels)}")
        
        if min_samples < cv_folds:
            cv_folds = max(2, min_samples)
            st.warning(f"Reduced CV folds to {cv_folds} due to insufficient samples per class (min: {min_samples})")
        
        # Ensure labels are integers for classification
        y = y.astype(int)
        
        # Perform cross-validated decoding
        results = self.cross_validate_decoder(X, y, classifier=classifier, cv_folds=cv_folds)
        
        # Add stimulus-specific information
        if group_by_identity:
            results['unique_stimuli'] = np.array([0, 1])
            results['n_stimuli'] = 2
            results['actual_stimulus_values'] = actual_values
            results['grouping'] = 'identity'
        else:
            results['unique_stimuli'] = self.unique_stimuli
            results['n_stimuli'] = len(self.unique_stimuli)
        results['time_window'] = time_window
        
        # Store actual stimulus values for plotting
        if not group_by_identity and hasattr(self, '_original_stimulus_values'):
            results['actual_stimulus_values'] = self._original_stimulus_values
        
        return results
    
    def time_resolved_stimulus_decoding(self, window_size=0.1, step_size=0.05, classifier='logistic', group_by_identity: bool = False):
        """
        Perform time-resolved stimulus decoding analysis.
        
        Args:
            window_size: size of sliding window (seconds)
            step_size: step size for sliding window (seconds)
            classifier: classifier type
            
        Returns:
            time-resolved decoding accuracies
        """
        if self.stimulus_labels is None:
            raise ValueError("No stimulus labels available for decoding")
        
        # Define time windows
        start_times = np.arange(-self.window_duration, self.window_duration - window_size, step_size)
        accuracies = []
        window_centers = []
        
        for start_time in start_times:
            end_time = start_time + window_size
            window_center = (start_time + end_time) / 2
            
            try:
                results = self.decode_stimuli(
                    time_window=(start_time, end_time), 
                    classifier=classifier, 
                    cv_folds=3,  # Fewer folds for speed
                    group_by_identity=group_by_identity
                )
                accuracies.append(results['mean_accuracy'])
                window_centers.append(window_center)
            except:
                accuracies.append(np.nan)
                window_centers.append(window_center)
        
        return {
            'time_centers': np.array(window_centers),
            'accuracies': np.array(accuracies),
            'window_size': window_size,
            'step_size': step_size
        }


class ChoiceDecoder(PopulationAnalyzer):
    """Specialized class for choice (Go/NoGo) decoding analysis."""
    
    def decode_choice(self, time_window=(-0.1, 0.5), classifier='logistic', cv_folds=5):
        """
        Decode Go/NoGo choice from population activity.
        
        Args:
            time_window: time window for analysis (seconds)
            classifier: classifier type
            cv_folds: cross-validation folds
            
        Returns:
            decoding results dictionary
        """
        if self.choice_labels is None:
            raise ValueError("No choice labels available for decoding")
        
        # Extract activity matrix
        X = self.extract_population_activity(time_window=time_window, average_trials=True)
        y = self.choice_labels
        
        # Remove trials with missing labels
        valid_trials = ~pd.isna(y)
        X = X[valid_trials]
        y = y[valid_trials]
        
        # Validate that we have enough samples for each class
        unique_labels, counts = np.unique(y, return_counts=True)
        min_samples = np.min(counts)
        
        if len(unique_labels) < 2:
            raise ValueError(f"Need at least 2 choice classes for decoding, found {len(unique_labels)}")
        
        if min_samples < cv_folds:
            cv_folds = max(2, min_samples)
            st.warning(f"Reduced CV folds to {cv_folds} due to insufficient samples per class (min: {min_samples})")
        
        # Perform cross-validated decoding
        results = self.cross_validate_decoder(X, y, classifier=classifier, cv_folds=cv_folds)
        
        # Add choice-specific information
        results['choice_types'] = np.unique(y)
        results['time_window'] = time_window
        
        return results
    
    def time_resolved_choice_decoding(self, window_size=0.1, step_size=0.05, classifier='logistic'):
        """
        Perform time-resolved choice decoding analysis.
        
        Args:
            window_size: size of sliding window (seconds)
            step_size: step size for sliding window (seconds)
            classifier: classifier type
            
        Returns:
            time-resolved decoding accuracies
        """
        if self.choice_labels is None:
            raise ValueError("No choice labels available for decoding")
        
        # Define time windows
        start_times = np.arange(-self.window_duration, self.window_duration - window_size, step_size)
        accuracies = []
        window_centers = []
        
        for start_time in start_times:
            end_time = start_time + window_size
            window_center = (start_time + end_time) / 2
            
            try:
                results = self.decode_choice(
                    time_window=(start_time, end_time), 
                    classifier=classifier, 
                    cv_folds=3  # Fewer folds for speed
                )
                accuracies.append(results['mean_accuracy'])
                window_centers.append(window_center)
            except:
                accuracies.append(np.nan)
                window_centers.append(window_center)
        
        return {
            'time_centers': np.array(window_centers),
            'accuracies': np.array(accuracies),
            'window_size': window_size,
            'step_size': step_size
        }


def plot_decoding_results(results, title="Decoding Results", actual_stimulus_values=None):
    """
    Plot decoding analysis results.
    
    Args:
        results: results dictionary from decoding analysis
        title: plot title
        actual_stimulus_values: array of actual stimulus values for labeling confusion matrix
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Cross-Validation Scores', 'Confusion Matrix', 
                       'Feature Importance', 'Classification Report'),
        specs=[[{"type": "scatter"}, {"type": "heatmap"}],
               [{"type": "bar"}, {"type": "table"}]]
    )
    
    # CV scores
    cv_scores = results['cv_scores']
    fig.add_trace(
        go.Scatter(
            x=list(range(len(cv_scores))),
            y=cv_scores,
            mode='markers+lines',
            name='CV Accuracy',
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    
    # Add mean line
    fig.add_hline(
        y=results['mean_accuracy'],
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {results['mean_accuracy']:.3f}",
        row=1, col=1
    )
    
    # Add chance level line
    if 'n_stimuli' in results:
        chance_level = 1.0 / results['n_stimuli']
    elif 'choice_types' in results:
        chance_level = 1.0 / len(results['choice_types'])
    else:
        # Default to binary classification chance level
        chance_level = 0.5
    
    fig.add_hline(
        y=chance_level,
        line_dash="dot",
        line_color="gray",
        annotation_text=f"Chance: {chance_level:.3f}",
        row=1, col=1
    )
    
    # Confusion matrix
    cm = results['confusion_matrix']
    
    # Create labels for confusion matrix axes
    if actual_stimulus_values is not None and len(actual_stimulus_values) == cm.shape[0]:
        # Use actual stimulus values as labels
        if isinstance(actual_stimulus_values[0], (int, float)):
            # Format numbers nicely
            labels = [f"{val:.1f}" if isinstance(val, float) else str(val) for val in actual_stimulus_values]
        else:
            labels = [str(val) for val in actual_stimulus_values]
    else:
        # Use category indices
        labels = [f"Cat {i}" for i in range(cm.shape[0])]
    
    fig.add_trace(
        go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale='Blues',
            showscale=True,
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 12}
        ),
        row=1, col=2
    )
    
    # Feature importance (if available)
    if results['feature_importance'] is not None:
        importance = np.abs(results['feature_importance'].flatten())
        fig.add_trace(
            go.Bar(
                x=list(range(len(importance))),
                y=importance,
                name='Feature Importance'
            ),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        showlegend=False,
        height=800
    )
    
    fig.update_xaxes(title_text="Fold", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", row=1, col=1)
    fig.update_xaxes(title_text="Predicted", row=1, col=2)
    fig.update_yaxes(title_text="True", row=1, col=2)
    fig.update_xaxes(title_text="Unit", row=2, col=1)
    fig.update_yaxes(title_text="Importance", row=2, col=1)
    
    return fig


def plot_time_resolved_decoding(time_results, title="Time-Resolved Decoding", chance_level=None):
    """
    Plot time-resolved decoding results.
    
    Args:
        time_results: results from time-resolved decoding
        title: plot title
        chance_level: chance level for this specific analysis (auto-computed if None)
    """
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=time_results['time_centers'],
            y=time_results['accuracies'],
            mode='lines+markers',
            name='Decoding Accuracy',
            line=dict(color='blue', width=2),
            marker=dict(size=4)
        )
    )
    
    # Add chance level
    if chance_level is None:
        chance_level = 0.5  # Default for binary classification
    
    fig.add_hline(
        y=chance_level,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Chance Level ({chance_level:.1%})"
    )
    
    # Add event onset line
    fig.add_vline(
        x=0,
        line_dash="dash",
        line_color="red",
        annotation_text="Event Onset"
    )
    
    fig.update_layout(
        title=title,
        xaxis_title="Time from Event (s)",
        yaxis_title="Decoding Accuracy",
        yaxis=dict(range=[0, 1]),
        height=400
    )
    
    return fig


class DimensionalityReducer(PopulationAnalyzer):
    """Specialized class for dimensionality reduction analyses."""
    
    def compute_pca(self, time_window=(-0.5, 1.0), n_components=10):
        """
        Perform PCA on population activity.
        
        Args:
            time_window: time window for analysis (seconds)
            n_components: number of PCA components
            
        Returns:
            PCA results dictionary
        """
        # Extract activity matrix [trials × units]
        X = self.extract_population_activity(time_window=time_window, average_trials=True)
        
        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit PCA
        pca = PCA(n_components=min(n_components, X_scaled.shape[1]))
        X_pca = pca.fit_transform(X_scaled)
        
        # Calculate condition-averaged trajectories
        condition_trajectories = {}
        if self.stimulus_labels is not None:
            for stim in np.unique(self.stimulus_labels):
                stim_mask = self.stimulus_labels == stim
                # Get actual stimulus value for labeling
                actual_stim = self._label_to_stimulus_map.get(stim, stim)
                if isinstance(actual_stim, (int, float)):
                    if isinstance(actual_stim, float):
                        stim_label = f'Stimulus_{actual_stim:.1f}'
                    else:
                        stim_label = f'Stimulus_{actual_stim}'
                else:
                    stim_label = f'Stimulus_{actual_stim}'
                condition_trajectories[stim_label] = X_pca[stim_mask].mean(axis=0)
        
        if self.choice_labels is not None:
            for choice in np.unique(self.choice_labels):
                choice_mask = self.choice_labels == choice
                condition_trajectories[f'Choice_{choice}'] = X_pca[choice_mask].mean(axis=0)
        
        return {
            'pca_model': pca,
            'transformed_data': X_pca,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
            'condition_trajectories': condition_trajectories,
            'scaler': scaler,
            'time_window': time_window,
            'trial_labels': {
                'stimulus': self.stimulus_labels,
                'choice': self.choice_labels
            },
            'analyzer': self  # Pass the analyzer object for color mapping
        }
    
    def compute_umap(self, time_window=(-0.5, 1.0), n_components=2, n_neighbors=15, min_dist=0.1):
        """
        Perform UMAP dimensionality reduction.
        
        Args:
            time_window: time window for analysis (seconds)
            n_components: number of UMAP components
            n_neighbors: UMAP n_neighbors parameter
            min_dist: UMAP min_dist parameter
            
        Returns:
            UMAP results dictionary
        """
        # Extract activity matrix [trials × units]
        X = self.extract_population_activity(time_window=time_window, average_trials=True)
        
        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Check if UMAP is available
        if UMAP is None:
            st.error("UMAP is not installed. Please install with: pip install umap-learn")
            return None
        
        # Fit UMAP
        try:
            umap_model = UMAP(n_components=n_components, n_neighbors=n_neighbors, 
                            min_dist=min_dist, random_state=42)
            X_umap = umap_model.fit_transform(X_scaled)
        except Exception as e:
            st.error(f"UMAP failed: {e}")
            return None
        
        return {
            'umap_model': umap_model,
            'transformed_data': X_umap,
            'scaler': scaler,
            'time_window': time_window,
            'trial_labels': {
                'stimulus': self.stimulus_labels,
                'choice': self.choice_labels
            },
            'analyzer': self  # Pass the analyzer object for color mapping
        }


class RepresentationalSimilarityAnalyzer(PopulationAnalyzer):
    """Specialized class for Representational Similarity Analysis (RSA)."""
    
    def compute_rsa(self, time_window=(-0.5, 1.0), metric='correlation'):
        """
        Compute Representational Similarity Analysis.
        
        Args:
            time_window: time window for analysis (seconds)
            metric: distance metric ('correlation', 'euclidean', 'cosine')
            
        Returns:
            RSA results dictionary
        """
        try:
            # Extract activity matrix [trials × units]
            X = self.extract_population_activity(time_window=time_window, average_trials=True)
            
            if X is None or X.shape[0] == 0:
                st.error("No valid activity data found for RSA analysis")
                return None
            
            # Compute condition-averaged responses
            condition_responses = {}
            condition_labels = []
            
            # Group by stimulus if available
            if self.stimulus_labels is not None:
                for stim in np.unique(self.stimulus_labels):
                    mask = self.stimulus_labels == stim
                    if np.sum(mask) > 0:
                        avg_response = X[mask].mean(axis=0)
                        # Get actual stimulus value for labeling
                        actual_stim = self._label_to_stimulus_map.get(stim, stim)
                        if isinstance(actual_stim, (int, float)):
                            # Format numbers nicely
                            if isinstance(actual_stim, float):
                                stim_label = f'Stim_{actual_stim:.1f}'
                            else:
                                stim_label = f'Stim_{actual_stim}'
                        else:
                            stim_label = f'Stim_{actual_stim}'
                        
                        condition_responses[stim_label] = avg_response
                        condition_labels.append(stim_label)
            
            # Group by choice if available
            if self.choice_labels is not None:
                for choice in np.unique(self.choice_labels):
                    mask = self.choice_labels == choice
                    if np.sum(mask) > 0:
                        avg_response = X[mask].mean(axis=0)
                        condition_responses[f'Choice_{choice}'] = avg_response
                        condition_labels.append(f'Choice_{choice}')
            
            if len(condition_responses) < 2:
                st.error("Need at least 2 conditions for RSA")
                return None
            
            # Create condition matrix [conditions × units]
            condition_matrix = np.array([condition_responses[label] for label in condition_labels])
            
            # Compute representational dissimilarity matrix (RDM)
            if metric == 'correlation':
                # 1 - correlation for dissimilarity
                rdm = 1 - np.corrcoef(condition_matrix)
                # For correlation-based RDM, we need to extract the upper triangle for linkage
                # since correlation matrix is already symmetric
                rdm_condensed = rdm[np.triu_indices_from(rdm, k=1)]
            else:
                # For other metrics, use pdist to compute pairwise distances
                rdm_condensed = pdist(condition_matrix, metric=metric)
                # Convert to square form for visualization
                rdm = squareform(rdm_condensed)
            
            # Hierarchical clustering
            linkage_matrix = linkage(rdm_condensed, method='average')
            
            return {
                'rdm': rdm,
                'condition_labels': condition_labels,
                'condition_matrix': condition_matrix,
                'linkage_matrix': linkage_matrix,
                'metric': metric,
                'time_window': time_window,
                'analyzer': self  # Pass the analyzer object for color mapping
            }
            
        except Exception as e:
            st.error(f"Error in RSA computation: {e}")
            return None


class jPCAAnalyzer(PopulationAnalyzer):
    """
    Class for performing jPCA (joint Principal Component Analysis) to identify 
    rotational dynamics in neural population activity.
    """
    
    def __init__(self, event_windows_matrix, stimuli_outcome_df, metadata):
        super().__init__(event_windows_matrix, stimuli_outcome_df, metadata)
        
    def compute_jpca(self, time_window=(-0.5, 1.0), n_components=6, max_skew=0.99):
        """
        Perform jPCA analysis to identify rotational dynamics.
        
        Args:
            time_window: time window for analysis (seconds)
            n_components: number of PCA components to use for jPCA
            max_skew: maximum skew-symmetric component to consider
            
        Returns:
            jPCA results dictionary
        """
        try:
            # Extract time-resolved activity matrix
            X = self.extract_population_activity(time_window=time_window, average_trials=False)
            
            if X.shape[0] < 2 or X.shape[2] < 2:
                raise ValueError("Need at least 2 trials and 2 time points for jPCA")
            
            # Reshape to [trials*time × units]
            n_trials, n_units, n_time = X.shape
            X_reshaped = X.transpose(0, 2, 1).reshape(-1, n_units)
            
            # Apply PCA to reduce dimensionality
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_reshaped)
            
            pca = PCA(n_components=min(n_components, n_units))
            X_pca = pca.fit_transform(X_scaled)
            
            # Reshape back to [trials × time × components]
            X_pca_reshaped = X_pca.reshape(n_trials, n_time, -1)
            
            # Compute cross-covariance matrix
            # Center the data
            X_centered = X_pca_reshaped - np.mean(X_pca_reshaped, axis=1, keepdims=True)
            
            # Compute cross-covariance between consecutive time points
            cross_cov = np.zeros((X_pca_reshaped.shape[2], X_pca_reshaped.shape[2]))
            for t in range(n_time - 1):
                cross_cov += X_centered[:, t, :].T @ X_centered[:, t+1, :]
            cross_cov /= (n_time - 1)
            
            # Find skew-symmetric component
            skew_sym = (cross_cov - cross_cov.T) / 2
            
            # SVD of skew-symmetric matrix
            U, S, Vt = np.linalg.svd(skew_sym)
            
            # Find pairs of components with similar singular values
            jpca_pairs = []
            for i in range(0, len(S)-1, 2):
                if i+1 < len(S):
                    # Check if singular values are similar (within 10%)
                    if abs(S[i] - S[i+1]) / max(S[i], S[i+1]) < 0.1:
                        jpca_pairs.append((i, i+1))
            
            # Project data onto jPCA space
            jpca_projection = U[:, :len(jpca_pairs)*2]
            X_jpca = X_pca @ jpca_projection
            
            # Reshape back to [trials × time × jpca_components]
            X_jpca_reshaped = X_jpca.reshape(n_trials, n_time, -1)
            
            # Compute rotation matrices for each pair
            rotation_matrices = []
            for pair_idx, (i, j) in enumerate(jpca_pairs):
                # Extract the 2D subspace for this pair
                pair_projection = U[:, [i, j]]
                X_pair = X_pca @ pair_projection
                X_pair_reshaped = X_pair.reshape(n_trials, n_time, 2)
                
                # Compute rotation matrix for this pair
                rot_matrix = self._compute_rotation_matrix(X_pair_reshaped)
                rotation_matrices.append(rot_matrix)
            
            # Create time axis
            time_axis = np.linspace(time_window[0], time_window[1], n_time)
            
            return {
                'X_jpca': X_jpca_reshaped,
                'jpca_pairs': jpca_pairs,
                'rotation_matrices': rotation_matrices,
                'singular_values': S,
                'U': U,
                'time_axis': time_axis,
                'pca_components': X_pca_reshaped,
                'explained_variance': pca.explained_variance_ratio_,
                'time_window': time_window,
                'analyzer': self
            }
            
        except Exception as e:
            st.error(f"Error in jPCA computation: {str(e)}")
            return None
    
    def _compute_rotation_matrix(self, X_2d):
        """
        Compute optimal rotation matrix for 2D data.
        
        Args:
            X_2d: 2D data of shape [trials × time × 2]
            
        Returns:
            rotation matrix
        """
        # Compute cross-covariance between consecutive time points
        n_trials, n_time, _ = X_2d.shape
        X_centered = X_2d - np.mean(X_2d, axis=1, keepdims=True)
        
        cross_cov = np.zeros((2, 2))
        for t in range(n_time - 1):
            cross_cov += X_centered[:, t, :].T @ X_centered[:, t+1, :]
        cross_cov /= (n_time - 1)
        
        # Find skew-symmetric component
        skew_sym = (cross_cov - cross_cov.T) / 2
        
        # Compute rotation matrix from skew-symmetric matrix
        # For 2x2 skew-symmetric matrix [[0, -a], [a, 0]], the rotation is exp([[0, -a], [a, 0]])
        a = skew_sym[0, 1]
        theta = np.arctan2(a, 1)  # Rotation angle
        
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        
        return rotation_matrix


def get_go_nogo_colors(n_stimuli, stimulus_go_nogo_map=None):
    """
    Generate color scheme for Go/NoGo stimuli.
    
    Args:
        n_stimuli: number of unique stimuli
        stimulus_go_nogo_map: mapping from stimulus to Go/NoGo status
        
    Returns:
        dict: color mapping for each stimulus
    """
    color_map = {}
    
    if stimulus_go_nogo_map:
        go_stimuli = [s for s, status in stimulus_go_nogo_map.items() if status == 'Go']
        nogo_stimuli = [s for s, status in stimulus_go_nogo_map.items() if status == 'NoGo']
        
        # Assign colors to Go stimuli (green shades)
        for i, stim in enumerate(go_stimuli):
            color_map[stim] = GO_COLORS[i % len(GO_COLORS)]
        
        # Assign colors to NoGo stimuli (red shades)
        for i, stim in enumerate(nogo_stimuli):
            color_map[stim] = NOGO_COLORS[i % len(NOGO_COLORS)]
    else:
        # Fallback: alternate between green and red
        for i in range(n_stimuli):
            if i % 2 == 0:
                color_map[i] = GO_COLORS[i % len(GO_COLORS)]
            else:
                color_map[i] = NOGO_COLORS[i % len(NOGO_COLORS)]
    
    return color_map


def plot_pca_results(pca_results, title="PCA Analysis"):
    """
    Plot PCA analysis results.
    
    Args:
        pca_results: results from compute_pca
        title: plot title
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Explained Variance', 'PC1 vs PC2', 
                       'PC1 vs PC3', 'Condition Trajectories'),
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "scatter"}]]
    )
    
    # Explained variance plot
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(pca_results['explained_variance_ratio']) + 1)),
            y=pca_results['explained_variance_ratio'],
            mode='markers+lines',
            name='Explained Variance',
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    
    # Cumulative variance
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(pca_results['cumulative_variance']) + 1)),
            y=pca_results['cumulative_variance'],
            mode='lines',
            name='Cumulative Variance',
            line=dict(color='red', dash='dash')
        ),
        row=1, col=1
    )
    
    # PC projections
    X_pca = pca_results['transformed_data']
    trial_labels = pca_results['trial_labels']
    
    # Get Go/NoGo color mapping
    stimulus_go_nogo_map = getattr(pca_results.get('analyzer', None), 'stimulus_go_nogo_map', {})
    color_map = get_go_nogo_colors(len(np.unique(trial_labels['stimulus'])), stimulus_go_nogo_map)
    
    # Color by stimulus if available
    if trial_labels['stimulus'] is not None:
        for stim in np.unique(trial_labels['stimulus']):
            mask = trial_labels['stimulus'] == stim
            color = color_map.get(stim, '#808080')  # Default gray if no mapping
            
            # Get actual stimulus value for legend
            actual_stim = pca_results.get('analyzer', {})._label_to_stimulus_map.get(stim, stim)
            if isinstance(actual_stim, (int, float)):
                if isinstance(actual_stim, float):
                    stim_display = f'{actual_stim:.1f}'
                else:
                    stim_display = str(actual_stim)
            else:
                stim_display = str(actual_stim)
            
            # Determine if this is Go or NoGo for legend
            go_nogo_status = stimulus_go_nogo_map.get(stim, 'Unknown')
            legend_name = f'Stimulus {stim_display} ({go_nogo_status})'
            
            fig.add_trace(
                go.Scatter(
                    x=X_pca[mask, 0],
                    y=X_pca[mask, 1],
                    mode='markers',
                    name=legend_name,
                    marker=dict(color=color, size=8),
                    showlegend=True
                ),
                row=1, col=2
            )
            
            if X_pca.shape[1] > 2:
                fig.add_trace(
                    go.Scatter(
                        x=X_pca[mask, 0],
                        y=X_pca[mask, 2],
                        mode='markers',
                        name=legend_name,
                        marker=dict(color=color, size=8),
                        showlegend=False
                    ),
                    row=2, col=1
                )
    
    # Condition trajectories
    if pca_results['condition_trajectories']:
        for i, (condition, trajectory) in enumerate(pca_results['condition_trajectories'].items()):
            # Extract stimulus value from condition name
            if 'Stimulus_' in condition:
                stim_str = condition.split('_', 1)[1]  # Get everything after 'Stimulus_'
                try:
                    stim_num = float(stim_str) if '.' in stim_str else int(stim_str)
                    # Find the corresponding label index for color mapping
                    label_idx = None
                    for idx, actual_stim in pca_results.get('analyzer', {})._label_to_stimulus_map.items():
                        if abs(actual_stim - stim_num) < 1e-6:  # Float comparison
                            label_idx = idx
                            break
                    
                    color = color_map.get(label_idx, '#808080')
                    go_nogo_status = stimulus_go_nogo_map.get(label_idx, 'Unknown')
                    
                    # Format stimulus value for display
                    if isinstance(stim_num, float):
                        stim_display = f'{stim_num:.1f}'
                    else:
                        stim_display = str(stim_num)
                    
                    legend_name = f'{stim_display} kHz ({go_nogo_status})'
                except (ValueError, TypeError):
                    color = '#808080'
                    legend_name = condition
            else:
                color = '#808080'  # Default gray for non-stimulus conditions
                legend_name = condition
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(trajectory))),
                    y=trajectory,
                    mode='lines+markers',
                    name=legend_name,
                    line=dict(color=color, width=2),
                    marker=dict(color=color, size=6)
                ),
                                 row=2, col=2
             )
    
    # Update layout
    fig.update_xaxes(title_text="PC", row=1, col=1)
    fig.update_yaxes(title_text="Variance Explained", row=1, col=1)
    fig.update_xaxes(title_text="PC1", row=1, col=2)
    fig.update_yaxes(title_text="PC2", row=1, col=2)
    fig.update_xaxes(title_text="PC1", row=2, col=1)
    fig.update_yaxes(title_text="PC3", row=2, col=1)
    fig.update_xaxes(title_text="Component", row=2, col=2)
    fig.update_yaxes(title_text="Loading", row=2, col=2)
    fig.update_layout(height=800, title_text=title)
    
    return fig


def plot_umap_results(umap_results, title="UMAP Analysis"):
    """
    Plot UMAP analysis results.
    
    Args:
        umap_results: results from compute_umap
        title: plot title
    """
    if umap_results is None:
        return None
        
    fig = go.Figure()
    
    X_umap = umap_results['transformed_data']
    trial_labels = umap_results['trial_labels']
    
    # Get Go/NoGo color mapping
    stimulus_go_nogo_map = getattr(umap_results.get('analyzer', None), 'stimulus_go_nogo_map', {})
    color_map = get_go_nogo_colors(len(np.unique(trial_labels['stimulus'])), stimulus_go_nogo_map)
    
    # Color by stimulus if available
    if trial_labels['stimulus'] is not None:
        for stim in np.unique(trial_labels['stimulus']):
            mask = trial_labels['stimulus'] == stim
            color = color_map.get(stim, '#808080')  # Default gray if no mapping
            
            # Get actual stimulus value for legend
            actual_stim = umap_results.get('analyzer', {})._label_to_stimulus_map.get(stim, stim)
            if isinstance(actual_stim, (int, float)):
                if isinstance(actual_stim, float):
                    stim_display = f'{actual_stim:.1f}'
                else:
                    stim_display = str(actual_stim)
            else:
                stim_display = str(actual_stim)
            
            # Determine if this is Go or NoGo for legend
            go_nogo_status = stimulus_go_nogo_map.get(stim, 'Unknown')
            legend_name = f'{stim_display} kHz ({go_nogo_status})'
            
            fig.add_trace(
                go.Scatter(
                    x=X_umap[mask, 0],
                    y=X_umap[mask, 1] if X_umap.shape[1] > 1 else np.zeros(np.sum(mask)),
                    mode='markers',
                    name=legend_name,
                    marker=dict(color=color, size=8)
                )
            )
    elif trial_labels['choice'] is not None:
        # Color by choice if no stimulus labels
        for choice in np.unique(trial_labels['choice']):
            mask = trial_labels['choice'] == choice
            fig.add_trace(
                go.Scatter(
                    x=X_umap[mask, 0],
                    y=X_umap[mask, 1] if X_umap.shape[1] > 1 else np.zeros(np.sum(mask)),
                    mode='markers',
                    name=f'Choice {choice}',
                    marker=dict(size=8)
                )
            )
    else:
        # No labels available
        fig.add_trace(
            go.Scatter(
                x=X_umap[:, 0],
                y=X_umap[:, 1] if X_umap.shape[1] > 1 else np.zeros(len(X_umap)),
                mode='markers',
                name='All Trials',
                marker=dict(size=8)
            )
        )
    
        
    
    fig.update_layout(
        title=title,
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2" if X_umap.shape[1] > 1 else "UMAP (1D)",
        height=600
    )
    
    return fig


def plot_rsa_results(rsa_results, title="Representational Similarity Analysis"):
    """
    Plot RSA analysis results.
    
    Args:
        rsa_results: results from compute_rsa
        title: plot title
    """
    if rsa_results is None:
        return None
    
    try:
        # Validate required fields
        required_fields = ['rdm', 'condition_labels', 'linkage_matrix']
        for field in required_fields:
            if field not in rsa_results:
                st.error(f"Missing required field '{field}' in RSA results")
                return None
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Representational Dissimilarity Matrix', 'Hierarchical Clustering'),
            specs=[[{"type": "heatmap"}, {"type": "scatter"}]]
        )
        
        # RDM heatmap
        rdm = rsa_results['rdm']
        condition_labels = rsa_results['condition_labels']
        
        # Get Go/NoGo color mapping for condition labels
        stimulus_go_nogo_map = getattr(rsa_results.get('analyzer', None), 'stimulus_go_nogo_map', {})
        
        # Create color annotations for condition labels
        label_colors = []
        for label in condition_labels:
            if 'Stim_' in label:
                # Extract stimulus value from label
                stim_str = label.split('_', 1)[1]  # Get everything after 'Stim_'
                try:
                    stim_num = float(stim_str) if '.' in stim_str else int(stim_str)
                    # Find the corresponding label index for Go/NoGo mapping
                    label_idx = None
                    for idx, actual_stim in rsa_results.get('analyzer', {})._label_to_stimulus_map.items():
                        if abs(actual_stim - stim_num) < 1e-6:  # Float comparison
                            label_idx = idx
                            break
                    
                    go_nogo_status = stimulus_go_nogo_map.get(label_idx, 'Unknown')
                    if go_nogo_status == 'Go':
                        label_colors.append(GO_COLORS[0])  # First green for Go
                    elif go_nogo_status == 'NoGo':
                        label_colors.append(NOGO_COLORS[0])  # First red for NoGo
                    else:
                        label_colors.append('#808080')  # Gray for unknown
                except (ValueError, TypeError):
                    label_colors.append('#808080')  # Gray for parsing errors
            else:
                label_colors.append('#808080')  # Gray for non-stimulus conditions
        
        fig.add_trace(
            go.Heatmap(
                z=rdm,
                x=condition_labels,
                y=condition_labels,
                colorscale='Viridis',
                showscale=True,
                text=np.round(rdm, 3),
                texttemplate="%{text}",
                textfont={"size": 10}
            ),
            row=1, col=1
        )
        
        # Dendrogram (simplified representation)
        linkage_matrix = rsa_results['linkage_matrix']
        
        # Create a simple dendrogram plot
        n_conditions = len(condition_labels)
        y_positions = np.arange(n_conditions)
        
        # Plot condition labels
        fig.add_trace(
            go.Scatter(
                x=np.zeros(n_conditions),
                y=y_positions,
                mode='markers+text',
                text=condition_labels,
                textposition="middle right",
                marker=dict(size=10),
                showlegend=False
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            height=600,
            title_text=title
        )
        
        fig.update_xaxes(title_text="Conditions", row=1, col=1)
        fig.update_yaxes(title_text="Conditions", row=1, col=1)
        fig.update_xaxes(title_text="Distance", row=1, col=2)
        fig.update_yaxes(title_text="Conditions", row=1, col=2)
        
        return fig
        
    except Exception as e:
        st.error(f"Error plotting RSA results: {e}")
        return None


def plot_jpca_results(jpca_results, title="jPCA Analysis"):
    """
    Plot jPCA analysis results.
    
    Args:
        jpca_results: results from compute_jpca
        title: plot title
    """
    if jpca_results is None:
        return None
    
    try:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Singular Values', 'jPCA Trajectories', 
                           'Rotation Matrices', 'Phase Space'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "heatmap"}, {"type": "scatter"}]]
        )
        
        # 1. Singular values
        singular_values = jpca_results['singular_values']
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(singular_values) + 1)),
                y=singular_values,
                mode='lines+markers',
                name='Singular Values',
                line=dict(color='blue', width=2),
                marker=dict(size=6)
            ),
            row=1, col=1
        )
        
        # Highlight jPCA pairs
        jpca_pairs = jpca_results['jpca_pairs']
        for i, (idx1, idx2) in enumerate(jpca_pairs):
            fig.add_trace(
                go.Scatter(
                    x=[idx1+1, idx2+1],
                    y=[singular_values[idx1], singular_values[idx2]],
                    mode='markers',
                    name=f'jPCA Pair {i+1}',
                    marker=dict(size=10, color='red', symbol='diamond'),
                    showlegend=True
                ),
                row=1, col=1
            )
        
        # 2. jPCA trajectories (first pair)
        if len(jpca_pairs) > 0:
            X_jpca = jpca_results['X_jpca']
            time_axis = jpca_results['time_axis']
            
            # Plot trajectories for first jPCA pair
            pair_idx = 0
            if X_jpca.shape[2] >= 2:
                # Average across trials
                mean_trajectory = np.mean(X_jpca, axis=0)
                
                fig.add_trace(
                    go.Scatter(
                        x=time_axis,
                        y=mean_trajectory[:, 0],
                        mode='lines+markers',
                        name='jPCA1',
                        line=dict(color='blue', width=2),
                        marker=dict(size=4)
                    ),
                    row=1, col=2
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=time_axis,
                        y=mean_trajectory[:, 1],
                        mode='lines+markers',
                        name='jPCA2',
                        line=dict(color='red', width=2),
                        marker=dict(size=4)
                    ),
                    row=1, col=2
                )
        
        # 3. Rotation matrices heatmap
        rotation_matrices = jpca_results['rotation_matrices']
        if len(rotation_matrices) > 0:
            # Show first rotation matrix
            rot_matrix = rotation_matrices[0]
            fig.add_trace(
                go.Heatmap(
                    z=rot_matrix,
                    colorscale='RdBu',
                    showscale=True,
                    text=np.round(rot_matrix, 3),
                    texttemplate="%{text}",
                    textfont={"size": 12}
                ),
                row=2, col=1
            )
        
        # 4. Phase space plot (first jPCA pair)
        if len(jpca_pairs) > 0 and X_jpca.shape[2] >= 2:
            # Plot phase space for first jPCA pair
            mean_trajectory = np.mean(X_jpca, axis=0)
            
            fig.add_trace(
                go.Scatter(
                    x=mean_trajectory[:, 0],
                    y=mean_trajectory[:, 1],
                    mode='lines+markers',
                    name='Phase Space',
                    line=dict(color='purple', width=3),
                    marker=dict(size=5, color=time_axis, colorscale='Viridis', 
                              showscale=True, colorbar=dict(title="Time (s)"))
                ),
                row=2, col=2
            )
            
            # Add start and end markers
            fig.add_trace(
                go.Scatter(
                    x=[mean_trajectory[0, 0]],
                    y=[mean_trajectory[0, 1]],
                    mode='markers',
                    name='Start',
                    marker=dict(size=10, color='green', symbol='diamond')
                ),
                row=2, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=[mean_trajectory[-1, 0]],
                    y=[mean_trajectory[-1, 1]],
                    mode='markers',
                    name='End',
                    marker=dict(size=10, color='red', symbol='diamond')
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title=title,
            height=800,
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Component", row=1, col=1)
        fig.update_yaxes(title_text="Singular Value", row=1, col=1)
        fig.update_xaxes(title_text="Time (s)", row=1, col=2)
        fig.update_yaxes(title_text="jPCA Component", row=1, col=2)
        fig.update_xaxes(title_text="jPCA2", row=2, col=1)
        fig.update_yaxes(title_text="jPCA1", row=2, col=1)
        fig.update_xaxes(title_text="jPCA1", row=2, col=2)
        fig.update_yaxes(title_text="jPCA2", row=2, col=2)
        
        return fig
        
    except Exception as e:
        st.error(f"Error plotting jPCA results: {e}")
        return None