"""
Color schemes and font styling for GLM visualizations.
"""
import matplotlib
import matplotlib.pyplot as plt

# ============================================================================
# Font Configuration
# ============================================================================

def configure_fonts():
    """Configure matplotlib to use modern scientific fonts."""
    matplotlib.rcParams['font.family'] = 'sans-serif'
    # Try Lato first, then Inter, then fallback to system sans-serif
    matplotlib.rcParams['font.sans-serif'] = ['Lato', 'Inter', 'DejaVu Sans', 'Helvetica', 'Liberation Sans']
    matplotlib.rcParams['font.size'] = 10
    matplotlib.rcParams['axes.labelsize'] = 11
    matplotlib.rcParams['axes.titlesize'] = 12
    matplotlib.rcParams['xtick.labelsize'] = 9
    matplotlib.rcParams['ytick.labelsize'] = 9
    matplotlib.rcParams['legend.fontsize'] = 9
    matplotlib.rcParams['figure.titlesize'] = 14


# ============================================================================
# Color Palettes
# ============================================================================

# Region colors (for ACx vs OFC comparisons)
REGION_COLORS = {
    'ACx': '#4A90E2',  # Blue
    'OFC': '#E74C3C',  # Red
}

# Feature type colors
FEATURE_COLORS = {
    'temporal': '#4A90E2',      # Blue for temporal features
    'categorical': '#50C878',   # Green for categorical features
    'spike_history': '#E74C3C', # Red for spike history
    'other': '#95A5A6',         # Gray for others
}

# Gradient colors for area plots
GRADIENT_COLORS = {
    'actual': 'orange',
    'predicted': 'green',
    'firing_rate': 'tab:blue',
}

# Colormap for connectivity matrices
CONNECTIVITY_CMAP = 'RdBu_r'


# ============================================================================
# Helper Functions
# ============================================================================

def get_feature_color(feature_name: str) -> str:
    """
    Get color for a feature based on its type.
    
    Parameters
    ----------
    feature_name : str
        Name of the feature
        
    Returns
    -------
    str
        Color hex code or color name
    """
    if feature_name in ['tone_onset', 'licks', 'outcome_onset']:
        return FEATURE_COLORS['temporal']
    elif feature_name.startswith('cat:') or feature_name.startswith('out:') or \
         feature_name.startswith('prev:') or feature_name == 'stimulus' or \
         feature_name.startswith('stim:'):
        return FEATURE_COLORS['categorical']
    elif feature_name == 'spike_history':
        return FEATURE_COLORS['spike_history']
    else:
        return FEATURE_COLORS['other']


def get_coefficient_colors(coefficients, positive_color='tab:blue', negative_color='tab:red'):
    """
    Get colors for coefficient bars based on sign.
    
    Parameters
    ----------
    coefficients : array-like
        Coefficient values
    positive_color : str
        Color for positive values
    negative_color : str
        Color for negative values
        
    Returns
    -------
    list
        List of colors for each coefficient
    """
    return [positive_color if c >= 0 else negative_color for c in coefficients]


# ============================================================================
# Plotly Colors
# ============================================================================

PLOTLY_REGION_COLORS = {
    'ACx': '#4A90E2',
    'OFC': '#E74C3C',
}

PLOTLY_FEATURE_LABELS = {
    'stimulus': 'Stimulus ID',
    'category': 'Category',
    'licks': 'Licks',
    'outcome': 'Outcome',
    'previous_outcome': 'Previous Outcome',
    'tone_onset': 'Tone Onset',
    'outcome_onset': 'Outcome Onset',
}

