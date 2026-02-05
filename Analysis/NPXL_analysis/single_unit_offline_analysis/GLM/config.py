"""
Configuration parameters for GLM analysis.
"""
import pynapple as nap

# ============================================================================
# Data Paths
# ============================================================================

BASE_PATH = r"Z:\\Shared\\Amichai\\NPXL\\Recs\\group7\\catgt_G7A2_novice_2b_4t_g1"


# ============================================================================
# Filtering Parameters
# ============================================================================

# Minimum average firing rate threshold (Hz) for selecting units
RATE_THRESHOLD = 10


# ============================================================================
# Binning Parameters
# ============================================================================

# Bin size for binning spike counts and predictors (in seconds)
BIN_SIZE = 0.01

# Bin size for preprocessing continuous data used in feature extraction (in seconds)
PREPROCESSING_BIN_SIZE = 0.005


# ============================================================================
# Epoch Parameters
# ============================================================================

# Window for peri-event epoching (in seconds, relative to event)
EPOCH_START = -1
EPOCH_END = 3

# Example epoch (for plotting): here from 30s to 90s
EXAMPLE_EPOCH = nap.IntervalSet(start=30, end=90)

# Which unit to show as example when plotting
EXAMPLE_NEURON_ID = 1


# ============================================================================
# Stimulus Categorization
# ============================================================================

# Thresholds for categorizing stimulus into "Go" or "NoGo"
LOW_BOUNDARY_THRESHOLD = 0.983
HIGH_BOUNDARY_THRESHOLD = 1.525


# ============================================================================
# Basis Function Parameters
# ============================================================================

# Number of raised cosine basis functions for all event-locked convolutions
N_BASIS_FUNCS = 8

# Window size (in seconds) after event, used for temporal event-aligned features (causal)
EVENT_WINDOW_SEC = 4  # seconds after event

# Acausal window for categorical features (captures effects before and after trial characteristics)
ACAUSAL_BEFORE_SEC = 1  # seconds before event
ACAUSAL_AFTER_SEC = 3   # seconds after event

# Spike history window size (in seconds) for spike history predictor
HISTORY_WINDOW_SEC = 1

# Acausal window for spike history in population GLM
HISTORY_ACAUSAL_BEFORE_SEC = 1  # seconds before spike
HISTORY_ACAUSAL_AFTER_SEC = 1   # seconds after spike


# ============================================================================
# Population GLM Parameters
# ============================================================================

# Number of neurons to include in population analysis
N_POPULATION = 1000

# Parameter to control whether to include spike history
INCLUDE_SPIKE_HISTORY = False
# Note: When True, creates ALL-TO-ALL connectivity - each neuron's spike history 
#       features are included and can affect ALL neurons via PopulationGLM coefficients


# ============================================================================
# GLM Solver Parameters
# ============================================================================

GLM_SOLVER = "LBFGS"
GLM_REGULARIZER = "Ridge"
GLM_REGULARIZER_STRENGTH = 0.1

