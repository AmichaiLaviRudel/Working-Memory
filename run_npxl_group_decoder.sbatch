#!/bin/bash
#SBATCH --job-name=npxl_group_decoder
#SBATCH --output=npxl_group_decoder_%j.out
#SBATCH --error=npxl_group_decoder_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=elscn.q
# If submission fails, list partitions with: sinfo -s

# Resource notes:
# - No GPU needed (sklearn classifiers run on CPU)
# - 8 CPUs: sklearn uses OpenMP threading for SVM; Logistic Regression is single-threaded
#   but multiple sessions can be processed faster with OMP parallelism
# - 64 GB: each session loads event matrices (~units × time × trials), peak usage
#   depends on session count and matrix size; 64 GB gives headroom for large cohorts

# -----------------------------------------------------------------------
# Configuration — edit these before submitting
# -----------------------------------------------------------------------
CONDA_ROOT="/ems/elsc-labs/mizrahi-a/amichai.lavi/miniconda3"
CONDA_ENV_NAME="analysis"        # <-- set to your analysis conda environment
CODE_ROOT="/ems/elsc-labs/mizrahi-a/Shared/Amichai/Code"

# Path to the NPXL monitoring CSV (same file the Streamlit app uses)
MONITORING_CSV="${CODE_ROOT}/DB/npxl_monitoring.csv"   # <-- update if needed

# Where to save the results (directory will be created if it doesn't exist)
OUTPUT_DIR="${CODE_ROOT}/DB/Results"
OUTPUT_CSV="${OUTPUT_DIR}/npxl_group_decoder_results.csv"

# Decoder settings (must match the time axis of your sessions)
DECODE_WINDOW_START=0.0      # seconds relative to tone onset
DECODE_WINDOW_STOP=0.25
AGGREGATION="Mean"            # Mean | Sum
CLASSIFIER="Logistic Regression"   # "Logistic Regression" | "RBF SVM" | "Linear SVM"
RANDOM_STATE=42
MIN_TRIALS=5

# Uncomment the next line to skip histology filtering (more units, no area guarantee):
# NO_HISTOLOGY_FLAG="--no_histology"

# -----------------------------------------------------------------------
# 1. Initialise conda
# -----------------------------------------------------------------------
source "${CONDA_ROOT}/etc/profile.d/conda.sh"

if ! conda env list | grep -q "^${CONDA_ENV_NAME} "; then
    echo "[setup] Creating conda environment '${CONDA_ENV_NAME}'..."
    conda create -y -n "${CONDA_ENV_NAME}" python=3.10
fi

conda activate "${CONDA_ENV_NAME}"

# -----------------------------------------------------------------------
# 2. Install required Python packages if missing
# -----------------------------------------------------------------------
python -c "import numpy"       2>/dev/null || pip install numpy
python -c "import pandas"      2>/dev/null || pip install pandas
python -c "import sklearn"     2>/dev/null || pip install scikit-learn
python -c "import streamlit"   2>/dev/null || pip install streamlit
python -c "import plotly"      2>/dev/null || pip install plotly

# -----------------------------------------------------------------------
# 3. Set PYTHONPATH so relative imports in the analysis code resolve
# -----------------------------------------------------------------------
export PYTHONPATH="${CODE_ROOT}:${CODE_ROOT}/DB:${PYTHONPATH}"

# Limit per-process threads to avoid over-subscription
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

# -----------------------------------------------------------------------
# 4. Create output directory
# -----------------------------------------------------------------------
mkdir -p "${OUTPUT_DIR}"

# -----------------------------------------------------------------------
# 5. Run the decoder
# -----------------------------------------------------------------------
echo "=== NPXL Group Decoder started at $(date) ==="
echo "    Job ID    : ${SLURM_JOB_ID}"
echo "    Node      : $(hostname)"
echo "    Python    : $(which python)"
echo "    Conda env : ${CONDA_DEFAULT_ENV}"
echo "    Monitoring: ${MONITORING_CSV}"
echo "    Output    : ${OUTPUT_CSV}"
echo ""

python "${CODE_ROOT}/DB/npxl_group_decoder_batch.py" \
    --monitoring_csv  "${MONITORING_CSV}" \
    --output          "${OUTPUT_CSV}" \
    --decode_window   ${DECODE_WINDOW_START} ${DECODE_WINDOW_STOP} \
    --aggregation     "${AGGREGATION}" \
    --classifier      "${CLASSIFIER}" \
    --random_state    ${RANDOM_STATE} \
    --min_trials      ${MIN_TRIALS} \
    ${NO_HISTOLOGY_FLAG:-}

EXIT_CODE=$?

echo ""
echo "=== NPXL Group Decoder finished at $(date) (exit code: ${EXIT_CODE}) ==="

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "Results saved to: ${OUTPUT_CSV}"
    # Copy results back to a local path if running from a cluster scratch dir
    # (uncomment and adjust LOCAL_RESULTS_DIR if needed):
    # LOCAL_RESULTS_DIR="/path/on/local/machine"
    # cp "${OUTPUT_CSV}" "${LOCAL_RESULTS_DIR}/"
    # cp "${OUTPUT_CSV%.csv}_failures.csv" "${LOCAL_RESULTS_DIR}/" 2>/dev/null || true
fi

echo ""
echo "=== Last 30 lines of output ==="
tail -n 30 "npxl_group_decoder_${SLURM_JOB_ID}.out" 2>/dev/null || true

exit ${EXIT_CODE}
