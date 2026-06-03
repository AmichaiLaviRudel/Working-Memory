#!/bin/bash
#SBATCH --job-name=npxl_group_decoder
#SBATCH --output=npxl_group_decoder_%j.out
#SBATCH --error=npxl_group_decoder_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu.q
# If submission fails, list partitions with: sinfo -s

# Resource notes:
# - Uses gpu.q for queue availability; sklearn decoders run on CPU only (no --gres)
# - 8 CPUs: sklearn uses OpenMP threading for SVM; Logistic Regression is single-threaded
#   but multiple sessions can be processed faster with OMP parallelism
# - 64 GB: each session loads event matrices (~units × time × trials), peak usage
#   depends on session count and matrix size; 64 GB gives headroom for large cohorts

# -----------------------------------------------------------------------
# Configuration — edit these before submitting
# -----------------------------------------------------------------------
CONDA_ROOT="/ems/elsc-labs/mizrahi-a/amichai.lavi/miniconda3"
# Use "base" — all required packages are available there via ~/.local/lib
# Change to "analysis" or "ks4" if you prefer an isolated env
CONDA_ENV_NAME="base"
CODE_ROOT="/ems/elsc-labs/mizrahi-a/Shared/Amichai/Code"

# Path to the NPXL monitoring CSV (same file the Streamlit app uses).
MONITORING_CSV="${CODE_ROOT}/DB/users_data/Amichai/NPXL recordings _experimental_data.csv"

# Where to save the results (directory will be created if it doesn't exist).
# npxl_group_decoder_batch.py appends _histology or _no_histology to OUTPUT_CSV's base name.
#   With histology:    OUTPUT_CSV=.../npxl_group_decoder_results_with_histology.csv  -> ..._with_histology_histology.csv
#   No histology:      OUTPUT_CSV=.../npxl_group_decoder_results.csv + --no_histology -> ..._no_histology.csv
# The batch script also writes sidecars next to the main CSV, including:
#   *_joint_probs.csv, *_model_agreement.csv, *_kappa_agreement.csv,
#   *_kappa_by_stimulus.csv, and psychometric agreement/disagreement CSVs.
OUTPUT_DIR="${CODE_ROOT}/DB/Results"
OUTPUT_CSV="${OUTPUT_DIR}/npxl_group_decoder_results.csv"

# Decoder settings (must match the time axis of your sessions)
DECODE_WINDOW_START=0.0      # seconds relative to tone onset
DECODE_WINDOW_STOP=0.25
AGGREGATION="Mean"            # Mean | Sum
CLASSIFIER="RBF SVM"   # "Logistic Regression" | "RBF SVM" | "Linear SVM"
RANDOM_STATE=42
MIN_TRIALS=20

# Uncomment the next line to skip histology filtering (more units, no area guarantee):
# NO_HISTOLOGY_FLAG="--no_histology"

# -----------------------------------------------------------------------
# 1. Initialise conda
# -----------------------------------------------------------------------
source "${CONDA_ROOT}/etc/profile.d/conda.sh"

echo "Available conda environments:"
conda info --envs

if [ "${CONDA_ENV_NAME}" = "base" ] || conda env list | grep -q "^${CONDA_ENV_NAME} "; then
    echo "[setup] Activating ${CONDA_ENV_NAME}..."
    conda activate "${CONDA_ENV_NAME}"
else
    echo "[setup] WARNING: env '${CONDA_ENV_NAME}' not found; using base"
    conda activate base
fi

# -----------------------------------------------------------------------
# 2. Install required Python packages if missing
#    tqdm is required by load_bpod_data (pulled in via NPXL_Preprocessing)
# -----------------------------------------------------------------------
_install_if_missing() {
    python -c "import ${1}" 2>/dev/null || pip install "${2:-$1}"
}
_install_if_missing numpy
_install_if_missing pandas
_install_if_missing scipy
_install_if_missing sklearn scikit-learn
_install_if_missing tqdm
_install_if_missing streamlit
_install_if_missing plotly

# Verify the full import chain before starting the long batch run
python - <<PYCODE || {
import sys
sys.path.insert(0, "${CODE_ROOT}")
sys.path.insert(0, "${CODE_ROOT}/DB")
from npxl_agreement_decoder import load_valid_sessions, _run_one_session_batch
print("[setup] npxl_agreement_decoder import OK")
PYCODE
    echo "[setup] Import check failed — installing analysis stack..."
    pip install numpy pandas scipy scikit-learn tqdm streamlit plotly openpyxl h5py
    python - <<PYCODE2
import sys
sys.path.insert(0, "${CODE_ROOT}")
sys.path.insert(0, "${CODE_ROOT}/DB")
from npxl_agreement_decoder import load_valid_sessions
print("[setup] npxl_agreement_decoder import OK (after pip install)")
PYCODE2
}

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
