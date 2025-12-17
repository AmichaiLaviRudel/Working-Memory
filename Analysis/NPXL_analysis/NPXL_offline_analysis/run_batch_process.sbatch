#!/bin/bash
#SBATCH --job-name=npxl_batch_refactored
#SBATCH --output=npxl_batch_refactored_%j.out
#SBATCH --error=npxl_batch_refactored_%j.err
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=64G
#SBATCH --partition=elscn.q
#
# NPXL Batch Processing (Refactored Pipeline)
# This job runs main_refactored.py for all catgt folders
# Generates tone/choice/outcome aligned plots with p-value filtering

# Try to initialize conda - adjust path if needed
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
elif [ -f ~/.bashrc ]; then
    source ~/.bashrc
fi

# Activate conda environment if conda is available
if command -v conda >/dev/null 2>&1; then
    # List available environments for debugging
    echo "Available conda environments:"
    conda info --envs
    
    # Try to activate working_memory, fall back to base if not found
    if conda env list | grep -q "working_memory"; then
        echo "Activating working_memory environment..."
        conda activate working_memory
    else
        echo "Warning: working_memory environment not found. Using base environment."
        conda activate base
        # Install required packages if not available
        python -c "import pandas" 2>/dev/null || python -m pip install --user pandas numpy scipy plotly scikit-learn streamlit
    fi
    
    # Verify pandas is available
    python - <<'PYCODE' || {
from importlib import import_module
mods = ("pandas", "numpy", "scipy", "plotly", "sklearn", "streamlit")
for mod in mods:
    import_module(mod)
print("Packages present:", ", ".join(mods))
PYCODE
        echo "Installing missing scientific stack..."
        python -m pip install --user pandas numpy scipy plotly scikit-learn streamlit
    }
else
    echo "Warning: conda not found. Using system Python."
fi

# Change to the script directory
cd /ems/elsc-labs/mizrahi-a/Shared/Amichai/Code/DB

# Limit thread usage to prevent excessive memory consumption
export OMP_NUM_THREADS=36
export MKL_NUM_THREADS=36
export NUMEXPR_NUM_THREADS=36

# Run the batch processing script
python Analysis/NPXL_analysis/NPXL_offline_analysis/batch_process.py --base-dir "/ems/elsc-labs/mizrahi-a/Shared/Amichai/NPXL/Recs"

echo "Batch processing job completed at $(date)"

# Show last 50 lines of output and error files
echo ""
echo "=== Last 50 lines of output (if available) ==="
tail -n 50 "npxl_batch_refactored_${SLURM_JOB_ID}.out" 2>/dev/null || echo "Output file not yet written or not found"

echo ""
echo "=== Last 50 lines of error output (if available) ==="
tail -n 50 "npxl_batch_refactored_${SLURM_JOB_ID}.err" 2>/dev/null || echo "Error file not yet written or not found"

echo ""
echo "Job finished. Check npxl_batch_refactored_${SLURM_JOB_ID}.out and npxl_batch_refactored_${SLURM_JOB_ID}.err for full output."
