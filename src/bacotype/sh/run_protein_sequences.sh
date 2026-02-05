#!/bin/bash
#SBATCH --job-name=protein_seqs_klebsiella
#SBATCH --output=protein_seqs_%A.out
#SBATCH --error=protein_seqs_%A.err
#SBATCH --time=24:00:00
#SBATCH --partition=cclake
#SBATCH --account=FLOTO-SL2-CPU
#SBATCH --nodes=1
#SBATCH --ntasks=120  
#SBATCH --mem=200G

# Script to run protein sequence extraction on HPC with CPU parallelization
# Usage:
#   sbatch src/bacotype/sh/run_protein_sequences.sh --n 10  # Test with 10 files
#   sbatch src/bacotype/sh/run_protein_sequences.sh         # Process all files
#   sbatch src/bacotype/sh/run_protein_sequences.sh --skip-existing  # Resume

# Force Python unbuffered output for real-time logging
export PYTHONUNBUFFERED=1
# Use work directory for UV cache to avoid disk space issues in home directory
export UV_CACHE_DIR=/home/dca36/rds/hpc-work/.uv_cache

# Ensure uv is in PATH (adjust this path to where uv is installed)
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

# Change to project directory
cd /home/dca36/workspace/Bacotype

echo "=========================================="
echo "Protein Sequence Extraction"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: ${SLURM_MEM_PER_NODE}M"
echo "Start time: $(date)"
echo "Arguments: $@"
echo "=========================================="

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "ERROR: uv not found in PATH"
    echo "PATH: $PATH"
    echo "Trying to use python directly from virtual environment..."
    
    # Try to use existing virtual environment
    if [ -d ".venv" ]; then
        source .venv/bin/activate
        python src/bacotype/pp/generate_protein_sequences.py "$@"
    else
        echo "ERROR: No .venv found and uv not available"
        exit 1
    fi
else
    echo "Using uv: $(which uv)"
    # Run the Python script with all passed arguments
    uv run python src/bacotype/pp/generate_protein_sequences.py "$@"
fi

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
