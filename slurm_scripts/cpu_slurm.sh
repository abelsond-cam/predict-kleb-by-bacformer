#!/bin/bash
#SBATCH --job-name=preprocess_amr_data    
#SBATCH --output=preprocess_amr_data_%j.out
#SBATCH --error=preprocess_amr_data_%j.err
#SBATCH --partition=icelake-himem 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=76
#SBATCH --time=01:00:00
#SBATCH --account=FLOTO-SL2-CPU

cd /home/dca36/workspace/Bacotype

# Force Python unbuffered output for real-time logging
export PYTHONUNBUFFERED=1

echo "========================================================================"
echo "Starting preprocess_amr_data to derive ESM embeddings for AMR prediction"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "========================================================================"
echo ""

# ESM embeddings
uv run python src/bacotype/pp/prepare_klebsiella_ast_splits_as_pt.py

echo ""
echo "========================================================================"
echo "Processing complete!"
echo "========================================================================"

# Run with: sbatch cpu_slurm.sh
# Check on progress with: squeue -u dca36
# To cancel: scancel jobid

