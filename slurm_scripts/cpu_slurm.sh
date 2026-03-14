#!/bin/bash
#SBATCH --job-name=preprocess_isolation_source_data    
#SBATCH --output=preprocess_isolation_source_data_%j.out     
#SBATCH --error=preprocess_isolation_source_data_%j.err
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=76
#SBATCH --time=00:30:00
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU  

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
# uv run python src/bacotype/pp/prepare_esmc_embeddings_and_labels_to_finetune_amr.py --skip-existing
uv run python src/bacotype/pp/add_paths_gff_fna_to_metadata.py

echo ""
echo "========================================================================"
echo "Processing complete!"
echo "========================================================================"

# Run with: sbatch cpu_slurm.sh
# Check on progress with: squeue -u dca36
# To cancel: scancel jobid

