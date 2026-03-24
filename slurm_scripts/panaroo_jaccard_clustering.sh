#!/bin/bash
#SBATCH --job-name=sv_clustering_CG11
#SBATCH --output=sv_clustering_CG11_%j.out
#SBATCH --error=sv_clustering_CG11_%j.err
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU
#SBATCH --mem=4G

cd /home/dca36/workspace/Bacotype

# Force unbuffered Python output for real-time logging
export PYTHONUNBUFFERED=1

echo "========================================================================"
echo "Panaroo SV clustering (KNN + Leiden)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 4G"
echo "========================================================================"
echo ""

# Change --strain or --filter-cutoff as needed
uv run python -u src/bacotype/tl/panaroo_jaccard_clustering.py \
  --strain CG11 \
  --filter-cutoff 50

echo ""
echo "========================================================================"
echo "Job complete!"
echo "========================================================================"

# Run with: sbatch slurm_scripts/panaroo_jaccard_clustering.sh
# Check: squeue -u dca36
# Cancel: scancel <jobid>
