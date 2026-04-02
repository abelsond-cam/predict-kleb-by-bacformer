#!/bin/bash
#SBATCH --job-name=panaroo_ref_sl258
#SBATCH --output=panaroo_ref_sl258_%j.out
#SBATCH --error=panaroo_ref_sl258_%j.err
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU
#SBATCH --mem=8G

cd /home/dca36/workspace/Bacotype

# Force unbuffered Python output for real-time logging
export PYTHONUNBUFFERED=1

echo "========================================================================"
echo "Panaroo GPA reference genome clustering"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================================================"
echo ""

# Change --strain or filter cutoffs as needed
uv run python -u src/bacotype/tl/panaroo_GPA_reference_genome.py \
  --directory-leaf SL258_sampled_all

# Optional overrides (uncomment to use):
# --sv-filter-cutoff 20
# --gpa-filter-cutoff 20
# --reference-top-n 15

echo ""
echo "========================================================================"
echo "Job complete!"
echo "========================================================================"

# Run with: sbatch slurm_scripts/panaroo_jaccard_clustering.sh
# Check: squeue -u dca36
# Cancel: scancel <jobid>
