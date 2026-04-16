#!/bin/bash
#SBATCH --job-name=panclustering_SL3010
#SBATCH --output=panclustering_SL3010_%j.out
#SBATCH --error=panclustering_SL3010_%j.err
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
echo "Panaroo SV + GPA clustering (KNN + Leiden)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 8G"
echo "========================================================================"
echo ""

# Change --strain or filter cutoffs as needed
uv run python -u src/bacotype/tl/panaroo_jaccard_clustering.py \
  --strain SL3010
  # --sv-filter-cutoff 20 \
  # --gpa-filter-cutoff 20

# Optional overrides (uncomment to use):
# --sv-filter-cutoff 20
# --gpa-filter-cutoff 20
# --strain CG147
# --strain-panaroo-dir /home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/panaroo_run/CG147_all_part1

# Example: use the directory as input AND output base (omit --strain to label from directory basename)
# uv run python -u src/bacotype/tl/panaroo_jaccard_clustering.py \
#   # --strain CG147_all_part1  # optional label override
#   --strain-panaroo-dir /home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/panaroo_run/CG147_all_part1

echo ""
echo "========================================================================"
echo "Job complete!"
echo "========================================================================"

# Run with: sbatch slurm_scripts/panaroo_jaccard_clustering.sh
# Check: squeue -u dca36
# Cancel: scancel <jobid>
