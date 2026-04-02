#!/bin/bash
#SBATCH --job-name=panaroo_ref_batch_all
#SBATCH --output=panaroo_ref_batch_all_%j.out
#SBATCH --error=panaroo_ref_batch_all_%j.err
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=02:30:00
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU

cd /home/dca36/workspace/Bacotype
export PYTHONUNBUFFERED=1

echo "========================================================================"
echo "Panaroo GPA reference genome batch runner"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "========================================================================"
echo ""

uv run python -u src/bacotype/tl/panaroo_GPA_batch_runner.py \
  --workers 8

# Quick test mode example:
# uv run python -u src/bacotype/tl/panaroo_GPA_batch_runner.py --workers 2 --test-n-subdir 3

echo ""
echo "========================================================================"
echo "Job complete!"
echo "========================================================================"
