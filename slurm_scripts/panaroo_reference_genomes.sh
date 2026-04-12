#!/bin/bash
#SBATCH --job-name=pangenomerge_sl17
#SBATCH --output=pangenomerge_sl17_%j.out
#SBATCH --error=pangenomerge_sl17_%j.err
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

# --- Input: uncomment exactly ONE line (flag + value; no default) ---
# INPUT_ARGS=(--directory-leaf SL258_sampled_all)
INPUT_ARGS=(--panaroo-dir /home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/pangenomerge/SL17)

echo "========================================================================"
echo "Panaroo GPA reference genome clustering"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================================================"
echo ""

echo "INPUT_ARGS: ${INPUT_ARGS[@]}"
echo ""

if [[ "${#INPUT_ARGS[@]}" -ne 2 ]]; then
  echo "ERROR: Uncomment exactly one INPUT_ARGS line: --directory-leaf <leaf> or --panaroo-dir <path>." >&2
  exit 1
fi
if [[ "${INPUT_ARGS[0]}" != --directory-leaf && "${INPUT_ARGS[0]}" != --panaroo-dir ]]; then
  echo "ERROR: First element of INPUT_ARGS must be --directory-leaf or --panaroo-dir." >&2
  exit 1
fi

uv run python -u src/bacotype/tl/panaroo_GPA_reference_genome.py \
  "${INPUT_ARGS[@]}"

# Optional overrides (append to uv run line above), e.g.:
# --gpa-filter-cutoff 20
# --reference-top-n 15

echo ""
echo "========================================================================"
echo "Job complete!"
echo "========================================================================"

# Run with: sbatch slurm_scripts/panaroo_reference_genomes.sh
# Check: squeue -u dca36
# Cancel: scancel <jobid>
