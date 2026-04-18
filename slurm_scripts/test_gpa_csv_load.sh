#!/bin/bash
#SBATCH --job-name=test_gpa_csv_load
#SBATCH --output=test_gpa_csv_load_%j.out
#SBATCH --error=test_gpa_csv_load_%j.err
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU
#SBATCH --mem=8G
#
# test_gpa_csv_load.sh
# ---------------------
# Runs: src/bacotype/tl/test_gpa_csv_load_standalone.py
#
# Minimal-imports sanity check for the streaming gene_presence_absence.csv
# loader. Copies the CSV to node-local scratch ($TMPDIR) first, then streams
# it into a uint16 count matrix and reports copy throughput, parse throughput,
# matrix size, and peak RSS.
#

cd /home/dca36/workspace/Bacotype

export PYTHONUNBUFFERED=1

# ---------------- User-editable settings ----------------
DATA_ROOT="/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david"
CSV="${DATA_ROOT}/processed/panaroo_with_reference_genome/SL17_part_0/gene_presence_absence.csv"
PROGRESS_EVERY=500
# --------------------------------------------------------

echo "========================================================================"
echo "Standalone GPA CSV loader timing test"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "TMPDIR: ${TMPDIR:-<unset>}"
echo "CSV: ${CSV}"
echo "========================================================================"
echo ""

uv run python -u src/bacotype/tl/test_gpa_csv_load_standalone.py \
    --csv "${CSV}" \
    --progress-every "${PROGRESS_EVERY}"

echo ""
echo "========================================================================"
echo "Done"
echo "========================================================================"

# Run with: sbatch slurm_scripts/test_gpa_csv_load.sh
# Check:    squeue -u dca36
# Cancel:   scancel <jobid>
