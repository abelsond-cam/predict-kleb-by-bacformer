#!/bin/bash
#SBATCH --job-name=gpa_distance_single_run
#SBATCH --output=gpa_distance_single_run_%j.out
#SBATCH --error=gpa_distance_single_run_%j.err
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU
#SBATCH --mem=8G
#
# gpa_distances_single_run.sh
# -----------------------------
# Runs: src/bacotype/tl/gpa_distances_single_run.py
#
# What it does:
#   Stratified post-Panaroo analysis on ONE directory that already contains
#   gene_presence_absence.Rtab. Runs the per-group analysis
#   (gpa_distances_single_group.py) on:
#     1) the whole set,
#     2) each major Clonal group (>= MIN_GROUP_SIZE) + pooled 'other',
#     3) within each major CG, each major K_locus (>= MIN_GROUP_SIZE) + pooled
#        '<CG>_other'.
#   Reference genomes (mgh78578 + RefSeq + complete Norway) are added back to
#   every subset before the group analysis so distances can always be computed.
#   All rows are written to a single detail TSV under
#   <panaroo_dir>/analysis/GPA_reference_genome/.
#
# Input (set variables below):
#   DIRECTORY_LEAF + PANAROO_RUN_ROOT  -> PANAROO_RUN_ROOT/DIRECTORY_LEAF
#   PANAROO_DIR (full path)            -> explicit directory mode
#
# For batching every run subfolder under PANAROO_RUN_ROOT, use
# gpa_distances_batch_runs.sh instead.
#

cd /home/dca36/workspace/Bacotype

# Force unbuffered Python output for real-time logging
export PYTHONUNBUFFERED=1

# ---------------- User-editable settings ----------------
DATA_ROOT="/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david"
PANAROO_RUN_ROOT="${DATA_ROOT}/processed/panaroo_with_reference_genome"
DIRECTORY_LEAF="SL17_part_0"  # Used when PANAROO_DIR is empty
PANAROO_DIR=""                 # Full path override; leave empty to use DIRECTORY_LEAF

METADATA_PATH="${DATA_ROOT}/final/metadata_final_curated_all_samples_and_columns.tsv"
MIN_GROUP_SIZE=250             # Min Clonal group / K_locus size for its own slice
GPA_FILTER_CUTOFF=""           # e.g. 20; leave empty for auto
MERGE_SMALL_CLUSTERS=""        # e.g. 15; leave empty for auto
REFERENCE_TOP_N=10
SHELL_CLOUD_CUTOFF=0.15
CORE_SHELL_CUTOFF=0.95
REPORT_TIMES=false
# --------------------------------------------------------

echo "========================================================================"
echo "GPA distances: single Panaroo run (gpa_distances_single_run.py)"
echo "Job ID: $SLURM_JOB_ID"
if [[ -n "${PANAROO_DIR}" ]]; then
  echo "Target mode: explicit directory"
  echo "PANAROO_DIR: ${PANAROO_DIR}"
else
  echo "Target mode: directory leaf under root"
  echo "PANAROO_RUN_ROOT: ${PANAROO_RUN_ROOT}"
  echo "DIRECTORY_LEAF: ${DIRECTORY_LEAF}"
fi
echo "========================================================================"
echo ""

if [[ -z "${PANAROO_DIR}" && -z "${DIRECTORY_LEAF}" ]]; then
  echo "ERROR: Set either PANAROO_DIR or DIRECTORY_LEAF near the top of this script." >&2
  exit 1
fi

CMD=(
  uv run python -u src/bacotype/tl/gpa_distances_single_run.py
  --panaroo-run-root "${PANAROO_RUN_ROOT}"
  --metadata "${METADATA_PATH}"
  --min-group-size "${MIN_GROUP_SIZE}"
  --reference-top-n "${REFERENCE_TOP_N}"
  --shell-cloud-cutoff "${SHELL_CLOUD_CUTOFF}"
  --core-shell-cutoff "${CORE_SHELL_CUTOFF}"
)

if [[ -n "${PANAROO_DIR}" ]]; then
  CMD+=(--panaroo-dir "${PANAROO_DIR}")
else
  CMD+=(--directory-leaf "${DIRECTORY_LEAF}")
fi
if [[ -n "${GPA_FILTER_CUTOFF}" ]]; then
  CMD+=(--gpa-filter-cutoff "${GPA_FILTER_CUTOFF}")
fi
if [[ -n "${MERGE_SMALL_CLUSTERS}" ]]; then
  CMD+=(--merge-small-clusters "${MERGE_SMALL_CLUSTERS}")
fi
if [[ "${REPORT_TIMES}" == "true" ]]; then
  CMD+=(--report-times true)
fi

"${CMD[@]}"

echo ""
echo "========================================================================"
echo "Job complete!"
echo "========================================================================"

# Run with: sbatch slurm_scripts/gpa_distances_single_run.sh
# Check: squeue -u dca36
# Cancel: scancel <jobid>
