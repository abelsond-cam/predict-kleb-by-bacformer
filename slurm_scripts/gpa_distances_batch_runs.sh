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
#
# gpa_distances_batch_runs.sh
# ---------------------------
# Runs: src/bacotype/tl/gpa_distances_batch_runs.py
#
# What it does:
#   Scans PANAROO_RUN_ROOT (set below) for each immediate subdirectory that
#   contains gene_presence_absence.Rtab, and runs the stratified distance
#   analysis (gpa_distances_single_run.py orchestrator) on every such folder
#   in parallel (--workers). For each Panaroo run the orchestrator produces a
#   detail TSV (whole set + per-CG + per-CG+K_locus rows). The batch itself
#   writes one compiled summary TSV (gpa_reference_batch_summary_<timestamp>.tsv,
#   one row per run = whole-set row) under --output-dir
#   (default: .../processed/pangenome_analysis).
#
# This is the "all completed Panaroo runs → one distance/clustering summary table"
# entrypoint. For a single folder only, use gpa_distances_single_run.sh instead.
#

cd /home/dca36/workspace/Bacotype
export PYTHONUNBUFFERED=1

# ---------------- User-editable settings ----------------
DATA_ROOT="/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david"
PANAROO_RUN_ROOT="${DATA_ROOT}/processed/panaroo_with_reference_genome"
METADATA_PATH="${DATA_ROOT}/final/metadata_final_curated_all_samples_and_columns.tsv"
OUTPUT_DIR="${DATA_ROOT}/processed/pangenome_analysis"
WORKERS=8
TEST_N_SUBDIR=""   # e.g. 3 for quick tests; leave empty for all
MIN_GROUP_SIZE=250          # Min Clonal group / K_locus size for its own slice
REFERENCE_TOP_N=10
GPA_FILTER_CUTOFF=""       # e.g. 20; leave empty for auto
MERGE_SMALL_CLUSTERS=""    # e.g. 15; leave empty for auto
SHELL_CLOUD_CUTOFF=0.15
CORE_SHELL_CUTOFF=0.95
REPORT_TIMES=false
# --------------------------------------------------------

echo "========================================================================"
echo "GPA distances: batch all Panaroo run subdirs (gpa_distances_batch_runs.py)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "PANAROO_RUN_ROOT: ${PANAROO_RUN_ROOT}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo "========================================================================"
echo ""

CMD=(
  uv run python -u src/bacotype/tl/gpa_distances_batch_runs.py
  --workers "${WORKERS}"
  --panaroo-run-root "${PANAROO_RUN_ROOT}"
  --metadata "${METADATA_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --min-group-size "${MIN_GROUP_SIZE}"
  --reference-top-n "${REFERENCE_TOP_N}"
  --shell-cloud-cutoff "${SHELL_CLOUD_CUTOFF}"
  --core-shell-cutoff "${CORE_SHELL_CUTOFF}"
)

if [[ -n "${TEST_N_SUBDIR}" ]]; then
  CMD+=(--test-n-subdir "${TEST_N_SUBDIR}")
fi
if [[ -n "${GPA_FILTER_CUTOFF}" ]]; then
  CMD+=(--gpa-filter-cutoff "${GPA_FILTER_CUTOFF}")
fi
if [[ -n "${MERGE_SMALL_CLUSTERS}" ]]; then
  CMD+=(--merge-small-clusters "${MERGE_SMALL_CLUSTERS}")
fi
if [[ "${REPORT_TIMES}" == "true" ]]; then
  CMD+=(--report-times)
fi

"${CMD[@]}"

echo ""
echo "========================================================================"
echo "Job complete!"
echo "========================================================================"
