#!/bin/bash
#SBATCH --job-name=icelake_himem_batch_runs
#SBATCH --output=iclake_himem_batch_runs_%j.out
#SBATCH --error=iclake_himem_batch_runs_%j.err
#SBATCH --partition=icelake-himem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=24:00:00
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
# Avoid writing .pyc back into the RDS-backed .venv when we execute its
# bin/python directly (keeps RDS access read-only, avoids NFS write latency).
export PYTHONDONTWRITEBYTECODE=1

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
# Cut-down modes for login-node / slow-HPC execution. Both default to false.
#   SKIP_CLUSTERING=true: skip scanpy neighbors/UMAP/Leiden/merge, UMAP plots,
#     quality metrics, and rank_genes_groups for every slice in every run.
#     scanpy is then never imported, which removes a multi-minute import cost
#     on congested RDS. Clustering columns emitted as NaN.
#   SKIP_JACCARD=true: skip MGH78578 / RefSeq / complete-Norway cohort Jaccard
#     summaries for every slice. Distance columns emitted as NaN; reference
#     counts preserved. Both true gives a pure stats-only batch.
SKIP_CLUSTERING=true
SKIP_JACCARD=true
# Stage project .venv to node-local scratch before running Python. On a
# congested RDS this can turn "imports hang for >1h" into "imports finish
# in seconds", and the cost is paid once for all WORKERS. Disable with
# STAGE_VENV=false if rsync itself is slow on a given node.
STAGE_VENV=false
# --------------------------------------------------------

echo "========================================================================"
echo "GPA distances: batch all Panaroo run subdirs (gpa_distances_batch_runs.py)"
echo "IceLake-HiMem"
echo "--------------------------------"
echo "Stage VENV: ${STAGE_VENV}"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "PANAROO_RUN_ROOT: ${PANAROO_RUN_ROOT}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo "========================================================================"
echo ""

# Pick python: staged local copy (fast imports on congested RDS) or uv run.
PYTHON_CMD=("uv" "run" "python" "-u")
if [[ "${STAGE_VENV}" == "true" ]]; then
  SCRATCH="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
  STAGED_VENV="${SCRATCH}/bacotype_venv_${SLURM_JOB_ID:-$$}"
  if [[ ! -L .venv && ! -d .venv ]]; then
    echo "WARNING: .venv not found; falling back to 'uv run'." >&2
  else
    VENV_REAL="$(readlink -f .venv)"
    echo "venv staging: resolved .venv -> ${VENV_REAL}"
    echo "venv staging: rsync -> ${STAGED_VENV}"
    rm -rf "${STAGED_VENV}"
    mkdir -p "${STAGED_VENV}"
    if /usr/bin/time -f "venv staging: rsync elapsed=%es peak_rss=%MkB" \
         rsync -a "${VENV_REAL}/" "${STAGED_VENV}/"; then
      if [[ -x "${STAGED_VENV}/bin/python" ]]; then
        PYTHON_CMD=("${STAGED_VENV}/bin/python" "-u")
        echo "venv staging: using ${STAGED_VENV}/bin/python (shared across ${WORKERS} workers)"
      else
        echo "WARNING: staged venv missing bin/python; falling back to 'uv run'." >&2
      fi
    else
      echo "WARNING: rsync failed; falling back to 'uv run'." >&2
    fi
  fi
else
  echo "venv staging: disabled (STAGE_VENV=false); using 'uv run'."
fi
echo ""

CMD=(
  "${PYTHON_CMD[@]}" src/bacotype/tl/gpa_distances_batch_runs.py
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
if [[ "${SKIP_CLUSTERING}" == "true" ]]; then
  CMD+=(--skip-clustering)
fi
if [[ "${SKIP_JACCARD}" == "true" ]]; then
  CMD+=(--skip-jaccard)
fi

"${CMD[@]}"
RC=$?

if [[ "${STAGE_VENV}" == "true" && -n "${STAGED_VENV:-}" && -d "${STAGED_VENV}" ]]; then
  echo ""
  echo "cleanup: removing staged venv ${STAGED_VENV}"
  rm -rf "${STAGED_VENV}" || echo "cleanup: rm -rf failed (ignored)"
fi

echo ""
echo "========================================================================"
echo "Job complete! (exit=${RC})"
echo "========================================================================"
exit "${RC}"

# Run with: sbatch slurm_scripts/gpa_distances_batch_runs.sh
# Check:    squeue -u dca36
# Cancel:   scancel <jobid>
# Toggle venv staging off for a run: edit STAGE_VENV=false near the top.
