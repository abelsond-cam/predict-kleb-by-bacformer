#!/bin/bash
#SBATCH --job-name=icelake_gpa_distance_single_run
#SBATCH --output=icelake_gpa_distance_single_run_%j.out
#SBATCH --error=gpa_distance_single_run_%j.err
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
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
# Avoid writing .pyc into the RDS-backed .venv when we run its bin/python
# directly (keeps RDS reads read-only and avoids further NFS write latency).
export PYTHONDONTWRITEBYTECODE=1

# ---------------- User-editable settings ----------------
DATA_ROOT="/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david"
PANAROO_RUN_ROOT="${DATA_ROOT}/processed/panaroo_with_reference_genome"
DIRECTORY_LEAF="SL258_part_0"  # Used when PANAROO_DIR is empty
PANAROO_DIR=""                 # Full path override; leave empty to use DIRECTORY_LEAF

METADATA_PATH="${DATA_ROOT}/final/metadata_final_curated_all_samples_and_columns.tsv"
MIN_GROUP_SIZE=250             # Min Clonal group / K_locus size for its own slice
GPA_FILTER_CUTOFF=""           # e.g. 20; leave empty for auto
MERGE_SMALL_CLUSTERS=""        # e.g. 15; leave empty for auto
REFERENCE_TOP_N=10
SHELL_CLOUD_CUTOFF=0.15
CORE_SHELL_CUTOFF=0.95
REPORT_TIMES=false
# Cut-down modes for login-node / slow-HPC execution. Both default to false.
#   SKIP_CLUSTERING=true: skip scanpy neighbors/UMAP/Leiden/merge, UMAP plots,
#     quality metrics, and rank_genes_groups. scanpy is then never imported,
#     which removes a multi-minute import cost on congested RDS. Clustering
#     columns in the output TSV are emitted as NaN.
#   SKIP_JACCARD=true: skip MGH78578 / RefSeq / complete-Norway cohort Jaccard
#     summaries. Distance columns emitted as NaN; refseq / norway / mgh counts
#     are preserved. With SKIP_CLUSTERING=true too, this gives a pure
#     presence/absence + feature-stats run (stats-only).
SKIP_CLUSTERING=false
SKIP_JACCARD=false
# Stage project .venv to node-local scratch before running Python.
# On a congested RDS this can turn "imports hang for >1h" into "imports
# finish in seconds". Disable by setting STAGE_VENV=false if the rsync
# itself is slow on a given run.
STAGE_VENV=false
# -------------------------------------------------------

echo "========================================================================"
echo "GPA distances: single Panaroo run (gpa_distances_single_run.py)"
echo "CC-Lake"
echo "--------------------------------"
echo "Stage VENV: ${STAGE_VENV}"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
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
        echo "venv staging: using ${STAGED_VENV}/bin/python"
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
  "${PYTHON_CMD[@]}" src/bacotype/tl/gpa_distances_single_run.py
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
CMD+=(--skip-clustering "${SKIP_CLUSTERING}")
CMD+=(--skip-jaccard "${SKIP_JACCARD}")

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

# Run with: sbatch slurm_scripts/gpa_distances_single_run.sh
# Check: squeue -u dca36
# Cancel: scancel <jobid>
# Toggle venv staging off for a run: edit STAGE_VENV=false near the top.
