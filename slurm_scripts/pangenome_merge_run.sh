#!/bin/bash
#SBATCH --job-name=merge_SL258
#SBATCH --output=/home/dca36/workspace/Bacotype/logs/pangenome_merge/merge_SL258_%j.out
#SBATCH --error=/home/dca36/workspace/Bacotype/logs/pangenome_merge/merge_SL258_%j.err
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=38
#SBATCH --time=24:00:00
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU
#
# Run pangenomerge from the cloned repo (not bioconda binary): micromamba env
# + PYTHONPATH so top-level custom_functions imports resolve.
#
# Configure (pick one):
#   1) Edit the defaults below, or
#   2) sbatch slurm_scripts/pangenome_merge_run.sh --component-graphs /path/to/paths.tsv --outdir /path/to/merge_out
#
# Optional env vars:
#   THREADS           default: SLURM_CPUS_PER_TASK
#   PANGENOME_MERGE_ROOT  default: /home/dca36/workspace/Bacotype/pangenome_merge
#   EXTRA_ARGS        extra args appended to the runner (quoted string), e.g. '--family-threshold 0.7'

set -euo pipefail

# Defaults — override with environment variables when submitting.

# COMPONENT_GRAPHS_TSV=/path/to/paths.tsv - requires full path
# ROOT DATA DIR:
ROOT_DATA_DIR="/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/pangenomerge"
# Component graphs to merge as a .tsv file
: "${COMPONENT_GRAPHS_TSV:=$ROOT_DATA_DIR/SL258_component_graphs.tsv}"  
# OUTDIR=/path/to/merge_out - requires full path
: "${OUTDIR:=$ROOT_DATA_DIR/SL258}"

# Other defaults
: "${PANGENOME_MERGE_ROOT:=/home/dca36/workspace/Bacotype/pangenome_merge}"
: "${THREADS:=${SLURM_CPUS_PER_TASK:-16}}"
: "${EXTRA_ARGS:=}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --component-graphs)
      COMPONENT_GRAPHS_TSV="$2"
      shift 2
      ;;
    --outdir)
      OUTDIR="$2"
      shift 2
      ;;
    --threads)
      THREADS="$2"
      shift 2
      ;;
    *)
      echo "Warning: ignoring unknown argument $1" >&2
      shift
      ;;
  esac
done

if [[ -z "$COMPONENT_GRAPHS_TSV" ]]; then
  echo "ERROR: Set COMPONENT_GRAPHS_TSV (TSV of Panaroo output directory paths) or pass --component-graphs PATH" >&2
  exit 1
fi
if [[ -z "$OUTDIR" ]]; then
  echo "ERROR: Set OUTDIR or pass --outdir PATH" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1

echo "========================================================================"
echo "pangenomerge (runner + PYTHONPATH workaround)"
echo "  COMPONENT_GRAPHS_TSV=${COMPONENT_GRAPHS_TSV}"
echo "  OUTDIR=${OUTDIR}"
echo "  THREADS=${THREADS}"
echo "  PANGENOME_MERGE_ROOT=${PANGENOME_MERGE_ROOT}"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "CPUs (Slurm): ${SLURM_CPUS_PER_TASK:-n/a}"
echo "========================================================================"
echo ""

if [[ ! -f "$COMPONENT_GRAPHS_TSV" ]]; then
  echo "ERROR: component graphs TSV not found: $COMPONENT_GRAPHS_TSV" >&2
  exit 1
fi

if command -v micromamba &>/dev/null; then
  eval "$(micromamba shell hook --shell bash)"
  micromamba activate pangenomerge
else
  echo "ERROR: micromamba not found on PATH" >&2
  exit 1
fi

cd "$PANGENOME_MERGE_ROOT"
export PYTHONPATH=".:pangenomerge"

# Optional: large temp space (edit if your site prefers SCRATCH)
if [[ -n "${SLURM_TMPDIR:-}" ]]; then
  export TMPDIR="${SLURM_TMPDIR}/pangenomerge_${SLURM_JOB_ID:-$$}"
  mkdir -p "$TMPDIR"
fi

set -x
python3 pangenomerge-runner.py \
  --component-graphs "$COMPONENT_GRAPHS_TSV" \
  --outdir "$OUTDIR" \
  --threads "$THREADS" \
  ${EXTRA_ARGS}
set +x

echo ""
echo "========================================================================"
echo "Job complete!"
echo "========================================================================"

# Examples:
#   COMPONENT_GRAPHS_TSV=/path/to/paths.tsv OUTDIR=/path/to/merge_out sbatch slurm_scripts/pangenome_merge_run.sh
#   sbatch slurm_scripts/pangenome_merge_run.sh   # after editing defaults at top
# Check: squeue -u $USER
# Cancel: scancel <jobid>
