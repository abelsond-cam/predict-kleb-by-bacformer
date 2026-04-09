#!/bin/bash
#SBATCH --job-name=panaroo_refmeta
#SBATCH --output=panaroo_refmeta_%A_%a.out
#SBATCH --error=panaroo_refmeta_%A_%a.err
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --time=36:00:00
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU
#SBATCH --array=1-1%1
#
# One Slurm array task per line in a TSV list file: runs panaroo_run_strain.sh with
# --sample-metadata-file for that path. Override the array on sbatch so 1..N matches
# the line count (see commented instructions at the bottom of this file).
#
# sbatch forwards arguments after the script path to this script as "$@".

while [[ $# -gt 0 ]]; do
  case "$1" in
    --list-file)
      if [[ -z "${2:-}" ]]; then echo "ERROR: --list-file needs a value" >&2; exit 1; fi
      LIST_FILE="$2"
      shift 2
      ;;
    --clean-mode)
      CLEAN_MODE="$2"
      shift 2
      ;;
    --outdir)
      OUTDIR="$2"
      shift 2
      ;;
    --n)
      N_SAMPLES="$2"
      shift 2
      ;;
    *)
      echo "Warning: ignoring unknown argument $1" >&2
      shift
      ;;
  esac
done

: "${LIST_FILE:=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/panaroo_ref_tsvs.list}"
: "${CLEAN_MODE:=strict}"
: "${N_SAMPLES:=-1}"
: "${OUTDIR:=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/panaroo_run}"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "ERROR: This script is for Slurm array jobs. Submit with sbatch --array=1-N%M ..." >&2
  exit 1
fi

if [[ ! -f "$LIST_FILE" || ! -r "$LIST_FILE" ]]; then
  echo "ERROR: LIST_FILE missing or unreadable: $LIST_FILE" >&2
  exit 1
fi

TSV=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST_FILE")
TSV="${TSV%%$'\r'}"
TSV="${TSV#"${TSV%%[![:space:]]*}"}"
TSV="${TSV%"${TSV##*[![:space:]]}"}"

if [[ -z "$TSV" ]]; then
  echo "ERROR: Line ${SLURM_ARRAY_TASK_ID} of LIST_FILE is empty: $LIST_FILE" >&2
  exit 1
fi

if [[ ! -f "$TSV" ]]; then
  echo "ERROR: TSV path from list line ${SLURM_ARRAY_TASK_ID} does not exist: $TSV" >&2
  exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STRAIN_SCRIPT="${SCRIPT_DIR}/panaroo_run_strain.sh"

echo "========================================================================"
echo "panaroo_run_strain_metadata_array: array_task=${SLURM_ARRAY_TASK_ID}  job=${SLURM_JOB_ID:-local}"
echo "  LIST_FILE=${LIST_FILE}"
echo "  TSV=$(printf %q "$TSV")"
echo "  OUTDIR=${OUTDIR}  clean_mode=${CLEAN_MODE}  n=${N_SAMPLES}"
echo "========================================================================"
echo ""

exec bash "$STRAIN_SCRIPT" \
  --sample-metadata-file "$TSV" \
  --outdir "$OUTDIR" \
  --clean-mode "$CLEAN_MODE" \
  --n "$N_SAMPLES"

# -----------------------------------------------------------------------------
# 1) Generate the list (one absolute path per line), then submit:
#
# find /home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/panaroo_with_reference_genome \
#   -maxdepth 1 -name '*.tsv' -type f | sort \
#   > /home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/panaroo_ref_tsvs.list
#
# 2) Submit the array so task IDs 1..N match line numbers (%M = max concurrent tasks):
#
# cd /home/dca36/workspace/Bacotype
# sbatch --array=1-$(wc -l < /home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/panaroo_ref_tsvs.list)%8 \
#   slurm_scripts/panaroo_run_strain_metadata_array.sh
#
# Optional: override list path or Panaroo output base:
#   ... slurm_scripts/panaroo_run_strain_metadata_array.sh \
#       --list-file /path/to/other.list --outdir /path/to/panaroo_run --clean-mode strict
#
# Reporting: logs panaroo_refmeta_<array_job_id>_<task_id>.out/.err ; then e.g.
#   sacct -j <ARRAY_JOB_ID> --format=JobID,JobName%25,State,ExitCode,Elapsed,AllocCPUS
# -----------------------------------------------------------------------------