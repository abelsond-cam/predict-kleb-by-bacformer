#!/bin/bash
#SBATCH --job-name=species
#SBATCH --output=species_%A_%a.out
#SBATCH --error=species_%A_%a.err
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --time=36:00:00
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU
#SBATCH --array=1-1%1
#
# One Slurm array task per line in a list file: each line is an absolute path to a
# sample-metadata TSV. This script runs panaroo_run_strain.sh with
# --sample-metadata-file for that path.
#
# Pipeline (details and fallback find commands at bottom of file):
#   (i)   Generate batch TSVs with panaroo_metadata_batching.py under
#         .../panaroo_with_reference_genome/batches/ (log + TSVs).
#   (ii)  Build .list files with generate_panaroo_ref_tsv_lists.sh pointing at
#         that batches/ directory (five phased lists + panaroo_ref_tsvs_all.list).
#   (iii) Submit with sbatch --array=1-$(wc -l < list)%M ...
#
# Before each task runs Panaroo, this script copies the listed metadata TSV from
#   batches/<stem>.tsv  ->  OUTDIR/<stem>/<stem>.tsv
# (idempotent if already copied). batches/ is left intact for list regeneration.
# Panaroo output goes under OUTDIR/<stem>/ (e.g. ROOT/SL101/).
#
# Default LIST_FILE (below) applies only when you do not pass --list-file.
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

: "${LIST_FILE:=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/panaroo_with_reference_genome/batches/panaroo_ref_tsvs_all.list}"
: "${CLEAN_MODE:=strict}"
: "${N_SAMPLES:=-1}"
: "${OUTDIR:=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/panaroo_with_reference_genome}"

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

stem=$(basename -- "$TSV" .tsv)
RUN_DIR="${OUTDIR}/${stem}"
STAGED_TSV="${RUN_DIR}/${stem}.tsv"

if [[ -f "$STAGED_TSV" ]]; then
  TSV="$STAGED_TSV"
elif [[ -f "$TSV" ]]; then
  mkdir -p "$RUN_DIR"
  cp -- "$TSV" "$STAGED_TSV"
  TSV="$STAGED_TSV"
else
  echo "ERROR: TSV from list line ${SLURM_ARRAY_TASK_ID} not found at:" >&2
  echo "  list path: $(printf %q "$TSV")" >&2
  echo "  staged:    $(printf %q "$STAGED_TSV")" >&2
  exit 1
fi

REPO_DIR=/home/dca36/workspace/Bacotype
STRAIN_SCRIPT="${REPO_DIR}/slurm_scripts/panaroo_run_strain.sh"

echo "========================================================================"
echo "panaroo_run_strain_metadata_array: array_task=${SLURM_ARRAY_TASK_ID}  job=${SLURM_JOB_ID:-local}"
echo "  LIST_FILE=${LIST_FILE}"
echo "  sample_metadata_tsv=$(printf %q "$TSV")  (staged under run-named subdir)"
echo "  OUTDIR=${OUTDIR}  clean_mode=${CLEAN_MODE}  n=${N_SAMPLES}"
echo "========================================================================"
echo ""

exec bash "$STRAIN_SCRIPT" \
  --sample-metadata-file "$TSV" \
  --outdir "$OUTDIR" \
  --clean-mode "$CLEAN_MODE" \
  --n "$N_SAMPLES"

# -----------------------------------------------------------------------------
# Layout after runs:
#   ROOT/                                     (= OUTDIR, default panaroo_with_reference_genome)
#     batches/                                (generated TSVs + log + .list files, untouched by runs)
#       SL101.tsv  SL258_part_0.tsv  ...
#       panaroo_batching.log
#       panaroo_ref_tsvs_*.list
#     SL101/                                  (created by this script at run time)
#       SL101.tsv                             (copied from batches/)
#       panaroo_input.txt  converted_gff/  gene_presence_absence.csv  ...
#     SL258_part_0/
#       ...
#
# Paths for copy-paste:
#   ROOT=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/panaroo_with_reference_genome
#   BATCHES=$ROOT/batches
#   REPO=/home/dca36/workspace/Bacotype
#
# (i) Generate batch TSVs + panaroo_batching.log under $BATCHES/:
#
#   cd "$REPO"
#   uv run python src/bacotype/pp/panaroo_metadata_batching.py
#   bash slurm_scripts/generate_panaroo_ref_tsv_lists.sh
#
# (ii) Submit: use --array=1-$(wc -l < FILE)%M so N matches the line count.
#
#   BATCHES=$ROOT/batches
#   cd "$REPO"
#   sbatch --array=1-$(wc -l < "$BATCHES/panaroo_ref_tsvs_sl258_parts.list")%8 \
#     slurm_scripts/panaroo_run_strain_metadata_array.sh --list-file "$BATCHES/panaroo_ref_tsvs_sl258_parts.list"
#   sbatch --array=1-$(wc -l < "$BATCHES/panaroo_ref_tsvs_split_parts_other.list")%8 \
#     slurm_scripts/panaroo_run_strain_metadata_array.sh --list-file "$BATCHES/panaroo_ref_tsvs_split_parts_other.list"
#   sbatch --array=1-$(wc -l < "$BATCHES/panaroo_ref_tsvs_large_single.list")%8 \
#     slurm_scripts/panaroo_run_strain_metadata_array.sh --list-file "$BATCHES/panaroo_ref_tsvs_large_single.list"
#   sbatch --array=1-$(wc -l < "$BATCHES/panaroo_ref_tsvs_species.list")%8 \
#     slurm_scripts/panaroo_run_strain_metadata_array.sh --list-file "$BATCHES/panaroo_ref_tsvs_species.list"
#   sbatch --array=1-$(wc -l < "$BATCHES/panaroo_ref_tsvs_kp_rare.list")%8 \
#     slurm_scripts/panaroo_run_strain_metadata_array.sh --list-file "$BATCHES/panaroo_ref_tsvs_kp_rare.list"
#
# Optional overrides (default list under $BATCHES/; default OUTDIR is $ROOT):
#   ... slurm_scripts/panaroo_run_strain_metadata_array.sh \
#       --list-file "$BATCHES/panaroo_ref_tsvs_all.list" \
#       --outdir "$ROOT" --clean-mode strict
#
# Reporting: logs panaroo_refmeta_<array_job_id>_<task_id>.out/.err ; e.g.
#   sacct -j <ARRAY_JOB_ID> --format=JobID,JobName%25,State,ExitCode,Elapsed,AllocCPUS
# -----------------------------------------------------------------------------