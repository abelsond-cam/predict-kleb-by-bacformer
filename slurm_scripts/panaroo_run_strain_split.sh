#!/bin/bash
#SBATCH --job-name=panaroo_split
#SBATCH --output=panaroo_split_%A_%a.out
#SBATCH --error=panaroo_split_%A_%a.err
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=38
#SBATCH --time=36:00:00
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU
#SBATCH --array=1-2
#
# Two-way Panaroo split: array task 1 -> part 1, task 2 -> part 2.
# Same Python entrypoint as panaroo_run_strain.sh with --split <1|2>.
# Requires --clonal-group OR --sublineage. OUTDIR must match the canonical
# CGx_all run so converted_gff symlinks resolve under CGx_all/converted_gff/.
#
#   sbatch slurm_scripts/panaroo_run_strain_split.sh --clonal-group CG14
#
# Config (CLI flags or edit defaults below):
#   CLONAL_GROUP / SUBLINEAGE    (one required)
#   SAMPLE_METADATA_FILE, CLEAN_MODE, OUTDIR  (same as panaroo_run_strain.sh)
#
# Note: sbatch forwards arguments after the script path to this script as "$@".
# Example: sbatch slurm_scripts/panaroo_run_strain_split.sh --clonal-group CG147

echo "========================================================================" >&2
echo "panaroo_run_strain_split.sh: raw batch arguments ($# args):" >&2
i=0
for _a in "$@"; do
  i=$((i + 1))
  printf "  [%d]=%q\n" "$i" "$_a" >&2
done
echo "========================================================================" >&2

while [[ $# -gt 0 ]]; do
  case "$1" in
    --clonal-group)
      if [[ -z "${2:-}" ]]; then echo "ERROR: --clonal-group needs a value" >&2; exit 1; fi
      CLONAL_GROUP="$2"
      shift 2
      ;;
    --sublineage)
      if [[ -z "${2:-}" ]]; then echo "ERROR: --sublineage needs a value" >&2; exit 1; fi
      SUBLINEAGE="$2"
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
    --sample-metadata-file)
      SAMPLE_METADATA_FILE="$2"
      shift 2
      ;;
    *)
      echo "Warning: ignoring unknown argument $1" >&2
      shift
      ;;
  esac
done

: "${CLEAN_MODE:=strict}"
: "${OUTDIR:=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/panaroo_run}"

if [[ -n "$CLONAL_GROUP" && -n "$SUBLINEAGE" ]]; then
  echo "ERROR: --clonal-group and --sublineage are mutually exclusive." >&2
  exit 1
fi
if [[ -z "$CLONAL_GROUP" && -z "$SUBLINEAGE" ]]; then
  echo "ERROR: Provide --clonal-group or --sublineage for split runs." >&2
  exit 1
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "ERROR: This script is intended for Slurm array jobs (#SBATCH --array=1-2)." >&2
  exit 1
fi

if [[ -n "$CLONAL_GROUP" ]]; then
  STRAIN_LABEL="$CLONAL_GROUP"
elif [[ -n "$SUBLINEAGE" ]]; then
  STRAIN_LABEL="$SUBLINEAGE"
fi

echo "Parsed strain label (exact, used for paths and Python): $(printf %q "$STRAIN_LABEL")" >&2

RUN_SUBDIR="${OUTDIR}/${STRAIN_LABEL}_part${SLURM_ARRAY_TASK_ID}"
PANAROO_INPUT="${RUN_SUBDIR}/panaroo_input.txt"
CONVERTED_GFF_DIR="${RUN_SUBDIR}/converted_gff"
GENE_PRESENCE_CSV="${RUN_SUBDIR}/gene_presence_absence.csv"

cd /home/dca36/workspace/Bacotype
export PYTHONUNBUFFERED=1

echo "========================================================================"
echo "MODE: Panaroo two-way SPLIT (Slurm array 1–2; use panaroo_run_strain_split.sh only)"
echo "  This is part ${SLURM_ARRAY_TASK_ID} of 2  (part 1 = first half of shuffled samples, part 2 = second)"
echo "  strain=${STRAIN_LABEL}  clean_mode=${CLEAN_MODE}"
echo "  RUN_SUBDIR=${RUN_SUBDIR}"
echo "  Reuses combined GFFs from: ${OUTDIR}/${STRAIN_LABEL}/converted_gff/*.gff (symlinks) when that run exists"
echo "Job ID: $SLURM_JOB_ID  ArrayJob ID: ${SLURM_ARRAY_JOB_ID:-?}  Array task: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME  CPUs: $SLURM_CPUS_PER_TASK"
echo "========================================================================"
echo ""

if command -v micromamba &>/dev/null; then
  eval "$(micromamba shell hook --shell bash)"
  micromamba activate panaroo
else
  echo "micromamba not found; ensure panaroo is on PATH"
fi

export TMPDIR="${RUN_SUBDIR}/tmp_${SLURM_JOB_ID:-$$}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$TMPDIR"

# Cleanup temp dirs on any exit; remove converted_gff only after success.
cleanup_tmp_dirs() {
  if [[ -n "${RUN_SUBDIR:-}" && -d "$RUN_SUBDIR" ]]; then
    shopt -s nullglob
    local tmp_dirs=("$RUN_SUBDIR"/tmp*)
    shopt -u nullglob
    local tmp_dir
    for tmp_dir in "${tmp_dirs[@]}"; do
      if [[ -d "$tmp_dir" ]]; then
        rm -rf "$tmp_dir"
        echo "Cleaned stale tmp dir: $tmp_dir"
      fi
    done
  fi
}

cleanup_run_dirs() {
  if [[ -n "${CONVERTED_GFF_DIR:-}" && -d "$CONVERTED_GFF_DIR" ]]; then
    rm -rf "$CONVERTED_GFF_DIR"
    echo "Cleaned converted_gff dir: $CONVERTED_GFF_DIR"
  fi
}

trap cleanup_tmp_dirs EXIT

echo "Using TMPDIR: $TMPDIR"
echo ""

# Quoted argv array — do not use a single unquoted string for strain (preserves case/spaces).
PY=(src/bacotype/pp/panaroo_run_strain.py)
if [[ -n "$CLONAL_GROUP" ]]; then
  PY+=(--clonal-group "$CLONAL_GROUP")
elif [[ -n "$SUBLINEAGE" ]]; then
  PY+=(--sublineage "$SUBLINEAGE")
fi
if [[ -n "${SAMPLE_METADATA_FILE:-}" ]]; then
  PY+=(--sample-metadata-file "$SAMPLE_METADATA_FILE")
fi
PY+=(--n -1 --split "${SLURM_ARRAY_TASK_ID}" --outdir "$OUTDIR")

echo "Running: uv run python $(printf '%q ' "${PY[@]}")" >&2
uv run python "${PY[@]}"

py_exit=$?
if [[ $py_exit -ne 0 ]]; then
  echo "ERROR: panaroo_run_strain.py exited with code $py_exit (split prep / metadata failed)." >&2
  echo "  Fix the Python error above; panaroo was not started." >&2
  echo "  (Would have written: $PANAROO_INPUT)" >&2
  exit "$py_exit"
fi

if [[ ! -f "$PANAROO_INPUT" ]]; then
  echo "ERROR: Panaroo input file missing after successful Python exit: $PANAROO_INPUT" >&2
  exit 1
fi

echo ""
echo "Running Panaroo: -i $PANAROO_INPUT -o $RUN_SUBDIR --clean-mode $CLEAN_MODE -t $SLURM_CPUS_PER_TASK"
echo ""

panaroo \
  -i "$PANAROO_INPUT" \
  -o "$RUN_SUBDIR" \
  --clean-mode "$CLEAN_MODE" \
  --remove-invalid-genes \
  -t "${SLURM_CPUS_PER_TASK:-76}"

panaroo_exit_code=$?
if [[ $panaroo_exit_code -ne 0 ]]; then
  echo "ERROR: panaroo failed with exit code $panaroo_exit_code." >&2
  echo "  TMPDIR=$TMPDIR" >&2
  echo "  CONVERTED_GFF_DIR=$CONVERTED_GFF_DIR" >&2
  exit "$panaroo_exit_code"
fi

echo ""
csv_found=0
if [[ -f "$GENE_PRESENCE_CSV" ]]; then
  csv_found=1
  echo "gene_presence_absence.csv dimensions:"
  awk -F',' 'END{print "rows=" NR-1 ", cols=" NF}' "$GENE_PRESENCE_CSV"
else
  echo "Warning: gene_presence_absence.csv not found: $GENE_PRESENCE_CSV"
fi

if [[ "$csv_found" -eq 1 ]]; then
  echo "Verified GPA output; cleaning intermediates for this part only."
  cleanup_run_dirs
else
  echo "Keeping converted_gff because GPA table is missing."
fi

echo ""
echo "========================================================================"
echo "Panaroo split part ${SLURM_ARRAY_TASK_ID} finished."
echo "========================================================================"
