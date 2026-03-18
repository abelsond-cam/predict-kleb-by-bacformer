#!/bin/bash
#SBATCH --job-name=panaroo_run_strain
#SBATCH --output=panaroo_run_strain_%j.out
#SBATCH --error=panaroo_run_strain_%j.err
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=38
#SBATCH --time=04:00:00
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU
# Run Panaroo for a strain (clonal group OR sublineage) or all samples:
#   build input list (one combined GFF+FASTA per sample) then run panaroo.
# Optionally provide --clonal-group or --sublineage; if neither is given, all
# samples in the metadata file are used.
# Config (edit these, set env, or pass as CLI flags):
#   CLONAL_GROUP              e.g. CG11         (--clonal-group CG258)
#   SUBLINEAGE                e.g. SL_123       (--sublineage SL_123)
#   SAMPLE_METADATA_FILE      path to TSV       (--sample-metadata-file /path/to/file.tsv)
#   CLEAN_MODE                default strict    (strict|moderate|sensitive or --clean-mode)
#   N_SAMPLES                 default -1        (all; use --n 10 for a test run)
#   OUTDIR                    base dir for run subdirs (e.g. panaroo_run/CG11_n10, or --outdir ...)

# Simple CLI flag parsing so `sbatch ... --n 10` etc. work.
while [[ $# -gt 0 ]]; do
  case "$1" in
    --n)
      N_SAMPLES="$2"
      shift 2
      ;;
    --clonal-group)
      CLONAL_GROUP="$2"
      shift 2
      ;;
    --sublineage)
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
: "${N_SAMPLES:=-1}"
: "${OUTDIR:=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/panaroo_run}"

# --clonal-group and --sublineage are mutually exclusive.
if [[ -n "$CLONAL_GROUP" && -n "$SUBLINEAGE" ]]; then
  echo "ERROR: --clonal-group and --sublineage are mutually exclusive. Provide one, not both." >&2
  exit 1
fi

PYTHON_STRAIN_ARGS=""
if [[ -n "$CLONAL_GROUP" ]]; then
  STRAIN_LABEL="$CLONAL_GROUP"
  PYTHON_STRAIN_ARGS="--clonal-group $CLONAL_GROUP"
elif [[ -n "$SUBLINEAGE" ]]; then
  STRAIN_LABEL="$SUBLINEAGE"
  PYTHON_STRAIN_ARGS="--sublineage $SUBLINEAGE"
else
  # All-samples mode: derive label from the metadata file stem.
  if [[ -n "$SAMPLE_METADATA_FILE" ]]; then
    STRAIN_LABEL="$(basename "$SAMPLE_METADATA_FILE" | sed 's/\.[^.]*$//')"
  else
    STRAIN_LABEL="metadata_final_curated_slimmed"
  fi
fi

PYTHON_METADATA_ARGS=""
if [[ -n "$SAMPLE_METADATA_FILE" ]]; then
  PYTHON_METADATA_ARGS="--sample-metadata-file $SAMPLE_METADATA_FILE"
fi

if [[ "$N_SAMPLES" == "-1" ]]; then
  RUN_SUBDIR_NAME="${STRAIN_LABEL}_all"
else
  RUN_SUBDIR_NAME="${STRAIN_LABEL}_n${N_SAMPLES}"
fi
RUN_SUBDIR="${OUTDIR}/${RUN_SUBDIR_NAME}"
PANAROO_INPUT="${RUN_SUBDIR}/panaroo_input.txt"

cd /home/dca36/workspace/Bacotype
export PYTHONUNBUFFERED=1

echo "========================================================================"
echo "Panaroo run: strain=${STRAIN_LABEL} n=${N_SAMPLES} clean_mode=${CLEAN_MODE}"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "========================================================================"
echo ""

# Activate panaroo env (micromamba)
if command -v micromamba &>/dev/null; then
  eval "$(micromamba shell hook --shell bash)"
  micromamba activate panaroo
else
  echo "micromamba not found; ensure panaroo is on PATH"
fi

# Build Panaroo input file (single-column list of combined GFF+FASTA files) and run subdir
uv run python src/bacotype/pp/panaroo_run_strain.py \
  $PYTHON_STRAIN_ARGS \
  $PYTHON_METADATA_ARGS \
  --n "$N_SAMPLES" \
  --outdir "$OUTDIR"

if [[ ! -f "$PANAROO_INPUT" ]]; then
  echo "ERROR: Panaroo input file not found: $PANAROO_INPUT"
  exit 1
fi

echo ""
echo "Running Panaroo: -i $PANAROO_INPUT -o $RUN_SUBDIR --clean-mode $CLEAN_MODE -t $SLURM_CPUS_PER_TASK"
echo ""

panaroo \
  -i "$PANAROO_INPUT" \
  -o "$RUN_SUBDIR" \
  --clean-mode "$CLEAN_MODE" \
  -t "${SLURM_CPUS_PER_TASK:-76}"

echo ""
echo "========================================================================"
echo "Panaroo run finished."
echo "========================================================================"

# Run with clonal group:       sbatch slurm_scripts/panaroo_run_strain.sh --clonal-group CG11
# Run with sublineage:         sbatch slurm_scripts/panaroo_run_strain.sh --sublineage SL123
# Test run (10 samples):       sbatch slurm_scripts/panaroo_run_strain.sh --clonal-group CG11 --n 10
# All samples (custom file):   sbatch slurm_scripts/panaroo_run_strain.sh --sample-metadata-file /path/to/samples.tsv
# All samples (default meta):  sbatch slurm_scripts/panaroo_run_strain.sh
