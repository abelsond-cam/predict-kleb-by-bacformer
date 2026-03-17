#!/bin/bash
#SBATCH --job-name=panaroo_run_strain
#SBATCH --output=panaroo_run_strain_%j.out
#SBATCH --error=panaroo_run_strain_%j.err
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=19
#SBATCH --time=04:00:00
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU
# Run Panaroo for a clonal group: build input list (one combined GFF+FASTA per sample) then run panaroo.
# Config (edit these, set env, or pass as CLI flags):
#   CLONAL_GROUP    default CG11      (or --clonal-group CG258)
#   CLEAN_MODE      default strict    (strict|moderate|sensitive or --clean-mode)
#   N_SAMPLES       default -1        (all; use --n 10 for a test run)
#   OUTDIR          base dir for run subdirs (e.g. panaroo_run/CG11_n10, or --outdir ...)

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
    --clean-mode)
      CLEAN_MODE="$2"
      shift 2
      ;;
    --outdir)
      OUTDIR="$2"
      shift 2
      ;;
    *)
      echo "Warning: ignoring unknown argument $1" >&2
      shift
      ;;
  esac
done

: "${CLONAL_GROUP:=CG11}"
: "${CLEAN_MODE:=strict}"
: "${N_SAMPLES:=-1}"
: "${OUTDIR:=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/panaroo_run}"

if [[ "$N_SAMPLES" == "-1" ]]; then
  RUN_SUBDIR_NAME="${CLONAL_GROUP}_all"
else
  RUN_SUBDIR_NAME="${CLONAL_GROUP}_n${N_SAMPLES}"
fi
RUN_SUBDIR="${OUTDIR}/${RUN_SUBDIR_NAME}"
PANAROO_INPUT="${RUN_SUBDIR}/panaroo_input.txt"

cd /home/dca36/workspace/Bacotype
export PYTHONUNBUFFERED=1

echo "========================================================================"
echo "Panaroo run: clonal_group=${CLONAL_GROUP} n=${N_SAMPLES} clean_mode=${CLEAN_MODE}"
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
  --clonal-group "$CLONAL_GROUP" \
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

# Run with: sbatch slurm_scripts/panaroo_run_strain.sh
# Test run (e.g. 10 samples): sbatch slurm_scripts/panaroo_run_strain.sh --n 10
# Different clonal group: sbatch slurm_scripts/panaroo_run_strain.sh --clonal-group CG258
