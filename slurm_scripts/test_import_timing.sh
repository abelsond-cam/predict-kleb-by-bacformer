#!/bin/bash
#SBATCH --job-name=test_icelake_himem_import_timing
#SBATCH --output=test_icelake_himem_import_timing_%j.out
#SBATCH --error=test_icelake_himem_import_timing_%j.err
#SBATCH --partition=cclake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:45:00
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU
#SBATCH --mem=8G
#
# test_cclake_import_timing.sh
# -----------------------
# Compares cold-import latency for the GPA dependency stack:
#
#   Phase A — ``uv run`` (project .venv symlink → RDS-backed packages)
#   Phase B — ``rsync`` the *resolved* venv to node-local scratch, then run the
#             same probe with ``$DEST/bin/python`` (no uv; all reads from local
#             disk, **does not use workspace quota**).
#
# Scratch dir: $SLURM_TMPDIR (preferred on many clusters), else $TMPDIR, else /tmp.
#
# We do **not** run ``du -sh`` on the venv by default: on NFS/RDS it walks the
# entire tree and can take many minutes. To try anyway: ``RUN_VENV_DU=1 sbatch …``
# (capped at 120s with ``timeout``).
#

set -euo pipefail
cd /home/dca36/workspace/Bacotype

export PYTHONUNBUFFERED=1

SCRATCH="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
DEST="${SCRATCH}/bacotype_venv_import_probe"
PROBE="src/bacotype/tl/test_import_timing_standalone.py"

echo "========================================================================"
echo "Import timing: IceLake-HiMem (uv) vs node-local venv copy"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURM_NODELIST:-$(hostname)}"
echo "Scratch: ${SCRATCH}"
echo "Venv copy destination: ${DEST}"
echo "========================================================================"
echo ""

if [[ ! -L .venv && ! -d .venv ]]; then
  echo "ERROR: .venv not found in $(pwd)" >&2
  exit 1
fi

VENV_REAL="$(readlink -f .venv)"
echo "Resolved .venv -> ${VENV_REAL}"
if [[ ! -d "${VENV_REAL}" ]]; then
  echo "ERROR: resolved venv path is not a directory: ${VENV_REAL}" >&2
  exit 1
fi

echo "========================================================================"
echo "Phase A: uv run (RDS-backed site-packages via symlinked .venv)"
echo "========================================================================"
/usr/bin/time -v uv run python -u "${PROBE}" --label "phase_a_uv_rds"

echo ""
echo "========================================================================"
echo "Phase B: rsync venv -> ${DEST}  (node-local scratch)"
echo "========================================================================"
rm -rf "${DEST}"
mkdir -p "${DEST}"
echo "rsync starting $(date -Is) ..."
/usr/bin/time -v rsync -a "${VENV_REAL}/" "${DEST}/"
echo "rsync finished $(date -Is)"

if [[ ! -x "${DEST}/bin/python" ]]; then
  echo "ERROR: ${DEST}/bin/python missing or not executable" >&2
  exit 1
fi

echo ""
echo "========================================================================"
echo "Phase C: copied venv python (cold imports from local disk)"
echo "========================================================================"
/usr/bin/time -v "${DEST}/bin/python" -u "${PROBE}" --label "phase_c_local_scratch"

echo ""
echo "========================================================================"
echo "All phases done $(date -Is)"
echo "========================================================================"
echo ""
echo "Optional cleanup (frees node-local space for other jobs on this node):"
echo "  rm -rf ${DEST}"

# Run with: sbatch slurm_scripts/test_import_timing.sh
