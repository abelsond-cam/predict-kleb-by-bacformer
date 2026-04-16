#!/bin/bash
#SBATCH --job-name=gff_count
#SBATCH --output=logs/gff_count_%j.out
#SBATCH --error=logs/gff_count_%j.err
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=76
#SBATCH --time=03:00:00
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU
#
# count_gff_features.sh
# ---------------------
# Stream every sample's gzipped GFF listed in metadata_final_curated_slimmed.tsv
# and count occurrences of each feature type (GFF column 3), writing per-sample
# counts to a sidecar TSV (+ per-sample errors). Uses Python multiprocessing
# across all allocated cores of a single icelake node.
#
# Pipeline:
#   (i)  This job runs count_gff_features.py -> writes:
#          david/final/gff_feature_counts.tsv
#          david/final/gff_feature_counts.errors.tsv   (only if any failures)
#   (ii) Locally (no sbatch needed), merge sidecar into metadata in place:
#          uv run python -m bacotype.pp.merge_gff_feature_counts_into_metadata
#
# Submit with: sbatch slurm_scripts/count_gff_features.sh
#
# Resume: re-submitting after a partial run skips Samples already present in
# the sidecar TSV, so no work is repeated.

set -euo pipefail

REPO_DIR=/home/dca36/workspace/Bacotype
cd "$REPO_DIR"

mkdir -p logs

echo "========================================================================"
echo "count_gff_features: job=${SLURM_JOB_ID:-local}"
echo "  node=$(hostname)  cpus_per_task=${SLURM_CPUS_PER_TASK:-?}"
echo "  start=$(date -Iseconds)"
echo "========================================================================"

uv run python -m bacotype.pp.count_gff_features

echo "========================================================================"
echo "count_gff_features: done=$(date -Iseconds)"
echo "Next step (run locally):"
echo "  uv run python -m bacotype.pp.merge_gff_feature_counts_into_metadata"
echo "========================================================================"
