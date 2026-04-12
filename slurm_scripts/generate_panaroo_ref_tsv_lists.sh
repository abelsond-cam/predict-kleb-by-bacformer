#!/usr/bin/env bash
# Generate phased .list files for panaroo_run_strain_metadata_array.sh from a
# panaroo_metadata_batching.py output directory.
#
# panaroo_ref_tsvs_all.list order (concatenation of the five category lists):
#   1) SL258 multi-part TSVs only          -> panaroo_ref_tsvs_sl258_parts.list
#   2) Other *_part_*.tsv (not SL258)      -> panaroo_ref_tsvs_split_parts_other.list
#   3) Large single-lineage TSVs           -> panaroo_ref_tsvs_large_single.list
#   4) species_*.tsv                       -> panaroo_ref_tsvs_species.list
#   5) kp_rare_sublineage_batch_*.tsv      -> panaroo_ref_tsvs_kp_rare.list
#
# All five partitions must be non-empty or the script exits with an error (wrong
# directory, batching not run, or unexpected layout). If you ever have only SL258
# part files and no other *_part_*, you will need a one-off list or to relax that
# check.
#
# Detailed batching rules and reference-genome handling are logged by
# panaroo_metadata_batching.py in panaroo_batching.log under the batch directory.
#
# Usage:
#   bash slurm_scripts/generate_panaroo_ref_tsv_lists.sh [BATCH_DIR]
#
# panaroo_metadata_batching.py writes TSVs + panaroo_batching.log under
#   <output_dir>/batches/
# This script's default BATCH_DIR is that batches folder. List files are written
# there next to the TSVs. find is not limited to depth 1 so lists can be
# regenerated after array jobs move each TSV into <batches>/<stem>/<stem>.tsv.
#
# Align SL258 filenames with --sl258-name (default SL258):
#   SL258_PREFIX=SL258 bash slurm_scripts/generate_panaroo_ref_tsv_lists.sh

set -euo pipefail

DEFAULT_BATCH_DIR="/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/panaroo_with_reference_genome/batches"
SL258_PREFIX="${SL258_PREFIX:-SL258}"

usage() {
  sed -n '1,/^set -euo pipefail$/p' "$0" | sed '$d'
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

BATCH_DIR="${1:-${BATCH_DIR:-$DEFAULT_BATCH_DIR}}"
if [[ -n "${2:-}" ]]; then
  echo "ERROR: this script takes at most one argument (BATCH_DIR); list files are always written there." >&2
  echo "  Extra argument: $2" >&2
  exit 1
fi

if [[ ! -d "$BATCH_DIR" ]]; then
  echo "ERROR: batch directory does not exist: $BATCH_DIR" >&2
  exit 1
fi

BATCH_DIR="$(cd "$BATCH_DIR" && pwd)"
LOG_FILE="${BATCH_DIR}/panaroo_batching.log"

sl258_glob="${SL258_PREFIX}_part_*.tsv"
mapfile -t paths_sl258 < <(find "$BATCH_DIR" -type f -name "$sl258_glob" | sort)
mapfile -t paths_other_parts < <(
  find "$BATCH_DIR" -type f -name '*_part_*.tsv' ! -name "$sl258_glob" | sort
)
mapfile -t paths_large_single < <(
  find "$BATCH_DIR" -type f -name '*.tsv' \
    ! -name '*_part_*' ! -name 'species_*' ! -name 'kp_rare_sublineage_batch_*' | sort
)
mapfile -t paths_species < <(find "$BATCH_DIR" -type f -name 'species_*.tsv' | sort)
mapfile -t paths_kp_rare < <(
  find "$BATCH_DIR" -type f -name 'kp_rare_sublineage_batch_*.tsv' | sort
)

fail_empty() {
  local category="$1" predicate="$2"
  echo "ERROR: no TSV files matched for category: ${category}" >&2
  echo "  Batch directory: ${BATCH_DIR}" >&2
  echo "  Expected (under BATCH_DIR, any depth): ${predicate}" >&2
  echo "  Fix: run panaroo_metadata_batching.py into this directory, or point BATCH_DIR at the correct path." >&2
  exit 1
}

if (( ${#paths_sl258[@]} == 0 )); then
  fail_empty "sl258_parts" "find BATCH_DIR -type f -name '${sl258_glob}'"
fi
if (( ${#paths_other_parts[@]} == 0 )); then
  fail_empty "split_parts_other" "find BATCH_DIR -type f -name '*_part_*.tsv' ! -name '${sl258_glob}'"
fi
if (( ${#paths_large_single[@]} == 0 )); then
  fail_empty "large_single" "find BATCH_DIR -type f -name '*.tsv' ! -name '*_part_*' ! -name 'species_*' ! -name 'kp_rare_sublineage_batch_*'"
fi
if (( ${#paths_species[@]} == 0 )); then
  fail_empty "species" "find BATCH_DIR -type f -name 'species_*.tsv'"
fi
if (( ${#paths_kp_rare[@]} == 0 )); then
  fail_empty "kp_rare" "find BATCH_DIR -type f -name 'kp_rare_sublineage_batch_*.tsv'"
fi

echo "========================================================================"
echo "generate_panaroo_ref_tsv_lists.sh"
echo "  Batch directory (TSVs, panaroo_batching.log, and *.list outputs): $BATCH_DIR"
echo "  Batching log (splits, refs, row counts): $LOG_FILE"
echo ""
echo "  Lists to write:"
echo "    - panaroo_ref_tsvs_sl258_parts.list       (SL258 multi-part batches)"
echo "    - panaroo_ref_tsvs_split_parts_other.list (other *_part_*.tsv)"
echo "    - panaroo_ref_tsvs_large_single.list      (one TSV per large unsplit SL)"
echo "    - panaroo_ref_tsvs_species.list           (species_*.tsv)"
echo "    - panaroo_ref_tsvs_kp_rare.list           (kp_rare_sublineage_batch_*.tsv)"
echo "    - panaroo_ref_tsvs_all.list               (concat of the five, fixed order)"
echo "========================================================================"
echo ""

out_sl258="${BATCH_DIR}/panaroo_ref_tsvs_sl258_parts.list"
out_other="${BATCH_DIR}/panaroo_ref_tsvs_split_parts_other.list"
out_large="${BATCH_DIR}/panaroo_ref_tsvs_large_single.list"
out_species="${BATCH_DIR}/panaroo_ref_tsvs_species.list"
out_kp="${BATCH_DIR}/panaroo_ref_tsvs_kp_rare.list"
out_all="${BATCH_DIR}/panaroo_ref_tsvs_all.list"

printf '%s\n' "${paths_sl258[@]}" >"$out_sl258"
printf '%s\n' "${paths_other_parts[@]}" >"$out_other"
printf '%s\n' "${paths_large_single[@]}" >"$out_large"
printf '%s\n' "${paths_species[@]}" >"$out_species"
printf '%s\n' "${paths_kp_rare[@]}" >"$out_kp"

cat "$out_sl258" "$out_other" "$out_large" "$out_species" "$out_kp" >"$out_all"

for f in "$out_sl258" "$out_other" "$out_large" "$out_species" "$out_kp" "$out_all"; do
  n=$(wc -l < "$f")
  echo "  wrote $n lines -> $f"
done

cat <<EOF

Done. Submit example (from repo root, adjust %concurrency):
  sbatch --array=1-\$(wc -l < ${out_sl258})%8 \\
    slurm_scripts/panaroo_run_strain_metadata_array.sh --list-file ${out_sl258}
  (repeat for split_parts_other, large_single, species, kp_rare, or use --list-file ${out_all} once)
EOF
