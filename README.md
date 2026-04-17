# Bacotype

Use Panaroo to define bacotypes by analysing gene presence/absence (GPA) in
sublineages, clonal groups and clusters, considering whole genomes (core and
accessory).

This repo combines three task areas. Each is briefly described below with the
main scripts to run and the most important options to tweak.

> Python entrypoints are always run with `uv run python ...` (the project
> ships a `pyproject.toml`). Most are wrapped by a Slurm script under
> `slurm_scripts/` for HPC execution.

---

## 1. Data preprocessing

Collect assemblies and annotations (GFF) and organise their file paths into
the project metadata TSV so that downstream Panaroo and analysis steps can
resolve every sample's FASTA + GFF.

Main scripts:
- [`src/bacotype/pp/add_paths_gff_fna_to_metadata.py`](src/bacotype/pp/add_paths_gff_fna_to_metadata.py)
  and Slurm wrapper
  [`slurm_scripts/add_paths_gff_fna_to_metadata.sh`](slurm_scripts/add_paths_gff_fna_to_metadata.sh):
  scans configured assembly / GFF roots and writes the resolved per-sample
  paths into the metadata TSV.
- [`src/bacotype/pp/count_gff_features.py`](src/bacotype/pp/count_gff_features.py)
  and Slurm wrapper
  [`slurm_scripts/count_gff_features.sh`](slurm_scripts/count_gff_features.sh):
  counts GFF feature types per sample (QC on annotations).
- [`src/bacotype/pp/merge_gff_feature_counts_into_metadata.py`](src/bacotype/pp/merge_gff_feature_counts_into_metadata.py):
  merges those counts back into the curated metadata TSV.
- Supporting utilities: `download_bakrep_gbff_files.py`,
  `add_bakta_gbff_downloaded_flag.py`, `add_poppunk_clusters_to_metadata.py`,
  `update_biosample_accessions.py`, `convert_ast_data.py`,
  `select_genomes_reference_comparison.py`.

Important settings to check before running:
- Assembly / GFF root directories in the preprocessing scripts (hard-coded or
  CLI-flag) — must point at the current data layout.
- Metadata TSV path — the curated `metadata_final_curated_*` file is the
  single source of truth used by the Panaroo and GPA-distance steps below.

Output: the curated metadata TSV at
`<DATA_ROOT>/final/metadata_final_curated_all_samples_and_columns.tsv`, which
all later steps read via a `--metadata` CLI flag.

---

## 2. Run Panaroo

Panaroo is executed via a single Python + shell pair that takes metadata
(either all rows, or a filtered subset) and builds the combined GFF+FASTA
input before invoking Panaroo. Three execution modes share this core:

### 2a. One strain (single job)

Run Panaroo on one clonal group, one sublineage, or one custom sample-list
TSV as a single Slurm job.

- Python: [`src/bacotype/pp/panaroo_run_strain.py`](src/bacotype/pp/panaroo_run_strain.py)
- Slurm: [`slurm_scripts/panaroo_run_strain.sh`](slurm_scripts/panaroo_run_strain.sh)

Examples:
```bash
# One clonal group:
sbatch slurm_scripts/panaroo_run_strain.sh --clonal-group CG11

# One sublineage:
sbatch slurm_scripts/panaroo_run_strain.sh --sublineage SL123

# Custom sample list (overrides CG/SL filters):
sbatch slurm_scripts/panaroo_run_strain.sh \
  --sample-metadata-file /path/to/samples.tsv

# Quick test with N samples:
sbatch slurm_scripts/panaroo_run_strain.sh --clonal-group CG11 --n 10
```

Important options (CLI flags, see header of the `.sh` for the full list):
- `--clonal-group` / `--sublineage` (mutually exclusive) — strain to Panaroo.
- `--sample-metadata-file` — arbitrary precomputed sample list.
- `--n` — cap for quick testing (`-1` = all).
- `--clean-mode` — Panaroo cleanup (`strict` | `moderate` | `sensitive`).
- `--outdir` — base directory for per-run subdirs (default:
  `.../processed/panaroo_run`).

### 2b. Whole dataset split by lineage, with reference genomes (array job)

To cover the whole dataset, we pre-compute per-lineage sample lists, include
the reference genomes in each, and submit them as a Slurm array. Each array
task calls the single-strain runner on one TSV.

- Python: [`src/bacotype/pp/panaroo_metadata_batching.py`](src/bacotype/pp/panaroo_metadata_batching.py)
  generates per-lineage batch TSVs plus a log under
  `<PANAROO_RUN_ROOT>/batches/`.
- Shell helper: [`slurm_scripts/generate_panaroo_ref_tsv_lists.sh`](slurm_scripts/generate_panaroo_ref_tsv_lists.sh)
  builds `.list` files (one path per line) grouping the generated TSVs into
  phased batches (`panaroo_ref_tsvs_*.list`).
- Slurm: [`slurm_scripts/panaroo_run_strain_metadata_array.sh`](slurm_scripts/panaroo_run_strain_metadata_array.sh)
  reads one TSV path per array index and execs `panaroo_run_strain.sh`.

Typical pipeline:
```bash
ROOT=/path/to/processed/panaroo_with_reference_genome
BATCHES=$ROOT/batches

# (i) Generate batch TSVs + .list files:
uv run python src/bacotype/pp/panaroo_metadata_batching.py
bash slurm_scripts/generate_panaroo_ref_tsv_lists.sh

# (ii) Submit one array per phased list (size N = line count of the list):
sbatch --array=1-$(wc -l < "$BATCHES/panaroo_ref_tsvs_sl258_parts.list")%8 \
  slurm_scripts/panaroo_run_strain_metadata_array.sh \
  --list-file "$BATCHES/panaroo_ref_tsvs_sl258_parts.list"
# (repeat for split_parts_other, large_single, species, kp_rare, or use
#  panaroo_ref_tsvs_all.list)
```

Important options:
- `--list-file` — which phased `.list` to submit (default: `panaroo_ref_tsvs_all.list`).
- `--outdir` — base run directory (default: `panaroo_with_reference_genome`).
- `--clean-mode`, `--n` — forwarded to the strain runner.
- Slurm `--array=1-N%M` — `N` must match the list's line count; `M` caps
  concurrent tasks.

Splitting large strains into parts is done by the precomputed lists
(replaces the old `panaroo_run_strain_split.sh`).

### 2c. Arbitrary precomputed sample list (single job)

Same script as 2a with `--sample-metadata-file /path/to/samples.tsv`. Useful
for one-off groupings outside the lineage batching scheme.

---

## 3. Analyse Panaroo GPA and distances

Post-Panaroo analysis computes a stratified view of each run's pangenome and
gene-presence distances to reference genomes (MGH78578 plus RefSeq and
complete Norway genomes). This is where bacotype / reference-genome decisions
are driven from.

There are three entrypoints, from narrowest to broadest scope:

### 3a. Single group (one sample set, one row)

Core analysis module for one sample set: pangenome features, KPSC filtering,
Jaccard distances + shared-gene counts vs. reference cohorts, clustering
metrics.

- Python: [`src/bacotype/tl/gpa_distances_single_group.py`](src/bacotype/tl/gpa_distances_single_group.py)

Not usually invoked directly — it is called by the orchestrator (3b) both on
the whole set and on each stratified subset. Call it directly only when you
already have a custom `gpa_df` / `meta_df` in Python.

### 3b. Single Panaroo run (whole set + stratified subsets, one detail TSV)

Orchestrator: runs 3a on the whole set of one Panaroo directory, then on
each major Clonal group (>= `MIN_GROUP_SIZE`) plus a pooled `other`, then on
each major K_locus within each major Clonal group (plus pooled `other`).
Reference genomes are always included in every subset.

- Python: [`src/bacotype/tl/gpa_distances_single_run.py`](src/bacotype/tl/gpa_distances_single_run.py)
- Slurm: [`slurm_scripts/gpa_distances_single_run.sh`](slurm_scripts/gpa_distances_single_run.sh)

Run:
```bash
sbatch slurm_scripts/gpa_distances_single_run.sh
```

Important settings (edit at the top of the `.sh`):
- `PANAROO_RUN_ROOT` + `DIRECTORY_LEAF` (or `PANAROO_DIR` for an explicit
  absolute path) — which Panaroo output to analyse.
- `METADATA_PATH` — curated metadata TSV.
- `MIN_GROUP_SIZE` (default 250) — minimum Clonal group / K_locus size to
  get its own stratified slice; smaller groups are pooled as `other`.
- `REFERENCE_TOP_N` (default 10) — top-N RefSeq / complete Norway genomes by
  lowest mean Jaccard.
- `GPA_FILTER_CUTOFF`, `MERGE_SMALL_CLUSTERS`,
  `SHELL_CLOUD_CUTOFF` / `CORE_SHELL_CUTOFF` — pangenome filtering / core-
  shell-cloud cutoffs.

Output: one detail TSV per run under
`<panaroo_dir>/analysis/GPA_reference_genome/`, with one row per group
(whole set, each Clonal group / `other`, each CG + K_locus / `other`).

### 3c. Batch of runs (every Panaroo run → one summary TSV)

Walks `PANAROO_RUN_ROOT`, picks every immediate subdirectory containing
`gene_presence_absence.Rtab`, and runs 3b on each in parallel. Per-run
detail TSVs are still written by 3b inside each run directory. The batch
itself compiles one summary TSV (one row per run = the whole-set row).

- Python: [`src/bacotype/tl/gpa_distances_batch_runs.py`](src/bacotype/tl/gpa_distances_batch_runs.py)
- Slurm: [`slurm_scripts/gpa_distances_batch_runs.sh`](slurm_scripts/gpa_distances_batch_runs.sh)

Run:
```bash
sbatch slurm_scripts/gpa_distances_batch_runs.sh
```

Important settings (edit at the top of the `.sh`):
- `PANAROO_RUN_ROOT` — directory whose immediate children are per-run
  Panaroo outputs.
- `OUTPUT_DIR` — where the compiled summary TSV is written (default:
  `.../processed/pangenome_analysis`).
- `WORKERS` — number of parallel per-run workers.
- `TEST_N_SUBDIR` — set to an integer to process only the first N leaves
  (smoke test); leave empty for all.
- `MIN_GROUP_SIZE`, `REFERENCE_TOP_N`, `GPA_FILTER_CUTOFF`,
  `MERGE_SMALL_CLUSTERS`, `SHELL_CLOUD_CUTOFF`, `CORE_SHELL_CUTOFF` —
  forwarded to 3b; same meaning as above.

Output: one compiled TSV
`gpa_reference_batch_summary_<timestamp>.tsv` under `OUTPUT_DIR`, plus the
per-run detail TSVs written under each Panaroo run directory.

---

## Choosing which analysis entrypoint to run

- Need bacotype / reference-genome decisions for **one** finished Panaroo
  run? → run 3b (`gpa_distances_single_run.sh`).
- Need the same for **every** finished Panaroo run in the current batching
  directory? → run 3c (`gpa_distances_batch_runs.sh`).
- Calling the analysis from Python code on a pre-built `gpa_df` /
  `meta_df`? → call 3a directly.
