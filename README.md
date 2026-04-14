# predict_kleb_by_bacformer

Use Bacformer to predict Klebsiella behaviour in AST testing and niche (using isolation source as a proxy for infection vs colonisation at a body site).

---

## Project flow (two prediction tasks)

The repo supports two fine-tuning tracks that share per-genome ESM embedding files (`{sample_accession}_esm_embeddings.pt`) but differ in label construction and training setup.

### Task 1 - Antibiotic resistance (AMR, multi-label per genome)

Goal: binary resistance labels per antibiotic (plus optional regression log MIC outputs during preprocessing).


| What runs                                                                                                                                   | Role                                                                                | Input                                                                                             | Output                                                                                                             |
| ------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| **Step 1:** `src/predict_kleb_by_bacformer/pp/preprocess_ebi_amr_records.py` (calls `src/predict_kleb_by_bacformer/pp/convert_ast_data.py`) | Preprocess EBI AST records and pivot to sample-level tables                         | Raw EBI AMR CSV; min-antibiotic threshold                                                         | `binary_ast.csv`, metadata CSV, `regression_log_mic.csv`, antibiogram                                              |
| **Step 2:** `src/predict_kleb_by_bacformer/pp/prepare_esmc_embeddings_and_labels_to_finetune_amr.py`                                        | Add train/validate/evaluate splits and merge AST labels into per-sample `.pt` files | `binary_ast.csv`, embedding files `{Sample}_esm_embeddings.pt`                                    | `binary_ast_with_split.csv`, `ast_training/{train,validate,evaluate}/{Sample}_with_ast.pt`, missing-samples report |
| **Step 3:** `src/predict_kleb_by_bacformer/tl/train_amr.py` (typically via `slurm_scripts/train_on_slurm_amr.sh`)                           | Fine-tune Bacformer for one selected antibiotic label                               | `ast_training/train`, `ast_training/validate`, `binary_ast_with_split.csv`, `--drug <antibiotic>` | Fine-tuned checkpoints and training logs                                                                           |


---

### Task 2 - Isolation source (binary pair task, e.g. blood vs respiratory)

Goal: create a pair-specific binary isolation-source task and train on pair-specific split outputs.


| What runs                                                                                                         | Role                                                                                                                                    | Input                                                                                                                                                  | Output                                                                                                                                                                        |
| ----------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Step 1:** `src/predict_kleb_by_bacformer/pp/stratified_isolation_source_sampling.py`                            | Select and balance two isolation-source categories with country + study-thread stratification                                           | Metadata TSV, `--isolation-sources <token1> <token2>`, optional `--ratio`, optional `--filter-by-study-setting`                                        | `/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/training_<token1>_<token2>/stratified_selected_isolation_source_metadata.tsv`                               |
| **Step 2:** `src/predict_kleb_by_bacformer/pp/prepare_esmc_embeddings_and_labels_to_finetune_isolation_source.py` | Resolve the same token pair, create pair-specific binary labels, split 70/10/20, prune missing embeddings, and write merged `.pt` files | `--isolation-sources <token1> <token2>`, optional `--input-metadata-file`, optional `--embeddings-dir`                                                   | `training_<token1>_<token2>/binary_<pair_slug>.csv`, `training_<token1>_<token2>/binary_<pair_slug>_with_split.csv`, `training_<token1>_<token2>/{train,validate,evaluate}/...` |
| **Step 3:** `src/predict_kleb_by_bacformer/tl/train_blood_infx.py`                                                | Train Bacformer from pair-specific split folders                                                                                        | Pair-specific `train` + `validate` folders and `binary_<pair_slug>_with_split.csv`                                                                     | Model checkpoints under a pair-specific `models/` folder                                                                                                                      |


---

## Placeholder - genomes to protein files to ESM embeddings

To be filled in next.

- Download or collect GFF files.
- Derive `.faa` protein FASTA files.
- Run ESM embedding generation so each retained sample has `{sample_accession}_esm_embeddings.pt`.

Until this section is completed, treat the embeddings directory as a prerequisite input to both prepare scripts.

---

## End-to-end example: blood vs respiratory

### Default workflow (auto-resolved directories)

```bash
uv run python src/predict_kleb_by_bacformer/pp/stratified_isolation_source_sampling.py \
  --isolation-sources blood respiratory
```

```bash
uv run python src/predict_kleb_by_bacformer/pp/prepare_esmc_embeddings_and_labels_to_finetune_isolation_source.py \
  --isolation-sources blood respiratory
```

The scripts infer and share:

- `/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/training_blood_respiratory/`
- `stratified_selected_isolation_source_metadata.tsv`
- `stratify_isolation_source_sampling.log`
- default embeddings dir: `/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/klebsiella_esm_embeddings`

### Optional overrides

```bash
uv run python src/predict_kleb_by_bacformer/pp/stratified_isolation_source_sampling.py \
  --isolation-sources blood respiratory \
  --metadata-file /path/to/metadata.tsv \
  --output-file /path/to/training_blood_respiratory/stratified_selected_isolation_source_metadata.tsv
```

```bash
uv run python src/predict_kleb_by_bacformer/pp/prepare_esmc_embeddings_and_labels_to_finetune_isolation_source.py \
  --isolation-sources blood respiratory \
  --input-metadata-file /path/to/training_blood_respiratory/stratified_selected_isolation_source_metadata.tsv \
  --embeddings-dir /path/to/klebsiella_esm_embeddings
```

---

## Related docs

More detail lives under `docs/`:

- `docs/amr_prediction.md`
- `docs/isolation_source_prediction.md`

