# Isolation Source Prediction Pipeline

This document describes the pair-based pipeline for predicting one isolation source against another (for example, blood vs respiratory) from genome embeddings.

## Overview

The workflow builds a binary task from two isolation source tokens and then prepares split `.pt` files for Bacformer fine-tuning.

## Prerequisites (shared with AMR pipeline)

- Assemblies and protein preprocessing available from the AMR flow.
- ESMc embedding files already generated as `{sample_accession}_esm_embeddings.pt`.

## Pipeline Steps

### 1. Stratified sample selection

Select two isolation source categories by token, then stratify by geography and study-type routing.

| Item | Details |
|------|---------|
| **Script** | [stratified_isolation_source_sampling.py](../src/predict_kleb_by_bacformer/pp/stratified_isolation_source_sampling.py) |
| **Key inputs** | Metadata TSV, `--isolation-sources <token1> <token2>`, optional `--ratio`, optional `--filter-by-study-setting` |
| **Key output** | `train_<token1>_vs_<token2>/stratified_selected_isolation_source_metadata.tsv` |

### 2. Prepare pair-specific labels and splits

Resolve the same token pair, create binary labels, prune missing embeddings, then write train/validate/evaluate `.pt` files.

| Item | Details |
|------|---------|
| **Script** | [prepare_esmc_embeddings_and_labels_to_finetune_isolation_source.py](../src/predict_kleb_by_bacformer/pp/prepare_esmc_embeddings_and_labels_to_finetune_isolation_source.py) |
| **Key inputs** | `--input-csv` stratified metadata, `--isolation-sources <token1> <token2>`, `--embeddings-dir` |
| **Key outputs** | `train_<pair_slug>/binary_<pair_slug>.csv`, `train_<pair_slug>/binary_<pair_slug>_with_split.csv`, `train_<pair_slug>/{train,validate,evaluate}/{sample}_with_<pair_slug>.pt` |

### 3. Fine-tune

Fine-tune Bacformer from the split `.pt` directories.

| Item | Details |
|------|---------|
| **Script** | [train_blood_infx.py](../src/predict_kleb_by_bacformer/tl/train_blood_infx.py) |
| **Key inputs** | Pair-specific split directories and pair-specific `binary_<pair_slug>_with_split.csv` |
| **Output** | Model checkpoints under a pair-specific `models/` directory |

## Example (blood vs respiratory)

```bash
uv run python src/predict_kleb_by_bacformer/pp/stratified_isolation_source_sampling.py \
  --isolation-sources blood respiratory \
  --metadata-file /path/to/metadata.tsv \
  --output-csv /path/to/train_blood_vs_respiratory/stratified_selected_isolation_source_metadata.tsv

uv run python src/predict_kleb_by_bacformer/pp/prepare_esmc_embeddings_and_labels_to_finetune_isolation_source.py \
  --input-csv /path/to/train_blood_vs_respiratory/stratified_selected_isolation_source_metadata.tsv \
  --isolation-sources blood respiratory \
  --embeddings-dir /path/to/klebsiella_esm_embeddings \
  --output-base /path/to/train_blood_vs_respiratory
```
