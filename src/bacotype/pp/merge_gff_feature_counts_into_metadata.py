"""
Merge GFF feature-count sidecar TSV into the slimmed metadata TSV in place.

Reads ``gff_feature_counts.tsv`` (produced by count_gff_features.py) and left-
merges its ``n_*`` columns onto ``metadata_final_curated_slimmed.tsv`` keyed on
``Sample``. Missing values are filled with 0 and cast to pandas nullable
``Int64``. The metadata file is overwritten atomically.

Run: ``uv run python -m bacotype.pp.merge_gff_feature_counts_into_metadata``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from bacotype.data_paths import data


METADATA_F: Path = data.klebsiella_metadata_file
SIDECAR_F: Path = data.final / "gff_feature_counts.tsv"


def _atomic_write_tsv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, sep="\t", index=False)
    os.replace(tmp, path)


def run(
    metadata_path: Path | None = None,
    sidecar_path: Path | None = None,
) -> None:
    """Left-merge sidecar n_* columns into metadata TSV and overwrite in place."""
    meta_path = Path(metadata_path) if metadata_path is not None else METADATA_F
    side_path = Path(sidecar_path) if sidecar_path is not None else SIDECAR_F

    print(f"Metadata file : {meta_path}")
    print(f"Sidecar file  : {side_path}")

    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file does not exist: {meta_path}")
    if not side_path.exists():
        raise FileNotFoundError(f"Sidecar file does not exist: {side_path}")

    counts = pd.read_csv(side_path, sep="\t", dtype={"Sample": str})
    feature_cols = [c for c in counts.columns if c != "Sample"]
    if not feature_cols:
        raise ValueError(f"Sidecar has no n_* columns: {side_path}")
    print(f"Sidecar rows  : {len(counts)}")
    print(f"Feature cols  : {len(feature_cols)} -> {feature_cols}")

    meta = pd.read_csv(meta_path, sep="\t", low_memory=False)
    n_meta = len(meta)
    print(f"Metadata rows : {n_meta}")

    overlap = [c for c in feature_cols if c in meta.columns]
    if overlap:
        print(f"Dropping {len(overlap)} pre-existing n_* columns before merge: {overlap}")
        meta = meta.drop(columns=overlap)

    merged = meta.merge(counts, on="Sample", how="left")
    if len(merged) != n_meta:
        raise RuntimeError(
            f"Row count changed after merge: before={n_meta} after={len(merged)}"
        )

    for c in feature_cols:
        merged[c] = merged[c].fillna(0).astype("Int64")

    n_matched = int(merged[feature_cols[0]].notna().sum())
    has_gff = (
        merged["gff_file"].fillna("").astype(str).str.strip().str.len() > 0
        if "gff_file" in merged.columns
        else pd.Series([True] * n_meta)
    )
    n_has_gff = int(has_gff.sum())
    n_has_gff_but_unmatched = int((has_gff & (merged[feature_cols[0]] == 0)).sum())
    print(f"Samples matched with counts          : {n_matched}/{n_meta}")
    print(f"Samples with gff_file                : {n_has_gff}")
    print(f"Samples with gff_file but 0 counts   : {n_has_gff_but_unmatched}")

    _atomic_write_tsv(merged, meta_path)
    print(f"Wrote {meta_path} ({len(merged)} rows, {len(merged.columns)} cols)")


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Merge GFF feature-count sidecar TSV into the slimmed metadata TSV in place.",
    )
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--sidecar", type=Path, default=None)
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    run(metadata_path=args.metadata, sidecar_path=args.sidecar)


if __name__ == "__main__":
    main()
