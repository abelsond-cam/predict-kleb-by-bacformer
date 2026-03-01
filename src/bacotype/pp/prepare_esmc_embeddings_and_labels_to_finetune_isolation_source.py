"""
Prepare ESMc embeddings and isolation-source labels as pytorch (.pt) files for finetuning.

Creates train/val/eval splits (70/10/20) from ESM embeddings and stratified
blood/stool metadata. Predicts blood (1) vs stool (0) from isolation_source_category.
Samples without embedding files are pruned automatically.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


INPUT_CSV_DEFAULT = Path(
    "/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/stratified_sample_blood_stool_by_country.csv"
)
EMBEDDINGS_DIR_DEFAULT = Path(
    "/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/klebsiella_esm_embeddings"
)
OUTPUT_BASE_DEFAULT = Path(
    "/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/blood_infx_training"
)

BLOOD_VAL = "blood"
STOOL_VAL = "faeces & rectal swabs"
LABEL_COLUMN = "blood_infxn"
PT_SUFFIX = "_with_blood_infx.pt"


def load_metadata_sheet(input_csv: Path) -> pd.DataFrame:
    """Load stratified blood/stool metadata and add Sample column."""
    df = pd.read_csv(input_csv)
    # Support sample_accession or phenotype-BioSample_ID
    if "sample_accession" in df.columns:
        df["Sample"] = df["sample_accession"].astype(str)
    elif "phenotype-BioSample_ID" in df.columns:
        df["Sample"] = df["phenotype-BioSample_ID"].astype(str)
    else:
        raise ValueError(
            f"Expected 'sample_accession' or 'phenotype-BioSample_ID', got {list(df.columns)}"
        )
    return df


def filter_and_create_blood_infxn(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to blood and stool only; create blood_infxn column.
    blood -> 1, faeces & rectal swabs -> 0.
    """
    iso_col = "isolation_source_category"
    if iso_col not in df.columns and "phenotype-isolation_source_category" in df.columns:
        iso_col = "phenotype-isolation_source_category"

    if iso_col not in df.columns:
        raise ValueError(
            f"Expected 'isolation_source_category' or 'phenotype-isolation_source_category', "
            f"got {list(df.columns)}"
        )

    filtered = df[df[iso_col].isin([BLOOD_VAL, STOOL_VAL])].copy()
    filtered[LABEL_COLUMN] = (filtered[iso_col] == BLOOD_VAL).astype(int)
    return filtered


def add_splits(df: pd.DataFrame, seed: int = 1) -> pd.DataFrame:
    """Add train_val_eval column with 70/10/20 split over unique Sample IDs."""
    rng = np.random.default_rng(seed)
    sample_ids = df["Sample"].unique()
    rng.shuffle(sample_ids)

    n_total = len(sample_ids)
    n_train = int(0.7 * n_total)
    n_val = int(0.1 * n_total)
    train_ids = set(sample_ids[:n_train])
    val_ids = set(sample_ids[n_train : n_train + n_val])
    eval_ids = set(sample_ids[n_train + n_val :])

    def _assign_split(sample_id: str) -> str:
        if sample_id in train_ids:
            return "train"
        if sample_id in val_ids:
            return "validate"
        return "evaluate"

    df = df.copy()
    df["train_val_eval"] = df["Sample"].map(_assign_split)
    return df


def validate_embeddings_and_prune(
    df_with_splits: pd.DataFrame,
    embeddings_dir: Path,
) -> tuple[pd.DataFrame, list[str]]:
    """Return pruned dataframe and list of missing sample IDs."""
    grouped = df_with_splits.groupby("Sample", as_index=False).first()
    missing_ids = []
    kept_ids = []

    for sample_id in grouped["Sample"]:
        embed_path = embeddings_dir / f"{sample_id}_esm_embeddings.pt"
        if embed_path.exists():
            kept_ids.append(sample_id)
        else:
            missing_ids.append(sample_id)
            print(f"WARNING: Embedding file not found for Sample {sample_id}: {embed_path}")

    kept_set = set(kept_ids)
    pruned_df = df_with_splits[df_with_splits["Sample"].isin(kept_set)].copy()
    return pruned_df, missing_ids


def write_split_files(
    df_with_splits: pd.DataFrame,
    embeddings_dir: Path,
    output_base: Path,
) -> None:
    """Write per-sample pytorch (.pt) files with embeddings and blood_infxn label."""
    for split_name in ("train", "validate", "evaluate"):
        (output_base / split_name).mkdir(parents=True, exist_ok=True)

    grouped = df_with_splits.groupby("Sample", as_index=False).first()
    print(f"Total unique samples (after pruning): {len(grouped)}")
    print(f"Label column: {LABEL_COLUMN}")

    for _, row in tqdm(
        grouped.iterrows(), total=len(grouped), desc="Writing split pytorch (.pt) files"
    ):
        sample_id = row["Sample"]
        split = row["train_val_eval"]
        if split not in ("train", "validate", "evaluate"):
            raise ValueError(f"Unexpected split value {split!r} for Sample {sample_id}")

        embed_path = embeddings_dir / f"{sample_id}_esm_embeddings.pt"
        if not embed_path.exists():
            print(f"WARNING: Embedding file not found for Sample {sample_id}: {embed_path}")
            continue

        data = torch.load(embed_path, map_location="cpu", weights_only=False)
        emb = data.get("prot_embeddings", data.get("protein_embeddings"))
        if emb is None:
            raise KeyError(f"'prot_embeddings' or 'protein_embeddings' key missing in {embed_path}")

        out: dict = {"Sample": sample_id, "prot_embeddings": emb}
        contig = data.get("contig_idx", data.get("contig_ids"))
        if contig is not None:
            out["contig_idx"] = contig
        if "attention_mask" in data:
            out["attention_mask"] = data["attention_mask"]
        if "special_tokens_mask" in data:
            out["special_tokens_mask"] = data["special_tokens_mask"]
        if "token_type_ids" in data:
            out["token_type_ids"] = data["token_type_ids"]

        out[LABEL_COLUMN] = int(row[LABEL_COLUMN])

        dest_dir = output_base / split
        dest_path = dest_dir / f"{sample_id}{PT_SUFFIX}"
        torch.save(out, dest_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare ESMc embeddings and blood_infxn labels as pytorch (.pt) splits. "
        "blood=1, faeces & rectal swabs=0. Prunes samples without embeddings."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=INPUT_CSV_DEFAULT,
        help="Path to stratified_sample_blood_stool_by_country.csv",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=EMBEDDINGS_DIR_DEFAULT,
        help="Directory containing pytorch (.pt) files named {Sample}_esm_embeddings.pt.",
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=OUTPUT_BASE_DEFAULT,
        help="Base directory for blood_infx_training/{train,validate,evaluate}/ outputs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for train/val/eval split.",
    )
    parser.add_argument(
        "--missing-out",
        type=Path,
        default=None,
        help="Optional path to write CSV of samples removed (missing embeddings).",
    )
    args = parser.parse_args()

    print(f"Loading metadata from: {args.input_csv}")
    df = load_metadata_sheet(args.input_csv)
    print(f"Total rows: {len(df)}")

    print(f"Filtering to {BLOOD_VAL} and {STOOL_VAL}, creating {LABEL_COLUMN}...")
    df = filter_and_create_blood_infxn(df)
    print(f"Rows after filter: {len(df)}")

    # Write binary_blood_infxn.csv (before splits, for reference)
    binary_path = args.input_csv.parent / "binary_blood_infxn.csv"
    binary_df = df.groupby("Sample", as_index=False).first()[["Sample", LABEL_COLUMN]]
    binary_df.to_csv(binary_path, index=False)
    print(f"Wrote {binary_path}")

    print("Adding train/validate/evaluate splits (70/10/20)...")
    df_splits = add_splits(df, seed=args.seed)

    print("Validating embedding files and pruning missing samples...")
    pruned_df, missing_ids = validate_embeddings_and_prune(df_splits, args.embeddings_dir)
    print(f"Samples with missing embeddings (removed): {len(missing_ids)}")
    print(f"Samples kept: {len(pruned_df['Sample'].unique())}")

    if missing_ids:
        missing_out = args.missing_out or (args.output_base / "blood_infxn_samples_not_in_dataset.csv")
        missing_out.parent.mkdir(parents=True, exist_ok=True)
        missing_mask = df_splits["Sample"].isin(missing_ids)
        removed_df = df_splits[missing_mask].drop_duplicates(subset=["Sample"])
        removed_df.to_csv(missing_out, index=False)
        print(f"Removed samples written to: {missing_out}")

    split_csv_path = args.input_csv.parent / "binary_blood_infxn_with_split.csv"
    print(f"Writing pruned sheet with splits to: {split_csv_path}")
    pruned_df.to_csv(split_csv_path, index=False)

    print("Writing per-sample pytorch (.pt) files with embeddings and blood_infxn labels...")
    write_split_files(pruned_df, args.embeddings_dir, args.output_base)

    print("Done.")


if __name__ == "__main__":
    main()
