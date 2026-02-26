import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


AST_CSV_DEFAULT = Path(
    "/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/binary_ast.csv"
)
EMBEDDINGS_DIR_DEFAULT = Path(
    "/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/klebsiella_esm_embeddings"
)
OUTPUT_BASE_DEFAULT = Path(
    "/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/ast_training"
)


def load_ast_sheet(ast_csv: Path) -> pd.DataFrame:
    """Load binary AST sheet and add a convenience Sample column."""
    df = pd.read_csv(ast_csv)
    # First column is phenotype-BioSample_ID
    if df.columns[0] != "phenotype-BioSample_ID":
        raise ValueError(
            f"Expected first column to be 'phenotype-BioSample_ID', got {df.columns[0]!r}"
        )
    # Add a convenience Sample column matching the IDs used in embedding filenames
    df["Sample"] = df["phenotype-BioSample_ID"].astype(str)
    return df


def add_splits(df: pd.DataFrame, seed: int = 1) -> pd.DataFrame:
    """Add train_val_eval column with 70/10/20 split over unique Sample IDs."""
    rng = np.random.default_rng(seed)
    sample_ids = df["Sample"].unique()
    rng.shuffle(sample_ids)

    n_total = len(sample_ids)
    n_train = int(0.7 * n_total)
    n_val = int(0.1 * n_total)
    # Remaining go to eval
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


def get_antibiotic_columns(df: pd.DataFrame) -> list[str]:
    """Return antibiotic columns (all except ID/split helper columns)."""
    exclude = {"phenotype-BioSample_ID", "Sample", "train_val_eval"}
    return [c for c in df.columns if c not in exclude]


def tensor_to_nested_list(t: torch.Tensor) -> list:
    """
    Convert a protein_embeddings tensor of shape [1, N, D] or [N, D]
    into a Python list-of-arrays suitable for parquet storage and
    downstream use with protein_embeddings_to_inputs.
    """
    if t.ndim == 3 and t.shape[0] == 1:
        t = t.squeeze(0)
    if t.ndim != 2:
        raise ValueError(f"Expected protein_embeddings with shape [1, N, D] or [N, D], got {tuple(t.shape)}")
    # Represent as a flat list of vectors (single contig)
    arr = t.detach().cpu().numpy()
    return [row for row in arr]


def write_split_parquet_files(
    df_with_splits: pd.DataFrame,
    embeddings_dir: Path,
    output_base: Path,
) -> None:
    """For each Sample, load .pt embeddings and write a per-sample parquet with AST labels."""
    antibiotic_cols = get_antibiotic_columns(df_with_splits)

    # Prepare output directories
    for split_name in ("train", "validate", "evaluate"):
        (output_base / split_name).mkdir(parents=True, exist_ok=True)

    # Work over unique samples, then attach AST row values
    grouped = df_with_splits.groupby("Sample", as_index=False).first()

    print(f"Total unique samples in AST sheet: {len(grouped)}")
    print(f"Antibiotic columns: {len(antibiotic_cols)}")

    for _, row in tqdm(grouped.iterrows(), total=len(grouped), desc="Writing split parquet files"):
        sample_id = row["Sample"]
        split = row["train_val_eval"]
        if split not in ("train", "validate", "evaluate"):
            raise ValueError(f"Unexpected split value {split!r} for Sample {sample_id}")

        pt_path = embeddings_dir / f"{sample_id}_esm_embeddings.pt"
        if not pt_path.exists():
            # Skip samples without embeddings, but warn
            print(f"WARNING: Embedding file not found for Sample {sample_id}: {pt_path}")
            continue

        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        emb = data.get("prot_embeddings", data.get("protein_embeddings"))
        if emb is None:
            raise KeyError(f"'prot_embeddings' or 'protein_embeddings' key missing in {pt_path}")
        prot_emb_list = tensor_to_nested_list(emb) if isinstance(emb, torch.Tensor) else emb

        out = {"Sample": sample_id, "prot_embeddings": prot_emb_list}
        contig_raw = data.get("contig_idx", data.get("contig_ids"))
        if contig_raw is not None:
            out["contig_idx"] = (
                contig_raw.squeeze().cpu().numpy() if isinstance(contig_raw, torch.Tensor) else contig_raw
            )
        for key in ("attention_mask", "special_tokens_mask", "token_type_ids"):
            if key in data:
                v = data[key]
                out[key] = v.squeeze().cpu().numpy() if isinstance(v, torch.Tensor) else v
        for col in antibiotic_cols:
            out[col] = row[col]

        out_df = pd.DataFrame([out])

        dest_dir = output_base / split
        dest_path = dest_dir / f"{sample_id}.parquet"
        out_df.to_parquet(dest_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare Klebsiella AMR train/val/eval parquet splits from .pt embeddings and binary_ast.csv."
    )
    parser.add_argument(
        "--ast-csv",
        type=Path,
        default=AST_CSV_DEFAULT,
        help="Path to binary_ast.csv",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=EMBEDDINGS_DIR_DEFAULT,
        help="Directory containing {Sample}_esm_embeddings.pt files.",
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=OUTPUT_BASE_DEFAULT,
        help="Base directory for ast_training/{train,validate,evaluate}/ outputs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for train/val/eval split.",
    )
    args = parser.parse_args()

    print(f"Loading AST sheet from: {args.ast_csv}")
    ast_df = load_ast_sheet(args.ast_csv)
    print(f"Total AST rows: {len(ast_df)}")

    print("Adding train/validate/evaluate splits (70/10/20)...")
    ast_df_splits = add_splits(ast_df, seed=args.seed)

    # Save annotated AST sheet next to original as binary_ast_with_split.csv
    split_csv_path = args.ast_csv.with_name("binary_ast_with_split.csv")
    print(f"Writing annotated AST sheet with splits to: {split_csv_path}")
    ast_df_splits.to_csv(split_csv_path, index=False)

    print("Writing per-sample parquet files with embeddings and AST labels...")
    write_split_parquet_files(ast_df_splits, args.embeddings_dir, args.output_base)

    print("Done.")


if __name__ == "__main__":
    main()

