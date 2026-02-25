import argparse
from pathlib import Path

import pandas as pd


def parse_missing_samples(log_path: Path) -> list[str]:
    """Parse the .out log and return a list of sample IDs with missing embeddings."""
    missing = []
    prefix = "WARNING: Embedding file not found for Sample "
    with log_path.open("r") as f:
        for line in f:
            if prefix in line:
                # Line format:
                # WARNING: Embedding file not found for Sample SAMEA104289870: /path/.../SAMEA104289870_esm_embeddings.pt
                try:
                    after = line.split(prefix, 1)[1]
                    sample_id = after.split(":", 1)[0].strip()
                    if sample_id.startswith("SAM"):
                        missing.append(sample_id)
                except (IndexError, ValueError):
                    continue
    # Deduplicate while preserving order
    seen = set()
    unique_missing = []
    for sid in missing:
        if sid not in seen:
            seen.add(sid)
            unique_missing.append(sid)
    return unique_missing


def prune_ast_csv(
    ast_csv: Path,
    missing_ids: list[str],
    output_missing_csv: Path,
) -> None:
    """Remove rows for missing_ids from AST CSV and write them to a separate file."""
    if not missing_ids:
        print("No missing sample IDs found in log; nothing to prune.")
        return

    print(f"Loading AST CSV from: {ast_csv}")
    df = pd.read_csv(ast_csv)
    print(f"Original rows in AST CSV: {len(df)}")

    # Determine which column holds the sample ID
    if "Sample" in df.columns:
        id_col = "Sample"
    elif "phenotype-BioSample_ID" in df.columns:
        id_col = "phenotype-BioSample_ID"
    else:
        # Fallback: use first column
        id_col = df.columns[0]
        print(f"WARNING: Using first column {id_col!r} as ID column.")

    mask_missing = df[id_col].astype(str).isin(missing_ids)
    removed_df = df[mask_missing].copy()
    kept_df = df[~mask_missing].copy()

    print(f"Rows to remove (missing embeddings): {len(removed_df)}")
    print(f"Rows kept: {len(kept_df)}")

    # Overwrite the AST CSV with only kept rows
    kept_df.to_csv(ast_csv, index=False)
    print(f"Updated AST CSV written (pruned) to: {ast_csv}")

    # Write removed rows (full rows) to a separate CSV for inspection
    output_missing_csv.parent.mkdir(parents=True, exist_ok=True)
    removed_df.to_csv(output_missing_csv, index=False)
    print(f"List of samples not in dataset written to: {output_missing_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Parse a preprocess_amr_data .out log, remove samples with missing embeddings "
            "from binary_ast_with_split.csv, and write them to ast_samples_not_in_dataset.csv."
        )
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        required=True,
        help="Path to preprocess_amr_data_*.out log file.",
    )
    parser.add_argument(
        "--ast-csv",
        type=Path,
        default=Path(
            "/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/binary_ast_with_split.csv"
        ),
        help="Path to binary_ast_with_split.csv to prune.",
    )
    parser.add_argument(
        "--missing-out",
        type=Path,
        default=Path(
            "/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/ast_samples_not_in_dataset.csv"
        ),
        help="Path to write CSV listing samples removed from AST (full rows).",
    )
    args = parser.parse_args()

    missing_ids = parse_missing_samples(args.log_path)
    print(f"Found {len(missing_ids)} unique sample IDs with missing embeddings in log.")
    prune_ast_csv(args.ast_csv, missing_ids, args.missing_out)


if __name__ == "__main__":
    main()

