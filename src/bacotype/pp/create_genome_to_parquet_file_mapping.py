#!/usr/bin/env python3
"""
Create a mapping of genome_id to parquet filename from ESMC embedding files.

This script processes parquet files in parallel to extract genome_id values
and creates a CSV mapping of which genomes are in which files.
"""

import argparse
import glob
import os
from multiprocessing import Pool

import pandas as pd
from tqdm import tqdm


def process_parquet_file(filepath: str) -> pd.DataFrame:
    """
    Process a single parquet file and extract genome_id mappings.

    Args:
        filepath: Path to the parquet file to process

    Returns
    -------
        DataFrame with columns ['genome_id', 'filename'] mapping each
        genome in the file to the file's basename
    """
    df = pd.read_parquet(filepath)
    filename = os.path.basename(filepath)

    # Extract genome_id from each row
    genome_ids = df["genome_id"].tolist()

    # Create mapping DataFrame
    return pd.DataFrame({"genome_id": genome_ids, "filename": [filename] * len(genome_ids)})


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Create genome_id to parquet file mapping from ESMC embeddings")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/projects/public/u5ah/data/genomes/atb/esmc-large",
        help="Directory containing parquet files (default: /projects/public/u5ah/data/genomes/atb/esmc-large)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: ./data/metadata/genome_to_parquet_file_mapping.csv)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=100,
        help="Maximum number of files to process (default: 100, use -1 for all files)",
    )
    parser.add_argument("--processes", type=int, default=10, help="Number of parallel processes (default: 10)")

    args = parser.parse_args()

    # Find all parquet files
    pattern = os.path.join(args.data_dir, "embeddings_*.parquet")
    all_files = sorted(glob.glob(pattern))

    if not all_files:
        raise ValueError(f"No parquet files found matching pattern: {pattern}")

    # Limit files if specified
    if args.max_files > 0:
        file_list = all_files[: args.max_files]
    else:
        file_list = all_files

    print(f"Found {len(all_files)} total parquet files")
    print(f"Processing {len(file_list)} files with {args.processes} processes")

    # Process files in parallel with progress bar
    with Pool(processes=args.processes) as pool:
        results = list(
            tqdm(pool.imap(process_parquet_file, file_list), total=len(file_list), desc="Processing files", unit="file")
        )

    # Accumulate all DataFrames
    print(f"Concatenating {len(results)} DataFrames...")
    final_df = pd.concat(results, ignore_index=True)

    # Determine output path
    if args.output is None:
        output_dir = "./data/metadata"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "genome_to_parquet_file_mapping.csv")
    else:
        output_path = args.output

    # Save to CSV
    print(f"Saving {len(final_df)} genome mappings to {output_path}")
    final_df.to_csv(output_path, index=False)
    print("Done!")


if __name__ == "__main__":
    main()
