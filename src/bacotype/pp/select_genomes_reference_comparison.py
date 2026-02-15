#!/usr/bin/env python3
"""Genome selection utility: select reference and comparison genomes for analysis."""

import argparse
import pandas as pd
from pathlib import Path

# Hardcoded metadata path
METADATA_PATH = "/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/final/metadata_final_curated_slimmed.tsv"
CLONAL_GROUP = "CG258"
KL_SUBSETS = ["KL106", "KL107"]  # comparison KL types

def select_genomes(n):
    """
    Select reference (is_refseq) and first n comparisons from each KL subset.
    
    Args:
        n: Number of comparison genomes to select per KL subset
    
    Returns:
        reference_accession: str
        comparison_accessions: list of str (grouped by KL)
    """
    df = pd.read_csv(METADATA_PATH, sep="\t", low_memory=False)
    
    # Filter to CG258
    cg_df = df[df["Clonal group"] == CLONAL_GROUP].copy()
    print(f"Total genomes in {CLONAL_GROUP}: {len(cg_df)}")
    
    # Reference: is_refseq == True
    ref_df = cg_df[cg_df["is_refseq"].astype(str).str.lower().isin(["true", "1", "yes"])]
    if len(ref_df) == 0:
        raise ValueError(f"No is_refseq=True genome found in {CLONAL_GROUP}")
    
    reference = ref_df.iloc[0]["sample_accession"]  # or "Sample" if needed
    reference_kl = ref_df.iloc[0]["K_locus"]
    print(f"\nReference: {reference} (K_locus: {reference_kl})")
    
    # Comparison: first n from each KL subset
    comparisons = []
    for kl in KL_SUBSETS:
        kl_df = cg_df[cg_df["K_locus"] == kl]
        subset = kl_df.head(n)["sample_accession"].tolist()
        print(f"\n{kl}: {len(subset)} genomes selected")
        for acc in subset:
            print(f"  - {acc}")
        comparisons.extend(subset)
    
    return reference, comparisons

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select reference and comparison genomes")
    parser.add_argument("--n", type=int, default=10, help="Number of comparison genomes per KL subset")
    args = parser.parse_args()
    
    print(f"===== Genome Selection (n={args.n}) =====")
    reference, comparisons = select_genomes(args.n)
    
    print(f"\n===== Summary =====")
    print(f"Reference: {reference}")
    print(f"Comparisons: {len(comparisons)} genomes")
