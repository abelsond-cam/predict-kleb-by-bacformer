#!/usr/bin/env python3
"""Genome selection utility: select reference and comparison genomes for analysis."""

import argparse
import pandas as pd
from pathlib import Path

# Hardcoded metadata path
METADATA_PATH = "/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/final/metadata_final_curated_slimmed.tsv"
OUTPUT_PATH = "/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/mgefinder/reference_comparison_sets.tsv"
CLONAL_GROUP = "CG258"
REFERENCE_KL_TYPES = ["KL63", "KL106", "KL107"]  # reference KL types
COMPARISON_KL_TYPES = ["KL106", "KL107"]  # comparison KL types

def select_genomes(n):
    """
    Select 3 reference genomes (one from each KL type) and n comparisons from each comparison KL subset.
    
    Args:
        n: Number of comparison genomes to select per KL subset
    
    Returns:
        references: list of dict with keys: sample_accession, Sample, K_locus
        comparisons: list of dict with keys: sample_accession, K_locus, Clonal group
    """
    df = pd.read_csv(METADATA_PATH, sep="\t", low_memory=False)
    
    # Filter to CG258
    cg_df = df[df["Clonal group"] == CLONAL_GROUP].copy()
    print(f"Total genomes in {CLONAL_GROUP}: {len(cg_df)}")
    
    # Get all is_refseq genomes
    ref_df = cg_df[cg_df["is_refseq"].astype(str).str.lower().isin(["true", "1", "yes"])]
    
    # Select 3 references: one from each KL type
    references = []
    for kl in REFERENCE_KL_TYPES:
        kl_ref_df = ref_df[ref_df["K_locus"] == kl]
        if len(kl_ref_df) == 0:
            print(f"ERROR: No is_refseq genome found for {CLONAL_GROUP} + {kl} combination")
            continue
        
        ref_genome = kl_ref_df.iloc[0]
        references.append({
            "sample_accession": ref_genome["sample_accession"],
            "Sample": ref_genome["Sample"],
            "K_locus": ref_genome["K_locus"]
        })
        print(f"\nReference ({kl}): {ref_genome['sample_accession']}")
    
    # Comparison: first n from each comparison KL subset
    comparisons = []
    for kl in COMPARISON_KL_TYPES:
        kl_df = cg_df[cg_df["K_locus"] == kl]
        subset = kl_df.head(n)
        print(f"\n{kl}: {len(subset)} genomes selected")
        for _, row in subset.iterrows():
            comparisons.append({
                "sample_accession": row["sample_accession"],
                "K_locus": row["K_locus"],
                "Clonal group": row["Clonal group"]
            })
            print(f"  - {row['sample_accession']}")
    
    return references, comparisons

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select reference and comparison genomes")
    parser.add_argument("--n", type=int, default=10, help="Number of comparison genomes per KL subset")
    args = parser.parse_args()
    
    print(f"===== Genome Selection (n={args.n}) =====")
    references, comparisons = select_genomes(args.n)
    
    # Build comparison set strings (same for all references)
    comparison_accessions = [c["sample_accession"] for c in comparisons]
    comparison_kls = [c["K_locus"] for c in comparisons]
    comparison_cgs = [c["Clonal group"] for c in comparisons]
    
    mge_comparison_set = ",".join(comparison_accessions)
    comparison_KL = ",".join(comparison_kls)
    comparison_CG = ",".join(comparison_cgs)
    
    # Build dataframe with one row per reference
    rows = []
    for ref in references:
        rows.append({
            "reference_sample_name": ref["Sample"],
            "reference_sample_accession": ref["sample_accession"],
            "mge_comparison_set": mge_comparison_set,
            "reference_CG": CLONAL_GROUP,
            "reference_KL": ref["K_locus"],
            "comparison_CG": comparison_CG,
            "comparison_KL": comparison_KL
        })
    
    df = pd.DataFrame(rows)
    
    # Ensure output directory exists
    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to TSV
    df.to_csv(output_path, sep="\t", index=False)
    print(f"\n===== Output saved to: {output_path} =====")
    
    # Print summary
    print(f"\n===== Summary =====")
    print(f"References: {len(references)}")
    for ref in references:
        print(f"  - {ref['sample_accession']} ({ref['K_locus']})")
    print(f"\nComparisons: {len(comparisons)} genomes")
    print(f"  - {len([c for c in comparisons if c['K_locus'] == 'KL106'])} from KL106")
    print(f"  - {len([c for c in comparisons if c['K_locus'] == 'KL107'])} from KL107")
