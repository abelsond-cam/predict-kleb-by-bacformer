#!/usr/bin/env python3
"""Update sample_accession values for RefSeq genomes by mapping assembly to BioSample accessions.

This script:
1. Loads metadata TSV
2. Extracts assembly accessions (GCF_*) from is_refseq=True rows
3. Queries NCBI datasets to map assembly -> BioSample accessions
4. Updates sample_accession column with BioSample accessions
5. Saves updated metadata back to file

For accessions that cannot be mapped, they remain unchanged as GCF_ format.
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

# Hardcoded paths
METADATA_PATH = Path("/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/final/metadata_final_curated_slimmed.tsv")
DATASETS_BIN = Path.home() / ".binaries" / "datasets"
DATAFORMAT_BIN = Path.home() / ".binaries" / "dataformat"


def extract_gcf_accession(full_accession: str) -> str:
    """
    Extract the GCF accession from a full assembly name.
    
    Args:
        full_accession: Full assembly string (e.g., 'GCF_020526085.1_ASM2052608v1_genomic')
    
    Returns:
        Just the accession part (e.g., 'GCF_020526085.1')
    """
    # Split by underscore and take first two parts (GCF_XXXXXXXX.X)
    parts = full_accession.split('_')
    if len(parts) >= 2 and parts[0] == 'GCF':
        return f"{parts[0]}_{parts[1]}"
    # If it doesn't match expected format, return as-is
    return full_accession


def query_ncbi_datasets(accessions: list[str], temp_dir: Path) -> dict[str, str]:
    """
    Query NCBI datasets to map assembly accessions to BioSample accessions.
    
    Args:
        accessions: List of assembly accessions (e.g., ['GCF_020526085.1'])
        temp_dir: Temporary directory for intermediate files
    
    Returns:
        Dictionary mapping assembly accession -> BioSample accession
        (Accessions that cannot be mapped will not be in the dictionary)
    """
    # Write accessions to temporary input file
    input_file = temp_dir / "gcf_list.txt"
    with open(input_file, 'w') as f:
        for acc in accessions:
            f.write(f"{acc}\n")
    
    print(f"Querying NCBI datasets for {len(accessions)} assembly accessions...")
    
    # Run datasets command piped to dataformat
    output_file = temp_dir / "assembly_to_biosample.tsv"
    
    try:
        # First command: datasets summary genome accession
        datasets_cmd = [
            str(DATASETS_BIN),
            "summary",
            "genome",
            "accession",
            "--inputfile",
            str(input_file),
            "--as-json-lines"
        ]
        
        # Second command: dataformat tsv genome
        dataformat_cmd = [
            str(DATAFORMAT_BIN),
            "tsv",
            "genome",
            "--fields",
            "accession,assminfo-biosample-accession"
        ]
        
        # Run datasets and pipe to dataformat
        with open(output_file, 'w') as outf:
            datasets_proc = subprocess.Popen(
                datasets_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            dataformat_proc = subprocess.Popen(
                dataformat_cmd,
                stdin=datasets_proc.stdout,
                stdout=outf,
                stderr=subprocess.PIPE
            )
            
            # Allow datasets_proc to receive a SIGPIPE if dataformat_proc exits
            datasets_proc.stdout.close()
            
            # Wait for both processes to complete
            dataformat_stderr = dataformat_proc.communicate()[1]
            datasets_return = datasets_proc.wait()
            
            if datasets_return != 0:
                datasets_stderr = datasets_proc.stderr.read()
                print(f"\nWARNING: datasets command reported issues:", file=sys.stderr)
                print(datasets_stderr.decode().strip(), file=sys.stderr)
                print("Continuing with partial results (unmapped accessions will remain as GCF_ format)...\n", file=sys.stderr)
            
            if dataformat_proc.returncode != 0:
                print(f"\nWARNING: dataformat command reported issues:", file=sys.stderr)
                print(dataformat_stderr.decode().strip(), file=sys.stderr)
                print("Continuing with partial results (unmapped accessions will remain as GCF_ format)...\n", file=sys.stderr)
    
    except FileNotFoundError as e:
        print(f"ERROR: Required binary not found: {e}", file=sys.stderr)
        print(f"Please ensure datasets and dataformat are installed at:", file=sys.stderr)
        print(f"  {DATASETS_BIN}", file=sys.stderr)
        print(f"  {DATAFORMAT_BIN}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to query NCBI datasets: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Parse the output TSV
    mapping = {}
    try:
        # Check if file has content
        if not output_file.exists() or output_file.stat().st_size == 0:
            print(f"WARNING: No output from NCBI datasets command", file=sys.stderr)
            print("All accessions will remain unchanged as GCF_ format", file=sys.stderr)
            return mapping
        
        df = pd.read_csv(output_file, sep='\t')
        
        # Check if dataframe is empty
        if len(df) == 0:
            print(f"WARNING: Empty results from NCBI", file=sys.stderr)
            print("All accessions will remain unchanged as GCF_ format", file=sys.stderr)
            return mapping
        
        # Check if we have the expected columns
        if 'Assembly Accession' not in df.columns or 'Assembly BioSample Accession' not in df.columns:
            print(f"ERROR: Unexpected column names in output: {df.columns.tolist()}", file=sys.stderr)
            print("All accessions will remain unchanged as GCF_ format", file=sys.stderr)
            return mapping
        
        for _, row in df.iterrows():
            assembly = row['Assembly Accession']
            biosample = row['Assembly BioSample Accession']
            if pd.notna(assembly) and pd.notna(biosample):
                mapping[assembly] = biosample
        
        print(f"Successfully mapped {len(mapping)} assemblies to biosamples")
        
    except Exception as e:
        print(f"WARNING: Failed to parse NCBI output: {e}", file=sys.stderr)
        print("Returning partial results (unmapped accessions will remain as GCF_ format)...", file=sys.stderr)
        return mapping
    
    return mapping


def main():
    parser = argparse.ArgumentParser(
        description="Update sample_accession for RefSeq genomes with BioSample accessions"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test with first 100 RefSeq rows without saving changes"
    )
    args = parser.parse_args()
    
    # Validate binaries exist
    if not DATASETS_BIN.exists():
        print(f"ERROR: datasets binary not found at {DATASETS_BIN}", file=sys.stderr)
        sys.exit(1)
    if not DATAFORMAT_BIN.exists():
        print(f"ERROR: dataformat binary not found at {DATAFORMAT_BIN}", file=sys.stderr)
        sys.exit(1)
    
    # Validate metadata exists
    if not METADATA_PATH.exists():
        print(f"ERROR: Metadata file not found at {METADATA_PATH}", file=sys.stderr)
        sys.exit(1)
    
    # Load metadata
    print(f"Loading metadata from {METADATA_PATH}")
    df = pd.read_csv(METADATA_PATH, sep='\t', low_memory=False)
    print(f"Loaded {len(df)} total rows")
    
    # Filter to is_refseq == True
    refseq_mask = df['is_refseq'].astype(str).str.lower().isin(['true', '1', 'yes'])
    refseq_df = df[refseq_mask].copy()
    total_refseq = len(refseq_df)
    print(f"Found {total_refseq} rows with is_refseq=True")
    
    if total_refseq == 0:
        print("No RefSeq rows to update. Exiting.")
        return
    
    # Apply dry-run limit
    if args.dry_run:
        print("Dry-run mode: limiting to first 100 rows")
        refseq_df = refseq_df.head(100)
    
    # Extract assembly accessions
    assembly_accessions = []
    accession_to_index = {}  # Map accession -> list of row indices in original df
    
    for idx, row in refseq_df.iterrows():
        full_acc = row['sample_accession']
        gcf_acc = extract_gcf_accession(str(full_acc))
        assembly_accessions.append(gcf_acc)
        
        if gcf_acc not in accession_to_index:
            accession_to_index[gcf_acc] = []
        accession_to_index[gcf_acc].append(idx)
    
    # Get unique accessions for querying
    unique_accessions = list(set(assembly_accessions))
    print(f"Extracted {len(unique_accessions)} unique assembly accessions")
    
    # Query NCBI datasets
    with tempfile.TemporaryDirectory() as temp_dir:
        mapping = query_ncbi_datasets(unique_accessions, Path(temp_dir))
    
    # Display sample mappings
    if mapping:
        print("\nSample mappings (first 10):")
        for i, (assembly, biosample) in enumerate(list(mapping.items())[:10]):
            print(f"  {assembly} → {biosample}")
        if len(mapping) > 10:
            print(f"  ... and {len(mapping) - 10} more")
    
    # Check for unmapped accessions
    unmapped = set(unique_accessions) - set(mapping.keys())
    if unmapped:
        print(f"\nINFO: {len(unmapped)} accessions could not be mapped (will remain as GCF_ format):")
        for acc in sorted(list(unmapped)[:10]):
            print(f"  {acc}")
        if len(unmapped) > 10:
            print(f"  ... and {len(unmapped) - 10} more")
        print(f"\nThese {len(unmapped)} rows will keep their original GCF_ accession values.")
    
    # Update dataframe
    num_updated = 0
    for assembly, biosample in mapping.items():
        if assembly in accession_to_index:
            for idx in accession_to_index[assembly]:
                df.at[idx, 'sample_accession'] = biosample
                num_updated += 1
    
    print(f"\n{num_updated} rows will be updated with BioSample accessions")
    print(f"{len(refseq_df) - num_updated} rows will remain unchanged (keeping GCF_ format)")
    
    # Save or show dry-run message
    if args.dry_run:
        print("\nDry-run complete. Use without --dry-run to save changes.")
        print("\nExample of updated rows:")
        updated_rows = df.loc[refseq_df.index, ['Sample', 'is_refseq', 'sample_accession']].head(5)
        print(updated_rows.to_string(index=False))
    else:
        print(f"\nSaving updated metadata to {METADATA_PATH}...")
        df.to_csv(METADATA_PATH, sep='\t', index=False)
        print("✓ Metadata updated successfully!")


if __name__ == "__main__":
    main()
