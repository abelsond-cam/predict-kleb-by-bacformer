#!/usr/bin/env python3
"""Update sample_accession values for RefSeq and NCTC genomes by mapping assembly to BioSample accessions.

This script:
1. Loads metadata TSV
2. Extracts assembly accessions (GCF_* for RefSeq, GCA_* for NCTC) from is_refseq=True or is_nctc=True rows
3. Queries NCBI datasets to map assembly -> BioSample accessions
4. Updates sample_accession column with BioSample accessions (SAM*)
5. Saves updated metadata back to file

For accessions that cannot be mapped, they remain unchanged as assembly accessions.
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


def extract_assembly_accession(full_accession: str) -> str:
    """
    Extract the assembly accession from a full assembly name.
    
    Args:
        full_accession: Full assembly string (e.g., 'GCF_020526085.1_ASM2052608v1_genomic' or 'GCA_900451185.1_44503_H01_genomic')
    
    Returns:
        Just the accession part (e.g., 'GCF_020526085.1' or 'GCA_900451185.1')
    """
    # Split by underscore and take first two parts (GCF_XXXXXXXX.X or GCA_XXXXXXXX.X)
    parts = full_accession.split('_')
    if len(parts) >= 2 and parts[0] in ('GCF', 'GCA'):
        return f"{parts[0]}_{parts[1]}"
    # If it doesn't match expected format, return as-is
    return full_accession


def query_ncbi_datasets(accessions: list[str], temp_dir: Path, retry_on_error: bool = True) -> dict[str, str]:
    """
    Query NCBI datasets to map assembly accessions to BioSample accessions.
    
    Args:
        accessions: List of assembly accessions (e.g., ['GCF_020526085.1'])
        temp_dir: Temporary directory for intermediate files
        retry_on_error: If True and batch fails, retry with smaller sub-batches
    
    Returns:
        Dictionary mapping assembly accession -> BioSample accession
        (Accessions that cannot be mapped will not be in the dictionary)
    """
    # Write accessions to temporary input file
    input_file = temp_dir / f"gcf_list_{len(accessions)}.txt"
    with open(input_file, 'w') as f:
        for acc in accessions:
            f.write(f"{acc}\n")
    
    print(f"Querying NCBI datasets for {len(accessions)} assembly accessions...")
    
    # Run datasets command piped to dataformat
    output_file = temp_dir / f"assembly_to_biosample_{len(accessions)}.tsv"
    
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
            
            # If retry is enabled, try to recover
            if retry_on_error:
                if len(accessions) > 10:
                    # Split in half and retry
                    print(f"Retrying with smaller sub-batches to isolate problematic accessions...", file=sys.stderr)
                    mid = len(accessions) // 2
                    first_half = accessions[:mid]
                    second_half = accessions[mid:]
                    
                    print(f"  Retrying first sub-batch ({len(first_half)} accessions)...", file=sys.stderr)
                    mapping.update(query_ncbi_datasets(first_half, temp_dir, retry_on_error=True))
                    
                    print(f"  Retrying second sub-batch ({len(second_half)} accessions)...", file=sys.stderr)
                    mapping.update(query_ncbi_datasets(second_half, temp_dir, retry_on_error=True))
                    
                    return mapping
                elif len(accessions) > 1:
                    # Try one at a time to isolate the exact problematic accession(s)
                    print(f"Batch small enough - trying {len(accessions)} accessions individually...", file=sys.stderr)
                    for acc in accessions:
                        result = query_ncbi_datasets([acc], temp_dir, retry_on_error=False)
                        if result:
                            mapping.update(result)
                        else:
                            print(f"    ✗ Failed to map: {acc}", file=sys.stderr)
                    
                    if mapping:
                        print(f"  ✓ Successfully mapped {len(mapping)} out of {len(accessions)} accessions", file=sys.stderr)
                    return mapping
                else:
                    # Single accession that failed
                    print(f"Single accession failed to map: {accessions[0]}", file=sys.stderr)
                    return mapping
            else:
                print("Retry disabled - all accessions in this batch will remain unchanged as GCF_ format", file=sys.stderr)
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
        description="Update sample_accession for RefSeq and NCTC genomes with BioSample accessions"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test with first 100 rows without saving changes"
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
    
    # Filter to is_refseq == True OR is_nctc == True
    refseq_mask = df['is_refseq'].astype(str).str.lower().isin(['true', '1', 'yes'])
    nctc_mask = df['is_nctc'].astype(str).str.lower().isin(['true', '1', 'yes'])
    combined_mask = refseq_mask | nctc_mask
    selected_df = df[combined_mask].copy()
    
    total_refseq = refseq_mask.sum()
    total_nctc = nctc_mask.sum()
    total_selected = len(selected_df)
    
    print(f"Found {total_refseq} rows with is_refseq=True")
    print(f"Found {total_nctc} rows with is_nctc=True")
    print(f"Total selected: {total_selected} rows")
    
    if total_selected == 0:
        print("No RefSeq or NCTC rows to update. Exiting.")
        return
    
    # Apply dry-run limit
    if args.dry_run:
        print("Dry-run mode: limiting to first 100 rows")
        selected_df = selected_df.head(100)
    
    # Extract assembly accessions (only process GCF_* and GCA_* accessions)
    assembly_accessions = []
    accession_to_index = {}  # Map accession -> list of row indices in original df
    skipped_sam_format = 0
    
    for idx, row in selected_df.iterrows():
        full_acc = row['sample_accession']
        # Only process if it starts with GCF_ or GCA_ (assembly accessions)
        if not (str(full_acc).startswith('GCF_') or str(full_acc).startswith('GCA_')):
            skipped_sam_format += 1
            continue
        
        assembly_acc = extract_assembly_accession(str(full_acc))
        assembly_accessions.append(assembly_acc)
        
        if assembly_acc not in accession_to_index:
            accession_to_index[assembly_acc] = []
        accession_to_index[assembly_acc].append(idx)
    
    print(f"Skipped {skipped_sam_format} rows already in SAM format (not assembly accessions)")
    
    # Get unique accessions for querying
    unique_accessions = list(set(assembly_accessions))
    print(f"Extracted {len(unique_accessions)} unique assembly accessions (GCF_ and GCA_)")
    
    # Query NCBI datasets in batches of 500 to avoid API limits
    batch_size = 500
    all_mappings = {}
    
    if len(unique_accessions) == 0:
        print("No assembly accessions (GCF_* or GCA_*) to query. Exiting.")
        return
    
    with tempfile.TemporaryDirectory() as temp_dir:
        total_batches = (len(unique_accessions) + batch_size - 1) // batch_size
        
        for i in range(0, len(unique_accessions), batch_size):
            batch = unique_accessions[i:i + batch_size]
            batch_num = i // batch_size + 1
            print(f"\n{'='*60}")
            print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} accessions)")
            print(f"{'='*60}")
            
            batch_mapping = query_ncbi_datasets(batch, Path(temp_dir))
            all_mappings.update(batch_mapping)
            print(f"Batch {batch_num} complete: {len(batch_mapping)} mappings retrieved")
        
        print(f"\n{'='*60}")
        print(f"All batches complete: {len(all_mappings)} total mappings retrieved")
        print(f"{'='*60}\n")
    
    mapping = all_mappings
    
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
        print(f"\nINFO: {len(unmapped)} accessions could not be mapped (will remain as assembly accessions):")
        for acc in sorted(list(unmapped)[:10]):
            print(f"  {acc}")
        if len(unmapped) > 10:
            print(f"  ... and {len(unmapped) - 10} more")
        print(f"\nThese {len(unmapped)} rows will keep their original assembly accession values.")
    
    # Update dataframe
    num_updated = 0
    for assembly, biosample in mapping.items():
        if assembly in accession_to_index:
            for idx in accession_to_index[assembly]:
                df.at[idx, 'sample_accession'] = biosample
                num_updated += 1
    
    # Calculate how many assembly rows will remain unchanged
    num_assembly_rows = len(assembly_accessions)  # Total assembly rows we tried to process
    num_assembly_unchanged = num_assembly_rows - num_updated
    
    print(f"\n{num_updated} rows will be updated with BioSample accessions")
    print(f"{num_assembly_unchanged} assembly accession rows will remain unchanged (could not be mapped)")
    print(f"{skipped_sam_format} rows were already in SAM format (skipped)")
    
    # Save or show dry-run message
    if args.dry_run:
        print("\nDry-run complete. Use without --dry-run to save changes.")
        print("\nExample of updated rows:")
        updated_rows = df.loc[selected_df.index, ['Sample', 'is_refseq', 'is_nctc', 'sample_accession']].head(5)
        print(updated_rows.to_string(index=False))
    else:
        print(f"\nSaving updated metadata to {METADATA_PATH}...")
        df.to_csv(METADATA_PATH, sep='\t', index=False)
        print("✓ Metadata updated successfully!")


if __name__ == "__main__":
    main()
