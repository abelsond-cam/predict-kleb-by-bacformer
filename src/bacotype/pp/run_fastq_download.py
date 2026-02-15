#!/usr/bin/env python3
"""Download FASTQ files using fastq-dl for selected genomes."""

import argparse
import subprocess
import sys
from pathlib import Path

# Import genome selection utility
from select_genomes_reference_comparison import select_genomes

# Hardcoded data paths (standalone - no bacotype package dependency)
FASTQ_OUTPUT_DIR = Path("/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/raw/fastq")


def check_fastq_exists(outdir):
    """
    Check if FASTQ files already exist in the output directory.
    
    Args:
        outdir: Path to the accession's output directory
    
    Returns:
        True if FASTQ files exist, False otherwise
    """
    if not outdir.exists():
        return False
    
    # Check for common FASTQ file extensions
    fastq_patterns = ["*.fastq", "*.fastq.gz", "*.fq", "*.fq.gz"]
    for pattern in fastq_patterns:
        if list(outdir.glob(pattern)):
            return True
    
    return False


def download_fastq(accessions, output_base_dir, skip_existing=True):
    """
    Download FASTQ files using fastq-dl.
    
    Args:
        accessions: List of SRA accession IDs
        output_base_dir: Base directory for FASTQ downloads
        skip_existing: If True, skip accessions that already have FASTQ files (default: True)
    
    Returns:
        Dictionary with success/failure/skipped counts
    """
    results = {"success": [], "failed": [], "skipped": []}
    
    for i, acc in enumerate(accessions, 1):
        print(f"\n[{i}/{len(accessions)}] Processing {acc}...")
        
        # Create output directory for this accession
        outdir = output_base_dir / acc
        outdir.mkdir(parents=True, exist_ok=True)
        
        # Check if already exists
        if skip_existing and check_fastq_exists(outdir):
            print(f"⊘ Skipping {acc} (FASTQ files already exist)")
            results["skipped"].append(acc)
            continue
        
        # Run fastq-dl
        try:
            cmd = ["fastq-dl", "--accession", acc, "--outdir", str(outdir)]
            print(f"Command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            print(f"✓ Successfully downloaded {acc}")
            if result.stdout:
                print(f"  Output: {result.stdout.strip()}")
            results["success"].append(acc)
            
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to download {acc}")
            print(f"  Error: {e.stderr.strip()}")
            results["failed"].append(acc)
        except Exception as e:
            print(f"✗ Unexpected error downloading {acc}: {e}")
            results["failed"].append(acc)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download FASTQ files for MGEfinder analysis")
    parser.add_argument("--n", type=int, default=10, help="Number of comparison genomes per KL subset")
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Re-download even if FASTQ files already exist (default: skip existing)"
    )
    args = parser.parse_args()
    
    skip_existing = not args.overwrite_existing
    mode_msg = "overwriting existing files" if args.overwrite_existing else "skipping existing files"
    print(f"===== FASTQ Download Workflow (n={args.n}, {mode_msg}) =====")
    
    # Select genomes
    print("\n--- Step 1: Genome Selection ---")
    reference, comparisons = select_genomes(args.n)
    
    # Combine all accessions for download
    all_accessions = [reference] + comparisons
    print(f"\n--- Step 2: Downloading {len(all_accessions)} FASTQ files ---")
    print(f"Output directory: {FASTQ_OUTPUT_DIR}")
    
    # Download
    results = download_fastq(all_accessions, FASTQ_OUTPUT_DIR, skip_existing=skip_existing)
    
    # Summary
    print("\n===== Download Summary =====")
    print(f"Total accessions: {len(all_accessions)}")
    print(f"Successfully downloaded: {len(results['success'])}")
    print(f"Skipped (already exist): {len(results['skipped'])}")
    print(f"Failed: {len(results['failed'])}")
    
    if results["failed"]:
        print("\nFailed accessions:")
        for acc in results["failed"]:
            print(f"  - {acc}")
        sys.exit(1)
    
    print("\n✓ All downloads completed successfully!")
