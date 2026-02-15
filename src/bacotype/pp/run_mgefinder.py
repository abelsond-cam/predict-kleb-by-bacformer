#!/usr/bin/env python3
"""Run MGEfinder workflow on downloaded FASTQ files."""

import argparse
import sys
from pathlib import Path

# Import genome selection utility
from select_genomes_reference_comparison import select_genomes

# Hardcoded data paths (standalone - no bacotype package dependency)
FASTQ_BASE_DIR = Path("/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/raw/fastq")
WARM_DIR = Path("/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/")


def find_fastq_files(accession, fastq_base_dir):
    """Find FASTQ files for a given accession.

    Args:
        accession: SRA accession ID
        fastq_base_dir: Base directory containing FASTQ files

    Returns
    -------
        List of FASTQ file paths
    """
    acc_dir = fastq_base_dir / accession
    if not acc_dir.exists():
        raise FileNotFoundError(f"FASTQ directory not found for {accession}: {acc_dir}")

    # Find all .fastq.gz files
    fastq_files = list(acc_dir.glob("*.fastq.gz")) + list(acc_dir.glob("*.fq.gz"))

    if not fastq_files:
        raise FileNotFoundError(f"No FASTQ files found in {acc_dir}")

    return sorted(fastq_files)


def run_mgefinder(reference, comparisons, fastq_base_dir, output_dir):
    """Run MGEfinder comparing reference against comparison genomes.

    Args:
        reference: Reference genome accession
        comparisons: List of comparison genome accessions
        fastq_base_dir: Base directory containing FASTQ files
        output_dir: Output directory for MGEfinder results

    Returns
    -------
        Success status
    """
    print("\n--- MGEfinder Workflow ---")
    print(f"Reference: {reference}")
    print(f"Comparisons: {len(comparisons)} genomes")
    print(f"FASTQ directory: {fastq_base_dir}")
    print(f"Output directory: {output_dir}")

    # Verify reference FASTQ files exist
    try:
        ref_files = find_fastq_files(reference, fastq_base_dir)
        print(f"\nReference FASTQ files: {len(ref_files)} found")
        for f in ref_files:
            print(f"  - {f.name}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return False

    # Verify comparison FASTQ files exist
    print("\nVerifying comparison FASTQ files...")
    missing = []
    for comp in comparisons:
        try:
            comp_files = find_fastq_files(comp, fastq_base_dir)
            print(f"  ✓ {comp}: {len(comp_files)} files")
        except FileNotFoundError:
            print(f"  ✗ {comp}: missing")
            missing.append(comp)

    if missing:
        print(f"\nError: Missing FASTQ files for {len(missing)} accessions")
        return False

    # TODO: Implement actual MGEfinder command execution
    print("\n--- Running MGEfinder ---")
    print("TODO: Implement MGEfinder command execution")
    print("This requires:")
    print("  1. Converting FASTQ to format expected by MGEfinder (if needed)")
    print("  2. Running MGEfinder workflow commands")
    print("  3. Processing and saving results")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MGEfinder on selected genomes")
    parser.add_argument("--n", type=int, default=10, help="Number of comparison genomes per KL subset")
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Output directory for MGEfinder results (default: auto-generated)"
    )
    args = parser.parse_args()

    print(f"===== MGEfinder Workflow (n={args.n}) =====")

    # Select genomes
    print("\n--- Step 1: Genome Selection ---")
    reference, comparisons = select_genomes(args.n)

    # Set output directory
    if args.output_dir is None:
        output_dir = WARM_DIR / "processed" / "mgefinder_results"
    else:
        output_dir = args.output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    # Run MGEfinder
    print("\n--- Step 2: Running MGEfinder ---")
    success = run_mgefinder(reference, comparisons, FASTQ_BASE_DIR, output_dir)

    if success:
        print("\n✓ MGEfinder workflow completed!")
    else:
        print("\n✗ MGEfinder workflow failed!")
        sys.exit(1)
