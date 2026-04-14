"""Generate protein sequences from Klebsiella genome assemblies (CPU-only).

This script extracts protein sequences from .bakta.gbff.gz files using parallel processing.
It runs Step 1 of the Bacformer pipeline (preprocess_genome_assembly) only.

The output protein sequences can then be used as input for GPU-based embedding generation.
"""

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count
from pathlib import Path

import pandas as pd
from bacformer.pp import preprocess_genome_assembly
from tqdm import tqdm

from predict_kleb_by_bacformer.data_paths import data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("protein_sequences_processing.log"),
    ],
)
logger = logging.getLogger(__name__)


def find_gbff_files(input_dir: Path, limit: int | None = None) -> list[Path]:
    """Recursively find all .bakta.gbff.gz files in the input directory.

    Args:
        input_dir: Directory to search for gbff files
        limit: Optional limit on number of files to return (must be positive)

    Returns
    -------
        List of paths to gbff files
    """
    logger.info(f"Searching for .bakta.gbff.gz files in {input_dir}")

    if limit is not None and limit > 0:
        # If limit is set to a positive number, only find the first n files (more efficient)
        logger.info(f"Looking for first {limit} files (early stopping enabled)")
        gbff_files = []
        for filepath in input_dir.rglob("*.bakta.gbff.gz"):
            gbff_files.append(filepath)
            if len(gbff_files) >= limit:
                break
        gbff_files = sorted(gbff_files)
    else:
        # Find all files (when limit is None, 0, or negative)
        if limit is not None and limit <= 0:
            logger.info(f"Limit is {limit}, processing all files")
        gbff_files = sorted(input_dir.rglob("*.bakta.gbff.gz"))

    logger.info(f"Found {len(gbff_files)} files to process")
    return gbff_files


def extract_sample_id(filepath: Path) -> str:
    """Extract sample ID from filepath.

    Example: /path/to/D000/SAMD00052611/SAMD00052611.bakta.gbff.gz -> SAMD00052611

    Args:
        filepath: Path to the gbff file

    Returns
    -------
        Sample ID string
    """
    # The sample ID is the parent directory name
    return filepath.parent.name


def check_output_exists(sample_id: str, output_dir: Path) -> bool:
    """Check if protein sequences file already exists for a given sample.

    Args:
        sample_id: Sample identifier
        output_dir: Output directory for protein sequences

    Returns
    -------
        True if output file exists, False otherwise
    """
    protein_file = output_dir / f"{sample_id}_protein_sequences.parquet"
    return protein_file.exists()


def save_to_parquet(data_dict: dict, output_path: Path) -> None:
    """Save data dictionary to parquet file.

    Args:
        data_dict: Dictionary containing data to save
        output_path: Path where parquet file should be saved
    """
    # Convert dict to DataFrame for parquet serialization
    df = pd.DataFrame([data_dict])
    df.to_parquet(output_path, engine="pyarrow", compression="snappy")


def process_single_genome(args_tuple: tuple) -> tuple[str, bool, str, float]:
    """Process a single genome file - extract protein sequences.

    Args:
        args_tuple: Tuple of (gbff_path, output_dir, skip_existing)

    Returns
    -------
        Tuple of (sample_id, success, error_message, processing_time)
    """
    gbff_path, output_dir, skip_existing = args_tuple
    sample_id = extract_sample_id(gbff_path)
    start_time = time.time()

    try:
        # Check if already processed
        if skip_existing and check_output_exists(sample_id, output_dir):
            return sample_id, True, "Already exists (skipped)", 0.0

        # Preprocess genome assembly to extract protein sequences
        genome_info = preprocess_genome_assembly(filepath=str(gbff_path))

        # Remove 'strain_name', 'accession_name', 'protein_name' keys from genome_info
        genome_info.pop("strain_name", None)
        genome_info.pop("accession_name", None)
        genome_info.pop("protein_name", None)
        # Add sample accession to the result as the first key
        genome_info = {"sample_id": sample_id, **genome_info}
        # Save to parquet
        protein_output_path = output_dir / f"{sample_id}_protein_sequences.parquet"
        save_to_parquet(genome_info, protein_output_path)
        elapsed = time.time() - start_time
        return sample_id, True, "", elapsed
    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = f"Error processing {sample_id}: {str(e)}"
        logger.error(error_msg)
        return sample_id, False, error_msg, elapsed


def main():
    """Main execution function."""
    logger.info("Script started")
    sys.stdout.flush()

    parser = argparse.ArgumentParser(
        description="Generate protein sequences from Klebsiella genomes (CPU-only)"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Limit number of genomes to process (for testing)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip genomes that have already been processed",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=76,
        help="Number of parallel workers (default: 76)",
    )

    args = parser.parse_args()

    # Setup paths
    input_dir = data.klebsiella_gbff_dir
    output_dir = data.klebsiella_protein_sequences_dir

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("PHASE 1: Setup")
    logger.info("=" * 60)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Workers requested: {args.workers}")
    sys.stdout.flush()

    # Find input files
    logger.info("")
    logger.info("PHASE 2: Discovering .bakta.gbff.gz files (this may take a while)")
    sys.stdout.flush()
    gbff_files = find_gbff_files(input_dir, limit=args.n)

    if not gbff_files:
        logger.error(f"No .bakta.gbff.gz files found in {input_dir}")
        sys.exit(1)

    # Filter out already processed files if requested
    logger.info("")
    logger.info("PHASE 3: Filtering (skip-existing check)")
    sys.stdout.flush()
    if args.skip_existing:
        original_count = len(gbff_files)
        gbff_files = [
            f for f in gbff_files if not check_output_exists(extract_sample_id(f), output_dir)
        ]
        skipped = original_count - len(gbff_files)
        if skipped > 0:
            logger.info(f"Skipping {skipped} already processed genomes")

    if not gbff_files:
        logger.info("All genomes have already been processed")
        return

    # Determine number of workers
    num_workers = min(args.workers, cpu_count(), len(gbff_files))

    logger.info("")
    logger.info("=" * 60)
    logger.info("PHASE 4: Parallel processing")
    logger.info("=" * 60)
    logger.info(f"Total samples to process: {len(gbff_files)}")
    logger.info(f"Worker processes: {num_workers}")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    sys.stdout.flush()

    # Prepare arguments for parallel processing
    process_args = [(f, output_dir, args.skip_existing) for f in gbff_files]

    # Record start time
    overall_start_time = time.time()

    # Process genomes in parallel
    results = {"success": [], "failed": []}
    error_log = []
    genome_times = []

    # tqdm: use file=sys.stdout and mininterval for batch/SLURM-friendly output
    with Pool(processes=num_workers) as pool:
        pbar = tqdm(
            total=len(gbff_files),
            desc="Processing genomes",
            unit="sample",
            file=sys.stdout,
            mininterval=5.0,
            dynamic_ncols=False,
            smoothing=0.1,
        )
        for sample_id, success, error_msg, elapsed in pool.imap_unordered(
            process_single_genome, process_args
        ):
            if success:
                results["success"].append(sample_id)
                if elapsed > 0:  # Don't count skipped files
                    genome_times.append(elapsed)
                if error_msg:  # "Already exists (skipped)"
                    logger.debug(f"{sample_id}: {error_msg}")
                else:
                    logger.info(f"{sample_id}: SUCCESS ({elapsed:.1f}s)")
            else:
                results["failed"].append(sample_id)
                error_log.append({"sample_id": sample_id, "error": error_msg})
                logger.warning(f"{sample_id}: FAILED")

            pbar.update(1)

            # Log progress and timing estimates periodically (every 25 samples)
            completed = len(results["success"]) + len(results["failed"])
            if len(genome_times) > 0 and completed % 25 == 0:
                avg_time = sum(genome_times) / len(genome_times)
                remaining = len(gbff_files) - completed
                est_remaining = timedelta(seconds=int(avg_time * remaining / num_workers))
                logger.info(
                    f"Progress: {completed}/{len(gbff_files)} samples | "
                    f"Avg {avg_time:.1f}s/sample | "
                    f"Est. remaining: {est_remaining}"
                )
                sys.stdout.flush()

        pbar.close()

    # Calculate total elapsed time
    overall_elapsed = time.time() - overall_start_time

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("PHASE 5: Summary")
    logger.info("=" * 60)
    logger.info(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total time elapsed: {timedelta(seconds=int(overall_elapsed))}")
    logger.info(f"Successfully processed: {len(results['success'])} genomes")
    logger.info(f"Failed: {len(results['failed'])} genomes")

    if genome_times:
        avg_time = sum(genome_times) / len(genome_times)
        min_time = min(genome_times)
        max_time = max(genome_times)
        logger.info("Timing statistics (for newly processed genomes):")
        logger.info(f"  - Average: {avg_time:.1f}s per genome")
        logger.info(f"  - Min: {min_time:.1f}s")
        logger.info(f"  - Max: {max_time:.1f}s")
        logger.info(f"  - Total genomes processed: {len(genome_times)}")
        logger.info(f"  - Throughput: {len(genome_times)/overall_elapsed*60:.1f} genomes/minute")

    if error_log:
        error_log_path = Path("protein_sequences_errors.log")
        with open(error_log_path, "w") as f:
            for entry in error_log:
                f.write(f"{entry['sample_id']}: {entry['error']}\n")
        logger.info(f"Error details saved to {error_log_path}")

    logger.info("=" * 60)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
