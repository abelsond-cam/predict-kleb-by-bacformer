"""
Add assembly_file and gff_file columns to metadata TSV.

For each Sample in the metadata:
- Finds the corresponding assembly file in david/raw/assemblies
- Finds the corresponding GFF file in either ncbi_gff3 (if is_refseq or is_nctc)
  or klebsiella_gff3 (otherwise)
- Adds absolute paths to new columns and prints coverage statistics
"""

from __future__ import annotations

import sys
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from tqdm import tqdm


DATA_DIR = Path("/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/")

METADATA_F = Path(
    "/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/final/metadata_final_curated_slimmed.tsv"
)

ASSEMBLY_DIR = DATA_DIR / "seb/assemblies_2"
NCBI_GFF_DIR = DATA_DIR / "david/raw/ncbi_gff3"
KLEBSIELLA_GFF_DIR = DATA_DIR / "david/raw/klebsiella_gff3"


def _list_files(directory: Path, recursive: bool = False) -> list[Path]:
    """List all files in the given directory, sorted by name.
    
    Args:
        directory: Directory to list files from
        recursive: If True, search recursively in subdirectories
    
    Includes symlinks without checking if targets exist (assumes they're valid).
    """
    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")
    
    if recursive:
        files = [p for p in directory.rglob("*") if p.is_file() or p.is_symlink()]
    else:
        files = [p for p in directory.iterdir() if p.is_file() or p.is_symlink()]
    
    return sorted(files)


def _find_matching_files(sample: str, files: Iterable[Path]) -> list[Path]:
    """Return all files whose basename contains the sample string as a substring."""
    if not isinstance(sample, str):
        sample = str(sample)
    return [p for p in files if sample in p.name]


def _summarise_matches_for_dir(
    total_files: int,
    used_files: set[Path],
    label: str,
) -> None:
    """Print summary of how many files in a directory were matched to at least one sample."""
    if total_files == 0:
        all_matched_str = "N/A (directory empty)"
    else:
        all_matched_str = str(len(used_files) == total_files)

    print(
        f"{label}: total_files={total_files}, "
        f"matched_at_least_once={len(used_files)}, "
        f"all_files_matched={all_matched_str}"
    )


def add_assembly_paths(df: pd.DataFrame, assembly_files: list[Path]) -> pd.DataFrame:
    """
    Add assembly_file column to metadata by matching Sample names to assembly filenames.

    Args:
        df: Metadata DataFrame with a Sample column
        assembly_files: Pre-computed list of all files in assembly directory

    For each Sample, searches the provided file list for files containing the sample name,
    and adds the absolute path to a new assembly_file column.
    Prints statistics about match coverage.
    """
    if "Sample" not in df.columns:
        raise KeyError("Expected 'Sample' column in metadata.")

    used_assembly_files: set[Path] = set()
    assembly_paths: list[str | None] = []

    for sample in tqdm(df["Sample"], total=len(df), desc="Adding assembly paths"):
        matches = _find_matching_files(sample, assembly_files)
        if not matches:
            assembly_paths.append(None)
            continue

        matches = sorted(matches, key=lambda p: p.name)
        chosen = matches[0]
        assembly_paths.append(str(chosen))
        used_assembly_files.add(chosen)

        if len(matches) > 1:
            msg = (
                f"Multiple assembly files matched sample {sample!r}: "
                f"{[p.name for p in matches]}. Using {chosen.name}"
            )
            print(msg, file=sys.stderr)

    df = df.copy()
    df["assembly_file"] = assembly_paths

    total_samples = len(df)
    n_found = df["assembly_file"].notna().sum()
    n_not_found = total_samples - n_found

    print("Assembly files:")
    print(f"  Samples in metadata: {total_samples}")
    print(f"  Samples with assembly_file: {n_found}")
    print(f"  Samples without assembly_file: {n_not_found}")

    _summarise_matches_for_dir(
        total_files=len(assembly_files),
        used_files=used_assembly_files,
        label="  Assembly directory coverage",
    )

    return df


def add_gff_paths(df: pd.DataFrame, ncbi_files: list[Path], kleb_files: list[Path]) -> pd.DataFrame:
    """
    Add gff_file column to metadata by matching Sample names to GFF filenames.

    Args:
        df: Metadata DataFrame with Sample, is_refseq, and is_nctc columns
        ncbi_files: Pre-computed list of all files in ncbi_gff3 directory
        kleb_files: Pre-computed list of all files in klebsiella_gff3 directory

    Uses is_refseq and is_nctc columns to determine search directory:
    - If either is_refseq or is_nctc is True: search in ncbi_gff3
    - If both are False: search in klebsiella_gff3
    Prints statistics about match coverage for both directories.
    """
    required_cols = {"Sample", "is_refseq", "is_nctc"}
    missing = required_cols.difference(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise KeyError(f"Missing required metadata columns: {missing_str}")

    used_ncbi: set[Path] = set()
    used_kleb: set[Path] = set()

    gff_paths: list[str | None] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Adding GFF paths"):
        sample = str(row["Sample"])
        is_refseq = bool(row["is_refseq"])
        is_nctc = bool(row["is_nctc"])

        search_ncbi = is_refseq or is_nctc
        files_to_search = ncbi_files if search_ncbi else kleb_files

        matches = _find_matching_files(sample, files_to_search)
        if not matches:
            gff_paths.append(None)
            continue

        matches = sorted(matches, key=lambda p: p.name)
        chosen = matches[0]
        gff_paths.append(str(chosen))

        if search_ncbi:
            used_ncbi.add(chosen)
        else:
            used_kleb.add(chosen)

        if len(matches) > 1:
            msg = (
                f"Multiple GFF files matched sample {sample!r} in "
                f"{'ncbi_gff3' if search_ncbi else 'klebsiella_gff3'}: "
                f"{[p.name for p in matches]}. Using {chosen.name}"
            )
            print(msg, file=sys.stderr)

    df = df.copy()
    df["gff_file"] = gff_paths

    total_samples = len(df)
    n_found = df["gff_file"].notna().sum()
    n_not_found = total_samples - n_found

    print("\nGFF files:")
    print(f"  Samples in metadata: {total_samples}")
    print(f"  Samples with gff_file: {n_found}")
    print(f"  Samples without gff_file: {n_not_found}")

    _summarise_matches_for_dir(
        total_files=len(ncbi_files),
        used_files=used_ncbi,
        label="  ncbi_gff3 directory coverage",
    )
    _summarise_matches_for_dir(
        total_files=len(kleb_files),
        used_files=used_kleb,
        label="  klebsiella_gff3 directory coverage",
    )

    return df


def run(metadata_path: Path | None = None) -> None:
    """
    Load metadata TSV, add assembly_file and gff_file columns, and overwrite the file.

    Args:
        metadata_path: Optional path to metadata TSV; defaults to METADATA_F constant.
    """
    meta_path = Path(metadata_path) if metadata_path is not None else METADATA_F

    print(f"Metadata file: {meta_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file does not exist: {meta_path}")
    print("  File exists: Yes")

    print(f"\nAssembly directory: {ASSEMBLY_DIR}")
    if not ASSEMBLY_DIR.exists():
        raise FileNotFoundError(f"Assembly directory does not exist: {ASSEMBLY_DIR}")
    print("  Directory exists: Yes")
    print("  Listing files (recursive)...")
    assembly_files = _list_files(ASSEMBLY_DIR, recursive=True)
    print(f"  Files found (including symlinks): {len(assembly_files)}")
    if len(assembly_files) == 0:
        raise ValueError(f"Assembly directory is empty: {ASSEMBLY_DIR}")

    print(f"\nNCBI GFF directory: {NCBI_GFF_DIR}")
    if not NCBI_GFF_DIR.exists():
        raise FileNotFoundError(f"NCBI GFF directory does not exist: {NCBI_GFF_DIR}")
    print("  Directory exists: Yes")
    print("  Listing files (non-recursive)...")
    ncbi_files = _list_files(NCBI_GFF_DIR, recursive=False)
    print(f"  Files found (including symlinks): {len(ncbi_files)}")
    if len(ncbi_files) == 0:
        raise ValueError(f"NCBI GFF directory is empty: {NCBI_GFF_DIR}")

    print(f"\nKlebsiella GFF directory: {KLEBSIELLA_GFF_DIR}")
    if not KLEBSIELLA_GFF_DIR.exists():
        raise FileNotFoundError(f"Klebsiella GFF directory does not exist: {KLEBSIELLA_GFF_DIR}")
    print("  Directory exists: Yes")
    print("  Listing files (non-recursive)...")
    kleb_files = _list_files(KLEBSIELLA_GFF_DIR, recursive=False)
    print(f"  Files found (including symlinks): {len(kleb_files)}")
    if len(kleb_files) == 0:
        raise ValueError(f"Klebsiella GFF directory is empty: {KLEBSIELLA_GFF_DIR}")

    print("\n" + "="*80)
    print("Loading metadata...")
    df = pd.read_csv(meta_path, sep="\t", low_memory=False)
    print(f"Loaded {len(df)} samples from metadata\n")

    df = add_assembly_paths(df, assembly_files)
    df = add_gff_paths(df, ncbi_files, kleb_files)

    print("\n" + "="*80)
    print(f"Writing updated metadata to: {meta_path}")
    df.to_csv(meta_path, sep="\t", index=False)
    print("Done!")


def main(argv: list[str] | None = None) -> None:
    """
    CLI entry point for adding assembly and GFF paths to metadata.

    Usage: python -m bacotype.pp.add_paths_gff_fna_to_metadata [optional_metadata_tsv]
    """
    if argv is None:
        argv = sys.argv[1:]

    if len(argv) > 1:
        raise SystemExit(
            "Usage: python -m bacotype.pp.add_paths_gff_fna_to_metadata "
            "[optional_metadata_tsv]"
        )

    metadata_path = Path(argv[0]) if argv else None
    run(metadata_path)


if __name__ == "__main__":
    main()

