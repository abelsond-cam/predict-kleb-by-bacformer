r"""
Build Panaroo input file from metadata for a chosen clonal group.

Reads metadata, filters by Clonal group, keeps only samples with both gff_file
and assembly_file present on disk. GFF files that are gzipped (.gz) are
decompressed into the run subdir (gff_unzipped/) so only the N samples needed
are unzipped; assemblies are likewise decompressed into assembly_unzipped/.
For each selected sample, a single Prokka-style combined GFF+FASTA file is
created via Panaroo's convert logic in converted_gff/, and the input file
lists only those combined GFF paths (one per line).
Used by slurm_scripts/panaroo_run_strain.sh.
"""

from __future__ import annotations

import argparse
import gzip
import sys
from pathlib import Path
import shutil

import pandas as pd


METADATA_FILE = Path(
    "/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/final/metadata_final_curated_slimmed.tsv"
)
BASE_DIR = Path("/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw")
DEFAULT_OUTDIR = Path(
    "/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/panaroo_run"
)
PANAROO_INPUT_FILENAME = "panaroo_input.txt"


GFF_UNZIPPED_SUBDIR = "gff_unzipped"
ASSEMBLY_UNZIPPED_SUBDIR = "assembly_unzipped"
CONVERTED_GFF_SUBDIR = "converted_gff"


def _abs_path(base: Path, rel: str | float) -> Path | None:
    """Build absolute path; return None if rel is null/NaN."""
    if pd.isna(rel) or rel is None or (isinstance(rel, str) and not rel.strip()):
        return None
    p = base / str(rel).lstrip("/")
    return p.resolve()


def _ensure_gff_unzipped(gff_path: Path, out_dir: Path, index: int) -> Path:
    """
    If gff_path ends with .gz, decompress to out_dir/gff_{index}.gff and return that path.
    Otherwise return gff_path unchanged.
    """
    if gff_path.suffix != ".gz":
        return gff_path
    out_path = out_dir / f"gff_{index}.gff"
    if out_path.exists():
        return out_path
    with gzip.open(gff_path, "rt") as f_in:
        with open(out_path, "w") as f_out:
            for line in f_in:
                # Drop any lines that start with "# " to match `sed -i '/^# /d'`
                if line.lstrip().startswith("# "):
                    continue
                f_out.write(line)
    return out_path


def _ensure_assembly_unzipped(assembly_path: Path, out_dir: Path, index: int) -> Path:
    """
    If assembly_path ends with .gz, decompress to out_dir/assembly_{index}.fna and return that path.
    Otherwise return assembly_path unchanged.
    """
    if assembly_path.suffix != ".gz":
        return assembly_path
    out_path = out_dir / f"assembly_{index}.fna"
    if out_path.exists():
        return out_path
    with gzip.open(assembly_path, "rt") as f_in:
        with open(out_path, "w") as f_out:
            for line in f_in:
                f_out.write(line)
    return out_path


import sys, os
import argparse
import gffutils as gff
from io import StringIO
from Bio import SeqIO


def clean_gff_string(gff_string):
    splitlines = gff_string.splitlines()
    lines_to_delete = []
    for index in range(len(splitlines)):
        if '##sequence-region' in splitlines[index]:
            lines_to_delete.append(index)
    for index in sorted(lines_to_delete, reverse=True):
        del splitlines[index]
    cleaned_gff = "\n".join(splitlines)
    return cleaned_gff


def convert(gfffile, outputfile, fastafile, is_ignore_overlapping):

    #Split file and parse
    with open(gfffile, 'r') as infile:
        lines = infile.read().replace(',','')

    if fastafile is None:
        split = lines.split('##FASTA')
        if len(split) != 2:
            print("Problem reading GFF3 file: ", gfffile)
            raise RuntimeError("Error reading GFF3 input!")
    else:
        with open(fastafile, 'r') as infile:
            fasta_lines = infile.read()
        split = [lines, fasta_lines]

    fasta_block = split[1]
    # Drop any leading comment lines before the first FASTA record to avoid
    # Biopython's deprecation warning about comments at the beginning of the
    # file. These lines start before any '>' header.
    fasta_lines = fasta_block.splitlines()
    start_idx = 0
    while start_idx < len(fasta_lines) and not fasta_lines[start_idx].lstrip().startswith(">"):
        start_idx += 1
    cleaned_fasta_block = "\n".join(fasta_lines[start_idx:])

    with StringIO(cleaned_fasta_block) as temp_fasta:
        sequences = list(SeqIO.parse(temp_fasta, 'fasta'))

    for seq in sequences:
        seq.description = ""

    parsed_gff = gff.create_db(clean_gff_string(split[0]),
                               dbfn=":memory:",
                               force=True,
                               keep_order=False,
                               merge_strategy="create_unique",
                               sort_attribute_values=True,
                               from_string=True)

    with open(outputfile, 'w') as outfile:
        # write gff part
        outfile.write("##gff-version 3\n")
        for seq in sequences:
            outfile.write(
                " ".join(["##sequence-region", seq.id, "1",
                          str(len(seq.seq))]) + "\n")

        prev_chrom = ""
        prev_end = -1
        ids = set()
        seen = set()
        seq_order = []
        for entry in parsed_gff.all_features(featuretype=(),
                                             order_by=('seqid', 'start')):
            entry.chrom = entry.chrom.split()[0]
            # skip non CDS
            if "CDS" not in entry.featuretype: continue
            # skip overlapping CDS if option is set
            if entry.chrom == prev_chrom and entry.start < prev_end and is_ignore_overlapping:
                continue
            # skip CDS that dont appear to be complete or have a premature stop codon

            premature_stop = False
            for sequence_index in range(len(sequences)):
                scaffold_id = sequences[sequence_index].id
                if scaffold_id == entry.seqid:
                    gene_sequence = sequences[sequence_index].seq[(
                        entry.start - 1):entry.stop]
                    if (len(gene_sequence) % 3 > 0) or (len(gene_sequence) <
                                                        34):
                        premature_stop = True
                        break
                    if entry.strand == "-":
                        gene_sequence = gene_sequence.reverse_complement()
                    if "*" in str(gene_sequence.translate())[:-1]:
                        premature_stop = True
                        break
            if premature_stop: continue

            c = 1
            while entry.attributes['ID'][0] in ids:
                entry.attributes['ID'][0] += "." + str(c)
                c += 1
            ids.add(entry.attributes['ID'][0])
            prev_chrom = entry.chrom
            prev_end = entry.end
            if entry.chrom not in seen:
                seq_order.append(entry.chrom)
                seen.add(entry.chrom)
            print(entry, file=outfile)

        # write fasta part
        outfile.write("##FASTA\n")
        sequences = [
            seq for x in seq_order for seq in sequences if seq.id == x
        ]
        if len(sequences) != len(seen):
            raise RuntimeError("Mismatch between fasta and GFF!")
        SeqIO.write(sequences, outfile, "fasta")

    return


def run(
    clonal_group: str,
    n: int,
    outdir: Path,
    metadata_file: Path,
    base_dir: Path,
) -> tuple[Path, Path]:
    """
    Filter metadata by clonal group, restrict to samples with both gff and assembly on disk.

    Optionally limit to first n, write Panaroo input file in run subdir.
    Returns (input_file_path, run_subdir_path).
    """
    df = pd.read_csv(metadata_file, sep="\t", low_memory=False)
    subset = df[df["Clonal group"] == clonal_group].copy()
    total_in_group = len(subset)
    if total_in_group == 0:
        print(f"No samples in clonal group '{clonal_group}'.")
        sys.exit(1)

    subset["gff_abs"] = subset["gff_file"].apply(lambda x: _abs_path(base_dir, x))
    subset["assembly_abs"] = subset["assembly_file"].apply(
        lambda x: _abs_path(base_dir, x)
    )
    has_gff = subset["gff_abs"].notna() & subset["gff_abs"].apply(
        lambda p: p.exists() if p is not None else False
    )
    has_assembly = subset["assembly_abs"].notna() & subset["assembly_abs"].apply(
        lambda p: p.exists() if p is not None else False
    )
    n_gff = int(has_gff.sum())
    n_assembly = int(has_assembly.sum())
    has_both = has_gff & has_assembly
    n_both = int(has_both.sum())

    print(f"Clonal group: {clonal_group}")
    print(f"  Total in group: {total_in_group}")
    print(f"  With gff (present on disk): {n_gff}")
    print(f"  With assembly (present on disk): {n_assembly}")
    print(f"  With both: {n_both}")

    if n_both == 0:
        print("No samples have both gff and assembly files. Exiting.")
        sys.exit(1)

    rows_both = subset.loc[has_both, ["gff_abs", "assembly_abs"]].copy()
    if n >= 1:
        rows_both = rows_both.head(n)
        print(f"  - Selected first {n} samples from clonal group {clonal_group}")

    n_written = len(rows_both)
    if n_written > n_both:
        print(f"  - Warning: {n_written} samples written, but only {n_both} have both gff and assembly files. Using all {n_both} samples.")

    if n == -1:
        run_subdir_name = f"{clonal_group}_all"
        print(f"  - Using all {n_written} samples from clonal group {clonal_group}")
    else:
        run_subdir_name = f"{clonal_group}_n{n}"
    run_subdir = outdir / run_subdir_name
    run_subdir.mkdir(parents=True, exist_ok=True)
    gff_unzipped_dir = run_subdir / GFF_UNZIPPED_SUBDIR
    gff_unzipped_dir.mkdir(exist_ok=True)
    assembly_unzipped_dir = run_subdir / ASSEMBLY_UNZIPPED_SUBDIR
    assembly_unzipped_dir.mkdir(exist_ok=True)
    converted_gff_dir = run_subdir / CONVERTED_GFF_SUBDIR
    converted_gff_dir.mkdir(exist_ok=True)
    input_path = run_subdir / PANAROO_INPUT_FILENAME

    with open(input_path, "w") as f:
        for i, (_, row) in enumerate(rows_both.iterrows()):
            sample_id = row["Sample"]
            gff_abs = row["gff_abs"]
            assembly_abs = row["assembly_abs"]
            gff_for_panaroo = _ensure_gff_unzipped(gff_abs, gff_unzipped_dir, i)
            assembly_for_panaroo = _ensure_assembly_unzipped(
                assembly_abs, assembly_unzipped_dir, i
            )
            combined_gff = converted_gff_dir / f"{sample_id}.gff"
            convert(
                str(gff_for_panaroo),
                str(combined_gff),
                str(assembly_for_panaroo),
                is_ignore_overlapping=True,
            )
            f.write(f"{combined_gff}\n")
            # Delete per-sample unzipped intermediates (only if we created them).
            # Note: Path.unlink() deletes files on disk (like `rm`).
            if gff_abs.suffix == ".gz":
                try:
                    gff_for_panaroo.unlink()
                except FileNotFoundError:
                    pass
            if assembly_abs.suffix == ".gz":
                try:
                    assembly_for_panaroo.unlink()
                except FileNotFoundError:
                    pass

    # We only need the combined GFF+FASTA files for Panaroo input; remove
    # per-sample unzipped intermediates to save space once conversion is done.
    try:
        shutil.rmtree(gff_unzipped_dir)
    except Exception as e:
        print(f"Warning: failed to remove {gff_unzipped_dir}: {e}")
    try:
        shutil.rmtree(assembly_unzipped_dir)
    except Exception as e:
        print(f"Warning: failed to remove {assembly_unzipped_dir}: {e}")

    print(f"Wrote {n_written} lines to {input_path}")
    print(f"Run subdir (for panaroo -o): {run_subdir}")
    return input_path, run_subdir


def main() -> None:
    """CLI entry point: parse args and run."""
    parser = argparse.ArgumentParser(
        description="Build Panaroo input file from metadata for a clonal group."
    )
    parser.add_argument(
        "--clonal-group",
        type=str,
        default="CG11",
        help="Clonal group to filter by (default: CG11)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=-1,
        help="Max number of samples to include; -1 = all (default: -1)",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help=f"Base output directory; run subdir created under it (default: {DEFAULT_OUTDIR})",
    )
    parser.add_argument(
        "--metadata-file",
        type=Path,
        default=METADATA_FILE,
        help="Path to metadata TSV (default: project default)",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=BASE_DIR,
        help="Base directory to prepend to gff_file and assembly_file paths (default: project default)",
    )
    args = parser.parse_args()
    run(
        clonal_group=args.clonal_group,
        n=args.n,
        outdir=args.outdir,
        metadata_file=args.metadata_file,
        base_dir=args.base_dir,
    )


if __name__ == "__main__":
    main()
