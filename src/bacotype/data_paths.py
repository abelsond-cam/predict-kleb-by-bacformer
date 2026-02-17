"""Data paths configuration for Bacotype project."""

from pathlib import Path


class DataPaths:
    """Central configuration for all data file and directory paths."""

    def __init__(self):
        """Initialize data paths with absolute paths to various data resources."""
        # Base directory for raw data
        self.warm: Path = Path("/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/")
        # Metadata file
        self.klebsiella_metadata_file: Path = Path(
            #"/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/final/metadata_final_curated_slimmed.tsv"
            self.warm / "final/metadata_final_curated_slimmed.tsv"
        )
        # Klebsiella assembly files mapping
        self.kpsc_assembly_files: Path = Path(
            #"/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/raw/all_kpsc_genome_file_mapping.txt"
            self.warm / "raw/kpsc_assembly_files.txt"
        )
        # Parsed protein sequences directory
        self.klebsiella_gbff_dir: Path = Path(
            #"/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/raw/klebsiella_gbff"
            self.warm / "raw/klebsiella_gbff"
        )
        # FASTQ files directory
        self.klebsiella_fastq_dir: Path = self.warm / "raw/fastq"
        # Parsed protein sequences directory
        self.klebsiella_protein_sequences_dir: Path = Path(
            #"/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/klebsiella_protein_sequences"
            self.warm / "processed/klebsiella_protein_sequences"
        )
        # ESM embeddings paths (to be populated)
        self.klebsiella_esm_embeddings_dir: Path = Path(
            #"/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/klebsiella_esm_embeddings"
            self.warm / "processed/klebsiella_esm_embeddings"
        )
        self.klebsiella_bacformer_embeddings_dir: Path = Path(
            #"/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/klebsiella_bacformer_embeddings"
            self.warm / "processed/klebsiella_bacformer_embeddings"
        )
        # AnnData objects directory
        self.klebsiella_anndata_dir: Path = Path(
            self.warm / "processed/klebsiella_anndata"
        )
        # ESM file mapping (to be populated)
        self.klebsiella_esm_file_mapping: Path | None = None  # To be set
        # Bacformer embeddings paths (to be populated)
        self.klebsiella_bacformer_file_mapping: Path | None = None  # To be set
        
        # General data directories
        self.raw: Path = self.warm / "raw"
        self.processed: Path = self.warm / "processed"
        self.final: Path = self.warm / "final"
        
        # Results visualization directory (in workspace for easy access)
        self.results_visualisations: Path = Path("/home/dca36/workspace/Bacotype/results/visualisations")

# Create singleton instance for import
data = DataPaths()
