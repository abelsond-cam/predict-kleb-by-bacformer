"""
Add poppunk_cluster column to metadata TSV from PopPUNK clusters CSV.

Joins on poppunk_clusters["Taxon"] = metadata["Sample"], adds Cluster as
poppunk_cluster, and overwrites the metadata file.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


POPPUNK_CLUSTERS = Path(
    "/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/final/combined_clusters_poppunk.csv"
)
METADATA = Path(
    "/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/final/metadata_final_curated_slimmed.tsv"
)


def main() -> None:
    """Load clusters and metadata, join on Taxon/Sample, add poppunk_cluster, save metadata."""
    clusters = pd.read_csv(POPPUNK_CLUSTERS, low_memory=False)
    meta = pd.read_csv(METADATA, sep="\t", low_memory=False)

    # Join on Taxon (clusters) = Sample (metadata); add Cluster as poppunk_cluster, drop Taxon
    meta = meta.merge(
        clusters[["Taxon", "Cluster"]], left_on="Sample", right_on="Taxon", how="left"
    )
    meta = meta.rename(columns={"Cluster": "poppunk_cluster"}).drop(columns=["Taxon"])

    meta.to_csv(METADATA, sep="\t", index=False)


if __name__ == "__main__":
    main()
