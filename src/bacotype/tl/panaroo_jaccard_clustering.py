#!/usr/bin/env python3
r"""
KNN + Leiden clustering of Panaroo structural-variant presence/absence.

Reads a tab-separated Rtab (rows = variants, cols = samples, 0/1), filters
rare variants, builds a Jaccard-based KNN graph, runs Leiden community
detection at multiple resolutions, merges sub-threshold clusters by
majority vote, and computes cluster quality metrics.

Outputs (written to {strain_dir}/analysis/SV_clustering/):

  clustering_log_{strain}.txt          full timed log
  sv_burden_pre_filter_{strain}.png    SV burden histogram (raw)
  sv_freq_dist_pre_filter_{strain}.png frequency distribution (raw)
  sv_core_shell_cloud_pre_filter_{strain}.png
  sv_burden_post_filter_{strain}.png   SV burden histogram (filtered)
  sv_freq_dist_post_filter_{strain}.png
  sv_core_shell_cloud_post_filter_{strain}.png
  umap_leiden_r1.0_{strain}.png
  umap_leiden_r0.5_{strain}.png
  umap_leiden_r0.3_{strain}.png
  umap_klocus_{strain}.png             (if K_locus in metadata)
  panaroo_anndata_{strain}.h5ad        AnnData with all cluster labels

Example:

  uv run python src/bacotype/tl/panaroo_jaccard_clustering.py --strain CG14
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist, squareform

from bacotype.tl.panaroo_structural_variants import (
    filter_structural_variants_by_prevalence,
    per_sample_counts_core_shell_cloud_structural_variants,
    structural_variant_frequency_distribution,
    structural_variants_per_sample,
)

RESOLUTIONS = [1.0, 0.5, 0.3]
MIN_CLUSTER_SIZE = 50
QUALITY_SUBSAMPLE_THRESHOLD = 2000
SECTION_BAR = "*" * 80


def _stderr_line_buffered() -> None:
    """Reduce 'silent hang' when stdout is fully buffered (pipes, some login nodes)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except (AttributeError, OSError, ValueError):
            pass


def _str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    v = value.strip().lower()
    if v in {"1", "true", "t", "yes", "y"}:
        return True
    if v in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _make_progress_logger(start: float, log_fh=None, report_times: bool = False):
    """Return a callable that prints labelled stages with elapsed / delta times.

    Writes to both stdout and *log_fh* (if provided).
    """
    last = [start]

    def log(label: str) -> None:
        now = time.perf_counter()
        total = now - start
        step = now - last[0]
        last[0] = now
        if report_times:
            msg = (
                f"[sv_clustering] {label}  "
                f"(total {total:.1f}s, since last {step:.1f}s)"
            )
        else:
            msg = f"[sv_clustering] {label}"
        print(msg, flush=True)
        if log_fh is not None:
            log_fh.write(msg + "\n")
            log_fh.flush()

    return log


def _log_section(log, title: str) -> None:
    log(SECTION_BAR)
    log(title)
    log(SECTION_BAR)


def _default_metadata_path(project_k_dir: str) -> str:
    return f"{project_k_dir.rstrip('/')}/final/metadata_final_curated_slimmed.tsv"


def _load_metadata(
    project_k_dir: str, metadata_arg: str | None, log
) -> tuple[str, pd.DataFrame]:
    """Load sample metadata indexed by ``Sample``."""
    if metadata_arg is None:
        meta_path = _default_metadata_path(project_k_dir)
    else:
        meta_path = metadata_arg

    meta_df = pd.read_csv(meta_path, sep="\t", low_memory=False)
    if "Sample" not in meta_df.columns:
        raise ValueError(
            f"Metadata file missing required column 'Sample': {meta_path}"
        )
    meta_df = (
        meta_df.drop_duplicates(subset=["Sample"], keep="first")
        .set_index("Sample")
    )
    meta_df.index = meta_df.index.astype(str)
    log(f"metadata: loaded {meta_path}  ({len(meta_df)} rows)")
    return meta_path, meta_df


def _build_adata_from_rtab(
    struct_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    sample_ids: np.ndarray,
    variant_ids: np.ndarray,
    log,
) -> ad.AnnData:
    """Create AnnData (obs=samples, var=variants) with metadata in ``obs``."""
    log("anndata: building sample x variant matrix (CSR)")
    X_u8 = struct_df.T.to_numpy(dtype=np.uint8, copy=False)
    X_sparse = csr_matrix(X_u8)

    sid = pd.Index(sample_ids.astype(str))
    missing_in_meta = sid.difference(meta_df.index)
    matched = len(sid) - len(missing_in_meta)
    log(
        f"metadata: matched {matched}/{len(sid)} samples "
        f"(metadata rows={len(meta_df)}, missing={len(missing_in_meta)})"
    )
    if len(missing_in_meta):
        print(
            f"    e.g. missing: {list(missing_in_meta)[:10]}", flush=True
        )

    obs = meta_df.reindex(sample_ids.astype(str))
    adata = ad.AnnData(
        X=X_sparse,
        obs=obs,
        var=pd.DataFrame(index=variant_ids.astype(str)),
    )
    adata.obs.index.name = "Sample"
    log(f"anndata: shape {adata.shape[0]} samples x {adata.shape[1]} variants")
    return adata


def _compute_k(n_samples: int) -> int:
    """Scale k for KNN from 25 (n<=1000) to 50 (n>=2000), linear between."""
    if n_samples <= 1000:
        return 25
    if n_samples >= 2000:
        return 50
    return 25 + round(25 * (n_samples - 1000) / 1000)


def _merge_small_clusters(
    adata: ad.AnnData,
    key: str,
    log,
    min_size: int = MIN_CLUSTER_SIZE,
) -> tuple[int, int, int]:
    """Reassign genomes in clusters smaller than *min_size* by kNN majority vote.

    For each genome in a sub-threshold cluster, the Leiden labels of its
    k nearest neighbours are examined and the modal label among neighbours
    belonging to clusters of >= *min_size* genomes is assigned.
    """
    labels = adata.obs[key].copy()
    counts = labels.value_counts()
    small_clusters = set(counts[counts < min_size].index)

    if not small_clusters:
        return 0, 0, 0

    large_clusters = set(counts.index) - small_clusters
    n_reassigned = 0

    conn = adata.obsp["connectivities"]

    for idx in range(adata.n_obs):
        if labels.iloc[idx] not in small_clusters:
            continue
        row = conn[idx]
        neighbor_indices = row.indices
        neighbor_labels = labels.iloc[neighbor_indices]
        valid = neighbor_labels[neighbor_labels.isin(large_clusters)]
        if len(valid) == 0:
            continue
        modal_label = valid.value_counts().idxmax()
        labels.iloc[idx] = modal_label
        n_reassigned += 1

    adata.obs[key] = labels
    new_counts = labels.value_counts()
    remaining_small = (new_counts < min_size).sum()
    return len(small_clusters), n_reassigned, int(remaining_small)


def _log_cluster_sizes(adata: ad.AnnData, key: str, log) -> None:
    """Log cluster count and size distribution."""
    counts = adata.obs[key].value_counts().sort_index()
    n_clusters = len(counts)
    log(
        f"clusters ({key}): {n_clusters} clusters, "
        f"sizes min={counts.min()} median={int(counts.median())} "
        f"max={counts.max()}"
    )


def _stratified_subsample(
    labels: pd.Series,
    max_n: int,
    min_per_cluster: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return indices for stratified subsampling, guaranteeing representation.

    Takes min(cluster_size, *min_per_cluster*) from every cluster, then fills
    remaining budget proportionally up to *max_n*.
    """
    cluster_ids = labels.unique()
    indices_by_cluster = {
        c: np.where(labels.values == c)[0] for c in cluster_ids
    }

    guaranteed: list[np.ndarray] = []
    leftover: list[tuple[str, np.ndarray]] = []

    for c, idxs in indices_by_cluster.items():
        take = min(len(idxs), min_per_cluster)
        chosen = rng.choice(idxs, size=take, replace=False)
        guaranteed.append(chosen)
        remaining = np.setdiff1d(idxs, chosen)
        if len(remaining):
            leftover.append((c, remaining))

    selected = np.concatenate(guaranteed)
    budget = max_n - len(selected)

    if budget > 0 and leftover:
        pool = np.concatenate([rem for _, rem in leftover])
        extra = min(budget, len(pool))
        selected = np.concatenate(
            [selected, rng.choice(pool, size=extra, replace=False)]
        )

    return np.sort(selected)


def _compute_quality_metrics(
    adata: ad.AnnData,
    key: str,
    log,
    max_subsample: int = QUALITY_SUBSAMPLE_THRESHOLD,
    min_per_cluster: int = MIN_CLUSTER_SIZE,
) -> dict[str, float]:
    """Compute and log Jaccard-based cluster quality metrics."""
    n = adata.n_obs
    labels = adata.obs[key]
    rng = np.random.default_rng(42)

    if n <= max_subsample:
        X = adata.X.toarray().astype(np.uint8, copy=False)
        sub_labels = labels
        log(f"quality ({key}): computing full pairwise Jaccard on {n} samples")
    else:
        sub_idx = _stratified_subsample(labels, max_subsample, min_per_cluster, rng)
        X = adata.X[sub_idx].toarray().astype(np.uint8, copy=False)
        sub_labels = labels.iloc[sub_idx]
        log(
            f"quality ({key}): subsampled {len(sub_idx)}/{n} samples "
            f"(stratified, min {min_per_cluster}/cluster)"
        )

    condensed = pdist(X, metric="jaccard")
    dist_sq = squareform(condensed)
    log(f"quality ({key}): pdist done, {len(condensed)} pairs")

    mean_svs = float(adata.X.toarray().astype(np.uint8).sum(axis=1).mean())

    global_mean = float(condensed.mean())
    global_sd = float(condensed.std(ddof=1)) if len(condensed) > 1 else 0.0

    log(f"quality ({key}): mean SVs per genome = {mean_svs:.1f}")
    log(f"quality ({key}): global Jaccard  mean={global_mean:.4f}  sd={global_sd:.4f}")
    global_shared = _jaccard_to_shared(global_mean, mean_svs)
    _log_shared_svs(global_mean, mean_svs, "global", log, key)

    cluster_ids = sorted(sub_labels.unique())

    within_dists_all: list[float] = []
    between_dists_all: list[float] = []

    for cid in cluster_ids:
        mask = (sub_labels.values == cid)
        idxs = np.where(mask)[0]
        if len(idxs) < 2:
            continue
        within = dist_sq[np.ix_(idxs, idxs)]
        triu_idx = np.triu_indices(len(idxs), k=1)
        w_vals = within[triu_idx]
        within_dists_all.extend(w_vals)
        w_mean = float(w_vals.mean())
        w_sd = float(w_vals.std(ddof=1)) if len(w_vals) > 1 else 0.0
        log(
            f"quality ({key}): cluster {cid}  n={len(idxs)}  "
            f"within Jaccard mean={w_mean:.4f} sd={w_sd:.4f}"
        )

    for i, c1 in enumerate(cluster_ids):
        for c2 in cluster_ids[i + 1:]:
            mask1 = np.where(sub_labels.values == c1)[0]
            mask2 = np.where(sub_labels.values == c2)[0]
            cross = dist_sq[np.ix_(mask1, mask2)].ravel()
            between_dists_all.extend(cross)

    pooled_within_mean = np.nan
    pooled_within_sd = np.nan
    pooled_between_mean = np.nan
    pooled_between_sd = np.nan

    if within_dists_all:
        w_arr = np.asarray(within_dists_all)
        pooled_within_mean = float(w_arr.mean())
        pooled_within_sd = float(w_arr.std(ddof=1))
        log(
            f"quality ({key}): pooled within-cluster Jaccard  "
            f"mean={pooled_within_mean:.4f}  sd={pooled_within_sd:.4f}"
        )
        _log_shared_svs(pooled_within_mean, mean_svs, "pooled within", log, key)

    if between_dists_all:
        b_arr = np.asarray(between_dists_all)
        pooled_between_mean = float(b_arr.mean())
        pooled_between_sd = float(b_arr.std(ddof=1))
        log(
            f"quality ({key}): pooled between-cluster Jaccard  "
            f"mean={pooled_between_mean:.4f}  sd={pooled_between_sd:.4f}"
        )
        _log_shared_svs(pooled_between_mean, mean_svs, "pooled between", log, key)

    centroids = []
    for cid in cluster_ids:
        mask = (sub_labels.values == cid)
        centroids.append(X[mask].mean(axis=0))
    if len(centroids) >= 2:
        centroid_dists = pdist(np.vstack(centroids), metric="jaccard")
        log(
            f"quality ({key}): mean centroid Jaccard = "
            f"{float(centroid_dists.mean()):.4f}"
        )

    if within_dists_all:
        gain = global_mean - pooled_within_mean
        within_shared = _jaccard_to_shared(pooled_within_mean, mean_svs)
        gain_svs = within_shared - global_shared
        global_shared_pct = 100.0 * global_shared / mean_svs if mean_svs > 0 else 0.0
        within_shared_pct = 100.0 * within_shared / mean_svs if mean_svs > 0 else 0.0
        gain_pct_points = within_shared_pct - global_shared_pct
        log(
            f"quality ({key}): clustering gain  "
            f"dJ={gain:.4f}  (~{gain_svs:.0f} additional shared SVs vs global)"
        )
        log(
            f"quality ({key}): shared SV proportion "
            f"{global_shared_pct:.1f}% -> {within_shared_pct:.1f}% "
            f"(+{gain_pct_points:.1f} percentage points)"
        )

        return {
            "global_mean": global_mean,
            "global_sd": global_sd,
            "global_shared": global_shared,
            "global_shared_pct": global_shared_pct,
            "within_mean": pooled_within_mean,
            "within_sd": pooled_within_sd,
            "within_shared": within_shared,
            "within_shared_pct": within_shared_pct,
            "gain_dj": gain,
            "gain_svs": gain_svs,
            "gain_pct_points": gain_pct_points,
            "between_mean": pooled_between_mean,
            "between_sd": pooled_between_sd,
            "mean_svs": mean_svs,
        }

    return {
        "global_mean": global_mean,
        "global_sd": global_sd,
        "global_shared": global_shared,
        "global_shared_pct": 100.0 * global_shared / mean_svs if mean_svs > 0 else 0.0,
        "within_mean": np.nan,
        "within_sd": np.nan,
        "within_shared": np.nan,
        "within_shared_pct": np.nan,
        "gain_dj": np.nan,
        "gain_svs": np.nan,
        "gain_pct_points": np.nan,
        "between_mean": pooled_between_mean,
        "between_sd": pooled_between_sd,
        "mean_svs": mean_svs,
    }


def _jaccard_to_shared(d_j: float, mean_svs: float) -> float:
    """Estimate shared SV count from Jaccard distance and mean SVs per genome.

    For two genomes A, B with |A|~|B|~mean_svs:
      |A ∩ B| = |A ∪ B| * (1 - dJ)
      |A ∪ B| = |A| + |B| - |A ∩ B|
    Solving: shared = mean_svs * (1 - dJ) / (1 + (1 - dJ) / 2)  [approx with |A|=|B|]
    Equivalently: shared = 2 * mean_svs * (1 - dJ) / (2 - dJ)  [... wait]
    """
    if d_j >= 1.0:
        return 0.0
    sim = 1.0 - d_j
    return 2.0 * mean_svs * sim / (1.0 + sim)


def _log_shared_svs(
    d_j: float, mean_svs: float, label: str, log, key: str
) -> None:
    shared = _jaccard_to_shared(d_j, mean_svs)
    log(
        f"quality ({key}): {label} -> estimated shared SVs = "
        f"{shared:.0f} / {mean_svs:.0f} mean"
    )


def _plot_umap_scatter(
    adata: ad.AnnData,
    color: str | None,
    out_path: str,
    title: str,
    log,
) -> None:
    """Plot UMAP using matplotlib (works with Agg backend)."""
    if "X_umap" not in adata.obsm:
        raise ValueError("UMAP missing: adata.obsm['X_umap'] not found.")

    umap = adata.obsm["X_umap"]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(title)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")

    if color is None:
        ax.scatter(
            umap[:, 0], umap[:, 1], s=6, c="lightgray", alpha=0.8,
            linewidths=0,
        )
    else:
        if color not in adata.obs.columns:
            raise ValueError(f"UMAP color '{color}' missing from adata.obs")
        labels = adata.obs[color].astype("object").fillna("NA")
        cats = labels.astype("category")
        codes = cats.cat.codes.to_numpy()
        categories = list(cats.cat.categories)

        cmap = plt.get_cmap("tab20")
        colors = [cmap(i % cmap.N) for i in range(len(categories))]
        point_colors = [colors[i] for i in codes]
        ax.scatter(
            umap[:, 0], umap[:, 1], s=6, c=point_colors, alpha=0.85,
            linewidths=0,
        )

        if len(categories) <= 20:
            handles = [
                plt.Line2D(
                    [0], [0], marker="o", color="w",
                    markerfacecolor=colors[i], markersize=6,
                )
                for i in range(len(categories))
            ]
            ax.legend(
                handles, categories, loc="best", frameon=False, fontsize=8
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"plot: saved {out_path}")


def _save_fig(fig: plt.Figure, path: str, log) -> None:
    """Save a matplotlib figure and close it."""
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"plot: saved {path}")


def main() -> int:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--strain", default="CG14", help="Strain / clonal group label"
    )
    p.add_argument(
        "--project-k-dir",
        default="/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/",
        help="Base project directory",
    )
    p.add_argument(
        "--rtab", default=None,
        help="Override path to .Rtab (default: struct_presence_absence.Rtab)",
    )
    p.add_argument(
        "--matrix", choices=("struct", "gene"), default="struct",
        help="Which Rtab to use (ignored if --rtab given)",
    )
    p.add_argument(
        "--filter-cutoff", type=int, default=10,
        help="Min number of genomes a variant must appear in to be kept",
    )
    p.add_argument(
        "--metadata", default=None,
        help="Path to metadata TSV (default: {project-k-dir}/final/metadata_final_curated_slimmed.tsv)",
    )
    p.add_argument(
        "--report-times",
        type=_str2bool,
        default=False,
        metavar="BOOL",
        help="Include timing suffixes in report/log lines (default: False)",
    )
    args = p.parse_args()

    _stderr_line_buffered()
    t0 = time.perf_counter()

    strain = args.strain
    strain_dir = (
        f"{args.project_k_dir.rstrip('/')}/processed/panaroo_run/{strain}_all/"
    )

    # ── output directory ──────────────────────────────────────────────
    out_dir = os.path.join(strain_dir, "analysis", "SV_clustering")
    os.makedirs(out_dir, exist_ok=True)

    log_path = os.path.join(out_dir, f"clustering_log_{strain}.txt")
    log_fh = open(log_path, "w")
    log = _make_progress_logger(t0, log_fh, report_times=args.report_times)

    log(
        f"start  pid={os.getpid()}  strain={strain}  "
        f"filter_cutoff={args.filter_cutoff}"
    )
    sc.settings.verbosity = 0

    _log_section(log, "Step 1: Load Rtab and filtering analysis")
    if args.rtab:
        rtab_path = os.path.abspath(args.rtab)
    else:
        name = (
            "struct_presence_absence.Rtab"
            if args.matrix == "struct"
            else "gene_presence_absence.Rtab"
        )
        rtab_path = os.path.abspath(os.path.join(strain_dir, name))

    log(f"load: reading {rtab_path}")
    struct_df = pd.read_csv(
        rtab_path, sep="\t", index_col=0, dtype=str, low_memory=False
    )
    struct_df.columns = struct_df.columns.astype(str)
    struct_df = (struct_df == "1").astype(np.uint8)
    log(
        f"load: done  {struct_df.shape[0]} variants x "
        f"{struct_df.shape[1]} samples"
    )

    # ── Step 1b: Pre-filter analysis ──────────────────────────────────
    res = structural_variants_per_sample(struct_df, strain)
    _save_fig(
        res.fig,
        os.path.join(out_dir, f"sv_burden_pre_filter_{strain}.png"),
        log,
    )

    fig_freq = structural_variant_frequency_distribution(struct_df, strain)
    _save_fig(
        fig_freq,
        os.path.join(out_dir, f"sv_freq_dist_pre_filter_{strain}.png"),
        log,
    )

    fig_csc = per_sample_counts_core_shell_cloud_structural_variants(
        struct_df, strain
    )
    _save_fig(
        fig_csc,
        os.path.join(out_dir, f"sv_core_shell_cloud_pre_filter_{strain}.png"),
        log,
    )

    # ── Step 1c: Filter ───────────────────────────────────────────────
    struct_df_filt = filter_structural_variants_by_prevalence(
        struct_df, min_prevalence=args.filter_cutoff
    )
    log(
        f"filter: {struct_df.shape[0]} -> {struct_df_filt.shape[0]} variants "
        f"(min_prevalence={args.filter_cutoff})"
    )

    # ── Step 1d: Post-filter analysis ─────────────────────────────────
    res_filt = structural_variants_per_sample(struct_df_filt, strain)
    _save_fig(
        res_filt.fig,
        os.path.join(out_dir, f"sv_burden_post_filter_{strain}.png"),
        log,
    )

    fig_freq2 = structural_variant_frequency_distribution(
        struct_df_filt, strain
    )
    _save_fig(
        fig_freq2,
        os.path.join(out_dir, f"sv_freq_dist_post_filter_{strain}.png"),
        log,
    )

    fig_csc2 = per_sample_counts_core_shell_cloud_structural_variants(
        struct_df_filt, strain
    )
    _save_fig(
        fig_csc2,
        os.path.join(out_dir, f"sv_core_shell_cloud_post_filter_{strain}.png"),
        log,
    )

    del struct_df  # free memory

    _log_section(log, "Step 2: Build AnnData with metadata")
    variant_ids = struct_df_filt.index.to_numpy()
    sample_ids = struct_df_filt.columns.to_numpy()

    meta_path, meta_df = _load_metadata(
        args.project_k_dir, args.metadata, log
    )
    adata = _build_adata_from_rtab(
        struct_df_filt, meta_df, sample_ids, variant_ids, log
    )
    del struct_df_filt

    _log_section(log, "Step 3: Build KNN graph (Jaccard)")
    n_samples = adata.n_obs
    k = _compute_k(n_samples)
    log(f"knn: n_samples={n_samples}, computed k={k}")

    try:
        # Preferred path: keep matrix in sparse form.
        sc.pp.neighbors(
            adata, n_neighbors=k, metric="jaccard", use_rep="X",
        )
    except ValueError as exc:
        if "Metric 'jaccard' not valid for sparse input" not in str(exc):
            raise
        # sklearn does not support sparse Jaccard. Fall back to dense boolean
        # representation for kNN construction while preserving Jaccard metric.
        log("knn: sparse Jaccard unsupported; retrying with dense boolean matrix")
        adata.obsm["X_jaccard_dense"] = adata.X.toarray().astype(bool, copy=False)
        sc.pp.neighbors(
            adata,
            n_neighbors=k,
            metric="jaccard",
            use_rep="X_jaccard_dense",
            transformer="sklearn",
        )
    log("knn: neighbors graph ready")

    _log_section(log, "Step 4: Compute UMAP")
    sc.tl.umap(adata)
    log("umap: done")

    _log_section(log, "Step 5: Leiden clustering + sub-threshold merge")
    merge_summary: dict[float, tuple[int, int, int]] = {}
    for r in RESOLUTIONS:
        key = f"leiden_r{r}"
        log("")
        _log_section(log, f"Resolution r={r}")
        log(f"leiden: resolution={r}")
        sc.tl.leiden(adata, resolution=r, key_added=key)

        raw_counts = adata.obs[key].value_counts()
        log(
            f"leiden ({key}): {len(raw_counts)} raw clusters, "
            f"sizes min={raw_counts.min()} max={raw_counts.max()}"
        )

        n_small_before, n_reassigned, n_small_after = _merge_small_clusters(
            adata, key, log, min_size=MIN_CLUSTER_SIZE
        )
        merge_summary[r] = (n_small_before, n_reassigned, n_small_after)
        _log_section(log, f"Merge summary r={r}")
        log(
            f"sub-threshold clusters (<{MIN_CLUSTER_SIZE}) before merge: {n_small_before}"
        )
        log(f"genomes reassigned by majority vote: {n_reassigned}")
        log(f"sub-threshold clusters (<{MIN_CLUSTER_SIZE}) after merge: {n_small_after}")
        _log_cluster_sizes(adata, key, log)

        _plot_umap_scatter(
            adata,
            color=key,
            out_path=os.path.join(out_dir, f"umap_leiden_r{r}_{strain}.png"),
            title=f"UMAP — Leiden r={r} — {strain}",
            log=log,
        )

    # K_locus UMAP (if available)
    if "K_locus" in adata.obs.columns:
        _plot_umap_scatter(
            adata,
            color="K_locus",
            out_path=os.path.join(out_dir, f"umap_klocus_{strain}.png"),
            title=f"UMAP — K_locus — {strain}",
            log=log,
        )

    _log_section(log, "Step 6: Quality assessment by resolution")
    resolution_summaries: list[dict[str, float | int | str]] = []
    for r in RESOLUTIONS:
        key = f"leiden_r{r}"
        log("")
        _log_section(log, f"Quality block r={r}")
        log(f"quality: starting assessment for {key}")
        metrics = _compute_quality_metrics(
            adata, key, log,
            max_subsample=QUALITY_SUBSAMPLE_THRESHOLD,
            min_per_cluster=MIN_CLUSTER_SIZE,
        )
        _log_section(log, f"Conclusion r={r}")
        log(
            f"Jaccard improvement (global - within): {metrics['gain_dj']:.4f}"
            if np.isfinite(metrics["gain_dj"])
            else "Jaccard improvement (global - within): not available"
        )
        if np.isfinite(metrics["gain_svs"]):
            log(
                f"Estimated additional shared SVs: {metrics['gain_svs']:.0f} "
                f"({metrics['gain_pct_points']:.1f} percentage points of mean genome SVs)"
            )
            log(
                f"Shared SVs: {metrics['global_shared']:.0f} "
                f"({metrics['global_shared_pct']:.1f}%) -> "
                f"{metrics['within_shared']:.0f} ({metrics['within_shared_pct']:.1f}%)"
            )
        else:
            log("Estimated additional shared SVs: not available")

        n_small_before, n_reassigned, n_small_after = merge_summary[r]
        resolution_summaries.append(
            {
                "strain": strain,
                "resolution": r,
                "n_samples": int(adata.n_obs),
                "k_neighbors": int(k),
                "min_cluster_size": int(MIN_CLUSTER_SIZE),
                "subclusters_before_merge": int(n_small_before),
                "genomes_reassigned": int(n_reassigned),
                "subclusters_after_merge": int(n_small_after),
                "global_jaccard_mean": float(metrics["global_mean"]),
                "global_jaccard_sd": float(metrics["global_sd"]),
                "within_jaccard_mean": float(metrics["within_mean"]),
                "within_jaccard_sd": float(metrics["within_sd"]),
                "between_jaccard_mean": float(metrics["between_mean"]),
                "between_jaccard_sd": float(metrics["between_sd"]),
                "jaccard_gain_global_minus_within": float(metrics["gain_dj"]),
                "mean_svs_per_genome": float(metrics["mean_svs"]),
                "shared_svs_global_est": float(metrics["global_shared"]),
                "shared_svs_within_est": float(metrics["within_shared"]),
                "shared_svs_gain_est": float(metrics["gain_svs"]),
                "shared_svs_global_pct": float(metrics["global_shared_pct"]),
                "shared_svs_within_pct": float(metrics["within_shared_pct"]),
                "shared_svs_gain_pct_points": float(metrics["gain_pct_points"]),
            }
        )

    _log_section(log, "Step 7: Save outputs")
    anndata_path = os.path.join(out_dir, f"panaroo_anndata_{strain}.h5ad")
    log(f"save: writing AnnData -> {anndata_path}")
    adata.write_h5ad(anndata_path)
    fsize_mb = os.path.getsize(anndata_path) / (1024 * 1024)
    log(f"save: done ({fsize_mb:.1f} MB)")

    summary_tsv_path = os.path.join(out_dir, f"sv_clustering_summary_{strain}.tsv")
    summary_csv_path = os.path.join(out_dir, f"sv_clustering_summary_{strain}.csv")
    summary_df = pd.DataFrame(resolution_summaries)
    summary_df.to_csv(summary_tsv_path, sep="\t", index=False)
    summary_df.to_csv(summary_csv_path, index=False)
    log(f"save: summary table -> {summary_tsv_path}")
    log(f"save: summary table -> {summary_csv_path}")

    total_wall = time.perf_counter() - t0
    _log_section(log, "Run summary")
    for r in RESOLUTIONS:
        n_small_before, n_reassigned, n_small_after = merge_summary[r]
        log(
            f"r={r}: merged_subclusters={n_small_before}, "
            f"genomes_reassigned={n_reassigned}, remaining_subclusters={n_small_after}"
        )
    log(f"DONE  total_wall={total_wall:.1f}s  exit=0")
    log_fh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
