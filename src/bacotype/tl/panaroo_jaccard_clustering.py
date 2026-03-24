#!/usr/bin/env python3
r"""
KNN + Leiden clustering of Panaroo structural-variant and gene presence/absence tables.

Reads both the SV and GPA Rtab files for a clonal group, filters rare features,
builds Jaccard-based KNN graphs, runs Leiden community detection at multiple
resolutions, merges sub-threshold clusters by majority vote, computes cluster
quality metrics, and saves a MuData object containing both modalities.

Outputs (written to {strain_dir}/analysis/{SV,GPA}_clustering/):

  clustering_log_{strain}.txt             combined timed log
  {prefix}_burden_pre_filter_{strain}.png
  {prefix}_freq_dist_pre_filter_{strain}.png
  {prefix}_core_shell_cloud_pre_filter_{strain}.png
  {prefix}_burden_post_filter_{strain}.png
  {prefix}_freq_dist_post_filter_{strain}.png
  {prefix}_core_shell_cloud_post_filter_{strain}.png
  umap_leiden_r{res}_{strain}.png
  umap_klocus_{strain}.png                (if K_locus in metadata)
  {prefix}_clustering_summary_{strain}.tsv / .csv

  panaroo_mudata_{strain}.h5mu            MuData with both modalities

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
import mudata as md
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist, squareform

from bacotype.tl.panaroo_pangenome_features import (
    feature_frequency_distribution,
    features_per_sample,
    filter_by_prevalence,
    per_sample_counts_core_shell_cloud,
)

RESOLUTIONS = [1.0, 0.5, 0.3]
MIN_CLUSTER_SIZE = 50
QUALITY_SUBSAMPLE_THRESHOLD = 2000
SECTION_BAR = "=" * 80


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

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
                f"[clustering] {label}  "
                f"(total {total:.1f}s, since last {step:.1f}s)"
            )
        else:
            msg = f"[clustering] {label}"
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


def _build_adata(
    binary_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    sample_ids: np.ndarray,
    feature_ids: np.ndarray,
    log,
) -> ad.AnnData:
    """Create AnnData (obs=samples, var=features) with metadata in ``obs``."""
    log("anndata: building sample x feature matrix (CSR)")
    X_u8 = binary_df.T.to_numpy(dtype=np.uint8, copy=False)
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
        var=pd.DataFrame(index=feature_ids.astype(str)),
    )
    adata.obs.index.name = "Sample"
    log(f"anndata: shape {adata.shape[0]} samples x {adata.shape[1]} features")
    return adata


def _compute_k(n_samples: int) -> int:
    """Scale k for KNN from 25 (n<=1000) to 50 (n>=2000), linear between."""
    if n_samples <= 1000:
        return 25
    if n_samples >= 2000:
        return 50
    return 25 + round(25 * (n_samples - 1000) / 1000)


# ---------------------------------------------------------------------------
# Clustering helpers
# ---------------------------------------------------------------------------

def _merge_small_clusters(
    adata: ad.AnnData,
    key: str,
    log,
    min_size: int = MIN_CLUSTER_SIZE,
) -> tuple[int, int, int]:
    """Reassign genomes in clusters < *min_size* by kNN majority vote."""
    labels = adata.obs[key].astype(str).copy()
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
    counts = adata.obs[key].astype(str).value_counts().sort_index()
    log(
        f"clusters ({key}): {len(counts)} clusters, "
        f"sizes min={counts.min()} median={int(counts.median())} "
        f"max={counts.max()}"
    )


def _stratified_subsample(
    labels: pd.Series,
    max_n: int,
    min_per_cluster: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return indices for stratified subsampling, guaranteeing representation."""
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


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------

def _jaccard_to_shared(d_j: float, mean_features: float) -> float:
    """Estimate shared feature count from Jaccard distance.

    For two genomes A, B with |A| ~ |B| ~ mean_features:
      shared = 2 * mean * sim / (1 + sim)  where sim = 1 - dJ.
    """
    if d_j >= 1.0:
        return 0.0
    sim = 1.0 - d_j
    return 2.0 * mean_features * sim / (1.0 + sim)


def _log_shared(
    d_j: float, mean_features: float, label: str, log, key: str,
    feature_label: str,
) -> None:
    shared = _jaccard_to_shared(d_j, mean_features)
    log(
        f"quality ({key}): {label} -> estimated shared {feature_label}s = "
        f"{shared:.0f} / {mean_features:.0f} mean"
    )


def _compute_quality_metrics(
    adata: ad.AnnData,
    key: str,
    log,
    feature_label: str = "structural variant",
    max_subsample: int = QUALITY_SUBSAMPLE_THRESHOLD,
    min_per_cluster: int = MIN_CLUSTER_SIZE,
) -> dict[str, float]:
    """Compute and log Jaccard-based cluster quality metrics."""
    n = adata.n_obs
    labels = adata.obs[key].astype(str)
    rng = np.random.default_rng(42)
    fl = feature_label

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

    mean_feats = float(adata.X.toarray().astype(np.uint8).sum(axis=1).mean())

    global_mean = float(condensed.mean())
    global_sd = float(condensed.std(ddof=1)) if len(condensed) > 1 else 0.0

    log(f"quality ({key}): mean {fl}s per genome = {mean_feats:.1f}")
    log(f"quality ({key}): global Jaccard  mean={global_mean:.4f}  sd={global_sd:.4f}")
    global_shared = _jaccard_to_shared(global_mean, mean_feats)
    _log_shared(global_mean, mean_feats, "global", log, key, fl)

    cluster_ids = sorted(sub_labels.unique())
    within_dists_all: list[float] = []
    between_dists_all: list[float] = []

    for cid in cluster_ids:
        mask = sub_labels.values == cid
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
        _log_shared(pooled_within_mean, mean_feats, "pooled within", log, key, fl)

    if between_dists_all:
        b_arr = np.asarray(between_dists_all)
        pooled_between_mean = float(b_arr.mean())
        pooled_between_sd = float(b_arr.std(ddof=1))
        log(
            f"quality ({key}): pooled between-cluster Jaccard  "
            f"mean={pooled_between_mean:.4f}  sd={pooled_between_sd:.4f}"
        )
        _log_shared(pooled_between_mean, mean_feats, "pooled between", log, key, fl)

    centroids = []
    for cid in cluster_ids:
        mask = sub_labels.values == cid
        centroids.append(X[mask].mean(axis=0))
    if len(centroids) >= 2:
        centroid_dists = pdist(np.vstack(centroids), metric="jaccard")
        log(
            f"quality ({key}): mean centroid Jaccard = "
            f"{float(centroid_dists.mean()):.4f}"
        )
    # Medoid-distance based summaries (requested reporting)
    # a) per-cluster mean distance to cluster medoid
    # b) mean distance to global medoid
    # c) mean distance to own-cluster medoid
    sample_to_own_medoid = np.full(X.shape[0], np.nan, dtype=float)

    global_medoid_idx = int(np.argmin(dist_sq.mean(axis=1)))
    mean_to_global_medoid = float(dist_sq[:, global_medoid_idx].mean())
    log(
        f"quality ({key}): mean Jaccard to global medoid sample (b) = "
        f"{mean_to_global_medoid:.4f}"
    )

    for cid in cluster_ids:
        idxs = np.where(sub_labels.values == cid)[0]
        if len(idxs) < 2:
            continue
        sub_d = dist_sq[np.ix_(idxs, idxs)]
        medoid_local = int(np.argmin(sub_d.mean(axis=1)))
        medoid_global_idx = idxs[medoid_local]
        d_to_medoid = dist_sq[idxs, medoid_global_idx]
        m = float(d_to_medoid.mean())
        sample_to_own_medoid[idxs] = d_to_medoid
        log(
            f"quality ({key}): cluster {cid} mean Jaccard to cluster medoid (a) = {m:.4f}"
        )

    finite_mask = np.isfinite(sample_to_own_medoid)
    if finite_mask.any():
        mean_to_own_cluster_medoid = float(sample_to_own_medoid[finite_mask].mean())
        gain = mean_to_global_medoid - mean_to_own_cluster_medoid
        global_shared_medoid = _jaccard_to_shared(mean_to_global_medoid, mean_feats)
        own_shared_medoid = _jaccard_to_shared(mean_to_own_cluster_medoid, mean_feats)
        gain_feats = own_shared_medoid - global_shared_medoid
        global_pct = 100.0 * global_shared_medoid / mean_feats if mean_feats > 0 else 0.0
        own_pct = 100.0 * own_shared_medoid / mean_feats if mean_feats > 0 else 0.0
        gain_pct = own_pct - global_pct

        log(
            f"quality ({key}): mean Jaccard to own cluster medoid (c) = "
            f"{mean_to_own_cluster_medoid:.4f}"
        )
        log(
            f"quality ({key}): clustering gain (b - c) = {gain:.4f}  "
            f"(~{gain_feats:.0f} additional shared {fl}s)"
        )
        log(
            f"quality ({key}): shared {fl} proportion "
            f"{global_pct:.1f}% -> {own_pct:.1f}% "
            f"(+{gain_pct:.1f} percentage points)"
        )
        return {
            "pairwise_global_jaccard_mean": global_mean,
            "pairwise_global_jaccard_sd": global_sd,
            "pairwise_global_shared_features_est": global_shared,
            "pairwise_global_shared_features_pct": 100.0 * global_shared / mean_feats if mean_feats > 0 else 0.0,
            "pairwise_within_jaccard_mean": pooled_within_mean,
            "pairwise_within_jaccard_sd": pooled_within_sd,
            "pairwise_within_shared_features_est": _jaccard_to_shared(pooled_within_mean, mean_feats) if np.isfinite(pooled_within_mean) else np.nan,
            "pairwise_within_shared_features_pct": (100.0 * _jaccard_to_shared(pooled_within_mean, mean_feats) / mean_feats) if (np.isfinite(pooled_within_mean) and mean_feats > 0) else np.nan,
            "pairwise_between_jaccard_mean": pooled_between_mean,
            "pairwise_between_jaccard_sd": pooled_between_sd,
            "mean_features_per_genome": mean_feats,
            "global_medoid_jaccard_mean": mean_to_global_medoid,
            "own_cluster_medoid_jaccard_mean": mean_to_own_cluster_medoid,
            "gain_jaccard_b_minus_c": gain,
            "gain_shared_features_est": gain_feats,
            "gain_shared_features_pct_points": gain_pct,
            "global_medoid_shared_features_est": global_shared_medoid,
            "global_medoid_shared_features_pct": global_pct,
            "own_cluster_medoid_shared_features_est": own_shared_medoid,
            "own_cluster_medoid_shared_features_pct": own_pct,
        }

    return {
        "pairwise_global_jaccard_mean": global_mean,
        "pairwise_global_jaccard_sd": global_sd,
        "pairwise_global_shared_features_est": global_shared,
        "pairwise_global_shared_features_pct": 100.0 * global_shared / mean_feats if mean_feats > 0 else 0.0,
        "pairwise_within_jaccard_mean": np.nan,
        "pairwise_within_jaccard_sd": np.nan,
        "pairwise_within_shared_features_est": np.nan,
        "pairwise_within_shared_features_pct": np.nan,
        "pairwise_between_jaccard_mean": pooled_between_mean,
        "pairwise_between_jaccard_sd": pooled_between_sd,
        "mean_features_per_genome": mean_feats,
        "global_medoid_jaccard_mean": np.nan,
        "own_cluster_medoid_jaccard_mean": np.nan,
        "gain_jaccard_b_minus_c": np.nan,
        "gain_shared_features_est": np.nan,
        "gain_shared_features_pct_points": np.nan,
        "global_medoid_shared_features_est": np.nan,
        "global_medoid_shared_features_pct": np.nan,
        "own_cluster_medoid_shared_features_est": np.nan,
        "own_cluster_medoid_shared_features_pct": np.nan,
    }


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

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
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"plot: saved {path}")


# ---------------------------------------------------------------------------
# Single-modality pipeline (shared by SV and GPA)
# ---------------------------------------------------------------------------

def _run_modality_pipeline(
    binary_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    strain: str,
    out_dir: str,
    log,
    *,
    modality_tag: str,
    feature_label: str,
    file_prefix: str,
    filter_cutoff: int,
) -> tuple[ad.AnnData, dict[float, tuple[int, int, int]], list[dict]]:
    """Run the full filter -> AnnData -> KNN -> UMAP -> Leiden -> quality pipeline.

    Returns ``(adata, merge_summary, resolution_summaries)``.
    """
    os.makedirs(out_dir, exist_ok=True)
    fl = feature_label

    # ── Pre-filter analysis ───────────────────────────────────────────
    res = features_per_sample(binary_df, strain, feature_label=fl)
    _save_fig(
        res.fig,
        os.path.join(out_dir, f"{file_prefix}_burden_pre_filter_{strain}.png"),
        log,
    )
    fig_freq = feature_frequency_distribution(binary_df, strain, feature_label=fl)
    _save_fig(
        fig_freq,
        os.path.join(out_dir, f"{file_prefix}_freq_dist_pre_filter_{strain}.png"),
        log,
    )
    fig_csc = per_sample_counts_core_shell_cloud(binary_df, strain, feature_label=fl)
    _save_fig(
        fig_csc,
        os.path.join(out_dir, f"{file_prefix}_core_shell_cloud_pre_filter_{strain}.png"),
        log,
    )

    # ── Filter ────────────────────────────────────────────────────────
    df_filt = filter_by_prevalence(
        binary_df, min_prevalence=filter_cutoff, feature_label=fl
    )
    log(
        f"filter: {binary_df.shape[0]} -> {df_filt.shape[0]} {fl}s "
        f"(min_prevalence={filter_cutoff})"
    )

    # ── Post-filter analysis ──────────────────────────────────────────
    res_filt = features_per_sample(df_filt, strain, feature_label=fl)
    _save_fig(
        res_filt.fig,
        os.path.join(out_dir, f"{file_prefix}_burden_post_filter_{strain}.png"),
        log,
    )
    fig_freq2 = feature_frequency_distribution(df_filt, strain, feature_label=fl)
    _save_fig(
        fig_freq2,
        os.path.join(out_dir, f"{file_prefix}_freq_dist_post_filter_{strain}.png"),
        log,
    )
    fig_csc2 = per_sample_counts_core_shell_cloud(df_filt, strain, feature_label=fl)
    _save_fig(
        fig_csc2,
        os.path.join(out_dir, f"{file_prefix}_core_shell_cloud_post_filter_{strain}.png"),
        log,
    )

    del binary_df

    # ── Build AnnData ─────────────────────────────────────────────────
    feature_ids = df_filt.index.to_numpy()
    sample_ids = df_filt.columns.to_numpy()
    adata = _build_adata(df_filt, meta_df, sample_ids, feature_ids, log)
    del df_filt

    # ── KNN graph ─────────────────────────────────────────────────────
    n_samples = adata.n_obs
    k = _compute_k(n_samples)
    log(f"knn: n_samples={n_samples}, computed k={k}")

    try:
        sc.pp.neighbors(adata, n_neighbors=k, metric="jaccard", use_rep="X")
    except ValueError as exc:
        if "Metric 'jaccard' not valid for sparse input" not in str(exc):
            raise
        log("knn: sparse Jaccard unsupported; retrying with dense boolean matrix")
        adata.obsm["X_jaccard_dense"] = adata.X.toarray().astype(bool, copy=False)
        sc.pp.neighbors(
            adata, n_neighbors=k, metric="jaccard",
            use_rep="X_jaccard_dense", transformer="sklearn",
        )
    log("knn: neighbors graph ready")

    # ── UMAP ──────────────────────────────────────────────────────────
    sc.tl.umap(adata)
    log("umap: done")

    # ── Leiden at multiple resolutions + merge ────────────────────────
    merge_summary: dict[float, tuple[int, int, int]] = {}
    for r in RESOLUTIONS:
        key = f"{modality_tag}_leiden_r{r}"
        log("")
        _log_section(log, f"{modality_tag.upper()} Resolution r={r}")
        sc.tl.leiden(adata, resolution=r, key_added=key)

        raw_counts = adata.obs[key].value_counts()
        log(
            f"leiden ({key}): {len(raw_counts)} raw clusters, "
            f"sizes min={raw_counts.min()} max={raw_counts.max()}"
        )

        n_small, n_reass, n_remain = _merge_small_clusters(
            adata, key, log, min_size=MIN_CLUSTER_SIZE
        )
        merge_summary[r] = (n_small, n_reass, n_remain)

        _log_section(log, f"Merge summary {modality_tag.upper()} r={r}")
        if n_small == 0:
            log(
                f"sub-threshold clusters (<{MIN_CLUSTER_SIZE}) before merge: 0 "
                "(merge step skipped)"
            )
        else:
            log(f"sub-threshold clusters (<{MIN_CLUSTER_SIZE}) before merge: {n_small}")
            log(f"genomes reassigned by majority vote: {n_reass}")
            log(f"sub-threshold clusters (<{MIN_CLUSTER_SIZE}) after merge: {n_remain}")
        _log_cluster_sizes(adata, key, log)

        _plot_umap_scatter(
            adata, color=key,
            out_path=os.path.join(out_dir, f"umap_{modality_tag}_leiden_r{r}_{strain}.png"),
            title=f"UMAP — {modality_tag.upper()} Leiden r={r} — {strain}",
            log=log,
        )

    # K_locus UMAP
    if "K_locus" in adata.obs.columns:
        _plot_umap_scatter(
            adata, color="K_locus",
            out_path=os.path.join(out_dir, f"umap_{modality_tag}_klocus_{strain}.png"),
            title=f"UMAP — {modality_tag.upper()} K_locus — {strain}",
            log=log,
        )

    # ── Quality assessment ────────────────────────────────────────────
    _log_section(log, f"{modality_tag.upper()} Quality assessment")
    resolution_summaries: list[dict] = []
    for r in RESOLUTIONS:
        key = f"{modality_tag}_leiden_r{r}"
        log("")
        _log_section(log, f"{modality_tag.upper()} Quality block r={r}")
        metrics = _compute_quality_metrics(
            adata, key, log, feature_label=fl,
            max_subsample=QUALITY_SUBSAMPLE_THRESHOLD,
            min_per_cluster=MIN_CLUSTER_SIZE,
        )

        _log_section(log, f"{modality_tag.upper()} Conclusion r={r}")
        log(
            f"Mean Jaccard to global medoid (b): {metrics['global_medoid_jaccard_mean']:.4f}"
            if np.isfinite(metrics["global_medoid_jaccard_mean"])
            else "Mean Jaccard to global medoid (b): not available"
        )
        log(
            f"Mean Jaccard to own cluster medoid (c): {metrics['own_cluster_medoid_jaccard_mean']:.4f}"
            if np.isfinite(metrics["own_cluster_medoid_jaccard_mean"])
            else "Mean Jaccard to own cluster medoid (c): not available"
        )
        log(
            f"Jaccard improvement (global medoid - own-cluster medoid; b - c): {metrics['gain_jaccard_b_minus_c']:.4f}"
            if np.isfinite(metrics["gain_jaccard_b_minus_c"])
            else "Jaccard improvement (global medoid - own-cluster medoid; b - c): not available"
        )
        if np.isfinite(metrics["gain_shared_features_est"]):
            log(
                f"Estimated additional shared {fl}s: {metrics['gain_shared_features_est']:.0f} "
                f"({metrics['gain_shared_features_pct_points']:.1f} percentage points of mean genome {fl}s)"
            )
            log(
                f"Shared {fl}s: {metrics['global_medoid_shared_features_est']:.0f} "
                f"({metrics['global_medoid_shared_features_pct']:.1f}%) -> "
                f"{metrics['own_cluster_medoid_shared_features_est']:.0f} "
                f"({metrics['own_cluster_medoid_shared_features_pct']:.1f}%)"
            )
        else:
            log(f"Estimated additional shared {fl}s: not available")

        n_small, n_reass, n_remain = merge_summary[r]
        resolution_summaries.append({
            "strain": strain,
            "modality": modality_tag,
            "resolution": r,
            "n_samples": int(adata.n_obs),
            "k_neighbors": int(k),
            "min_cluster_size": int(MIN_CLUSTER_SIZE),
            "subclusters_before_merge": int(n_small),
            "genomes_reassigned": int(n_reass),
            "subclusters_after_merge": int(n_remain),
            "pairwise_global_jaccard_mean": float(metrics["pairwise_global_jaccard_mean"]),
            "pairwise_global_jaccard_sd": float(metrics["pairwise_global_jaccard_sd"]),
            "pairwise_within_jaccard_mean": float(metrics["pairwise_within_jaccard_mean"]),
            "pairwise_within_jaccard_sd": float(metrics["pairwise_within_jaccard_sd"]),
            "pairwise_between_jaccard_mean": float(metrics["pairwise_between_jaccard_mean"]),
            "pairwise_between_jaccard_sd": float(metrics["pairwise_between_jaccard_sd"]),
            "global_medoid_jaccard_mean": float(metrics["global_medoid_jaccard_mean"]),
            "own_cluster_medoid_jaccard_mean": float(metrics["own_cluster_medoid_jaccard_mean"]),
            "gain_jaccard_b_minus_c": float(metrics["gain_jaccard_b_minus_c"]),
            "mean_features_per_genome": float(metrics["mean_features_per_genome"]),
            "global_medoid_shared_features_est": float(metrics["global_medoid_shared_features_est"]),
            "own_cluster_medoid_shared_features_est": float(metrics["own_cluster_medoid_shared_features_est"]),
            "gain_shared_features_est": float(metrics["gain_shared_features_est"]),
            "global_medoid_shared_features_pct": float(metrics["global_medoid_shared_features_pct"]),
            "own_cluster_medoid_shared_features_pct": float(metrics["own_cluster_medoid_shared_features_pct"]),
            "gain_shared_features_pct_points": float(metrics["gain_shared_features_pct_points"]),
        })

    # Save summary table
    summary_df = pd.DataFrame(resolution_summaries)
    for ext, sep in [(".tsv", "\t"), (".csv", ",")]:
        p = os.path.join(out_dir, f"{file_prefix}_clustering_summary_{strain}{ext}")
        summary_df.to_csv(p, sep=sep, index=False)
        log(f"save: summary table -> {p}")

    return adata, merge_summary, resolution_summaries


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def main() -> int:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--strain", default="CG14", help="Strain / clonal group label")
    p.add_argument(
        "--project-k-dir",
        default="/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/",
        help="Base project directory",
    )
    p.add_argument("--sv-rtab", default=None, help="Override path to SV .Rtab")
    p.add_argument("--gpa-rtab", default=None, help="Override path to GPA .Rtab")
    p.add_argument(
        "--sv-filter-cutoff", type=int, default=10,
        help="Min genomes for SV prevalence filter",
    )
    p.add_argument(
        "--gpa-filter-cutoff", type=int, default=10,
        help="Min genomes for GPA prevalence filter",
    )
    p.add_argument("--metadata", default=None, help="Path to metadata TSV")
    p.add_argument(
        "--report-times", type=_str2bool, default=False, metavar="BOOL",
        help="Include timing in report/log lines (default: False)",
    )
    args = p.parse_args()

    _stderr_line_buffered()
    t0 = time.perf_counter()
    sc.settings.verbosity = 0

    strain = args.strain
    strain_dir = (
        f"{args.project_k_dir.rstrip('/')}/processed/panaroo_run/{strain}_all/"
    )

    # Log file sits at the top analysis level
    analysis_dir = os.path.join(strain_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    log_path = os.path.join(analysis_dir, f"clustering_log_{strain}.txt")
    log_fh = open(log_path, "w")
    log = _make_progress_logger(t0, log_fh, report_times=args.report_times)

    log(
        f"start  pid={os.getpid()}  strain={strain}  "
        f"sv_filter_cutoff={args.sv_filter_cutoff}  "
        f"gpa_filter_cutoff={args.gpa_filter_cutoff}"
    )

    # ── Load metadata (shared) ────────────────────────────────────────
    _log_section(log, "METADATA")
    _, meta_df = _load_metadata(args.project_k_dir, args.metadata, log)

    # ==================================================================
    # STRUCTURAL VARIANT ANALYSIS
    # ==================================================================
    _log_section(log, "STRUCTURAL VARIANT ANALYSIS")

    sv_rtab = args.sv_rtab or os.path.abspath(
        os.path.join(strain_dir, "struct_presence_absence.Rtab")
    )
    log(f"load SV: reading {sv_rtab}")
    sv_df = pd.read_csv(sv_rtab, sep="\t", index_col=0, dtype=str, low_memory=False)
    sv_df.columns = sv_df.columns.astype(str)
    sv_df = (sv_df == "1").astype(np.uint8)
    log(f"load SV: {sv_df.shape[0]} variants x {sv_df.shape[1]} samples")

    sv_out_dir = os.path.join(analysis_dir, "SV_clustering")
    adata_sv, sv_merge, sv_summaries = _run_modality_pipeline(
        sv_df, meta_df, strain, sv_out_dir, log,
        modality_tag="sv",
        feature_label="structural variant",
        file_prefix="sv",
        filter_cutoff=args.sv_filter_cutoff,
    )
    del sv_df

    # ==================================================================
    # GENE PRESENCE ABSENCE ANALYSIS
    # ==================================================================
    _log_section(log, "GENE PRESENCE ABSENCE ANALYSIS")

    gpa_rtab = args.gpa_rtab or os.path.abspath(
        os.path.join(strain_dir, "gene_presence_absence.Rtab")
    )
    log(f"load GPA: reading {gpa_rtab}")
    gpa_df = pd.read_csv(gpa_rtab, sep="\t", index_col=0, dtype=str, low_memory=False)
    gpa_df.columns = gpa_df.columns.astype(str)
    gpa_df = (gpa_df == "1").astype(np.uint8)
    log(f"load GPA: {gpa_df.shape[0]} genes x {gpa_df.shape[1]} samples")

    gpa_out_dir = os.path.join(analysis_dir, "GPA_clustering")
    adata_gpa, gpa_merge, gpa_summaries = _run_modality_pipeline(
        gpa_df, meta_df, strain, gpa_out_dir, log,
        modality_tag="gpa",
        feature_label="gene",
        file_prefix="gpa",
        filter_cutoff=args.gpa_filter_cutoff,
    )
    del gpa_df

    # ==================================================================
    # SAVE MUDATA
    # ==================================================================
    _log_section(log, "SAVE COMBINED MUDATA")

    mdata = md.MuData({"sv": adata_sv, "gpa": adata_gpa})
    mdata.update()

    mudata_path = os.path.join(analysis_dir, f"panaroo_mudata_{strain}.h5mu")
    log(f"save: writing MuData -> {mudata_path}")
    mdata.write(mudata_path)
    fsize_mb = os.path.getsize(mudata_path) / (1024 * 1024)
    log(f"save: done ({fsize_mb:.1f} MB)")

    total_wall = time.perf_counter() - t0
    _log_section(log, "FINAL SUMMARY")
    log(f"DONE  total_wall={total_wall:.1f}s  exit=0")
    log_fh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
