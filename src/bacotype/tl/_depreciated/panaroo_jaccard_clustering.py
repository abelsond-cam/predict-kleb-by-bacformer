#!/usr/bin/env python3
r"""
KNN + Leiden clustering of Panaroo structural-variant and gene presence/absence tables.

Reads both the SV and GPA Rtab files for a clonal group, filters rare features,
builds Jaccard-based KNN graphs, runs Leiden community detection at multiple
resolutions, merges sub-threshold clusters by majority vote, computes cluster
quality metrics, and saves a MuData object containing both modalities.

Outputs (written to {strain_dir}/analysis/):

  clustering_log_{strain}.txt             combined timed log

  {SV,GPA}_clustering/
    {prefix}_burden_pre_filter_{strain}.png
    {prefix}_freq_dist_pre_filter_{strain}.png
    {prefix}_core_shell_cloud_pre_filter_{strain}.png
    {prefix}_burden_post_filter_{strain}.png
    {prefix}_freq_dist_post_filter_{strain}.png
    {prefix}_core_shell_cloud_post_filter_{strain}.png
    umap_leiden_r{res}_{strain}.png
    umap_klocus_{strain}.png              (if K_locus in metadata)
    {prefix}_clustering_summary_{strain}.tsv / .csv

  GPA_by_SV_clusters/
    rank_genes_wilcoxon_heatmap_sv_r{res}_{strain}.png
    rank_genes_wilcoxon_dotplot_sv_r{res}_{strain}.png
    rank_genes_wilcoxon_matrixplot_sv_r{res}_{strain}.png
    rank_genes_groups_sv_r{res}_{strain}.tsv

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

from bacotype.tl.panaroo_gpa_by_sv_clusters import (
    run_gpa_by_sv_cross_analysis,
    set_log_path_root as set_gpa_by_sv_log_path_root,
)
from bacotype.tl.gpa_distances_cluster_metrics import (
    log_medoid_report,
    log_resolution_one_comparison,
    medoid_metrics_from_dist_sq,
)
from bacotype.tl.gpa_matrix_utils import (
    feature_frequency_distribution,
    features_per_sample,
    filter_by_prevalence,
    per_sample_counts_core_shell_cloud,
)

RESOLUTIONS = [1.0, 0.5, 0.3]
MIN_CLUSTER_SIZE = 50
QUALITY_SUBSAMPLE_THRESHOLD = 2000
SECTION_BAR = "=" * 80
_LOG_PATH_ROOT: str | None = None


def _set_log_path_root(strain_dir: str) -> None:
    global _LOG_PATH_ROOT
    _LOG_PATH_ROOT = os.path.dirname(strain_dir.rstrip("/"))


def _fmt_log_path(path: str) -> str:
    if _LOG_PATH_ROOT is None:
        return path
    try:
        rel = os.path.relpath(path, _LOG_PATH_ROOT)
    except ValueError:
        return path
    return rel if not rel.startswith("..") else path


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
        suffix = (
            f"  (total {total:.1f}s, since last {step:.1f}s)"
            if report_times
            else ""
        )
        msg = f"{label}{suffix}"
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
    log(f"metadata: loaded {_fmt_log_path(meta_path)}  ({len(meta_df)} rows)")
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


def _default_filter_cutoff(n_samples: int) -> int:
    """Default prevalence filter cutoff.

    maximum(5, 0.01*n_samples) using floor for the 0.01*n_samples term.
    """
    return max(5, int(0.01 * n_samples))


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

def _compute_quality_metrics(
    adata: ad.AnnData,
    key: str,
    log,
    feature_label: str = "structural variant",
    max_subsample: int = QUALITY_SUBSAMPLE_THRESHOLD,
    min_per_cluster: int = MIN_CLUSTER_SIZE,
) -> dict:
    """Subsampling + Jaccard pdist for medoid-based cluster quality metrics."""
    n = adata.n_obs
    labels = adata.obs[key].astype(str)
    rng = np.random.default_rng(42)
    _ = feature_label

    if n <= max_subsample:
        X = adata.X.toarray().astype(np.uint8, copy=False)
        sub_labels = labels
    else:
        sub_idx = _stratified_subsample(labels, max_subsample, min_per_cluster, rng)
        X = adata.X[sub_idx].toarray().astype(np.uint8, copy=False)
        sub_labels = labels.iloc[sub_idx]

    condensed = pdist(X, metric="jaccard")
    dist_sq = squareform(condensed)

    mean_feats = float(adata.X.toarray().astype(np.uint8).sum(axis=1).mean())
    return medoid_metrics_from_dist_sq(dist_sq, sub_labels, mean_feats)


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
    log(f"plot: saved {_fmt_log_path(out_path)}")


def _plot_umap_refseq_highlight(
    adata: ad.AnnData,
    refseq_col: str,
    out_path: str,
    title: str,
    log,
) -> None:
    """Plot UMAP with RefSeq genomes highlighted in bright red."""
    if "X_umap" not in adata.obsm:
        raise ValueError("UMAP missing: adata.obsm['X_umap'] not found.")
    if refseq_col not in adata.obs.columns:
        log(f"plot: skipping RefSeq UMAP ({refseq_col} not in adata.obs)")
        return

    umap = adata.obsm["X_umap"]
    ref_mask = adata.obs[refseq_col].to_numpy(dtype=bool, copy=False)
    n_ref = int(ref_mask.sum())
    n_total = int(len(ref_mask))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(title)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")

    non_ref = ~ref_mask
    if non_ref.any():
        ax.scatter(
            umap[non_ref, 0],
            umap[non_ref, 1],
            s=6,
            c="lightgray",
            alpha=0.65,
            linewidths=0,
            label=f"non-RefSeq (n={int(non_ref.sum())})",
        )
    if ref_mask.any():
        ax.scatter(
            umap[ref_mask, 0],
            umap[ref_mask, 1],
            s=48,
            c="#ff0000",
            alpha=0.98,
            linewidths=0.5,
            edgecolors="black",
            label=f"RefSeq (n={n_ref})",
            zorder=3,
        )
    else:
        ax.text(
            0.02,
            0.98,
            "RefSeq (n=0)",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="red",
        )

    if n_ref > 0:
        ax.legend(loc="best", frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(
        f"plot: saved {_fmt_log_path(out_path)} "
        f"(RefSeq highlighted: {n_ref}/{n_total})"
    )


def _save_fig(fig: plt.Figure, path: str, log) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"plot: saved {_fmt_log_path(path)}")


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
    merge_small_clusters: int | None = None,
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
    merge_min_size = (
        int(merge_small_clusters)
        if merge_small_clusters is not None
        else max(10, int(0.01 * n_samples))
    )
    log(f"merge: n_samples={n_samples} merge_small_clusters_min_size={merge_min_size}")

    try:
        sc.pp.neighbors(adata, n_neighbors=k, metric="jaccard", use_rep="X")
    except Exception as exc:  # noqa: BLE001
        # Some runs fail inside pynndescent/numba for sparse Jaccard even when
        # the metric is nominally supported. Fall back to dense + sklearn.
        log(
            "knn: sparse Jaccard neighbor build failed "
            f"({exc}); retrying with dense boolean matrix + sklearn"
        )
        adata.obsm["X_jaccard_dense"] = adata.X.toarray().astype(bool, copy=False)
        sc.pp.neighbors(
            adata,
            n_neighbors=k,
            metric="jaccard",
            use_rep="X_jaccard_dense",
            transformer="sklearn",
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
            adata, key, log, min_size=merge_min_size
        )
        merge_summary[r] = (n_small, n_reass, n_remain)

        _log_section(log, f"Merge summary {modality_tag.upper()} r={r}")
        if n_small == 0:
            log(
                f"sub-threshold clusters (<{merge_min_size}) before merge: 0 "
                "(merge step skipped)"
            )
        else:
            log(f"sub-threshold clusters (<{merge_min_size}) before merge: {n_small}")
            log(f"genomes reassigned by majority vote: {n_reass}")
            log(f"sub-threshold clusters (<{merge_min_size}) after merge: {n_remain}")
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

    # RefSeq-highlight UMAP
    _plot_umap_refseq_highlight(
        adata,
        refseq_col="is_refseq",
        out_path=os.path.join(out_dir, f"umap_{modality_tag}_refseq_{strain}.png"),
        title=f"UMAP — {modality_tag.upper()} RefSeq highlighted — {strain}",
        log=log,
    )

    # ── Quality assessment ────────────────────────────────────────────
    _log_section(log, f"{modality_tag.upper()} Quality assessment")
    n_obs = int(adata.n_obs)
    if n_obs <= QUALITY_SUBSAMPLE_THRESHOLD:
        log(f"Analyzing all {n_obs} samples for medoid metrics")
    else:
        log(
            f"Subsampled {QUALITY_SUBSAMPLE_THRESHOLD}/{n_obs} samples, "
            f"stratified by minimum cluster size {MIN_CLUSTER_SIZE}"
        )
    mean_feats_all = float(adata.X.toarray().astype(np.uint8).sum(axis=1).mean())
    log(f"Mean {fl}s per genome in analyzed set = {mean_feats_all:.1f}")
    resolution_summaries: list[dict] = []
    for r in RESOLUTIONS:
        key = f"{modality_tag}_leiden_r{r}"
        log("")
        metrics = _compute_quality_metrics(
            adata, key, log, feature_label=fl,
            max_subsample=QUALITY_SUBSAMPLE_THRESHOLD,
            min_per_cluster=MIN_CLUSTER_SIZE,
        )
        log_medoid_report(
            log,
            f"{modality_tag.upper()} medoid metrics — Leiden r={r}",
            fl,
            metrics,
        )

        n_small, n_reass, n_remain = merge_summary[r]
        resolution_summaries.append({
            "strain": strain,
            "modality": modality_tag,
            "resolution": r,
            "n_samples": int(adata.n_obs),
            "k_neighbors": int(k),
            "min_cluster_size": int(merge_min_size),
            "subclusters_before_merge": int(n_small),
            "genomes_reassigned": int(n_reass),
            "subclusters_after_merge": int(n_remain),
            "global_medoid_jaccard_mean": float(metrics["global_medoid_jaccard_mean"]),
            "own_cluster_medoid_jaccard_mean": float(metrics["own_cluster_medoid_jaccard_mean"]),
            "gain_jaccard_b_minus_c": float(metrics["gain_jaccard_b_minus_c"]),
            "gain_similarity": float(metrics["gain_similarity"]),
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
        log(f"save: summary table -> {_fmt_log_path(p)}")

    return adata, merge_summary, resolution_summaries


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def _summary_row_at_resolution(summaries: list[dict], r: float) -> dict | None:
    for row in summaries:
        if abs(float(row["resolution"]) - float(r)) < 1e-9:
            return row
    return None


def main() -> int:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--strain",
        default=None,
        help=(
            "Strain / clonal group label (descriptor for output naming). "
            "If --strain-panaroo-dir is provided and --strain is omitted, "
            "the label defaults to the directory basename."
        ),
    )
    p.add_argument(
        "--project-k-dir",
        default="/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/",
        help="Base project directory",
    )
    p.add_argument(
        "--strain-panaroo-dir",
        "-strain-panaroo-dir",
        dest="strain_panaroo_dir",
        default=None,
        help=(
            "Override Panaroo input directory (used as-is). "
            "Output is written to <this-dir>/analysis. "
            "If --strain is omitted, output labels default to the directory basename."
        ),
    )
    p.add_argument("--sv-rtab", default=None, help="Override path to SV .Rtab")
    p.add_argument("--gpa-rtab", default=None, help="Override path to GPA .Rtab")
    p.add_argument(
        "--sv-filter-cutoff",
        type=int,
        default=None,
        help=(
            "Min genomes for SV prevalence filter. "
            "If omitted, uses maximum(5, 0.01*n_samples) (floor)."
        ),
    )
    p.add_argument(
        "--gpa-filter-cutoff",
        type=int,
        default=None,
        help=(
            "Min genomes for GPA prevalence filter. "
            "If omitted, uses maximum(5, 0.01*n_samples) (floor)."
        ),
    )
    p.add_argument("--metadata", default=None, help="Path to metadata TSV")
    p.add_argument(
        "--report-times", type=_str2bool, default=False, metavar="BOOL",
        help="Include timing in report/log lines (default: False)",
    )
    p.add_argument(
        "--gpa-by-sv-leiden-resolution", type=float, default=1.0,
        help="SV Leiden resolution to use for GPA cross-analysis (default: 1.0)",
    )
    p.add_argument(
        "--merge-small-clusters",
        type=int,
        default=None,
        help=(
            "Recombine clusters smaller than this size by kNN majority vote. "
            "If omitted, uses max(10, 0.01*n_samples) (floor)."
        ),
    )
    args = p.parse_args()

    _stderr_line_buffered()
    t0 = time.perf_counter()
    sc.settings.verbosity = 0

    if args.strain_panaroo_dir is not None:
        strain_dir = args.strain_panaroo_dir
        inferred_label = os.path.basename(os.path.normpath(strain_dir))
        strain = args.strain if args.strain is not None else inferred_label
    else:
        strain = args.strain if args.strain is not None else "CG14"
        strain_dir = (
            f"{args.project_k_dir.rstrip('/')}/processed/panaroo_run/{strain}_all/"
        )
    _set_log_path_root(strain_dir)
    set_gpa_by_sv_log_path_root(strain_dir)

    # Log file sits at the top analysis level
    analysis_dir = os.path.join(strain_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    log_path = os.path.join(analysis_dir, f"clustering_log_{strain}.txt")
    log_fh = open(log_path, "w")
    log = _make_progress_logger(t0, log_fh, report_times=args.report_times)

    log(
        f"start  pid={os.getpid()}  strain={strain}  "
        f"sv_filter_cutoff_arg={args.sv_filter_cutoff if args.sv_filter_cutoff is not None else 'computed'}  "
        f"gpa_filter_cutoff_arg={args.gpa_filter_cutoff if args.gpa_filter_cutoff is not None else 'computed'}  "
        f"merge_small_clusters_arg={args.merge_small_clusters if args.merge_small_clusters is not None else 'computed'}"
    )
    log(f"path context: analysis root = {analysis_dir}")

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
    log(f"load SV: reading {_fmt_log_path(sv_rtab)}")
    sv_df = pd.read_csv(sv_rtab, sep="\t", index_col=0, dtype=str, low_memory=False)
    sv_df.columns = sv_df.columns.astype(str)
    sv_df = (sv_df == "1").astype(np.uint8)
    log(f"load SV: {sv_df.shape[0]} variants x {sv_df.shape[1]} samples")
    n_samples_sv = int(sv_df.shape[1])
    sv_filter_cutoff = (
        args.sv_filter_cutoff
        if args.sv_filter_cutoff is not None
        else _default_filter_cutoff(n_samples_sv)
    )
    log(
        f"filter cutoff (SV): n_samples={n_samples_sv} -> {sv_filter_cutoff} "
        f"(min genomes)"
    )

    sv_out_dir = os.path.join(analysis_dir, "SV_clustering")
    adata_sv, sv_merge, sv_summaries = _run_modality_pipeline(
        sv_df, meta_df, strain, sv_out_dir, log,
        modality_tag="sv",
        feature_label="structural variant",
        file_prefix="sv",
        filter_cutoff=sv_filter_cutoff,
        merge_small_clusters=args.merge_small_clusters,
    )
    del sv_df

    # ==================================================================
    # GENE PRESENCE ABSENCE ANALYSIS
    # ==================================================================
    _log_section(log, "GENE PRESENCE ABSENCE ANALYSIS")

    gpa_rtab = args.gpa_rtab or os.path.abspath(
        os.path.join(strain_dir, "gene_presence_absence.Rtab")
    )
    log(f"load GPA: reading {_fmt_log_path(gpa_rtab)}")
    gpa_df = pd.read_csv(gpa_rtab, sep="\t", index_col=0, dtype=str, low_memory=False)
    gpa_df.columns = gpa_df.columns.astype(str)
    gpa_df = (gpa_df == "1").astype(np.uint8)
    log(f"load GPA: {gpa_df.shape[0]} genes x {gpa_df.shape[1]} samples")
    n_samples_gpa = int(gpa_df.shape[1])
    gpa_filter_cutoff = (
        args.gpa_filter_cutoff
        if args.gpa_filter_cutoff is not None
        else _default_filter_cutoff(n_samples_gpa)
    )
    log(
        f"filter cutoff (GPA): n_samples={n_samples_gpa} -> {gpa_filter_cutoff} "
        f"(min genomes)"
    )

    gpa_out_dir = os.path.join(analysis_dir, "GPA_clustering")
    adata_gpa, gpa_merge, gpa_summaries = _run_modality_pipeline(
        gpa_df, meta_df, strain, gpa_out_dir, log,
        modality_tag="gpa",
        feature_label="gene",
        file_prefix="gpa",
        filter_cutoff=gpa_filter_cutoff,
        merge_small_clusters=args.merge_small_clusters,
    )
    del gpa_df

    # ==================================================================
    # GPA BY SV CLUSTERS (cross-modality analysis)
    # ==================================================================
    _log_section(log, "GPA BY SV CLUSTERS (cross-analysis)")
    gpa_by_sv_dir = os.path.join(analysis_dir, "GPA_by_SV_clusters")
    gpa_sv_res = args.gpa_by_sv_leiden_resolution
    log(f"cross-analysis: GPA vs SV medoids + markers @ sv_leiden_r{gpa_sv_res}")
    metrics_gpa_vs_sv, adata_gpa = run_gpa_by_sv_cross_analysis(
        adata_sv=adata_sv,
        adata_gpa=adata_gpa,
        strain=strain,
        out_dir=gpa_by_sv_dir,
        resolution=gpa_sv_res,
        log=log,
    )
    if not adata_gpa.obs_names.equals(adata_sv.obs_names):
        adata_sv = adata_sv[adata_gpa.obs_names].copy()

    log_resolution_one_comparison(
        log,
        metrics_sv=_summary_row_at_resolution(sv_summaries, 1.0),
        metrics_gpa_vs_sv=metrics_gpa_vs_sv,
        metrics_gpa=_summary_row_at_resolution(gpa_summaries, 1.0),
        resolution_sv_gpa=1.0,
        resolution_gpa_sv_cross=gpa_sv_res,
        feature_label_sv="structural variant",
        feature_label_gpa="gene",
    )

    # ==================================================================
    # SAVE MUDATA
    # ==================================================================
    _log_section(log, "SAVE COMBINED MUDATA")

    mdata = md.MuData({"sv": adata_sv, "gpa": adata_gpa})
    mdata.update()

    mudata_path = os.path.join(analysis_dir, f"panaroo_mudata_{strain}.h5mu")
    log(f"save: writing MuData -> {_fmt_log_path(mudata_path)}")
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
