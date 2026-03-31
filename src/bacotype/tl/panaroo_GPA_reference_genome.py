#!/usr/bin/env python3
r"""
GPA-only Jaccard clustering for a Panaroo directory leaf.

Loads ``gene_presence_absence.Rtab`` from:
  /home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/panaroo_run/<directory_leaf>/

Then:
  - validates all GPA samples are present in metadata
  - filters rare genes by prevalence
  - runs Jaccard KNN, UMAP, Leiden (r=0.3), and small-cluster merge
  - produces post-filter GPA plots + non-core penetrance density plot
  - runs Wilcoxon rank_genes_groups by GPA Leiden clusters
  - logs RefSeq-focused quality summary and reference-vs-all Jaccard distances
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
from scipy.spatial.distance import cdist
from bacotype.tl.panaroo_jaccard_medoid_metrics import (
    log_medoid_report,
    medoid_metrics_from_dist_sq,
)
from bacotype.tl.panaroo_pangenome_features import (
    filter_by_prevalence,
)

PANAROO_RUN_ROOT = (
    "/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/panaroo_run"
)
DEFAULT_METADATA_PATH = (
    "/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/final/"
    "metadata_final_curated_all_samples_and_columns.tsv"
)
LEIDEN_RESOLUTION = 0.3
QUALITY_SUBSAMPLE_THRESHOLD = 2000
MIN_CLUSTER_SIZE = 50
SECTION_BAR = "=" * 80
_LOG_PATH_ROOT: str | None = None


def _set_log_path_root(panaroo_dir: str) -> None:
    global _LOG_PATH_ROOT
    _LOG_PATH_ROOT = os.path.dirname(panaroo_dir.rstrip("/"))


def _fmt_log_path(path: str) -> str:
    if _LOG_PATH_ROOT is None:
        return path
    try:
        rel = os.path.relpath(path, _LOG_PATH_ROOT)
    except ValueError:
        return path
    return rel if not rel.startswith("..") else path


def _stderr_line_buffered() -> None:
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


def _save_fig(fig: plt.Figure, path: str, log) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"plot: saved {_fmt_log_path(path)}")


def _default_filter_cutoff(n_samples: int) -> int:
    return max(2, int(0.01 * n_samples))


def _compute_k(n_samples: int) -> int:
    if n_samples <= 1000:
        return 25
    if n_samples >= 2000:
        return 50
    return 25 + round(25 * (n_samples - 1000) / 1000)


def _load_metadata(meta_path: str, log) -> pd.DataFrame:
    meta_df = pd.read_csv(meta_path, sep="\t", low_memory=False)
    if "Sample" not in meta_df.columns:
        raise ValueError(f"Metadata missing required column 'Sample': {meta_path}")
    meta_df = (
        meta_df.drop_duplicates(subset=["Sample"], keep="first")
        .set_index("Sample")
    )
    meta_df.index = meta_df.index.astype(str)
    log(f"metadata: loaded {_fmt_log_path(meta_path)}  ({len(meta_df)} rows)")
    return meta_df


def _build_adata(
    binary_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    sample_ids: np.ndarray,
    feature_ids: np.ndarray,
    log,
) -> ad.AnnData:
    sid = pd.Index(sample_ids.astype(str))
    missing_in_meta = sid.difference(meta_df.index)
    if len(missing_in_meta):
        show = ", ".join(map(str, list(missing_in_meta[:5])))
        raise ValueError(
            "Metadata missing GPA samples. "
            f"missing_count={len(missing_in_meta)} first5=[{show}]"
        )

    X_u8 = binary_df.T.to_numpy(dtype=np.uint8, copy=False)
    X_sparse = csr_matrix(X_u8)
    obs = meta_df.reindex(sid)
    adata = ad.AnnData(
        X=X_sparse,
        obs=obs,
        var=pd.DataFrame(index=feature_ids.astype(str)),
    )
    adata.obs.index.name = "Sample"
    log(f"anndata: shape {adata.shape[0]} samples x {adata.shape[1]} genes")
    return adata


def _merge_small_clusters(
    adata: ad.AnnData,
    key: str,
    min_size: int,
) -> tuple[int, int, int]:
    labels = adata.obs[key].astype(str).copy()
    counts = labels.value_counts()
    small_clusters = set(counts[counts < min_size].index)
    if not small_clusters:
        return 0, 0, 0

    large_clusters = set(counts.index) - small_clusters
    conn = adata.obsp["connectivities"]
    n_reassigned = 0
    for idx in range(adata.n_obs):
        if labels.iloc[idx] not in small_clusters:
            continue
        row = conn[idx]
        neighbor_indices = row.indices
        neighbor_labels = labels.iloc[neighbor_indices]
        valid = neighbor_labels[neighbor_labels.isin(large_clusters)]
        if len(valid) == 0:
            continue
        labels.iloc[idx] = valid.value_counts().idxmax()
        n_reassigned += 1

    adata.obs[key] = labels
    new_counts = labels.value_counts()
    remaining_small = int((new_counts < min_size).sum())
    return len(small_clusters), n_reassigned, remaining_small


def _plot_umap_scatter(
    adata: ad.AnnData,
    color: str,
    out_path: str,
    title: str,
    log,
) -> None:
    if "X_umap" not in adata.obsm:
        raise ValueError("UMAP missing: adata.obsm['X_umap'] not found.")
    if color not in adata.obs.columns:
        raise ValueError(f"UMAP color '{color}' missing from adata.obs")

    umap = adata.obsm["X_umap"]
    labels = adata.obs[color].astype("object").fillna("NA")
    cats = labels.astype("category")
    codes = cats.cat.codes.to_numpy()
    categories = list(cats.cat.categories)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(title)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % cmap.N) for i in range(len(categories))]
    point_colors = [colors[i] for i in codes]
    ax.scatter(umap[:, 0], umap[:, 1], s=6, c=point_colors, alpha=0.85, linewidths=0)
    if len(categories) <= 20:
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=colors[i],
                markersize=6,
            )
            for i in range(len(categories))
        ]
        ax.legend(handles, categories, loc="best", frameon=False, fontsize=8)
    fig.tight_layout()
    _save_fig(fig, out_path, log)


def _plot_umap_refseq_highlight(
    adata: ad.AnnData,
    out_path: str,
    title: str,
    log,
) -> None:
    if "X_umap" not in adata.obsm:
        raise ValueError("UMAP missing: adata.obsm['X_umap'] not found.")
    if "is_refseq" not in adata.obs.columns:
        log("plot: skipping RefSeq UMAP (is_refseq missing in metadata)")
        return

    umap = adata.obsm["X_umap"]
    ref_mask = _series_to_bool(adata.obs["is_refseq"]).to_numpy()
    n_ref = int(ref_mask.sum())
    non_ref = ~ref_mask

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(title)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
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
        ax.legend(loc="best", frameon=False, fontsize=8)
    fig.tight_layout()
    _save_fig(fig, out_path, log)


def _plot_noncore_penetrance_density(
    gpa_df_unfiltered: pd.DataFrame,
    strain_label: str,
    out_path: str,
    log,
    core_high: float = 0.95,
) -> None:
    """Plot non-core penetrance histogram weighted by genome count.

    Each non-core gene contributes its penetrance value once per genome that
    carries it, so high-penetrance genes dominate in proportion to how many
    genomes they inhabit. Histogram bars are normalized by
    (n_genomes * mean_genes_per_genome).
    """
    n_samples = gpa_df_unfiltered.shape[1]
    if n_samples == 0:
        raise ValueError("Cannot plot penetrance density with zero samples.")

    counts = gpa_df_unfiltered.sum(axis=1).astype(float)
    penetrance = counts / float(n_samples)
    non_core_mask = penetrance < core_high
    non_core_pen = penetrance[non_core_mask].to_numpy(dtype=float, copy=False)
    non_core_cnt = counts[non_core_mask].round().astype(int).to_numpy(copy=False)

    if len(non_core_pen) == 0:
        log("plot: non-core penetrance density skipped (no non-core genes)")
        return

    # Build weighted array: each gene's penetrance value repeated count times
    x = np.repeat(non_core_pen, non_core_cnt)
    n_genome_gene_pairs = len(x)
    mean_genes_per_genome = float(gpa_df_unfiltered.sum(axis=0).mean())
    denom = float(n_samples) * mean_genes_per_genome
    weights = np.full(n_genome_gene_pairs, 1.0 / denom, dtype=float)
    log(
        f"plot: non-core penetrance density — "
        f"{non_core_mask.sum()} non-core genes, "
        f"{n_genome_gene_pairs} genome-gene pairs"
    )

    bins = np.arange(0.0, float(core_high) + 1e-9, 0.05)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(x, bins=bins, weights=weights, edgecolor="black", linewidth=0.3)
    ax.set_xlim(0, float(core_high))
    ax.set_xticks(bins)
    ax.set_xticklabels([f"{b:.2f}" for b in bins], rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Gene penetrance")
    ax.set_ylabel("Genes per genome (normalized by mean genome genes)")
    ax.set_title(f"Non-core gene penetrance — {strain_label}")
    fig.tight_layout()
    _save_fig(fig, out_path, log)


def _gpa_category_masks(
    frac: pd.Series,
    shell_cloud_cutoff: float,
    core_shell_cutoff: float,
) -> dict[str, pd.Series]:
    return {
        "ubiquitous": frac > 0.999,
        "core": (frac > 0.99) & (frac <= 0.999),
        "soft_core": (frac >= core_shell_cutoff) & (frac <= 0.99),
        "shell": (frac >= shell_cloud_cutoff) & (frac < core_shell_cutoff),
        "cloud": frac < shell_cloud_cutoff,
    }


def _plot_gpa_distribution_and_log(
    df: pd.DataFrame,
    strain_label: str,
    out_path: str,
    log,
    shell_cloud_cutoff: float,
    core_shell_cutoff: float,
) -> None:
    n_samples = int(df.shape[1])
    n_genes = int(df.shape[0])
    freq = df.sum(axis=1)
    frac = freq / float(n_samples)
    masks = _gpa_category_masks(frac, shell_cloud_cutoff, core_shell_cutoff)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(frac.to_numpy(dtype=float), bins=100, edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Fraction of samples carrying the gene")
    ax.set_ylabel("Number of genes")
    ax.set_title(f"Gene frequency distribution — {strain_label} (n={n_genes})")
    ax.axvline(shell_cloud_cutoff, color="red", ls="--", lw=1, label=f"shell/cloud={shell_cloud_cutoff:.2f}")
    ax.axvline(core_shell_cutoff, color="orange", ls="--", lw=1, label=f"core/shell={core_shell_cutoff:.2f}")
    ax.axvline(0.99, color="blue", ls="--", lw=1, label="core=0.99")
    ax.axvline(0.999, color="green", ls="--", lw=1, label="ubiquitous=0.999")
    ax.legend(loc="best", fontsize=8, frameon=False)
    fig.tight_layout()
    _save_fig(fig, out_path, log)

    log(f"Total genes: {n_genes}")
    label_order = [
        ("ubiquitous", "Ubiquitous (>99.9%)"),
        ("core", "Core (>99% and <=99.9%)"),
        ("soft_core", f"Soft core (>={core_shell_cutoff*100:.1f}% and <=99%)"),
        ("shell", f"Shell (>={shell_cloud_cutoff*100:.1f}% and <{core_shell_cutoff*100:.1f}%)"),
        ("cloud", f"Cloud (<{shell_cloud_cutoff*100:.1f}%)"),
    ]
    for key, label in label_order:
        n = int(masks[key].sum())
        pct = (100.0 * n / n_genes) if n_genes > 0 else 0.0
        log(f"  {label}: {n:>6} ({pct:.1f}%)")


def _plot_per_sample_category_counts(
    df: pd.DataFrame,
    strain_label: str,
    out_path: str,
    log,
    shell_cloud_cutoff: float,
    core_shell_cutoff: float,
) -> None:
    n_samples = int(df.shape[1])
    frac = df.sum(axis=1) / float(n_samples)
    masks = _gpa_category_masks(frac, shell_cloud_cutoff, core_shell_cutoff)

    cat_series = {
        "core": df.loc[masks["core"]].sum(axis=0),
        "soft_core": df.loc[masks["soft_core"]].sum(axis=0),
        "shell": df.loc[masks["shell"]].sum(axis=0),
        "cloud": df.loc[masks["cloud"]].sum(axis=0),
    }
    titles = {
        "core": "Core genes (>99% and <=99.9% penetrance)",
        "soft_core": f"Soft-core genes (>= {core_shell_cutoff*100:.0f}% and <=99% penetrance)",
        "shell": f"Shell genes (>= {shell_cloud_cutoff*100:.0f}% and < {core_shell_cutoff*100:.0f}% penetrance)",
        "cloud": f"Cloud genes (< {shell_cloud_cutoff*100:.0f}% penetrance)",
    }

    fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=True)
    for ax, key in zip(axes, ["core", "soft_core", "shell", "cloud"]):
        ax.hist(cat_series[key].to_numpy(dtype=float), bins=50, edgecolor="black", linewidth=0.3)
        ax.set_title(f"{titles[key]}\n{strain_label}", fontsize=9)
        ax.set_xlabel("Genes per genome")
        ax.set_ylabel("Number of samples")
    fig.tight_layout()
    _save_fig(fig, out_path, log)


def _stratified_subsample(
    labels: pd.Series,
    max_n: int,
    min_per_cluster: int,
    rng: np.random.Generator,
) -> np.ndarray:
    cluster_ids = labels.unique()
    indices_by_cluster = {c: np.where(labels.values == c)[0] for c in cluster_ids}
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
        selected = np.concatenate([selected, rng.choice(pool, size=extra, replace=False)])
    return np.sort(selected)


def _compute_quality_metrics(adata: ad.AnnData, key: str) -> dict:
    n = adata.n_obs
    labels = adata.obs[key].astype(str)
    rng = np.random.default_rng(42)
    if n <= QUALITY_SUBSAMPLE_THRESHOLD:
        X = adata.X.toarray().astype(np.uint8, copy=False)
        sub_labels = labels
    else:
        sub_idx = _stratified_subsample(labels, QUALITY_SUBSAMPLE_THRESHOLD, MIN_CLUSTER_SIZE, rng)
        X = adata.X[sub_idx].toarray().astype(np.uint8, copy=False)
        sub_labels = labels.iloc[sub_idx]

    # Use dense Jaccard via cdist to avoid creating unnecessary squareform intermediates.
    dist_sq = cdist(X, X, metric="jaccard")
    mean_feats = float(adata.X.toarray().astype(np.uint8).sum(axis=1).mean())
    return medoid_metrics_from_dist_sq(dist_sq, sub_labels, mean_feats)


def _run_rank_genes_groups(
    adata_gpa: ad.AnnData,
    cluster_key: str,
    out_dir: str,
    leaf_label: str,
    log,
    n_genes: int = 20,
) -> None:
    if cluster_key not in adata_gpa.obs.columns:
        log(f"rank_genes_groups: '{cluster_key}' missing; skipping")
        return

    unique_labels = adata_gpa.obs[cluster_key].astype(str).unique()
    if len(unique_labels) < 2:
        log("rank_genes_groups: only 1 cluster — skipping marker analysis")
        return

    rg_adata = adata_gpa.copy()
    rg_adata.obs["gpa_cluster"] = adata_gpa.obs[cluster_key].astype(str).values
    var_name_set = set(rg_adata.var_names.astype(str))
    collisions = [
        col for col in rg_adata.obs.columns if col != "gpa_cluster" and str(col) in var_name_set
    ]
    if collisions:
        rg_adata.obs = rg_adata.obs.drop(columns=collisions)
        show = ", ".join(collisions[:5])
        suffix = "..." if len(collisions) > 5 else ""
        log(
            f"rank_genes_groups: dropped {len(collisions)} obs columns "
            f"that overlap gene names ({show}{suffix})"
        )

    log(f"rank_genes_groups: groupby=gpa_cluster, method=wilcoxon, n_genes={n_genes}")
    sc.tl.rank_genes_groups(rg_adata, groupby="gpa_cluster", method="wilcoxon")

    old_figdir = sc.settings.figdir
    sc.settings.figdir = out_dir
    res_tag = f"gpa_r{LEIDEN_RESOLUTION}_{leaf_label}"
    plots = [
        (
            "heatmap",
            sc.pl.rank_genes_groups_heatmap,
            {"n_genes": n_genes, "show_gene_labels": True},
        ),
        ("dotplot", sc.pl.rank_genes_groups_dotplot, {"n_genes": n_genes}),
        ("matrixplot", sc.pl.rank_genes_groups_matrixplot, {"n_genes": n_genes}),
    ]
    for name, fn, kwargs in plots:
        fname = f"rank_genes_wilcoxon_{name}_{res_tag}.png"
        try:
            fn(rg_adata, save=f"_{fname}", show=False, **kwargs)
            log(f"plot: saved {_fmt_log_path(os.path.join(out_dir, fname))} (via sc.pl)")
        except Exception as exc:  # noqa: BLE001
            log(f"plot: {name} failed ({exc}); skipping")
    sc.settings.figdir = old_figdir

    try:
        df = sc.get.rank_genes_groups_df(rg_adata, group=None)
        tsv_path = os.path.join(out_dir, f"rank_genes_groups_{res_tag}.tsv")
        df.to_csv(tsv_path, sep="\t", index=False)
        log(f"save: rank_genes_groups table -> {_fmt_log_path(tsv_path)}")
    except Exception as exc:  # noqa: BLE001
        log(f"save: rank_genes_groups table failed ({exc})")


def _series_to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    s = series.astype(str).str.strip().str.lower()
    return s.isin({"1", "true", "t", "yes", "y"})


def _refseq_summary_and_distances(
    adata_gpa: ad.AnnData,
    log,
) -> None:
    _log_section(log, "FINAL REFERENCE SUMMARY")
    obs = adata_gpa.obs.copy()

    ratio = pd.to_numeric(obs.get("largest_contig"), errors="coerce") / pd.to_numeric(
        obs.get("total_size"), errors="coerce"
    )
    contig_count = pd.to_numeric(obs.get("contig_count"), errors="coerce")

    ref_mask = (
        _series_to_bool(obs["is_refseq"])
        if "is_refseq" in obs.columns
        else pd.Series(False, index=obs.index)
    )
    n_ref = int(ref_mask.sum())
    log(f"refseq: genomes in set = {n_ref}")

    ref_ids: pd.Index
    if n_ref > 0:
        ratio_ref = ratio[ref_mask]
        contig_ref = contig_count[ref_mask]
        ratio_ref_clean = ratio_ref.dropna()
        contig_ref_clean = contig_ref.dropna()
        if ratio_ref_clean.empty:
            log("refseq: largest_contig_ratio stats unavailable (all missing)")
        else:
            log(
                "refseq: largest_contig_ratio "
                f"min={ratio_ref_clean.min():.5f} "
                f"median={ratio_ref_clean.median():.5f} "
                f"max={ratio_ref_clean.max():.5f}"
            )
        if contig_ref_clean.empty:
            log("refseq: contig_count stats unavailable (all missing)")
        else:
            log(
                "refseq: contig_count "
                f"min={contig_ref_clean.min():.0f} "
                f"median={contig_ref_clean.median():.0f} "
                f"max={contig_ref_clean.max():.0f}"
            )
        ref_ids = obs.index[ref_mask]
    else:
        log("WARNING: No RefSeq genomes in this set.")
        ratio_all = ratio.dropna().sort_values()
        if ratio_all.empty:
            log("fallback ratio stats unavailable (largest_contig/total_size all missing)")
            ref_ids = obs.index[: min(10, len(obs))]
        else:
            top10 = ratio_all.tail(min(10, len(ratio_all)))
            log(
                "fallback largest_contig_ratio (all samples): "
                f"min={ratio_all.min():.5f} "
                f"median={ratio_all.median():.5f} "
                f"max={ratio_all.max():.5f} "
                f"median_top10={top10.median():.5f}"
            )
            ref_ids = top10.index

    if len(ref_ids) == 0:
        log("reference-vs-all distances: skipped (no reference samples selected)")
        return

    X_all = adata_gpa.X.toarray().astype(bool, copy=False)
    ref_pos = adata_gpa.obs_names.get_indexer(ref_ids)
    ref_pos = ref_pos[ref_pos >= 0]
    if len(ref_pos) == 0:
        log("reference-vs-all distances: skipped (reference ids not present in adata)")
        return

    mean_features = float(X_all.astype(np.uint8).sum(axis=1).mean())

    X_ref = X_all[ref_pos]
    dist = cdist(X_ref, X_all, metric="jaccard")  # shape (n_ref, n_samples)

    # Per-reference-genome mean Jaccard to all samples
    mean_per_ref = dist.mean(axis=1)  # shape (n_ref,)
    mean_of_means = float(mean_per_ref.mean())
    min_mean = float(mean_per_ref.min())
    max_mean = float(mean_per_ref.max())
    sd_of_means = float(mean_per_ref.std(ddof=1)) if len(mean_per_ref) > 1 else 0.0

    _log_section(log, "MEAN DISTANCE TO REFERENCE GENOMES")
    log(
        f"Jaccard distances (mean Jaccard per reference genome to all samples): "
        f"mean={mean_of_means:.5f}  sd=\u00b1{sd_of_means:.5f}  "
        f"range=[{min_mean:.5f}, {max_mean:.5f}]  "
        f"(n_ref={len(ref_pos)}, n_samples={X_all.shape[0]})"
    )

    # Shared-gene and differ-per-genome estimates at min, mean, max distances
    from bacotype.tl.panaroo_jaccard_medoid_metrics import jaccard_to_shared

    log(f"Translation to shared genes (mean genome size {mean_features:.0f}):")
    for label, d_j in [("Min ", min_mean), ("Mean", mean_of_means), ("Max ", max_mean)]:
        shared = jaccard_to_shared(d_j, mean_features)
        differ = mean_features - shared
        pct = 100.0 * shared / mean_features if mean_features > 0 else 0.0
        log(
            f"  {label} (Jaccard {d_j:.5f}): "
            f"shared={shared:.0f} genes ({pct:.1f}%), "
            f"differ={differ:.0f} genes per genome"
        )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory-leaf", required=True, help="Panaroo directory leaf, e.g. CG11_all")
    p.add_argument(
        "--metadata",
        default=DEFAULT_METADATA_PATH,
        help=f"Metadata TSV path (default: {DEFAULT_METADATA_PATH})",
    )
    p.add_argument(
        "--gpa-filter-cutoff",
        type=int,
        default=None,
        help="Min genomes for GPA prevalence filter. Default: max(2, int(0.01*n_samples)).",
    )
    p.add_argument(
        "--merge-small-clusters",
        type=int,
        default=None,
        help="Merge clusters smaller than this size by kNN majority vote. Default: max(10, int(0.01*n_samples)).",
    )
    p.add_argument(
        "--shell-cloud-cutoff",
        type=float,
        default=0.15,
        help="Penetrance cutoff between shell and cloud genes (default: 0.15).",
    )
    p.add_argument(
        "--core-shell-cutoff",
        type=float,
        default=0.95,
        help="Penetrance cutoff between soft-core and shell genes (default: 0.95).",
    )
    p.add_argument(
        "--report-times",
        type=_str2bool,
        default=False,
        metavar="BOOL",
        help="Include timing in report/log lines (default: False).",
    )
    args = p.parse_args()
    if not (0.0 < args.shell_cloud_cutoff < args.core_shell_cutoff < 1.0):
        raise ValueError(
            "Invalid cutoffs: require 0 < shell_cloud_cutoff < core_shell_cutoff < 1."
        )

    _stderr_line_buffered()
    t0 = time.perf_counter()
    sc.settings.verbosity = 0

    panaroo_dir = os.path.join(PANAROO_RUN_ROOT, args.directory_leaf)
    if not os.path.isdir(panaroo_dir):
        raise FileNotFoundError(f"Panaroo directory not found: {panaroo_dir}")
    _set_log_path_root(panaroo_dir)

    analysis_dir = os.path.join(panaroo_dir, "analysis", "GPA_reference_genome")
    os.makedirs(analysis_dir, exist_ok=True)
    log_path = os.path.join(analysis_dir, f"clustering_log_{args.directory_leaf}.txt")
    log_fh = open(log_path, "w")
    log = _make_progress_logger(t0, log_fh, report_times=args.report_times)

    log(f"start pid={os.getpid()} directory_leaf={args.directory_leaf}")
    log(f"path context: input={_fmt_log_path(panaroo_dir)}")
    log(f"path context: analysis={_fmt_log_path(analysis_dir)}")

    _log_section(log, "METADATA")
    meta_df = _load_metadata(args.metadata, log)

    _log_section(log, "LOAD GPA")
    gpa_rtab = os.path.join(panaroo_dir, "gene_presence_absence.Rtab")
    if not os.path.isfile(gpa_rtab):
        raise FileNotFoundError(f"Required file not found: {gpa_rtab}")
    log(f"load GPA: reading {_fmt_log_path(gpa_rtab)}")
    gpa_df = pd.read_csv(gpa_rtab, sep="\t", index_col=0, dtype=str, low_memory=False)
    gpa_df.columns = gpa_df.columns.astype(str)
    gpa_df = (gpa_df == "1").astype(np.uint8)
    log(f"load GPA: {gpa_df.shape[0]} genes x {gpa_df.shape[1]} samples")

    _log_section(log, "GENES IN A GENOME")
    n_samples = int(gpa_df.shape[1])
    genes_per_genome = gpa_df.sum(axis=0)
    log(
        f"genes per genome (unfiltered): "
        f"mean={genes_per_genome.mean():.1f} "
        f"sd={genes_per_genome.std(ddof=1):.1f} "
        f"(n_samples={n_samples})"
    )
    filter_cutoff = (
        args.gpa_filter_cutoff
        if args.gpa_filter_cutoff is not None
        else _default_filter_cutoff(n_samples)
    )
    log(f"filter cutoff (GPA): n_samples={n_samples} -> {filter_cutoff} (min genomes)")
    gpa_df_filt = filter_by_prevalence(gpa_df, min_prevalence=filter_cutoff, feature_label="gene")
    n_removed = int(gpa_df.shape[0] - gpa_df_filt.shape[0])
    log(
        f"filter: {gpa_df.shape[0]} -> {gpa_df_filt.shape[0]} genes "
        f"(removed={n_removed}, min_prevalence={filter_cutoff})"
    )
    genes_per_genome_filt = gpa_df_filt.sum(axis=0)
    log(
        f"genes per genome (post-filter): "
        f"mean={genes_per_genome_filt.mean():.1f} "
        f"sd={genes_per_genome_filt.std(ddof=1):.1f}"
    )

    _plot_gpa_distribution_and_log(
        gpa_df_filt,
        args.directory_leaf,
        os.path.join(analysis_dir, f"gpa_freq_dist_post_filter_{args.directory_leaf}.png"),
        log,
        shell_cloud_cutoff=args.shell_cloud_cutoff,
        core_shell_cutoff=args.core_shell_cutoff,
    )
    _plot_per_sample_category_counts(
        gpa_df_filt,
        args.directory_leaf,
        os.path.join(analysis_dir, f"gpa_core_softcore_shell_cloud_post_filter_{args.directory_leaf}.png"),
        log,
        shell_cloud_cutoff=args.shell_cloud_cutoff,
        core_shell_cutoff=args.core_shell_cutoff,
    )
    _plot_noncore_penetrance_density(
        gpa_df,
        args.directory_leaf,
        os.path.join(analysis_dir, f"gpa_noncore_penetrance_density_{args.directory_leaf}.png"),
        log,
        core_high=args.core_shell_cutoff,
    )

    _log_section(log, "ANNDATA + CLUSTERING")
    adata_gpa = _build_adata(
        gpa_df_filt,
        meta_df,
        gpa_df_filt.columns.to_numpy(),
        gpa_df_filt.index.to_numpy(),
        log,
    )
    del gpa_df_filt

    n_samples = adata_gpa.n_obs
    k = _compute_k(n_samples)
    merge_min_size = (
        int(args.merge_small_clusters)
        if args.merge_small_clusters is not None
        else max(10, int(0.01 * n_samples))
    )
    log(f"knn: n_samples={n_samples}, computed k={k}")
    log(f"merge: n_samples={n_samples} merge_small_clusters_min_size={merge_min_size}")
    try:
        sc.pp.neighbors(adata_gpa, n_neighbors=k, metric="jaccard", use_rep="X")
    except Exception as exc:  # noqa: BLE001
        log(
            "knn: sparse Jaccard neighbor build failed "
            f"({exc}); retrying with dense boolean matrix + sklearn"
        )
        adata_gpa.obsm["X_jaccard_dense"] = adata_gpa.X.toarray().astype(bool, copy=False)
        sc.pp.neighbors(
            adata_gpa,
            n_neighbors=k,
            metric="jaccard",
            use_rep="X_jaccard_dense",
            transformer="sklearn",
        )
    log("knn: neighbors graph ready")

    sc.tl.umap(adata_gpa)
    log("umap: done")

    key = f"gpa_leiden_r{LEIDEN_RESOLUTION}"
    sc.tl.leiden(adata_gpa, resolution=LEIDEN_RESOLUTION, key_added=key)
    raw_counts = adata_gpa.obs[key].value_counts()
    log(
        f"leiden ({key}): {len(raw_counts)} raw clusters, "
        f"sizes min={raw_counts.min()} max={raw_counts.max()}"
    )
    n_small, n_reass, n_remain = _merge_small_clusters(adata_gpa, key, merge_min_size)
    if n_small == 0:
        log(f"merge summary: sub-threshold clusters (<{merge_min_size}) before merge: 0")
    else:
        log(f"merge summary: sub-threshold clusters (<{merge_min_size}) before merge: {n_small}")
        log(f"merge summary: genomes reassigned by majority vote: {n_reass}")
        log(f"merge summary: sub-threshold clusters (<{merge_min_size}) after merge: {n_remain}")

    _plot_umap_scatter(
        adata_gpa,
        color=key,
        out_path=os.path.join(analysis_dir, f"umap_gpa_leiden_r{LEIDEN_RESOLUTION}_{args.directory_leaf}.png"),
        title=f"UMAP - GPA Leiden r={LEIDEN_RESOLUTION} - {args.directory_leaf}",
        log=log,
    )
    if "K_locus" in adata_gpa.obs.columns:
        _plot_umap_scatter(
            adata_gpa,
            color="K_locus",
            out_path=os.path.join(analysis_dir, f"umap_gpa_klocus_{args.directory_leaf}.png"),
            title=f"UMAP - GPA K_locus - {args.directory_leaf}",
            log=log,
        )
    _plot_umap_refseq_highlight(
        adata_gpa,
        out_path=os.path.join(analysis_dir, f"umap_gpa_refseq_{args.directory_leaf}.png"),
        title=f"UMAP - GPA RefSeq highlighted - {args.directory_leaf}",
        log=log,
    )

    _log_section(log, "GPA QUALITY ASSESSMENT")
    metrics = _compute_quality_metrics(adata_gpa, key)
    log_medoid_report(
        log,
        f"GPA medoid metrics - Leiden r={LEIDEN_RESOLUTION}",
        "gene",
        metrics,
    )

    summary_df = pd.DataFrame(
        [
            {
                "directory_leaf": args.directory_leaf,
                "modality": "gpa",
                "resolution": LEIDEN_RESOLUTION,
                "n_samples": int(adata_gpa.n_obs),
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
            }
        ]
    )
    for ext, sep in [(".tsv", "\t"), (".csv", ",")]:
        p_out = os.path.join(analysis_dir, f"gpa_clustering_summary_{args.directory_leaf}{ext}")
        summary_df.to_csv(p_out, sep=sep, index=False)
        log(f"save: summary table -> {_fmt_log_path(p_out)}")

    _log_section(log, "MARKER GENES (GPA BY GPA CLUSTER)")
    _run_rank_genes_groups(adata_gpa, key, analysis_dir, args.directory_leaf, log)

    _refseq_summary_and_distances(adata_gpa, log)

    total_wall = time.perf_counter() - t0
    _log_section(log, "FINAL SUMMARY")
    log(f"DONE total_wall={total_wall:.1f}s exit=0")
    log_fh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
