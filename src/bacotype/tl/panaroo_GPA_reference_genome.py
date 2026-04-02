#!/usr/bin/env python3
r"""
GPA-only Jaccard clustering for a Panaroo directory leaf.

Loads ``gene_presence_absence.Rtab`` from:
  /home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/panaroo_run/<directory_leaf>/

Then:
  - validates all GPA samples are present in metadata
  - filters rare genes by prevalence
  - runs Jaccard KNN, UMAP, Leiden (r=0.3), and small-cluster merge
  - produces post-filter GPA plots
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
    jaccard_to_shared,
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


def _plot_umap_reference_highlights(
    adata: ad.AnnData,
    out_path: str,
    title: str,
    log,
) -> None:
    if "X_umap" not in adata.obsm:
        raise ValueError("UMAP missing: adata.obsm['X_umap'] not found.")
    if "is_refseq" not in adata.obs.columns:
        log("plot: skipping reference-highlight UMAP (is_refseq missing in metadata)")
        return

    umap = adata.obsm["X_umap"]
    ref_mask = _series_to_bool(adata.obs["is_refseq"]).to_numpy()
    if "is_complete_norway_genome" in adata.obs.columns:
        norway_mask = _series_to_bool(adata.obs["is_complete_norway_genome"]).to_numpy()
    else:
        norway_mask = np.zeros(adata.n_obs, dtype=bool)

    ref_only = ref_mask & ~norway_mask
    base = ~(ref_mask | norway_mask)
    n_norway = int(norway_mask.sum())

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(title)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    if base.any():
        ax.scatter(
            umap[base, 0],
            umap[base, 1],
            s=6,
            c="lightgray",
            alpha=0.65,
            linewidths=0,
            label=f"other (n={int(base.sum())})",
            zorder=1,
        )
    if ref_only.any():
        ax.scatter(
            umap[ref_only, 0],
            umap[ref_only, 1],
            s=48,
            c="#ff0000",
            alpha=0.98,
            linewidths=0.5,
            edgecolors="black",
            label=f"RefSeq only (n={int(ref_only.sum())})",
            zorder=2,
        )
    if norway_mask.any():
        ax.scatter(
            umap[norway_mask, 0],
            umap[norway_mask, 1],
            s=48,
            c="#228B22",
            alpha=0.98,
            linewidths=0.5,
            edgecolors="black",
            label=f"complete Norway (n={n_norway})",
            zorder=4,
        )
    ax.legend(loc="best", frameon=False, fontsize=8)
    fig.tight_layout()
    _save_fig(fig, out_path, log)


def _gpa_category_masks(
    frac: pd.Series,
    shell_cloud_cutoff: float,
    core_shell_cutoff: float,
) -> dict[str, pd.Series]:
    return {
        "ubiquitous": frac > 0.999,
        # Core intentionally includes ubiquitous genes.
        "core": frac > 0.99,
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
        ("core", "Core (>99%, includes ubiquitous)"),
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
        "core": "Core genes (>99% penetrance, includes ubiquitous)",
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


def _compute_per_genome_category_stats(
    df: pd.DataFrame,
    shell_cloud_cutoff: float,
    core_shell_cutoff: float,
) -> dict[str, float]:
    n_samples = int(df.shape[1])
    frac = df.sum(axis=1) / float(n_samples)
    masks = _gpa_category_masks(frac, shell_cloud_cutoff, core_shell_cutoff)
    cat_series = {
        "core": df.loc[masks["core"]].sum(axis=0).to_numpy(dtype=float),
        "soft_core": df.loc[masks["soft_core"]].sum(axis=0).to_numpy(dtype=float),
        "shell": df.loc[masks["shell"]].sum(axis=0).to_numpy(dtype=float),
        "cloud": df.loc[masks["cloud"]].sum(axis=0).to_numpy(dtype=float),
    }
    genome_size = df.sum(axis=0).to_numpy(dtype=float)

    def _mean_sd(arr: np.ndarray) -> tuple[float, float]:
        if len(arr) == 0:
            return 0.0, 0.0
        sd = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        return float(arr.mean()), sd

    mean_genome, sd_genome = _mean_sd(genome_size)
    mean_core, sd_core = _mean_sd(cat_series["core"])
    mean_soft, sd_soft = _mean_sd(cat_series["soft_core"])
    mean_shell, sd_shell = _mean_sd(cat_series["shell"])
    mean_cloud, sd_cloud = _mean_sd(cat_series["cloud"])
    return {
        "mean_genome_size": mean_genome,
        "sd_genome_size": sd_genome,
        "mean_core_genes": mean_core,
        "sd_core_genes": sd_core,
        "mean_softcore_genes": mean_soft,
        "sd_softcore_genes": sd_soft,
        "mean_shell_genes": mean_shell,
        "sd_shell_genes": sd_shell,
        "mean_cloud_genes": mean_cloud,
        "sd_cloud_genes": sd_cloud,
    }


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


def _filter_gpa_to_kpsc(gpa_df: pd.DataFrame, meta_df: pd.DataFrame, log) -> tuple[pd.DataFrame, int, int]:
    if "kpsc_final_list" not in meta_df.columns:
        raise ValueError("Metadata missing required column 'kpsc_final_list'")
    sample_ids = pd.Index(gpa_df.columns.astype(str))
    missing_in_meta = sample_ids.difference(meta_df.index)
    if len(missing_in_meta):
        show = ", ".join(map(str, list(missing_in_meta[:5])))
        raise ValueError(
            "Metadata missing GPA samples before kpsc filter. "
            f"missing_count={len(missing_in_meta)} first5=[{show}]"
        )
    kpsc_mask = _series_to_bool(meta_df.reindex(sample_ids)["kpsc_final_list"])
    keep_ids = sample_ids[kpsc_mask.to_numpy()]
    n_raw = int(len(sample_ids))
    n_kpsc = int(len(keep_ids))
    n_drop = n_raw - n_kpsc
    log(f"kpsc filter: raw GPA samples={n_raw} kept={n_kpsc} dropped={n_drop}")
    return gpa_df.loc[:, keep_ids].copy(), n_raw, n_kpsc


def _species_is_klebsiella_pneumoniae(species_val: object) -> bool:
    s = str(species_val).strip().lower()
    return s == "klebsiella pneumoniae"


def _classify_run(
    gpa_sample_ids: pd.Index,
    meta_df: pd.DataFrame,
    log,
) -> dict[str, object]:
    needed = ["kpsc_final_list", "Sublineage", "Clonal group"]
    missing = [col for col in needed if col not in meta_df.columns]
    if missing:
        raise ValueError(f"Metadata missing required columns for run classification: {missing}")

    sid = pd.Index(gpa_sample_ids.astype(str))
    sid_set = set(sid.astype(str))
    missing_in_meta = sid.difference(meta_df.index)
    if len(missing_in_meta):
        show = ", ".join(map(str, list(missing_in_meta[:5])))
        raise ValueError(
            "Metadata missing GPA samples for run classification. "
            f"missing_count={len(missing_in_meta)} first5=[{show}]"
        )

    kpsc_meta = meta_df.loc[_series_to_bool(meta_df["kpsc_final_list"])].copy()
    run_meta = kpsc_meta.reindex(sid)
    run_meta = run_meta.dropna(subset=["Sublineage", "Clonal group"])
    if run_meta.empty:
        return {
            "n_unique_sublineages": 0,
            "sublineages_complete": False,
            "n_unique_clonal_groups": 0,
            "clonal_groups_complete": False,
            "species": "",
            "strain": "",
            "samples_in_strain": 0,
            "run_classification": "unknown",
        }

    run_subs = set(run_meta["Sublineage"].astype(str).unique())
    run_cgs = set(run_meta["Clonal group"].astype(str).unique())
    # Species label: only if all kpsc-filtered samples in this run share exactly one non-null species value.
    species_unique_set: set[str] = set()
    if "species" in kpsc_meta.columns:
        sp_series = kpsc_meta.reindex(sid)["species"]
        sp_unique = list(pd.Series(sp_series).dropna().astype(str).unique())
        species_unique_set = set(sp_unique)
        species_label = sp_unique[0] if len(sp_unique) == 1 else ""
    else:
        species_label = ""

    # All kpsc samples in the sublineages touched by this run (for sublineage-level completeness).
    full_cgs_for_run_subs = set(
        kpsc_meta.loc[kpsc_meta["Sublineage"].astype(str).isin(run_subs), "Clonal group"]
        .dropna()
        .astype(str)
        .unique()
    )
    full_samples_for_run_subs = set(
        kpsc_meta.loc[kpsc_meta["Sublineage"].astype(str).isin(run_subs)].index.astype(str)
    )

    sublineages_complete = bool(
        full_samples_for_run_subs and sid_set >= full_samples_for_run_subs
    )

    # True iff for every clonal group present in the run, all kpsc_final_list samples with that CG
    # (in full metadata) are included in this GPA run. This is NOT "all CGs in the sublineage".
    clonal_groups_complete = True
    for cg in run_cgs:
        cg_samples = kpsc_meta.loc[kpsc_meta["Clonal group"].astype(str) == cg].index.astype(str)
        cg_set = set(cg_samples)
        if not cg_set:
            clonal_groups_complete = False
            break
        if not cg_set <= sid_set:
            clonal_groups_complete = False
            break

    # Entire sublineage represented: every CG that exists in kpsc for these sublineages is in the run.
    all_cgs_of_sublineages_in_run = bool(
        full_cgs_for_run_subs and run_cgs == full_cgs_for_run_subs
    )

    all_kp_species = True
    if "species" in meta_df.columns:
        sp_run = kpsc_meta.reindex(sid)["species"]
        all_kp_species = bool(sp_run.notna().all() and sp_run.map(_species_is_klebsiella_pneumoniae).all())
    else:
        log("run classification: species column missing; skipping non-kp-species override")

    if not all_kp_species:
        run_type = "non-kp-species"
    else:
        n_set_samples = int(len(sid_set))
        # clonal-group level
        if len(run_cgs) == 1:
            if clonal_groups_complete:
                run_type = "clonal-group"
            elif n_set_samples > 1000:
                run_type = "clonal-group-split"
            else:
                run_type = "unknown"
        # sublineage level
        elif len(run_subs) == 1 and len(run_cgs) > 1:
            if (
                all_cgs_of_sublineages_in_run
                and sublineages_complete
            ):
                run_type = "sublineage"
            elif n_set_samples > 1000 and not sublineages_complete:
                run_type = "sublineage-split"
            else:
                run_type = "sublineage-other"
        elif len(run_subs) > 1:
            run_type = "rare-lineage"
        else:
            run_type = "unknown"

    out = {
        "n_unique_sublineages": int(len(run_subs)),
        "sublineages_complete": bool(sublineages_complete),
        "n_unique_clonal_groups": int(len(run_cgs)),
        "clonal_groups_complete": bool(clonal_groups_complete),
        "species": species_label,
        # "strain" is the unique identifier we want to carry forward for plotting.
        # It is *independent* of the completeness checks used for run_classification.
        "strain": "",
        "samples_in_strain": 0,
        "run_classification": run_type,
    }

    # strain selection precedence:
    # 1) only one clonal group in the GPA sample -> use that CG name
    # 2) else, only one sublineage (but >1 CG) -> use that sublineage name
    # 3) else, only one species in the GPA sample -> use that species name
    # 4) else -> mixed_batch
    if len(run_cgs) == 1:
        strain_val = next(iter(run_cgs))
        samples_in_strain = int(
            (kpsc_meta["Clonal group"].astype(str) == strain_val).sum()
        )
    elif len(run_cgs) > 1 and len(run_subs) == 1:
        strain_val = next(iter(run_subs))
        samples_in_strain = int(
            (kpsc_meta["Sublineage"].astype(str) == strain_val).sum()
        )
    elif species_label != "":
        strain_val = species_label
        samples_in_strain = int(
            (kpsc_meta["species"].astype(str) == strain_val).sum()
        ) if "species" in kpsc_meta.columns else 0
    else:
        strain_val = "mixed_batch"
        # union across all clonal groups/sublineages/species touched by this run in metadata
        mask = pd.Series(False, index=kpsc_meta.index)
        mask |= kpsc_meta["Clonal group"].astype(str).isin(run_cgs)
        mask |= kpsc_meta["Sublineage"].astype(str).isin(run_subs)
        if "species" in kpsc_meta.columns and species_unique_set:
            sp_col = kpsc_meta["species"]
            mask |= sp_col.notna() & sp_col.astype(str).isin(species_unique_set)
        samples_in_strain = int(mask.sum())

    out["strain"] = strain_val
    out["samples_in_strain"] = int(samples_in_strain)
    log(
        "run classification: "
        f"type={out['run_classification']} "
        f"n_sublineages={out['n_unique_sublineages']} "
        f"sublineages_complete={out['sublineages_complete']} "
        f"n_clonal_groups={out['n_unique_clonal_groups']} "
        f"clonal_groups_complete={out['clonal_groups_complete']} "
        f"(CG-complete means every CG in the run includes all its kpsc_final_list samples)"
    )
    return out


def _mgh78578_sample_in_gpa(meta_df: pd.DataFrame, gpa_columns: pd.Index) -> str | None:
    if "is_mgh78578" not in meta_df.columns:
        return None
    col = meta_df.reindex(gpa_columns.astype(str))["is_mgh78578"]
    if pd.api.types.is_bool_dtype(col):
        mask = col.fillna(False).to_numpy(dtype=bool)
    else:
        mask = _series_to_bool(col).to_numpy()
    hits = gpa_columns.astype(str).to_numpy()[mask]
    if len(hits) == 0:
        return None
    return str(hits[0])


def _empty_cohort_flat_keys(key_prefix: str) -> dict[str, object]:
    return {
        f"{key_prefix}_min_mean_jaccard": np.nan,
        f"{key_prefix}_mean_mean_jaccard": np.nan,
        f"{key_prefix}_max_mean_jaccard": np.nan,
        f"{key_prefix}_min_shared_genes": np.nan,
        f"{key_prefix}_mean_shared_genes": np.nan,
        f"{key_prefix}_max_shared_genes": np.nan,
        f"{key_prefix}_min_shared_pct": np.nan,
        f"{key_prefix}_mean_shared_pct": np.nan,
        f"{key_prefix}_max_shared_pct": np.nan,
        f"{key_prefix}_top_genome_ids": [],
        f"{key_prefix}_top_mean_jaccards": [],
    }


def _cohort_jaccard_flat_summary(
    adata_gpa: ad.AnnData,
    ref_positions: np.ndarray,
    mean_features: float,
    top_n: int,
    log,
    *,
    section_title: str,
    log_tag: str,
    key_prefix: str,
) -> dict[str, object]:
    """Min/mean/max of per-genome mean Jaccard to all samples; shared-gene translation; top-N best matches."""
    out = _empty_cohort_flat_keys(key_prefix)
    if len(ref_positions) == 0:
        return out

    X_all = adata_gpa.X.toarray().astype(bool, copy=False)
    ref_pos = ref_positions[ref_positions >= 0]
    if len(ref_pos) == 0:
        return out

    X_ref = X_all[ref_pos]
    dist = cdist(X_ref, X_all, metric="jaccard")
    mean_per_ref = dist.mean(axis=1)
    mean_of_means = float(mean_per_ref.mean())
    min_mean = float(mean_per_ref.min())
    max_mean = float(mean_per_ref.max())
    sd_of_means = float(mean_per_ref.std(ddof=1)) if len(mean_per_ref) > 1 else 0.0

    _log_section(log, section_title)
    log(
        f"{log_tag}: Jaccard distances (mean Jaccard per genome to all samples): "
        f"mean={mean_of_means:.5f}  sd=\u00b1{sd_of_means:.5f}  "
        f"range=[{min_mean:.5f}, {max_mean:.5f}]  "
        f"(n_cohort={len(ref_pos)}, n_samples={X_all.shape[0]})"
    )
    log(f"{log_tag}: translation to shared genes (mean genome size {mean_features:.0f}):")
    for label, d_j in [("Min ", min_mean), ("Mean", mean_of_means), ("Max ", max_mean)]:
        shared = jaccard_to_shared(d_j, mean_features)
        differ = mean_features - shared
        pct = 100.0 * shared / mean_features if mean_features > 0 else 0.0
        log(
            f"  {label} (Jaccard {d_j:.5f}): "
            f"shared={shared:.0f} genes ({pct:.1f}%), "
            f"differ={differ:.0f} genes per genome"
        )

    min_shared = jaccard_to_shared(min_mean, mean_features)
    mean_shared = jaccard_to_shared(mean_of_means, mean_features)
    max_shared = jaccard_to_shared(max_mean, mean_features)
    mf = mean_features
    out.update(
        {
            f"{key_prefix}_min_mean_jaccard": float(min_mean),
            f"{key_prefix}_mean_mean_jaccard": float(mean_of_means),
            f"{key_prefix}_max_mean_jaccard": float(max_mean),
            f"{key_prefix}_min_shared_genes": float(min_shared),
            f"{key_prefix}_mean_shared_genes": float(mean_shared),
            f"{key_prefix}_max_shared_genes": float(max_shared),
            f"{key_prefix}_min_shared_pct": float(100.0 * min_shared / mf if mf > 0 else 0.0),
            f"{key_prefix}_mean_shared_pct": float(100.0 * mean_shared / mf if mf > 0 else 0.0),
            f"{key_prefix}_max_shared_pct": float(100.0 * max_shared / mf if mf > 0 else 0.0),
        }
    )
    order = np.argsort(mean_per_ref)
    take = order[: min(top_n, len(order))]
    top_ids = [str(adata_gpa.obs_names[ref_pos[i]]) for i in take]
    top_j = [float(mean_per_ref[i]) for i in take]
    out[f"{key_prefix}_top_genome_ids"] = top_ids
    out[f"{key_prefix}_top_mean_jaccards"] = top_j
    return out


def _log_mgh78578_genome_categories(
    gpa_df_stats: pd.DataFrame,
    mgh_id: str,
    shell_cloud_cutoff: float,
    core_shell_cutoff: float,
    log,
) -> None:
    n_samples = int(gpa_df_stats.shape[1])
    if mgh_id not in gpa_df_stats.columns or n_samples == 0:
        return
    frac = gpa_df_stats.sum(axis=1) / float(n_samples)
    masks = _gpa_category_masks(frac, shell_cloud_cutoff, core_shell_cutoff)
    col = gpa_df_stats[mgh_id]
    genome_size = int(col.sum())
    log(
        f"global reference (mgh78578) genome size (genes in filtered GPA): {genome_size}"
    )
    for key, label in [
        ("core", "core (>99% cohort penetrance)"),
        ("soft_core", "soft_core"),
        ("shell", "shell"),
        ("cloud", "cloud"),
    ]:
        n = int(gpa_df_stats.loc[masks[key], mgh_id].astype(np.uint8).sum())
        log(f"  {label} genes carried: {n}")


def _global_reference_mgh78578_summary(
    adata_gpa: ad.AnnData,
    mgh_id: str | None,
    mean_genome_size_excl_mgh: float,
    log,
) -> dict[str, object]:
    nan_keys = {
        "global_ref_mean_jaccard_to_others": np.nan,
        "global_ref_mean_shared_genes": np.nan,
        "global_ref_mean_shared_pct": np.nan,
        "global_ref_sd_shared_genes": np.nan,
        "global_ref_divergent_genes_mean": np.nan,
    }
    if mgh_id is None:
        _log_section(log, "GLOBAL REFERENCE (MGH78578)")
        log(
            "Global reference genome not in GPA, distances to it cannot be calculated"
        )
        return {"global_ref_in_gpa": 0, **nan_keys}

    m_pos = adata_gpa.obs_names.get_loc(mgh_id)
    if not isinstance(m_pos, (int, np.integer)):
        log("global ref: ambiguous index for mgh78578; skipping distances")
        return {"global_ref_in_gpa": 0, **nan_keys}

    X_all = adata_gpa.X.toarray().astype(bool, copy=False)
    row_m = X_all[int(m_pos)]
    other_mask = np.ones(adata_gpa.n_obs, dtype=bool)
    other_mask[int(m_pos)] = False
    X_others = X_all[other_mask]
    dist = cdist(row_m.reshape(1, -1), X_others, metric="jaccard")
    mean_j = float(dist.mean())
    shared_counts = (row_m.astype(np.uint8) & X_others.astype(np.uint8)).sum(axis=1).astype(float)
    mean_shared = float(shared_counts.mean())
    sd_shared = (
        float(shared_counts.std(ddof=1)) if len(shared_counts) > 1 else 0.0
    )
    mg = mean_genome_size_excl_mgh
    pct = float(100.0 * mean_shared / mg) if mg > 0 else 0.0
    divergent = float(mg - mean_shared)

    _log_section(log, "GLOBAL REFERENCE (MGH78578)")
    log(
        f"Mean Jaccard to all other samples in set: {mean_j:.5f}"
    )
    log(f"Mean shared genes with members of set: {mean_shared:.0f}")
    log(f"% shared genes (vs mean genome size excl. mgh78578): {pct:.1f}%")
    log(f"S.D. of shared genes with members of set: {sd_shared:.2f}")
    log(f"Divergent genes (mean genome excl. mgh − mean shared): {divergent:.0f}")

    return {
        "global_ref_in_gpa": 1,
        "global_ref_mean_jaccard_to_others": mean_j,
        "global_ref_mean_shared_genes": mean_shared,
        "global_ref_mean_shared_pct": pct,
        "global_ref_sd_shared_genes": sd_shared,
        "global_ref_divergent_genes_mean": divergent,
    }


def _refseq_summary_and_distances(
    adata_gpa: ad.AnnData,
    log,
    reference_top_n: int,
) -> dict[str, object]:
    ref_summary: dict[str, object] = {
        "n_refseq_genomes": 0,
        **_empty_cohort_flat_keys("ref"),
    }
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
    ref_summary["n_refseq_genomes"] = n_ref
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
        return ref_summary

    ref_pos = adata_gpa.obs_names.get_indexer(ref_ids)
    ref_pos = ref_pos[ref_pos >= 0]
    if len(ref_pos) == 0:
        log("reference-vs-all distances: skipped (reference ids not present in adata)")
        return ref_summary

    X_all = adata_gpa.X.toarray().astype(bool, copy=False)
    mean_features = float(X_all.astype(np.uint8).sum(axis=1).mean())

    cohort = _cohort_jaccard_flat_summary(
        adata_gpa,
        ref_pos,
        mean_features,
        reference_top_n,
        log,
        section_title="MEAN DISTANCE TO REFERENCE GENOMES",
        log_tag="refseq",
        key_prefix="ref",
    )
    ref_summary.update(cohort)
    return ref_summary


def _norway_complete_summary_and_distances(
    adata_gpa: ad.AnnData,
    log,
    reference_top_n: int,
) -> dict[str, object]:
    count_key = "n_norway_complete_genomes"
    out: dict[str, object] = {
        count_key: 0,
        **_empty_cohort_flat_keys("norway"),
    }
    obs = adata_gpa.obs
    if "is_complete_norway_genome" not in obs.columns:
        log("complete Norway: column is_complete_norway_genome missing; skipping")
        return out

    mask = _series_to_bool(obs["is_complete_norway_genome"]).to_numpy()
    n_n = int(mask.sum())
    out[count_key] = n_n
    _log_section(log, "COMPLETE NORWAY GENOMES IN SET")
    log(f"complete Norway: genomes in set = {n_n}")
    if n_n == 0:
        log("complete Norway: no genomes — distance summary not computed")
        return out

    ref_pos = np.nonzero(mask)[0]
    X_all = adata_gpa.X.toarray().astype(bool, copy=False)
    mean_features = float(X_all.astype(np.uint8).sum(axis=1).mean())

    cohort = _cohort_jaccard_flat_summary(
        adata_gpa,
        ref_pos,
        mean_features,
        reference_top_n,
        log,
        section_title="MEAN DISTANCE TO COMPLETE NORWAY GENOMES",
        log_tag="norway_complete",
        key_prefix="norway",
    )
    out.update(cohort)
    return out


def run_gpa_analysis(
    directory_leaf: str,
    metadata_path: str = DEFAULT_METADATA_PATH,
    gpa_filter_cutoff: int | None = None,
    merge_small_clusters: int | None = None,
    shell_cloud_cutoff: float = 0.15,
    core_shell_cutoff: float = 0.95,
    report_times: bool = False,
    reference_top_n: int = 10,
) -> dict[str, object]:
    if not (0.0 < shell_cloud_cutoff < core_shell_cutoff < 1.0):
        raise ValueError(
            "Invalid cutoffs: require 0 < shell_cloud_cutoff < core_shell_cutoff < 1."
        )
    if reference_top_n < 1:
        raise ValueError("reference_top_n must be >= 1")

    _stderr_line_buffered()
    t0 = time.perf_counter()
    sc.settings.verbosity = 0

    panaroo_dir = os.path.join(PANAROO_RUN_ROOT, directory_leaf)
    if not os.path.isdir(panaroo_dir):
        raise FileNotFoundError(f"Panaroo directory not found: {panaroo_dir}")
    _set_log_path_root(panaroo_dir)

    analysis_dir = os.path.join(panaroo_dir, "analysis", "GPA_reference_genome")
    os.makedirs(analysis_dir, exist_ok=True)
    log_path = os.path.join(analysis_dir, f"clustering_log_{directory_leaf}.txt")
    log_fh = open(log_path, "w")
    log = _make_progress_logger(t0, log_fh, report_times=report_times)
    try:
        log(f"start pid={os.getpid()} directory_leaf={directory_leaf}")
        log(f"path context: input={_fmt_log_path(panaroo_dir)}")
        log(f"path context: analysis={_fmt_log_path(analysis_dir)}")

        _log_section(log, "METADATA")
        meta_df = _load_metadata(metadata_path, log)

        _log_section(log, "LOAD GPA")
        gpa_rtab = os.path.join(panaroo_dir, "gene_presence_absence.Rtab")
        if not os.path.isfile(gpa_rtab):
            raise FileNotFoundError(f"Required file not found: {gpa_rtab}")
        log(f"load GPA: reading {_fmt_log_path(gpa_rtab)}")
        gpa_df = pd.read_csv(gpa_rtab, sep="\t", index_col=0, dtype=str, low_memory=False)
        gpa_df.columns = gpa_df.columns.astype(str)
        gpa_df = (gpa_df == "1").astype(np.uint8)
        log(f"load GPA: {gpa_df.shape[0]} genes x {gpa_df.shape[1]} samples")

        _log_section(log, "KPSC FILTER + RUN CLASSIFICATION")
        gpa_df, n_gpa_samples_raw, n_kpsc_samples = _filter_gpa_to_kpsc(gpa_df, meta_df, log)
        if n_kpsc_samples == 0:
            raise ValueError("No kpsc_final_list=True samples remain after filtering.")
        run_meta = _classify_run(pd.Index(gpa_df.columns.astype(str)), meta_df, log)

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
            gpa_filter_cutoff
            if gpa_filter_cutoff is not None
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

        mgh_id = _mgh78578_sample_in_gpa(meta_df, pd.Index(gpa_df_filt.columns.astype(str)))
        if mgh_id is not None:
            gpa_df_stats = gpa_df_filt.drop(columns=[mgh_id], errors="ignore")
            log(f"mgh78578 (global ref) present in GPA ({mgh_id}); pangenome stats/plots exclude this sample")
        else:
            gpa_df_stats = gpa_df_filt

        _plot_gpa_distribution_and_log(
            gpa_df_stats,
            directory_leaf,
            os.path.join(analysis_dir, f"gpa_freq_dist_post_filter_{directory_leaf}.png"),
            log,
            shell_cloud_cutoff=shell_cloud_cutoff,
            core_shell_cutoff=core_shell_cutoff,
        )
        _plot_per_sample_category_counts(
            gpa_df_stats,
            directory_leaf,
            os.path.join(analysis_dir, f"gpa_core_softcore_shell_cloud_post_filter_{directory_leaf}.png"),
            log,
            shell_cloud_cutoff=shell_cloud_cutoff,
            core_shell_cutoff=core_shell_cutoff,
        )
        per_genome_category_stats = _compute_per_genome_category_stats(
            gpa_df_stats,
            shell_cloud_cutoff=shell_cloud_cutoff,
            core_shell_cutoff=core_shell_cutoff,
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
            int(merge_small_clusters)
            if merge_small_clusters is not None
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
            out_path=os.path.join(analysis_dir, f"umap_gpa_leiden_r{LEIDEN_RESOLUTION}_{directory_leaf}.png"),
            title=f"UMAP - GPA Leiden r={LEIDEN_RESOLUTION} - {directory_leaf}",
            log=log,
        )
        if "K_locus" in adata_gpa.obs.columns:
            _plot_umap_scatter(
                adata_gpa,
                color="K_locus",
                out_path=os.path.join(analysis_dir, f"umap_gpa_klocus_{directory_leaf}.png"),
                title=f"UMAP - Coloured by GPA K_locus - {directory_leaf}",
                log=log,
            )
        if "Clonal group" in adata_gpa.obs.columns:
            _plot_umap_scatter(
                adata_gpa,
                color="Clonal group",
                out_path=os.path.join(analysis_dir, f"umap_gpa_clonal_group_{directory_leaf}.png"),
                title=f"UMAP - Coloured by GPA Clonal Group - {directory_leaf}",
                log=log,
            )
        _plot_umap_reference_highlights(
            adata_gpa,
            out_path=os.path.join(analysis_dir, f"umap_gpa_refseq_{directory_leaf}.png"),
            title=f"UMAP - RefSeq and complete Norway - {directory_leaf}",
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

        _log_section(log, "MARKER GENES (GPA BY GPA CLUSTER)")
        _run_rank_genes_groups(adata_gpa, key, analysis_dir, directory_leaf, log)

        mean_genome_excl_mgh = float(gpa_df_stats.sum(axis=0).mean()) if gpa_df_stats.shape[1] else 0.0
        global_ref_stats = _global_reference_mgh78578_summary(
            adata_gpa,
            mgh_id,
            mean_genome_excl_mgh,
            log,
        )
        if mgh_id is not None:
            _log_mgh78578_genome_categories(
                gpa_df_stats,
                mgh_id,
                shell_cloud_cutoff=shell_cloud_cutoff,
                core_shell_cutoff=core_shell_cutoff,
                log=log,
            )

        refseq_stats = _refseq_summary_and_distances(
            adata_gpa, log, reference_top_n=reference_top_n
        )
        norway_stats = _norway_complete_summary_and_distances(
            adata_gpa, log, reference_top_n=reference_top_n
        )

        summary_df = pd.DataFrame(
            [
                {
                    "directory_leaf": directory_leaf,
                    "modality": "gpa",
                    "resolution": LEIDEN_RESOLUTION,
                    "n_samples": int(adata_gpa.n_obs),
                    "n_gpa_samples_raw": int(n_gpa_samples_raw),
                    "n_kpsc_samples": int(n_kpsc_samples),
                    "n_unique_sublineages": int(run_meta["n_unique_sublineages"]),
                    "sublineages_complete": bool(run_meta["sublineages_complete"]),
                    "n_unique_clonal_groups": int(run_meta["n_unique_clonal_groups"]),
                    "clonal_groups_complete": bool(run_meta["clonal_groups_complete"]),
                    "species": str(run_meta.get("species", "")),
                    "strain": str(run_meta.get("strain", "")),
                    "samples_in_strain": int(run_meta.get("samples_in_strain", 0)),
                    "run_classification": str(run_meta["run_classification"]),
                    "n_leiden_clusters": int(len(raw_counts)),
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
                    **per_genome_category_stats,
                    **global_ref_stats,
                    **refseq_stats,
                    **norway_stats,
                }
            ]
        )
        # Keep the compiled tables compact: round float columns to 2 dp.
        float_cols = [
            c for c in summary_df.columns if pd.api.types.is_float_dtype(summary_df[c])
        ]
        if float_cols:
            summary_df[float_cols] = summary_df[float_cols].round(2)
        p_out = os.path.join(analysis_dir, f"gpa_clustering_summary_{directory_leaf}.tsv")
        summary_df.to_csv(p_out, sep="\t", index=False)
        log(f"save: summary table -> {_fmt_log_path(p_out)}")

        total_wall = time.perf_counter() - t0
        _log_section(log, "FINAL SUMMARY")
        log(f"DONE total_wall={total_wall:.1f}s exit=0")
        result = summary_df.iloc[0].to_dict()
        result["status"] = "ok"
        return result
    except Exception as exc:
        total_wall = time.perf_counter() - t0
        _log_section(log, "FINAL SUMMARY")
        log(f"ERROR total_wall={total_wall:.1f}s error={exc}")
        return {
            "directory_leaf": directory_leaf,
            "status": "error",
            "error": str(exc),
        }
    finally:
        log_fh.close()


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
    p.add_argument(
        "--reference-top-n",
        type=int,
        default=10,
        help="Top-N reference genomes by lowest mean Jaccard to all samples (RefSeq and Norway).",
    )
    args = p.parse_args()

    result = run_gpa_analysis(
        directory_leaf=args.directory_leaf,
        metadata_path=args.metadata,
        gpa_filter_cutoff=args.gpa_filter_cutoff,
        merge_small_clusters=args.merge_small_clusters,
        shell_cloud_cutoff=args.shell_cloud_cutoff,
        core_shell_cutoff=args.core_shell_cutoff,
        report_times=args.report_times,
        reference_top_n=args.reference_top_n,
    )
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
