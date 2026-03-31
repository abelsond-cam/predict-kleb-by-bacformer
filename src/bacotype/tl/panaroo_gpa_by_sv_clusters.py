"""Cross-analysis: GPA homogeneity relative to SV medoids + marker genes.

Uses SV-derived cluster medoids (in SV space), reports GPA Jaccard from each
sample to its cluster SV medoid vs global GPA medoid (same metrics as the
clustering pipeline), and runs scanpy rank_genes_groups (Wilcoxon).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.spatial.distance import pdist, squareform

from bacotype.tl.panaroo_jaccard_medoid_metrics import (
    log_medoid_report,
    medoid_metrics_gpa_vs_sv_medoids,
)

if TYPE_CHECKING:
    from collections.abc import Callable

QUALITY_SUBSAMPLE_THRESHOLD = 2000
MIN_PER_CLUSTER = 50
_SECTION_BAR = "=" * 80
_LOG_PATH_ROOT: str | None = None


def set_log_path_root(strain_dir: str) -> None:
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


def _stratified_subsample(
    labels: pd.Series,
    max_n: int,
    min_per_cluster: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return sorted indices for stratified subsampling (same as clustering)."""
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


def _log_section(log: Callable[[str], None], title: str) -> None:
    log(_SECTION_BAR)
    log(title)
    log(_SECTION_BAR)


def _sv_cluster_medoid_rows(
    dist_sq_sv: np.ndarray,
    sub_labels: pd.Series,
) -> dict[str, int]:
    """Map cluster id -> subsample row index of SV medoid for that cluster."""
    cluster_ids = sorted(sub_labels.unique())
    out: dict[str, int] = {}
    lab = sub_labels.astype(str).values

    for cid in cluster_ids:
        idxs = np.where(lab == cid)[0]
        if len(idxs) == 0:
            continue
        if len(idxs) == 1:
            out[str(cid)] = int(idxs[0])
            continue
        sub_d = dist_sq_sv[np.ix_(idxs, idxs)]
        medoid_local = int(np.argmin(sub_d.mean(axis=1)))
        out[str(cid)] = int(idxs[medoid_local])
    return out


def _run_rank_genes_groups(
    adata_gpa: ad.AnnData,
    sv_labels: pd.Series,
    out_dir: str,
    strain: str,
    resolution: float,
    log: Callable[[str], None],
    n_genes: int = 20,
) -> None:
    """Wilcoxon rank_genes_groups on GPA grouped by SV cluster, save plots + TSV."""
    unique_labels = sv_labels.astype(str).unique()
    if len(unique_labels) < 2:
        log("rank_genes_groups: only 1 SV cluster — skipping marker analysis")
        return

    rg_adata = adata_gpa.copy()
    rg_adata.obs["sv_cluster"] = sv_labels.astype(str).values
    # scanpy plotting fails when keys are present in both obs columns and var_names.
    # Keep group label and drop only conflicting metadata columns in the plotting copy.
    var_name_set = set(rg_adata.var_names.astype(str))
    collisions = [
        col for col in rg_adata.obs.columns
        if col != "sv_cluster" and str(col) in var_name_set
    ]
    if collisions:
        rg_adata.obs = rg_adata.obs.drop(columns=collisions)
        show = ", ".join(collisions[:5])
        suffix = "..." if len(collisions) > 5 else ""
        log(
            f"rank_genes_groups: dropped {len(collisions)} obs columns "
            f"that overlap gene names ({show}{suffix})"
        )
    log(f"rank_genes_groups: groupby=sv_cluster, method=wilcoxon, n_genes={n_genes}")
    sc.tl.rank_genes_groups(rg_adata, groupby="sv_cluster", method="wilcoxon")

    res_tag = f"sv_r{resolution}_{strain}"
    old_figdir = sc.settings.figdir
    sc.settings.figdir = out_dir

    plots = [
        ("heatmap", sc.pl.rank_genes_groups_heatmap, {"n_genes": n_genes, "show_gene_labels": True}),
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


def run_gpa_by_sv_cross_analysis(
    adata_sv: ad.AnnData,
    adata_gpa: ad.AnnData,
    strain: str,
    out_dir: str,
    resolution: float,
    log: Callable[[str], None],
    max_subsample: int = QUALITY_SUBSAMPLE_THRESHOLD,
    min_per_cluster: int = MIN_PER_CLUSTER,
    n_genes: int = 20,
) -> tuple[dict | None, ad.AnnData]:
    """GPA Jaccard vs SV medoids + marker genes.

    Returns
    -------
    (medoid_metrics | None, adata_gpa_to_use)
        Second value is intersection-aligned with ``adata_sv`` when names
        differ; use it for MuData so modalities share the same observations.
    """
    os.makedirs(out_dir, exist_ok=True)

    sv_key = f"sv_leiden_r{resolution}"
    if sv_key not in adata_sv.obs.columns:
        log(f"gpa_by_sv: SV label key '{sv_key}' not found — skipping")
        return None, adata_gpa

    if not adata_gpa.obs_names.equals(adata_sv.obs_names):
        log("gpa_by_sv: WARNING — obs_names differ; intersecting + ordering to SV")
        common = adata_sv.obs_names.intersection(adata_gpa.obs_names)
        adata_gpa = adata_gpa[common].copy()
        adata_sv_use = adata_sv[adata_gpa.obs_names].copy()
    else:
        adata_sv_use = adata_sv

    sv_labels_full = adata_sv_use.obs[sv_key].astype(str)
    n = adata_gpa.n_obs
    rng = np.random.default_rng(42)

    if n <= max_subsample:
        sub_idx = np.arange(n, dtype=int)
        X_sv = adata_sv_use.X.toarray().astype(np.uint8, copy=False)
        X_gpa = adata_gpa.X.toarray().astype(np.uint8, copy=False)
        sub_labels = sv_labels_full.copy()
        log(f"gpa_by_sv: SV medoids + GPA Jaccard on all {n} samples")
    else:
        sub_idx = _stratified_subsample(
            sv_labels_full, max_subsample, min_per_cluster, rng,
        )
        X_sv = adata_sv_use.X[sub_idx].toarray().astype(np.uint8, copy=False)
        X_gpa = adata_gpa.X[sub_idx].toarray().astype(np.uint8, copy=False)
        sub_labels = sv_labels_full.iloc[sub_idx].reset_index(drop=True)
        log(
            f"gpa_by_sv: stratified subsample {len(sub_idx)}/{n} "
            f"(min {min_per_cluster}/cluster)"
        )

    mean_genes = float(adata_gpa.X.toarray().astype(np.uint8).sum(axis=1).mean())

    dist_sq_sv = squareform(pdist(X_sv, metric="jaccard"))
    dist_sq_gpa = squareform(pdist(X_gpa, metric="jaccard"))
    log(f"gpa_by_sv: SV + GPA pdist done (subsample size {X_gpa.shape[0]})")

    med_map = _sv_cluster_medoid_rows(dist_sq_sv, sub_labels)
    metrics = medoid_metrics_gpa_vs_sv_medoids(
        dist_sq_gpa,
        sub_labels,
        med_map,
        mean_genes,
    )

    log_medoid_report(
        log,
        f"GPA vs SV medoids — SV Leiden r={resolution}",
        "gene",
        metrics,
    )

    _log_section(log, "Marker genes (GPA by SV cluster)")
    _run_rank_genes_groups(
        adata_gpa, sv_labels_full, out_dir, strain, resolution, log,
        n_genes=n_genes,
    )

    return metrics, adata_gpa
