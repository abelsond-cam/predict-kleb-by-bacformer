"""Medoid-based Jaccard summaries and shared Jaccard → feature-count helpers.

Used by SV/GPA clustering quality and GPA-vs-SV-medoid cross-analysis so
reporting stays consistent.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

SECTION_BAR = "=" * 80


def jaccard_to_shared(d_j: float, mean_features: float) -> float:
    """Estimate shared feature count from Jaccard distance.

    For two genomes A, B with |A| ~ |B| ~ mean_features:
      shared = 2 * mean * sim / (1 + sim)  where sim = 1 - dJ.
    """
    if d_j >= 1.0:
        return 0.0
    sim = 1.0 - d_j
    return 2.0 * mean_features * sim / (1.0 + sim)


def medoid_metrics_from_dist_sq(
    dist_sq: np.ndarray,
    labels: np.ndarray | pd.Series,
    mean_features_per_genome: float,
) -> dict[str, Any]:
    """Medoid summaries from a Jaccard distance matrix and cluster labels.

    Global medoid minimizes mean Jaccard distance to all samples. Per-cluster
    medoid minimizes mean distance within the cluster. Single-member clusters
    use that sample as medoid (distance 0).
    """
    if isinstance(labels, pd.Series):
        lab = labels.astype(str).values
    else:
        lab = np.asarray(labels).astype(str)

    cluster_ids = sorted(np.unique(lab))
    per_cluster: list[dict[str, float | str | int]] = []
    sample_to_own = np.full(dist_sq.shape[0], np.nan, dtype=float)

    global_medoid_idx = int(np.argmin(dist_sq.mean(axis=1)))
    mean_global = float(dist_sq[:, global_medoid_idx].mean())

    for cid in cluster_ids:
        idxs = np.where(lab == cid)[0]
        n = int(len(idxs))
        if n == 0:
            continue
        if n == 1:
            m_row = int(idxs[0])
            sample_to_own[m_row] = 0.0
            per_cluster.append({
                "cluster_id": str(cid),
                "n": n,
                "mean_jaccard_to_cluster_medoid": 0.0,
            })
            continue

        sub_d = dist_sq[np.ix_(idxs, idxs)]
        medoid_local = int(np.argmin(sub_d.mean(axis=1)))
        medoid_g = int(idxs[medoid_local])
        d_to_m = dist_sq[idxs, medoid_g]
        sample_to_own[idxs] = d_to_m
        per_cluster.append({
            "cluster_id": str(cid),
            "n": n,
            "mean_jaccard_to_cluster_medoid": float(d_to_m.mean()),
        })

    finite = np.isfinite(sample_to_own)
    if not finite.any():
        return _empty_medoid_payload(
            mean_features_per_genome, per_cluster,
        )

    mean_own = float(sample_to_own[finite].mean())
    return _finalize_medoid_dict(
        per_cluster,
        mean_global,
        mean_own,
        mean_features_per_genome,
    )


def medoid_metrics_gpa_vs_sv_medoids(
    dist_sq_gpa: np.ndarray,
    subsample_labels: pd.Series | np.ndarray,
    cluster_to_medoid_row: dict[str, int],
    mean_features_per_genome: float,
) -> dict[str, Any]:
    """GPA Jaccard vs SV-derived medoid rows (indices into *dist_sq_gpa*)."""
    if isinstance(subsample_labels, pd.Series):
        lab = subsample_labels.astype(str).values
    else:
        lab = np.asarray(subsample_labels).astype(str)

    cluster_ids = sorted(np.unique(lab))
    per_cluster: list[dict[str, float | str | int]] = []
    n_sub = dist_sq_gpa.shape[0]
    d_own = np.full(n_sub, np.nan, dtype=float)

    for cid in cluster_ids:
        idxs = np.where(lab == cid)[0]
        n = int(len(idxs))
        if n == 0:
            continue
        m_row = cluster_to_medoid_row.get(str(cid))
        if m_row is None:
            continue
        block = dist_sq_gpa[idxs, m_row]
        d_own[idxs] = block
        per_cluster.append({
            "cluster_id": str(cid),
            "n": n,
            "mean_jaccard_to_cluster_medoid": float(block.mean()),
        })

    global_medoid_idx = int(np.argmin(dist_sq_gpa.mean(axis=1)))
    mean_global = float(dist_sq_gpa[:, global_medoid_idx].mean())

    finite = np.isfinite(d_own)
    if not finite.any():
        return _empty_medoid_payload(
            mean_features_per_genome, per_cluster,
        )

    mean_own = float(d_own[finite].mean())
    return _finalize_medoid_dict(
        per_cluster,
        mean_global,
        mean_own,
        mean_features_per_genome,
    )


def _finalize_medoid_dict(
    per_cluster: list[dict[str, float | str | int]],
    mean_global: float,
    mean_own: float,
    mean_feats: float,
) -> dict[str, Any]:
    gain_j = mean_global - mean_own
    gain_sim = (1.0 - mean_own) - (1.0 - mean_global)
    g_shared = jaccard_to_shared(mean_global, mean_feats)
    o_shared = jaccard_to_shared(mean_own, mean_feats)
    gain_feats = o_shared - g_shared
    g_pct = 100.0 * g_shared / mean_feats if mean_feats > 0 else 0.0
    o_pct = 100.0 * o_shared / mean_feats if mean_feats > 0 else 0.0
    gain_pct = o_pct - g_pct

    return {
        "per_cluster": per_cluster,
        "mean_features_per_genome": mean_feats,
        "global_medoid_jaccard_mean": mean_global,
        "own_cluster_medoid_jaccard_mean": mean_own,
        "gain_jaccard_b_minus_c": gain_j,
        "gain_similarity": gain_sim,
        "gain_shared_features_est": gain_feats,
        "gain_shared_features_pct_points": gain_pct,
        "global_medoid_shared_features_est": g_shared,
        "global_medoid_shared_features_pct": g_pct,
        "own_cluster_medoid_shared_features_est": o_shared,
        "own_cluster_medoid_shared_features_pct": o_pct,
    }


def _empty_medoid_payload(
    mean_feats: float,
    per_cluster: list[dict[str, float | str | int]],
) -> dict[str, Any]:
    nan = np.nan
    return {
        "per_cluster": per_cluster,
        "mean_features_per_genome": mean_feats,
        "global_medoid_jaccard_mean": nan,
        "own_cluster_medoid_jaccard_mean": nan,
        "gain_jaccard_b_minus_c": nan,
        "gain_similarity": nan,
        "gain_shared_features_est": nan,
        "gain_shared_features_pct_points": nan,
        "global_medoid_shared_features_est": nan,
        "global_medoid_shared_features_pct": nan,
        "own_cluster_medoid_shared_features_est": nan,
        "own_cluster_medoid_shared_features_pct": nan,
    }


def log_medoid_report(
    log: Callable[[str], None],
    step_title: str,
    feature_label: str,
    metrics: dict[str, Any],
) -> None:
    """One format for SV, GPA, and GPA-vs-SV medoid blocks."""
    fl = feature_label
    log(SECTION_BAR)
    log(step_title)
    log(SECTION_BAR)

    for row in metrics.get("per_cluster", []):
        log(
            f"Cluster {row['cluster_id']}  n={row['n']}  "
            f"mean Jaccard to cluster medoid = {row['mean_jaccard_to_cluster_medoid']:.4f}"
        )

    mg = metrics.get("global_medoid_jaccard_mean", np.nan)
    mo = metrics.get("own_cluster_medoid_jaccard_mean", np.nan)
    if np.isfinite(mg):
        log(f"Mean Jaccard (sample → global medoid) = {mg:.4f}")
    else:
        log("Mean Jaccard (sample → global medoid) = not available")

    if np.isfinite(mo):
        log(f"Mean Jaccard (sample → own cluster medoid) = {mo:.4f}")
    else:
        log("Mean Jaccard (sample → own cluster medoid) = not available")

    gj = metrics.get("gain_jaccard_b_minus_c", np.nan)
    gs = metrics.get("gain_similarity", np.nan)
    if np.isfinite(gj):
        log(
            f"Improvement: Δ Jaccard (global − own cluster medoid) = {gj:.4f}; "
            f"Δ similarity (Jaccard) = {gs:.4f}"
        )
    else:
        log("Improvement: not available")

    gfe = metrics.get("gain_shared_features_est", np.nan)
    if np.isfinite(gfe):
        log(
            f"Estimated extra shared {fl}s vs global medoid: {gfe:.0f} "
            f"({metrics['gain_shared_features_pct_points']:.1f} percentage points "
            f"of mean genome {fl}s)"
        )
        log(
            f"Shared {fl}s (est.): {metrics['global_medoid_shared_features_est']:.0f} "
            f"({metrics['global_medoid_shared_features_pct']:.1f}%) → "
            f"{metrics['own_cluster_medoid_shared_features_est']:.0f} "
            f"({metrics['own_cluster_medoid_shared_features_pct']:.1f}%)"
        )
    elif np.isfinite(mo):
        log(f"Estimated extra shared {fl}s: not available")


def log_resolution_one_comparison(
    log: Callable[[str], None],
    *,
    metrics_sv: dict[str, Any] | None,
    metrics_gpa_vs_sv: dict[str, Any] | None,
    metrics_gpa: dict[str, Any] | None,
    resolution_sv_gpa: float,
    resolution_gpa_sv_cross: float,
    feature_label_sv: str,
    feature_label_gpa: str,
) -> None:
    log(SECTION_BAR)
    log(
        "COMPARISON: medoid gains — "
        f"SV & GPA clustering @ r={resolution_sv_gpa:g}; "
        f"GPA vs SV medoids @ SV r={resolution_gpa_sv_cross:g}"
    )
    log(SECTION_BAR)

    def _line(label: str, m: dict[str, Any] | None, fl: str) -> None:
        if m is None or not np.isfinite(m.get("gain_jaccard_b_minus_c", np.nan)):
            log(f"{label}: not available")
            return
        log(
            f"{label}: Δ Jaccard = {m['gain_jaccard_b_minus_c']:.4f}, "
            f"Δ similarity = {m['gain_similarity']:.4f}, "
            f"extra shared {fl}s (est.) ≈ {m['gain_shared_features_est']:.0f}"
        )

    _line(
        f"SV clustering (SV space, r={resolution_sv_gpa:g})",
        metrics_sv,
        feature_label_sv,
    )
    _line(
        f"GPA vs SV medoids (GPA space, SV labels r={resolution_gpa_sv_cross:g})",
        metrics_gpa_vs_sv,
        feature_label_gpa,
    )
    _line(
        f"GPA clustering (GPA space, r={resolution_sv_gpa:g})",
        metrics_gpa,
        feature_label_gpa,
    )
