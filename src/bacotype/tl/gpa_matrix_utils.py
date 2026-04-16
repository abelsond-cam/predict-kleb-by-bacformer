"""Plots and summaries for Panaroo presence/absence matrices (Rtab-style).

Works for both structural-variant and gene presence/absence tables.
The ``feature_label`` parameter controls axis labels, titles, and print output
(e.g. ``"structural variant"`` or ``"gene"``).
"""

from __future__ import annotations

from typing import NamedTuple

import matplotlib.pyplot as plt
import pandas as pd


class FeaturePerSampleResult(NamedTuple):
    """Return type for :func:`features_per_sample`."""

    fig: plt.Figure
    counts_per_sample: pd.Series
    n_samples: int


def features_per_sample(
    df: pd.DataFrame,
    strain: str,
    feature_label: str = "structural variant",
    *,
    bins: int = 60,
    figsize: tuple[float, float] = (8, 4),
) -> FeaturePerSampleResult:
    """Per-sample burden: column sums, summary stats, histogram."""
    counts_per_sample = df.sum(axis=0)
    n_samples = df.shape[1]
    fl = feature_label  # short alias for print lines

    print(f"Present {fl}s (panaroo) per sample (column sums)")
    print(f"  n_samples: {n_samples}")
    print(f"  min:  {int(counts_per_sample.min())} {fl}s per sample")
    print(f"  max:  {int(counts_per_sample.max())} {fl}s per sample")
    print(f"  mean: {counts_per_sample.mean():.2f} {fl}s per sample")
    print(f"  sd:   {counts_per_sample.std(ddof=1):.2f} {fl}s per sample")

    fl_cap = fl.capitalize()
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(counts_per_sample.to_numpy(), bins=bins, edgecolor="black", linewidth=0.3)
    ax.set_xlabel(f"Present {fl}s per sample")
    ax.set_ylabel("Number of samples")
    ax.set_title(f"{fl_cap} burden — {strain}")
    fig.tight_layout()

    return FeaturePerSampleResult(fig=fig, counts_per_sample=counts_per_sample, n_samples=n_samples)


def feature_frequency_distribution(
    df: pd.DataFrame,
    strain: str,
    feature_label: str = "structural variant",
    *,
    bins: int = 100,
    figsize: tuple[float, float] = (10, 4),
    shell_low: float = 0.05,
    core_high: float = 0.95,
) -> plt.Figure:
    """Histogram of feature prevalence across samples; core/shell/cloud counts."""
    n_samples = df.shape[1]
    n_features = df.shape[0]
    freq = df.sum(axis=1)
    frac = freq / n_samples
    fl = feature_label
    fl_cap = fl.capitalize()

    pct_low = 100 * shell_low
    pct_high = 100 * core_high
    panaroo_cutoff = 0.01

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(frac, bins=bins, edgecolor="black", linewidth=0.3)
    ax.set_xlabel(f"Fraction of samples carrying the {fl}")
    ax.set_ylabel(f"Number of {fl}s")
    ax.set_title(f"{fl_cap} frequency distribution — {strain} (n={n_features})")
    ax.axvline(shell_low, color="red", ls="--", lw=1, label=f"{pct_low:g}%")
    ax.axvline(core_high, color="blue", ls="--", lw=1, label=f"{pct_high:g}%")
    ax.legend()
    fig.tight_layout()

    n_core = int((frac > core_high).sum())
    n_shell = int(((frac >= shell_low) & (frac <= core_high)).sum())
    n_cloud = int((frac < shell_low).sum())
    n_panaroo_cutoff = int((frac < panaroo_cutoff).sum())
    n_ubiquitous = int((frac == 1).sum())

    print(f"Total {fl}s: {n_features}")
    print(f"  Core  (>{pct_high:g}% samples):  {n_core:>6}  ({100 * n_core / n_features:.1f}%)")
    print(f"  Shell ({pct_low:g}–{pct_high:g}%):         {n_shell:>6}  ({100 * n_shell / n_features:.1f}%)")
    print(f"  Cloud (<{pct_low:g}%):           {n_cloud:>6}  ({100 * n_cloud / n_features:.1f}%)")
    print(
        f"  Panaroo cutoff (<{panaroo_cutoff * 100:g}%):     {n_panaroo_cutoff:>6}  ({100 * n_panaroo_cutoff / n_features:.1f}%)"
    )
    print(f"  Ubiquitous (100%):              {n_ubiquitous:>6}  ({100 * n_ubiquitous / n_features:.1f}%)")
    return fig


def per_sample_counts_core_shell_cloud(
    df: pd.DataFrame,
    strain: str,
    feature_label: str = "structural variant",
    gene_frac: pd.Series | None = None,
    *,
    bins: int = 50,
    figsize: tuple[float, float] = (14, 4),
    shell_low: float = 0.05,
    core_high: float = 0.95,
) -> plt.Figure:
    """Per-sample counts split by core / shell / cloud (by prevalence thresholds)."""
    fl = feature_label
    fl_short = "SVs" if "structural" in fl.lower() else "genes"

    if gene_frac is None:
        freq = df.sum(axis=1)
        n_samples = df.shape[1]
        gene_frac = freq / n_samples

    category = pd.Series("shell", index=df.index, name="category")
    category[gene_frac > core_high] = "core"
    category[gene_frac < shell_low] = "cloud"

    core_mask = (category == "core").values
    shell_mask = (category == "shell").values
    cloud_mask = (category == "cloud").values

    total = df.sum(axis=0)
    sample_core = df.loc[core_mask].sum(axis=0)
    sample_shell = df.loc[shell_mask].sum(axis=0)
    sample_cloud = df.loc[cloud_mask].sum(axis=0)

    per_sample = pd.DataFrame(
        {
            "core": sample_core,
            "shell": sample_shell,
            "cloud": sample_cloud,
            "total": total,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
    for ax, cat in zip(axes, ["core", "shell", "cloud"]):
        ax.hist(per_sample[cat], bins=bins, edgecolor="black", linewidth=0.3)
        ax.set_xlabel(f"# {cat} {fl}s per sample")
        ax.set_ylabel("Samples")
        ax.set_title(f"{cat.capitalize()} {fl_short}")
    axes[0].set_ylabel("Number of samples")
    fig.suptitle(f"Per-sample {fl} counts by category — {strain}", y=1.02)
    fig.tight_layout()

    return fig


def filter_by_prevalence(
    df: pd.DataFrame,
    min_prevalence: int = 5,
    feature_label: str = "structural variant",
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """Drop features present in fewer than ``min_prevalence`` total samples."""
    n_samples = df.shape[1]
    if n_samples == 0:
        return df.copy()

    row_sum = df.sum(axis=1)
    keep = row_sum >= min_prevalence
    out = df.loc[keep].copy()
    if verbose:
        print(
            f"filter_by_prevalence ({feature_label}): "
            f"{df.shape[0]} -> {out.shape[0]} features "
            f"(min_prevalence={min_prevalence} samples)"
        )
    return out
