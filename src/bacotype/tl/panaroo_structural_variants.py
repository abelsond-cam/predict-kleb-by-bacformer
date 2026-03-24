"""Plots and summaries for Panaroo structural-variant presence/absence matrices (Rtab-style)."""

from __future__ import annotations

from typing import NamedTuple

import matplotlib.pyplot as plt
import pandas as pd


class StructuralVariantsPerSampleResult(NamedTuple):
    fig: plt.Figure
    genes_per_sample: pd.Series
    n_samples: int


def structural_variants_per_sample(
    struct_df: pd.DataFrame,
    strain: str,
    *,
    bins: int = 60,
    figsize: tuple[float, float] = (8, 4),
) -> StructuralVariantsPerSampleResult:
    """Per-sample burden: column sums, summary stats, histogram."""
    genes_per_sample = struct_df.sum(axis=0)
    n_samples = struct_df.shape[1]

    print("Present structural variants (panaroo structural variants) per sample (column sums)")
    print(f"  n_samples: {n_samples}")
    print(f"  min:  {int(genes_per_sample.min())} structural variants per sample")
    print(f"  max:  {int(genes_per_sample.max())} structural variants per sample")
    print(f"  mean: {genes_per_sample.mean():.2f} structural variants per sample")
    print(f"  sd:   {genes_per_sample.std(ddof=1):.2f} structural variants per sample")

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(genes_per_sample.to_numpy(), bins=bins, edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Present structural variants per sample")
    ax.set_ylabel("Number of samples")
    ax.set_title(f"Structural variant burden — {strain}")
    fig.tight_layout()

    return StructuralVariantsPerSampleResult(fig=fig, genes_per_sample=genes_per_sample, n_samples=n_samples)


def structural_variant_frequency_distribution(
    struct_df: pd.DataFrame,
    strain: str,
    *,
    bins: int = 100,
    figsize: tuple[float, float] = (10, 4),
    shell_low: float = 0.05,
    core_high: float = 0.95,
) -> plt.Figure:
    """Histogram of variant prevalence across samples; core/shell/cloud counts."""
    n_samples = struct_df.shape[1]
    n_variants = struct_df.shape[0]
    gene_freq = struct_df.sum(axis=1)
    gene_frac = gene_freq / n_samples

    pct_low = 100 * shell_low
    pct_high = 100 * core_high

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(gene_frac, bins=bins, edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Fraction of samples carrying the structural variant")
    ax.set_ylabel("Number of structural variants")
    ax.set_title(f"Structural variant frequency distribution — {strain} (n={n_variants})")
    ax.axvline(shell_low, color="red", ls="--", lw=1, label=f"{pct_low:g}%")
    ax.axvline(core_high, color="blue", ls="--", lw=1, label=f"{pct_high:g}%")
    ax.legend()
    fig.tight_layout()

    n_core = int((gene_frac > core_high).sum())
    n_shell = int(((gene_frac >= shell_low) & (gene_frac <= core_high)).sum())
    n_cloud = int((gene_frac < shell_low).sum())

    print(f"Total structural variants: {n_variants}")
    print(f"  Core  (>{pct_high:g}% samples):  {n_core:>6}  ({100 * n_core / n_variants:.1f}%)")
    print(f"  Shell ({pct_low:g}–{pct_high:g}%):         {n_shell:>6}  ({100 * n_shell / n_variants:.1f}%)")
    print(f"  Cloud (<{pct_low:g}%):           {n_cloud:>6}  ({100 * n_cloud / n_variants:.1f}%)")

    return fig


def per_sample_counts_core_shell_cloud_structural_variants(
    struct_df: pd.DataFrame,
    strain: str,
    gene_frac: pd.Series | None = None,
    *,
    bins: int = 50,
    figsize: tuple[float, float] = (14, 4),
    shell_low: float = 0.05,
    core_high: float = 0.95
) -> plt.Figure:
    """Per-sample counts split by core / shell / cloud (by prevalence thresholds)."""
    if gene_frac is None:
        gene_freq = struct_df.sum(axis=1)
        n_samples = struct_df.shape[1]
        gene_frac = gene_freq / n_samples

    gene_category = pd.Series("shell", index=struct_df.index, name="category")
    gene_category[gene_frac > core_high] = "core"
    gene_category[gene_frac < shell_low] = "cloud"

    core_mask = (gene_category == "core").values
    shell_mask = (gene_category == "shell").values
    cloud_mask = (gene_category == "cloud").values

    genes_per_sample = struct_df.sum(axis=0)
    sample_core = struct_df.loc[core_mask].sum(axis=0)
    sample_shell = struct_df.loc[shell_mask].sum(axis=0)
    sample_cloud = struct_df.loc[cloud_mask].sum(axis=0)

    per_sample = pd.DataFrame(
        {
            "core": sample_core,
            "shell": sample_shell,
            "cloud": sample_cloud,
            "total": genes_per_sample,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
    for ax, cat in zip(axes, ["core", "shell", "cloud"]):
        ax.hist(per_sample[cat], bins=bins, edgecolor="black", linewidth=0.3)
        ax.set_xlabel(f"# {cat} structural variants per sample")
        ax.set_ylabel("Samples")
        ax.set_title(f"{cat.capitalize()} SVs")
    axes[0].set_ylabel("Number of samples")
    fig.suptitle(f"Per-sample structural variant counts by category — {strain}", y=1.02)
    fig.tight_layout()

    return fig


def filter_structural_variants_by_prevalence(
    struct_df: pd.DataFrame,
    min_prevalence: int = 5,
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """Drop variants present in fewer than ``min_prevalence`` total number of samples."""
    n_samples = struct_df.shape[1]
    if n_samples == 0:
        return struct_df.copy()

    # Sum across samples for each variant (row)
    row_sum = struct_df.sum(axis=1)
    keep = row_sum >= min_prevalence
    out = struct_df.loc[keep].copy()
    if verbose:
        print(
            f"filter_structural_variants_by_prevalence: {struct_df.shape[0]} -> {out.shape[0]} variants "
            f"(min_prevalence={min_prevalence} samples)"
        )
    return out
