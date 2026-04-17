#!/usr/bin/env python3
r"""
Stratified GPA distance orchestrator for a single Panaroo run.

Loads ``gene_presence_absence.csv`` from a Panaroo output directory as a
uint16 gene-count matrix, applies KPSC and (binary-view) prevalence filters,
then runs the per-group analysis
(``gpa_distances_single_group.run_gpa_analysis``) on:

1. The whole set (one row, ``group_level='whole_set'``).
2. Each major Clonal group (count >= ``min_group_size``) plus a pooled
   ``other`` slice (``group_level='clonal_group'``). Reference genomes
   (mgh78578, RefSeq, and complete Norway) are added back to each slice so
   they are always available for distance comparisons.
3. Within each major Clonal group (not ``other``), each major K_locus
   (count >= ``min_group_size``) plus a pooled ``<CG>_other`` slice
   (``group_level='cg_klocus'``). Reference genomes are again added back.

Writes a single detail TSV at
``<panaroo_dir>/analysis/GPA_reference_genome/gpa_distances_detail_<run_label>.tsv``.
Returns the whole-set summary row (augmented with ``detail_tsv_path``) for
batch-runner compatibility.
"""

from __future__ import annotations

import argparse
import os
import re
import time

import numpy as np
import pandas as pd

from bacotype.tl.gpa_distances_single_group import (
    DEFAULT_METADATA_PATH,
    PANAROO_RUN_ROOT,
    _classify_run,
    _default_filter_cutoff,
    _filter_gpa_to_kpsc,
    _load_gff_feature_counts,
    _load_gpa_counts_from_csv,
    _load_metadata,
    _series_to_bool,
    _str2bool,
)
from bacotype.tl.gpa_distances_single_group import (
    run_gpa_analysis as run_single_group_analysis,
)
from bacotype.tl.gpa_matrix_utils import filter_by_prevalence

DEFAULT_MIN_GROUP_SIZE = 250
DEFAULT_GFF_FEATURE_COUNTS_PATH = (
    "/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/final/gff_feature_counts.tsv"
)


def _identify_reference_sample_ids(
    gpa_sample_ids: pd.Index,
    meta_df: pd.DataFrame,
) -> set[str]:
    """Return sample IDs whose metadata marks them as a reference genome.

    A sample counts as a reference if any of ``is_mgh78578``, ``is_refseq``,
    or ``is_complete_norway_genome`` is True. Only IDs that appear in
    ``gpa_sample_ids`` are returned.
    """
    result: set[str] = set()
    sid = pd.Index(gpa_sample_ids.astype(str))
    reindexed = meta_df.reindex(sid)
    for col in ("is_mgh78578", "is_refseq", "is_complete_norway_genome"):
        if col not in reindexed.columns:
            continue
        mask = _series_to_bool(reindexed[col]).to_numpy()
        result.update(sid[mask].astype(str).tolist())
    return result


def _split_groups(
    meta_for_samples: pd.DataFrame,
    col: str,
    min_group_size: int,
    other_label: str = "other",
) -> list[tuple[str, list[str]]]:
    """Split samples by metadata column value.

    Groups with count >= ``min_group_size`` keep their own entry. All other
    samples (including those with missing/empty values) are pooled into a
    single ``(other_label, combined_ids)`` tuple. Major groups are ordered
    by descending size; ``other`` comes last.
    """
    series = meta_for_samples[col]
    series_str = series.astype(str)
    missing_mask = series.isna() | series_str.isin({"", "nan", "None", "NaN"})
    value_counts = series_str[~missing_mask].value_counts()

    major_vals = value_counts[value_counts >= min_group_size].index.tolist()
    minor_set = set(value_counts[value_counts < min_group_size].index.tolist())

    groups: list[tuple[str, list[str]]] = []
    for val in major_vals:
        ids = (
            meta_for_samples.index[(~missing_mask) & (series_str == val)]
            .astype(str)
            .tolist()
        )
        groups.append((str(val), ids))
    other_mask = missing_mask | series_str.isin(minor_set)
    other_ids = meta_for_samples.index[other_mask].astype(str).tolist()
    if other_ids:
        groups.append((other_label, other_ids))
    return groups


def _build_group_gpa_df(
    gpa_df_filt: pd.DataFrame,
    group_ids: list[str] | set[str],
    ref_ids: set[str],
) -> pd.DataFrame:
    """Return ``gpa_df_filt`` restricted to columns in group_ids ∪ ref_ids."""
    wanted = set(map(str, group_ids)) | set(map(str, ref_ids))
    cols = [c for c in gpa_df_filt.columns.astype(str) if c in wanted]
    return gpa_df_filt.loc[:, cols]


def _sanitize_label(label: str) -> str:
    """Make a label safe for use as a filesystem component."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(label))


def _subset_analysis_dir(main_analysis_dir: str, group_label: str) -> str:
    return os.path.join(main_analysis_dir, "groups", _sanitize_label(group_label))


def _inject_run_metadata(
    row: dict[str, object],
    *,
    n_gpa_samples_raw: int,
    n_kpsc_samples: int,
    run_meta: dict[str, object],
) -> dict[str, object]:
    """Overlay whole-run context onto a summary row (mutates and returns)."""
    row["n_gpa_samples_raw"] = int(n_gpa_samples_raw)
    row["n_kpsc_samples"] = int(n_kpsc_samples)
    row["n_unique_sublineages"] = int(run_meta["n_unique_sublineages"])
    row["sublineages_complete"] = bool(run_meta["sublineages_complete"])
    row["n_unique_clonal_groups"] = int(run_meta["n_unique_clonal_groups"])
    row["clonal_groups_complete"] = bool(run_meta["clonal_groups_complete"])
    row["species"] = str(run_meta.get("species", ""))
    row["strain"] = str(run_meta.get("strain", ""))
    row["samples_in_strain"] = int(run_meta.get("samples_in_strain", 0))
    row["run_classification"] = str(run_meta["run_classification"])
    return row


def run_gpa_analysis(
    directory_leaf: str | None = None,
    panaroo_dir: str | None = None,
    panaroo_run_root: str = PANAROO_RUN_ROOT,
    metadata_path: str = DEFAULT_METADATA_PATH,
    gff_feature_counts_path: str = DEFAULT_GFF_FEATURE_COUNTS_PATH,
    min_group_size: int = DEFAULT_MIN_GROUP_SIZE,
    gpa_filter_cutoff: int | None = None,
    merge_small_clusters: int | None = None,
    shell_cloud_cutoff: float = 0.15,
    core_shell_cutoff: float = 0.95,
    report_times: bool = False,
    reference_top_n: int = 10,
) -> dict[str, object]:
    """Run stratified GPA distance analysis for a single Panaroo run.

    Returns the whole-set summary row (augmented with
    ``detail_tsv_path``) for batch-runner compatibility. All rows
    (whole set, per-CG, per-CG+K_locus) are written to a single detail TSV.
    """
    if (directory_leaf is not None) == (panaroo_dir is not None):
        raise ValueError(
            "Provide exactly one of directory_leaf (under panaroo_run_root) or panaroo_dir."
        )
    if min_group_size < 1:
        raise ValueError("min_group_size must be >= 1")

    t0 = time.perf_counter()

    if panaroo_dir is not None:
        panaroo_dir = os.path.abspath(panaroo_dir)
        run_label = os.path.basename(os.path.normpath(panaroo_dir))
    else:
        assert directory_leaf is not None
        panaroo_dir = os.path.join(panaroo_run_root, directory_leaf)
        run_label = directory_leaf

    if not os.path.isdir(panaroo_dir):
        raise FileNotFoundError(f"Panaroo directory not found: {panaroo_dir}")

    main_analysis_dir = os.path.join(panaroo_dir, "analysis", "GPA_reference_genome")
    os.makedirs(main_analysis_dir, exist_ok=True)

    orch_log_path = os.path.join(main_analysis_dir, f"orchestrator_log_{run_label}.txt")
    detail_path = os.path.join(main_analysis_dir, f"gpa_distances_detail_{run_label}.tsv")
    whole_row: dict[str, object] = {}

    with open(orch_log_path, "w") as orch_fh:

        def olog(msg: str) -> None:
            print(msg, flush=True)
            orch_fh.write(msg + "\n")
            orch_fh.flush()

        def _helper_log(msg: str) -> None:
            olog(f"  [prep] {msg}")

        try:
            olog(f"orchestrator start pid={os.getpid()} run_label={run_label}")
            olog(f"panaroo_dir={panaroo_dir}")
            olog(f"min_group_size={min_group_size}")

            meta_df = _load_metadata(metadata_path, _helper_log)
            gff_counts_df = _load_gff_feature_counts(
                gff_feature_counts_path, _helper_log
            )

            gpa_csv = os.path.join(panaroo_dir, "gene_presence_absence.csv")
            if not os.path.isfile(gpa_csv):
                raise FileNotFoundError(f"Required file not found: {gpa_csv}")
            olog(f"loading GPA: {gpa_csv}")
            gpa_counts_df_raw = _load_gpa_counts_from_csv(gpa_csv, _helper_log)
            olog(
                f"GPA raw: {gpa_counts_df_raw.shape[0]} genes x "
                f"{gpa_counts_df_raw.shape[1]} samples"
            )

            gpa_counts_df_kpsc, n_gpa_samples_raw, n_kpsc_samples = _filter_gpa_to_kpsc(
                gpa_counts_df_raw, meta_df, _helper_log
            )
            if n_kpsc_samples == 0:
                raise ValueError(
                    "No kpsc_final_list=True samples remain after filtering."
                )
            del gpa_counts_df_raw

            run_meta = _classify_run(
                pd.Index(gpa_counts_df_kpsc.columns.astype(str)), meta_df, _helper_log
            )

            n_samples_whole = int(gpa_counts_df_kpsc.shape[1])
            filter_cutoff = (
                int(gpa_filter_cutoff)
                if gpa_filter_cutoff is not None
                else _default_filter_cutoff(n_samples_whole)
            )
            gpa_df_kpsc_bin = (gpa_counts_df_kpsc > 0).astype(np.uint8)
            gpa_df_filt_bin = filter_by_prevalence(
                gpa_df_kpsc_bin, min_prevalence=filter_cutoff, feature_label="gene"
            )
            gpa_df_filt = gpa_counts_df_kpsc.loc[gpa_df_filt_bin.index]
            olog(
                f"prevalence filter (cutoff={filter_cutoff}): "
                f"{gpa_counts_df_kpsc.shape[0]} -> {gpa_df_filt.shape[0]} genes"
            )
            del gpa_counts_df_kpsc, gpa_df_kpsc_bin, gpa_df_filt_bin

            olog("=== running whole-set analysis ===")
            whole_member_ids = pd.Index(gpa_df_filt.columns.astype(str))
            whole_row = run_single_group_analysis(
                gpa_df=gpa_df_filt,
                meta_df=meta_df,
                gff_counts_df=gff_counts_df,
                group_label=run_label,
                analysis_dir=main_analysis_dir,
                group_level="whole_set",
                parent_group="",
                member_sample_ids=whole_member_ids,
                gpa_filter_cutoff=filter_cutoff,
                merge_small_clusters=merge_small_clusters,
                shell_cloud_cutoff=shell_cloud_cutoff,
                core_shell_cutoff=core_shell_cutoff,
                report_times=report_times,
                reference_top_n=reference_top_n,
            )
            if whole_row.get("status") == "ok":
                _inject_run_metadata(
                    whole_row,
                    n_gpa_samples_raw=n_gpa_samples_raw,
                    n_kpsc_samples=n_kpsc_samples,
                    run_meta=run_meta,
                )
            whole_row["directory_leaf"] = run_label
            rows: list[dict[str, object]] = [whole_row]

            filt_sids = pd.Index(gpa_df_filt.columns.astype(str))
            ref_ids = _identify_reference_sample_ids(filt_sids, meta_df)
            olog(
                f"reference genome IDs preserved in subsets: n={len(ref_ids)} "
                f"(mgh78578 + RefSeq + complete Norway)"
            )

            filt_cols_set = set(filt_sids.astype(str))
            assert ref_ids <= filt_cols_set, (
                "ref_ids contains IDs not present in gpa_df_filt columns: "
                f"{sorted(ref_ids - filt_cols_set)[:5]}"
            )

            meta_for_samples = meta_df.reindex(filt_sids).copy()

            cg_groups: list[tuple[str, list[str]]] = []
            if "Clonal group" in meta_for_samples.columns:
                cg_groups = _split_groups(
                    meta_for_samples,
                    "Clonal group",
                    min_group_size,
                    other_label="other",
                )
                olog(
                    f"Clonal group split: {len(cg_groups)} slices "
                    f"({', '.join(f'{n}:{len(i)}' for n, i in cg_groups)})"
                )
            else:
                olog("skipping Clonal group split ('Clonal group' column missing)")

            for cg_name, cg_ids in cg_groups:
                group_label = _sanitize_label(cg_name)
                cg_id_set = set(map(str, cg_ids))
                assert cg_id_set <= filt_cols_set, (
                    f"CG slice {group_label}: cg_ids not subset of gpa_df_filt columns: "
                    f"{sorted(cg_id_set - filt_cols_set)[:5]}"
                )
                belonging_refs = sorted(cg_id_set & ref_ids)
                nonbelonging_refs = sorted(ref_ids - cg_id_set)
                olog(
                    f"=== CG subset: {group_label} "
                    f"(n_group={len(cg_ids)}, +refs={len(ref_ids)}) ==="
                )
                olog(
                    f"  {group_label}: members (stats) n={len(cg_ids)} "
                    f"(belonging refs n={len(belonging_refs)}); "
                    f"non-belonging refs n={len(nonbelonging_refs)} "
                    f"(kept in Jaccard/UMAP only)"
                )
                if belonging_refs:
                    olog(
                        f"    belonging refs: {', '.join(belonging_refs[:10])}"
                        + (" ..." if len(belonging_refs) > 10 else "")
                    )
                if nonbelonging_refs:
                    olog(
                        f"    non-belonging refs: {', '.join(nonbelonging_refs[:10])}"
                        + (" ..." if len(nonbelonging_refs) > 10 else "")
                    )
                subset_df = _build_group_gpa_df(gpa_df_filt, cg_ids, ref_ids)
                subset_dir = _subset_analysis_dir(main_analysis_dir, group_label)
                row = run_single_group_analysis(
                    gpa_df=subset_df,
                    meta_df=meta_df,
                    gff_counts_df=gff_counts_df,
                    group_label=group_label,
                    analysis_dir=subset_dir,
                    group_level="clonal_group",
                    parent_group="",
                    member_sample_ids=pd.Index(sorted(cg_id_set)),
                    gpa_filter_cutoff=filter_cutoff,
                    merge_small_clusters=merge_small_clusters,
                    shell_cloud_cutoff=shell_cloud_cutoff,
                    core_shell_cutoff=core_shell_cutoff,
                    report_times=report_times,
                    reference_top_n=reference_top_n,
                )
                row["directory_leaf"] = run_label
                rows.append(row)

            if "K_locus" in meta_for_samples.columns:
                for cg_name, cg_ids in cg_groups:
                    if cg_name == "other":
                        continue
                    cg_meta = meta_for_samples.reindex(cg_ids)
                    kl_groups = _split_groups(
                        cg_meta,
                        "K_locus",
                        min_group_size,
                        other_label="other",
                    )
                    olog(
                        f"CG={cg_name} K_locus split: {len(kl_groups)} slices "
                        f"({', '.join(f'{n}:{len(i)}' for n, i in kl_groups)})"
                    )
                    for kl_name, kl_ids in kl_groups:
                        group_label = (
                            f"{_sanitize_label(cg_name)}"
                            f"_{_sanitize_label(kl_name)}"
                        )
                        kl_id_set = set(map(str, kl_ids))
                        assert kl_id_set <= filt_cols_set, (
                            f"CG+KL slice {group_label}: kl_ids not subset of "
                            f"gpa_df_filt columns: {sorted(kl_id_set - filt_cols_set)[:5]}"
                        )
                        belonging_refs = sorted(kl_id_set & ref_ids)
                        nonbelonging_refs = sorted(ref_ids - kl_id_set)
                        olog(
                            f"=== CG+KL subset: {group_label} "
                            f"(n_group={len(kl_ids)}, +refs={len(ref_ids)}) ==="
                        )
                        olog(
                            f"  {group_label}: members (stats) n={len(kl_ids)} "
                            f"(belonging refs n={len(belonging_refs)}); "
                            f"non-belonging refs n={len(nonbelonging_refs)} "
                            f"(kept in Jaccard/UMAP only)"
                        )
                        if belonging_refs:
                            olog(
                                f"    belonging refs: {', '.join(belonging_refs[:10])}"
                                + (" ..." if len(belonging_refs) > 10 else "")
                            )
                        if nonbelonging_refs:
                            olog(
                                f"    non-belonging refs: {', '.join(nonbelonging_refs[:10])}"
                                + (" ..." if len(nonbelonging_refs) > 10 else "")
                            )
                        subset_df = _build_group_gpa_df(gpa_df_filt, kl_ids, ref_ids)
                        subset_dir = _subset_analysis_dir(
                            main_analysis_dir, group_label
                        )
                        row = run_single_group_analysis(
                            gpa_df=subset_df,
                            meta_df=meta_df,
                            gff_counts_df=gff_counts_df,
                            group_label=group_label,
                            analysis_dir=subset_dir,
                            group_level="cg_klocus",
                            parent_group=cg_name,
                            member_sample_ids=pd.Index(sorted(kl_id_set)),
                            gpa_filter_cutoff=filter_cutoff,
                            merge_small_clusters=merge_small_clusters,
                            shell_cloud_cutoff=shell_cloud_cutoff,
                            core_shell_cutoff=core_shell_cutoff,
                            report_times=report_times,
                            reference_top_n=reference_top_n,
                        )
                        row["directory_leaf"] = run_label
                        rows.append(row)
            else:
                olog("skipping K_locus splits ('K_locus' column missing)")

            detail_df = pd.DataFrame(rows)
            detail_df.to_csv(detail_path, sep="\t", index=False)
            olog(f"detail TSV: {detail_path} ({len(rows)} rows)")

            total_wall = time.perf_counter() - t0
            olog(f"orchestrator done total_wall={total_wall:.1f}s rows={len(rows)}")
        except Exception as exc:  # noqa: BLE001
            total_wall = time.perf_counter() - t0
            olog(f"orchestrator ERROR total_wall={total_wall:.1f}s error={exc}")
            return {
                "directory_leaf": run_label,
                "group_label": run_label,
                "group_level": "whole_set",
                "parent_group": "",
                "status": "error",
                "error": str(exc),
            }

    whole_row["detail_tsv_path"] = detail_path
    return whole_row


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    input_group = p.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--directory-leaf",
        default=None,
        help="Panaroo directory name under PANAROO_RUN_ROOT, e.g. CG11_all",
    )
    input_group.add_argument(
        "--panaroo-dir",
        default=None,
        metavar="DIR",
        help=(
            "Full path to Panaroo output directory containing gene_presence_absence.csv "
            "(does not use PANAROO_RUN_ROOT). Output label is the directory basename."
        ),
    )
    p.add_argument(
        "--metadata",
        default=DEFAULT_METADATA_PATH,
        help=f"Metadata TSV path (default: {DEFAULT_METADATA_PATH})",
    )
    p.add_argument(
        "--gff-feature-counts",
        default=DEFAULT_GFF_FEATURE_COUNTS_PATH,
        help=(
            "GFF feature counts TSV path keyed by 'Sample' "
            f"(default: {DEFAULT_GFF_FEATURE_COUNTS_PATH})"
        ),
    )
    p.add_argument(
        "--panaroo-run-root",
        default=PANAROO_RUN_ROOT,
        help=f"Root used with --directory-leaf (default: {PANAROO_RUN_ROOT})",
    )
    p.add_argument(
        "--min-group-size",
        type=int,
        default=DEFAULT_MIN_GROUP_SIZE,
        help=(
            "Minimum Clonal group / K_locus size to get its own slice; "
            f"smaller groups are pooled into 'other' (default: {DEFAULT_MIN_GROUP_SIZE})."
        ),
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
        panaroo_dir=args.panaroo_dir,
        panaroo_run_root=args.panaroo_run_root,
        metadata_path=args.metadata,
        gff_feature_counts_path=args.gff_feature_counts,
        min_group_size=args.min_group_size,
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
