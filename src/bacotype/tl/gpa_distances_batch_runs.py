#!/usr/bin/env python3
"""Batch runner for stratified GPA distance analysis across Panaroo runs.

Discovers every immediate subdirectory of ``--panaroo-run-root`` that contains
a ``gene_presence_absence.Rtab`` file and runs the stratified distance
analysis orchestrator
(:func:`bacotype.tl.gpa_distances_single_run.run_gpa_analysis`) on each one
in parallel using a :class:`concurrent.futures.ProcessPoolExecutor`.

Per-run outputs (whole-set + per-Clonal-group + per-CG/K_locus detail TSVs)
are written by the orchestrator inside each Panaroo run directory. This batch
runner additionally compiles one combined summary TSV
``gpa_reference_batch_summary_<timestamp>.tsv`` containing the whole-set row
for each discovered run, under ``--output-dir``.

Typical invocation is via the Slurm wrapper
``slurm_scripts/gpa_distances_batch_runs.sh``.
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from bacotype.tl.gpa_distances_single_run import (
    DEFAULT_METADATA_PATH,
    DEFAULT_MIN_GROUP_SIZE,
    PANAROO_RUN_ROOT,
    run_gpa_analysis,
)

DEFAULT_OUTPUT_DIR = (
    "/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/pangenome_analysis"
)


def _discover_leaves(root: str) -> list[str]:
    """Return names of Panaroo-run subdirectories under ``root``.

    A subdirectory qualifies iff it contains a ``gene_presence_absence.Rtab``
    file at its top level. Only immediate children of ``root`` are inspected
    (no recursive descent). The return value is the leaf directory names
    (not full paths), sorted alphabetically, ready to pass as
    ``directory_leaf`` to
    :func:`bacotype.tl.gpa_distances_single_run.run_gpa_analysis`.
    """
    leaves: list[str] = []
    for entry in sorted(os.listdir(root)):
        run_dir = os.path.join(root, entry)
        gpa_rtab = os.path.join(run_dir, "gene_presence_absence.Rtab")
        if os.path.isdir(run_dir) and os.path.isfile(gpa_rtab):
            leaves.append(entry)
    return leaves


def main() -> int:
    """CLI entrypoint for the batch GPA distance analysis.

    Parses CLI arguments, discovers Panaroo run subdirectories under
    ``--panaroo-run-root``, and fans them out to parallel workers that call
    :func:`bacotype.tl.gpa_distances_single_run.run_gpa_analysis` for each
    run. When all workers complete, the whole-set summary rows are compiled
    into a single TSV under ``--output-dir``.

    Key options:
      * ``--workers``: number of parallel processes.
      * ``--panaroo-run-root``: directory whose immediate children are the
        per-run Panaroo output folders (each containing
        ``gene_presence_absence.Rtab``).
      * ``--metadata``: path to the curated metadata TSV.
      * ``--output-dir``: where the compiled batch summary TSV is written.
      * ``--min-group-size``: minimum Clonal group / K_locus size to get its
        own stratified slice inside each run (smaller groups pooled as
        ``other``).
      * ``--test-n-subdir``: optional cap on how many discovered leaves are
        processed (useful for smoke tests).
      * ``--reference-top-n``, ``--gpa-filter-cutoff``,
        ``--merge-small-clusters``, ``--shell-cloud-cutoff``,
        ``--core-shell-cutoff``, ``--report-times``: tuning knobs forwarded
        to the orchestrator.

    Returns ``0`` when at least one run succeeds, otherwise ``1``.
    """
    p = argparse.ArgumentParser(description="Batch runner for GPA reference genome analysis.")
    p.add_argument("--workers", type=int, default=10, help="Number of parallel workers.")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for compiled summary output.")
    p.add_argument("--metadata", default=DEFAULT_METADATA_PATH, help="Metadata TSV path.")
    p.add_argument(
        "--panaroo-run-root",
        default=PANAROO_RUN_ROOT,
        help=f"Root containing per-run Panaroo subdirectories (default: {PANAROO_RUN_ROOT}).",
    )
    p.add_argument("--test-n-subdir", type=int, default=None, help="Optional: process only first N leaves.")
    p.add_argument(
        "--min-group-size",
        type=int,
        default=DEFAULT_MIN_GROUP_SIZE,
        help=(
            "Minimum Clonal group / K_locus size to get its own stratified slice; "
            f"smaller groups are pooled into 'other' (default: {DEFAULT_MIN_GROUP_SIZE})."
        ),
    )
    p.add_argument("--gpa-filter-cutoff", type=int, default=None, help="Override GPA prevalence filter.")
    p.add_argument("--merge-small-clusters", type=int, default=None, help="Override merge-small-clusters value.")
    p.add_argument("--shell-cloud-cutoff", type=float, default=0.15, help="Shell/cloud penetrance cutoff.")
    p.add_argument("--core-shell-cutoff", type=float, default=0.95, help="Core/shell penetrance cutoff.")
    p.add_argument("--report-times", action="store_true", help="Enable timing suffixes in worker logs.")
    p.add_argument(
        "--reference-top-n",
        type=int,
        default=10,
        help="Top-N RefSeq / complete Norway genomes by lowest mean Jaccard (passed to run_gpa_analysis).",
    )
    args = p.parse_args()

    # Validate CLI arguments before doing any filesystem work or fan-out.
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.test_n_subdir is not None and args.test_n_subdir < 1:
        raise ValueError("--test-n-subdir must be >= 1 when provided")

    os.makedirs(args.output_dir, exist_ok=True)
    # Discover per-run Panaroo leaves (immediate children with a GPA .Rtab).
    leaves = _discover_leaves(args.panaroo_run_root)
    if args.test_n_subdir is not None:
        leaves = leaves[: args.test_n_subdir]

    if not leaves:
        print("No valid panaroo run subdirectories found.")
        return 1

    t0 = time.perf_counter()
    print(f"Batch start: n_subdirs={len(leaves)} workers={args.workers}", flush=True)
    print(f"Panaroo run root: {args.panaroo_run_root}", flush=True)
    print(f"Output dir: {args.output_dir}", flush=True)

    # Fan out one orchestrator call per Panaroo run across a process pool.
    # Each worker writes its own detail TSV (whole set + per-CG + per-CG/K_locus)
    # inside the run's analysis directory; here we only collect the whole-set
    # summary row returned by the orchestrator for the compiled batch table.
    results: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(
                run_gpa_analysis,
                directory_leaf=leaf,
                panaroo_run_root=args.panaroo_run_root,
                metadata_path=args.metadata,
                min_group_size=args.min_group_size,
                gpa_filter_cutoff=args.gpa_filter_cutoff,
                merge_small_clusters=args.merge_small_clusters,
                shell_cloud_cutoff=args.shell_cloud_cutoff,
                core_shell_cutoff=args.core_shell_cutoff,
                report_times=args.report_times,
                reference_top_n=args.reference_top_n,
            ): leaf
            for leaf in leaves
        }
        for fut in as_completed(futures):
            leaf = futures[fut]
            try:
                result = fut.result()
            except Exception as exc:  # noqa: BLE001
                result = {"directory_leaf": leaf, "status": "error", "error": str(exc)}
            results.append(result)
            print(f"done: {leaf} status={result.get('status', 'unknown')}", flush=True)

    # Compile one summary TSV: one row per Panaroo run (the whole-set row).
    df = pd.DataFrame(results)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_tsv = os.path.join(args.output_dir, f"gpa_reference_batch_summary_{stamp}.tsv")
    df.to_csv(out_tsv, sep="\t", index=False)

    n_ok = int((df.get("status", pd.Series(dtype=str)) == "ok").sum())
    n_err = int((df.get("status", pd.Series(dtype=str)) == "error").sum())
    total_wall = time.perf_counter() - t0
    print(f"Batch complete: ok={n_ok} error={n_err} wall={total_wall:.1f}s", flush=True)
    print(f"Saved: {out_tsv}", flush=True)
    return 0 if n_ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
