#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from bacotype.tl.panaroo_GPA_reference_genome import (
    DEFAULT_METADATA_PATH,
    PANAROO_RUN_ROOT,
    run_gpa_analysis,
)

DEFAULT_OUTPUT_DIR = (
    "/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/pangenome_analysis"
)


def _discover_leaves(root: str) -> list[str]:
    leaves: list[str] = []
    for entry in sorted(os.listdir(root)):
        run_dir = os.path.join(root, entry)
        gpa_rtab = os.path.join(run_dir, "gene_presence_absence.Rtab")
        if os.path.isdir(run_dir) and os.path.isfile(gpa_rtab):
            leaves.append(entry)
    return leaves


def main() -> int:
    p = argparse.ArgumentParser(description="Batch runner for GPA reference genome analysis.")
    p.add_argument("--workers", type=int, default=10, help="Number of parallel workers.")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for compiled summary output.")
    p.add_argument("--metadata", default=DEFAULT_METADATA_PATH, help="Metadata TSV path.")
    p.add_argument("--test-n-subdir", type=int, default=None, help="Optional: process only first N leaves.")
    p.add_argument("--gpa-filter-cutoff", type=int, default=None, help="Override GPA prevalence filter.")
    p.add_argument("--merge-small-clusters", type=int, default=None, help="Override merge-small-clusters value.")
    p.add_argument("--shell-cloud-cutoff", type=float, default=0.15, help="Shell/cloud penetrance cutoff.")
    p.add_argument("--core-shell-cutoff", type=float, default=0.95, help="Core/shell penetrance cutoff.")
    p.add_argument("--report-times", action="store_true", help="Enable timing suffixes in worker logs.")
    args = p.parse_args()

    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.test_n_subdir is not None and args.test_n_subdir < 1:
        raise ValueError("--test-n-subdir must be >= 1 when provided")

    os.makedirs(args.output_dir, exist_ok=True)
    leaves = _discover_leaves(PANAROO_RUN_ROOT)
    if args.test_n_subdir is not None:
        leaves = leaves[: args.test_n_subdir]

    if not leaves:
        print("No valid panaroo run subdirectories found.")
        return 1

    t0 = time.perf_counter()
    print(f"Batch start: n_subdirs={len(leaves)} workers={args.workers}", flush=True)
    print(f"Output dir: {args.output_dir}", flush=True)

    results: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(
                run_gpa_analysis,
                directory_leaf=leaf,
                metadata_path=args.metadata,
                gpa_filter_cutoff=args.gpa_filter_cutoff,
                merge_small_clusters=args.merge_small_clusters,
                shell_cloud_cutoff=args.shell_cloud_cutoff,
                core_shell_cutoff=args.core_shell_cutoff,
                report_times=args.report_times,
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

    df = pd.DataFrame(results)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(args.output_dir, f"gpa_reference_batch_summary_{stamp}.csv")
    out_tsv = os.path.join(args.output_dir, f"gpa_reference_batch_summary_{stamp}.tsv")
    df.to_csv(out_csv, index=False)
    df.to_csv(out_tsv, sep="\t", index=False)

    n_ok = int((df.get("status", pd.Series(dtype=str)) == "ok").sum())
    n_err = int((df.get("status", pd.Series(dtype=str)) == "error").sum())
    total_wall = time.perf_counter() - t0
    print(f"Batch complete: ok={n_ok} error={n_err} wall={total_wall:.1f}s", flush=True)
    print(f"Saved: {out_csv}", flush=True)
    print(f"Saved: {out_tsv}", flush=True)
    return 0 if n_ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
