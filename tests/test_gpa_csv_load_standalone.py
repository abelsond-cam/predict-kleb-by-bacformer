#!/usr/bin/env python3
"""Standalone timing harness for the streaming gene_presence_absence.csv loader.

Minimal-imports (stdlib + numpy) script that mirrors the streaming parse logic
from ``bacotype.tl.gpa_distances_single_group._load_gpa_counts_from_csv``.

Steps
-----
1. Copy the input CSV to node-local scratch (``--scratch``, or ``$TMPDIR``, or
   ``/tmp``). Time the copy and report throughput in MB/s.
2. Stream the local copy row-by-row, converting each cell to a uint16 count
   (0 if empty, else ``cell.count(";") + 1``). Print a progress line every
   ``--progress-every`` rows with elapsed time and rows/s.
3. Report final matrix shape, uint16 matrix size (GB), and peak resident RSS.

Intended as a sanity check independent of the full pipeline (no scanpy /
anndata / matplotlib imports), so we can isolate parse performance and
filesystem latency.
"""
from __future__ import annotations

import argparse
import csv
import os
import resource
import shutil
import sys
import tempfile
import time

import numpy as np

DEFAULT_CSV = (
    "/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/"
    "panaroo_with_reference_genome/SL17_part_0/gene_presence_absence.csv"
)
DROP_COL_NAMES = frozenset({"Non-unique Gene name", "Annotation"})


def _log(msg: str) -> None:
    print(msg, flush=True)


def _peak_rss_gb() -> float:
    """Linux reports ru_maxrss in kilobytes."""
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kb / (1024 * 1024)


def _stream_parse(path: str, progress_every: int) -> tuple[int, int, float]:
    """Stream CSV into uint16 matrix; return (n_genes, n_samples, seconds)."""
    t0 = time.perf_counter()
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        try:
            gene_idx = header.index("Gene")
        except ValueError as exc:
            raise ValueError(f"Missing 'Gene' column in header of {path}") from exc

        sample_indices = [
            i for i, n in enumerate(header)
            if i != gene_idx and n not in DROP_COL_NAMES
        ]
        n_samples = len(sample_indices)
        _log(f"header parsed: {n_samples} sample columns (dropped {len(header) - 1 - n_samples} meta cols)")

        gene_names: list[str] = []
        row_arrays: list[np.ndarray] = []
        for i, row in enumerate(reader, start=1):
            gene_names.append(row[gene_idx])
            row_arrays.append(
                np.fromiter(
                    (
                        0 if not (cell := row[ci]) else cell.count(";") + 1
                        for ci in sample_indices
                    ),
                    dtype=np.uint16,
                    count=n_samples,
                )
            )
            if progress_every > 0 and i % progress_every == 0:
                elapsed = time.perf_counter() - t0
                rate = i / elapsed if elapsed > 0 else 0.0
                _log(
                    f"  {i} rows | {elapsed:6.1f}s | {rate:6.0f} rows/s "
                    f"| peak RSS {_peak_rss_gb():.2f} GB"
                )

    seconds = time.perf_counter() - t0
    if row_arrays:
        counts = np.stack(row_arrays)
    else:
        counts = np.zeros((0, n_samples), dtype=np.uint16)
    return counts.shape[0], n_samples, seconds


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=DEFAULT_CSV, help="Path to gene_presence_absence.csv")
    ap.add_argument(
        "--scratch", default=None,
        help="Node-local scratch directory for the copy (default: $TMPDIR, else /tmp)",
    )
    ap.add_argument("--progress-every", type=int, default=500,
                    help="Rows between progress logs; <=0 to silence")
    ap.add_argument("--skip-copy", action="store_true",
                    help="Read the source directly, skip the scratch-copy step")
    args = ap.parse_args(argv)

    src = args.csv
    if not os.path.isfile(src):
        _log(f"ERROR: CSV not found: {src}")
        return 2
    src_bytes = os.path.getsize(src)
    _log(f"source: {src} ({src_bytes / (1024 * 1024):.1f} MB)")

    if args.skip_copy:
        path_to_read = src
        _log("skip-copy: reading source path directly (no scratch copy)")
    else:
        scratch = args.scratch or os.environ.get("TMPDIR") or tempfile.gettempdir()
        os.makedirs(scratch, exist_ok=True)
        dst = os.path.join(scratch, os.path.basename(src))
        _log(f"copy: {src} -> {dst}")
        t_c = time.perf_counter()
        shutil.copyfile(src, dst)
        copy_s = time.perf_counter() - t_c
        dst_bytes = os.path.getsize(dst)
        mb_s = (dst_bytes / copy_s) / (1024 * 1024) if copy_s > 0 else float("inf")
        _log(
            f"copy done: {dst_bytes / (1024 * 1024):.1f} MB in {copy_s:.1f}s "
            f"({mb_s:.1f} MB/s)"
        )
        path_to_read = dst

    _log("-" * 72)
    _log(f"streaming parse starting (progress every {args.progress_every} rows)")
    _log("-" * 72)
    n_genes, n_samples, parse_s = _stream_parse(path_to_read, args.progress_every)
    _log("-" * 72)

    matrix_gb = (n_genes * n_samples * 2) / (1024 ** 3)
    file_mb = os.path.getsize(path_to_read) / (1024 * 1024)
    parse_mb_s = (file_mb / parse_s) if parse_s > 0 else float("inf")
    rows_s = (n_genes / parse_s) if parse_s > 0 else float("inf")
    _log(
        f"parse done: {n_genes} genes x {n_samples} samples in {parse_s:.1f}s "
        f"({rows_s:.0f} rows/s, {parse_mb_s:.1f} MB/s)"
    )
    _log(f"uint16 matrix footprint: {matrix_gb:.2f} GB")
    _log(f"peak RSS: {_peak_rss_gb():.2f} GB")

    if not args.skip_copy:
        try:
            os.remove(path_to_read)
            _log(f"cleaned up scratch copy: {path_to_read}")
        except OSError as e:
            _log(f"warning: could not remove scratch copy: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
