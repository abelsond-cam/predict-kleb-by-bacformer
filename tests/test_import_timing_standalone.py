#!/usr/bin/env python3
"""Time heavy package imports, one at a time (cold interpreter per invocation).

Designed for two use cases:

1. **RDS baseline:** ``uv run python .../test_import_timing_standalone.py``
   (uses project ``.venv`` symlink → RDS-backed site-packages).

2. **Local venv copy:** ``/path/to/copied/venv/bin/python .../test_import_timing_standalone.py``
   After ``rsync -a`` (or ``cp -a``) of the *resolved* venv tree to e.g.
   ``$SLURM_TMPDIR/bacotype_venv`` — uses node-local disk, not workspace quota.

Each run is a **fresh process**, so import times reflect cold loads from whatever
filesystem backs ``sys.prefix``.

No scanpy execution — only ``import`` statements and a tiny matplotlib backend
set (same pattern as ``gpa_distances_single_group.py``).
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from collections.abc import Callable


def _banner(title: str) -> None:
    print("=" * 72, flush=True)
    print(title, flush=True)
    print("=" * 72, flush=True)


def _print_env() -> None:
    print(f"pid={os.getpid()}", flush=True)
    print(f"executable={sys.executable}", flush=True)
    print(f"prefix={sys.prefix}", flush=True)
    print(f"VIRTUAL_ENV={os.environ.get('VIRTUAL_ENV', '<unset>')}", flush=True)
    try:
        import site

        paths = site.getsitepackages()
        if paths:
            print(f"site-packages[0]={paths[0]}", flush=True)
    except Exception as e:  # noqa: BLE001 — diagnostic script
        print(f"site.getsitepackages failed: {e}", flush=True)


def _time_step(fn: Callable[[], None]) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def main() -> int:
    """CLI entry: print env, time each import, print total."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--label",
        default="",
        help="Short tag printed in banner (e.g. rds, local_tmp)",
    )
    args = ap.parse_args()

    tag = f" ({args.label})" if args.label else ""
    _banner(f"Import timing probe{tag}")
    _print_env()
    print("-" * 72, flush=True)

    def _matplotlib_agg() -> None:
        import matplotlib

        matplotlib.use("Agg")

    # Order: dependencies roughly before dependents; same stack as GPA scripts.
    steps: list[tuple[str, Callable[[], None]]] = [
        ("numpy", lambda: importlib.import_module("numpy")),
        ("pandas", lambda: importlib.import_module("pandas")),
        ("scipy", lambda: importlib.import_module("scipy")),
        ("matplotlib", _matplotlib_agg),
        ("seaborn", lambda: importlib.import_module("seaborn")),
        ("Bio (biopython)", lambda: importlib.import_module("Bio")),
        ("gffutils", lambda: importlib.import_module("gffutils")),
        ("mudata", lambda: importlib.import_module("mudata")),
        ("anndata", lambda: importlib.import_module("anndata")),
        ("scanpy", lambda: importlib.import_module("scanpy")),
    ]

    total = 0.0
    for label, fn in steps:
        try:
            dt = _time_step(fn)
            total += dt
            print(f"{label:22s}  {dt:8.3f}s", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"{label:22s}  FAILED after {total:.3f}s total so far: {e}", flush=True)
            return 1

    print("-" * 72, flush=True)
    print(f"{'TOTAL (listed imports)':22s}  {total:8.3f}s", flush=True)
    _banner("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
