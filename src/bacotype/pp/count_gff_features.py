"""
Count GFF feature types (column 3) for every sample in the metadata TSV.

For each row of metadata_final_curated_slimmed.tsv, open the sample's gzipped
GFF, count occurrences of each feature type (GFF column 3), and write a sidecar
TSV with one row per sample and one column per feature type (prefixed ``n_``).
Failed files are written to a separate ``.errors.tsv``.

Designed to run as a single Slurm job on one icelake node using Python
multiprocessing across all allocated cores. Supports resume: if the sidecar
TSV already exists, already-processed samples are skipped.

Run: ``uv run python -m bacotype.pp.count_gff_features``.
"""

from __future__ import annotations

import argparse
import gzip
import multiprocessing as mp
import os
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

from bacotype.data_paths import data


BASE_DIR: Path = data.warm.parent  # /home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw
METADATA_F: Path = data.klebsiella_metadata_file
SIDECAR_F: Path = data.final / "gff_feature_counts.tsv"
ERRORS_F: Path = data.final / "gff_feature_counts.errors.tsv"


def count_features(
    args: tuple[str, str],
) -> tuple[str, dict[str, int] | None, str | None]:
    """Stream a gzipped GFF and return (sample, {feature_type: count}, error)."""
    sample, abs_path = args
    counts: Counter[str] = Counter()
    try:
        with gzip.open(abs_path, "rt") as fh:
            for line in fh:
                if not line or line[0] == "#":
                    continue
                f = line.split("\t", 4)
                if len(f) >= 3:
                    counts[f[2]] += 1
    except Exception as e:
        return sample, None, f"{type(e).__name__}: {e}"
    return sample, dict(counts), None


def _load_jobs(
    metadata_path: Path, sidecar_path: Path
) -> tuple[list[tuple[str, str]], pd.DataFrame | None, int, int]:
    """Return (jobs, existing_sidecar_df, n_total_with_gff, n_skipped_resume)."""
    df = pd.read_csv(
        metadata_path,
        sep="\t",
        usecols=["Sample", "gff_file"],
        dtype=str,
        low_memory=False,
    )
    df["gff_file"] = df["gff_file"].fillna("").str.strip()
    df = df[df["gff_file"].str.len() > 0].copy()
    n_total = len(df)

    existing_df: pd.DataFrame | None = None
    done_samples: set[str] = set()
    if sidecar_path.exists():
        existing_df = pd.read_csv(sidecar_path, sep="\t", dtype={"Sample": str})
        done_samples = set(existing_df["Sample"].astype(str))

    df = df[~df["Sample"].isin(done_samples)].copy()
    n_skipped = n_total - len(df)

    df["abs_path"] = df["gff_file"].map(lambda p: str(BASE_DIR / p))
    jobs: list[tuple[str, str]] = list(
        zip(df["Sample"].astype(str), df["abs_path"].astype(str))
    )
    return jobs, existing_df, n_total, n_skipped


def _format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m{s:02d}s"


def _atomic_write_tsv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, sep="\t", index=False)
    os.replace(tmp, path)


def _build_counts_df(rows: list[tuple[str, dict[str, int]]]) -> pd.DataFrame:
    """Build a DataFrame with columns ``Sample, n_<type>...`` (sorted)."""
    if not rows:
        return pd.DataFrame(columns=["Sample"])
    samples = [r[0] for r in rows]
    dicts = [r[1] for r in rows]
    counts_df = pd.DataFrame.from_records(dicts).fillna(0).astype(int)
    counts_df.insert(0, "Sample", samples)
    feature_cols = sorted(c for c in counts_df.columns if c != "Sample")
    rename_map = {c: f"n_{c}" for c in feature_cols}
    counts_df = counts_df[["Sample", *feature_cols]].rename(columns=rename_map)
    return counts_df


def run(
    metadata_path: Path | None = None,
    sidecar_path: Path | None = None,
    errors_path: Path | None = None,
    workers: int | None = None,
    flush_every: int = 10_000,
) -> None:
    """Count GFF features for every sample and write sidecar + errors TSVs."""
    meta_path = Path(metadata_path) if metadata_path is not None else METADATA_F
    side_path = Path(sidecar_path) if sidecar_path is not None else SIDECAR_F
    err_path = Path(errors_path) if errors_path is not None else ERRORS_F

    if workers is None:
        workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))

    print(f"Metadata file : {meta_path}")
    print(f"Base dir      : {BASE_DIR}")
    print(f"Sidecar out   : {side_path}")
    print(f"Errors out    : {err_path}")
    print(f"Workers       : {workers}")

    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file does not exist: {meta_path}")

    jobs, existing_df, n_total, n_skipped = _load_jobs(meta_path, side_path)
    n_jobs = len(jobs)
    print(f"Samples with gff_file       : {n_total}")
    print(f"Already in sidecar (resume) : {n_skipped}")
    print(f"New jobs to process         : {n_jobs}")

    if n_jobs == 0 and existing_df is not None:
        print("Nothing to do; sidecar already complete.")
        return

    side_path.parent.mkdir(parents=True, exist_ok=True)

    successes: list[tuple[str, dict[str, int]]] = []
    errors: list[tuple[str, str, str]] = []
    sample_to_path = dict(jobs)

    start = time.monotonic()
    last_log = start
    done = 0

    def _flush(final: bool = False) -> None:
        new_df = _build_counts_df(successes)
        if existing_df is not None and len(existing_df) > 0:
            combined = pd.concat([existing_df, new_df], ignore_index=True, sort=False)
            feature_cols = sorted(c for c in combined.columns if c != "Sample")
            combined = combined[["Sample", *feature_cols]].fillna(0)
            for c in feature_cols:
                combined[c] = combined[c].astype(int)
        else:
            combined = new_df
        _atomic_write_tsv(combined, side_path)
        if errors:
            err_df = pd.DataFrame(
                errors, columns=["Sample", "gff_file", "error"]
            )
            _atomic_write_tsv(err_df, err_path)
        if final:
            print(f"Wrote sidecar: {side_path} ({len(combined)} rows)")
            if errors:
                print(f"Wrote errors : {err_path} ({len(errors)} rows)")

    try:
        with mp.Pool(processes=workers) as pool:
            for sample, counts, err in pool.imap_unordered(
                count_features, jobs, chunksize=64
            ):
                done += 1
                if err is not None or counts is None:
                    errors.append((sample, sample_to_path.get(sample, ""), err or ""))
                else:
                    successes.append((sample, counts))

                now = time.monotonic()
                if done % 2_000 == 0 or (now - last_log) > 30:
                    elapsed = now - start
                    rate = done / elapsed if elapsed > 0 else 0.0
                    remaining = n_jobs - done
                    eta = remaining / rate if rate > 0 else 0.0
                    print(
                        f"  progress: {done}/{n_jobs} "
                        f"({100.0 * done / n_jobs:.1f}%) "
                        f"rate={rate:.1f} files/s  "
                        f"elapsed={_format_eta(elapsed)} eta={_format_eta(eta)}  "
                        f"errors={len(errors)}",
                        flush=True,
                    )
                    last_log = now

                if done % flush_every == 0:
                    _flush(final=False)
    finally:
        _flush(final=True)

    elapsed = time.monotonic() - start
    print(
        f"Done. processed={done} successes={len(successes)} "
        f"errors={len(errors)} elapsed={_format_eta(elapsed)}"
    )


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Count GFF feature types per sample and write a sidecar TSV.",
    )
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--sidecar", type=Path, default=None)
    parser.add_argument("--errors", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--flush-every", type=int, default=10_000)
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    run(
        metadata_path=args.metadata,
        sidecar_path=args.sidecar,
        errors_path=args.errors,
        workers=args.workers,
        flush_every=args.flush_every,
    )


if __name__ == "__main__":
    main()
