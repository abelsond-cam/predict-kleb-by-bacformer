"""Split curated sample metadata into Panaroo-sized TSV batches.

Sublineage splits, species batches, and rare Klebsiella pneumoniae packs with
deterministic shuffles and reference-genome handling.

Run: uv run python src/bacotype/pp/panaroo_metadata_batching.py
"""

# ruff: noqa: D102, D103

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO, cast

import numpy as np
import pandas as pd

DEFAULT_METADATA = Path(
    "/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/final/metadata_final_curated_slimmed.tsv"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/panaroo_with_reference_genome"
)
LOG_FILENAME = "panaroo_batching.log"


def as_bool(series: pd.Series) -> pd.Series:
    s = series.copy()
    if s.dtype == object:
        return s.map(
            lambda x: (
                str(x).strip().lower() in ("true", "1", "yes", "t")
                if pd.notna(x) and str(x).strip() != ""
                else False
            )
        )
    return s.fillna(False).astype(bool)


class RunLog:
    """Write the same lines to stdout and a single log file (overwrite each run)."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file: TextIO = path.open("w", encoding="utf-8")

    def close(self) -> None:
        self._file.close()

    def line(self, msg: str = "") -> None:
        print(msg)
        self._file.write(msg + "\n")
        self._file.flush()


def _strip_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in df.columns if str(c).startswith("_")], errors="ignore")


def count_refs(df: pd.DataFrame) -> tuple[int, int]:
    n_g = (
        int(as_bool(cast(pd.Series, df["is_mgh78578"])).sum())
        if "is_mgh78578" in df.columns
        else 0
    )
    n_n = (
        int(as_bool(cast(pd.Series, df["is_complete_norway_genome"])).sum())
        if "is_complete_norway_genome" in df.columns
        else 0
    )
    return n_g, n_n


def write_tsv(
    path: Path,
    df: pd.DataFrame,
    log: RunLog,
    written_counts: Counter[str],
) -> None:
    _strip_for_csv(df).to_csv(path, sep="\t", index=False)
    n = len(df)
    n_g, n_n = count_refs(df)
    warn = ""
    if n_g > 1:
        warn = "  WARNING: n_global > 1"
    elif n_g == 0:
        warn = "  NOTE: n_global == 0"
    log.line(f"  WROTE {path}")
    log.line(f"    rows={n}  n_global={n_g}  n_norway={n_n}{warn}")
    for sid in df["Sample"].astype(str):
        written_counts[sid] += 1


def append_global_dedupe(df: pd.DataFrame, global_row: pd.DataFrame) -> pd.DataFrame:
    out = pd.concat([df, global_row], ignore_index=True)
    return out.drop_duplicates(subset=["Sample"], keep="first")


def shuffle_two_parts(core: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match panaroo_run_strain._shuffle_and_part: part0 = first ceil(n/2), part1 = rest."""
    n = len(core)
    if n == 0:
        return core.copy(), core.copy()
    order = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(order)
    shuffled = core.iloc[order].reset_index(drop=True)
    mid = (n + 1) // 2
    return shuffled.iloc[:mid].copy(), shuffled.iloc[mid:].copy()


def shuffle_n_parts(core: pd.DataFrame, n_parts: int, seed: int) -> list[pd.DataFrame]:
    n = len(core)
    if n == 0:
        return [core.copy() for _ in range(n_parts)]
    order = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(order)
    shuffled = core.iloc[order].reset_index(drop=True)
    idx_groups = np.array_split(np.arange(n), n_parts)
    return [shuffled.iloc[g].copy() for g in idx_groups]


def add_refs_to_batch(
    core_batch: pd.DataFrame,
    global_row: pd.DataFrame,
    norway_in_sl: pd.DataFrame,
) -> pd.DataFrame:
    out = pd.concat([core_batch, global_row, norway_in_sl], ignore_index=True)
    return out.drop_duplicates(subset=["Sample"], keep="first")


def species_to_basename(species: str) -> str:
    safe = re.sub(r"[^\w\-. ]+", "_", str(species))
    safe = re.sub(r"\s+", "_", safe.strip())
    return safe[:200] if safe else "unknown_species"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Batch metadata TSVs for Panaroo runs.")
    p.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA,
        help="Input metadata TSV",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for batch TSVs and log",
    )
    p.add_argument("--min-sublineage", type=int, default=250)
    p.add_argument("--split-low", type=int, default=3500)
    p.add_argument("--split-high", type=int, default=7000)
    p.add_argument("--sl258-name", type=str, default="SL258")
    p.add_argument("--sl258-parts", type=int, default=5)
    p.add_argument("--kp-batch-min", type=int, default=1500)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    metadata_path = args.metadata.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / LOG_FILENAME
    log = RunLog(log_path)

    try:
        log.line(f"=== panaroo_metadata_batching {datetime.now(timezone.utc).isoformat()} ===")
        log.line(f"metadata={metadata_path}")
        log.line(f"output_dir={out_dir}")
        log.line(
            "args: "
            f"min_sublineage={args.min_sublineage} split_low={args.split_low} "
            f"split_high={args.split_high} sl258_name={args.sl258_name!r} "
            f"sl258_parts={args.sl258_parts} kp_batch_min={args.kp_batch_min} seed={args.seed}"
        )
        log.line("")

        metadata = pd.read_csv(metadata_path, sep="\t", low_memory=False)
        log.line(f"Loaded metadata: {len(metadata)} rows from {metadata_path}")

        kpsc = metadata.loc[metadata["kpsc_final_list"]].copy()
        log.line(f"After kpsc_final_list filter: {len(kpsc)} rows")

        if "is_mgh78578" not in kpsc.columns:
            log.line("ERROR: column is_mgh78578 missing")
            sys.exit(1)
        global_rows = kpsc.loc[as_bool(kpsc["is_mgh78578"])]
        if len(global_rows) != 1:
            log.line(f"ERROR: expected exactly one is_mgh78578 row, got {len(global_rows)}")
            sys.exit(1)
        global_row = global_rows.iloc[:1].copy()
        gid = str(global_row["Sample"].iloc[0])
        log.line(f"Global reference Sample: {gid}")
        log.line("")

        has_sl = kpsc["Sublineage"].notna()
        kpsc.loc[has_sl, "_sl_norm"] = (
            kpsc.loc[has_sl, "Sublineage"].astype(str).str.strip().str.upper()
        )
        sub = kpsc[has_sl]
        vc = sub.groupby("_sl_norm", sort=False).size()
        min_sl = args.min_sublineage
        large_sl_norms = set(vc[vc >= min_sl].index)
        sl258_norm = args.sl258_name.strip().upper()

        log.line("--- Phase: large sublineages (split or single file) ---")
        log.line(f"Sublineages with n >= {min_sl}: {len(large_sl_norms)}")
        written_counts: Counter[str] = Counter()

        for sl_norm in sorted(large_sl_norms):
            sl_df = kpsc[has_sl & (kpsc["_sl_norm"] == sl_norm)].copy()
            label = str(sl_df["Sublineage"].iloc[0]).strip()
            n = len(sl_df)
            norway_in_sl = sl_df.loc[as_bool(sl_df["is_complete_norway_genome"])].copy()
            ref_b = as_bool(sl_df["is_mgh78578"]) | as_bool(sl_df["is_complete_norway_genome"])
            core = sl_df.loc[~ref_b].copy()

            if sl_norm == sl258_norm:
                log.line(f"  {label}: n={n} -> SL258 path ({args.sl258_parts} parts)")
                parts = shuffle_n_parts(core, args.sl258_parts, args.seed)
                for i, part_core in enumerate(parts):
                    batch = add_refs_to_batch(part_core, global_row, norway_in_sl)
                    out_path = out_dir / f"{label}_part_{i}.tsv"
                    write_tsv(out_path, batch, log, written_counts)
                continue

            if n >= args.split_high and sl_norm != sl258_norm:
                log.line(
                    f"ERROR: sublineage {label!r} has n={n} >= {args.split_high} "
                    f"but is not {args.sl258_name!r}. Aborting."
                )
                sys.exit(1)

            if args.split_low < n < args.split_high:
                log.line(f"  {label}: n={n} -> 2-way split (core n={len(core)})")
                p0, p1 = shuffle_two_parts(core, args.seed)
                for i, part_core in enumerate((p0, p1)):
                    batch = add_refs_to_batch(part_core, global_row, norway_in_sl)
                    out_path = out_dir / f"{label}_part_{i}.tsv"
                    write_tsv(out_path, batch, log, written_counts)
                continue

            if min_sl <= n <= args.split_low:
                log.line(f"  {label}: n={n} -> single file (+ global if missing)")
                batch = append_global_dedupe(sl_df, global_row)
                out_path = out_dir / f"{label}.tsv"
                write_tsv(out_path, batch, log, written_counts)
                continue

            log.line(f"ERROR: uncategorized large sublineage {label!r} n={n}")
            sys.exit(1)

        rem_mask = kpsc["Sublineage"].isna() | ~kpsc["_sl_norm"].isin(large_sl_norms)
        remaining = kpsc.loc[rem_mask].copy()
        log.line("")
        log.line("--- Phase: remaining (small / null Sublineage) ---")
        log.line(f"Remaining rows: {len(remaining)}")

        kp_name = "Klebsiella pneumoniae"
        non_kp = remaining.loc[remaining["species"] != kp_name]
        log.line(f"Non–{kp_name} species groups: {non_kp['species'].nunique()}")

        for species, grp in non_kp.groupby("species", sort=False):
            batch = append_global_dedupe(grp, global_row)
            base = species_to_basename(species)
            out_path = out_dir / f"species_{base}.tsv"
            log.line(f"  species batch: {species!r} -> {out_path.name}")
            write_tsv(out_path, batch, log, written_counts)

        kp_rem = remaining.loc[remaining["species"] == kp_name].copy()
        log.line("")
        log.line("--- Phase: Klebsiella pneumoniae rare sublineages (greedy) ---")
        log.line(f"KP remaining rows: {len(kp_rem)}")

        if len(kp_rem) > 0:
            kp_rem["_sl_ord"] = kp_rem["Sublineage"].map(
                lambda x: str(x).strip().upper() if pd.notna(x) else ""
            )
            sl_sizes = kp_rem.groupby("_sl_ord", sort=False).size().sort_values(ascending=False)
            ordered_sl = list(sl_sizes.index)
            batch_i = 0
            current_parts: list[pd.DataFrame] = []
            current_count = 0

            for sl_o in ordered_sl:
                sl_part = kp_rem.loc[kp_rem["_sl_ord"] == sl_o]
                current_parts.append(sl_part)
                current_count += len(sl_part)
                if current_count > args.kp_batch_min:
                    batch_df = pd.concat(current_parts, ignore_index=True)
                    batch_df = append_global_dedupe(batch_df, global_row)
                    out_path = out_dir / f"kp_rare_sublineage_batch_{batch_i}.tsv"
                    log.line(
                        f"  flush batch {batch_i}: {current_count} rows "
                        f"across {len(current_parts)} sublineage keys"
                    )
                    write_tsv(out_path, batch_df, log, written_counts)
                    batch_i += 1
                    current_parts = []
                    current_count = 0

            if current_parts:
                batch_df = pd.concat(current_parts, ignore_index=True)
                batch_df = append_global_dedupe(batch_df, global_row)
                out_path = out_dir / f"kp_rare_sublineage_batch_{batch_i}.tsv"
                log.line(
                    f"  final batch {batch_i}: {len(batch_df)} rows "
                    f"across {len(current_parts)} sublineage keys"
                )
                write_tsv(out_path, batch_df, log, written_counts)

        log.line("")
        log.line("--- Coverage check (kpsc_final_list samples vs written) ---")
        all_ids = set(kpsc["Sample"].astype(str))
        missing_non_global = sorted(s for s in all_ids if s != gid and written_counts[s] == 0)
        dup_non_global = sorted(s for s in all_ids if s != gid and written_counts[s] > 1)
        if written_counts[gid] < 1:
            log.line("WARNING: global reference sample never written to any output file.")
        if missing_non_global:
            log.line(
                f"WARNING: {len(missing_non_global)} non-global samples never written "
                f"(showing up to 20): {missing_non_global[:20]}"
            )
        if dup_non_global:
            log.line(
                f"WARNING: {len(dup_non_global)} non-global samples appear in multiple "
                f"outputs (showing up to 20): {dup_non_global[:20]}"
            )
        if not missing_non_global and not dup_non_global:
            log.line(
                "Each non-global kpsc_final_list sample appears in exactly one output file; "
                f"global reference appears in {written_counts[gid]} files (expected >1)."
            )
        log.line("")
        log.line(f"Done. Log: {log_path}")
    finally:
        log.close()


if __name__ == "__main__":
    main()
