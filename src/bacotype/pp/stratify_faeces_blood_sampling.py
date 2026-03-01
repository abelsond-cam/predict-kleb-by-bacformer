#!/usr/bin/env python3
"""
Stratify blood and faeces/rectal swab samples by country.

This script balances blood and faeces samples across geographic locations
with a configurable ratio, sampling at country level only.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np


# Constants
DEFAULT_RATIO = 2.0
TEST_RATIOS = [1.0, 2.0, 2.5, 3.0]
ISOLATION_SOURCES = ['blood', 'faeces & rectal swabs']


def setup_logging(log_file: str = "stratify_sampling.log"):
    """Configure logging to both file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )


def _log_category_breakdown(
    df: pd.DataFrame,
    category_col: str,
    header: str,
    top_n: int = None,
    exclude_vals: List[str] = None,
    include_blood_faeces: bool = True,
) -> None:
    """
    Log category counts. Optionally include blood/faeces sub-breakdown per category.

    Args:
        df: DataFrame with category_col and isolation_source_category
        category_col: Column to break down
        header: Header line for the report
        top_n: If set, show only top N categories by total
        exclude_vals: If set, exclude these values from the breakdown
        include_blood_faeces: If False, log only category: total (no blood/faeces lines)
    """
    if category_col not in df.columns:
        logging.info(f"  {header}: column '{category_col}' not found")
        return
    crosstab = pd.crosstab(df[category_col], df['isolation_source_category'])
    if exclude_vals:
        crosstab = crosstab[~crosstab.index.isin(exclude_vals)]
    totals = crosstab.sum(axis=1)
    totals = totals.sort_values(ascending=False)
    if top_n:
        totals = totals.head(top_n)
    logging.info(f"  {header}:")
    for cat_val in totals.index:
        label = "NA" if pd.isna(cat_val) else str(cat_val)
        row = crosstab.loc[cat_val]
        n_blood = int(row.get('blood', 0)) if isinstance(row, pd.Series) else 0
        n_faeces = int(row.get('faeces & rectal swabs', 0)) if isinstance(row, pd.Series) else 0
        total = n_blood + n_faeces
        if include_blood_faeces:
            logging.info(f"    {label}:")
            logging.info(f"      Total: {total:,}")
            logging.info(f"      blood: {n_blood:,}")
            logging.info(f"      faeces & rectal swabs: {n_faeces:,}")
        else:
            logging.info(f"    {label}: {total:,}")


def _log_final_country_table(df_init: pd.DataFrame, df_final: pd.DataFrame) -> None:
    """
    Log final table by country: Country | Faeces (Init -> Sample) | Blood (Init -> Sample) | Sampled ratio
    """
    if df_final.empty or 'country_parsed' not in df_final.columns:
        return
    if 'country_parsed' not in df_init.columns or 'isolation_source_category' not in df_init.columns:
        return

    iso_col = 'isolation_source_category'

    def _counts_by_country(df: pd.DataFrame) -> pd.DataFrame:
        ctab = pd.crosstab(df['country_parsed'], df[iso_col])
        return ctab

    init_ct = _counts_by_country(df_init)
    final_ct = _counts_by_country(df_final)

    all_countries = sorted(
        set(init_ct.index) | set(final_ct.index),
        key=lambda x: (str(x) if pd.notna(x) else "")
    )
    rows = []
    for country in all_countries:
        if pd.isna(country):
            continue
        init_row = init_ct.loc[country] if country in init_ct.index else pd.Series()
        final_row = final_ct.loc[country] if country in final_ct.index else pd.Series()
        i_f = int(init_row.get('faeces & rectal swabs', 0))
        i_b = int(init_row.get('blood', 0))
        s_f = int(final_row.get('faeces & rectal swabs', 0))
        s_b = int(final_row.get('blood', 0))
        ratio_str = f"{s_b / s_f:.2f}" if s_f > 0 else "-"
        rows.append((country, i_f, s_f, i_b, s_b, ratio_str))

    rows.sort(key=lambda r: r[2] + r[4], reverse=True)

    logging.info(f"\n{'='*80}")
    logging.info("FINAL SAMPLES BY COUNTRY")
    logging.info(f"{'='*80}")
    logging.info(f"{'Country':<30} {'Faeces (Init -> Sample)':<25} {'Blood (Init -> Sample)':<25} {'Sampled ratio':<15}")
    logging.info("-" * 95)
    for country, i_f, s_f, i_b, s_b, ratio_str in rows:
        faeces_str = f"{i_f:,} -> {s_f:,}"
        blood_str = f"{i_b:,} -> {s_b:,}"
        logging.info(f"{str(country):<30} {faeces_str:<25} {blood_str:<25} {ratio_str:<15}")
    logging.info("")


def calculate_ratio_bounds(ratio: float) -> Tuple[float, float]:
    """
    Calculate acceptable ratio bounds.
    
    Args:
        ratio: Target ratio (e.g., 2.0)
    
    Returns:
        Tuple of (lower_bound, upper_bound) where acceptable ratios are
        between 1/ratio and ratio (e.g., for ratio=2: 0.5 to 2.0)
    """
    return (1.0 / ratio, ratio)


def load_and_filter_data(
    metadata_file: str,
    filter_by_study_setting: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Load metadata and apply initial filters.
    Does NOT filter by amr_study - that split is done in main for dual-thread stratification.

    Args:
        metadata_file: Path to the metadata TSV file
        filter_by_study_setting: If True, filter to study_setting contains "Hospital"

    Returns:
        Tuple of (filtered_dataframe, filter_counts_dict)
    """
    logging.info(f"Loading metadata from: {metadata_file}")

    df = pd.read_csv(metadata_file, sep="\t", low_memory=False)
    initial_count = len(df)
    logging.info(f"Initial dataset size: {initial_count:,} samples")

    filter_counts = {'initial': initial_count}

    # Filter 1: Isolation source category
    logging.info(f"\nFiltering to isolation sources: {ISOLATION_SOURCES}")
    _log_category_breakdown(
        df, 'isolation_source_category', "isolation_source_category (pre-filter, top 6)",
        top_n=6, include_blood_faeces=False
    )
    df = df[df['isolation_source_category'].isin(ISOLATION_SOURCES)]
    filter_counts['after_isolation_source'] = len(df)
    logging.info(f"After isolation source filter: {len(df):,} samples")

    # Filter 2: Host category == "human"
    logging.info('\nFiltering to host_category == "human"')
    _log_category_breakdown(df, 'host_category', "host_category (pre-filter, top 5 other)", top_n=5, exclude_vals=['human'])
    df = df[df['host_category'] == 'human']
    filter_counts['after_host_category'] = len(df)
    logging.info(f"After host_category filter: {len(df):,} samples")

    # Filter 3: Study setting (conditional)
    logging.info('\nStudy setting:')
    _log_category_breakdown(df, 'study_setting', "study_setting (pre-filter, all categories)")
    if filter_by_study_setting:
        df = df[df['study_setting'].str.contains('Hospital', case=False, na=False)]
        filter_counts['after_study_setting'] = len(df)
        logging.info(f"After study_setting filter: {len(df):,} samples (filter applied)")
    else:
        filter_counts['after_study_setting'] = len(df)
        logging.info(f"Study setting filter: NOT APPLIED (--filter-by-study-setting not set). All {len(df):,} samples retained.")

    # AMR study routing (no filter - report only; split done in main)
    logging.info("\namr_study routing (samples assigned to AMR, Surveillance, NA threads):")
    _log_category_breakdown(df, 'amr_study', "amr_study (all categories)")
    mask_amr = df['amr_study'].str.contains('amr', case=False, na=False)
    mask_surveillance = df['amr_study'].str.contains('surveillance', case=False, na=False) & ~mask_amr
    mask_na = df['amr_study'].isna()
    n_amr = mask_amr.sum()
    n_surveillance = mask_surveillance.sum()
    n_na = mask_na.sum()
    n_unassigned = len(df) - n_amr - n_surveillance - n_na
    logging.info(f"  AMR thread (amr/AMR plus control): {n_amr:,} samples")
    logging.info(f"  Surveillance thread: {n_surveillance:,} samples")
    logging.info(f"  NA thread: {n_na:,} samples")
    if n_unassigned > 0:
        logging.info(f"  Unassigned (excluded from stratification): {n_unassigned:,} samples")

    # Summary of isolation sources in retained df
    logging.info("\nBreakdown by isolation source (after pre-amr filters):")
    source_counts = df['isolation_source_category'].value_counts()
    for source, count in source_counts.items():
        logging.info(f"  {source}: {count:,}")

    return df, filter_counts


def sample_to_ratio(df: pd.DataFrame, ratio: float, isolation_col: str = 'isolation_source_category') -> pd.DataFrame:
    """
    Keep all samples from the smaller group and sample from the larger group.
    Target: larger_sampled = ratio * smaller_count (capped at available).
    
    E.g. 67 blood, 436 faeces with ratio=3: keep all 67 blood, sample 3*67=201 faeces.
    
    Args:
        df: DataFrame subset to sample
        ratio: Target ratio (larger_group : smaller_group)
        isolation_col: Column name for isolation source
    
    Returns:
        Sampled DataFrame
    """
    counts = df[isolation_col].value_counts()
    n_blood = counts.get('blood', 0)
    n_faeces = counts.get('faeces & rectal swabs', 0)

    if n_blood == 0 or n_faeces == 0:
        return df

    # Keep ALL of the smaller group; sample min(ratio * smaller, larger) from the larger
    if n_blood < n_faeces:
        n_smaller, n_larger = n_blood, n_faeces
        smaller_val, larger_val = 'blood', 'faeces & rectal swabs'
    else:
        n_smaller, n_larger = n_faeces, n_blood
        smaller_val, larger_val = 'faeces & rectal swabs', 'blood'

    target_larger = min(int(ratio * n_smaller), n_larger)
    smaller_df = df[df[isolation_col] == smaller_val]
    larger_df = df[df[isolation_col] == larger_val].sample(n=target_larger, random_state=42)
    return pd.concat([smaller_df, larger_df])


def stratify_by_country(
    df: pd.DataFrame,
    ratio: float,
    thread_label: str = "",
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Apply country-level stratification. All countries with both blood and faeces
    are processed; none are deferred to region.

    Args:
        df: Input dataframe
        ratio: Target ratio
        thread_label: Optional label for log context (e.g. "AMR", "Surveillance")

    Returns:
        Tuple of (stratified_dataframe, sampling_log)
    """
    lower_bound, upper_bound = calculate_ratio_bounds(ratio)

    stratified_samples = []
    sampling_log = []

    title = f"COUNTRY-LEVEL STRATIFICATION{f' [{thread_label}]' if thread_label else ''}"
    logging.info(f"\n{'='*80}")
    logging.info(title)
    logging.info(f"{'='*80}")
    logging.info(f"Target ratio: {ratio} (acceptable range: {lower_bound:.2f} to {upper_bound:.2f})")
    logging.info("")
    
    for country in df['country_parsed'].unique():
        if pd.isna(country):
            continue

        country_df = df[df['country_parsed'] == country]

        counts = country_df['isolation_source_category'].value_counts()
        n_blood = counts.get('blood', 0)
        n_faeces = counts.get('faeces & rectal swabs', 0)

        if n_blood == 0 or n_faeces == 0:
            continue

        current_ratio = n_blood / n_faeces

        if lower_bound <= current_ratio <= upper_bound:
            stratified_samples.append(country_df)
            action = 'accepted_all'
            sampled_blood = n_blood
            sampled_faeces = n_faeces
        else:
            sampled_df = sample_to_ratio(country_df, ratio)
            stratified_samples.append(sampled_df)
            action = 'country_sampled'
            sampled_counts = sampled_df['isolation_source_category'].value_counts()
            sampled_blood = sampled_counts.get('blood', 0)
            sampled_faeces = sampled_counts.get('faeces & rectal swabs', 0)

        sampling_log.append({
            'location': country,
            'location_type': 'country',
            'initial_faeces': n_faeces,
            'sampled_faeces': sampled_faeces,
            'initial_blood': n_blood,
            'sampled_blood': sampled_blood,
            'initial_ratio': current_ratio,
            'action': action
        })

    stratified_df = pd.concat(stratified_samples, ignore_index=True) if stratified_samples else pd.DataFrame()

    logging.info(f"Countries processed: {len(sampling_log)}")
    logging.info(f"Samples stratified at country level: {len(stratified_df):,}")

    return stratified_df, sampling_log


def stratify_by_location(
    df: pd.DataFrame,
    ratio: float,
    thread_label: str = "",
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Main stratification function - country-level only.

    Args:
        df: Input dataframe
        ratio: Target ratio
        thread_label: Optional label for log context (e.g. "AMR", "Surveillance")

    Returns:
        Tuple of (final_stratified_dataframe, sampling_log)
    """
    final_df, complete_log = stratify_by_country(df, ratio, thread_label=thread_label)
    
    summary_title = f"FINAL STRATIFICATION SUMMARY{f' [{thread_label}]' if thread_label else ''}"
    logging.info(f"\n{'='*80}")
    logging.info(summary_title)
    logging.info(f"{'='*80}")
    logging.info(f"Total samples after stratification: {len(final_df):,}")
    
    # Final counts by isolation source
    if not final_df.empty:
        final_counts = final_df['isolation_source_category'].value_counts()
        logging.info("\nFinal breakdown by isolation source:")
        for source, count in final_counts.items():
            logging.info(f"  {source}: {count:,}")
        
        n_blood = final_counts.get('blood', 0)
        n_faeces = final_counts.get('faeces & rectal swabs', 0)
        if n_faeces > 0:
            final_ratio = n_blood / n_faeces
            logging.info(f"\nFinal blood:faeces ratio: {final_ratio:.2f}")
    
    return final_df, complete_log


def create_detailed_report(
    sampling_log: List[Dict],
    ratio: float,
    thread_label: str = "",
):
    """
    Create detailed country breakdown report.

    Args:
        sampling_log: List of sampling decisions
        ratio: The ratio used
        thread_label: Optional label for log context (e.g. "AMR", "Surveillance")
    """
    title = f"DETAILED BREAKDOWN FOR RATIO = {ratio}{f' [{thread_label}]' if thread_label else ''}"
    logging.info(f"\n{'='*80}")
    logging.info(title)
    logging.info(f"{'='*80}")
    logging.info("")
    
    # Sort by total initial samples (descending)
    sorted_log = sorted(
        sampling_log,
        key=lambda x: x['initial_blood'] + x['initial_faeces'],
        reverse=True
    )
    
    # Header
    logging.info(f"{'Location':<30} {'Type':<10} {'Faeces':<15} {'Blood':<15} {'Ratio':<10} {'Action':<20}")
    logging.info(f"{'':^30} {'':^10} {'Init → Sample':<15} {'Init → Sample':<15} {'(B/F)':<10} {'':<20}")
    logging.info("-" * 115)
    
    # Detail rows
    for entry in sorted_log:
        location = entry['location'][:28]  # Truncate if too long
        loc_type = entry['location_type']
        
        faeces_str = f"{entry['initial_faeces']:,} → {entry['sampled_faeces']:,}"
        blood_str = f"{entry['initial_blood']:,} → {entry['sampled_blood']:,}"
        ratio_str = f"{entry['initial_ratio']:.2f}"
        action = entry['action'].replace('_', ' ').title()
        
        logging.info(
            f"{location:<30} {loc_type:<10} {faeces_str:<15} {blood_str:<15} "
            f"{ratio_str:<10} {action:<20}"
        )
    
    logging.info("")


def test_multiple_ratios(
    df: pd.DataFrame,
    test_ratios: List[float],
    filter_counts: Dict[str, int],
):
    """
    Test stratification with multiple ratios.

    Args:
        df: Filtered dataframe
        test_ratios: List of ratios to test
        filter_counts: Dictionary of filter stage counts
    """
    logging.info(f"\n{'='*80}")
    logging.info("TESTING MULTIPLE RATIOS")
    logging.info(f"{'='*80}")
    logging.info("")
    
    results = {}
    
    for ratio in test_ratios:
        logging.info(f"\n{'-'*80}")
        logging.info(f"RATIO: {ratio}")
        logging.info(f"{'-'*80}")
        
        stratified_df, sampling_log = stratify_by_location(df.copy(), ratio)
        
        # Store results
        results[ratio] = {
            'stratified_df': stratified_df,
            'sampling_log': sampling_log,
            'final_count': len(stratified_df)
        }
        
        # Create detailed report for default ratio
        if ratio == DEFAULT_RATIO:
            create_detailed_report(sampling_log, ratio)
    
    # Summary table across all ratios
    logging.info(f"\n{'='*80}")
    logging.info("SUMMARY ACROSS ALL RATIOS")
    logging.info(f"{'='*80}")
    logging.info("")
    logging.info(f"{'Ratio':<10} {'Final Count':<15} {'Blood':<15} {'Faeces':<15} {'Blood/Faeces':<15}")
    logging.info("-" * 70)
    
    for ratio in test_ratios:
        stratified_df = results[ratio]['stratified_df']
        if not stratified_df.empty:
            counts = stratified_df['isolation_source_category'].value_counts()
            n_blood = counts.get('blood', 0)
            n_faeces = counts.get('faeces & rectal swabs', 0)
            final_ratio = n_blood / n_faeces if n_faeces > 0 else 0
            
            logging.info(
                f"{ratio:<10} {len(stratified_df):<15,} {n_blood:<15,} "
                f"{n_faeces:<15,} {final_ratio:<15.2f}"
            )
    
    logging.info("")
    
    return results


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Stratify blood and faeces samples by geographic location'
    )
    parser.add_argument(
        '--ratio',
        type=float,
        default=DEFAULT_RATIO,
        help=f'Target blood:faeces ratio (default: {DEFAULT_RATIO})'
    )
    parser.add_argument(
        '--metadata-file',
        type=str,
        default="/Users/davidabelson/Library/CloudStorage/OneDrive-UniversityofCambridge/Aaron Weimann's files - project_k/data/final/metadata/metadata_final_curated_slimmed.tsv",
        help='Path to metadata TSV file'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        default='stratify_sampling.log',
        help='Path to output log file'
    )
    parser.add_argument(
        '--filter-by-study-setting',
        action='store_true',
        help='Filter to study_setting contains "Hospital" (default: no filter, all samples retained)'
    )
    parser.add_argument(
        '--test-all-ratios',
        action='store_true',
        help='Test all predefined ratios instead of just the specified one'
    )
    parser.add_argument(
        '--output-csv',
        type=str,
        help='Optional: Save stratified samples to CSV file'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_file)
    
    logging.info("=" * 80)
    logging.info("BLOOD VS FAECES/RECTAL SWABS STRATIFICATION")
    logging.info("=" * 80)
    logging.info("")
    
    try:
        df, filter_counts = load_and_filter_data(
            args.metadata_file,
            filter_by_study_setting=args.filter_by_study_setting,
        )

        if df.empty:
            logging.error("No data remaining after filtering!")
            return

        # Split by amr_study: AMR, Surveillance, NA (three separate threads)
        mask_amr = df['amr_study'].str.contains('amr', case=False, na=False)
        mask_surveillance = df['amr_study'].str.contains('surveillance', case=False, na=False) & ~mask_amr
        mask_na = df['amr_study'].isna()
        df_amr = df[mask_amr].copy()
        df_surveillance = df[mask_surveillance].copy()
        df_na = df[mask_na].copy()

        logging.info(f"\n{'='*80}")
        logging.info("THREE-THREAD STRATIFICATION (AMR + Surveillance + NA)")
        logging.info(f"{'='*80}")
        logging.info(f"AMR thread (amr/AMR plus control): {len(df_amr):,} samples")
        logging.info(f"Surveillance thread: {len(df_surveillance):,} samples")
        logging.info(f"NA thread: {len(df_na):,} samples")
        logging.info("")

        # Test multiple ratios or single ratio
        if args.test_all_ratios:
            results = test_multiple_ratios(
                df_amr,  # test_multiple_ratios expects single df; run on combined or amr only
                TEST_RATIOS,
                filter_counts,
            )
            final_df = results[DEFAULT_RATIO]['stratified_df']
            # test_multiple_ratios does not support multi-thread; use AMR only for --test-all-ratios
            if not df_surveillance.empty or not df_na.empty:
                logging.info("Note: --test-all-ratios runs on AMR thread only. Surveillance and NA samples not included.")
        else:
            logging.info(f"\nUsing single ratio: {args.ratio}")

            stratified_amr = pd.DataFrame()
            log_amr = []
            stratified_surveillance = pd.DataFrame()
            log_surveillance = []

            if not df_amr.empty:
                stratified_amr, log_amr = stratify_by_location(
                    df_amr, args.ratio, thread_label="AMR"
                )
                create_detailed_report(log_amr, args.ratio, "AMR")

            if not df_surveillance.empty:
                stratified_surveillance, log_surveillance = stratify_by_location(
                    df_surveillance, args.ratio, thread_label="Surveillance"
                )
                create_detailed_report(log_surveillance, args.ratio, "Surveillance")

            stratified_na = pd.DataFrame()
            if not df_na.empty:
                stratified_na, log_na = stratify_by_location(
                    df_na, args.ratio, thread_label="NA"
                )
                create_detailed_report(log_na, args.ratio, "NA")

            # Combine
            parts = [p for p in [stratified_amr, stratified_surveillance, stratified_na] if not p.empty]
            final_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

            # Combined summary
            logging.info(f"\n{'='*80}")
            logging.info("COMBINED STRATIFICATION SUMMARY")
            logging.info(f"{'='*80}")
            if not stratified_amr.empty:
                amr_counts = stratified_amr['isolation_source_category'].value_counts()
                n_blood = amr_counts.get('blood', 0)
                n_faeces = amr_counts.get('faeces & rectal swabs', 0)
                logging.info(f"AMR thread: {len(stratified_amr):,} samples (blood: {n_blood:,}, faeces: {n_faeces:,})")
            if not stratified_surveillance.empty:
                surv_counts = stratified_surveillance['isolation_source_category'].value_counts()
                n_blood = surv_counts.get('blood', 0)
                n_faeces = surv_counts.get('faeces & rectal swabs', 0)
                logging.info(f"Surveillance thread: {len(stratified_surveillance):,} samples (blood: {n_blood:,}, faeces: {n_faeces:,})")
            if not stratified_na.empty:
                na_counts = stratified_na['isolation_source_category'].value_counts()
                n_blood = na_counts.get('blood', 0)
                n_faeces = na_counts.get('faeces & rectal swabs', 0)
                logging.info(f"NA thread: {len(stratified_na):,} samples (blood: {n_blood:,}, faeces: {n_faeces:,})")
            if not final_df.empty:
                final_counts = final_df['isolation_source_category'].value_counts()
                n_blood = final_counts.get('blood', 0)
                n_faeces = final_counts.get('faeces & rectal swabs', 0)
                logging.info(f"Combined total: {len(final_df):,} samples (blood: {n_blood:,}, faeces: {n_faeces:,})")

            # Final table by country
            df_init_parts = [p for p in [df_amr, df_surveillance, df_na] if not p.empty]
            df_init = pd.concat(df_init_parts, ignore_index=True) if df_init_parts else pd.DataFrame()
            _log_final_country_table(df_init, final_df)
        
        # Save to CSV if requested
        if args.output_csv and not final_df.empty:
            final_df.to_csv(args.output_csv, index=False, sep='\t')
            logging.info(f"\nStratified samples saved to: {args.output_csv}")
        
        logging.info("\n" + "=" * 80)
        logging.info("STRATIFICATION COMPLETE")
        logging.info("=" * 80)
        
    except Exception as e:
        logging.error(f"\nError during stratification: {e}")
        import traceback
        logging.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
