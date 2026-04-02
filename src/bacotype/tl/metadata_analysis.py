#!/usr/bin/env python3
"""
Analyze parsed metadata from metadata_curation.py output.

Generates:
1. Total sample count
2. Species distribution histogram (top 10, log scale)
3. K. pneumoniae ST groups histogram with cumulative line
"""

import argparse
import sys
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Global color mappings and configurations
ISOLATION_SOURCE_COLORS = {
    'urine, urinary catheter': '#FFF9C4',  # pale non-fluorescent yellow
    'urine': '#FFF9C4',  # pale non-fluorescent yellow
    'faeces & rectal swabs': '#6B3F13',  # dark poo brown
    'blood': '#8B0000',  # dark non fluorescent red
    'upper airway': 'steelblue',
    'lower respiratory, endotracheal': 'steelblue'
}

# Stack order for isolation sources (bottom to top)
ISOLATION_SOURCE_STACK_ORDER = [
    'urine, urinary catheter',
    'faeces & gut',
    'upper airway',
    'lower respiratory, endotracheal',
    'blood'
]

REGION_COLORS = {
    'N. America': '#4575b4',      # Medium blue
    'W. Europe': '#91bfdb',       # Light blue
    'E. Asia': '#fee090',         # Light orange
    'Africa': '#fc8d59',          # Medium orange
    'E. Europe': '#d73027',       # Red
    'Oceania': '#91cf60',         # Light green
    'Central & S. America': '#1a9850',  # Dark green
    'M. East, Central Asia': '#8c510a'  # Brown
}

# Stack order for regions (bottom to top)
REGION_STACK_ORDER = [
    'Central & S. America',
    'Oceania',
    'E. Europe',
    'Africa',
    'M. East, Central Asia',
    'E. Asia',
    'W. Europe',
    'N. America',
]

# Host category colors and stack order
HOST_CATEGORY_COLORS = {
    'animals (other)': '#8B4513',  # SaddleBrown
    'birds (other)': '#87CEEB',  # SkyBlue
    'clinical environment': '#9370DB',  # MediumPurple
    'domestic dogs and cats': '#FFA500',  # Orange
    'grazing livestock': '#228B22',  # ForestGreen
    'human': '#FFFDD0',  # Cream
    'insect': '#FFD700',  # Gold
    'poultry': '#FF6347',  # Tomato
    'poultry livestock': '#FF6347',  # Tomato (alternative name)
    'wastewater, water': '#4682B4',  # SteelBlue
    'Missing': 'gray'
}

HOST_CATEGORY_STACK_ORDER = [
    'human',  # First (bottom of stack) - dominates
    'wastewater & water',
    'grazing livestock',
    'poultry livestock',  # Alternative name
    'vegetable, plant or soil',  # Alternative name
    'domestic dogs and cats',
    'clinical environment or surface',
    'meat products',
    'animals (other)',
    'birds (other)',
    'insect',
    'Missing',
]

# AMR study colors and stack order
AMR_STUDY_COLORS = {
    'Surveillance': 'steelblue',
    'AMR': 'darkred',
    'AMR plus control': '#CC5500',  # burnt orange
    'Missing': 'gray'
}

AMR_STUDY_STACK_ORDER = [
    'AMR',
    'AMR plus control',
    'Surveillance',
    'Missing'
]

# Study setting colors and stack order
STUDY_SETTING_COLORS = {
    'Hospital': 'steelblue',
    'Mixed': 'lightblue',
    'Community': 'darkgreen',
    'Missing': 'gray'
}

STUDY_SETTING_STACK_ORDER = [
    'Hospital',
    'Mixed',
    'Community',
    'Missing'
]

# Descriptor configuration mappings
DESCRIPTOR_COLORS = {
    'isolation_source_category': ISOLATION_SOURCE_COLORS,
    'region': REGION_COLORS,
    'amr_study': AMR_STUDY_COLORS,
    'study_setting': STUDY_SETTING_COLORS,
    'host_catetory': HOST_CATEGORY_COLORS
}

DESCRIPTOR_STACK_ORDER = {
    'isolation_source_category': ISOLATION_SOURCE_STACK_ORDER,
    'region': REGION_STACK_ORDER,
    'amr_study': AMR_STUDY_STACK_ORDER,
    'study_setting': STUDY_SETTING_STACK_ORDER,
    'host_catetory': HOST_CATEGORY_STACK_ORDER
}

def get_subspecies_config(subspecies_choice: str) -> tuple[str, str, str]:
    """Map subspecies CLI choice to DataFrame column, label, and filename token.
    
    Args:
        subspecies_choice: One of 'ST', 'CG', 'SL' (case-insensitive)
    
    Returns:
        Tuple of (column_name, label_text, filename_token) for the chosen subspecies dimension
    """
    choice = subspecies_choice.upper()
    mapping = {
        'ST': ('ST', 'Sequence Type', 'st'),
        'CG': ('Clonal group', 'Clonal Group', 'cg'),
        'SL': ('Sublineage', 'Sub-Lineage', 'sl')
    }
    
    if choice not in mapping:
        raise ValueError(f"Invalid subspecies choice '{subspecies_choice}'. Must be one of: ST, CG, SL")
    
    return mapping[choice]


def load_data(file_path: str) -> pd.DataFrame:
    """Load parsed metadata TSV file."""
    print(f"Loading data from: {file_path}")
    df = pd.read_csv(file_path, sep="\t", low_memory=False)
    print(f"Loaded {len(df):,} rows")
    return df


def plot_all_kleb_species_histogram(df: pd.DataFrame, output_path: Path) -> None:
    """Create histogram of top 10 species with log scale, separated by is_kpsc flag."""
    print("Creating species histogram...")
    
    # Check if is_kpsc column exists, fallback to kpsc if needed
    kpsc_col = None
    if 'is_kpsc' in df.columns:
        kpsc_col = 'is_kpsc'
    elif 'kpsc' in df.columns:
        kpsc_col = 'kpsc'
        print("Warning: 'is_kpsc' column not found. Using 'kpsc' column instead.")
    else:
        print("Warning: Neither 'is_kpsc' nor 'kpsc' column found. Using default behavior without Kpsc separation.")
        # Fallback to original behavior
        species_counts = df['species'].value_counts()
        top_10 = species_counts.head(10)
        top_10_sorted = top_10.sort_values(ascending=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(top_10_sorted)), top_10_sorted.values, 
                color='steelblue', edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(len(top_10_sorted)))
        ax.set_yticklabels(top_10_sorted.index, fontsize=9)
        ax.set_xscale('log')
        ax.set_xlabel('Number of Samples (log scale)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Species', fontsize=11, fontweight='bold')
        ax.set_title('Top 10 Species Distribution', fontsize=13, fontweight='bold', pad=15)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        for i, (idx, val) in enumerate(top_10_sorted.items()):
            ax.text(val, i, f' {val:,}', va='center', fontsize=8, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved species histogram to: {output_path}")
        plt.close()
        return
    
    # Count species
    species_counts = df['species'].value_counts()
    
    # Create a mapping of species to is_kpsc status (use the most common is_kpsc value for each species)
    species_kpsc_map = df.groupby('species')[kpsc_col].apply(lambda x: x.mode()[0] if len(x.mode()) > 0 else False).to_dict()
    
    # Get top 10 species
    top_10 = species_counts.head(10)
    
    # Split into is_kpsc=True and is_kpsc=False
    kpsc_true_species = [sp for sp in top_10.index if species_kpsc_map.get(sp, False)]
    kpsc_false_species = [sp for sp in top_10.index if not species_kpsc_map.get(sp, False)]
    
    # Sort each group by count (ascending for plotting from bottom to top)
    kpsc_true_counts = species_counts[kpsc_true_species].sort_values(ascending=True)
    kpsc_false_counts = species_counts[kpsc_false_species].sort_values(ascending=True)
    
    # Calculate total counts for each group
    total_kpsc_true = kpsc_true_counts.sum()
    total_kpsc_false = kpsc_false_counts.sum()
    
    # Combine: is_kpsc=False at bottom (plotted first), is_kpsc=True at top (plotted second)
    combined_species = pd.concat([kpsc_false_counts, kpsc_true_counts])
    
    # Create colors array: grey for is_kpsc=False, steelblue for is_kpsc=True
    colors = ['grey' if not species_kpsc_map.get(sp, False) else 'steelblue' 
              for sp in combined_species.index]
    
    print(f"  is_kpsc=True species: {len(kpsc_true_species)}")
    print(f"  is_kpsc=False species: {len(kpsc_false_species)}")
    print(f"  Total is_kpsc=True samples: {total_kpsc_true:,}")
    print(f"  Total is_kpsc=False samples: {total_kpsc_false:,}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create horizontal bar chart
    ax.barh(range(len(combined_species)), combined_species.values, 
            color=colors, edgecolor='black', linewidth=0.5)
    
    # Set y-axis labels
    ax.set_yticks(range(len(combined_species)))
    ax.set_yticklabels(combined_species.index, fontsize=9)
    
    # Set log scale for x-axis
    ax.set_xscale('log')
    ax.set_xlabel('Number of Samples (log scale)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Species', fontsize=11, fontweight='bold')
    ax.set_title('Top 10 Species Distribution (is_kpsc in steel blue)', fontsize=13, fontweight='bold', pad=15)
    
    # Add grid
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add value labels on bars
    for i, (idx, val) in enumerate(combined_species.items()):
        ax.text(val, i, f' {val:,}', va='center', fontsize=8, fontweight='bold')
    
    # Add total count labels at specified positions (using axes coordinates)
    ax.text(0.8, 0.8, f'n={total_kpsc_true:,}', transform=ax.transAxes, 
            fontsize=11, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='steelblue', alpha=0.3))
    ax.text(0.8, 0.2, f'n={total_kpsc_false:,}', transform=ax.transAxes, 
            fontsize=11, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='grey', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved species histogram to: {output_path}")
    plt.close()


########################### METADATA CROSS TAB PLOTS ###########################

def plot_all_kleb_metadata_counts(df: pd.DataFrame, output_path: Path, x_axis: str, stack_by: str, top_n: int = 7, title: str = None) -> None:
    """Create stacked count bar chart for non-ST plots.
    
    Generic function for count plots where x-axis is not ST groups (uses crosstab).
    
    Args:
        df: DataFrame with metadata
        output_path: Path to save the plot
        x_axis: Column for x-axis ('isolation_source_category', 'amr_study', 'study_setting')
        stack_by: Column for stacking
        top_n: Number of top x-axis categories (default: 7)
        title: Optional custom title
    """
    print(f"Creating {x_axis} counts plot stacked by {stack_by}...")
    
    # Check if columns exist
    if x_axis not in df.columns or stack_by not in df.columns:
        print("Warning: Required columns not found. Skipping plot.")
        return
    
    df_plot = df.copy()
    
    # Prepare x_axis column
    if x_axis == 'isolation_source_category':
        # Filter out unhelpful categories
        unhelpful_patterns = ["(unhelpful)", "(other)", "(not specified)"]
        x_counts_all = df_plot[x_axis].value_counts()
        x_counts_all = x_counts_all[x_counts_all.index.notna()]
        
        if len(x_counts_all) > 0:
            mask = pd.Series([True] * len(x_counts_all), index=x_counts_all.index)
            for pattern in unhelpful_patterns:
                mask = mask & ~x_counts_all.index.astype(str).str.contains(pattern, na=False, regex=False)
            x_categories = x_counts_all[mask].head(top_n).index.tolist()
        else:
            x_categories = []
        
        if len(x_categories) == 0:
            print(f"Warning: No valid {x_axis} categories found. Skipping plot.")
            return
        
        df_plot = df_plot[df_plot[x_axis].isin(x_categories)].copy()
        df_plot = df_plot[df_plot[x_axis].notna()].copy()
        x_col_to_use = x_axis
    else:
        # For amr_study and study_setting: create clean column with fillna('Missing')
        x_col_to_use = f'{x_axis}_clean'
        df_plot[x_col_to_use] = df_plot[x_axis].fillna('Missing')
    
    # Prepare stack_by column
    if stack_by == 'isolation_source_category':
        # No fillna for isolation_source
        stack_col_to_use = stack_by
    else:
        # For amr_study and study_setting: create clean column
        stack_col_to_use = f'{stack_by}_clean'
        df_plot[stack_col_to_use] = df_plot[stack_by].fillna('Missing')
    
    if len(df_plot) == 0:
        print("Warning: No samples after filtering. Skipping plot.")
        return
    
    # Create crosstab
    crosstab = pd.crosstab(df_plot[x_col_to_use], df_plot[stack_col_to_use])
    
    # Reorder rows (x-axis)
    if x_axis == 'isolation_source_category':
        # Sort by total descending
        crosstab['_total'] = crosstab.sum(axis=1)
        crosstab = crosstab.sort_values('_total', ascending=False)
        crosstab = crosstab.drop('_total', axis=1)
    else:
        # Use DESCRIPTOR_STACK_ORDER for fixed categories
        if x_axis in DESCRIPTOR_STACK_ORDER:
            x_order = DESCRIPTOR_STACK_ORDER[x_axis]
            x_order = [cat for cat in x_order if cat in crosstab.index]
            crosstab = crosstab.reindex(x_order)
    
    # Reorder columns (stack) using DESCRIPTOR_STACK_ORDER
    if stack_by in DESCRIPTOR_STACK_ORDER:
        stack_order = DESCRIPTOR_STACK_ORDER[stack_by]
        for col in stack_order:
            if col not in crosstab.columns:
                crosstab[col] = 0
        crosstab = crosstab[stack_order]
    
    # Get colors for stack_by
    if stack_by in DESCRIPTOR_COLORS:
        colors = [DESCRIPTOR_COLORS[stack_by].get(col, 'gray') for col in crosstab.columns]
    else:
        colors = plt.cm.tab10(range(len(crosstab.columns)))
    
    # Truncate x-axis labels if isolation_source_category
    if x_axis == 'isolation_source_category':
        truncated_labels = []
        for label in crosstab.index:
            words = str(label).split()[:3]
            truncated_labels.append(' '.join(words))
    else:
        truncated_labels = list(crosstab.index)
    
    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot stacked bars
    crosstab.plot(kind='bar', stacked=True, ax=ax, width=0.8, color=colors)
    
    # Set axis labels
    ax.set_xticks(range(len(crosstab)))
    ax.set_xticklabels(truncated_labels, rotation=45, ha='right')
    ax.set_ylabel('Number of Samples', fontsize=12)
    
    # Set title
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        x_title_map = {
            'isolation_source_category': 'Isolation Source Category',
            'amr_study': 'AMR Study Type',
            'study_setting': 'Study Setting'
        }
        stack_title_map = {
            'isolation_source_category': 'Isolation Source',
            'amr_study': 'AMR Study Type',
            'study_setting': 'Study Setting'
        }
        x_title = x_title_map.get(x_axis, x_axis.replace('_', ' ').title())
        stack_title = stack_title_map.get(stack_by, stack_by.replace('_', ' ').title())
        ax.set_title(f'{stack_title} by {x_title}', fontsize=14, fontweight='bold')
    
    ax.grid(axis='y', alpha=0.3)
    
    # Set legend labels: replace "Missing" with "Not reviewed" for amr_study
    legend_labels = []
    for col in crosstab.columns:
        if stack_by == 'amr_study' and col == 'Missing':
            legend_labels.append("Not reviewed")
        else:
            legend_labels.append(col)
    
    stack_title_for_legend = {
        'isolation_source_category': 'Isolation Source',
        'amr_study': 'AMR Study Type',
        'study_setting': 'Study Setting'
    }
    legend_title = stack_title_for_legend.get(stack_by, stack_by.replace('_', ' ').title())
    ax.legend(title=legend_title, labels=legend_labels, bbox_to_anchor=(1.02, 1), loc='upper left')
    
    # Add total count labels on top
    for i, (idx, row) in enumerate(crosstab.iterrows()):
        total = row.sum()
        if total > 0:
            ax.text(i, total, f'{int(total):,}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved counts plot to: {output_path}")
    plt.close()


########################### METADATA PROPORTIONS PLOTS ###########################
def plot_all_kleb_metadata_proportions(df: pd.DataFrame, output_path: Path, x_axis: str, stack_by: str, top_n: int = None) -> None:
    """Create stacked proportion bar chart for non-ST plots.
    
    Generic function for proportion plots where x-axis is not ST groups.
    
    Note: Expects df to be pre-filtered to K. pneumoniae samples with kpsc_final_list=True.
    
    Args:
        df: DataFrame with K. pneumoniae metadata (pre-filtered)
        output_path: Path to save the plot
        x_axis: Column name for x-axis (e.g., 'region')
        stack_by: Column name for stacking (e.g., 'isolation_source_category')
        top_n: Optional limit on x-axis categories
    """
    print(f"Creating {x_axis} proportions plot stacked by {stack_by}...")
    print(f"Using {len(df):,} K. pneumoniae samples")
    
    # Check if columns exist
    if x_axis not in df.columns or stack_by not in df.columns:
        print("Warning: Required columns not found. Skipping plot.")
        return
    
    # Remove rows with missing values
    filtered = df[df[x_axis].notna() & df[stack_by].notna()].copy()
    
    if len(filtered) == 0:
        print("Warning: No samples found. Skipping plot.")
        return
    
    # Get x-axis categories ordered by count (descending)
    x_counts = filtered[x_axis].value_counts()
    # Filter to only include categories with > 100 samples
    x_counts = x_counts[x_counts > 5000]
    if top_n:
        x_categories = x_counts.head(top_n).index.tolist()
    else:
        x_categories = x_counts.index.tolist()
    
    # Get stack categories - for isolation_source use top 4, for others use all
    if stack_by == 'isolation_source_category':
        stack_counts = filtered[stack_by].value_counts()
        stack_categories = stack_counts.head(4).index.tolist()
    else:
        stack_categories = filtered[stack_by].unique().tolist()
    
    # Order categories using DESCRIPTOR_STACK_ORDER if available
    if stack_by in DESCRIPTOR_STACK_ORDER:
        stack_order_list = DESCRIPTOR_STACK_ORDER[stack_by]
        ordered_categories = []
        for cat in stack_order_list:
            if cat in stack_categories:
                ordered_categories.append(cat)
        for cat in stack_categories:
            if cat not in ordered_categories:
                ordered_categories.append(cat)
        stack_categories = ordered_categories
    
    print(f"Plotting {len(x_categories)} {x_axis} categories")
    
    # Calculate proportions for each x-axis category
    proportions_data = {}
    sample_counts_dict = {}
    
    for x_cat in x_categories:
        x_data = filtered[filtered[x_axis] == x_cat]
        x_data_filtered = x_data[x_data[stack_by].isin(stack_categories)]
        total = len(x_data_filtered)
        sample_counts_dict[x_cat] = len(x_data)
        
        if total == 0:
            proportions_data[x_cat] = pd.Series([0.0] * len(stack_categories), index=stack_categories)
        else:
            category_counts = x_data_filtered[stack_by].value_counts()
            proportions = pd.Series(0.0, index=stack_categories)
            for cat in stack_categories:
                proportions[cat] = category_counts.get(cat, 0) / total
            proportions_data[x_cat] = proportions
    
    proportions_df = pd.DataFrame(proportions_data).T
    
    # Get colors
    if stack_by in DESCRIPTOR_COLORS:
        colors = [DESCRIPTOR_COLORS[stack_by].get(cat, 'gray') for cat in stack_categories]
    else:
        colors = plt.cm.tab10(range(len(stack_categories)))
    
    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(14, 7))
    x_pos = range(len(proportions_df))
    
    # Plot stacked bars in reverse order
    bottom = np.zeros(len(proportions_df))
    categories_reversed = list(reversed(stack_categories))
    colors_reversed = list(reversed(colors))
    
    for i, category in enumerate(categories_reversed):
        values = proportions_df[category].values
        ax.bar(x_pos, values, bottom=bottom, label=category, 
               color=colors_reversed[i], edgecolor='black', linewidth=0.5, alpha=0.8)
        bottom += values
    
    # Add n= labels above bars
    for i, x_cat in enumerate(proportions_df.index):
        count = sample_counts_dict.get(x_cat, 0)
        ax.text(i, 1.02, f'n={count:,}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Set axis labels and styling (truncate to 3 words max)
    truncated_labels = []
    for label in proportions_df.index:
        words = str(label).split()[:3]
        truncated_labels.append(' '.join(words))
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(truncated_labels, rotation=45, fontsize=9, ha='right')
    ax.set_xlabel(x_axis.replace('_', ' ').title(), fontsize=11, fontweight='bold')
    ax.set_ylabel('Proportion of all samples', fontsize=11, fontweight='bold')
    ax.set_title(f'{stack_by} distribution by {x_axis}', 
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_ylim(0, 1.3)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    
    # Add legend (reverse to match stack order)
    handles, labels_list = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels_list[::-1], loc='upper right', fontsize=9)
    
    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved proportions plot to: {output_path}")
    plt.close()



########################### SUBSPECIES (SL/ST/CG) PLOTS ###########################

def plot_kpsc_subspecies_cumulative_histogram(df: pd.DataFrame, output_path: Path, group_label: str) -> None:
    """Create histogram of K. pneumoniae subspecies groups with cumulative line.
    
    Args:
        df: K. pneumoniae DataFrame with 'subspecies_group' column (pre-filtered, non-NA)
        output_path: Path to save the plot
        group_label: Human-readable label for the subspecies dimension (e.g., "Sequence Type", "Clonal Group", "Sub-Lineage")
    
    Note: Expects df to be pre-filtered to K. pneumoniae samples with kpsc_final_list=True.
    """
    print(f"Creating K. pneumoniae {group_label} groups plot...")
    print(f"Using {len(df):,} K. pneumoniae samples")
    
    # Count subspecies groups
    subspecies_counts = df['subspecies_group'].value_counts()
    
    # Get top 50 ST groups
    top_30 = subspecies_counts.head(50)
    
    # Sum remaining into "Other"
    other_count = subspecies_counts.iloc[50:].sum() if len(subspecies_counts) > 50 else 0
    
    # Create combined series with "Other" first, then top 50 sorted lowest to highest
    if other_count > 0:
        other_series = pd.Series({'Other': other_count})
        # Sort top 50 by frequency (lowest to highest)
        top_30_sorted = top_30.sort_values(ascending=True)
        # Combine: Other first, then top 50
        subspecies_counts_sorted = pd.concat([other_series, top_30_sorted])
    else:
        # If no "Other", just sort top 50
        subspecies_counts_sorted = top_30.sort_values(ascending=True)
    
    print(f"Plotting {len(subspecies_counts_sorted)} ST groups (top 50 + Other)")
    
    # Convert counts to percentages
    total_samples = len(df)
    subspecies_percentages = (subspecies_counts_sorted / total_samples) * 100
    
    # Create figure with dual axes
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    # Determine colors for each bar based on position
    # Bars are sorted ascending (lowest to highest), so:
    # - "Other" is at position 0 (if it exists)
    # - The LAST bars are the most common STs (top 10)
    colors = []
    num_bars = len(subspecies_percentages)
    
    for i, subspecies_name in enumerate(subspecies_percentages.index):
        if subspecies_name == 'Other':
            # Other category: light blue
            colors.append('lightblue')
        else:
            # Calculate position from the end (excluding "Other")
            # If "Other" exists, it's at position 0, so real STs start at position 1
            has_other = 'Other' in subspecies_percentages.index
            if has_other:
                position_from_end = num_bars - 1 - i  # Position from the end
            else:
                position_from_end = num_bars - 1 - i
            
            # Top 10 STs are the last 10 bars (position from end: 0-9)
            if position_from_end < 10:
                colors.append('darkblue')
            # Everything else is light blue
            else:
                colors.append('lightblue')
    
    # Create histogram on primary axis (using percentages) with color coding
    ax1.bar(range(len(subspecies_percentages)), subspecies_percentages.values,
            color=colors, edgecolor='black', linewidth=0.3, alpha=0.7)
    
    # Set x-axis labels
    ax1.set_xticks(range(len(subspecies_percentages)))
    ax1.set_xticklabels(subspecies_percentages.index, rotation=90, fontsize=7, ha='center')
    ax1.set_xlabel(f'{group_label} Group', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Percent of All Samples (%)', fontsize=11, fontweight='bold', color='darkblue')
    ax1.tick_params(axis='y', labelcolor='darkblue')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)
    
    # Calculate cumulative percentage (based on percentages)
    cumulative_pct = subspecies_percentages.cumsum()
    
    # Smooth the cumulative line using rolling mean
    window_size = max(3, len(cumulative_pct) // 50)  # Adaptive window size
    smoothed_pct = cumulative_pct.rolling(window=window_size, center=True, min_periods=1).mean()
    
    # Create secondary axis for cumulative line
    ax2 = ax1.twinx()
    ax2.plot(range(len(cumulative_pct)), smoothed_pct,
             color='red', linewidth=2.5, label='Cumulative % (smoothed)')
    ax2.set_ylabel('Cumulative Percentage (%)', fontsize=11, fontweight='bold', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, 100)
    
    # Add legend
    ax2.legend(loc='upper right', fontsize=9)
    
    ax1.set_title(f'K. pneumoniae {group_label} Distribution',
                  fontsize=13, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved {group_label} groups plot to: {output_path}")
    plt.close()


def plot_kpsc_subspecies_proportions(df: pd.DataFrame, output_path: Path, stack_by: str, group_label: str, subspecies_order: list = None, top_n: int = 10) -> list:
    """Create stacked proportion bar chart of K. pneumoniae subspecies groups.
    
    Generic function for subspecies-based proportion plots. Works with any descriptor variable
    that has configuration in DESCRIPTOR_COLORS and DESCRIPTOR_STACK_ORDER.
    
    Args:
        df: DataFrame with K. pneumoniae metadata (pre-filtered) with 'subspecies_group' column
        output_path: Path to save the plot
        stack_by: Column name for stacking ('isolation_source_category', 'region', etc.)
        group_label: Human-readable label for the subspecies dimension
        subspecies_order: Optional subspecies order list (if None, creates new order and returns it)
        top_n: Number of top subspecies to show individually (default: 10)
    
    Returns:
        List of subspecies group names in order
    
    Note: Expects df to be pre-filtered to K. pneumoniae samples with kpsc_final_list=True.
    """
    print(f"Creating {group_label} proportions plot stacked by {stack_by}...")
    print(f"Using {len(df):,} K. pneumoniae samples")
    
    # Check if stack_by column exists
    if stack_by not in df.columns:
        print(f"Warning: '{stack_by}' column not found. Skipping plot.")
        return None if subspecies_order is None else subspecies_order
    
    # Remove rows with missing subspecies_group or stack_by
    filtered = df[df['subspecies_group'].notna() & df[stack_by].notna()].copy()
    
    if len(filtered) == 0:
        print(f"Warning: No samples with subspecies_group and {stack_by} values found. Skipping plot.")
        return None if subspecies_order is None else subspecies_order
    
    # Get or create subspecies grouping
    subspecies_counts_all = filtered['subspecies_group'].value_counts()
    
    if subspecies_order is None:
        # Create new subspecies grouping: top_n + Other
        top_subspecies = subspecies_counts_all.head(top_n).index.tolist()
        remaining_subspecies = subspecies_counts_all.iloc[top_n:].index.tolist() if len(subspecies_counts_all) > top_n else []
        
        print(f"Grouping: {len(top_subspecies)} top {group_label}s, {len(remaining_subspecies)} in Other")
        
        # Calculate proportions and counts for each subspecies group
        proportions_data = {}
        sample_counts_dict = {}
        
        # Helper to calculate proportions
        def calc_props(subspecies_list, label, categories):
            if len(subspecies_list) == 0:
                return pd.Series([0.0] * len(categories), index=categories, name=label)
            
            group_data = filtered[filtered['subspecies_group'].isin(subspecies_list)]
            group_data_filtered = group_data[group_data[stack_by].isin(categories)]
            total = len(group_data_filtered)
            
            if total == 0:
                return pd.Series([0.0] * len(categories), index=categories, name=label)
            
            category_counts = group_data_filtered[stack_by].value_counts()
            proportions = pd.Series(0.0, index=categories, name=label)
            for cat in categories:
                proportions[cat] = category_counts.get(cat, 0) / total
            
            return proportions
        
        # Get stack categories - for isolation_source use top 4, for others use all
        if stack_by == 'isolation_source_category':
            desc_counts = filtered[stack_by].value_counts()
            stack_categories = desc_counts.head(4).index.tolist()
        else:
            # For region and others, get all unique categories
            stack_categories = filtered[stack_by].unique().tolist()
        
        # Order categories using DESCRIPTOR_STACK_ORDER if available
        if stack_by in DESCRIPTOR_STACK_ORDER:
            stack_order_list = DESCRIPTOR_STACK_ORDER[stack_by]
            ordered_categories = []
            for cat in stack_order_list:
                if cat in stack_categories:
                    ordered_categories.append(cat)
            # Add any remaining categories not in stack order
            for cat in stack_categories:
                if cat not in ordered_categories:
                    ordered_categories.append(cat)
            stack_categories = ordered_categories
        
        print(f"Stack categories for {stack_by}: {stack_categories}")
        
        # Individual top N subspecies
        for st in top_subspecies:
            subspecies_key = str(st)
            proportions_data[subspecies_key] = calc_props([st], subspecies_key, stack_categories)
            sample_counts_dict[subspecies_key] = subspecies_counts_all[st]
        
        # Other group
        if len(remaining_subspecies) > 0:
            proportions_data['Other'] = calc_props(remaining_subspecies, 'Other', stack_categories)
            sample_counts_dict['Other'] = subspecies_counts_all[remaining_subspecies].sum()
        
        # Convert to DataFrame - order with top N by count (descending), then Other at the end
        proportions_df = pd.DataFrame(proportions_data).T
        
        # Sort top subspecies by count (descending)
        top_sorted = sorted([k for k in sample_counts_dict.keys() if k != 'Other'],
                          key=lambda x: sample_counts_dict[x],
                          reverse=True)
        
        # Add "Other" at the end if it exists
        if 'Other' in sample_counts_dict:
            subspecies_order_final = top_sorted + ['Other']
        else:
            subspecies_order_final = top_sorted
        
        proportions_df = proportions_df.reindex(subspecies_order_final)
        subspecies_order_to_return = proportions_df.index.tolist()
    else:
        # Use existing ST order
        # Map ST order to ST lists
        subspecies_group_mapping = {}
        top_subspecies = []
        remaining_subspecies = []
        
        for subspecies_group_name in subspecies_order:
            if subspecies_group_name == 'Other':
                # Get all remaining subspecies beyond top_n
                remaining_subspecies = subspecies_counts_all.iloc[top_n:].index.tolist() if len(subspecies_counts_all) > top_n else []
                subspecies_group_mapping['Other'] = remaining_subspecies
            else:
                # Individual subspecies
                try:
                    subspecies_val = int(subspecies_group_name) if subspecies_group_name.isdigit() else subspecies_group_name
                    if subspecies_val in subspecies_counts_all.index:
                        subspecies_group_mapping[subspecies_group_name] = [subspecies_val]
                        if subspecies_val in subspecies_counts_all.head(top_n).index:
                            top_subspecies.append(subspecies_val)
                except (ValueError, KeyError):
                    pass
        
        # Get stack categories
        if stack_by == 'isolation_source_category':
            desc_counts = filtered[stack_by].value_counts()
            stack_categories = desc_counts.head(4).index.tolist()
        else:
            stack_categories = filtered[stack_by].unique().tolist()
        
        # Order categories
        if stack_by in DESCRIPTOR_STACK_ORDER:
            stack_order_list = DESCRIPTOR_STACK_ORDER[stack_by]
            ordered_categories = []
            for cat in stack_order_list:
                if cat in stack_categories:
                    ordered_categories.append(cat)
            for cat in stack_categories:
                if cat not in ordered_categories:
                    ordered_categories.append(cat)
            stack_categories = ordered_categories
        
        # Helper to calculate proportions
        def calc_props(subspecies_list, label, categories):
            if len(subspecies_list) == 0:
                return pd.Series([0.0] * len(categories), index=categories, name=label)
            
            group_data = filtered[filtered['subspecies_group'].isin(subspecies_list)]
            group_data_filtered = group_data[group_data[stack_by].isin(categories)]
            total = len(group_data_filtered)
            
            if total == 0:
                return pd.Series([0.0] * len(categories), index=categories, name=label)
            
            category_counts = group_data_filtered[stack_by].value_counts()
            proportions = pd.Series(0.0, index=categories, name=label)
            for cat in categories:
                proportions[cat] = category_counts.get(cat, 0) / total
            
            return proportions
        
        proportions_data = {}
        sample_counts_dict = {}
        
        for subspecies_group_name in subspecies_order:
            if subspecies_group_name in subspecies_group_mapping:
                subspecies_list = subspecies_group_mapping[subspecies_group_name]
                proportions_data[subspecies_group_name] = calc_props(subspecies_list, subspecies_group_name, stack_categories)
                
                if subspecies_group_name == 'Other':
                    sample_counts_dict[subspecies_group_name] = subspecies_counts_all[subspecies_list].sum() if subspecies_list else 0
                else:
                    subspecies_val = subspecies_list[0] if subspecies_list else None
                    if subspecies_val is not None:
                        sample_counts_dict[subspecies_group_name] = subspecies_counts_all.get(subspecies_val, 0)
        
        proportions_df = pd.DataFrame(proportions_data).T
        subspecies_order_to_return = subspecies_order
    
    print(f"Plotting {len(proportions_df)} {group_label} groups")
    
    # Get colors from DESCRIPTOR_COLORS
    if stack_by in DESCRIPTOR_COLORS:
        colors = [DESCRIPTOR_COLORS[stack_by].get(cat, 'gray') for cat in stack_categories]
    else:
        # Fallback colors
        colors = plt.cm.tab10(range(len(stack_categories)))
    
    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(14, 7))
    x_pos = range(len(proportions_df))
    
    # Plot stacked bars in reverse order (first in STACK_ORDER at bottom)
    bottom = np.zeros(len(proportions_df))
    categories_reversed = list(reversed(stack_categories))
    colors_reversed = list(reversed(colors))
    
    for i, category in enumerate(categories_reversed):
        values = proportions_df[category].values
        ax.bar(x_pos, values, bottom=bottom, label=category, 
               color=colors_reversed[i], edgecolor='black', linewidth=0.5, alpha=0.8)
        bottom += values
    
    # Add n= labels above bars
    for i, subspecies_group in enumerate(proportions_df.index):
        count = sample_counts_dict.get(subspecies_group, 0)
        ax.text(i, 1.02, f'n={count:,}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Set axis labels and styling
    ax.set_xticks(x_pos)
    ax.set_xticklabels(proportions_df.index, rotation=45, fontsize=9, ha='right')
    ax.set_xlabel(f'{group_label}', fontsize=11, fontweight='bold')
    ax.set_ylabel('Proportion of all samples', fontsize=11, fontweight='bold')
    ax.set_title(f'{stack_by} distribution by {group_label}', 
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_ylim(0, 1.5)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    
    # Add legend (reverse to match stack order)
    handles, labels_list = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels_list[::-1], loc='upper right', fontsize=9)
    
    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved {group_label} proportions plot to: {output_path}")
    plt.close()
    
    return subspecies_order_to_return


def plot_kpsc_subspecies_counts(df: pd.DataFrame, output_path: Path, stack_by: str, group_label: str) -> None:
    """Create stacked count bar chart of K. pneumoniae subspecies groups.
    
    Generic function for subspecies-based count plots (uses crosstab).
    
    Args:
        df: DataFrame with K. pneumoniae metadata (pre-filtered) with 'subspecies_group' column
        output_path: Path to save the plot
        stack_by: Column name for stacking ('amr_study', 'study_setting', etc.)
        group_label: Human-readable label for the subspecies dimension
    
    Note: Expects df to be pre-filtered to K. pneumoniae samples with kpsc_final_list=True.
    """
    print(f"Creating {group_label} counts plot stacked by {stack_by}...")
    print(f"Using {len(df):,} K. pneumoniae samples")
    
    # Check if stack_by column exists
    if stack_by not in df.columns:
        print(f"Warning: '{stack_by}' column not found. Skipping plot.")
        return
    
    # Remove rows with missing subspecies_group
    filtered = df[df['subspecies_group'].notna()].copy()
    
    if len(filtered) == 0:
        print("Warning: No samples with subspecies_group values found. Skipping plot.")
        return
    
    # Create clean column: convert NaN to 'Missing'
    stack_by_clean = f'{stack_by}_clean'
    filtered[stack_by_clean] = filtered[stack_by].fillna('Missing')
    
    # Subspecies grouping: top 30 + Other
    subspecies_counts = filtered['subspecies_group'].value_counts()
    top_30 = subspecies_counts.head(30)
    other_count = subspecies_counts.iloc[50:].sum() if len(subspecies_counts) > 50 else 0
    
    # Create subspecies_label mapping
    filtered['subspecies_label'] = filtered['subspecies_group'].apply(lambda x: str(x) if x in top_30.index else 'Other')
    
    # Create crosstab
    crosstab = pd.crosstab(filtered['subspecies_label'], filtered[stack_by_clean])
    
    # Reorder rows: Other first, then top 50 ascending by count
    if other_count > 0:
        other_series = pd.Series({'Other': other_count})
        top_30_sorted = top_30.sort_values(ascending=True)
        subspecies_counts_sorted = pd.concat([other_series, top_30_sorted])
    else:
        subspecies_counts_sorted = top_30.sort_values(ascending=True)
    
    subspecies_order = []
    for st in subspecies_counts_sorted.index:
        subspecies_label = 'Other' if st == 'Other' else str(st)
        if subspecies_label in crosstab.index:
            subspecies_order.append(subspecies_label)
    crosstab = crosstab.reindex(subspecies_order)
    
    # Reorder columns using DESCRIPTOR_STACK_ORDER
    if stack_by in DESCRIPTOR_STACK_ORDER:
        stack_order_list = DESCRIPTOR_STACK_ORDER[stack_by]
        for col in stack_order_list:
            if col not in crosstab.columns:
                crosstab[col] = 0
        crosstab = crosstab[stack_order_list]
    
    # Get colors from DESCRIPTOR_COLORS
    if stack_by in DESCRIPTOR_COLORS:
        colors = [DESCRIPTOR_COLORS[stack_by].get(col, 'gray') for col in crosstab.columns]
    else:
        colors = plt.cm.tab10(range(len(crosstab.columns)))
    
    print(f"Plotting {len(crosstab)} {group_label} groups (top 50 + Other)")
    
    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot stacked bars
    crosstab.plot(kind='bar', stacked=True, ax=ax, width=0.8, color=colors)
    
    # Set x-axis labels
    ax.set_xticks(range(len(crosstab)))
    ax.set_xticklabels(crosstab.index, rotation=90, fontsize=7, ha='center')
    ax.set_xlabel(f'{group_label} Group', fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Samples', fontsize=11, fontweight='bold')
    
    # Create title
    title_map = {
        'amr_study': 'AMR Study Type',
        'study_setting': 'Study Setting'
    }
    title_text = title_map.get(stack_by, stack_by.replace('_', ' ').title())
    ax.set_title(f'{title_text} by {group_label}', fontsize=13, fontweight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Set legend labels: replace "Missing" with "Not reviewed" for amr_study
    legend_labels = []
    for col in crosstab.columns:
        if stack_by == 'amr_study' and col == 'Missing':
            legend_labels.append("Not reviewed")
        else:
            legend_labels.append(col)
    ax.legend(title=title_text, labels=legend_labels, bbox_to_anchor=(1.02, 1), loc='upper left')
    
    # Add total count labels on top
    for i, (idx, row) in enumerate(crosstab.iterrows()):
        total = row.sum()
        if total > 0:
            ax.text(i, total, f'{int(total):,}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved {group_label} counts plot to: {output_path}")
    plt.close()


def plot_kpsc_subspecies_by_K_locus(df: pd.DataFrame, output_path: Path, subspecies_descriptor: str = 'Sublineage', top_n: int = 10) -> None:
    """Create stacked bar chart of subspecies groups by K_locus.
    
    Shows all K_locus values as separate segments with black dividers.
    Only labels segments that comprise >5% of the subspecies group.
    
    Args:
        df: DataFrame with K. pneumoniae metadata (pre-filtered)
        output_path: Path to save the plot
        subspecies_descriptor: Column name for subspecies grouping ('Sublineage' or 'Clonal group')
        top_n: Number of top subspecies to show individually (remaining grouped as "Other")
    
    Note: Expects df to be pre-filtered to K. pneumoniae samples with kpsc_final_list=True.
    """
    # Map subspecies_descriptor to column and display label
    descriptor_mapping = {
        'Sublineage': ('Sublineage', 'Sub-Lineage'),
        'Clonal group': ('Clonal group', 'Clonal Group')
    }
    
    if subspecies_descriptor not in descriptor_mapping:
        raise ValueError(f"Invalid subspecies_descriptor '{subspecies_descriptor}'. Must be 'Sublineage' or 'Clonal group'")
    
    column_name, group_label = descriptor_mapping[subspecies_descriptor]
    
    print(f"Creating {group_label} by K_locus plot...")
    print(f"Using {len(df):,} K. pneumoniae samples")
    
    # Remove rows with missing column or K_locus
    filtered = df[df[column_name].notna() & df['K_locus'].notna()].copy()
    
    # Count subspecies groups
    subspecies_counts_all = filtered[column_name].value_counts()
    
    # Get top N subspecies and combine rest as "Other"
    top_subspecies = subspecies_counts_all.head(top_n).index.tolist()
    remaining_subspecies = subspecies_counts_all.iloc[top_n:].index.tolist() if len(subspecies_counts_all) > top_n else []
    
    # Create list of groups to plot (top_n + "Other")
    groups_to_plot = [str(st) for st in top_subspecies]
    if len(remaining_subspecies) > 0:
        groups_to_plot.append('Other')
    
    # Calculate K_locus proportions for each group
    all_k_locus_data = {}  # Dict of {group_name: {k_locus: proportion}}
    sample_counts_dict = {}
    
    for group_name in groups_to_plot:
        if group_name == 'Other':
            # Combine all remaining subspecies
            group_data = filtered[filtered[column_name].isin(remaining_subspecies)]
        else:
            # Individual subspecies group
            group_data = filtered[filtered[column_name] == group_name]
        
        total = len(group_data)
        sample_counts_dict[group_name] = total
        
        if total == 0:
            all_k_locus_data[group_name] = {}
            continue
        
        # Count ALL K_locus values for this group
        locus_counts = group_data['K_locus'].value_counts()
        
        # Calculate proportions for ALL K_locus values
        k_locus_proportions = {}
        for locus, count in locus_counts.items():
            k_locus_proportions[str(locus)] = count / total
        
        all_k_locus_data[group_name] = k_locus_proportions
    
    print(f"Plotting {len(groups_to_plot)} {group_label} groups (top {top_n} + Other)")
    
    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Single color for all segments
    bar_color = 'steelblue'
    
    # Plot each group
    x_pos = range(len(groups_to_plot))
    
    for bar_idx, group_name in enumerate(groups_to_plot):
        k_locus_proportions = all_k_locus_data.get(group_name, {})
        
        if not k_locus_proportions:
            continue
        
        # Sort K_locus by proportion (descending) for consistent stacking
        sorted_loci = sorted(k_locus_proportions.items(), key=lambda x: x[1], reverse=True)
        
        current_bottom = 0.0
        
        for locus_label, proportion in sorted_loci:
            # Plot segment
            ax.bar(bar_idx, proportion, bottom=current_bottom,
                   color=bar_color, edgecolor='black', linewidth=1.0, alpha=0.8)
            
            # Add label only if segment >= 5%
            if proportion >= 0.05:
                label_y = current_bottom + proportion / 2
                ax.text(bar_idx, label_y, locus_label,
                       ha='center', va='center',
                       fontsize=7, fontweight='bold',
                       color='white',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6, edgecolor='white', linewidth=0.5))
            
            current_bottom += proportion
    
    # Add sample count labels above bars (n=XXX)
    for i, group_name in enumerate(groups_to_plot):
        if group_name in sample_counts_dict:
            count = sample_counts_dict[group_name]
            ax.text(i, 1.02, f'n={count:,}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Set x-axis labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(groups_to_plot, rotation=45, fontsize=9, ha='right')
    ax.set_xlabel(f'{group_label}', fontsize=11, fontweight='bold')
    ax.set_ylabel('Proportion of all samples', fontsize=11, fontweight='bold')
    ax.set_title(f'K_locus by {group_label}',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_ylim(0, 1.1)
    
    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved {group_label} by K_locus plot to: {output_path}")
    plt.close()


def plot_kpsc_subspecies_by_isolation_source_by_region(df: pd.DataFrame, output_path: Path, subspecies_order: list, group_label: str, top_n: int = 10) -> None:
    """Create stacked bar chart of subspecies groups by isolation source category, one subplot per region.
    
    Uses the same subspecies groups in the same order as the overall isolation source plot.
    For subspecies groups with < 100 samples in a region, bars are left blank but labels and counts are shown.
    
    Args:
        df: DataFrame with K. pneumoniae metadata (pre-filtered) with 'subspecies_group' column
        output_path: Path to save the plot
        subspecies_order: List of subspecies group names in the order they should appear (from overall isolation source plot)
        group_label: Human-readable label for the subspecies dimension
        top_n: Number of top subspecies to show individually (used for grouping logic)
    
    Note: Expects df to be pre-filtered to K. pneumoniae samples with kpsc_final_list=True.
    """
    print(f"Creating {group_label} by isolation source by region plot...")
    print(f"Using {len(df):,} K. pneumoniae samples")
    
    # Remove rows with missing subspecies_group, isolation_source_category, or region
    filtered = df[
        df['subspecies_group'].notna() & 
        df['isolation_source_category'].notna() &
        df['region'].notna()
    ].copy()
    
    # Count subspecies groups to identify subspecies lists (for grouping logic)
    subspecies_counts_all = filtered['subspecies_group'].value_counts()
    
    # Get top N subspecies and remaining (2-tier grouping: top_n + Other)
    top_subspecies = subspecies_counts_all.head(top_n).index.tolist()
    remaining_subspecies = subspecies_counts_all.iloc[top_n:].index.tolist() if len(subspecies_counts_all) > top_n else []
    
    # Create mapping from subspecies group name to subspecies list
    subspecies_group_mapping = {}
    for st in top_subspecies:
        subspecies_group_mapping[str(st)] = [st]
    if len(remaining_subspecies) > 0:
        subspecies_group_mapping['Other'] = remaining_subspecies
    
    # Identify all regions present in data
    region_counts = filtered['region'].value_counts()
    all_regions = region_counts.index.tolist()
    print(f"Found {len(all_regions)} regions: {all_regions}")
    
    # Identify top 4 isolation source categories (same as overall plot)
    iso_counts = filtered['isolation_source_category'].value_counts()
    top_4_categories = iso_counts.head(4).index.tolist()
    
    # Order categories according to ISOLATION_SOURCE_STACK_ORDER (bottom to top)
    ordered_categories = []
    for cat in ISOLATION_SOURCE_STACK_ORDER:
        if cat in top_4_categories:
            ordered_categories.append(cat)
    for cat in top_4_categories:
        if cat not in ordered_categories:
            ordered_categories.append(cat)
    top_4_categories = ordered_categories
    
    # Map exact category names to colors using global constant
    colors = [ISOLATION_SOURCE_COLORS.get(cat, 'gray') for cat in top_4_categories]
    colors_reversed = list(reversed(colors))
    categories_reversed = list(reversed(top_4_categories))
    
    # Create figure with subplots (one per region)
    n_regions = len(all_regions)
    fig, axes = plt.subplots(n_regions, 1, figsize=(14, 3 * n_regions))
    
    # Handle case where there's only one region (axes won't be an array)
    if n_regions == 1:
        axes = [axes]
    
    # Process each region
    for region_idx, region in enumerate(all_regions):
        ax = axes[region_idx]
        region_data = filtered[filtered['region'] == region].copy()
        
        # Calculate proportions for each subspecies group in this region
        proportions_data = {}
        sample_counts_regional = {}
        
        for subspecies_group_name in subspecies_order:
            if subspecies_group_name not in subspecies_group_mapping:
                # Skip if not in mapping (shouldn't happen, but be safe)
                continue
            
            subspecies_list = subspecies_group_mapping[subspecies_group_name]
            group_data = region_data[region_data['subspecies_group'].isin(subspecies_list)]
            
            # Count total samples for this subspecies group in this region
            count_in_region = len(group_data)
            sample_counts_regional[subspecies_group_name] = count_in_region
            
            # If < 100 samples, leave proportions as zeros (will show blank bar)
            if count_in_region < 100:
                proportions_data[subspecies_group_name] = pd.Series(
                    [0.0] * len(top_4_categories), 
                    index=top_4_categories, 
                    name=subspecies_group_name
                )
            else:
                # Calculate proportions (same logic as overall plot)
                group_data_top4 = group_data[group_data['isolation_source_category'].isin(top_4_categories)]
                total_top4 = len(group_data_top4)
                
                if total_top4 == 0:
                    proportions_data[subspecies_group_name] = pd.Series(
                        [0.0] * len(top_4_categories), 
                        index=top_4_categories, 
                        name=subspecies_group_name
                    )
                else:
                    category_counts = group_data_top4['isolation_source_category'].value_counts()
                    proportions = pd.Series(0.0, index=top_4_categories, name=subspecies_group_name)
                    for cat in top_4_categories:
                        count = category_counts.get(cat, 0)
                        proportions[cat] = count / total_top4
                    proportions_data[subspecies_group_name] = proportions
        
        # Convert to DataFrame (use subspecies_order to ensure correct ordering)
        proportions_df = pd.DataFrame(proportions_data).T
        proportions_df = proportions_df.reindex(subspecies_order)
        
        # Get x positions
        x_pos = range(len(proportions_df))
        
        # Plot stacked bars (only for ST groups with >= 100 samples)
        bottom = np.zeros(len(proportions_df))
        
        for i, category in enumerate(categories_reversed):
            values = proportions_df[category].values
            ax.bar(x_pos, values, bottom=bottom, label=category,
                   color=colors_reversed[i], edgecolor='black', linewidth=0.5, alpha=0.8)
            bottom += values
        
        # Add sample count labels above bars (n=XXX) - always show, even if bar is blank
        for i, subspecies_group in enumerate(subspecies_order):
            count = sample_counts_regional.get(subspecies_group, 0)
            ax.text(i, 1.02, f'n={count:,}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Set x-axis labels
        ax.set_xticks(x_pos)
        ax.set_xticklabels(proportions_df.index, rotation=90, fontsize=9, ha='center')
        ax.set_xlabel(f'{group_label} Group', fontsize=11, fontweight='bold')
        ax.set_ylabel('Proportion of all samples', fontsize=11, fontweight='bold')
        ax.set_title(region, fontsize=12, fontweight='bold', pad=10)
        ax.set_ylim(0, 1.3)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        
        # Add grid
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Add legend only to last subplot
        if region_idx == n_regions - 1:
            handles, labels_leg = ax.get_legend_handles_labels()
            ax.legend(handles[::-1], labels_leg[::-1], loc='lower right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved {group_label} by isolation source by region plot to: {output_path}")
    plt.close()





########################### MAIN FUNCTION ##################################


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Analyze parsed metadata from metadata_curation.py',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--input',
        type=str,
        default="/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/final/metadata_final_curated_slimmed.tsv",
        help='Path to parsed_metadata.tsv file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default="/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/pangenome_analysis/visualisations",
        help='Directory to save output PNG files'
    )
    parser.add_argument(
        '--top-n-st',
        type=int,
        default=10,
        help='Number of top STs to show individually (default: 10)'
    )
    parser.add_argument(
        '--plot-all-klebsiella-species',
        action='store_true',
        default=False,
        help='Plot all K. pneumoniae samples without kpsc_final_list filtering (default: False)'
    )
    parser.add_argument(
        '--subspecies',
        type=str,
        default='SL',
        help='Subspecies dimension for grouping: ST (Sequence Type), CG (Clonal Group), or SL (Sub-Lineage, default)'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    
    # Set output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging to file
    log_file_path = output_dir / 'kleb_metadata_analysis.log'
    log_capture = StringIO()
    original_stdout = sys.stdout
    
    # Redirect stdout to capture output
    sys.stdout = log_capture
    
    try:
        # Load data
        df = load_data(str(input_path))
        
        # Analyze sample counts
        """Print total number of samples."""
        total_samples = len(df)
        print(f"\n{'='*60}")
        print(f"Total number of samples: {total_samples:,}")
        print(f"{'='*60}\n")
        
        # Species-level plots (use full dataframe)
        plot_all_kleb_species_histogram(df, output_dir / 'species_histogram.png')
        
        # Filter to K. pneumoniae (and optionally to kpsc_final_list)
        if args.plot_all_klebsiella_species:
            # Use all K. pneumoniae samples without kpsc_final_list filtering
            df_kpsc = df[df['species'] == 'Klebsiella pneumoniae'].copy()
            print(f"Using all {len(df_kpsc):,} K. pneumoniae samples (kpsc_final_list filtering disabled)")
        else:
            # Default: filter to kpsc_final_list=True
            if 'kpsc_final_list' not in df.columns:
                print("Warning: 'kpsc_final_list' column not found. Using all K. pneumoniae samples.")
                df_kpsc = df[df['species'] == 'Klebsiella pneumoniae'].copy()
            else:
                df_kpsc = df[
                    (df['species'] == 'Klebsiella pneumoniae') & 
                    df['kpsc_final_list']
                ].copy()
                print(f"Filtered to {len(df_kpsc):,} K. pneumoniae samples with kpsc_final_list=True")
        
        # Configure subspecies grouping dimension
        try:
            group_col, group_label, subspecies_token = get_subspecies_config(args.subspecies)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        
        # Check that the chosen column exists
        if group_col not in df_kpsc.columns:
            print(f"Error: Column '{group_col}' not found in data. Cannot use subspecies={args.subspecies}", file=sys.stderr)
            sys.exit(1)
        
        # Create subspecies-specific dataframe with subspecies_group column
        df_kpsc_sub = df_kpsc.copy()
        df_kpsc_sub['subspecies_group'] = df_kpsc_sub[group_col]
        
        # Filter out rows with missing subspecies_group
        df_kpsc_sub = df_kpsc_sub[df_kpsc_sub['subspecies_group'].notna()].copy()
        print(f"Using subspecies dimension: {group_label} (column: {group_col})")
        print(f"After removing NA in {group_col}: {len(df_kpsc_sub):,} samples")
        
        # All subsequent subspecies-based plots use df_kpsc_sub and group_label
        plot_kpsc_subspecies_cumulative_histogram(df_kpsc_sub, output_dir / f'kpsc_{subspecies_token}_histogram.png', group_label)
        
        # Subspecies-based proportion plots (create subspecies order first, then reuse)
        subspecies_order = plot_kpsc_subspecies_proportions(df_kpsc_sub, output_dir / f'isolation_source_by_{subspecies_token}.png', 
                                               stack_by='isolation_source_category', group_label=group_label, 
                                               top_n=args.top_n_st)
        
        plot_kpsc_subspecies_proportions(df_kpsc_sub, output_dir / f'region_by_{subspecies_token}.png', 
                                    stack_by='region', group_label=group_label, 
                                    subspecies_order=subspecies_order, top_n=args.top_n_st)
        
        # K_locus plots for both Sublineage and Clonal group (independent of subspecies dimension)
        plot_kpsc_subspecies_by_K_locus(df_kpsc_sub, output_dir / 'k_locus_by_sublineage.png',
                                   subspecies_descriptor='Sublineage', top_n=args.top_n_st)
        plot_kpsc_subspecies_by_K_locus(df_kpsc_sub, output_dir / 'k_locus_by_clonal_group.png',
                                   subspecies_descriptor='Clonal group', top_n=args.top_n_st)
        
        # Specialized subspecies proportion plots (use subspecies_order from isolation_source plot)
        plot_kpsc_subspecies_by_isolation_source_by_region(df_kpsc_sub, output_dir / f'isolation_source_by_{subspecies_token}_by_region.png', 
                                                       subspecies_order=subspecies_order, group_label=group_label, top_n=args.top_n_st)
        
        # Non-subspecies proportion plot (uses full dataset df)
        plot_all_kleb_metadata_proportions(df, output_dir / 'region_by_isolation_source.png', 
                        x_axis='isolation_source_category', stack_by='region')
        
        # Subspecies-based count plots
        plot_kpsc_subspecies_counts(df_kpsc_sub, output_dir / f'amr_study_by_{subspecies_token}.png', 
                              stack_by='amr_study', group_label=group_label)
        plot_kpsc_subspecies_counts(df_kpsc_sub, output_dir / f'study_setting_by_{subspecies_token}.png', 
                              stack_by='study_setting', group_label=group_label)
        
        # Comparison of two categorical variables (not ST) - uses full dataset df
        plot_all_kleb_metadata_counts(df, output_dir / 'amr_study_by_isolation_source.png',
                   x_axis='isolation_source_category', stack_by='amr_study', top_n=12)
        plot_all_kleb_metadata_counts(df_kpsc, output_dir / 'study_setting_by_isolation_source.png',
                   x_axis='isolation_source_category', stack_by='study_setting', top_n=12)
        plot_all_kleb_metadata_counts(df_kpsc, output_dir / 'amr_study_by_study_setting.png',
                   x_axis='study_setting', stack_by='amr_study')
        # Isolation source category by host category
        plot_all_kleb_metadata_counts(df_kpsc, output_dir / 'host_catetory_by_isolation_source.png',
                   x_axis='isolation_source_category', stack_by='host_catetory', top_n=12)
        
        print(f"\n{'='*60}")
        print("Analysis complete!")
        print(f"Output files saved to: {output_dir}")
        print(f"{'='*60}\n")
        
        # Write captured output to log file
        log_content = log_capture.getvalue()
        with open(log_file_path, 'w') as f:
            f.write(log_content)
        
        # Also print to console
        sys.stdout = original_stdout
        print(log_content, end='')
        
    except Exception as e:
        # Restore stdout before error handling
        sys.stdout = original_stdout
        error_msg = f"Error: {str(e)}"
        print(error_msg, file=sys.stderr)
        
        # Write error to log file too
        with open(log_file_path, 'w') as f:
            f.write(log_capture.getvalue())
            f.write(f"\n{error_msg}\n")
        
        sys.exit(1)


if __name__ == '__main__':
    main()
