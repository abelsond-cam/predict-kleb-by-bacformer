#!/usr/bin/env python
"""Command-line script to process Klebsiella AST data.

This script loads EBI AMR records, converts resistance phenotypes to binary,
converts MIC values to log scale, filters antibiotics, creates metadata and
pivot tables, and generates visualizations.

Usage:
    python src/bacotype/pp/process_klebsiella_ast.py
    
Or with uv:
    uv run python src/bacotype/pp/process_klebsiella_ast.py
"""

from bacotype.pp.convert_ast_data import process_klebsiella_ast_data


def main():
    """Main entry point for the AST data processing pipeline."""
    # Input file path
    input_file = '/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/raw/klebsiella_ebi_amr_records_20260216.csv'
    
    print("=" * 80)
    print("KLEBSIELLA AST DATA PROCESSING - COMMAND LINE SCRIPT")
    print("=" * 80)
    print(f"\nInput file: {input_file}")
    print("Minimum antibiotic count: 1000")
    print("\n" + "=" * 80 + "\n")
    
    try:
        # Run the pipeline
        results = process_klebsiella_ast_data(
            input_file=input_file,
            min_antibiotic_count=1000
        )
        
        print("\n" + "=" * 80)
        print("PROCESSING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nOutput files created:")
        for key, path in results['paths'].items():
            print(f"  - {key}: {path}")
        print(f"\nTotal antibiotics kept: {len(results['kept_antibiotics'])}")
        print(f"Metadata samples: {len(results['metadata'])}")
        print(f"Binary AST shape: {results['binary_ast'].shape}")
        print(f"Regression log_mic shape: {results['regression_log_mic'].shape}")
        
    except FileNotFoundError as e:
        print("\n" + "=" * 80)
        print("ERROR: INPUT FILE NOT FOUND!")
        print("=" * 80)
        print(f"\nThe input file does not exist: {input_file}")
        print("\nPlease check that the file path is correct and the file exists.")
        print(f"\nError details: {e}")
        return 1
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("ERROR: PROCESSING FAILED!")
        print("=" * 80)
        print(f"\nError: {e}")
        print("\nFull traceback:")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
