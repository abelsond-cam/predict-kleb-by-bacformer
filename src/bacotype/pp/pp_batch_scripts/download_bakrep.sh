#!/bin/bash
#SBATCH --job-name=bakrep_download
#SBATCH --output=bakrep_download_%j.out
#SBATCH --error=bakrep_download_%j.err
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=76
#SBATCH --time=00:10:00
#SBATCH --account=FLOTO-SL2-CPU

# Default values
N=10
TSV_FILE="/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/final/metadata_final_curated_slimmed.tsv"
OUTPUT_DIR="/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/raw/klebsiella_gbff"
NCORES=76
BATCH_SIZE=100  # Number of samples per batch download
SKIP_EXISTING=true  # Default to skip existing files
UPDATE_FLAGS=true  # Default to auto-update metadata flags

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --n)
            N="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --overwrite-existing)
            SKIP_EXISTING=false
            shift
            ;;
        --no-flag-update)
            UPDATE_FLAGS=false
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --n <number>              : Number of samples to process"
            echo "                              10 = test run (default)"
            echo "                              -1 = all samples"
            echo "  --batch-size <size>       : Samples per batch (default: 100)"
            echo "                              Use 1 to retry failed samples individually"
            echo "  --overwrite-existing      : Re-download even if files exist (default: skip)"
            echo "  --no-flag-update          : Skip automatic metadata flag update (default: update)"
            exit 1
            ;;
    esac
done

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate my-bio

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Extract and filter sample IDs to a temporary file
TEMP_SAMPLE_FILE=$(mktemp)
TEMP_FILTERED_FILE=$(mktemp)

echo "Filtering metadata with Python..."
if [ "$SKIP_EXISTING" = true ]; then
    echo "  - Excluding samples where bakta_gbff_downloaded = True"
else
    echo "  - Including all samples (overwrite mode)"
fi
echo "  - Including only samples where sample_accession starts with 'SAM'"

# Export TSV_FILE so Python subprocess can access it
export TSV_FILE
export SKIP_EXISTING

# Use Python to filter the metadata TSV
python3 <<'EOF' > "$TEMP_FILTERED_FILE"
import pandas as pd
import sys
import os

tsv_file = os.environ.get('TSV_FILE')
skip_existing = os.environ.get('SKIP_EXISTING', 'true').lower() == 'true'

if not tsv_file:
    print("ERROR: TSV_FILE environment variable not set", file=sys.stderr)
    sys.exit(1)

# Load metadata
df = pd.read_csv(tsv_file, sep='\t', low_memory=False)
initial_count = len(df)

# Filter 1: Exclude samples where bakta_gbff_downloaded is True (if skip_existing is enabled)
if skip_existing:
    if 'bakta_gbff_downloaded' in df.columns:
        # Handle various boolean representations
        df = df[~df['bakta_gbff_downloaded'].astype(str).str.lower().isin(['true', '1', 'yes'])]
        print(f"After bakta_gbff_downloaded filter: {len(df):,} samples", file=sys.stderr)
    else:
        print("bakta_gbff_downloaded column not found; skipping this filter", file=sys.stderr)
else:
    print("Skip-existing disabled; including all samples", file=sys.stderr)

# Filter 2: Only include samples where sample_accession starts with "SAM"
if 'sample_accession' not in df.columns:
    print("ERROR: 'sample_accession' column not found in metadata", file=sys.stderr)
    sys.exit(1)

df = df[df['sample_accession'].astype(str).str.startswith('SAM')]
print(f"After SAM prefix filter: {len(df):,} samples", file=sys.stderr)

# Output filtered sample IDs
print(f"Filtered from {initial_count:,} to {len(df):,} samples", file=sys.stderr)
for sample_id in df['sample_accession']:
    print(sample_id)
EOF

# Apply --n limit if specified
if [ "$N" -eq -1 ]; then
    echo "Using ALL filtered sample IDs..."
    cat "$TEMP_FILTERED_FILE" > "$TEMP_SAMPLE_FILE"
else
    echo "Taking first $N filtered sample IDs..."
    head -n "$N" "$TEMP_FILTERED_FILE" > "$TEMP_SAMPLE_FILE"
fi

TOTAL=$(wc -l < "$TEMP_SAMPLE_FILE")
echo "Final sample count for download: $TOTAL"

echo "Sample IDs saved to: $TEMP_SAMPLE_FILE"
echo "First few sample IDs:"
head -n 3 "$TEMP_SAMPLE_FILE"

# Cleanup temp filtered file
rm -f "$TEMP_FILTERED_FILE"

# Create log directory
LOG_DIR="${OUTPUT_DIR}/logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# Split samples into batches and create batch files
BATCH_DIR=$(mktemp -d)
echo "Creating batches of $BATCH_SIZE samples each..."
split -l $BATCH_SIZE -d "$TEMP_SAMPLE_FILE" "${BATCH_DIR}/batch_"

# Count number of batches
NUM_BATCHES=$(ls -1 ${BATCH_DIR}/batch_* | wc -l)
echo "Created $NUM_BATCHES batches"
echo "Batch files saved to: $BATCH_DIR"

# Function to download a batch of samples
download_batch() {
    local BATCH_FILE=$1
    local OUTPUT_DIR=$2
    local LOG_DIR=$3
    
    local BATCH_NAME=$(basename "$BATCH_FILE")
    local BATCH_LOG="${LOG_DIR}/${BATCH_NAME}.log"
    
    # Create comma-delimited list from batch file
    local SAMPLE_LIST=$(cat "$BATCH_FILE" | paste -sd,)
    local NUM_SAMPLES=$(cat "$BATCH_FILE" | wc -l)
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting batch: $BATCH_NAME ($NUM_SAMPLES samples)" | tee -a "$BATCH_LOG"
    echo "Samples: $SAMPLE_LIST" >> "$BATCH_LOG"
    
    if bakrep download \
        -e "$SAMPLE_LIST" \
        -d "$OUTPUT_DIR" \
        -m tool:bakta,filetype:gbff \
        >> "$BATCH_LOG" 2>&1; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS: $BATCH_NAME" | tee -a "$BATCH_LOG"
        return 0
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED: $BATCH_NAME" | tee -a "$BATCH_LOG"
        return 1
    fi
}

# Export function and variables for xargs
export -f download_batch
export OUTPUT_DIR
export LOG_DIR

# Run batch downloads in parallel
echo ""
echo "Starting parallel batch download with $NCORES cores at $(date)"
echo "Processing $NUM_BATCHES batches ($TOTAL total samples)..."
echo "Logs will be saved to: $LOG_DIR"
echo ""

# Use xargs for parallel processing
ls -1 ${BATCH_DIR}/batch_* | \
    xargs -I {} -P $NCORES bash -c 'download_batch "$1" "$2" "$3"' _ {} "$OUTPUT_DIR" "$LOG_DIR"

EXIT_CODE=$?

# Create summary report
SUMMARY_FILE="${OUTPUT_DIR}/download_summary_$(date +%Y%m%d_%H%M%S).txt"
echo "Download Summary - $(date)" > "$SUMMARY_FILE"
echo "===========================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Total samples: $TOTAL" >> "$SUMMARY_FILE"
echo "Total batches: $NUM_BATCHES" >> "$SUMMARY_FILE"
echo "Batch size: $BATCH_SIZE samples/batch" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Successful batches: $(grep -l "SUCCESS:" ${LOG_DIR}/batch_*.log 2>/dev/null | wc -l)" >> "$SUMMARY_FILE"
echo "Failed batches: $(grep -l "FAILED:" ${LOG_DIR}/batch_*.log 2>/dev/null | wc -l)" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Failed batches:" >> "$SUMMARY_FILE"
grep -l "FAILED:" ${LOG_DIR}/batch_*.log 2>/dev/null | xargs -I {} basename {} .log >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Log directory: $LOG_DIR" >> "$SUMMARY_FILE"

cat "$SUMMARY_FILE"

# Verify downloaded files and update metadata flags
if [ "$UPDATE_FLAGS" = true ]; then
    echo ""
    echo "============================================"
    echo "Verifying downloads and updating metadata flags..."
    echo "============================================"
    
    # Count actual .bakta.gbff.gz files in output directory
    echo "Scanning $OUTPUT_DIR for .bakta.gbff.gz files..."
    ACTUAL_FILES=$(find "$OUTPUT_DIR" -name "*.bakta.gbff.gz" -type f | wc -l)
    echo "Found $ACTUAL_FILES .bakta.gbff.gz files in output directory"
    
    # Export variables for Python script
    export TSV_FILE
    export OUTPUT_DIR
    
    # Run Python script to update flags
    python3 <<'EOFFLAGS'
import pandas as pd
import sys
import os
from pathlib import Path

tsv_file = Path(os.environ['TSV_FILE'])
output_dir = Path(os.environ['OUTPUT_DIR'])

print(f"\nScanning {output_dir} for *.bakta.gbff.gz files...")
gbff_files = list(output_dir.rglob("*.bakta.gbff.gz"))
print(f"Found {len(gbff_files)} .bakta.gbff.gz files")

# Extract sample accessions from directory structure
# Assuming structure: OUTPUT_DIR/XXXX/SAMXXXXXXX/SAMXXXXXXX.bakta.gbff.gz
sample_accessions = set()
for filepath in gbff_files:
    # The parent directory name should be the sample_accession
    sample_acc = filepath.parent.name
    sample_accessions.add(sample_acc)

print(f"Extracted {len(sample_accessions)} unique sample accessions from file paths")

# Load metadata
print(f"\nLoading metadata from {tsv_file}...")
df = pd.read_csv(tsv_file, sep='\t', low_memory=False)
print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

# Check which column to use
if 'sample_accession' not in df.columns:
    print("ERROR: 'sample_accession' column not found in metadata", file=sys.stderr)
    sys.exit(1)

# Update bakta_gbff_downloaded flag based on sample_accession
df['bakta_gbff_downloaded'] = df['sample_accession'].isin(sample_accessions)

# Print statistics
num_downloaded = df['bakta_gbff_downloaded'].sum()
num_total = len(df)
print(f"\nResults:")
print(f"  Samples with bakta.gbff.gz: {num_downloaded:,} / {num_total:,} ({num_downloaded/num_total*100:.1f}%)")
print(f"  Samples without file: {num_total - num_downloaded:,}")

# Save updated metadata
print(f"\nUpdating metadata file: {tsv_file}")
df.to_csv(tsv_file, sep='\t', index=False)
print("✓ Metadata flags updated successfully!")
EOFFLAGS

    if [ $? -eq 0 ]; then
        echo "✓ Metadata flags updated successfully"
    else
        echo "✗ Failed to update metadata flags"
    fi
fi

# Cleanup
rm -f "$TEMP_SAMPLE_FILE"
rm -rf "$BATCH_DIR"

# Final summary
echo ""
echo "============================================"
echo "Download job completed at $(date)"
echo "Exit code: $EXIT_CODE"
echo ""
echo "Total samples: $TOTAL"
echo "Batches processed: $NUM_BATCHES"
echo ""
echo "Summary report: $SUMMARY_FILE"
echo "Batch logs: $LOG_DIR/"
echo "============================================"

exit $EXIT_CODE