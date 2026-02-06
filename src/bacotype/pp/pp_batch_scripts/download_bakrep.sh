#!/bin/bash
#SBATCH --job-name=bakrep_download
#SBATCH --output=bakrep_download_%j.out
#SBATCH --error=bakrep_download_%j.err
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=76
#SBATCH --time=02:00:00
#SBATCH --account=FLOTO-SL2-CPU

# Default values
N=10
TSV_FILE="/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/final/metadata_final_curated_slimmed.tsv"
OUTPUT_DIR="/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/raw/klebsiella_gff"
NCORES=76
BATCH_SIZE=100  # Number of samples per batch download

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --n)
            N="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --n <number>"
            echo "  --n 10    : Download first 10 samples (test run)"
            echo "  --n -1    : Download all samples"
            exit 1
            ;;
    esac
done

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate my-bio

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Extract sample IDs to a temporary file
TEMP_SAMPLE_FILE=$(mktemp)
if [ "$N" -eq -1 ]; then
    echo "Extracting ALL sample IDs from TSV..."
    tail -n +2 "$TSV_FILE" | cut -f1 > "$TEMP_SAMPLE_FILE"
    TOTAL=$(wc -l < "$TEMP_SAMPLE_FILE")
    echo "Total samples: $TOTAL"
else
    echo "Extracting first $N sample IDs from TSV..."
    tail -n +2 "$TSV_FILE" | cut -f1 | head -n "$N" > "$TEMP_SAMPLE_FILE"
    TOTAL=$N
    echo "Sample count: $TOTAL"
fi

echo "Sample IDs saved to: $TEMP_SAMPLE_FILE"
echo "First few sample IDs:"
head -n 3 "$TEMP_SAMPLE_FILE"

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
        -m tool:bakta,filetype:gff3 \  #-m tool:bakta,filetype:gbff3
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