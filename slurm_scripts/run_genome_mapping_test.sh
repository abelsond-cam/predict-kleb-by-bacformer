#!/bin/bash

#SBATCH --job-name=genome_mapping_test
#SBATCH --output=genome_mapping_test_%j.out
#SBATCH --error=genome_mapping_test_%j.err
#SBATCH --time=00:05:00         # 5 minutes should be plenty for 1000 files
#SBATCH --nodes=1
#SBATCH --ntasks=100            # Use 100 cores for testing

# Change to the workspace directory
cd /home/u5ah/dca36.u5ah/Workspace/Bacotype

# Run the script with micromamba - test with 1000 files
micromamba run -n pytorch_env python src/bacotype/pp/create_genome_to_parquet_file_mapping.py \
    --processes 100 \
    --max-files 1000

echo "Job completed at $(date)"
