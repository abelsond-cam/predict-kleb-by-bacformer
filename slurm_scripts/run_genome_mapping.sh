#!/bin/bash

#SBATCH --job-name=genome_mapping
#SBATCH --output=genome_mapping_%j.out
#SBATCH --error=genome_mapping_%j.err
#SBATCH --time=02:00:00         # 2 hours should be plenty with 120 processes
#SBATCH --nodes=1
#SBATCH --ntasks=120            # Use 120 of 144 cores (leave some for overhead)

# Change to the workspace directory
cd /home/u5ah/dca36.u5ah/Workspace/Bacotype

# Run the script with micromamba
# Using 120 processes to fully utilize the Grace CPU Superchip (144 cores)
micromamba run -n pytorch_env python src/bacotype/pp/create_genome_to_parquet_file_mapping.py \
    --processes 120 \
    --max-files -1

echo "Job completed at $(date)"
