#!/bin/bash
#SBATCH --job-name=fastq_dl
#SBATCH --time=4:00:00
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --account=FLOTO-SL2-CPU

eval "$(micromamba shell hook --shell bash)"
micromamba activate fastq-dl

cd /home/dca36/workspace/Bacotype
python src/bacotype/pp/run_fastq_download.py "$@"
