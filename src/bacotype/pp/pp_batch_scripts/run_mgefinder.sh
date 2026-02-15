#!/bin/bash
#SBATCH --job-name=mgefinder
#SBATCH --time=4:00:00
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --account=FLOTO-SL2-CPU

eval "$(micromamba shell hook --shell bash)"
micromamba activate mgefinder

cd /home/dca36/workspace/Bacotype
python src/bacotype/pp/run_mgefinder.py "$@"
