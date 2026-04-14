#!/bin/bash
#SBATCH --job-name=isol_src_train
#SBATCH --output=isol_src_train_%j.out
#SBATCH --error=isol_src_train_%j.err
#SBATCH --time=36:00:00
#SBATCH --partition=ampere
#SBATCH --account=FLOTO-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=250G
#SBATCH --open-mode=append

cd /home/dca36/workspace/predict_kleb_by_bacformer

python_script="src/predict_kleb_by_bacformer/tl/train_isolation_source.py"
isolation_sources="blood respiratory"

warmup_proportion=0.1
lr=0.00015
eval_steps=850  # There are ~ 7,000 in training set, so this is one epoch, with step size of 8
model_name_or_path="macwiatrak/bacformer-large-masked-MAG"

# Load any necessary modules
module purge
module load cuda/12.4
module load cudnn/8.9_cuda-12.4

# Force Python unbuffered output for real-time logging
export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=info

echo "========================================================================"
echo "Fine-tuning Bacformer for isolation-source pair prediction (.pt files)"
echo "Isolation sources: $isolation_sources"
echo "Python script: $python_script"
echo "eval_steps: $eval_steps"
echo "Learning rate: $lr"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME, GPU: $CUDA_VISIBLE_DEVICES"
echo "========================================================================"
echo ""

uv run python "$python_script" \
  --isolation-sources $isolation_sources \
  --lr "$lr" \
  --model-name-or-path "$model_name_or_path" \
  --warmup-proportion "$warmup_proportion" \
  --num-workers 15 \
  --grad-accumulation-steps 8 \
  --batch-size 1 \
  --eval-steps "$eval_steps" \
  --max-steps 100000 \
  --early-stopping-patience 30

echo ""
echo "End of script... check the .out and .err logs for any errors and for training progress"
