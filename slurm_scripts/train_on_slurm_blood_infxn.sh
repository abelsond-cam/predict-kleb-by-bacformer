#!/bin/bash
#SBATCH --job-name=blood_infxn
#SBATCH --output=blood_infxn_%A_%a.out
#SBATCH --error=blood_infxn_%A_%a.err
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

species=klebsiella_pneumoniae
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

echo "Training blood_infxn model (blood vs stool) from pytorch (.pt) files"
echo "species: $species"
echo "Training set 70%, validation set 10%, test set 20%"
echo "eval_steps: $eval_steps"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Learning rate: $lr"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME, GPU: $CUDA_VISIBLE_DEVICES"

echo "Finetuned model for blood_infxn prediction (Bacformer finetuning, linear head)"
uv run python src/predict_kleb_by_bacformer/tl/train_blood_infx.py  \
--train-data-dir /home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/blood_infx_training/train \
--val-data-dir /home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/blood_infx_training/validate \
--sheet-path /home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/binary_blood_infxn_with_split.csv \
--lr $lr \
--model-name-or-path $model_name_or_path \
--warmup-proportion $warmup_proportion \
--num-workers 15 \
--grad-accumulation-steps 8 \
--batch-size 1 \
--eval-steps $eval_steps \
--max-steps 100000 \
--early-stopping-patience 30 \
--output-dir /home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/blood_infx_training/blood_infxn_lr_${lr}_finetuned

echo "End of script... check the .out and .err logs for any errors and for training progress"
