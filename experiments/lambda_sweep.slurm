#!/bin/bash
#SBATCH --job-name=lambda_sweep
#SBATCH --output=lambda_sweep_%A_%a.out
#SBATCH --error=lambda_sweep_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --partition=tier3
#SBATCH --array=0-5         # 6 lambda values, one job per value

# Lambda values to sweep — index maps to array task ID
LAMBDAS=(0.5 1.0 1.5 2.0 3.0 5.0)
LAMBDA=${LAMBDAS[$SLURM_ARRAY_TASK_ID]}

echo "========================================"
echo "Lambda sweep  |  task=$SLURM_ARRAY_TASK_ID  lambda=$LAMBDA"
echo "========================================"

# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate codegen

cd ~/testing/secure-codegen/experiments

OUTPUT_DIR="./eval_results_lambda_${LAMBDA}"

python3 evaluate_on_dataset.py \
    --dataset_path  ~/testing/secure-codegen/data/combined_dataset \
    --scorer_path   ~/testing/secure-codegen/data/models/security_scorer/final \
    --n_samples     200 \
    --per_cwe       30 \
    --max_new_tokens 128 \
    --gamma         5 \
    --security_lambda "$LAMBDA" \
    --threshold     0.5 \
    --temperature   0.7 \
    --output_dir    "$OUTPUT_DIR"

echo "Done  lambda=$LAMBDA  results -> $OUTPUT_DIR"
