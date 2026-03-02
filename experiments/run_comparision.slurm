#!/bin/bash
#SBATCH --job-name=spec_comparison
#SBATCH --output=comparison_%j.out
#SBATCH --error=comparison_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --partition=tier3

# Load modules (if needed)
# module load cuda/12.4

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate codegen  # Replace with your actual env name

# Go to project directory
cd testing/secure-codegen/experiments

# Run test
python3 test_comparison.py

echo "Comparison complete!"