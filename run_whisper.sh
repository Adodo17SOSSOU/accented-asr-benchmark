#!/bin/bash
#SBATCH --job-name=run_whisper
#SBATCH --output=logs/whisper.out
#SBATCH --error=logs/whisper.err
#SBATCH --partition=L40S
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

echo "Starting at $(date) on $(hostname)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate asr_env
export HF_HUB_READ_TIMEOUT=60

python run_whisper.py

echo "Finished at $(date)"

