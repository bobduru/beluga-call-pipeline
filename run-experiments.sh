#!/usr/bin/env bash

# ===== Slurm options =====
#SBATCH --partition=a100-40
#SBATCH --gpus=1
#SBATCH --job-name=s_opt3
#SBATCH --output="/home/%u/voxaboxen/slurm-logs/%A.log"

# ensure environment is created in scratch and synced
uv sync

# run training using the allocated GPU
srun uv run training/run_optimization_3sec.py