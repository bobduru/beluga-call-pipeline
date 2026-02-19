#!/usr/bin/env bash

# ===== Slurm options =====
#SBATCH --partition=a100-40
#SBATCH --gpus=1
#SBATCH --job-name=spects_embeddings
#SBATCH --output="/home/%u/voxaboxen/slurm-logs/%A.log"

# ensure environment is created in scratch and synced
uv sync

uv run python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device Count: {torch.cuda.device_count()}') if torch.cuda.is_available() else None"
