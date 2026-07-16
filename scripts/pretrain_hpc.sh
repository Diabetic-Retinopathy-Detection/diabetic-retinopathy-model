#!/bin/bash
#SBATCH --job-name=dr-pretrain
#SBATCH --partition=gpu
#SBATCH --qos=public_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --output=/scratch/users/%u/tb-logs/%j.out
#SBATCH --error=/scratch/users/%u/tb-logs/%j.err

# ─── Full ViT-B/16 pretraining on HPC ─────────────────────────────
#
# Usage:
#   sbatch scripts/pretrain_hpc.sh
#
# Monitor:
#   squeue -u $USER
#   tail -f /scratch/users/$USER/tb-logs/<jobid>.out
# ────────────────────────────────────────────────────────────────────

SCRATCH="/scratch/users/${USER}"
SIF="${SCRATCH}/dr-train.sif"

mkdir -p "${SCRATCH}/checkpoints" "${SCRATCH}/tb-logs" "${SCRATCH}/mlflow"

singularity exec --nv --writable-tmpfs \
    --home /tmp \
    --pwd /app \
    --bind "${SCRATCH}/data:/app/data" \
    --bind "${SCRATCH}/checkpoints:/app/checkpoints" \
    --bind "${SCRATCH}/tb-logs:/app/logs" \
    --bind "${SCRATCH}/mlflow:/scratch/mlflow" \
    --env MLFLOW_TRACKING_URI="sqlite:////scratch/mlflow/mlflow.db" \
    "${SIF}" \
    uv run python -m dr_model.training.cli \
        --phase pretrain --device cuda \
        --config configs/pretrain_default.yaml \
        --data-index-path /app/data/dataset.pkl \
        --data-dir /app/data \
        --seed 42
