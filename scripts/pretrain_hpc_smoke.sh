#!/bin/bash -l
#SBATCH --job-name=dr-smoke
#SBATCH --partition=gpu
#SBATCH --qos=public_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/users/%u/tb-logs/%j.out
#SBATCH --error=/scratch/users/%u/tb-logs/%j.err

# ─── Smoke test: verify full pipeline on HPC (multi-GPU capable) ───
#
# Runs the tiny smoke config through the DDP path. Use 2 GPUs to prove
# gradient sync works before launching a full run:
#   ./scripts/submit_train.sh smoke 2
#
# Monitor:
#   squeue -u $USER
#   tail -f /scratch/users/$USER/tb-logs/<jobid>.out
# ────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRATCH="/scratch/users/${USER}"
SIF="${SCRATCH}/dr-train.sif"
N_GPUS="${N_GPUS:-${SLURM_GPUS_ON_NODE:-1}}"

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
    uv run torchrun --nnodes=1 --nproc-per-node="${N_GPUS}" -m dr_model.training.cli \
        --phase pretrain --device cuda \
        --config configs/pretrain_smoke.yaml \
        --data-index-path /app/data/smoke_dataset.pkl \
        --data-dir /app/data \
        --seed 42
