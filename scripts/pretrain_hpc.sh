#!/bin/bash -l
#SBATCH --job-name=dr-pretrain
#SBATCH --partition=gpu
#SBATCH --qos=public_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --output=/scratch/users/%u/tb-logs/%j.out
#SBATCH --error=/scratch/users/%u/tb-logs/%j.err

# ─── Full ViT-B/16 pretraining on HPC (multi-GPU DDP) ─────────────
#
# Allocates $N_GPUS GPUs (default: 1) and launches one torchrun
# process per GPU. Slurm sets CUDA_VISIBLE_DEVICES to the allocated
# GPUs, which torchrun's LOCAL_RANK indexes via cuda:{local_rank}.
#
# Recommended entry point — overrides the allocation:
#   ./scripts/submit_train.sh full 4
#
# Or submit directly:
#   N_GPUS=4 sbatch --gres=gpu:4 scripts/pretrain_hpc.sh
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
        --config configs/pretrain_default.yaml \
        --data-index-path /app/data/dataset.pkl \
        --data-dir /app/data \
        --seed 42
