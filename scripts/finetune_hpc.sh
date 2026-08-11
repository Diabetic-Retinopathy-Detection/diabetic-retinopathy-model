#!/bin/bash -l
#SBATCH --job-name=dr-finetune
#SBATCH --partition=gpu
#SBATCH --qos=public_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=47:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --output=/scratch/users/%u/tb-logs/%j.out
#SBATCH --error=/scratch/users/%u/tb-logs/%j.err

# ─── Full ViT-B/16 fine-tuning on HPC (multi-GPU DDP) ─────────────
#
# Allocates $N_GPUS GPUs (default: 1) and launches one torchrun
# process per GPU. Slurm sets CUDA_VISIBLE_DEVICES to the allocated
# GPUs, which torchrun's LOCAL_RANK indexes via cuda:{local_rank}.
#
# Recommended entry point — overrides the allocation:
#   ./scripts/submit_train.sh finetune 4
#
# Override the config or seed checkpoint via env:
#   FINETUNE_CONFIG=configs/finetune_aptos.yaml \
#   FINETUNE_CHECKPOINT=checkpoints/foo/checkpoint.pt \
#   ./scripts/submit_train.sh finetune 4
#
# Or submit directly:
#   N_GPUS=4 sbatch --gres=gpu:4 scripts/finetune_hpc.sh
#
# Monitor:
#   squeue -u $USER
#   tail -f /scratch/users/$USER/tb-logs/<jobid>.out
# ────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRATCH="/scratch/users/${USER}"
SIF="${SCRATCH}/dr-train.sif"
N_GPUS="${N_GPUS:-${SLURM_GPUS_ON_NODE:-1}}"
CONFIG="${FINETUNE_CONFIG:-configs/finetune_ddr.yaml}"
CHECKPOINT="${FINETUNE_CHECKPOINT:-checkpoints/full-pretraining/checkpoint.pt}"

mkdir -p "${SCRATCH}/checkpoints" "${SCRATCH}/tb-logs" "${SCRATCH}/mlflow"

singularity exec --nv --writable-tmpfs \
    --home /tmp \
    --pwd /app \
    --bind "${SCRATCH}/data:/app/data" \
    --bind "${SCRATCH}/checkpoints:/app/checkpoints" \
    --bind "${SCRATCH}/tb-logs:/app/logs" \
    --bind "${SCRATCH}/mlflow:/scratch/mlflow" \
    --env MLFLOW_TRACKING_URI="sqlite:////scratch/mlflow/mlflow.db" \
    --env PYTHONDONTWRITEBYTECODE=1 \
    "${SIF}" \
    uv run torchrun --nnodes=1 --nproc-per-node="${N_GPUS}" -m dr_model.training.cli \
        --phase finetune --device cuda \
        --config "${CONFIG}" \
        --finetune-checkpoint "${CHECKPOINT}" \
        --seed 42
