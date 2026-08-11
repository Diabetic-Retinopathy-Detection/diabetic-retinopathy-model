#!/bin/bash -l
#SBATCH --job-name=dr-finetune-smoke
#SBATCH --partition=gpu
#SBATCH --qos=public_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/users/%u/tb-logs/%j.out
#SBATCH --error=/scratch/users/%u/tb-logs/%j.err

# ─── Smoke test: verify fine-tuning pipeline on HPC (multi-GPU capable) ───
#
# Runs one epoch through the DDP path with the real DDR data and the real
# seed checkpoint (proves checkpoint load + gradient sync + logging).
# Use 2 GPUs to prove gradient sync before a full run:
#   ./scripts/submit_train.sh finetune-smoke 2
#
# Monitor:
#   squeue -u $USER
#   tail -f /scratch/users/$USER/tb-logs/<jobid>.out
# ────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRATCH="/scratch/users/${USER}"
SIF="${SCRATCH}/dr-train.sif"
N_GPUS="${N_GPUS:-${SLURM_GPUS_ON_NODE:-1}}"
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
        --config configs/finetune_ddr_smoke.yaml \
        --finetune-checkpoint "${CHECKPOINT}" \
        --seed 42
