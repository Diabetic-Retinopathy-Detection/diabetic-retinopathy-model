#!/bin/bash
# Submit a pretraining or fine-tuning Slurm job, optionally overriding the
# GPU count.
#
# Usage:
#   ./scripts/submit_train.sh [full|smoke|finetune|finetune-smoke] [N_GPUS]
#
# Examples:
#   ./scripts/submit_train.sh full 4            # 4x GPU full pretraining
#   ./scripts/submit_train.sh full              # 1x GPU full pretraining
#   ./scripts/submit_train.sh smoke 2           # 2x GPU pretrain smoke (DDP sync)
#   ./scripts/submit_train.sh finetune 4        # 4x GPU DDR fine-tuning
#   ./scripts/submit_train.sh finetune-smoke 2  # 2x GPU finetune smoke
#
# Fine-tuning overrides (propagated to the job via --export=ALL):
#   FINETUNE_CONFIG=configs/finetune_aptos.yaml ./scripts/submit_train.sh finetune 4
#   FINETUNE_CHECKPOINT=checkpoints/foo/checkpoint.pt ./scripts/submit_train.sh finetune 4
#
# The command-line --gres overrides the #SBATCH --gres default in the
# script, and N_GPUS is exported into the job so torchrun launches one
# process per allocated GPU.
set -euo pipefail

MODE="${1:-full}"
N_GPUS="${2:-}"

case "${MODE}" in
    full)
        SCRIPT="scripts/pretrain_hpc.sh"
        ;;
    smoke)
        SCRIPT="scripts/pretrain_hpc_smoke.sh"
        ;;
    finetune)
        SCRIPT="scripts/finetune_hpc.sh"
        ;;
    finetune-smoke)
        SCRIPT="scripts/finetune_hpc_smoke.sh"
        ;;
    *)
        echo "Unknown mode '${MODE}' (expected 'full', 'smoke', 'finetune', or 'finetune-smoke')" >&2
        exit 1
        ;;
esac

if [[ -n "${N_GPUS}" ]]; then
    exec sbatch --gres="gpu:${N_GPUS}" --export="ALL,N_GPUS=${N_GPUS}" "${SCRIPT}"
else
    exec sbatch --export=ALL "${SCRIPT}"
fi
