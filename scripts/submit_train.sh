#!/bin/bash
# Submit a pretraining Slurm job, optionally overriding the GPU count.
#
# Usage:
#   ./scripts/submit_train.sh [full|smoke] [N_GPUS]
#
# Examples:
#   ./scripts/submit_train.sh full 4    # 4x GPU full pretraining (gpu partition)
#   ./scripts/submit_train.sh full      # 1x GPU full pretraining
#   ./scripts/submit_train.sh smoke 2   # 2x GPU smoke test (proves DDP sync)
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
    *)
        echo "Unknown mode '${MODE}' (expected 'full' or 'smoke')" >&2
        exit 1
        ;;
esac

if [[ -n "${N_GPUS}" ]]; then
    exec sbatch --gres="gpu:${N_GPUS}" --export="ALL,N_GPUS=${N_GPUS}" "${SCRIPT}"
else
    exec sbatch --export=ALL "${SCRIPT}"
fi
