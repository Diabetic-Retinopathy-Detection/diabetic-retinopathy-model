# HPC Deployment

Train on HPC clusters using Singularity containers with GPU acceleration.

## Prerequisites

- A **Linux amd64 machine** with Docker (for building the image)
- **Singularity/Apptainer** on the HPC cluster
- Data uploaded to cluster scratch storage

## 1. Build the Docker image

Build on a **Linux amd64** machine (not macOS — `docker buildx` cross-compilation for Singularity is unreliable):

```bash
git clone https://github.com/Diabetic-Retinopathy-Detection/diabetic-retinopathy-model.git
cd diabetic-retinopathy-model
docker build -f Dockerfile.training -t dr-train:local .
```

The image includes CUDA 12.2, Python 3.11, and all dependencies pre-installed.

## 2. Transfer to cluster

```bash
# Save Docker image as tar
docker save dr-train:local -o /tmp/dr-train.tar

# Transfer to cluster (resumable with rsync)
rsync -avP --progress /tmp/dr-train.tar hpc:/scratch/users/$USER/

# On cluster — convert tar to Singularity SIF
ssh hpc
singularity build /scratch/users/$USER/dr-train.sif \
    docker-archive:///scratch/users/$USER/dr-train.tar

# Clean up tar (SIF is self-contained)
rm /scratch/users/$USER/dr-train.tar
```

## 3. Prepare writable directories on cluster

Singularity mounts the container as read-only. All writable data (logs, checkpoints, MLflow) must live on scratch:

```bash
mkdir -p /scratch/users/$USER/data
mkdir -p /scratch/users/$USER/checkpoints
mkdir -p /scratch/users/$USER/tb-logs
mkdir -p /scratch/users/$USER/mlflow
```

## 4. Transfer data to cluster

```bash
# From your local machine — rsync cropped images, saliency maps, and pickle index
rsync -avP --progress /path/to/cropped/  hpc:/scratch/users/$USER/data/cropped/
rsync -avP --progress /path/to/saliency/ hpc:/scratch/users/$USER/data/saliency/
scp /path/to/dataset.pkl hpc:/scratch/users/$USER/data/
```

## 5. Run training

### Interactive

```bash
singularity exec --nv --writable-tmpfs \
    --home /tmp \
    --pwd /app \
    --bind /scratch/users/$USER/data:/app/data \
    --bind /scratch/users/$USER/checkpoints:/app/checkpoints \
    --bind /scratch/users/$USER/tb-logs:/app/logs \
    --bind /scratch/users/$USER/mlflow:/scratch/mlflow \
    --env MLFLOW_TRACKING_URI="sqlite:////scratch/mlflow/mlflow.db" \
    /scratch/users/$USER/dr-train.sif \
    uv run python -m dr_model.training.cli \
        --phase pretrain --device cuda \
        --config configs/pretrain_default.yaml \
        --data-index-path /app/data/dataset.pkl \
        --data-dir /app/data \
        --seed 42
```

### SLURM batch job (single or multi-GPU)

A ready-to-use SLURM script is included at `scripts/pretrain_hpc.sh`, with a
smoke variant at `scripts/pretrain_hpc_smoke.sh`. The recommended entry point
is `scripts/submit_train.sh`, which overrides the GPU count:

```bash
# 1x GPU (single-process, no DDP)
./scripts/submit_train.sh full 1

# 4x GPU — single-node DDP, one torchrun process per GPU
./scripts/submit_train.sh full 4

# 2x GPU smoke test — proves gradient sync before a long run
./scripts/submit_train.sh smoke 2
```

The command-line `--gres=gpu:N` overrides the `#SBATCH --gres=gpu:1` default in
the script, and `N_GPUS` is exported into the job. Inside the job, Slurm sets
`CUDA_VISIBLE_DEVICES` to the allocated GPUs and the script launches:

```bash
uv run torchrun --nnodes=1 --nproc-per-node="${N_GPUS}" -m dr_model.training.cli \
    --phase pretrain --device cuda \
    --config configs/pretrain_default.yaml \
    --data-index-path /app/data/dataset.pkl \
    --data-dir /app/data \
    --seed 42
```

**Note the `-m` form**: `torchrun ... -m dr_model.training.cli` — torchrun takes
the module via its own `-m` flag (do **not** write `torchrun ... python -m ...`).
Each rank detects the torchrun environment and pins itself to `cuda:{local_rank}`;
the `--device cuda` argument is ignored in favour of the per-rank device.

Monitor with:

```bash
squeue -u $USER
tail -f /scratch/users/$USER/tb-logs/<jobid>.out
```

### SLURM batch script example

```bash
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
```

Submit with: `./scripts/submit_train.sh full 4`

## Single-GPU Linux machine (local development)

A Linux box with one GPU (no Slurm) is ideal for testing CUDA-only paths —
AMP, real GPU memory behaviour, and the Linux/amd64 Docker build that
Singularity requires. Clone the repo and run directly:

```bash
git clone https://github.com/Diabetic-Retinopathy-Detection/diabetic-retinopathy-model.git
cd diabetic-retinopathy-model
uv sync          # installs deps + project into .venv

# Smoke test on the GPU (validates CUDA, AMP off, determinism off)
uv run python -m dr_model.training.cli \
    --phase pretrain --device cuda \
    --config configs/pretrain_smoke.yaml

# Multi-GPU check on a 1-GPU box: run the DDP launcher with 1 process —
# exercises the torchrun path (rank 0) that the cluster will use with N GPUs
uv run torchrun --nnodes=1 --nproc-per-node=1 -m dr_model.training.cli \
    --phase pretrain --device cuda \
    --config configs/pretrain_smoke.yaml
```

`--device cuda` falls back to `cuda:0`; with `--device auto` the first visible
GPU is used. For real data, add `--config configs/pretrain_default.yaml
--data-index-path <path>/dataset.pkl --data-dir <root>`.

This is also the right machine to build the amd64 image for the cluster:
`make docker-build-train` (see [Section 1](#1-build-the-docker-image)).

## Singularity flags explained

| Flag | Why |
|---|---|
| `--nv` | Enable NVIDIA GPU passthrough |
| `--writable-tmpfs` | Writable overlay (tiny — only for uv cache, not data) |
| `--home /tmp` | Prevents uv from discovering Python from user's home directory |
| `--pwd /app` | Set working directory to the container's project |
| `--bind ...:/app/data` | Mount training data |
| `--bind ...:/app/checkpoints` | Mount checkpoint output directory |
| `--bind ...:/app/logs` | Mount TensorBoard log directory |
| `--bind ...:/scratch/mlflow` | Mount MLflow database directory |
| `--env MLFLOW_TRACKING_URI=...` | Point MLflow to SQLite on scratch (not file store) |

**Note:** We use `uv run python -m dr_model.training.cli` and `uv run dr-train` interchangeably — both work, since `uv run` puts the project's `dr-train` entry point on `PATH`. The `python -m` form is used with `torchrun`, which has its own `-m` flag.

## MLflow on cluster

The container's resolved MLflow version determines which tracking backends are
supported. The commands above explicitly configure SQLite on scratch:

```
MLFLOW_TRACKING_URI=sqlite:////scratch/mlflow/mlflow.db
```

This creates `mlflow.db` on scratch. View results locally:

```bash
scp hpc:/scratch/users/$USER/mlflow/mlflow.db ./
mlflow ui
# → http://localhost:5000
```

Or log to a remote MLflow server instead:

```bash
--env MLFLOW_TRACKING_URI=http://mlflow-server:5000
```

## Troubleshooting

| Issue | Cause | Fix |
|---|---|---|
| `image (linux/arm64) does not satisfy required platform (linux/amd64)` | Docker image built on macOS arm64 | Build on a Linux amd64 machine |
| `database or disk is full` | MLflow writing to tiny tmpfs overlay | Bind-mount mlflow dir + set `MLFLOW_TRACKING_URI` to SQLite on scratch |
| `No space left on device` | TensorBoard/checkpoints writing to tmpfs | Bind-mount `logs/` and `checkpoints/` to scratch |
| `unexpected pos 64 vs 0` | `torch.save` failing on tmpfs | Bind-mount `checkpoints/` to scratch |
| `No module named dr_model` | `python -m` invoked with the system Python instead of the venv | Use `uv run python -m dr_model.training.cli` |
| `No such file or directory: dr-train` | Entry point called without `uv run` | Use `uv run dr-train` |
| MLflow `filestore in maintenance mode` | The installed MLflow version requires a database backend | Use the explicitly configured `sqlite:///` URI |
| `Failed to initialize cache at .cache/uv` | Home directory is read-only | Use `--writable-tmpfs` and `--home /tmp` |
| `CUDA out of memory` | Batch too large for GPU VRAM | Reduce `batch_size` in config |
| `NVIDIA driver not detected` | Missing `--nv` flag | Add `--nv` to `singularity exec` |
| `can't open file '.../python'` | `torchrun ... python -m ...` | Use torchrun's own `-m`: `torchrun ... -m dr_model.training.cli` |
| `Unable to resolve hostname` / NCCL timeout on multi-GPU | Wrong network interface chosen | Set `NCCL_SOCKET_IFNAME` (e.g. `ib0` or `eth0`) in the Slurm script or `--env` |
| Only rank 0 logs metrics | Rank gating is intentional | Checkpoints, TensorBoard, and MLflow are written by rank 0 only |

## GPU partitions

| Partition | GPU Type | VRAM | Time Limit | Preemptible | Notes |
|---|---|---|---|---|---|
| `gpu` | H200 | 141 GB | 2 days | No | Recommended for training |
| `gpu` | A100-40GB | 40 GB | 2 days | No | Also available on this partition |
| `interruptible_gpu` | A100 (MIG) | ~5 GB | Infinite | Yes | **Avoid** — MIG slices too small |
| `tier1_gpu` | H200 | 141 GB | Infinite | No | High priority, often fully allocated |

**Important**: Some nodes on `interruptible_gpu` have MIG (Multi-Instance GPU) enabled, splitting each A100 into ~5 GB slices. Always use `--partition=gpu` to get full GPUs.
