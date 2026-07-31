# Distributed Training (DDP)

Single-node, multi-GPU pretraining via `torch.distributed` and `torchrun`.
The design keeps **all distributed runtime concerns in one module**
(`dr_model.training.distributed`) — training logic never inspects environment
variables or calls `torch.distributed` directly.

## Launch model

`torchrun` starts one process per GPU and sets four environment variables:

| Variable | Meaning |
|---|---|
| `RANK` | Global rank across all processes |
| `LOCAL_RANK` | Rank within the node — equals the GPU index |
| `WORLD_SIZE` | Total number of processes |
| `MASTER_ADDR` / `MASTER_PORT` | Rendezvous address for the process group |

```bash
uv run torchrun --nnodes=1 --nproc-per-node=4 -m dr_model.training.cli \
    --phase pretrain --device cuda
```

`--device` is **ignored** under CUDA torchrun — each rank must pin itself to
its own GPU, so the CLI always uses `cuda:{local_rank}`. Explicit CPU runs
(`--device cpu`) keep the requested device and use the `gloo` backend, which is
how the multi-process unit tests exercise real DDP without a GPU.

## Bootstrap flow

1. **Detect** — `detect_distributed_context(cli_device)` reads `RANK`/
   `LOCAL_RANK`/`WORLD_SIZE` and builds a `DistributedContext`. Without a
   `torchrun` environment, the context is a plain single-process run
   (`enabled=False`).
2. **Init** — `init_distributed(ctx)` creates the process group: NCCL on CUDA,
   gloo otherwise.
3. **Wrap** — `wrap_model(model, ctx)` converts BatchNorm layers to
   `SyncBatchNorm` on CUDA (the pretrainer's projection heads use `BatchNorm1d`,
   whose statistics get noisy at small per-GPU batch sizes), then wraps the
   model in `DistributedDataParallel` pinned to the local GPU.
4. **Teardown** — `cleanup_distributed()` destroys the process group. It is
   idempotent and runs unconditionally in the CLI's `finally` block.

## Rank-0 gating

All side effects that must happen exactly once run on rank 0 only:

- MLflow run initialisation and metric logging
- TensorBoard writer creation and scalar writes
- Checkpoint and encoder saves
- Progress prints and finalisation

Other ranks only train and participate in gradient sync. Gating is applied via
`is_rank_zero(ctx)` in the CLI and the `@rank_zero_only` decorator on logging
helpers. Collectives are **never** gated — all ranks must reach them.

## Data sharding

`PretrainDataModule` handles two things for distributed runs:

- **Deterministic subsampling** — the optional `dataset_ratio < 1.0` subset is
  selected with a *locally seeded* `random.Random(config.seed)`, so every rank
  picks the identical subset regardless of launch order.
- **Sharding** — when `torch.distributed.is_initialized()`, the dataloader uses
  a `DistributedSampler` (stored as `self.sampler`). The training loop calls
  `sampler.set_epoch(epoch)` each epoch so shards rotate between epochs.

## Global metric reduction

Per-rank epoch losses are combined with an `all_reduce` of **sums and counts**,
not by averaging per-rank means:

```python
def _reduce_epoch_metrics(cl_sum, ss_sum, steps, world_size, device):
    # all_reduce SUM on cl_sum, ss_sum, steps → global_mean = sum / count
```

Averaging per-rank means is only correct when every rank processes the same
number of batches (e.g. `drop_last=True` with equal shards); sums + counts stay
correct even when shards are uneven.

## Checkpoint integrity

`unwrap_model()` is the single source of truth on every persistence path, so
checkpoints and encoder dumps never contain `module.`-prefixed keys (which
would otherwise appear from a DDP wrapper's `module` attribute). This keeps
checkpoints loadable by the existing resume path unchanged.

The TensorBoard `SummaryWriter` is owned by the CLI — it is created and closed
there; the training loop never closes it. This makes ownership explicit and
avoids double-close between ranks.

## Verification

The distributed path is tested without a GPU:

- `tests/pretrain_loop/test_distributed.py` runs a **real 2-process DDP
  training** on the `gloo` backend (`torch.multiprocessing.start_processes`,
  spawn) — forward, backward, gradient sync, and rank-0 checkpoint production
  all execute.
- It also covers torchrun env detection, the `DistributedSampler` boundary,
  checkpoint key format, and rank-0 gating.

And end-to-end through the CLI:

```bash
uv run torchrun --nnodes=1 --nproc-per-node=2 -m dr_model.training.cli \
    --phase pretrain --device cpu --config configs/pretrain_smoke.yaml
```

For the HPC cluster (Slurm + Singularity), see [HPC Deployment](hpc.md).
