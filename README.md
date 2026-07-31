# Diabetic Retinopathy Model

ViT-based model for diabetic retinopathy classification — training, inference, and serving.

## Project structure

```
src/dr_model/
├── config.py                 # Pydantic Settings — all hyper-parameters in one place
├── model/
│   ├── backbone.py           # ViT encoder + classification head
│   └── pretrain.py           # Pretrainer — saliency-guided MoCo v3 (SSiT)
├── data/
│   ├── pair_dataset.py       # Paired image/saliency dataset + transforms
│   └── pretrain_datamodule.py# PretrainDataModule — pickle index + DistributedSampler
├── training/
│   ├── cli.py                # Training entry-point (dr-train)
│   ├── distributed.py        # DDP runtime: torchrun detection, rank gating, wrap/unwrap
│   ├── pretrain_loop.py      # Pretraining loop (schedules, checkpoints, metrics)
│   └── report.py             # TensorBoard → PDF training report
├── logging/                  # MLflow + TensorBoard loggers
├── inference/                # predict() — single-image inference pipeline
├── serve/                    # FastAPI serving endpoint
└── utils/                    # device resolution, determinism, timer
```

## ViTBackbone

The core model interface. Pure `nn.Module` — no training loop or serving logic.

### Quick start

```python
from dr_model.config import Settings
from dr_model.model import ViTBackbone

config = Settings()                    # loads from configs/pretrain_default.yaml
model = ViTBackbone(config)            # instantiates ViT-B/16 (patch=16, dim=768, depth=12)

import torch
x = torch.randn(1, 3, 224, 224)       # (batch, channels, height, width)
logits = model(x)                      # (1, 5) — 5 DR severity classes
```

### Architecture

All architecture parameters are driven by `Settings` — no magic constants in the class body.

| Parameter | Config field | Default | Description |
|---|---|---|---|
| `patch_size` | `Settings.patch_size` | 16 | Spatial size of each image patch |
| `embed_dim` | `Settings.embed_dim` | 768 | Transformer hidden dimension |
| `depth` | `Settings.depth` | 12 | Number of transformer blocks |
| `num_heads` | `Settings.num_heads` | 12 | Attention heads per block |
| `mlp_ratio` | `Settings.mlp_ratio` | 4.0 | MLP hidden dim / embed_dim |
| `num_classes` | `Settings.num_classes` | 5 | Output classes (DR severity grades) |
| `image_size` | `Settings.image_size` | (224, 224) | Input resolution (H, W) |

Standard ViT-B/16 configuration per Dosovitskiy et al. (2020).

### Forward pass

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """(B, 3, H, W) -> (B, num_classes) logits"""
```

1. Patch embedding via Conv2d: `(B, 3, H, W)` → `(B, N, E)` where `N = (H/P) * (W/P)`
2. Prepend learnable class token: `(B, N+1, E)`
3. Add positional embedding (class invariant: `pos_embed.shape[1] == N + 1`)
4. Transformer encoder blocks with pre-norm attention
5. Classification from class token only → `(B, num_classes)` logits

### Checkpoint loading

```python
model.load_checkpoint("checkpoints/vit_p16_e768_d12_h12_c5.pt")
```

Supports both bare `state_dict` files and VAE-style wrapper format `{"state_dict": ..., "model_config": ...}`.

Set `strict=False` when loading a pretrained encoder with a different `num_classes` (e.g. SSiT ImageNet-1k → DR 5-class):

```python
model.load_checkpoint("ssiT_pretrained.pt", strict=False)
```

Size-mismatched keys (typically the classification head) are silently skipped in non-strict mode.

## Configuration

`Settings` is a Pydantic `BaseSettings` class that loads from YAML, `.env` files, and environment variables.

### model_name (auto-generated)

`model_name` is a `@computed_field` — derived from architecture parameters, not manually set:

```python
config = Settings()
config.model_name  # "vit_p16_e768_d12_h12_c5"
```

Format: `vit_p{patch_size}_e{embed_dim}_d{depth}_h{num_heads}_c{num_classes}`

This name is used for checkpoint filenames, logging, and serving metadata. Non-encoded parameters (`mlp_ratio`, `drop_rate`, etc.) are stored in the checkpoint file itself.

### image_size normalisation

`image_size` accepts either a tuple or a scalar int. Scalars are normalised to a square:

```python
Settings(image_size=256).image_size   # (256, 256)
Settings(image_size=(224, 192)).image_size  # (224, 192)
```

### Config priority

1. Environment variables
2. `.env` file
3. YAML file (default: `configs/pretrain_default.yaml`)
4. Class defaults

Override the YAML path via the `DR_CONFIG_FILE` environment variable.

## Data

Preprocessed images live in a shared directory outside this repo:

```
DiabeticRetinopathy/
├── data/               # shared, gitignored
│   ├── cropped/        # JPEG fundus crops
│   ├── saliency/       # .npy saliency maps
│   └── dataset.pkl     # image-saliency pair index (relative paths)
├── preprocess-retina-datasets/
└── diabetic-retinopathy-model/   # this repo
```

Set `data_dir` in `configs/pretrain_default.yaml` or as an environment variable (`DR_DATA_DIR`) to point at the data directory on your machine.

## Reports

Export TensorBoard training data to a PDF report:

```bash
uv run dr-report --logdir logs/vit_p16_e768_d12_h12_c5/
uv run dr-report --logdir logs/vit_p16_e768_d12_h12_c5/ --output my_report.pdf
```

## Training

### Smoke test

Quick local validation with a tiny model and a handful of images:

```bash
# 1. Create a small pickle index (first 80 images with matching saliency maps)
uv run python scripts/create_smoke_dataset.py

# 2. Run one pretraining epoch on CPU
uv run dr-train --phase pretrain --device cpu --config configs/pretrain_smoke.yaml

# 3. DDP smoke — two processes, no GPU needed (gloo backend on CPU)
uv run torchrun --nnodes=1 --nproc-per-node=2 -m dr_model.training.cli \
    --phase pretrain --device cpu --config configs/pretrain_smoke.yaml
```

This logs to MLflow (`mlflow` experiment `dr-pretrain-smoke`) and TensorBoard (`logs/`).

### Full pretraining

```bash
uv run dr-train --phase pretrain
```

Uses `configs/pretrain_default.yaml` by default. Override with `--config` or `DR_CONFIG_FILE`.

### Distributed / multi-GPU (DDP)

Single-node data-parallel training via `torchrun`. Each process detects the
`torchrun` environment, pins itself to `cuda:{local_rank}`, and the model is
wrapped in `DistributedDataParallel` (with BatchNorm converted to
`SyncBatchNorm` on CUDA).

```bash
# Single GPU
uv run dr-train --phase pretrain --device cuda

# N GPUs on one workstation
uv run torchrun --nnodes=1 --nproc-per-node=N -m dr_model.training.cli \
    --phase pretrain --device cuda

# HPC cluster via Slurm + Singularity (see docs/hpc.md)
./scripts/submit_train.sh smoke 2     # 2x GPU — proves DDP gradient sync
./scripts/submit_train.sh full 4      # 4x GPU full pretraining
```

Notes:

- Only rank 0 writes checkpoints and logs to MLflow/TensorBoard; other ranks
  only train and participate in gradient sync.
- Under CUDA torchrun, `--device` is ignored — each rank always uses
  `cuda:{local_rank}`.
- Per-epoch losses are all-reduced across ranks (sums + counts), so global
  averages stay correct even with uneven shards.
- Checkpoints are saved from the *unwrapped* model, so keys never carry a
  `module.` prefix.

## Testing

```bash
uv run python -m pytest tests/ -v
```

## Type checking and linting

```bash
uv run mypy src/
uv run ruff check src/ tests/
```
