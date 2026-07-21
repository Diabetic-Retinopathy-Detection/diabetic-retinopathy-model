# Diabetic Retinopathy Model

ViT-based model for diabetic retinopathy classification — training, inference, and serving.

## Project structure

```
src/dr_model/
├── config.py            # Pydantic Settings — all hyper-parameters in one place
├── model/
│   ├── __init__.py      # Public API: ViTBackbone
│   └── backbone.py      # Vision Transformer encoder + classification head
├── inference/
│   └── __init__.py      # predict() — single-image inference pipeline
├── serve/
│   └── app.py           # FastAPI serving endpoint
└── training/
    └── cli.py           # Training entry-point (dr-train)
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

## Testing

```bash
uv run python -m pytest tests/ -v
```

## Type checking and linting

```bash
uv run mypy src/
uv run ruff check src/ tests/
```
