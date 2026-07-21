# Configuration

`Settings` is a Pydantic `BaseSettings` class that loads from YAML, `.env` files, and environment variables. It is the single source of truth for every hyper-parameter — architecture, training, data, and serving.

## model_name (auto-generated)

`model_name` is a `@computed_field` — derived from architecture parameters, not manually set:

```python
config = Settings()
config.model_name  # "vit_p16_e768_d12_h12_c5"
```

Format: `vit_p{patch_size}_e{embed_dim}_d{depth}_h{num_heads}_c{num_classes}`

This name is used for checkpoint filenames, logging, and serving metadata. Non-encoded parameters (`mlp_ratio`, `drop_rate`, etc.) are stored in the checkpoint file itself.

## image_size normalisation

`image_size` accepts either a tuple or a scalar int. Scalars are normalised to a square:

```python
Settings(image_size=256).image_size   # (256, 256)
Settings(image_size=(224, 192)).image_size  # (224, 192)
```

## Config priority

1. Environment variables
2. `.env` file
3. YAML file (default: `configs/pretrain_default.yaml`)
4. Class defaults

Override the YAML path via the `DR_CONFIG_FILE` environment variable.

## CLI usage

Both entry points accept the same flags:

```bash
# Registered entry point (pyproject.toml)
uv run dr-train --phase pretrain --device mps --seed 42

# Direct script
uv run scripts/pretrain.py --device mps --seed 42
```

| Flag | Description |
|---|---|
| `--phase` | `pretrain` or `finetune` (default: `pretrain`) |
| `--device` | `auto`, `cpu`, `cuda`, `mps` (default: `auto`) |
| `--seed` | Random seed, `-1` to disable (overrides config) |
| `--config` | Path to YAML config file |
| `--resume` | Path to `checkpoint.pt` to resume from |
| `--data-index-path` | Path to pretraining pickle index (overrides config) |

### Examples

```bash
# Use a different pickle index
uv run dr-train --phase pretrain --data-index-path /other/dataset.pkl

# Custom config + resume
uv run dr-train --phase pretrain --config configs/pretrain_custom.yaml --resume checkpoints/vit_.../checkpoint.pt

# Deterministic run on MPS
uv run dr-train --phase pretrain --device mps --seed 42
```

Override via YAML `data_index_path:` or environment variable `DR_DATA_INDEX_PATH`.

## data_dir

Absolute path to the shared data directory containing images and saliency maps.
Override via YAML `data_dir:` or environment variable `DR_DATA_DIR`.

See [Modules — Pretraining Data](modules.md#pretraining-data) for the directory layout.

## seed

Random seed for reproducible training. Defaults to `-1` (disabled), matching SSiT convention. When `seed >= 0`, `setup_determinism()` is called before model creation — this sets PyTorch/CUDA/cuDNN seeds and enables deterministic algorithms.

Override via CLI: `--seed 42` or YAML `seed: 42`.
