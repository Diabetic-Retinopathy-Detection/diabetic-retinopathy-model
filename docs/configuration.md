# Configuration

`Settings` is a Pydantic `BaseSettings` class that loads from YAML, `.env` files, and environment variables. It is the single source of truth for every hyper-parameter — architecture, training, data, and serving.

## model_name (auto-generated)

`model_name` is a `@computed_field` — derived from architecture parameters, not manually set:

```python
config = Settings()
config.model_name  # "vit_p16_e768_d12_h12_c5"
```

Format: `vit_p{patch_size}_e{embed_dim}_d{depth}_h{num_heads}_c{num_classes}`

This name is used for pretraining checkpoint directory names, logging, and
serving metadata. It does not encode every configuration value, and
checkpoints do not persist a complete `Settings` object.

## image_size normalisation

`image_size` accepts either a tuple or a scalar int. Scalars are normalised to a square:

```python
Settings(image_size=256).image_size   # (256, 256)
Settings(image_size=(224, 192)).image_size  # (224, 192)
```

## Config priority

1. Constructor arguments
2. Environment variables
3. `.env` file
4. YAML file (default: `configs/pretrain_default.yaml`)
5. Class defaults

Override the YAML path via the `DR_CONFIG_FILE` environment variable.

## CLI usage

Both entry points accept the same flags:

```bash
# Registered entry point (pyproject.toml)
uv run dr-train --phase pretrain --device mps --seed 42

# Direct module invocation
uv run python -m dr_model.training.cli --phase pretrain --device mps --seed 42
```

| Flag | Description |
|---|---|
| `--phase` | `pretrain` or `finetune` (default: `pretrain`) |
| `--device` | `auto`, `cpu`, `cuda`, `mps` (default: `auto`) |
| `--seed` | Random seed, `-1` to disable (overrides config) |
| `--config` | Path to YAML config file |
| `--resume` | Path to a checkpoint to resume from (pretraining and fine-tuning where supported) |
| `--data-index-path` | Path to pretraining pickle index (overrides config) |
| `--data-dir` | Root data directory containing `cropped/` and `saliency/` (overrides config) |
| `--finetune-epochs` | Number of fine-tuning epochs (overrides config) |
| `--batch-size` | Batch size (overrides config) |
| `--finetune-checkpoint` | Pretrain checkpoint seeding the fine-tuning trunk (overrides config) |
| `--num-workers` | DataLoader worker processes (overrides config) |
| `--train-on-train-and-valid` / `--no-train-on-train-and-valid` | Train on a virtual concatenation of the train and validation splits |
| `--skip-validation` / `--no-skip-validation` | Skip validation during fine-tuning |
| `--deterministic` | Enable deterministic algorithms for the run |
| `--finetune-extra-epochs` | Extend a resumed fine-tuning run by this many epochs |

### Examples

```bash
# Use a different pickle index
uv run dr-train --phase pretrain --data-index-path /other/dataset.pkl

# Custom config + resume
uv run dr-train --phase pretrain --config configs/pretrain_custom.yaml --resume checkpoints/vit_.../checkpoint.pt

# Seeded run on MPS; deterministic algorithms require the explicit flag
uv run dr-train --phase pretrain --device mps --seed 42 --deterministic

# Final fixed-epoch fit on train + validation without validation metrics
uv run dr-train --phase finetune --config configs/finetune_ddr.yaml \
    --device cuda --finetune-epochs 7 \
    --train-on-train-and-valid --skip-validation
```

Override via YAML `data_index_path:` or environment variable `DR_DATA_INDEX_PATH`.

## data_dir

Absolute path to the shared data directory containing images and saliency maps.
Override via YAML `data_dir:` or environment variable `DR_DATA_DIR`.

See [Modules — Pretraining Data](modules.md#pretraining-data) for the directory layout.

## seed

Random seed for reproducible training. Defaults to `-1` (disabled), matching
SSiT convention. When `seed >= 0`, the run seeds PyTorch, CUDA, cuDNN, and data
loader workers before model creation. Deterministic algorithm selection is a
separate option controlled by `deterministic_algorithms: true` or the
`--deterministic` CLI flag.

Override via CLI: `--seed 42` or YAML `seed: 42`.
