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
