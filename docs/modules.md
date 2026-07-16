# API Reference

::: dr_model.config

## Vision Transformer

The backbone implements a standard ViT-B/16 (Dosovitskiy et al., 2020).
All architecture parameters are read from `Settings` -- no magic constants.

### Forward pass

1. **Patch embedding** via a single `Conv2d`: `(B, 3, H, W)` -> `(B, N, E)` where `N = (H/P) * (W/P)`.
2. **Prepend learnable class token**: `(B, N, E)` -> `(B, N+1, E)`.
3. **Add positional embedding** (class-invariant: `pos_embed.shape[1] == N + 1`).
4. **Transformer encoder blocks** with pre-norm attention + pre-norm MLP.
5. **Classification head** reads from the class token only -> `(B, num_classes)` logits.

### `forward_features(x, pmap=None)`

Returns the full encoder sequence `[CLS, patch_1, ..., patch_N]` without the classification head.
When `pmap` is provided, applies saliency-guided patch masking: only the top-k patches (by saliency score) are fed to the transformer blocks. This is used by the momentum encoder during pretraining.

::: dr_model.model.backbone

## Pretrainer

Saliency-supervised self-distillation pretrainer (SSiT). Two simultaneous objectives:

1. **MoCo v3-style contrastive learning** across two augmented views.
2. **Saliency segmentation auxiliary task** that forces the encoder to localise clinically relevant regions.

### Architecture

- **Base encoder**: `ViTBackbone` trained by gradient. Head replaced by a 3-layer projector MLP (`embed_dim` -> `mlp_dim` -> `dim`).
- **Momentum encoder**: `ViTBackbone` updated via EMA. Same architecture, `requires_grad=False`.
- **Predictor**: 2-layer MLP on top of the base encoder's projection (BYOL-style asymmetry).
- **Saliency segmentor**: `Conv2d(embed_dim, patch_size^2, 1)` + `PixelShuffle(patch_size)` applied to patch tokens from the base encoder.
- **Saliency pool**: `MaxPool2d(patch_size, patch_size)` downsamples saliency maps to patch resolution for momentum encoder masking.

### Forward pass

```
forward(x1, x2, m1, m2, momentum_m) -> (contrastive_loss, saliency_loss)
```

1. Pool saliency maps: `mp1 = pool(m1)`, `mp2 = pool(m2)`.
2. Base encoder forward: `t1, f1 = _encode(base_encoder, x1)` -- full image, no masking.
3. Predictor: `q1 = predictor(t1)`.
4. Momentum update (no grad): `_update_momentum_encoder(momentum_m)`.
5. Momentum encoder forward: `k1, _ = _encode(momentum_encoder, x1, mp1)` -- with saliency-guided masking.
6. Contrastive loss: `contrastive_loss(q1, k2) + contrastive_loss(q2, k1)`.
7. Saliency loss: `saliency_segmentation_loss(f1, m1) + saliency_segmentation_loss(f2, m2)`.

### InfoNCE contrastive loss

L2-normalize queries and keys, `concat_all_gather` on keys (no-op in single-GPU), einsum cosine similarity logits, cross-entropy on diagonal labels. Handles both distributed and single-GPU without `all_gather` crash.

### Saliency segmentation loss

Strip CLS token from features, reshape to `(B, C, H, W)`, pass through segmentor, `binary_cross_entropy_with_logits` against thresholded saliency map.

::: dr_model.model.pretrain

## Pretraining Data

Both repos share a data directory outside either repo:

```
DiabeticRetinopathy/
├── data/                    # shared, gitignored
│   ├── cropped/             # JPEG fundus crops
│   ├── saliency/            # .npy saliency maps
│   └── dataset.pkl          # portable index (relative paths)
├── preprocess-retina-datasets/   # writes into data/
└── diabetic-retinopathy-model/   # reads from data/
```

`PairDataset` loads image-saliency pairs from the pickle index produced by
`preprocess-retina-datasets`. Paths in the pickle are stored **relative** to a common
root directory, making the index portable across machines.

`TransformWithMask` applies asymmetric student/teacher augmentation with paired spatial transforms. Every spatial transform (crop, rotation, flip) is applied to the image and mask with the same sampled parameters. Colour-only transforms are applied to the image only.

`PretrainDataModule` loads the pickle, resolves relative paths, optionally subsamples via shuffle-then-truncate, and exposes a single `train_dataloader()`.

::: dr_model.data

## Inference

::: dr_model.inference

## Serving

::: dr_model.serve.app

## Pretraining Loop

Runs the full pretraining loop for the saliency-guided MoCo v3 contrastive objective.
Follows the schedule and optimiser design from the SSiT paper.

### Schedules

All schedules are driven by fractional training progress `t = step_ratio / max_epochs`:

- **Learning rate**: linear warmup from 0 to `learning_rate` over `warmup_epochs`, then cosine decay to zero.
- **Momentum**: cosine ramp from `momentum_base` (0.99) to `momentum_max` (1.0) for the teacher EMA update.
- **Saliency weight** (`lambda_s`): optional cosine decay from `lambda_s` to zero when `ss_decay=True`. Disabled by default.

### Checkpoints

- Full training state (model, optimizer, epoch, scaler) saved to `checkpoint.pt` every `save_every` epochs.
- Encoder-only weights saved to `epoch_{N}_encoder.pt` at the same intervals.
- Final checkpoint always saved after the last epoch.
- `resume_path` restores model, optimizer, epoch, and scaler for interrupted runs.

### Mixed precision

- CUDA: full AMP support via `torch.amp.GradScaler`.
- MPS/CPU: no scaler (PyTorch limitation). Precision flag is ignored.

### Logging

TensorBoard logging when a `SummaryWriter` is provided: contrastive loss, saliency loss, total loss, learning rate, and momentum per epoch.

::: dr_model.training.pretrain_loop

## Training CLI

### Running pretraining

```bash
uv run dr-train --phase pretrain --device mps --seed 42
```

Loads `Settings` from the YAML config (default: `configs/pretrain_default.yaml`), builds the `PretrainDataModule` + `Pretrainer`, and runs the contrastive-saliency loop. Key flags: `--device`, `--seed`, `--data-index-path`, `--config`, `--resume`.

### Training reports

Export TensorBoard scalars to a PDF report:

```bash
uv run dr-report --logdir logs/vit_p16_e768_d12_h12_c5/
uv run dr-report --logdir logs/vit_p16_e768_d12_h12_c5/ --output report.pdf
```

Generates a multi-page PDF with loss curves (contrastive, saliency, total), learning rate schedule, momentum schedule, and a summary page with final/best metrics.

::: dr_model.training.report

### Pretraining loop

`pretrain(model, train_dataloader, config, device=..., writer=..., resume_path=...)` runs the full MoCo v3 contrastive + saliency segmentation pretraining loop.

- **LR schedule**: linear warmup for `warmup_epochs`, then cosine decay to zero.
- **Momentum schedule**: cosine ramp from `momentum_base` to `momentum_max` (fractional progress `t ∈ [0, 1]`).
- **Lambda_s schedule**: optional cosine decay of saliency loss weight (enabled via `ss_decay`).
- **AMP**: enabled when `precision == "16-mixed"` and `device.type == "cuda"`.
- **Checkpointing**: interval saves at `save_every` epochs + final epoch. Saves both full training state (`checkpoint.pt`) and encoder-only weights (`epoch_{N}_encoder.pt`).
- **TensorBoard**: logs `loss/contrastive`, `loss/saliency`, `loss/total`, `lr`, `momentum_m` per epoch.

The caller resolves the device and moves the model before calling `pretrain()`. This keeps device logic out of the loop and simplifies testing.

::: dr_model.training.pretrain_loop

### Running pretraining

```bash
uv run dr-train --phase pretrain --device mps --seed 42
```

Loads `Settings` from the YAML config (default: `configs/pretrain_default.yaml`), builds the `PretrainDataModule` + `Pretrainer`, and runs the contrastive-saliency loop. Key flags: `--device`, `--seed`, `--data-index-path`, `--config`, `--resume`.

### Training reports

Export TensorBoard scalars to a PDF report:

```bash
uv run dr-report --logdir logs/vit_p16_e768_d12_h12_c5/
uv run dr-report --logdir logs/vit_p16_e768_d12_h12_c5/ --output report.pdf
```

Generates a multi-page PDF with loss curves (contrastive, saliency, total), learning rate schedule, momentum schedule, and a summary page with final/best metrics.

::: dr_model.training.report

### Pretraining loop

`pretrain(model, train_dataloader, config, device=..., writer=..., resume_path=...)` runs the full MoCo v3 contrastive + saliency segmentation pretraining loop.

- **LR schedule**: linear warmup for `warmup_epochs`, then cosine decay to zero.
- **Momentum schedule**: cosine ramp from `momentum_base` to `momentum_max` (fractional progress `t ∈ [0, 1]`).
- **Lambda_s schedule**: optional cosine decay of saliency loss weight (enabled via `ss_decay`).
- **AMP**: enabled when `precision == "16-mixed"` and `device.type == "cuda"`.
- **Checkpointing**: interval saves at `save_every` epochs + final epoch. Saves both full training state (`checkpoint.pt`) and encoder-only weights (`epoch_{N}_encoder.pt`).
- **TensorBoard**: logs `loss/contrastive`, `loss/saliency`, `loss/total`, `lr`, `momentum_m` per epoch.
- **MLflow**: logs the same 5 metrics per epoch, plus all Settings fields as params (when `config.mlflow == True`).

The caller resolves the device and moves the model before calling `pretrain()`. This keeps device logic out of the loop and simplifies testing.

::: dr_model.training.pretrain_loop

::: dr_model.training.cli

## Experiment Logging

Dual logging to MLflow (run comparison) and TensorBoard (live curves). Both record the same metrics — see [Experiment Tracking](experiment-tracking.md) for usage.

::: dr_model.logging

## Utils

Device resolution and determinism helpers for reproducible training.

- `resolve_device(device_str)` — `"auto"` prefers CUDA, then MPS, then CPU. Other strings pass through.
- `setup_determinism(seed)` — sets PyTorch/CUDA/cuDNN seeds and enables deterministic algorithms. Call before any CUDA work.
- `setup_cublas_workspace()` — sets `CUBLAS_WORKSPACE_CONFIG` for deterministic cuBLAS. Called automatically by `setup_determinism`.

::: dr_model.utils
