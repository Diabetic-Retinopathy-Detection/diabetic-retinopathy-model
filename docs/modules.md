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

::: dr_model.model.backbone

## Pretraining Data

`PairDataset` loads image-saliency pairs from the pickle index produced by
`build-dataset-index`. Paths in the pickle are stored **relative** to a common
root directory, making the index portable across machines. The dataset resolves
these paths against `Settings.data_dir`, which should be set via the
`DR_DATA_DIR` environment variable per machine.

::: dr_model.data

## Inference

::: dr_model.inference

## Serving

::: dr_model.serve.app

## Training

::: dr_model.training.cli
