# Evaluation — kNN Protocol

Linear-probe-free evaluation of a pretrained encoder: for each test image, the
k nearest train images are found by cosine similarity in the frozen encoder's
feature space, and distance-weighted voting assigns the predicted class.

Ported from SSiT's `knn.py` (itself based on DINO's `eval_knn.py`). Metrics
reported: **Top-1 accuracy** and **Quadratic Weighted Kappa (QWK)**.

## Requirements

- A pretrained encoder checkpoint — either an encoder-only
  `epoch_{N}_encoder.pt` or a full `checkpoint.pt` (the loader strips the
  `base_encoder.` prefix automatically).
- The evaluation dataset in `torchvision.datasets.ImageFolder` layout with
  `train/` and `test/` subdirectories, one folder per class (grades `0`-`4`;
  grade 5 "unreadable" is excluded, matching SSiT's 5-class protocol).
- Per-dataset mean/std normalisation stats. Supported datasets:
  `ddr`, `aptos2019`, `messidor2` (default `ddr`).

## Running the evaluation

```zsh
uv run knn-probe \
    --method knn \
    --encoder checkpoints/full-pretraining/epoch_60_encoder.pt \
    --data-path ../datasets/DDR-ImageFolder \
    --dataset ddr \
    --device mps \
    --output ../datasets/DDR-ImageFolder/knn_results_epoch60.json
```

`--device auto` prefers CUDA, then MPS, then CPU — `mps` is typical on a Mac,
`cuda` on the cluster. The `--output` flag writes the results JSON.

Flags:

| Flag            | Default       | Description                                     |
| --------------- | ------------- | ----------------------------------------------- |
| `--method`      | — (required)  | `knn` (only method implemented).                |
| `--encoder`     | — (required)  | Path to the pretrained encoder checkpoint.      |
| `--data-path`   | — (required)  | ImageFolder root containing `train/` + `test/`. |
| `--dataset`     | `ddr`         | Dataset name for normalisation stats.           |
| `--input-size`  | `224`         | Resize target.                                  |
| `--batch-size`  | `32`          | Feature-extraction batch size.                  |
| `--num-workers` | `4`           | DataLoader workers.                             |
| `--knn`         | `5 10 20`     | Neighbour counts to evaluate.                   |
| `--temperature` | `0.07`        | Softmax temperature for distance weighting.     |
| `--device`      | `auto`        | `auto`, `cuda`, `mps`, or `cpu`.                |
| `--output`      | —             | Optional JSON path for the results.             |

Example output:

```zsh
Using device: mps
Loaded bare encoder weights (missing: 0, unexpected: 0)
Data loaded: <n> train, <m> test images.
Extracting train features...
Extracting test features...
Features extracted. Starting kNN classification.
5-NN: Acc=61.64%, Kappa=0.4876
10-NN: Acc=62.25%, Kappa=0.4992
20-NN: Acc=61.59%, Kappa=0.4790
Results saved to ../datasets/DDR-ImageFolder/knn_results_epoch60.json
```

## Interpreting results

The results JSON follows this schema:

```json
{
  "k-NN": {
    "5-NN":  {"accuracy": 61.64, "kappa": 0.488},
    "10-NN": {"accuracy": 62.25, "kappa": 0.499},
    "20-NN": {"accuracy": 61.59, "kappa": 0.479}
  }
}
```

Convention: save each run as `knn_results_epoch{N}.json` next to the
ImageFolder, so checkpoints across pretraining epochs can be compared
(e.g. `knn_results_epoch40.json`, `knn_results_epoch43.json`).

## API Reference

::: dr_model.evaluation.cli

::: dr_model.evaluation.knn
