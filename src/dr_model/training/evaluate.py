"""Standalone evaluation of a fine-tuned checkpoint on one split.

Loads trained ``best_validation_weights.pt``-style checkpoints, evaluates
them through the same ``_evaluate`` path used during training, and writes
the full artifact bundle (metrics JSON, confusion matrix, raw labels,
predictions, and probabilities) to the external artifact directory.
No MLflow or TensorBoard involvement.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

from dr_model.config import Settings
from dr_model.data.finetune_datamodule import FinetuneDataModule
from dr_model.model.finetune import Finetuner
from dr_model.training.artifacts import write_finetune_artifacts
from dr_model.training.finetune_loop import (
    _evaluate,
    _load_model_checkpoint,
    build_finetune_criterion,
)
from dr_model.utils import resolve_device


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dr-evaluate",
        description="Evaluate a fine-tuned checkpoint on the validation or test split.",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a .pt training checkpoint.")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Dataset root containing train/val/test split dirs (overrides config).",
    )
    parser.add_argument(
        "--split",
        choices=["validation", "test"],
        default="validation",
        help="Which split to evaluate.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Artifact run name (defaults to the checkpoint's parent directory name).",
    )
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cpu, cuda, mps.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (overrides config).")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers (overrides config).")
    args = parser.parse_args(argv)

    if args.config is not None:
        os.environ["DR_CONFIG_FILE"] = args.config
    else:
        os.environ.setdefault("DR_CONFIG_FILE", "configs/finetune_default.yaml")

    config = Settings()
    updates: dict[str, object] = {}
    if args.dataset_root is not None:
        updates["finetune_dataset_root"] = Path(args.dataset_root)
    if args.batch_size is not None:
        updates["batch_size"] = args.batch_size
    if args.num_workers is not None:
        updates["num_workers"] = args.num_workers
    if updates:
        config = config.model_copy(update=updates)

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    checkpoint_path = Path(args.checkpoint)
    run_name = args.run_name or checkpoint_path.parent.name.lower()

    model: torch.nn.Module = Finetuner(config).to(device)
    _load_model_checkpoint(checkpoint_path, model, device)

    datamodule = FinetuneDataModule(config)
    datamodule.setup()
    loader = datamodule.val_dataloader() if args.split == "validation" else datamodule.test_dataloader()

    result = _evaluate(model, loader, build_finetune_criterion(config), device)

    output = write_finetune_artifacts(
        config.artifact_dir,
        config,
        [],
        result if args.split == "validation" else None,
        result if args.split == "test" else None,
        run_name=run_name,
    )

    metrics = result.metrics
    auc = "n/a" if metrics.auc_macro is None else f"{metrics.auc_macro:.4f}"
    print(
        f"{args.split}: kappa={metrics.kappa:.4f}  accuracy={metrics.accuracy:.4f}  "
        f"f1_macro={metrics.f1_macro:.4f}  f1_weighted={metrics.f1_weighted:.4f}  "
        f"auc_macro={auc}  n={len(result.labels)}"
    )
    print(f"Artifacts written to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
