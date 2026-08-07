"""Fine-tune a ViT backbone for supervised DR grading (SSiT method).

Research entry point for the supervised fine-tuning stage (DRC-49).  Loads
a YAML config, builds a :class:`~dr_model.model.finetune.Finetuner` (seeding
the trunk from ``finetune_checkpoint`` when given), and runs the supervised
fine-tuning loop, reporting quadratic Cohen's kappa on the validation split
each epoch.

Usage::

    python scripts/finetune.py --config configs/finetune_default.yaml
"""

from __future__ import annotations

import argparse
import os
import sys

# Must be set before importing torch (CuBLAS reads it at init).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch.nn as nn

os.environ.setdefault("DR_CONFIG_FILE", "configs/finetune_default.yaml")

from dr_model.config import Settings
from dr_model.data.finetune_datamodule import FinetuneDataModule
from dr_model.logging.tensorboard_utils import init_tensorboard_logger
from dr_model.model.finetune import Finetuner
from dr_model.training.finetune_loop import run
from dr_model.utils import resolve_device, setup_determinism


def _build_config(args: argparse.Namespace) -> Settings:
    """Load settings, applying command-line overrides on top of the YAML."""
    if args.config is not None:
        os.environ["DR_CONFIG_FILE"] = args.config

    config = Settings()
    updates: dict[str, object] = {}
    if args.seed is not None:
        updates["seed"] = args.seed
    if args.finetune_epochs is not None:
        updates["finetune_epochs"] = args.finetune_epochs
    if args.batch_size is not None:
        updates["batch_size"] = args.batch_size
    if args.finetune_checkpoint is not None:
        updates["finetune_checkpoint"] = args.finetune_checkpoint
    if args.num_workers is not None:
        updates["num_workers"] = args.num_workers
    if updates:
        config = config.model_copy(update=updates)
    return config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="finetune",
        description="Fine-tune a ViT backbone for supervised DR grading (SSiT).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (overrides DR_CONFIG_FILE env).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto, cpu, cuda, mps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config). Set to -1 to disable.",
    )
    parser.add_argument(
        "--finetune-epochs",
        type=int,
        default=None,
        help="Number of fine-tuning epochs (overrides config).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config).",
    )
    parser.add_argument(
        "--finetune-checkpoint",
        type=str,
        default=None,
        help="Pretrain checkpoint to seed the trunk (overrides config).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader worker processes (overrides config).",
    )
    args = parser.parse_args(argv)

    config = _build_config(args)

    device = resolve_device(args.device)
    if config.seed >= 0:
        setup_determinism(config.seed, deterministic_algorithms=config.deterministic_algorithms)
    print(f"Using device: {device}")

    dm = FinetuneDataModule(config)
    dm.setup()

    model: nn.Module = Finetuner(config, checkpoint_path=config.finetune_checkpoint).to(device)

    writer = None
    if config.tensorboard:
        writer = init_tensorboard_logger(config, run_name=f"finetune_{config.model_name}")

    try:
        run(config=config, model=model, datamodule=dm, device=device, writer=writer)
    finally:
        if writer is not None:
            writer.close()

    print("Fine-tuning complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
