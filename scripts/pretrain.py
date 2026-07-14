"""Pretraining entry point.

Loads settings from the YAML config (or ``DR_CONFIG_FILE`` env var),
builds the ``Pretrainer`` + ``PretrainDataModule``, and runs the
contrastive-saliency pretraining loop.

Usage::

    uv run scripts/pretrain.py
    uv run scripts/pretrain.py --config configs/pretrain_custom.yaml
    uv run scripts/pretrain.py --resume checkpoints/vit_.../checkpoint.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

os.environ.setdefault("DR_CONFIG_FILE", "configs/pretrain_default.yaml")

from dr_model.config import Settings
from dr_model.data.pretrain_datamodule import PretrainDataModule
from dr_model.model.pretrain import Pretrainer
from dr_model.training.pretrain_loop import pretrain


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SSiT pretraining")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint.pt to resume from.",
    )
    args = parser.parse_args(argv)

    if args.config is not None:
        os.environ["DR_CONFIG_FILE"] = args.config

    config = Settings()
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    dm = PretrainDataModule(config)
    dm.setup()
    dl = dm.train_dataloader()

    model = Pretrainer(config).to(device)

    log_dir = config.log_dir / config.model_name
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    resume_path = Path(args.resume) if args.resume else None

    pretrain(
        model=model,
        train_dataloader=dl,
        config=config,
        writer=writer,
        resume_path=resume_path,
    )
    print("Pretraining complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
