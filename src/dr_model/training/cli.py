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
from dr_model.utils import resolve_device, setup_determinism


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dr-train",
        description="Train a diabetic retinopathy model (pretrain or fine-tune).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (overrides DR_CONFIG_FILE env).",
    )
    parser.add_argument(
        "--phase",
        type=str,
        choices=["pretrain", "finetune"],
        default="pretrain",
        help="Training phase.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint.pt to resume from.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config). Set to -1 to disable.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto, cpu, cuda, mps.",
    )
    args = parser.parse_args(argv)

    if args.config is not None:
        os.environ["DR_CONFIG_FILE"] = args.config

    config = Settings()
    if args.seed is not None:
        config = config.model_copy(update={"seed": args.seed})
    device = resolve_device(args.device)
    if config.seed >= 0:
        setup_determinism(config.seed)
    print(f"Using device: {device}")

    if args.phase == "pretrain":
        return _run_pretrain(config, device, args.resume)
    print(f"Phase '{args.phase}' not implemented yet.")
    return 1


def _run_pretrain(config: Settings, device: torch.device, resume: str | None) -> int:
    dm = PretrainDataModule(config)
    dm.setup()
    dl = dm.train_dataloader()

    model = Pretrainer(config).to(device)

    log_dir = config.log_dir / config.model_name
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    resume_path = Path(resume) if resume else None

    pretrain(
        model=model,
        train_dataloader=dl,
        config=config,
        device=device,
        writer=writer,
        resume_path=resume_path,
    )
    print("Pretraining complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
