from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

os.environ.setdefault("DR_CONFIG_FILE", "configs/pretrain_default.yaml")

from dr_model.config import Settings
from dr_model.data.pretrain_datamodule import PretrainDataModule
from dr_model.logging import end_mlflow_run, init_mlflow_run, init_tensorboard_logger
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
    parser.add_argument(
        "--data-index-path",
        type=str,
        default=None,
        help="Path to pretraining pickle index (overrides config).",
    )
    args = parser.parse_args(argv)

    if args.config is not None:
        os.environ["DR_CONFIG_FILE"] = args.config

    config = Settings()
    updates: dict[str, object] = {}
    if args.seed is not None:
        updates["seed"] = args.seed
    if args.data_index_path is not None:
        updates["data_index_path"] = Path(args.data_index_path)
    if updates:
        config = config.model_copy(update=updates)
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

    if config.mlflow:
        init_mlflow_run(
            config,
            experiment_name=config.mlflow_experiment_name,
            run_name_prefix="pretrain",
            device=device,
        )

    writer = None
    if config.tensorboard:
        writer = init_tensorboard_logger(config, run_name=f"pretrain_{config.model_name}")

    resume_path = Path(resume) if resume else None

    try:
        pretrain(
            model=model,
            train_dataloader=dl,
            config=config,
            device=device,
            writer=writer,
            resume_path=resume_path,
        )
    finally:
        if writer is not None:
            writer.close()
        if config.mlflow:
            end_mlflow_run()

    print("Pretraining complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
