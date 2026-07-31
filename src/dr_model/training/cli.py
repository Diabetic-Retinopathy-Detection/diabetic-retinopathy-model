from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Must be set before importing torch (CuBLAS reads it at init).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import mlflow
import torch
import torch.nn as nn

os.environ.setdefault("DR_CONFIG_FILE", "configs/pretrain_default.yaml")

from dr_model.config import Settings
from dr_model.data.pretrain_datamodule import PretrainDataModule
from dr_model.logging import end_mlflow_run, init_mlflow_run, init_tensorboard_logger
from dr_model.model.pretrain import Pretrainer
from dr_model.training.distributed import (
    DistributedContext,
    cleanup_distributed,
    detect_distributed_context,
    init_distributed,
    is_rank_zero,
    wrap_model,
)
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
        "--deterministic",
        action="store_true",
        default=None,
        help="Enable full GPU determinism (slower, bitwise reproducible). Overrides config.deterministic_algorithms.",
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
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Root data directory containing cropped/ and saliency/ (overrides config).",
    )
    args = parser.parse_args(argv)

    if args.config is not None:
        os.environ["DR_CONFIG_FILE"] = args.config

    config = Settings()
    updates: dict[str, object] = {}
    if args.seed is not None:
        updates["seed"] = args.seed
    if args.deterministic is not None:
        updates["deterministic_algorithms"] = args.deterministic
    if args.data_index_path is not None:
        updates["data_index_path"] = Path(args.data_index_path)
    if args.data_dir is not None:
        updates["data_dir"] = Path(args.data_dir)
    if updates:
        config = config.model_copy(update=updates)

    ctx = detect_distributed_context(args.device)
    device = ctx.device if ctx.enabled else resolve_device(args.device)
    if config.seed >= 0:
        setup_determinism(config.seed, deterministic_algorithms=config.deterministic_algorithms)
    if is_rank_zero(ctx):
        print(f"Using device: {device}")

    if args.phase == "pretrain":
        return _run_pretrain(config, device, args.resume, ctx)
    print(f"Phase '{args.phase}' not implemented yet.")
    return 1


def _run_pretrain(
    config: Settings,
    device: torch.device,
    resume: str | None,
    ctx: DistributedContext | None = None,
) -> int:
    if ctx is None:
        ctx = detect_distributed_context(str(device))
    if ctx.enabled:
        device = ctx.device
    init_distributed(ctx)

    dm = PretrainDataModule(config)
    dm.setup()
    dl = dm.train_dataloader()

    model: nn.Module = Pretrainer(config).to(device)
    model = wrap_model(model, ctx)

    writer = None
    if is_rank_zero(ctx):
        if config.mlflow:
            init_mlflow_run(
                config,
                experiment_name=config.mlflow_experiment_name,
                run_name_prefix="pretrain",
                device=device,
            )
            mlflow.log_dict(config.model_dump(), "config.yaml")

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
            rank=ctx.rank,
            world_size=ctx.world_size,
            sampler=dm.sampler,
        )
    finally:
        if writer is not None:
            writer.close()
        if is_rank_zero(ctx) and config.mlflow:
            end_mlflow_run()
        cleanup_distributed()

    if is_rank_zero(ctx):
        print("Pretraining complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
