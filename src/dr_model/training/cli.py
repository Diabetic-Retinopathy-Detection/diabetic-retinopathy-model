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

from dr_model.config import Settings
from dr_model.data.finetune_datamodule import FinetuneDataModule
from dr_model.data.pretrain_datamodule import PretrainDataModule
from dr_model.logging import end_mlflow_run, init_mlflow_run, init_tensorboard_logger
from dr_model.model.finetune import Finetuner
from dr_model.model.pretrain import Pretrainer
from dr_model.training.distributed import (
    DistributedContext,
    cleanup_distributed,
    detect_distributed_context,
    init_distributed,
    is_rank_zero,
    wrap_model,
)
from dr_model.training.finetune_loop import run
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
        help="Pretrain checkpoint seeding the fine-tuning trunk (overrides config).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader worker processes (overrides config).",
    )
    args = parser.parse_args(argv)

    if args.config is not None:
        os.environ["DR_CONFIG_FILE"] = args.config
    else:
        default_config = (
            "configs/pretrain_default.yaml" if args.phase == "pretrain" else "configs/finetune_default.yaml"
        )
        os.environ.setdefault("DR_CONFIG_FILE", default_config)

    config = Settings()
    updates = _collect_updates(args)
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
    return _run_finetune(config, device, ctx)


def _collect_updates(args: argparse.Namespace) -> dict[str, object]:
    """Map CLI overrides onto ``Settings`` field names."""
    updates: dict[str, object] = {}
    if args.seed is not None:
        updates["seed"] = args.seed
    if args.deterministic is not None:
        updates["deterministic_algorithms"] = args.deterministic
    if args.data_index_path is not None:
        updates["data_index_path"] = Path(args.data_index_path)
    if args.data_dir is not None:
        updates["data_dir"] = Path(args.data_dir)
    if args.finetune_epochs is not None:
        updates["finetune_epochs"] = args.finetune_epochs
    if args.batch_size is not None:
        updates["batch_size"] = args.batch_size
    if args.finetune_checkpoint is not None:
        updates["finetune_checkpoint"] = args.finetune_checkpoint
    if args.num_workers is not None:
        updates["num_workers"] = args.num_workers
    return updates


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


def _run_finetune(
    config: Settings,
    device: torch.device,
    ctx: DistributedContext | None = None,
) -> int:
    if ctx is None:
        ctx = detect_distributed_context(str(device))
    if ctx.enabled:
        device = ctx.device
    init_distributed(ctx)

    dm = FinetuneDataModule(config)
    dm.setup()

    model: nn.Module = Finetuner(config, checkpoint_path=config.finetune_checkpoint).to(device)
    model = wrap_model(model, ctx)

    writer = None
    if is_rank_zero(ctx):
        if config.mlflow:
            init_mlflow_run(
                config,
                experiment_name=config.mlflow_experiment_name,
                run_name_prefix="finetune",
                device=device,
            )
            mlflow.log_dict(config.model_dump(), "config.yaml")

        if config.tensorboard:
            writer = init_tensorboard_logger(config, run_name=f"finetune_{config.model_name}")

    try:
        run(
            config=config,
            model=model,
            datamodule=dm,
            device=device,
            writer=writer,
            rank=ctx.rank,
            world_size=ctx.world_size,
        )
    finally:
        if writer is not None:
            writer.close()
        if is_rank_zero(ctx) and config.mlflow:
            end_mlflow_run()
        cleanup_distributed()

    if is_rank_zero(ctx):
        print("Fine-tuning complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
