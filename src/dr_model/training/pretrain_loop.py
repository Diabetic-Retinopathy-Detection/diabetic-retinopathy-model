"""Pretraining loop for saliency-guided MoCo v3 contrastive learning.

Follows the schedule and optimiser design from SSiT
(``checkpoints/referencematerial/SSiT/train.py``) — cosine LR warmup,
cosine momentum ramp, optional cosine saliency-weight decay — without
reimplementing distributed training (single-process for now).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler

from dr_model.config import Settings
from dr_model.model.pretrain import Pretrainer
from dr_model.training.distributed import rank_zero_only, unwrap_model
from dr_model.utils.timer import Timer

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter


def _adjust_lr(optimizer: torch.optim.Optimizer, config: Settings, step_ratio: float) -> float:
    """Linear warmup then cosine decay to zero."""
    if step_ratio < config.warmup_epochs:
        lr = config.learning_rate * step_ratio / config.warmup_epochs
    else:
        progress = (step_ratio - config.warmup_epochs) / (config.max_epochs - config.warmup_epochs)
        decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = config.learning_rate * decay
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def _adjust_momentum(config: Settings, t: float) -> float:
    """Cosine ramp from ``momentum_base`` to ``momentum_max``.

    Parameters
    ----------
    t
        Fractional progress in ``[0, 1]``.
    """
    return 1.0 - 0.5 * (1.0 + math.cos(math.pi * t)) * (1.0 - config.momentum_base)


def _adjust_lambda_s(config: Settings, t: float) -> float:
    """Cosine decay from ``lambda_s`` to zero (SSiT ``adjust_lambda_ss``).

    Parameters
    ----------
    t
        Fractional progress in ``[0, 1]``.
    """
    return config.lambda_s * 0.5 * (1.0 + math.cos(math.pi * t))


@rank_zero_only
def _log_epoch(
    epoch: int,
    config: Settings,
    avg_cl: float,
    avg_ss: float,
    lr: float,
    moco_m: float,
    t_epoch: float,
    writer: SummaryWriter | None,
    rank: int = 0,
) -> None:
    """Print and log epoch metrics to TensorBoard and MLflow (rank 0 only)."""
    print(
        f"Epoch {epoch + 1}/{config.max_epochs} — "
        f"cl_loss={avg_cl:.4f}  ss_loss={avg_ss:.4f}  "
        f"total={avg_cl + avg_ss:.4f}  lr={lr:.6f}"
    )

    if writer is not None:
        writer.add_scalar("loss/contrastive", avg_cl, epoch)
        writer.add_scalar("loss/saliency", avg_ss, epoch)
        writer.add_scalar("loss/total", avg_cl + avg_ss, epoch)
        writer.add_scalar("lr", lr, epoch)
        writer.add_scalar("momentum_m", moco_m, epoch)
        writer.add_scalar("time/epoch", t_epoch, epoch)

    if config.mlflow:
        import mlflow

        mlflow.log_metrics(
            {
                "loss/contrastive": avg_cl,
                "loss/saliency": avg_ss,
                "loss/total": avg_cl + avg_ss,
                "lr": lr,
                "momentum_m": moco_m,
                "time/epoch": t_epoch,
            },
            step=epoch,
        )


def _save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: AdamW,
    scaler: torch.cuda.amp.GradScaler | None,
) -> None:
    """Persist full training state for potential resume.

    The state dict is taken from the *unwrapped* model so keys never carry
    a ``module.`` prefix under DDP.
    """
    state: dict[str, object] = {
        "epoch": epoch,
        "state_dict": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    torch.save(state, path)


def _save_encoder(path: Path, model: nn.Module) -> None:
    """Persist only the base encoder weights for downstream fine-tuning."""
    torch.save(cast(Pretrainer, unwrap_model(model)).base_encoder.state_dict(), path)


@rank_zero_only
def _finalize_training(
    timer: Timer,
    writer: SummaryWriter | None,
    config: Settings,
    rank: int = 0,
) -> None:
    """Log total wall time (rank 0 only).

    Writer teardown is owned by the CLI — this helper never closes it.
    """
    t_total = timer.total()
    if writer is not None:
        writer.add_scalar("time/total", t_total)
    if config.mlflow:
        import mlflow

        mlflow.log_metric("time/total", t_total)


def _reduce_epoch_metrics(
    cl_sum: float,
    ss_sum: float,
    steps: int,
    world_size: int,
    device: torch.device,
) -> tuple[float, float]:
    """Combine per-rank epoch loss sums and step counts into a global mean.

    Sums *and* counts are all-reduced (rather than averaging per-rank means),
    which stays correct even when ranks process different numbers of batches.
    """
    if world_size <= 1:
        return cl_sum / steps, ss_sum / steps

    cl_t = torch.tensor([cl_sum], device=device)
    ss_t = torch.tensor([ss_sum], device=device)
    cnt_t = torch.tensor([float(steps)], device=device)
    torch.distributed.all_reduce(cl_t, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(ss_t, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(cnt_t, op=torch.distributed.ReduceOp.SUM)
    return cl_t.item() / cnt_t.item(), ss_t.item() / cnt_t.item()


def pretrain(
    model: nn.Module,
    train_dataloader: DataLoader,
    config: Settings,
    *,
    device: torch.device,
    writer: SummaryWriter | None = None,
    resume_path: Path | None = None,
    rank: int = 0,
    world_size: int = 1,
    sampler: DistributedSampler | None = None,
) -> None:
    """Run the full pretraining loop.

    Parameters
    ----------
    model
        ``Pretrainer`` (possibly DDP-wrapped), moved to device by the caller.
    train_dataloader
        Yields ``(x1, x2, m1, m2)`` batches.
    config
        Application settings — drives all schedules and thresholds.
    device
        Target device.  Caller is responsible for moving the model.
    writer
        Optional TensorBoard writer.  ``None`` disables logging.  Owned by
        the caller — this loop never closes it.
    resume_path
        Path to a ``checkpoint.pt`` to resume from.
    rank
        Global rank of this process.
    world_size
        Total number of processes.
    sampler
        Optional :class:`DistributedSampler` — reshuffled each epoch via
        ``set_epoch`` so shards rotate between epochs.
    """
    use_amp = config.precision == "16-mixed" and device.type == "cuda"
    scaler: torch.cuda.amp.GradScaler | None = None
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=0.1,
    )

    raw_model = cast(Pretrainer, unwrap_model(model))

    start_epoch = 0
    if resume_path is not None and resume_path.exists():
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        raw_model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        if scaler is not None and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])

    save_dir = config.checkpoint_dir / config.model_name
    save_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    timer = Timer()
    for epoch in range(start_epoch, config.max_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        epoch_cl_sum = 0.0
        epoch_ss_sum = 0.0
        epoch_steps = 0
        for step, (x1, x2, m1, m2) in enumerate(train_dataloader):
            step_ratio = epoch + step / len(train_dataloader)
            lr = _adjust_lr(optimizer, config, step_ratio)
            t = min(max(step_ratio / config.max_epochs, 0.0), 1.0)
            moco_m = _adjust_momentum(config, t)
            ls = _adjust_lambda_s(config, t) if config.ss_decay else config.lambda_s

            x1, x2 = x1.to(device), x2.to(device)
            m1, m2 = m1.to(device), m2.to(device)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    cl_loss, ss_loss = model(x1, x2, m1, m2, moco_m)
                    loss = config.lambda_c * cl_loss + ls * ss_loss
                optimizer.zero_grad()
                scaler.scale(loss).backward()  # type: ignore[union-attr]
                scaler.unscale_(optimizer)  # type: ignore[union-attr]
                nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_val)
                scaler.step(optimizer)  # type: ignore[union-attr]
                scaler.update()  # type: ignore[union-attr]
            else:
                cl_loss, ss_loss = model(x1, x2, m1, m2, moco_m)
                loss = config.lambda_c * cl_loss + ls * ss_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_val)
                optimizer.step()

            epoch_cl_sum += cl_loss.item()
            epoch_ss_sum += ss_loss.item()
            epoch_steps += 1

        avg_cl, avg_ss = _reduce_epoch_metrics(epoch_cl_sum, epoch_ss_sum, epoch_steps, world_size, device)

        _log_epoch(epoch, config, avg_cl, avg_ss, lr, moco_m, timer.lap(), writer, rank=rank)

        if rank == 0 and (epoch + 1) % config.save_every == 0 and (epoch + 1) < config.max_epochs:
            _save_checkpoint(save_dir / "checkpoint.pt", epoch, raw_model, optimizer, scaler)
            _save_encoder(save_dir / f"epoch_{epoch + 1}_encoder.pt", raw_model)

    if rank == 0:
        _save_checkpoint(save_dir / "checkpoint.pt", config.max_epochs - 1, raw_model, optimizer, scaler)
        _save_encoder(save_dir / f"epoch_{config.max_epochs}_encoder.pt", raw_model)

    _finalize_training(timer, writer, config, rank=rank)
