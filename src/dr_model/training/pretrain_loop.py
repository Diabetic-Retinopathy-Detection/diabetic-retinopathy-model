"""Pretraining loop for saliency-guided MoCo v3 contrastive learning.

Follows the schedule and optimiser design from SSiT
(``checkpoints/referencematerial/SSiT/train.py``) — cosine LR warmup,
cosine momentum ramp, optional cosine saliency-weight decay — without
reimplementing distributed training (single-process for now).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dr_model.config import Settings
from dr_model.model.pretrain import Pretrainer

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


def _save_checkpoint(
    path: Path,
    epoch: int,
    model: Pretrainer,
    optimizer: AdamW,
    scaler: torch.amp.GradScaler | None,
) -> None:
    """Persist full training state for potential resume."""
    state: dict[str, object] = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    torch.save(state, path)


def _save_encoder(path: Path, model: Pretrainer) -> None:
    """Persist only the base encoder weights for downstream fine-tuning."""
    torch.save(model.base_encoder.state_dict(), path)


def pretrain(
    model: Pretrainer,
    train_dataloader: DataLoader,
    config: Settings,
    *,
    device: torch.device,
    writer: SummaryWriter | None = None,
    resume_path: Path | None = None,
) -> None:
    """Run the full pretraining loop.

    Parameters
    ----------
    model
        ``Pretrainer`` instance (moved to device by the caller).
    train_dataloader
        Yields ``(x1, x2, m1, m2)`` batches.
    config
        Application settings — drives all schedules and thresholds.
    device
        Target device.  Caller is responsible for moving the model.
    writer
        Optional TensorBoard writer.  ``None`` disables logging.
    resume_path
        Path to a ``checkpoint.pt`` to resume from.
    """
    use_amp = config.precision == "16-mixed" and device.type == "cuda"
    scaler: torch.amp.GradScaler | None = None
    if use_amp:
        scaler = torch.amp.GradScaler("cuda")

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=0.1,
    )

    start_epoch = 0
    if resume_path is not None and resume_path.exists():
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        if scaler is not None and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])

    save_dir = config.checkpoint_dir / config.model_name
    save_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    for epoch in range(start_epoch, config.max_epochs):
        epoch_cl_loss = 0.0
        epoch_ss_loss = 0.0
        for step, (x1, x2, m1, m2) in enumerate(train_dataloader):
            step_ratio = epoch + step / len(train_dataloader)
            lr = _adjust_lr(optimizer, config, step_ratio)
            t = min(max(step_ratio / config.max_epochs, 0.0), 1.0)
            moco_m = _adjust_momentum(config, t)
            ls = _adjust_lambda_s(config, t) if config.ss_decay else config.lambda_s

            x1, x2 = x1.to(x1.device), x2.to(x1.device)
            m1, m2 = m1.to(x1.device), m2.to(x1.device)

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

            epoch_cl_loss += cl_loss.item()
            epoch_ss_loss += ss_loss.item()

        steps = len(train_dataloader)
        avg_cl = epoch_cl_loss / steps
        avg_ss = epoch_ss_loss / steps

        if writer is not None:
            writer.add_scalar("loss/contrastive", avg_cl, epoch)
            writer.add_scalar("loss/saliency", avg_ss, epoch)
            writer.add_scalar("loss/total", avg_cl + avg_ss, epoch)
            writer.add_scalar("lr", lr, epoch)
            writer.add_scalar("momentum_m", moco_m, epoch)

        if (epoch + 1) % config.save_every == 0 and (epoch + 1) < config.max_epochs:
            _save_checkpoint(save_dir / "checkpoint.pt", epoch, model, optimizer, scaler)
            _save_encoder(save_dir / f"epoch_{epoch + 1}_encoder.pt", model)

    _save_checkpoint(save_dir / "checkpoint.pt", config.max_epochs - 1, model, optimizer, scaler)
    _save_encoder(save_dir / f"epoch_{config.max_epochs}_encoder.pt", model)

    if writer is not None:
        writer.close()
