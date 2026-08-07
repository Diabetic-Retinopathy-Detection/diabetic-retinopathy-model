"""Supervised fine-tuning loop for DR grading.

Mirrors SSiT's ``eval.py`` fine-tuning procedure: AdamW with a cosine LR
schedule (linear warmup, then cosine decay to ``finetune_min_lr``), plain
cross-entropy loss, and quadratic-weighted Cohen's kappa reported on the
validation split each epoch.  Single-process only.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dr_model.config import Settings
from dr_model.data.finetune_datamodule import FinetuneDataModule
from dr_model.utils.timer import Timer

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter


def adjust_lr(optimizer: torch.optim.Optimizer, config: Settings, step_ratio: float) -> float:
    """Linear warmup then cosine decay to ``finetune_min_lr``.

    Matches SSiT's ``adjust_learning_rate`` (``train.py``) when
    ``finetune_min_lr`` is zero.
    """
    if step_ratio < config.finetune_warmup_epochs:
        lr = config.finetune_lr * step_ratio / config.finetune_warmup_epochs
    else:
        progress = (step_ratio - config.finetune_warmup_epochs) / (
            config.finetune_epochs - config.finetune_warmup_epochs
        )
        decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = config.finetune_min_lr + (config.finetune_lr - config.finetune_min_lr) * decay
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    val_dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Mean validation loss and quadratic-weighted Cohen's kappa."""
    model.eval()
    val_loss_sum = 0.0
    steps = 0
    all_labels: list[int] = []
    all_preds: list[int] = []
    for images, labels in val_dataloader:
        labels_cpu = labels.tolist()
        logits = model(images.to(device))
        val_loss_sum += criterion(logits, labels.to(device)).item()
        all_labels.extend(labels_cpu)
        all_preds.extend(logits.argmax(dim=1).tolist())
        steps += 1
    model.train()
    kappa = float(cohen_kappa_score(all_labels, all_preds, weights="quadratic"))
    return val_loss_sum / max(steps, 1), kappa


def _log_epoch(
    epoch: int,
    config: Settings,
    train_loss: float,
    val_loss: float,
    kappa: float,
    lr: float,
    t_epoch: float,
    writer: SummaryWriter | None,
) -> None:
    """Print and log epoch metrics to TensorBoard and MLflow."""
    print(
        f"Epoch {epoch + 1}/{config.finetune_epochs} — "
        f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
        f"kappa={kappa:.4f}  lr={lr:.6f}"
    )

    if writer is not None:
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("kappa/val", kappa, epoch)
        writer.add_scalar("lr", lr, epoch)
        writer.add_scalar("time/epoch", t_epoch, epoch)

    if config.mlflow:
        import mlflow

        mlflow.log_metrics(
            {
                "loss/train": train_loss,
                "loss/val": val_loss,
                "kappa/val": kappa,
                "lr": lr,
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
    """Persist full training state for potential resume."""
    state: dict[str, object] = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    torch.save(state, path)


def run(
    config: Settings,
    model: nn.Module,
    datamodule: FinetuneDataModule,
    *,
    device: torch.device,
    writer: SummaryWriter | None = None,
) -> None:
    """Run the full supervised fine-tuning loop.

    Parameters
    ----------
    config
        Application settings — drives the schedule and checkpoints.
    model
        :class:`~dr_model.model.finetune.Finetuner`, moved to device by the
        caller.  All parameters are trained (full fine-tuning, matching SSiT).
    datamodule
        Initialised (``setup()`` already called) fine-tuning data module.
    device
        Target device.  Caller is responsible for moving the model.
    writer
        Optional TensorBoard writer.  ``None`` disables logging.  Owned by
        the caller — this loop never closes it.
    """
    use_amp = config.precision == "16-mixed" and device.type == "cuda"
    scaler: torch.cuda.amp.GradScaler | None = None
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()

    optimizer = AdamW(
        model.parameters(),
        lr=config.finetune_lr,
        weight_decay=config.finetune_weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()

    save_dir = config.checkpoint_dir / "finetune"
    save_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    timer = Timer()
    best_kappa = -1.0
    for epoch in range(config.finetune_epochs):
        epoch_loss_sum = 0.0
        epoch_steps = 0
        for images, labels in train_dataloader:
            step_ratio = epoch + epoch_steps / len(train_dataloader)
            lr = adjust_lr(optimizer, config, step_ratio)

            images = images.to(device)
            labels = labels.to(device)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits = model(images)
                    loss = criterion(logits, labels)
                optimizer.zero_grad()
                scaler.scale(loss).backward()  # type: ignore[union-attr]
                scaler.unscale_(optimizer)  # type: ignore[union-attr]
                nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_val)
                scaler.step(optimizer)  # type: ignore[union-attr]
                scaler.update()  # type: ignore[union-attr]
            else:
                logits = model(images)
                loss = criterion(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_val)
                optimizer.step()

            epoch_loss_sum += loss.item()
            epoch_steps += 1

        avg_train_loss = epoch_loss_sum / max(epoch_steps, 1)
        avg_val_loss, kappa = _evaluate(model, val_dataloader, criterion, device)

        _log_epoch(epoch, config, avg_train_loss, avg_val_loss, kappa, lr, timer.lap(), writer)

        if kappa > best_kappa:
            best_kappa = kappa
            _save_checkpoint(save_dir / "best_validation_weights.pt", epoch, model, optimizer, scaler)
        if (epoch + 1) % config.save_every == 0 and (epoch + 1) < config.finetune_epochs:
            _save_checkpoint(save_dir / f"epoch_{epoch + 1}.pt", epoch, model, optimizer, scaler)

    _save_checkpoint(
        save_dir / f"epoch_{config.finetune_epochs}.pt",
        config.finetune_epochs - 1,
        model,
        optimizer,
        scaler,
    )
