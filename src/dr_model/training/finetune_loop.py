"""Supervised fine-tuning loop for DR grading.

Mirrors SSiT's ``eval.py`` fine-tuning procedure: AdamW with a cosine LR
schedule (linear warmup, then cosine decay to ``finetune_min_lr``), plain
cross-entropy loss, and quadratic-weighted Cohen's kappa reported on the
validation split each epoch.  Supports single-node DDP via ``torchrun`` —
each rank trains on a sharded train split, validation metrics are reduced
across ranks, and checkpoints are written by rank 0.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler

from dr_model.config import Settings
from dr_model.data.finetune_datamodule import FinetuneDataModule
from dr_model.training.artifacts import write_finetune_artifacts
from dr_model.training.distributed import rank_zero_only, unwrap_model
from dr_model.training.metrics import ClassificationMetrics, calculate_classification_metrics
from dr_model.utils.timer import Timer

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter


class SquaredCDFLoss(nn.Module):
    """Mean squared difference between predicted and target CDFs."""

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        probabilities = torch.softmax(logits, dim=1)
        targets = F.one_hot(labels, num_classes=logits.shape[1]).to(dtype=logits.dtype)
        predicted_cdf = probabilities.cumsum(dim=1)
        target_cdf = targets.cumsum(dim=1)
        return (predicted_cdf - target_cdf).square().mean()


class SquaredWassersteinLoss(nn.Module):
    """Squared 2-Wasserstein loss for one-hot ordinal class labels."""

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        probabilities = torch.softmax(logits, dim=1)
        class_positions = torch.arange(logits.shape[1], device=logits.device, dtype=logits.dtype)
        distances = (class_positions.unsqueeze(0) - labels.unsqueeze(1)).square()
        return (probabilities * distances).sum(dim=1).mean()


def build_finetune_criterion(config: Settings) -> nn.Module:
    """Build the configured supervised fine-tuning loss."""
    if config.finetune_loss == "squared_wasserstein":
        return SquaredWassersteinLoss()
    if config.finetune_loss == "squared_cdf":
        return SquaredCDFLoss()
    if config.finetune_loss == "cross_entropy":
        return nn.CrossEntropyLoss()
    msg = (
        f"Unsupported finetune_loss: {config.finetune_loss!r}; expected "
        "'squared_wasserstein', 'squared_cdf', or 'cross_entropy'"
    )
    raise ValueError(msg)


def adjust_lr(
    optimizer: torch.optim.Optimizer,
    config: Settings,
    step_ratio: float,
    total_epochs: int | None = None,
) -> float:
    """Linear warmup then cosine decay to ``finetune_min_lr``.

    Matches SSiT's ``adjust_learning_rate`` (``train.py``) when
    ``finetune_min_lr`` is zero.
    """
    schedule_epochs = total_epochs if total_epochs is not None else config.finetune_epochs
    if step_ratio < config.finetune_warmup_epochs:
        lr = config.finetune_lr * step_ratio / config.finetune_warmup_epochs
    else:
        progress = (step_ratio - config.finetune_warmup_epochs) / (schedule_epochs - config.finetune_warmup_epochs)
        decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = config.finetune_min_lr + (config.finetune_lr - config.finetune_min_lr) * decay
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    world_size: int = 1,
) -> ClassificationMetrics:
    """Evaluate a classification split and calculate all classification metrics.

    Loss sums and step counts are all-reduced across ranks so the mean is
    computed over the whole split; predictions, labels, and probabilities are
    gathered so metrics are also computed on the full split (not per-rank).
    """
    model.eval()
    loss_sum = 0.0
    steps = 0
    all_labels: list[int] = []
    all_preds: list[int] = []
    all_probabilities: list[list[float]] = []
    for images, labels in dataloader:
        labels_cpu = labels.tolist()
        logits = model(images.to(device))
        loss_sum += criterion(logits, labels.to(device)).item()
        all_labels.extend(labels_cpu)
        all_preds.extend(logits.argmax(dim=1).tolist())
        all_probabilities.extend(torch.softmax(logits, dim=1).cpu().tolist())
        steps += 1
    model.train()

    if world_size > 1:
        loss_t = torch.tensor([loss_sum], device=device)
        cnt_t = torch.tensor([float(steps)], device=device)
        torch.distributed.all_reduce(loss_t, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(cnt_t, op=torch.distributed.ReduceOp.SUM)
        loss_sum = loss_t.item()
        steps = int(cnt_t.item())

        labels_by_rank: list[list[int]] = [[] for _ in range(world_size)]
        preds_by_rank: list[list[int]] = [[] for _ in range(world_size)]
        probabilities_by_rank: list[list[list[float]]] = [[] for _ in range(world_size)]
        torch.distributed.all_gather_object(labels_by_rank, all_labels)
        torch.distributed.all_gather_object(preds_by_rank, all_preds)
        torch.distributed.all_gather_object(probabilities_by_rank, all_probabilities)
        all_labels = [label for labels in labels_by_rank for label in labels]
        all_preds = [pred for preds in preds_by_rank for pred in preds]
        all_probabilities = [probability for probabilities in probabilities_by_rank for probability in probabilities]

    num_classes = len(all_probabilities[0]) if all_probabilities else 0
    return calculate_classification_metrics(
        all_labels,
        all_preds,
        all_probabilities,
        loss=loss_sum / max(steps, 1),
        num_classes=num_classes,
    )


@rank_zero_only
def _log_epoch(
    epoch: int,
    config: Settings,
    train_loss: float,
    val_metrics: ClassificationMetrics | None,
    lr: float,
    t_epoch: float,
    writer: SummaryWriter | None,
    rank: int = 0,
) -> None:
    """Print and log epoch metrics to TensorBoard and MLflow (rank 0 only)."""
    message = f"Epoch {epoch + 1}/{config.finetune_epochs} — train_loss={train_loss:.4f}"
    if val_metrics is not None:
        message += f"  val_loss={val_metrics.loss:.4f}  kappa={val_metrics.kappa:.4f}"
    print(f"{message}  lr={lr:.6f}")

    if writer is not None:
        writer.add_scalar("loss/train", train_loss, epoch)
        if val_metrics is not None:
            writer.add_scalar("loss/val", val_metrics.loss, epoch)
            writer.add_scalar("kappa/val", val_metrics.kappa, epoch)
            _log_metrics(writer.add_scalar, val_metrics, "val", epoch)
        writer.add_scalar("lr", lr, epoch)
        writer.add_scalar("time/epoch", t_epoch, epoch)

    if config.mlflow:
        import mlflow

        metrics = {"loss/train": train_loss, "lr": lr}
        if val_metrics is not None:
            metrics.update(_metric_scalars(val_metrics, "val"))
        mlflow.log_metrics(metrics, step=epoch)


def _metric_scalars(metrics: ClassificationMetrics, split: str) -> dict[str, float]:
    """Flatten aggregate and per-class metrics into logging scalar names."""
    values: dict[str, float] = {
        f"loss/{split}": metrics.loss,
        f"kappa/{split}": metrics.kappa,
        f"accuracy/{split}": metrics.accuracy,
        f"f1/macro/{split}": metrics.f1_macro,
        f"f1/weighted/{split}": metrics.f1_weighted,
    }
    if metrics.auc_macro is not None:
        values[f"auc/macro/{split}"] = metrics.auc_macro
    if metrics.auc_weighted is not None:
        values[f"auc/weighted/{split}"] = metrics.auc_weighted
    for index, value in enumerate(metrics.f1_per_class):
        values[f"f1/class_{index}/{split}"] = value
    for index, auc_value in enumerate(metrics.auc_per_class):
        if auc_value is not None:
            values[f"auc/class_{index}/{split}"] = auc_value
    for index, value in enumerate(metrics.recall_per_class):
        values[f"recall/class_{index}/{split}"] = value
    return values


def _log_metrics(
    add_scalar: Callable[[str, float, int], None], metrics: ClassificationMetrics, split: str, epoch: int
) -> None:
    """Log metric scalars through a TensorBoard-compatible callback."""
    for tag, value in _metric_scalars(metrics, split).items():
        if tag.startswith("loss/") or tag.startswith("kappa/"):
            continue
        add_scalar(tag, value, epoch)


@rank_zero_only
def _log_split_metrics(
    metrics: ClassificationMetrics,
    split: str,
    step: int,
    writer: SummaryWriter | None,
) -> None:
    """Log metrics for a completed validation or test split."""
    values = _metric_scalars(metrics, split)
    if writer is not None:
        for tag, value in values.items():
            writer.add_scalar(tag, value, step)
    try:
        import mlflow

        if mlflow.active_run() is not None:
            mlflow.log_metrics(values, step=step)
    except ImportError:
        pass


def _save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: AdamW,
    scaler: torch.cuda.amp.GradScaler | None,
    *,
    total_epochs: int | None = None,
    kappa: float | None = None,
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
    if total_epochs is not None:
        state["total_epochs"] = total_epochs
    if kappa is not None:
        state["kappa"] = kappa
    torch.save(state, path)


def _load_model_checkpoint(path: Path, model: nn.Module, device: torch.device) -> None:
    """Load model weights from a training checkpoint for held-out evaluation."""
    state = torch.load(path, map_location=device)
    unwrap_model(model).load_state_dict(state["state_dict"])


def _evaluate_test(
    config: Settings,
    model: nn.Module,
    datamodule: FinetuneDataModule,
    criterion: nn.Module,
    device: torch.device,
    world_size: int,
    save_dir: Path,
    stop_epoch: int,
    writer: SummaryWriter | None,
) -> None:
    """Evaluate the selected final checkpoint on the held-out test split."""
    checkpoint_name = f"epoch_{stop_epoch}.pt" if config.skip_validation else "best_validation_weights.pt"
    checkpoint = save_dir / checkpoint_name
    if not checkpoint.exists():
        return
    _load_model_checkpoint(checkpoint, model, device)
    metrics = _evaluate(model, datamodule.test_dataloader(), criterion, device, world_size)
    _log_split_metrics(metrics, "test", stop_epoch, writer)
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        write_finetune_artifacts(
            config.artifact_dir,
            config,
            [],
            None,
            metrics,
            run_name=f"{config.model_name}_{config.finetune_loss}",
        )


def _restore_checkpoint(
    path: str | Path | None,
    model: nn.Module,
    optimizer: AdamW,
    scaler: torch.cuda.amp.GradScaler | None,
    device: torch.device,
    default_total_epochs: int,
) -> tuple[int, int, float]:
    if path is None:
        return 0, default_total_epochs, -1.0
    state = torch.load(path, map_location=device)
    unwrap_model(model).load_state_dict(state["state_dict"])
    optimizer.load_state_dict(state["optimizer"])
    if scaler is not None and "scaler" in state:
        scaler.load_state_dict(state["scaler"])
    start_epoch = int(state["epoch"]) + 1
    total_epochs = int(state.get("total_epochs", default_total_epochs))
    best_kappa = float(state.get("kappa", -1.0))
    return start_epoch, total_epochs, best_kappa


def run(
    config: Settings,
    model: nn.Module,
    datamodule: FinetuneDataModule,
    *,
    device: torch.device,
    writer: SummaryWriter | None = None,
    rank: int = 0,
    world_size: int = 1,
    resume_path: str | Path | None = None,
    extra_epochs: int = 0,
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
        Under DDP its train loader shards the dataset per rank and exposes
        the sampler as ``datamodule.sampler``.
    device
        Target device.  Caller is responsible for moving the model.
    writer
        Optional TensorBoard writer.  ``None`` disables logging.  Owned by
        the caller — this loop never closes it.
    rank
        Global rank of this process.
    world_size
        Total number of processes.
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
    criterion = build_finetune_criterion(config)

    start_epoch = 0
    start_epoch, total_epochs, best_kappa = _restore_checkpoint(
        resume_path, model, optimizer, scaler, device, config.finetune_epochs
    )

    stop_epoch = start_epoch + extra_epochs if resume_path is not None and extra_epochs else config.finetune_epochs

    train_dataloader = datamodule.train_dataloader()
    val_dataloader = None if config.skip_validation else datamodule.val_dataloader()
    sampler: DistributedSampler | None = datamodule.sampler

    save_dir = config.checkpoint_dir / "finetune"
    save_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    timer = Timer()
    for epoch in range(start_epoch, stop_epoch):
        if sampler is not None:
            sampler.set_epoch(epoch)

        epoch_loss_sum = 0.0
        epoch_steps = 0
        for images, labels in train_dataloader:
            step_ratio = epoch + epoch_steps / len(train_dataloader)
            lr = adjust_lr(optimizer, config, step_ratio, total_epochs=total_epochs)

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
        if val_dataloader is None:
            kappa = None
        else:
            val_metrics = _evaluate(model, val_dataloader, criterion, device, world_size)
            kappa = val_metrics.kappa

        _log_epoch(
            epoch,
            config,
            avg_train_loss,
            val_metrics if val_dataloader is not None else None,
            lr,
            timer.lap(),
            writer,
            rank=rank,
        )

        if rank == 0 and kappa is not None and kappa > best_kappa:
            best_kappa = kappa
            _save_checkpoint(
                save_dir / "best_validation_weights.pt",
                epoch,
                model,
                optimizer,
                scaler,
                total_epochs=total_epochs,
                kappa=kappa,
            )
        if rank == 0 and (epoch + 1) % config.save_every == 0 and (epoch + 1) < stop_epoch:
            _save_checkpoint(
                save_dir / f"epoch_{epoch + 1}.pt",
                epoch,
                model,
                optimizer,
                scaler,
                total_epochs=total_epochs,
                kappa=best_kappa if not config.skip_validation else None,
            )

    if rank == 0:
        _save_checkpoint(
            save_dir / f"epoch_{stop_epoch}.pt",
            stop_epoch - 1,
            model,
            optimizer,
            scaler,
            total_epochs=total_epochs,
            kappa=best_kappa if not config.skip_validation else None,
        )

    _evaluate_test(config, model, datamodule, criterion, device, world_size, save_dir, stop_epoch, writer)
