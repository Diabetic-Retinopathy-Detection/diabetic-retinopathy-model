"""Distributed training helpers for single-node DDP via ``torchrun``.

All distributed *runtime* concerns live here — detecting a ``torchrun``
launch, initialising/tearing down the process group, rank gating, and
unwrapping DDP-wrapped models.  Training logic elsewhere never inspects
environment variables or calls ``torch.distributed`` directly.
"""

from __future__ import annotations

import functools
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from typing import TypeVar, cast

import torch
import torch.nn as nn

from dr_model.utils.device import resolve_device

_T = TypeVar("_T")
_M = TypeVar("_M", bound=nn.Module)


@dataclass(frozen=True)
class DistributedContext:
    """Runtime facts about the current process.

    Attributes
    ----------
    enabled : bool
        Whether the process is part of a distributed run.
    rank : int
        Global rank across all processes.
    local_rank : int
        Rank within the current node (GPU index when ``enabled``).
    world_size : int
        Total number of processes.
    device : torch.device
        Device this process trains on.
    """

    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device


def detect_distributed_context(cli_device: str) -> DistributedContext:
    """Detect a ``torchrun`` launch and build the per-process context.

    ``torchrun`` sets ``RANK``/``LOCAL_RANK``/``WORLD_SIZE`` in the
    environment.  When present, the process is distributed:

    * CUDA-capable runs (``cli_device`` of ``"auto"`` or ``cuda*``) always
      train on ``cuda:{local_rank}``, ignoring ``--device`` — each rank must
      pin itself to its own GPU.
    * Explicit CPU runs (``cli_device="cpu"``) honour the requested device
      and fall back to the ``gloo`` backend.  This is how the multi-process
      unit tests exercise real DDP without a GPU.

    Otherwise the context is a plain single-process run with ``enabled=False``.
    """
    if "RANK" not in os.environ and "LOCAL_RANK" not in os.environ:
        return DistributedContext(False, 0, 0, 1, torch.device(resolve_device(cli_device)))

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if cli_device in ("auto",) or cli_device.startswith("cuda"):
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device(resolve_device(cli_device))
    else:
        device = torch.device(cli_device)

    return DistributedContext(True, rank, local_rank, world_size, device)


def init_distributed(ctx: DistributedContext) -> None:
    """Initialise the process group — NCCL on CUDA, gloo otherwise.

    No-op for a non-distributed context.

    A bounded ``timeout`` keeps a hung or crashed peer from blocking
    every other rank for the default 30 minutes — it surfaces as an error
    after roughly two minutes instead.
    """
    if not ctx.enabled:
        return
    backend = "nccl" if ctx.device.type == "cuda" else "gloo"
    torch.distributed.init_process_group(backend=backend, timeout=timedelta(seconds=120))


def cleanup_distributed() -> None:
    """Destroy the process group if one exists.

    Idempotent — safe to call in every teardown path, including single-
    process runs where no group was ever created.

    In multi-process runs every rank joins a ``barrier()`` before the
    store is torn down.  Destroying too early — while a peer is still
    finishing collectives (e.g. gloo's object transport) — can leave the
    rank-0 TCPStore waiting for a connection that never drains, which
    hangs teardown (pytorch/pytorch#75097).
    """
    if torch.distributed.is_initialized():
        if torch.distributed.get_world_size() > 1:
            torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def is_rank_zero(ctx: DistributedContext) -> bool:
    """Return True for rank 0 (and for any non-distributed run)."""
    return ctx.rank == 0


def rank_zero_only(func: Callable[..., _T]) -> Callable[..., _T | None]:
    """Skip *func* unless called with ``rank=0`` (the default).

    Intended for logging/finalisation helpers that must run exactly once.
    The caller passes ``rank`` as a keyword argument.
    """

    @functools.wraps(func)
    def wrapper(*args: object, **kwargs: object) -> _T | None:
        if kwargs.get("rank", 0) != 0:
            return None
        return func(*args, **kwargs)

    return wrapper


def unwrap_model(model: _M) -> _M:
    """Return the inner module of a DDP wrapper, or the model itself.

    Single source of truth for every persistence path so checkpoints never
    contain ``module.``-prefixed keys.
    """
    if hasattr(model, "module"):
        return cast(_M, model.module)
    return model


def wrap_model(model: nn.Module, ctx: DistributedContext) -> nn.Module:
    """Wrap *model* for distributed training, or return it unchanged.

    On CUDA, BatchNorm layers are converted to :class:`nn.SyncBatchNorm`
    first (the pretrainer's projection heads use ``BatchNorm1d``, whose
    statistics get noisy at small per-GPU batch sizes), then the model is
    wrapped in :class:`DistributedDataParallel` pinned to the local GPU.
    On CPU (``gloo`` test contexts) the model is wrapped without
    ``device_ids``.  Non-distributed runs are returned untouched.
    """
    if not ctx.enabled:
        return model
    if ctx.device.type == "cuda":
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    return nn.parallel.DistributedDataParallel(
        model,
        device_ids=[ctx.local_rank] if ctx.device.type == "cuda" else None,
    )
