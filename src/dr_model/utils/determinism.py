"""Determinism and cuBLAS workspace setup for reproducible training."""

from __future__ import annotations

import os

import torch


def setup_cublas_workspace() -> None:
    """Set ``CUBLAS_WORKSPACE_CONFIG`` for deterministic cuBLAS on CUDA >= 10.2.

    Must be called before any CUDA work starts.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def setup_determinism(seed: int = 42) -> None:
    """Configure PyTorch for deterministic behaviour as far as possible.

    Parameters
    ----------
    seed : int
        Random seed for ``torch.manual_seed`` and CUDA seeds.
    """
    setup_cublas_workspace()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
