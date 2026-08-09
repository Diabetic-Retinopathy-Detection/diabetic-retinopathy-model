"""Determinism setup — seeding always applied; GPU determinism optional."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

_SEED: int | None = None


def setup_cublas_workspace() -> None:
    """Set ``CUBLAS_WORKSPACE_CONFIG`` for deterministic cuBLAS on CUDA >= 10.2.

    Only meaningful when deterministic algorithms are enabled; called
    automatically by :func:`setup_determinism`.  CuBLAS reads the variable
    lazily at the first kernel launch, so no need to set it at import time.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def setup_determinism(seed: int, deterministic_algorithms: bool = False) -> int:
    """Configure RNG seeds and optionally GPU determinism.

    Parameters
    ----------
    seed : int
        Random seed for torch, Python, and numpy RNGs.
    deterministic_algorithms : bool
        If True, also enable ``torch.use_deterministic_algorithms``,
        ``cudnn.deterministic``, and set ``CUBLAS_WORKSPACE_CONFIG``
        (slower but fully reproducible).  Default False — seeds RNGs only
        (cheap, good enough for most runs).

    Returns
    -------
    int
        The seed that was stored, for informational use.
    """
    global _SEED
    _SEED = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic_algorithms:
        setup_cublas_workspace()
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return seed


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker reseeding — pass as ``worker_init_fn=``.

    Seeds torch, Python ``random``, and ``numpy`` RNGs with
    ``_SEED + worker_id``.  Does nothing if ``setup_determinism`` was
    never called (``_SEED`` is ``None``).
    """
    seed = _SEED
    if seed is not None:
        base = seed + worker_id
        random.seed(base)
        np.random.seed(base)
        torch.manual_seed(base)
