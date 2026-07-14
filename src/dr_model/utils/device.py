"""Resolve a user-facing device string into a :class:`torch.device`."""

from __future__ import annotations

import torch


def resolve_device(device_str: str) -> torch.device:
    """Resolve a device string to a :class:`torch.device`.

    Parameters
    ----------
    device_str : str
        ``"auto"`` prefers CUDA, then MPS, then CPU.  Any other string
        is passed directly (e.g. ``"cpu"``, ``"cuda"``, ``"cuda:0"``,
        ``"mps"``).

    Returns
    -------
    torch.device
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if device_str.startswith("cuda"):
        torch.cuda.empty_cache()
    return torch.device(device_str)
