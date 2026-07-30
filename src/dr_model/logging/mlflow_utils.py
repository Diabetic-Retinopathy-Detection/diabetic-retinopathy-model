"""MLflow experiment tracking utilities.

Provides ``init_mlflow_run`` and ``end_mlflow_run`` — thin wrappers
around the MLflow API that log all ``Settings`` fields as params, plus
the git commit hash and CLI command as tags.
"""

from __future__ import annotations

import contextlib
import platform
import subprocess
import sys
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from dr_model.config import Settings

import mlflow


def init_mlflow_run(
    config: Settings,
    experiment_name: str,
    run_name_prefix: str,
    device: torch.device | None = None,
) -> mlflow.Run:  # type: ignore[name-defined]
    """Initialise and start an MLflow run.

    Parameters
    ----------
    config
        Application settings — all scalar fields are logged as params.
    experiment_name
        MLflow experiment name (created if it doesn't exist).
    run_name_prefix
        Human-readable prefix; a timestamp is appended automatically.
    device
        Optional device used for training.  When provided the GPU name
        is logged as a tag (if CUDA).

    Returns
    -------
    mlflow.Run
        The active run.
    """
    if config.mlflow_tracking_uri:
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)

    mlflow.set_experiment(experiment_name)
    run_name = f"{run_name_prefix}_{datetime.now():%Y%m%d_%H%M%S}"
    mlflow.start_run(run_name=run_name)

    for key, value in vars(config).items():
        if isinstance(value, (str, int, float, bool)):
            mlflow.log_param(key, value)

    with contextlib.suppress(Exception):
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            mlflow.set_tag("git_commit", result.stdout.strip())

    mlflow.set_tag("command", " ".join(sys.argv))
    _log_system_tags(device)

    return mlflow.active_run()


def _log_system_tags(device: torch.device | None = None) -> None:
    """Log environment metadata as MLflow tags."""
    import torch

    mlflow.set_tag("torch_version", torch.__version__)
    mlflow.set_tag("cuda_version", torch.version.cuda or "none")
    mlflow.set_tag("platform", platform.platform())
    mlflow.set_tag("python_version", platform.python_version())

    if device is not None and device.type == "cuda":
        with contextlib.suppress(Exception):
            mlflow.set_tag("gpu_name", torch.cuda.get_device_name(device))

    with contextlib.suppress(ImportError):
        import psutil

        mlflow.set_tag("ram_gb", round(psutil.virtual_memory().total / 1e9, 1))


def end_mlflow_run() -> None:
    """End the active MLflow run."""
    mlflow.end_run()
