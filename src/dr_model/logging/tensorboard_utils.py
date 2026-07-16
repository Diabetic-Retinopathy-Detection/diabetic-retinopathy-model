"""TensorBoard logging utilities.

Provides ``init_tensorboard_logger`` — creates a ``SummaryWriter``
in the configured log directory.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from torch.utils.tensorboard import SummaryWriter

if TYPE_CHECKING:
    from dr_model.config import Settings


def init_tensorboard_logger(config: Settings, run_name: str | None = None) -> SummaryWriter:
    """Create a TensorBoard ``SummaryWriter``.

    Parameters
    ----------
    config
        Application settings — ``log_dir`` is used as the parent directory.
    run_name
        Subdirectory name.  Defaults to ``pretrain_YYYYMMDD_HHMMSS``.

    Returns
    -------
    SummaryWriter
        Writer ready for ``add_scalar`` calls.
    """
    if run_name is None:
        run_name = f"pretrain_{datetime.now():%Y%m%d_%H%M%S}"
    log_dir = config.log_dir / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(log_dir))
