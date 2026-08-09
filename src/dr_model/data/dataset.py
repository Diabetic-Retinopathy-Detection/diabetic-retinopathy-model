"""Supervised grading dataset over a pre-split ImageFolder tree.

The fine-tuning datasets (APTOS 2019, Messidor-2, DDR) are prepared by
``utils/crop.py`` into a ``<dataset_root>/{train,val,test}/{0..4}/`` layout
where each subdirectory name is the DR grade.  ``GradingDataset`` wraps a
*single* split directory in :class:`torchvision.datasets.ImageFolder`;
labels come from the subdirectory names.  Nothing here parses CSVs, resolves
original labels, or defines splits — that information is consumed during
preprocessing and encoded in the filesystem.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from torch import Tensor
from torch.utils.data import Dataset
from torchvision import datasets


class GradingDataset(Dataset):
    """Thin wrapper around ``ImageFolder`` for one fine-tuning split.

    Parameters
    ----------
    root : Path
        A single split directory, e.g. ``<dataset_root>/train``.  It must
        contain one subdirectory per DR grade (``0``-``4``).
    transform : Callable | None
        Per-sample transform pipeline applied by ``ImageFolder``.
    """

    def __init__(self, root: Path, transform: Callable | None = None) -> None:
        root = Path(root)
        if not root.exists():
            msg = f"Dataset split directory not found: {root}"
            raise FileNotFoundError(msg)
        if not any(p.is_dir() for p in root.iterdir()):
            msg = f"No class subdirectories found in: {root}"
            raise ValueError(msg)
        self.dataset = datasets.ImageFolder(str(root), transform=transform)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        img, label = self.dataset[index]
        return img, label
