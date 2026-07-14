"""Pretraining dataset: loads image-saliency pairs from the packaged pickle index.

The pickle produced by ``build-dataset-index`` stores relative paths and a
root directory.  This module resolves those paths against a configurable
``data_dir`` so the same pickle works on any machine.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


class PairDataset(Dataset):
    """Image-saliency pair dataset for saliency-guided pretraining.

    Parameters
    ----------
    pickle_path : Path
        Path to the ``.pkl`` index produced by ``build-dataset-index``.
    data_dir : Path
        Root directory to resolve relative paths against.  Typically
        ``Settings.data_dir``.
    """

    def __init__(self, pickle_path: Path, data_dir: Path) -> None:
        with pickle_path.open("rb") as f:
            index = pickle.load(f)  # noqa: S301

        self.pairs: list[tuple[Path, Path]] = index["pairs"]
        self.data_dir = data_dir

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Load an image-saliency pair.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``image``  — shape ``(3, H, W)``, float32 in [0, 1].
            ``saliency`` — shape ``(1, H, W)``, float32 in [0, 1].
        """
        img_rel, sal_rel = self.pairs[idx]

        img_path = self.data_dir / img_rel
        sal_path = self.data_dir / sal_rel

        image = Image.open(img_path).convert("RGB")
        image = torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1) / 255.0

        saliency = torch.tensor(np.load(sal_path), dtype=torch.float32)
        if saliency.ndim == 2:
            saliency = saliency.unsqueeze(0)

        return image, saliency
