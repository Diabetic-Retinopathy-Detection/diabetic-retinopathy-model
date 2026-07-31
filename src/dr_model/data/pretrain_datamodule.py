"""Pretraining data module.

Loads the pickle index produced by ``preprocess-retina-datasets``,
resolves relative paths, optionally subsamples, and exposes a single
``train_dataloader()``.
"""

from __future__ import annotations

import pickle
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, DistributedSampler

from dr_model.config import Settings
from dr_model.data.constants import EYEPACS_MEAN, EYEPACS_STD
from dr_model.data.pair_dataset import DATA_AUG, PairDataset, TransformWithMask
from dr_model.utils.determinism import worker_init_fn


class PretrainDataModule:
    """Plain-Python data module for saliency-guided pretraining.

    Parameters
    ----------
    config : Settings
        Application settings.  ``data_index_path`` must be set.
    """

    def __init__(self, config: Settings) -> None:
        self.config = config
        self.dataset: PairDataset | None = None
        self.sampler: DistributedSampler | None = None

    def setup(self) -> None:
        """Load the pickle index and build the dataset."""
        if self.config.data_index_path is None:
            msg = "config.data_index_path must be set for pretraining"
            raise ValueError(msg)

        pkl_path = Path(self.config.data_index_path)
        if not pkl_path.exists():
            msg = f"Pickle index not found: {pkl_path}"
            raise FileNotFoundError(msg)

        with pkl_path.open("rb") as f:
            index = pickle.load(f)  # noqa: S301

        data_dir = Path(self.config.data_dir)
        raw_pairs: list[tuple[Path, Path]] = index["pairs"]
        pairs = [(data_dir / img, data_dir / sal) for img, sal in raw_pairs]

        if self.config.dataset_ratio < 1.0:
            rng = random.Random(self.config.seed)  # noqa: S311
            pairs = list(pairs)
            rng.shuffle(pairs)
            pairs = pairs[: int(len(pairs) * self.config.dataset_ratio)]

        transform = TransformWithMask(
            input_size=self.config.input_size,
            mean=EYEPACS_MEAN,
            std=EYEPACS_STD,
            data_aug=DATA_AUG,
        )
        self.dataset = PairDataset(pairs, transform=transform)

    def train_dataloader(self) -> DataLoader:
        """Return the pretraining train dataloader.

        When ``torch.distributed`` is initialised, the dataset is sharded
        with a :class:`DistributedSampler` (stored as ``self.sampler``) so
        each rank trains on a disjoint slice.  The training loop must call
        ``sampler.set_epoch(epoch)`` each epoch to reshuffle.
        """
        if self.dataset is None:
            msg = "Call setup() before train_dataloader()"
            raise RuntimeError(msg)

        sampler: DistributedSampler | None = None
        shuffle = True
        if torch.distributed.is_initialized():
            sampler = DistributedSampler(self.dataset, shuffle=True, drop_last=True)
            self.sampler = sampler
            shuffle = False

        return DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=True,
            shuffle=shuffle,
            drop_last=True,
            sampler=sampler,
            worker_init_fn=worker_init_fn if self.config.seed >= 0 else None,
        )
