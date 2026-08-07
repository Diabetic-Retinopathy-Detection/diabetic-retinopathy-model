"""Supervised fine-tuning data module over pre-split ImageFolder trees.

Loads the ``train/``, ``val/``, ``test/`` splits that ``utils/crop.py``
materialised on disk (one subdirectory per DR grade, ``0``-``4``) and
exposes one :class:`~torch.utils.data.DataLoader` per split.  There is no
splitting or label-resolution logic here — the filesystem already encodes
the labels and the splits.

The train pipeline reproduces SSiT's ``eval.py`` ``data_transforms``
exactly (flips, mild ``RandomResizedCrop``, colour jitter, rotation,
affine).  The eval pipeline uses SSiT's plain ``Resize → ToTensor →
Normalize``.  Both run at ``finetune_input_size`` (384), distinct from the
pretraining resolution ``input_size`` (224).
"""

from __future__ import annotations

from torch.utils.data import DataLoader
from torchvision import transforms

from dr_model.config import Settings
from dr_model.data.dataset import GradingDataset
from dr_model.utils.determinism import worker_init_fn

SPLITS = ("train", "val", "test")


def build_train_transform(config: Settings) -> transforms.Compose:  # type: ignore[no-any-unimported]
    """Augmented train pipeline, matching SSiT ``eval.py`` ``data_transforms``."""
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomResizedCrop(
            config.finetune_input_size,
            scale=(0.87, 1.15),
            ratio=(0.7, 1.3),
        ),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
        transforms.RandomRotation(degrees=(-180, 180)),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.Resize((config.finetune_input_size, config.finetune_input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.finetune_mean, std=config.finetune_std),
    ])


def build_eval_transform(config: Settings) -> transforms.Compose:  # type: ignore[no-any-unimported]
    """Deterministic eval pipeline: plain resize to ``finetune_input_size``."""
    return transforms.Compose([
        transforms.Resize((config.finetune_input_size, config.finetune_input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.finetune_mean, std=config.finetune_std),
    ])


class FinetuneDataModule:
    """Plain-Python data module for supervised fine-tuning.

    Parameters
    ----------
    config : Settings
        Application settings.  ``finetune_dataset_root`` must point at a
        dataset root containing ``train/``, ``val/``, and ``test/``.
    """

    def __init__(self, config: Settings) -> None:
        self.config = config
        self.train_dataset: GradingDataset | None = None
        self.val_dataset: GradingDataset | None = None
        self.test_dataset: GradingDataset | None = None

    def setup(self) -> None:
        """Validate the split tree and build the three datasets."""
        root = self.config.finetune_dataset_root
        if root is None:
            msg = "config.finetune_dataset_root must be set for fine-tuning"
            raise ValueError(msg)

        for split in SPLITS:
            split_dir = root / split
            if not split_dir.is_dir():
                msg = f"Fine-tuning split directory not found: {split_dir}"
                raise FileNotFoundError(msg)

        self.train_dataset = GradingDataset(root / "train", transform=build_train_transform(self.config))
        self.val_dataset = GradingDataset(root / "val", transform=build_eval_transform(self.config))
        self.test_dataset = GradingDataset(root / "test", transform=build_eval_transform(self.config))

    def train_dataloader(self) -> DataLoader:
        """Shuffled train loader with ``drop_last=True`` (matches SSiT)."""
        if self.train_dataset is None:
            msg = "Call setup() before train_dataloader()"
            raise RuntimeError(msg)
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            shuffle=True,
            drop_last=True,
            worker_init_fn=worker_init_fn if self.config.seed >= 0 else None,
        )

    def val_dataloader(self) -> DataLoader:
        """Unshuffled val loader."""
        if self.val_dataset is None:
            msg = "Call setup() before val_dataloader()"
            raise RuntimeError(msg)
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            shuffle=False,
            drop_last=False,
            worker_init_fn=worker_init_fn if self.config.seed >= 0 else None,
        )

    def test_dataloader(self) -> DataLoader:
        """Unshuffled test loader."""
        if self.test_dataset is None:
            msg = "Call setup() before test_dataloader()"
            raise RuntimeError(msg)
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            shuffle=False,
            drop_last=False,
            worker_init_fn=worker_init_fn if self.config.seed >= 0 else None,
        )
