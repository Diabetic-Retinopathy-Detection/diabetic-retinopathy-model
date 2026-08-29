from __future__ import annotations

import shutil
from collections.abc import Sized
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import RandomSampler, SequentialSampler

from dr_model.config import Settings
from dr_model.data import FinetuneDataModule, GradingDataset
from dr_model.data.constants import DATASET_STATS
from dr_model.data.finetune_datamodule import build_eval_transform, build_train_transform

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

GRADES = (0, 1, 2, 3, 4)
APTOS_MEAN: list[float] = DATASET_STATS["aptos2019"]["mean"]
APTOS_STD: list[float] = DATASET_STATS["aptos2019"]["std"]


def _settings(root: Path | None, batch_size: int = 4) -> Settings:
    return Settings(
        finetune_dataset_root=root,
        finetune_input_size=384,
        finetune_mean=APTOS_MEAN,
        finetune_std=APTOS_STD,
        batch_size=batch_size,
        num_workers=0,
    )


def _make_split_tree(root: Path, images_per_grade: int = 2) -> Path:
    for grade in GRADES:
        grade_dir = root / str(grade)
        grade_dir.mkdir(parents=True, exist_ok=True)
        for i in range(images_per_grade):
            arr = np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8)
            Image.fromarray(arr).save(grade_dir / f"img_{grade}_{i}.jpeg")
    return root


def _make_dataset_root(tmp_path: Path, images_per_grade: int = 2) -> Path:
    root = tmp_path / "dataset"
    for split in ("train", "val", "test"):
        _make_split_tree(root / split, images_per_grade)
    return root


class TestGradingDataset:
    def test_returns_tensor_and_grade(self, tmp_path: Path) -> None:
        root = _make_split_tree(tmp_path / "train")
        ds = GradingDataset(root, transform=build_eval_transform(_settings(root)))

        img, label = ds[0]

        assert img.shape == (3, 384, 384)
        assert img.dtype == torch.float32
        assert isinstance(label, int)
        assert label in GRADES

    def test_label_is_subdirectory_name(self, tmp_path: Path) -> None:
        root = _make_split_tree(tmp_path / "train")
        ds = GradingDataset(root, transform=build_eval_transform(_settings(root)))

        for i in range(len(ds)):
            img_path, label = ds.dataset.samples[i]
            assert label == int(Path(img_path).parent.name)

    def test_labels_cover_all_grades(self, tmp_path: Path) -> None:
        root = _make_split_tree(tmp_path / "train")
        ds = GradingDataset(root, transform=build_eval_transform(_settings(root)))

        labels = {ds[i][1] for i in range(len(ds))}
        assert labels == set(GRADES)

    @pytest.mark.parametrize("name", ["aptos2019", "messidor2"])
    def test_same_contract_across_datasets(self, tmp_path: Path, name: str) -> None:
        root = _make_dataset_root(tmp_path, images_per_grade=1)
        root.rename(tmp_path / name)
        ds = GradingDataset(tmp_path / name / "train", transform=build_eval_transform(_settings(root)))

        assert len(ds) == len(GRADES)
        for i in range(len(ds)):
            img, label = ds[i]
            assert img.shape == (3, 384, 384)
            assert img.dtype == torch.float32
            assert label in GRADES

    def test_len(self, tmp_path: Path) -> None:
        root = _make_split_tree(tmp_path / "train", images_per_grade=3)
        ds = GradingDataset(root, transform=build_eval_transform(_settings(root)))

        assert len(ds) == 3 * len(GRADES)

    def test_missing_root_raises(self, tmp_path: Path) -> None:
        missing = tmp_path / "does-not-exist"
        with pytest.raises(FileNotFoundError, match="split directory not found"):
            GradingDataset(missing)

    def test_empty_root_raises(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ValueError, match="class subdirectories"):
            GradingDataset(empty)


class TestFinetuneDataModule:
    @pytest.mark.parametrize("missing", ["train", "val", "test"])
    def test_missing_split_raises(self, tmp_path: Path, missing: str) -> None:
        root = _make_dataset_root(tmp_path)
        shutil.rmtree(root / missing)

        dm = FinetuneDataModule(_settings(root))
        with pytest.raises(FileNotFoundError, match="Fine-tuning split directory not found"):
            dm.setup()

    def test_root_none_raises(self) -> None:
        dm = FinetuneDataModule(_settings(None))
        with pytest.raises(ValueError, match="finetune_dataset_root must be set"):
            dm.setup()

    def test_setup_builds_three_datasets(self, tmp_path: Path) -> None:
        root = _make_dataset_root(tmp_path, images_per_grade=2)
        dm = FinetuneDataModule(_settings(root))
        dm.setup()

        assert dm.train_dataset is not None
        assert dm.val_dataset is not None
        assert dm.test_dataset is not None
        assert isinstance(dm.train_dataset, Sized)
        assert isinstance(dm.val_dataset, Sized)
        assert isinstance(dm.test_dataset, Sized)
        expected = 2 * len(GRADES)
        assert len(dm.train_dataset) == expected
        assert len(dm.val_dataset) == expected
        assert len(dm.test_dataset) == expected

    def test_dataloader_flags(self, tmp_path: Path) -> None:
        root = _make_dataset_root(tmp_path)
        dm = FinetuneDataModule(_settings(root))
        dm.setup()

        train_loader = dm.train_dataloader()
        assert isinstance(train_loader.sampler, RandomSampler)
        assert train_loader.drop_last is True

        val_loader = dm.val_dataloader()
        assert isinstance(val_loader.sampler, SequentialSampler)
        assert val_loader.drop_last is False

        test_loader = dm.test_dataloader()
        assert isinstance(test_loader.sampler, SequentialSampler)
        assert test_loader.drop_last is False

    def test_dataloaders_iterate_one_batch(self, tmp_path: Path) -> None:
        root = _make_dataset_root(tmp_path, images_per_grade=2)
        dm = FinetuneDataModule(_settings(root, batch_size=4))
        dm.setup()

        loaders: list[DataLoader] = [
            dm.train_dataloader(),
            dm.val_dataloader(),
            dm.test_dataloader(),
        ]
        for loader in loaders:
            images, labels = next(iter(loader))
            assert images.shape == (4, 3, 384, 384)
            assert images.dtype == torch.float32
            assert labels.shape == (4,)
            assert set(labels.tolist()) <= set(GRADES)

    def test_dataloader_before_setup_raises(self, tmp_path: Path) -> None:
        root = _make_dataset_root(tmp_path)
        dm = FinetuneDataModule(_settings(root))
        with pytest.raises(RuntimeError, match="Call setup"):
            dm.train_dataloader()


class TestTransforms:
    def _image(self) -> Image.Image:
        arr = np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8)
        return Image.fromarray(arr)

    def test_both_transforms_output_384(self) -> None:
        config = _settings(None)
        img = self._image()

        assert build_train_transform(config)(img).shape == (3, 384, 384)
        assert build_eval_transform(config)(img).shape == (3, 384, 384)

    def test_train_and_eval_transforms_distinct(self) -> None:
        config = _settings(None)
        train_tf = build_train_transform(config)
        eval_tf = build_eval_transform(config)
        img = self._image()

        torch.manual_seed(0)
        eval_out = eval_tf(img)
        assert any(not torch.equal(train_tf(img), eval_out) for _ in range(5))


class TestExports:
    def test_dataset_stats_keys(self) -> None:
        assert {"ddr", "aptos2019", "messidor2"} <= set(DATASET_STATS)
        assert "eyepacs" in DATASET_STATS

    def test_package_exports(self) -> None:
        assert FinetuneDataModule is not None
        assert GradingDataset is not None
