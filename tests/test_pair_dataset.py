from __future__ import annotations

import pickle
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from dr_model.config import Settings
from dr_model.data import EYEPACS_MEAN, EYEPACS_STD, PairDataset, PretrainDataModule, TransformWithMask
from dr_model.data.pair_dataset import DATA_AUG

INPUT_SIZE = 224


def _make_image(path: Path, size: tuple[int, int] = (INPUT_SIZE, INPUT_SIZE)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)


def _make_saliency(path: Path, size: tuple[int, int] = (INPUT_SIZE, INPUT_SIZE)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.random.rand(*size).astype(np.float32)
    np.save(str(path), arr)


def _make_index(tmp_path: Path, n: int) -> Path:
    data_dir = tmp_path / "data"
    pairs = []
    for i in range(n):
        name = f"img_{i}"
        _make_image(data_dir / "cropped" / f"{name}.jpeg")
        _make_saliency(data_dir / "saliency" / f"{name}.npy")
        pairs.append((Path(f"cropped/{name}.jpeg"), Path(f"saliency/{name}.npy")))

    index_path = tmp_path / "index.pkl"
    with index_path.open("wb") as f:
        pickle.dump({"pairs": pairs}, f)
    return index_path


def _make_transform() -> TransformWithMask:
    return TransformWithMask(INPUT_SIZE, EYEPACS_MEAN, EYEPACS_STD, DATA_AUG)


class TestPairDataset:
    def test_returns_four_tensors(self, tmp_path: Path) -> None:
        index_path = _make_index(tmp_path, 1)
        transform = _make_transform()

        with index_path.open("rb") as f:
            index = pickle.load(f)  # noqa: S301
        data_dir = tmp_path / "data"
        pairs = [(data_dir / img, data_dir / sal) for img, sal in index["pairs"]]

        ds = PairDataset(pairs, transform=transform)
        img_stu, img_tea, mask_stu, mask_tea = ds[0]

        assert img_stu.shape == (3, INPUT_SIZE, INPUT_SIZE)
        assert img_tea.shape == (3, INPUT_SIZE, INPUT_SIZE)
        assert mask_stu.shape == (1, INPUT_SIZE, INPUT_SIZE)
        assert mask_tea.shape == (1, INPUT_SIZE, INPUT_SIZE)
        assert img_stu.dtype == torch.float32
        assert mask_stu.dtype == torch.float32

    def test_images_normalised(self, tmp_path: Path) -> None:
        index_path = _make_index(tmp_path, 1)
        transform = _make_transform()

        with index_path.open("rb") as f:
            index = pickle.load(f)  # noqa: S301
        data_dir = tmp_path / "data"
        pairs = [(data_dir / img, data_dir / sal) for img, sal in index["pairs"]]

        ds = PairDataset(pairs, transform=transform)
        img_stu, img_tea, _, _ = ds[0]

        assert img_stu.mean().item() != 0.5
        assert img_tea.mean().item() != 0.5

    def test_masks_unnormalised(self, tmp_path: Path) -> None:
        index_path = _make_index(tmp_path, 1)
        transform = _make_transform()

        with index_path.open("rb") as f:
            index = pickle.load(f)  # noqa: S301
        data_dir = tmp_path / "data"
        pairs = [(data_dir / img, data_dir / sal) for img, sal in index["pairs"]]

        ds = PairDataset(pairs, transform=transform)
        _, _, mask_stu, mask_tea = ds[0]

        assert mask_stu.min() >= 0.0
        assert mask_stu.max() <= 1.0
        assert mask_tea.min() >= 0.0
        assert mask_tea.max() <= 1.0

    def test_student_crop_smaller_than_teacher(self, tmp_path: Path) -> None:
        index_path = _make_index(tmp_path, 1)
        transform = _make_transform()

        with index_path.open("rb") as f:
            index = pickle.load(f)  # noqa: S301
        data_dir = tmp_path / "data"
        pairs = [(data_dir / img, data_dir / sal) for img, sal in index["pairs"]]

        ds = PairDataset(pairs, transform=transform)

        stu_areas = []
        tea_areas = []
        for _ in range(100):
            img_stu, img_tea, _, _ = ds[0]
            stu_areas.append(img_stu.mean().item())
            tea_areas.append(img_tea.mean().item())

        assert np.mean(stu_areas) != np.mean(tea_areas)

    def test_missing_image_raises(self, tmp_path: Path) -> None:
        transform = _make_transform()
        bad_path = tmp_path / "nonexistent.jpeg"
        sal_path = tmp_path / "saliency.npy"
        np.save(str(sal_path), np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=np.float32))

        ds = PairDataset([(bad_path, sal_path)], transform=transform)
        with pytest.raises(FileNotFoundError, match="Image not found"):
            ds[0]

    def test_missing_saliency_raises(self, tmp_path: Path) -> None:
        transform = _make_transform()
        img_path = tmp_path / "img.jpeg"
        Image.fromarray(np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)).save(img_path)
        bad_path = tmp_path / "nonexistent.npy"

        ds = PairDataset([(img_path, bad_path)], transform=transform)
        with pytest.raises(FileNotFoundError, match="Saliency map not found"):
            ds[0]

    def test_transform_none_raises(self) -> None:
        with pytest.raises(ValueError, match="requires a TransformWithMask"):
            PairDataset([], transform=None)

    def test_len(self, tmp_path: Path) -> None:
        index_path = _make_index(tmp_path, 5)
        transform = _make_transform()

        with index_path.open("rb") as f:
            index = pickle.load(f)  # noqa: S301
        data_dir = tmp_path / "data"
        pairs = [(data_dir / img, data_dir / sal) for img, sal in index["pairs"]]

        ds = PairDataset(pairs, transform=transform)
        assert len(ds) == 5


class TestPretrainDataModule:
    def test_data_index_path_none_raises(self) -> None:
        config = Settings(data_index_path=None)
        dm = PretrainDataModule(config)
        with pytest.raises(ValueError, match="data_index_path must be set"):
            dm.setup()

    def test_missing_pickle_raises(self, tmp_path: Path) -> None:
        config = Settings(data_index_path=tmp_path / "missing.pkl")
        dm = PretrainDataModule(config)
        with pytest.raises(FileNotFoundError, match="Pickle index not found"):
            dm.setup()

    def test_dataset_ratio(self, tmp_path: Path) -> None:
        index_path = _make_index(tmp_path, 20)
        config = Settings(
            data_index_path=index_path,
            data_dir=tmp_path / "data",
            dataset_ratio=0.5,
            input_size=INPUT_SIZE,
        )

        random.seed(42)
        dm = PretrainDataModule(config)
        dm.setup()

        assert dm.dataset is not None
        assert len(dm.dataset) == 10

    def test_full_ratio(self, tmp_path: Path) -> None:
        index_path = _make_index(tmp_path, 10)
        config = Settings(
            data_index_path=index_path,
            data_dir=tmp_path / "data",
            dataset_ratio=1.0,
            input_size=INPUT_SIZE,
        )

        dm = PretrainDataModule(config)
        dm.setup()

        assert dm.dataset is not None
        assert len(dm.dataset) == 10

    def test_dataloader_one_batch(self, tmp_path: Path) -> None:
        index_path = _make_index(tmp_path, 8)
        config = Settings(
            data_index_path=index_path,
            data_dir=tmp_path / "data",
            dataset_ratio=1.0,
            input_size=INPUT_SIZE,
            batch_size=4,
            num_workers=0,
        )

        dm = PretrainDataModule(config)
        dm.setup()
        loader = dm.train_dataloader()

        batch = next(iter(loader))
        img_stu, img_tea, mask_stu, mask_tea = batch
        assert img_stu.shape == (4, 3, INPUT_SIZE, INPUT_SIZE)
        assert img_tea.shape == (4, 3, INPUT_SIZE, INPUT_SIZE)
        assert mask_stu.shape == (4, 1, INPUT_SIZE, INPUT_SIZE)
        assert mask_tea.shape == (4, 1, INPUT_SIZE, INPUT_SIZE)

    def test_dataloader_before_setup_raises(self) -> None:
        config = Settings()
        dm = PretrainDataModule(config)
        with pytest.raises(RuntimeError, match="Call setup"):
            dm.train_dataloader()


class TestImports:
    def test_eyepacs_constants(self) -> None:
        assert len(EYEPACS_MEAN) == 3
        assert len(EYEPACS_STD) == 3
        assert all(isinstance(v, float) for v in EYEPACS_MEAN)
        assert all(isinstance(v, float) for v in EYEPACS_STD)

    def test_data_aug(self) -> None:
        assert "brightness" in DATA_AUG
        assert "scale_stu" in DATA_AUG
        assert "scale_tea" in DATA_AUG
        assert "degrees" in DATA_AUG
