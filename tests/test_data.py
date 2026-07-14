from __future__ import annotations

import pickle
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from dr_model.data import PairDataset


def _make_image(path: Path, size: tuple[int, int] = (224, 224)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _make_saliency(path: Path, size: tuple[int, int] = (224, 224)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.random.rand(*size).astype(np.float32)
    np.save(str(path), arr)


def _make_pickle(pkl_path: Path, root: str | None, pairs: list[tuple[str, str]]) -> None:
    data = {"root": root, "pairs": [(Path(a), Path(b)) for a, b in pairs]}
    with pkl_path.open("wb") as f:
        pickle.dump(data, f)


class TestPairDataset:
    def test_loads_relative_paths(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        img_dir = data_dir / "cropped"
        sal_dir = data_dir / "saliency"

        _make_image(img_dir / "a.jpeg")
        _make_saliency(sal_dir / "a.npy")
        _make_pickle(
            tmp_path / "index.pkl",
            root=str(data_dir),
            pairs=[("cropped/a.jpeg", "saliency/a.npy")],
        )

        ds = PairDataset(tmp_path / "index.pkl", data_dir)
        assert len(ds) == 1

        image, _saliency = ds[0]
        assert image.shape == (3, 224, 224)
        assert image.dtype == torch.float32
        assert _saliency.shape == (1, 224, 224)
        assert _saliency.dtype == torch.float32

    def test_loads_absolute_paths(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        img_dir = data_dir / "cropped"
        sal_dir = data_dir / "saliency"

        _make_image(img_dir / "a.jpeg")
        _make_saliency(sal_dir / "a.npy")

        abs_img = str((img_dir / "a.jpeg").absolute())
        abs_sal = str((sal_dir / "a.npy").absolute())
        _make_pickle(
            tmp_path / "index.pkl",
            root=None,
            pairs=[(abs_img, abs_sal)],
        )

        ds = PairDataset(tmp_path / "index.pkl", data_dir)
        image, _saliency = ds[0]
        assert image.shape == (3, 224, 224)

    def test_len(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        img_dir = data_dir / "cropped"
        sal_dir = data_dir / "saliency"

        for name in ("a", "b", "c"):
            _make_image(img_dir / f"{name}.jpeg")
            _make_saliency(sal_dir / f"{name}.npy")

        _make_pickle(
            tmp_path / "index.pkl",
            root=str(data_dir),
            pairs=[
                ("cropped/a.jpeg", "saliency/a.npy"),
                ("cropped/b.jpeg", "saliency/b.npy"),
                ("cropped/c.jpeg", "saliency/c.npy"),
            ],
        )

        ds = PairDataset(tmp_path / "index.pkl", data_dir)
        assert len(ds) == 3

    def test_missing_image_raises(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        _make_pickle(
            tmp_path / "index.pkl",
            root=str(data_dir),
            pairs=[("cropped/missing.jpeg", "saliency/missing.npy")],
        )

        ds = PairDataset(tmp_path / "index.pkl", data_dir)
        with pytest.raises((FileNotFoundError, OSError)):
            ds[0]

    def test_pixel_values_in_range(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        img_dir = data_dir / "cropped"
        sal_dir = data_dir / "saliency"

        _make_image(img_dir / "a.jpeg")
        _make_saliency(sal_dir / "a.npy")
        _make_pickle(
            tmp_path / "index.pkl",
            root=str(data_dir),
            pairs=[("cropped/a.jpeg", "saliency/a.npy")],
        )

        ds = PairDataset(tmp_path / "index.pkl", data_dir)
        image, _saliency = ds[0]
        assert image.min() >= 0.0
        assert image.max() <= 1.0
        assert _saliency.min() >= 0.0
