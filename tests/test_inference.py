from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

import dr_model.inference as inference
from dr_model.config import Settings


def _settings() -> Settings:
    return Settings(
        patch_size=16,
        embed_dim=32,
        depth=2,
        num_heads=4,
        image_size=(32, 32),
        finetune_input_size=32,
        finetune_mean=[0.5, 0.25, 0.1],
        finetune_std=[0.25, 0.5, 0.2],
        serving_checkpoint="model.pt",
    )


def test_preprocess_image_matches_eval_normalization() -> None:
    image = Image.new("RGB", (8, 8), color=(255, 128, 51))

    tensor = inference.preprocess_image(image, _settings())

    assert tensor.shape == (1, 3, 32, 32)
    assert tensor.dtype == torch.float32
    assert torch.allclose(tensor[0, :, 0, 0], torch.tensor([2.0, 0.5039, 0.5]), atol=1e-4)


def test_load_model_caches_model_and_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[tuple[Any, str]] = []

    class FakeModel:
        def __init__(self, config: Settings, checkpoint_path: str) -> None:
            created.append((config, checkpoint_path))

        def eval(self) -> FakeModel:
            return self

    monkeypatch.setattr(inference, "Settings", _settings)
    monkeypatch.setattr(inference, "Finetuner", FakeModel)
    monkeypatch.setattr(inference, "_MODEL", None)
    monkeypatch.setattr(inference, "_CONFIG", None)

    first_model, first_config = inference.load_model()
    second_model, second_config = inference.load_model()

    assert first_model is second_model
    assert first_config is second_config
    assert created == [(first_config, "model.pt")]


def test_predict_returns_named_probabilities(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _settings()

    class FakeModel:
        def __call__(self, _inputs: torch.Tensor) -> torch.Tensor:
            return torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0]])

    monkeypatch.setattr(inference, "load_model", lambda: (FakeModel(), config))

    probabilities = inference.predict(Image.new("RGB", (8, 8)))

    assert tuple(probabilities) == inference.CLASS_NAMES
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert max(probabilities, key=lambda label: probabilities[label]) == "Proliferative DR"


def test_predict_accepts_numpy_arrays_and_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _settings()

    class FakeModel:
        def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
            assert inputs.shape == (1, 3, 32, 32)
            return torch.zeros((1, 5))

    monkeypatch.setattr(inference, "load_model", lambda: (FakeModel(), config))
    image_path = tmp_path / "image.png"
    Image.new("RGB", (8, 8)).save(image_path)

    assert len(inference.predict(image_path)) == 5
    assert len(inference.predict(np.zeros((8, 8, 3), dtype="uint8"))) == 5
