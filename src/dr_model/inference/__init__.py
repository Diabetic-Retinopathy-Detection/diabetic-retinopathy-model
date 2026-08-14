"""Inference pipeline: preprocessing transforms and predict()."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torchvision import transforms

from dr_model.config import Settings
from dr_model.model.finetune import Finetuner

_MODEL: Finetuner | None = None
_CONFIG: Settings | None = None

CLASS_NAMES = ("No DR", "Mild", "Moderate", "Severe", "Proliferative DR")


def load_model() -> tuple[Finetuner, Settings]:
    global _MODEL, _CONFIG
    if _MODEL is not None and _CONFIG is not None:
        return _MODEL, _CONFIG

    config = Settings()
    model = Finetuner(config, checkpoint_path=config.serving_checkpoint)
    model.eval()

    _MODEL = model
    _CONFIG = config
    return model, config


def preprocess_image(image: Image.Image, config: Settings) -> Tensor:
    transform = transforms.Compose([
        transforms.Resize((config.finetune_input_size, config.finetune_input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.finetune_mean, std=config.finetune_std),
    ])
    tensor = transform(image.convert("RGB")).unsqueeze(0)
    return cast(Tensor, tensor)


def predict(image: Image.Image | np.ndarray | str | Path) -> dict[str, float]:
    """Run inference on a single fundus image.

    Parameters
    ----------
    image : PIL Image, numpy array, or path to an image file.

    Returns
    -------
    dict[str, float]
        Mapping from class label to predicted probability.
    """
    model, config = load_model()

    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert("RGB")
    else:
        image = image.convert("RGB")

    x = preprocess_image(image, config)

    with torch.inference_mode():
        logits = model(x)
        probs = cast(list[float], torch.softmax(logits, dim=1)[0].cpu().tolist())

    return dict(zip(CLASS_NAMES, probs, strict=True))
