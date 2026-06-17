"""Inference pipeline: preprocessing transforms and predict()."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


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
    raise NotImplementedError("predict() is not implemented yet — this is a scaffold.")
