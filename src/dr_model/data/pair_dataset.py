"""Saliency-guided pretraining dataset with asymmetric student/teacher augmentation.

Reproduces the ``PairGenerator`` + ``TransformWithMask`` logic from SSiT
(``checkpoints/SSiT/data.py``) inside the project's own package.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from torchvision.transforms import transforms

DATA_AUG: dict[str, object] = {
    "brightness": 0.4,
    "contrast": 0.4,
    "saturation": 0.2,
    "hue": 0.1,
    "scale_stu": (0.08, 0.8),
    "scale_tea": (0.8, 1.0),
    "degrees": (-180, 180),
}


class GaussianBlur:
    """Apply Gaussian blur with a sampled sigma."""

    def __init__(self, sigma: list[float]) -> None:
        self.sigma = sigma

    def __call__(self, img: Image.Image) -> Image.Image:
        sigma = random.uniform(self.sigma[0], self.sigma[1])  # noqa: S311
        return img.filter(ImageFilter.GaussianBlur(radius=sigma))


class Solarize:
    """Invert pixels above a threshold."""

    def __init__(self, threshold: int = 128) -> None:
        self.threshold = threshold

    def __call__(self, img: Image.Image) -> Image.Image:
        return ImageOps.solarize(img, self.threshold)


class TransformWithMask:
    """Asymmetric student/teacher augmentation with paired spatial transforms.

    Every spatial transform (crop, rotation, flip) is applied to the image and
    mask with the same sampled parameters.  Colour-only transforms are applied
    to the image only.
    """

    def __init__(
        self,
        input_size: int,
        mean: list[float],
        std: list[float],
        data_aug: dict[str, object],
    ) -> None:
        self.input_size = input_size
        self.mean = mean
        self.std = std

        self.scale_stu: tuple[float, float] = data_aug["scale_stu"]  # type: ignore[assignment]
        self.scale_tea: tuple[float, float] = data_aug["scale_tea"]  # type: ignore[assignment]
        self.degrees: tuple[float, float] = data_aug["degrees"]  # type: ignore[assignment]
        self.brightness: float = data_aug["brightness"]  # type: ignore[assignment]
        self.contrast: float = data_aug["contrast"]  # type: ignore[assignment]
        self.saturation: float = data_aug["saturation"]  # type: ignore[assignment]
        self.hue: float = data_aug["hue"]  # type: ignore[assignment]

        self.student_crop = transforms.RandomResizedCrop(input_size, scale=self.scale_stu, ratio=(0.7, 1.3 / 0.7))
        self.teacher_crop = transforms.RandomResizedCrop(input_size, scale=self.scale_tea, ratio=(0.7, 1.3 / 0.7))

        self.student_blur = transforms.RandomApply([GaussianBlur(sigma=[0.1, 2.0])], p=1.0)
        self.teacher_blur = transforms.RandomApply([GaussianBlur(sigma=[0.1, 2.0])], p=0.1)
        self.teacher_solarize = transforms.RandomApply([Solarize()], p=0.2)

        self.jitter = transforms.RandomApply(
            [transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)],
            p=0.8,
        )
        self.grayscale = transforms.RandomGrayscale(p=0.2)
        self.rotation = transforms.RandomRotation(degrees=self.degrees)

    def _resized_crop_with_mask(
        self, tf: transforms.RandomResizedCrop, img: Image.Image, mask: Image.Image
    ) -> tuple[Image.Image, Image.Image]:
        assert isinstance(tf, transforms.RandomResizedCrop)  # noqa: S101
        i, j, h, w = tf.get_params(img, tf.scale, tf.ratio)
        img = F.resized_crop(img, i, j, h, w, tf.size, interpolation=F.InterpolationMode.BICUBIC)
        mask = F.resized_crop(mask, i, j, h, w, tf.size, interpolation=F.InterpolationMode.NEAREST)
        return img, mask

    def _rotation_with_mask(
        self, tf: transforms.RandomRotation, img: Image.Image, mask: Image.Image, p: float
    ) -> tuple[Image.Image, Image.Image]:
        assert isinstance(tf, transforms.RandomRotation)  # noqa: S101
        if random.random() < p:  # noqa: S311
            angle = tf.get_params(tf.degrees)
            img = F.rotate(
                img,
                angle,
                interpolation=F.InterpolationMode.BILINEAR,
                expand=tf.expand,
                center=tf.center,
                fill=tf.fill,
            )
            mask = F.rotate(
                mask,
                angle,
                interpolation=F.InterpolationMode.NEAREST,
                expand=tf.expand,
                center=tf.center,
                fill=tf.fill,
            )
        return img, mask

    def _hflip_with_mask(self, img: Image.Image, mask: Image.Image, p: float) -> tuple[Image.Image, Image.Image]:
        if random.random() < p:  # noqa: S311
            img = F.hflip(img)
            mask = F.hflip(mask)
        return img, mask

    def _vflip_with_mask(self, img: Image.Image, mask: Image.Image, p: float) -> tuple[Image.Image, Image.Image]:
        if random.random() < p:  # noqa: S311
            img = F.vflip(img)
            mask = F.vflip(mask)
        return img, mask

    def _student_view(self, img: Image.Image, mask: Image.Image) -> tuple[Tensor, Tensor]:
        img, mask = self._resized_crop_with_mask(self.student_crop, img, mask)
        img = self.jitter(img)
        img = self.grayscale(img)
        img = self.student_blur(img)
        img, mask = self._rotation_with_mask(self.rotation, img, mask, p=0.8)
        img, mask = self._hflip_with_mask(img, mask, p=0.5)
        img, mask = self._vflip_with_mask(img, mask, p=0.5)

        img_tensor = F.to_tensor(img)
        img_tensor = F.normalize(img_tensor, mean=self.mean, std=self.std)
        mask_tensor = F.to_tensor(mask)
        return img_tensor, mask_tensor

    def _teacher_view(self, img: Image.Image, mask: Image.Image) -> tuple[Tensor, Tensor]:
        img, mask = self._resized_crop_with_mask(self.teacher_crop, img, mask)
        img = self.jitter(img)
        img = self.grayscale(img)
        img = self.teacher_blur(img)
        img = self.teacher_solarize(img)
        img, mask = self._rotation_with_mask(self.rotation, img, mask, p=0.8)
        img, mask = self._hflip_with_mask(img, mask, p=0.5)
        img, mask = self._vflip_with_mask(img, mask, p=0.5)

        img_tensor = F.to_tensor(img)
        img_tensor = F.normalize(img_tensor, mean=self.mean, std=self.std)
        mask_tensor = F.to_tensor(mask)
        return img_tensor, mask_tensor

    def __call__(self, img: Image.Image, mask: Image.Image) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        img_stu, mask_stu = self._student_view(img.copy(), mask.copy())
        img_tea, mask_tea = self._teacher_view(img, mask)
        return img_stu, img_tea, mask_stu, mask_tea


class PairDataset(Dataset):
    """Image-saliency pair dataset for saliency-guided pretraining.

    Parameters
    ----------
    data : list[tuple[Path, Path]]
        Pre-loaded index of ``(image_path, saliency_path)`` pairs.
    transform : TransformWithMask | None
        Augmentation pipeline.  ``None`` is rejected — a pair dataset
        without transforms has no valid use case in pretraining.
    """

    def __init__(self, data: list[tuple[Path, Path]], transform: TransformWithMask | None = None) -> None:
        if transform is None:
            msg = "PairDataset requires a TransformWithMask instance"
            raise ValueError(msg)
        self.data = data
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        img_path, mask_path = self.data[index]

        if not img_path.exists():
            msg = f"Image not found: {img_path}"
            raise FileNotFoundError(msg)
        if not mask_path.exists():
            msg = f"Saliency map not found: {mask_path}"
            raise FileNotFoundError(msg)

        img = Image.open(img_path).convert("RGB")
        mask = Image.fromarray(np.uint8(np.load(mask_path) * 255))

        return self.transform(img, mask)
