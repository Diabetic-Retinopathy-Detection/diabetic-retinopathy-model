"""kNN evaluation of pretrained ViT encoders on DR classification datasets.

Ported from SSiT's ``knn.py`` (based on DINO's ``eval_knn.py``).  The
encoder is frozen — no gradient updates.  For each test image, cosine
similarity against all train features finds k nearest neighbours, and
distance-weighted voting assigns the predicted class.

Metrics reported: Top-1 Accuracy and Quadratic Weighted Kappa.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision import datasets, transforms

from dr_model.config import Settings
from dr_model.model.backbone import ViTBackbone, interpolate_pos_embed

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


# ── Dataset mean/std (from SSiT funcs.py) ──────────────────────────

_DATASET_STATS: dict[str, tuple[list[float], list[float]]] = {
    "ddr": (
        [0.423737496137619, 0.2609460651874542, 0.128403902053833],
        [0.29482534527778625, 0.20167365670204163, 0.13668020069599152],
    ),
    "aptos2019": (
        [0.46100369095802307, 0.246780663728714, 0.07989078760147095],
        [0.24873991310596466, 0.13842609524726868, 0.08025242388248444],
    ),
    "messidor2": (
        [0.48436370491981506, 0.2238118201494217, 0.07583174854516983],
        [0.2939208149909973, 0.14721707999706268, 0.06350880116224289],
    ),
}


# ── Data loading ───────────────────────────────────────────────────


class _ReturnIndexDataset(datasets.ImageFolder):
    """ImageFolder subclass that also returns the sample index."""

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        img, _lab = super().__getitem__(idx)
        return img, idx


def _build_transform(dataset: str, input_size: int) -> transforms.Compose:
    mean, std = _DATASET_STATS[dataset]
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


# ── Feature extraction ─────────────────────────────────────────────


@torch.no_grad()
def _extract_features(
    model: ViTBackbone,
    loader: DataLoader,
    device: torch.device,
) -> Tensor:
    # The dataloader runs without shuffling, so batch order equals dataset
    # order and concatenation preserves sample indices. Avoids
    # ``Tensor.index_copy_``, which is not implemented on the MPS device.
    feats_list: list[Tensor] = []
    for samples, _ in loader:
        samples = samples.to(device, non_blocking=True)
        feats_list.append(model.forward_features(samples)[:, 0].clone())  # CLS token

    if not feats_list:
        msg = "No features extracted — empty dataloader?"
        raise RuntimeError(msg)
    return torch.cat(feats_list, dim=0)


# ── kNN classifier ─────────────────────────────────────────────────


def quadratic_weighted_kappa(conf_mat: Tensor) -> float:
    """Compute QWK from a confusion matrix (torch or numpy)."""
    import numpy as np

    mat: np.ndarray = conf_mat.cpu().numpy() if isinstance(conf_mat, Tensor) else conf_mat
    n = mat.shape[0]
    weighted = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            weighted[i][j] = 1.0 - ((i - j) ** 2) / ((n - 1) ** 2)

    gt_count = mat.sum(axis=1)
    pred_count = mat.sum(axis=0)
    expected = np.outer(gt_count, pred_count)

    conf_norm = mat / mat.sum()
    expected_norm = expected / expected.sum()

    observed = (conf_norm * weighted).sum()
    expected_val = (expected_norm * weighted).sum()
    return float((observed - expected_val) / (1.0 - expected_val))


@torch.no_grad()
def knn_classifier(
    train_features: Tensor,
    train_labels: Tensor,
    test_features: Tensor,
    test_labels: Tensor,
    k: int,
    temperature: float,
    num_classes: int = 5,
) -> tuple[float, float]:
    """Run kNN classification and return (top1_accuracy, qwk)."""
    train_features = train_features.t()
    num_test = test_labels.shape[0]
    top1 = 0.0
    total = 0

    conf_mat = torch.zeros(num_classes, num_classes, device=train_features.device)

    chunk_size = max(1, num_test // 50)
    for start in range(0, num_test, chunk_size):
        end = min(start + chunk_size, num_test)
        features = test_features[start:end]
        targets = test_labels[start:end]
        batch_size = targets.shape[0]

        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)

        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved = torch.gather(candidates, 1, indices)

        one_hot = torch.zeros(batch_size * k, num_classes, device=train_features.device)
        one_hot.scatter_(1, retrieved.view(-1, 1), 1)
        weights = distances.clone().div_(temperature).exp_()
        probs = torch.sum(
            one_hot.view(batch_size, -1, num_classes) * weights.view(batch_size, -1, 1),
            dim=1,
        )
        _, predictions = probs.sort(1, True)

        correct = predictions.eq(targets.data.view(-1, 1))
        top1 += correct.narrow(1, 0, 1).sum().item()
        total += targets.size(0)

        tgt = targets.data.view(-1, 1)
        for i, p in enumerate(predictions.narrow(1, 0, 1)):
            conf_mat[int(tgt[i])][int(p.item())] += 1

    top1 = top1 * 100.0 / total
    kappa = quadratic_weighted_kappa(conf_mat)
    return top1, kappa


# ── Public API ──────────────────────────────────────────────────────


def evaluate_knn(
    encoder_path: str | Path,
    data_path: str | Path,
    dataset: str = "ddr",
    input_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    nb_knn: list[int] | None = None,
    temperature: float = 0.07,
    device: str = "auto",
) -> dict[str, dict[str, dict[str, float]]]:
    """Evaluate a pretrained encoder using kNN classification.

    Parameters
    ----------
    encoder_path
        Path to ``epoch_N.pt`` checkpoint (contains ``state_dict`` with
        ``base_encoder.*`` keys from pretraining).
    data_path
        Path to ImageFolder dataset root (contains ``train/`` and ``test/``).
    dataset
        Dataset name for mean/std normalisation (``ddr``, ``aptos2019``,
        ``messidor2``).
    input_size
        Resize target for images.
    batch_size
        Inference batch size.
    num_workers
        DataLoader workers.
    nb_knn
        List of k values to evaluate. Default ``[5, 10, 20]``.
    temperature
        Temperature for distance-weighted voting.
    device
        Device string (``"auto"``, ``"cuda"``, ``"cpu"``).

    Returns
    -------
    dict
        ``{"k-NN": {"5-NN": {"accuracy": ..., "kappa": ...}, ...}}``
    """
    if nb_knn is None:
        nb_knn = [5, 10, 20]

    from dr_model.utils.device import resolve_device

    dev = resolve_device(device)
    print(f"Using device: {dev}")

    # ── load encoder ────────────────────────────────────────────────
    model = _load_encoder(encoder_path, input_size).to(dev)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # ── load data ───────────────────────────────────────────────────
    transform = _build_transform(dataset, input_size)
    train_dataset = _ReturnIndexDataset(os.path.join(data_path, "train"), transform=transform)
    test_dataset = _ReturnIndexDataset(os.path.join(data_path, "test"), transform=transform)

    train_loader: DataLoader[tuple[Tensor, int]] = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader: DataLoader[tuple[Tensor, int]] = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print(f"Data loaded: {len(train_dataset)} train, {len(test_dataset)} test images.")

    # ── extract features ────────────────────────────────────────────
    print("Extracting train features...")
    train_features = _extract_features(model, train_loader, dev)
    print("Extracting test features...")
    test_features = _extract_features(model, test_loader, dev)

    train_features = F.normalize(train_features, dim=1, p=2)
    test_features = F.normalize(test_features, dim=1, p=2)

    train_labels = torch.tensor([s[-1] for s in train_dataset.samples], device=dev)
    test_labels = torch.tensor([s[-1] for s in test_dataset.samples], device=dev)

    # ── kNN classification ──────────────────────────────────────────
    print("Features extracted. Starting kNN classification.")
    results: dict[str, dict[str, dict[str, float]]] = {"k-NN": {}}

    for k in nb_knn:
        acc, kappa = knn_classifier(
            train_features,
            train_labels,
            test_features,
            test_labels,
            k=k,
            temperature=temperature,
        )
        key = f"{k}-NN"
        results["k-NN"][key] = {"accuracy": acc, "kappa": kappa}
        print(f"{key}: Acc={acc:.2f}%, Kappa={kappa:.4f}")

    return results


def _load_encoder(encoder_path: str | Path, input_size: int) -> ViTBackbone:
    """Build a frozen encoder at *input_size* and load *encoder_path* weights.

    Handles both ``{"state_dict": ...}`` checkpoints with a ``base_encoder.``
    prefix and bare encoder state dicts.  The positional embedding is
    interpolated (SSiT/DINO style) when the checkpoint's training resolution
    differs from *input_size*.
    """
    model = ViTBackbone(Settings(), input_size=input_size)

    ckpt = torch.load(encoder_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)

    prefix = "base_encoder."
    stripped = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}

    def _interpolate(source: dict[str, Tensor]) -> None:
        pos = source.get("pos_embed")
        if pos is None:
            return
        expected = model.pos_embed.shape[1] - 1
        if pos.shape[1] - 1 != expected:
            print(f"Interpolating pos_embed: {pos.shape[1] - 1} -> {expected} patches")
            source["pos_embed"] = interpolate_pos_embed(pos, expected)

    if stripped:
        _interpolate(stripped)
        msg = model.load_state_dict(stripped, strict=False)
        print(f"Loaded encoder weights (missing: {len(msg.missing_keys)}, unexpected: {len(msg.unexpected_keys)})")
    else:
        _interpolate(state_dict)
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded bare encoder weights (missing: {len(msg.missing_keys)}, unexpected: {len(msg.unexpected_keys)})")

    return model
