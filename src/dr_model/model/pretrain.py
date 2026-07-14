"""Saliency-supervised self-distillation pretrainer.

Mirrors the ``SSiT`` class from ``checkpoints/referencematerial/SSiT/ssit.py``,
adapted to :class:`dr_model.model.backbone.ViTBackbone`.

Two simultaneous objectives:

1. MoCo v3-style contrastive learning across two augmented views.
2. Saliency segmentation auxiliary task that forces the encoder to localise
   clinically relevant regions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from dr_model.config import Settings
from dr_model.model.backbone import ViTBackbone


def _build_mlp(
    num_layers: int,
    input_dim: int,
    mlp_dim: int,
    output_dim: int,
    last_bn: bool = True,
) -> nn.Sequential:
    """Build an MLP with optional BatchNorm between layers.

    Follows SimCLR's design: the final BN layer has no affine parameters
    (``affine=False``).
    """
    layers: list[nn.Module] = []
    for i in range(num_layers):
        dim_in = input_dim if i == 0 else mlp_dim
        dim_out = output_dim if i == num_layers - 1 else mlp_dim
        layers.append(nn.Linear(dim_in, dim_out, bias=False))
        if i < num_layers - 1:
            layers.append(nn.BatchNorm1d(dim_out))
            layers.append(nn.ReLU(inplace=True))
        elif last_bn:
            layers.append(nn.BatchNorm1d(dim_out, affine=False))
    return nn.Sequential(*layers)


@torch.no_grad()
def concat_all_gather(tensor: Tensor) -> Tensor:
    """Concatenate tensors from all processes in distributed training.

    Returns the input unchanged when not in a distributed setting.
    """
    if not torch.distributed.is_initialized():
        return tensor
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    return torch.cat(tensors_gather, dim=0)


class Pretrainer(nn.Module):
    """Saliency-guided self-distillation pretrainer.

    Wraps two :class:`ViTBackbone` instances — a base encoder trained by
    gradient and a momentum encoder updated via EMA — with contrastive and
    saliency segmentation objectives.

    Parameters
    ----------
    config : Settings
        Application settings containing architecture and pretraining
        hyper-parameters.
    """

    def __init__(self, config: Settings) -> None:
        super().__init__()

        self.T = config.temperature
        self.saliency_threshold = config.saliency_threshold

        mlp_dim = config.mlp_dim
        dim = config.dim
        patch_size = config.patch_size

        self.base_encoder = ViTBackbone(config)
        self.momentum_encoder = ViTBackbone(config)

        hidden_dim = self.base_encoder.head.weight.shape[1]
        del self.base_encoder.head, self.momentum_encoder.head
        self.base_encoder.head = _build_mlp(3, hidden_dim, mlp_dim, dim)  # type: ignore[attr-defined]
        self.momentum_encoder.head = _build_mlp(3, hidden_dim, mlp_dim, dim)  # type: ignore[attr-defined]

        self.predictor = _build_mlp(2, dim, mlp_dim, dim)

        self.saliency_segmentor = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim, out_channels=patch_size**2, kernel_size=1),
            nn.PixelShuffle(upscale_factor=patch_size),
        )

        if config.pool_mode == "max":
            self.pool = nn.MaxPool2d(kernel_size=patch_size, stride=patch_size)
        elif config.pool_mode == "avg":
            self.pool = nn.AvgPool2d(kernel_size=patch_size, stride=patch_size)
        else:
            self.pool = None  # type: ignore[assignment]

        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters(), strict=True):
            param_m.data.copy_(param_b.data)
            param_m.requires_grad = False

    @torch.no_grad()
    def _update_momentum_encoder(self, m: float) -> None:
        """EMA update of the momentum encoder."""
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters(), strict=True):
            param_m.data = param_m.data * m + param_b.data * (1.0 - m)

    def contrastive_loss(self, q: Tensor, k: Tensor) -> Tensor:
        """InfoNCE contrastive loss between query and key projections."""
        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)
        k = concat_all_gather(k)

        logits = torch.einsum("nc,mc->nm", [q, k]) / self.T
        N = logits.shape[0]
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        labels = torch.arange(N, dtype=torch.long, device=q.device) + N * rank
        return F.cross_entropy(logits, labels) * (2 * self.T)

    def saliency_segmentation_loss(self, f: Tensor, m: Tensor) -> Tensor:
        """BCE loss between upsampled patch features and thresholded saliency."""
        f = f[:, 1:]
        m = (m > self.saliency_threshold).float()

        B, L, C = f.shape
        H = W = int(L**0.5)
        f = f.permute(0, 2, 1).reshape(B, C, H, W)
        ss = self.saliency_segmentor(f)

        return F.binary_cross_entropy_with_logits(ss, m)

    def _encode(self, encoder: ViTBackbone, x: Tensor, pmap: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Forward through encoder, returning (projection, full_features)."""
        features = encoder.forward_features(x, pmap=pmap)
        projection = encoder.head(features[:, 0])  # type: ignore[union-attr]
        return projection, features

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        m1: Tensor,
        m2: Tensor,
        momentum_m: float,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass.

        Parameters
        ----------
        x1, x2 : Tensor
            Two augmented views, shape ``(B, 3, H, W)``.
        m1, m2 : Tensor
            Corresponding saliency maps, shape ``(B, 1, H, W)``.
        momentum_m : float
            Current momentum coefficient for EMA update.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(contrastive_loss, saliency_loss)`` — both scalar tensors.
        """
        mp1 = None if self.pool is None else self.pool(m1)
        mp2 = None if self.pool is None else self.pool(m2)

        t1, f1 = self._encode(self.base_encoder, x1)
        t2, f2 = self._encode(self.base_encoder, x2)

        q1 = self.predictor(t1)
        q2 = self.predictor(t2)

        with torch.no_grad():
            self._update_momentum_encoder(momentum_m)
            k1, _ = self._encode(self.momentum_encoder, x1, mp1)
            k2, _ = self._encode(self.momentum_encoder, x2, mp2)

        cl_loss = self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)
        sp_loss = self.saliency_segmentation_loss(f1, m1) + self.saliency_segmentation_loss(f2, m2)
        return cl_loss, sp_loss
