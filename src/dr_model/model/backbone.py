"""Vision Transformer backbone for diabetic retinopathy classification.

Pure PyTorch ``nn.Module`` — no training loop or serving logic.
All architecture parameters are read from :class:`dr_model.config.Settings`.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import torch
from torch import Tensor, nn

from dr_model.config import Settings


class PatchEmbed(nn.Module):
    """2-D image to patch embedding via a single convolution."""

    def __init__(self, patch_size: int, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:
        return cast(Tensor, self.proj(x).flatten(2).transpose(1, 2))


class Block(nn.Module):
    """Transformer encoder block: pre-norm attention + pre-norm MLP."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
        drop_rate: float,
        attn_drop_rate: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=attn_drop_rate,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(drop_rate),
        )
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop(h)
        x = x + self.mlp(self.norm2(x))
        return x


class PatchSampler:
    """Select top-k patches by saliency score for saliency-guided masking."""

    def __init__(self, patch_size: int, mask_ratio: float) -> None:
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

    def __call__(self, pmap: Tensor) -> Tensor:
        B, _C, H, W = pmap.shape
        num_sample = int((1 - self.mask_ratio) * H * W)
        feat_idx = pmap.flatten(1).argsort(descending=True)[:, :num_sample]
        feat_idx += 1  # class embedding index
        cls_idx = torch.zeros((B, 1), dtype=torch.int64, device=pmap.device)
        return torch.cat([cls_idx, feat_idx], dim=1)


class ViTBackbone(nn.Module):
    """Vision Transformer backbone for DR classification.

    Instantiates cleanly from :class:`dr_model.config.Settings` with zero
    magic constants — every architecture parameter is config-driven.

    Parameters
    ----------
    config : Settings
        Application settings containing architecture hyper-parameters.

    Attributes
    ----------
    Class invariant: ``pos_embed.shape[1] == num_patches + 1``
        The +1 accounts for the learnable class token prepended to the patch
        sequence.
    """

    def __init__(self, config: Settings) -> None:
        super().__init__()

        patch_size = config.patch_size
        embed_dim = config.embed_dim
        depth = config.depth
        num_heads = config.num_heads
        mlp_ratio = config.mlp_ratio
        drop_rate = config.drop_rate
        attn_drop_rate = config.attn_drop_rate
        num_classes = config.num_classes

        img_h, img_w = config.image_size
        num_patches = (img_h // patch_size) * (img_w // patch_size)

        self.patch_embed = PatchEmbed(patch_size, embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.Sequential(*[
            Block(embed_dim, num_heads, mlp_ratio, drop_rate, attn_drop_rate) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.Linear(embed_dim, num_classes)

        self.patch_sampler = PatchSampler(patch_size, mask_ratio=0.25)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward_features(self, x: Tensor, pmap: Tensor | None = None) -> Tensor:
        """Run the encoder without the classification head.

        When *pmap* is provided, applies saliency-guided patch masking: only
        the top-k patches (by saliency score) are fed to the transformer
        blocks.  This is used by the momentum encoder in pretraining.
        """
        x = self.patch_embed(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + self.pos_embed

        if pmap is not None:
            active_idx = self.patch_sampler(pmap)
            active_idx = active_idx.unsqueeze(-1).repeat(1, 1, x.shape[-1])
            x = torch.gather(x, dim=1, index=active_idx)

        x = self.pos_drop(x)
        x = self.blocks(x)
        return cast(Tensor, self.norm(x))

    def forward(self, x: Tensor) -> Tensor:
        """Run the full encoder + classification head.

        Parameters
        ----------
        x : Tensor
            Input images, shape ``(B, 3, H, W)``.

        Returns
        -------
        Tensor
            Logits, shape ``(B, num_classes)``.
        """
        x = self.patch_embed(x)

        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1)

        x = self.pos_drop(x + self.pos_embed)

        x = self.blocks(x)
        x = self.norm(x)

        return cast(Tensor, self.head(x[:, 0]))

    @torch.no_grad()
    def load_checkpoint(self, path: str | Path, strict: bool = True) -> None:
        """Load model weights from a checkpoint file.

        Parameters
        ----------
        path : str | Path
            Path to a ``.pt`` / ``.pth`` checkpoint.  Supports both bare
            ``state_dict`` files and the VAE-style wrapper format
            ``{"state_dict": ..., "model_config": ...}``.
        strict : bool, optional
            If ``True`` (default), all keys must match exactly.  Set to
            ``False`` when loading a pretrained encoder with a different
            ``num_classes`` (e.g. SSiT ImageNet-1k → DR 5-class).
            Size-mismatched keys (typically the classification head) are
            silently skipped in non-strict mode.
        """
        ckpt_path = Path(path)
        if not ckpt_path.exists():
            msg = f"Checkpoint not found: {ckpt_path}"
            raise FileNotFoundError(msg)

        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

        if strict:
            self.load_state_dict(state_dict, strict=True)
        else:
            model_state = self.state_dict()
            filtered = {k: v for k, v in state_dict.items() if k in model_state and v.shape == model_state[k].shape}
            self.load_state_dict(filtered, strict=False)
