"""Supervised fine-tuning model for DR grading.

Mirrors SSiT's ``eval.py`` fine-tuning: the backbone's classification head
is removed so the trunk emits the raw ``[CLS; patch tokens]`` sequence, the
representation is the concatenation of the CLS token and the mean of all
patch tokens (§III-B of the SSiT paper), and a single linear layer — no
hidden layers — maps it to logits.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
from torch import Tensor

from dr_model.config import Settings
from dr_model.model.backbone import ViTBackbone, interpolate_pos_embed


class Finetuner(nn.Module):
    """Supervised classifier on top of a ViT backbone.

    Parameters
    ----------
    config : Settings
        Application settings.  The trunk is built at
        ``config.finetune_input_size`` (distinct from the pretraining
        ``config.input_size``) so a ``pos_embed`` trained at a lower
        resolution is interpolated when a checkpoint is loaded.
    checkpoint_path : str | None
        Optional pretrain checkpoint to seed the trunk with.  Accepts both
        DRC-46 formats: a full ``checkpoint.pt`` (``state_dict`` with
        ``base_encoder.*`` keys) or a bare ``epoch_{N}_encoder.pt``
        (unprefixed ``base_encoder`` weights).
    """

    def __init__(self, config: Settings, checkpoint_path: str | None = None) -> None:
        super().__init__()
        self.backbone = ViTBackbone(config, input_size=config.finetune_input_size)
        self.backbone.head = nn.Identity()  # type: ignore[assignment]
        self.head = nn.Linear(2 * config.embed_dim, config.num_classes)
        self._init_head()

        if checkpoint_path is not None:
            self._load_pretrained(checkpoint_path)

    def _init_head(self) -> None:
        """Initialise the classification head with the backbone's convention."""
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Return logits, shape ``(B, num_classes)``.

        The representation is ``[z^L_class ; mean_i z^L_i]`` — the CLS token
        concatenated with the mean of all patch tokens — matching SSiT's
        ``feat_concat`` (§III-B).
        """
        feats = self.backbone.forward_features(x)
        cls = feats[:, 0]
        mean = feats[:, 1:].mean(dim=1)
        rep = torch.cat((cls, mean), dim=1)
        return cast(Tensor, self.head(rep))

    def _load_pretrained(self, checkpoint_path: str | Path) -> None:
        """Load trunk weights from a DRC-46 pretrain checkpoint.

        The pretrain classification head (an MLP projector in ``head.*``)
        is dropped; only the encoder trunk is transferred.  The positional
        embedding is interpolated when the checkpoint was trained at a
        different resolution than ``finetune_input_size``.
        """
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            msg = f"Checkpoint not found: {ckpt_path}"
            raise FileNotFoundError(msg)

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

        has_prefix = any(str(k).startswith("base_encoder.") for k in state_dict)
        filtered: dict[str, Tensor] = {}
        for key, value in state_dict.items():
            if has_prefix:
                if not key.startswith("base_encoder."):
                    continue
                key = key[len("base_encoder.") :]
            if key.startswith("head."):
                continue
            filtered[f"backbone.{key}"] = value

        if "backbone.pos_embed" in filtered:
            expected = self.backbone.pos_embed.shape[1] - 1
            filtered["backbone.pos_embed"] = interpolate_pos_embed(filtered["backbone.pos_embed"], expected)

        missing, unexpected = self.load_state_dict(filtered, strict=False)
        if set(missing) != {"head.weight", "head.bias"}:
            msg = f"Unexpected missing keys when loading {ckpt_path}: {sorted(missing)}"
            raise RuntimeError(msg)
        if unexpected:
            msg = f"Unexpected keys when loading {ckpt_path}: {sorted(unexpected)}"
            raise RuntimeError(msg)

        print(f"Loaded trunk weights from {ckpt_path} (missing classification head keys: {sorted(missing)})")
