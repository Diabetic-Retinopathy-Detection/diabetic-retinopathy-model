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
        Optional checkpoint to seed the trunk with.  Accepts both DRC-46
        pretrain formats (a full ``checkpoint.pt`` with ``base_encoder.*``
        keys or a bare ``epoch_{N}_encoder.pt`` with unprefixed weights) and
        a DRC-49 fine-tuned checkpoint (``backbone.*`` plus a trained
        ``head.*`` classifier).
    """

    def __init__(self, config: Settings, checkpoint_path: str | None = None) -> None:
        super().__init__()
        self.backbone = ViTBackbone(config, input_size=config.finetune_input_size)
        self.backbone.head = nn.Identity()  # type: ignore[assignment]
        self.head = nn.Linear(2 * config.embed_dim, config.num_classes)
        self._init_head()

        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)

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

    def _load_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Load trunk (and, for fine-tuned checkpoints, head) weights.

        The checkpoint format is detected from its key prefixes:

        - ``base_encoder.`` — DRC-46 full checkpoint: keep only
          ``base_encoder.*``, strip the prefix, drop the pretrain projector
          (``head.*``), and remap to ``backbone.*``.
        - ``backbone.`` — DRC-49 fine-tuned checkpoint: ``backbone.*`` is the
          trunk as-is; top-level ``head.*`` is a trained classifier head.
        - otherwise — DRC-46 bare encoder: keep unprefixed keys, drop the
          projector ``head.*``, and remap to ``backbone.*``.

        The positional embedding is interpolated when the checkpoint was
        trained at a different resolution than ``finetune_input_size``
        (idempotent when the resolutions already match).  A compatible
        fine-tuned head is loaded on top; otherwise the randomly initialised
        head is kept.
        """
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            msg = f"Checkpoint not found: {ckpt_path}"
            raise FileNotFoundError(msg)

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

        keys = list(state_dict)
        if any(k.startswith("base_encoder.") for k in keys):
            trunk: dict[str, Tensor] = {
                f"backbone.{k[len('base_encoder.') :]}": v
                for k, v in state_dict.items()
                if k.startswith("base_encoder.") and not k[len("base_encoder.") :].startswith("head.")
            }
            head: dict[str, Tensor] = {}
        elif any(k.startswith("backbone.") for k in keys):
            trunk = {k: v for k, v in state_dict.items() if k.startswith("backbone.")}
            head = {k: v for k, v in state_dict.items() if k.startswith("head.")}
        else:
            trunk = {f"backbone.{k}": v for k, v in state_dict.items() if not k.startswith("head.")}
            head = {}

        if "backbone.pos_embed" in trunk:
            expected = self.backbone.pos_embed.shape[1] - 1
            trunk["backbone.pos_embed"] = interpolate_pos_embed(trunk["backbone.pos_embed"], expected)

        missing, unexpected = self.load_state_dict(trunk, strict=False)
        if not set(missing) <= {"head.weight", "head.bias"}:
            msg = f"Unexpected missing keys when loading {ckpt_path}: {sorted(missing)}"
            raise RuntimeError(msg)
        if unexpected:
            msg = f"Unexpected keys when loading {ckpt_path}: {sorted(unexpected)}"
            raise RuntimeError(msg)

        print(f"Loaded trunk weights from {ckpt_path}.")
        if head:
            self._load_classifier_head(head, ckpt_path)

    def _load_classifier_head(self, head: dict[str, Tensor], ckpt_path: Path) -> None:
        """Load a fine-tuned classifier head, keeping it random on any mismatch.

        Only keys whose name *and* shape match ``self.head`` are kept, which
        avoids ``load_state_dict``'s ``strict=False`` size-mismatch error.
        When every expected key matches, the head is overwritten; otherwise it
        stays randomly initialised (trunk only).
        """
        expected = self.head.state_dict()
        compatible = {
            k[len("head.") :]: v
            for k, v in head.items()
            if k.startswith("head.") and v.shape == expected[k[len("head.") :]].shape
        }
        if set(compatible) == set(expected):
            self.head.load_state_dict(compatible, strict=True)
            print(f"Loaded classifier head from {ckpt_path}.")
        else:
            incompatible = sorted(set(expected) - set(compatible))
            print(f"Kept randomly initialised classifier head (incompatible keys: {incompatible}); trunk only.")
