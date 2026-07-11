from __future__ import annotations

import os
from pathlib import Path

from pydantic import computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

YAML_FILE = os.getenv("DR_CONFIG_FILE", "configs/pretrain_default.yaml")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        yaml_file=YAML_FILE,
    )

    # ── paths ──────────────────────────────────────────────────────
    checkpoint_dir: Path = Path("checkpoints")
    log_dir: Path = Path("logs")
    data_dir: Path | None = None

    # ── architecture ───────────────────────────────────────────────
    # Standard ViT-B/16 configuration per Dosovitskiy et al. (2020).
    patch_size: int = 16
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    # Training-time only — inference uses model.eval() to disable these.
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0

    # ── model ──────────────────────────────────────────────────────
    pretrained_weights: str | None = None
    num_classes: int = 5

    # ── training ───────────────────────────────────────────────────
    batch_size: int = 32
    learning_rate: float = 1e-4
    max_epochs: int = 100
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1
    precision: str = "16-mixed"

    # ── data ───────────────────────────────────────────────────────
    image_size: tuple[int, int] = (224, 224)
    num_workers: int = 4
    train_split: float = 0.8
    val_split: float = 0.1

    # ── serving ────────────────────────────────────────────────────
    host: str = "0.0.0.0"  # noqa: S104
    port: int = 8000
    workers: int = 1
    max_batch_size: int = 16

    # ── computed ───────────────────────────────────────────────────

    @field_validator("image_size", mode="before")
    @classmethod
    def _normalise_image_size(cls, v: object) -> tuple[int, int]:
        if isinstance(v, int):
            return (v, v)
        return v  # type: ignore[return-value]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def model_name(self) -> str:
        return f"vit_p{self.patch_size}_e{self.embed_dim}_d{self.depth}_h{self.num_heads}_c{self.num_classes}"
