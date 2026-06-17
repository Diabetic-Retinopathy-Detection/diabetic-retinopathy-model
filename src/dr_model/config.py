from __future__ import annotations

import os
from pathlib import Path

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

    # ── model ──────────────────────────────────────────────────────
    model_name: str = "vit_base"
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
