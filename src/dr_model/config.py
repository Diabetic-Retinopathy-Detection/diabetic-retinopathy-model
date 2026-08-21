"""Application configuration.

``Settings`` is the single source of truth for every hyper-parameter in the
system — architecture, training, data, and serving.  It is a Pydantic
``BaseSettings`` subclass, so values resolve from (in descending priority):

1. Constructor arguments
2. Environment variables
3. ``.env`` file
4. YAML file (``configs/pretrain_default.yaml`` unless ``DR_CONFIG_FILE`` is set)
5. Class defaults

``model_name`` is a ``@computed_field`` derived from the architecture
parameters (``patch_size``, ``embed_dim``, ``depth``, ``num_heads``,
``num_classes``).  It is read-only and always in sync — there is no manual
override.  Use it for checkpoint filenames, logging, and serving metadata.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import computed_field, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        yaml_file = os.getenv("DR_CONFIG_FILE", "configs/pretrain_default.yaml")
        yaml_values = _load_yaml_config(Path(yaml_file))
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            _YamlConfigSettingsSource(settings_cls, yaml_values),
        )

    # ── paths ──────────────────────────────────────────────────────
    checkpoint_dir: Path = Path("checkpoints")
    log_dir: Path = Path("logs")
    data_dir: Path = Path("../data")

    # ── architecture ───────────────────────────────────────────────
    patch_size: int = 16
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0

    # ── model ──────────────────────────────────────────────────────
    pretrained_weights: str | None = None
    num_classes: int = 5

    # ── pretraining ────────────────────────────────────────────────
    dim: int = 256
    mlp_dim: int = 4096
    temperature: float = 1.0
    saliency_threshold: float = 0.25
    pool_mode: str = "max"

    # ── training ───────────────────────────────────────────────────
    batch_size: int = 32
    learning_rate: float = 1e-4
    max_epochs: int = 100
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1
    precision: str = "16-mixed"
    warmup_epochs: int = 40
    momentum_base: float = 0.99
    momentum_max: float = 1.0
    lambda_c: float = 1.0
    lambda_s: float = 10.0
    ss_decay: bool = False
    save_every: int = 20
    seed: int = -1
    deterministic_algorithms: bool = False

    # ── data ───────────────────────────────────────────────────────
    image_size: tuple[int, int] = (224, 224)
    input_size: int = 224
    data_index_path: Path | None = None
    dataset_ratio: float = 1.0
    num_workers: int = 4
    pin_memory: bool = True
    train_split: float = 0.8
    val_split: float = 0.1

    # ── fine-tuning ────────────────────────────────────────────────
    finetune_dataset_root: Path | None = None
    finetune_input_size: int = 384
    finetune_mean: list[float] = [0.46100369095802307, 0.246780663728714, 0.07989078760147095]
    finetune_std: list[float] = [0.24873991310596466, 0.13842609524726868, 0.08025242388248444]
    finetune_lr: float = 2e-5
    finetune_min_lr: float = 0.0
    finetune_weight_decay: float = 1e-5
    finetune_warmup_epochs: int = 5
    finetune_epochs: int = 25
    finetune_checkpoint: str | None = None
    finetune_loss: str = "squared_emd"
    train_on_train_and_valid: bool = False
    skip_validation: bool = False

    # ── experiment tracking ─────────────────────────────────────
    mlflow: bool = True
    mlflow_tracking_uri: str | None = None
    mlflow_experiment_name: str = "dr-pretrain"
    tensorboard: bool = True

    # ── serving ────────────────────────────────────────────────────
    host: str = "0.0.0.0"  # noqa: S104
    port: int = 8000
    workers: int = 1
    max_batch_size: int = 16
    serving_checkpoint: str = "checkpoints/finetune/best_validation_weights.pt"

    # ── computed ───────────────────────────────────────────────────

    @field_validator("image_size", mode="before")
    @classmethod
    def _normalise_image_size(cls, v: object) -> tuple[int, int]:
        if isinstance(v, int):
            return (v, v)
        if isinstance(v, (list, tuple)) and len(v) == 2:
            return (int(v[0]), int(v[1]))
        msg = f"image_size must be an int or a 2-element sequence, got {type(v)}"
        raise ValueError(msg)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def model_name(self) -> str:
        return f"vit_p{self.patch_size}_e{self.embed_dim}_d{self.depth}_h{self.num_heads}_c{self.num_classes}"


class _YamlConfigSettingsSource(YamlConfigSettingsSource):
    """Settings source backed by a resolved base-plus-overlay YAML mapping."""

    def __init__(self, settings_cls: type[BaseSettings], values: dict[str, Any]) -> None:
        super().__init__(settings_cls, yaml_file=None)
        self._values = values

    def __call__(self) -> dict[str, Any]:
        return self._values


def _load_yaml_config(path: Path, seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Load a YAML config and recursively merge its optional ``_base_`` file."""
    path = path.resolve()
    if path in seen:
        chain = " -> ".join(str(item) for item in (*seen, path))
        raise ValueError from ValueError(chain)
    with path.open(encoding="utf-8") as stream:
        values = yaml.safe_load(stream) or {}
    if not isinstance(values, dict):
        raise TypeError

    base_name = values.pop("_base_", None)
    if base_name is None:
        return values
    if not isinstance(base_name, str):
        raise TypeError

    base = _load_yaml_config(path.parent / base_name, (*seen, path))
    return _merge_yaml_values(base, values)


def _merge_yaml_values(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Merge overlay values into base values, recursively for mappings."""
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _merge_yaml_values(merged[key], value)
        else:
            merged[key] = value
    return merged
