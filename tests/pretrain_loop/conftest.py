from __future__ import annotations

from dr_model.config import Settings


def _make_config(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "patch_size": 4,
        "embed_dim": 32,
        "depth": 2,
        "num_heads": 2,
        "mlp_ratio": 4.0,
        "num_classes": 5,
        "dim": 16,
        "mlp_dim": 64,
        "temperature": 1.0,
        "saliency_threshold": 0.25,
        "pool_mode": "max",
        "batch_size": 4,
        "learning_rate": 0.01,
        "max_epochs": 10,
        "warmup_epochs": 5,
        "momentum_base": 0.99,
        "momentum_max": 1.0,
        "lambda_c": 1.0,
        "lambda_s": 10.0,
        "ss_decay": False,
        "save_every": 5,
        "gradient_clip_val": 1.0,
        "precision": "32-true",
        "input_size": 16,
        "image_size": (16, 16),
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]
