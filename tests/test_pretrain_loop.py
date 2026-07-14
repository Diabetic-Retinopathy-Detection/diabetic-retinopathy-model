"""Tests for the pretraining loop — schedules, checkpointing, and integration."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from dr_model.config import Settings
from dr_model.model.pretrain import Pretrainer
from dr_model.training.pretrain_loop import (
    _adjust_lambda_s,
    _adjust_lr,
    _adjust_momentum,
    _save_checkpoint,
    _save_encoder,
    pretrain,
)

# ---------------------------------------------------------------------------
# Settings fixtures
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------


class TestLearningRateSchedule:
    def test_warmup_linear(self) -> None:
        config = _make_config(learning_rate=0.1, warmup_epochs=10, max_epochs=100)
        dummy_optimizer = torch.optim.SGD([torch.zeros(1)], lr=0.0)
        for step in range(1, 11):
            lr = _adjust_lr(dummy_optimizer, config, float(step))
            expected = 0.1 * step / 10
            assert lr == pytest.approx(expected, abs=1e-7)

    def test_cosine_decay_at_end(self) -> None:
        config = _make_config(learning_rate=0.1, warmup_epochs=10, max_epochs=100)
        dummy_optimizer = torch.optim.SGD([torch.zeros(1)], lr=0.0)
        lr = _adjust_lr(dummy_optimizer, config, 100.0)
        assert lr == pytest.approx(0.0, abs=1e-7)

    def test_cosine_peak(self) -> None:
        config = _make_config(learning_rate=0.1, warmup_epochs=0, max_epochs=100)
        dummy_optimizer = torch.optim.SGD([torch.zeros(1)], lr=0.0)
        lr = _adjust_lr(dummy_optimizer, config, 0.0)
        assert lr == pytest.approx(0.1, abs=1e-7)

    def test_lr_set_on_param_groups(self) -> None:
        config = _make_config(learning_rate=0.05, warmup_epochs=5, max_epochs=20)
        dummy_optimizer = torch.optim.SGD([torch.zeros(1)], lr=0.0)
        lr = _adjust_lr(dummy_optimizer, config, 2.5)
        assert dummy_optimizer.param_groups[0]["lr"] == lr


# ---------------------------------------------------------------------------
# Momentum schedule
# ---------------------------------------------------------------------------


class TestMomentumSchedule:
    def test_starts_at_base(self) -> None:
        config = _make_config(momentum_base=0.99)
        m = _adjust_momentum(config, 0.0)
        assert m == pytest.approx(0.99, abs=1e-6)

    def test_ends_at_one(self) -> None:
        config = _make_config(momentum_base=0.99)
        m = _adjust_momentum(config, 1.0)
        assert m == pytest.approx(1.0, abs=1e-6)

    def test_monotonically_increases(self) -> None:
        config = _make_config(momentum_base=0.99)
        values = [_adjust_momentum(config, t) for t in [i / 10 for i in range(11)]]
        for i in range(len(values) - 1):
            assert values[i] <= values[i + 1] + 1e-7

    def test_no_oscillation_over_many_epochs(self) -> None:
        config = _make_config(momentum_base=0.99, max_epochs=300)
        t_values = [e / 300 for e in range(301)]
        values = [_adjust_momentum(config, t) for t in t_values]
        for i in range(len(values) - 1):
            assert values[i] <= values[i + 1] + 1e-7


# ---------------------------------------------------------------------------
# Lambda_s schedule
# ---------------------------------------------------------------------------


class TestLambdaSSchedule:
    def test_starts_at_lambda_s(self) -> None:
        config = _make_config(lambda_s=10.0)
        v = _adjust_lambda_s(config, 0.0)
        assert v == pytest.approx(10.0, abs=1e-6)

    def test_decays_to_zero(self) -> None:
        config = _make_config(lambda_s=10.0)
        v = _adjust_lambda_s(config, 1.0)
        assert v == pytest.approx(0.0, abs=1e-6)

    def test_no_oscillation_over_many_epochs(self) -> None:
        config = _make_config(lambda_s=10.0, max_epochs=300)
        t_values = [e / 300 for e in range(301)]
        values = [_adjust_lambda_s(config, t) for t in t_values]
        for i in range(len(values) - 1):
            assert values[i] >= values[i + 1] - 1e-7


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


class TestCheckpointing:
    def test_save_and_load_checkpoint(self, tmp_path: Path) -> None:
        config = _make_config()
        model = Pretrainer(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        ckpt_path = tmp_path / "checkpoint.pt"
        _save_checkpoint(ckpt_path, 5, model, optimizer, None)

        loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert loaded["epoch"] == 5
        assert "state_dict" in loaded
        assert "optimizer" in loaded

    def test_save_encoder(self, tmp_path: Path) -> None:
        config = _make_config()
        model = Pretrainer(config)
        enc_path = tmp_path / "epoch_5_encoder.pt"
        _save_encoder(enc_path, model)

        state = torch.load(enc_path, map_location="cpu", weights_only=True)
        assert any(k.startswith("patch_embed") for k in state)
        assert not any(k.startswith("momentum") for k in state)


# ---------------------------------------------------------------------------
# Integration: full loop with tiny data
# ---------------------------------------------------------------------------


class TestPretrainIntegration:
    def test_runs_without_error(self, tmp_path: Path) -> None:
        config = _make_config(
            max_epochs=2,
            save_every=2,
            checkpoint_dir=tmp_path / "ckpts",
            log_dir=tmp_path / "logs",
        )
        model = Pretrainer(config)
        B, C, H, W = 4, 3, 16, 16
        x1 = torch.randn(B, C, H, W)
        x2 = torch.randn(B, C, H, W)
        m1 = torch.rand(B, 1, H, W)
        m2 = torch.rand(B, 1, H, W)
        ds = TensorDataset(x1, x2, m1, m2)
        dl = DataLoader(ds, batch_size=4)

        pretrain(model, dl, config)

        assert (tmp_path / "ckpts" / config.model_name / "checkpoint.pt").exists()
        assert (tmp_path / "ckpts" / config.model_name / f"epoch_{config.max_epochs}_encoder.pt").exists()

    def test_lambda_s_decay_flag(self, tmp_path: Path) -> None:
        config = _make_config(
            max_epochs=2,
            save_every=100,
            checkpoint_dir=tmp_path / "ckpts",
            log_dir=tmp_path / "logs",
            ss_decay=True,
            lambda_s=5.0,
        )
        model = Pretrainer(config)
        B, C, H, W = 4, 3, 16, 16
        ds = TensorDataset(
            torch.randn(B, C, H, W),
            torch.randn(B, C, H, W),
            torch.rand(B, 1, H, W),
            torch.rand(B, 1, H, W),
        )
        dl = DataLoader(ds, batch_size=4)
        pretrain(model, dl, config)

    def test_resume_from_checkpoint(self, tmp_path: Path) -> None:
        config = _make_config(
            max_epochs=1,
            save_every=100,
            checkpoint_dir=tmp_path / "ckpts",
            log_dir=tmp_path / "logs",
        )
        model = Pretrainer(config)
        B, C, H, W = 4, 3, 16, 16
        ds = TensorDataset(
            torch.randn(B, C, H, W),
            torch.randn(B, C, H, W),
            torch.rand(B, 1, H, W),
            torch.rand(B, 1, H, W),
        )
        dl = DataLoader(ds, batch_size=4)
        pretrain(model, dl, config)

        ckpt_path = tmp_path / "ckpts" / config.model_name / "checkpoint.pt"
        assert ckpt_path.exists()

        config2 = _make_config(
            max_epochs=2,
            save_every=100,
            checkpoint_dir=tmp_path / "ckpts2",
            log_dir=tmp_path / "logs2",
        )
        model2 = Pretrainer(config2)
        pretrain(model2, dl, config2, resume_path=ckpt_path)
