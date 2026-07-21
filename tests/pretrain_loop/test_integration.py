from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from dr_model.model.pretrain import Pretrainer
from dr_model.training.pretrain_loop import pretrain

from .conftest import _make_config


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
