from __future__ import annotations

from pathlib import Path

import torch

from dr_model.model.pretrain import Pretrainer
from dr_model.training.pretrain_loop import _save_checkpoint, _save_encoder

from .conftest import _make_config


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
