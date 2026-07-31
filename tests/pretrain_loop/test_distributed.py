"""Tests for distributed training — DDP bootstrap, sampling, and persistence."""

from __future__ import annotations

import os
import socket
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, RandomSampler, TensorDataset

from dr_model.config import Settings
from dr_model.data.pretrain_datamodule import PretrainDataModule
from dr_model.model.pretrain import Pretrainer
from dr_model.training.distributed import (
    cleanup_distributed,
    detect_distributed_context,
    init_distributed,
    wrap_model,
)
from dr_model.training.pretrain_loop import (
    _finalize_training,
    _log_epoch,
    _save_checkpoint,
    _save_encoder,
    pretrain,
)
from dr_model.utils.timer import Timer

from .conftest import _make_config


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _ddp_config_dict(checkpoint_dir: str) -> dict[str, object]:
    return {
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
        "max_epochs": 1,
        "warmup_epochs": 1,
        "momentum_base": 0.99,
        "momentum_max": 1.0,
        "lambda_c": 1.0,
        "lambda_s": 10.0,
        "ss_decay": False,
        "save_every": 100,
        "gradient_clip_val": 1.0,
        "precision": "32-true",
        "input_size": 16,
        "image_size": (16, 16),
        "num_workers": 0,
        "seed": -1,
        "checkpoint_dir": checkpoint_dir,
        "log_dir": checkpoint_dir,
        "mlflow": False,
        "tensorboard": False,
    }


class TestDetectDistributedContext:
    def test_no_torchrun_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.delenv("LOCAL_RANK", raising=False)
        monkeypatch.delenv("WORLD_SIZE", raising=False)

        ctx = detect_distributed_context("cpu")

        assert not ctx.enabled
        assert ctx.rank == 0
        assert ctx.world_size == 1

    def test_torchrun_env_cpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RANK", "1")
        monkeypatch.setenv("LOCAL_RANK", "1")
        monkeypatch.setenv("WORLD_SIZE", "2")

        ctx = detect_distributed_context("cpu")

        assert ctx.enabled
        assert ctx.rank == 1
        assert ctx.local_rank == 1
        assert ctx.world_size == 2
        assert ctx.device == torch.device("cpu")

    def test_torchrun_env_auto_pins_to_cuda_or_cpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "1")

        ctx = detect_distributed_context("auto")

        assert ctx.enabled
        if torch.cuda.is_available():
            assert ctx.device == torch.device("cuda:0")
        else:
            assert ctx.device.type in ("cpu", "mps")

    def test_init_and_cleanup_idempotent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        port = _free_port()
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "1")
        monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
        monkeypatch.setenv("MASTER_PORT", str(port))

        ctx = detect_distributed_context("cpu")
        init_distributed(ctx)
        try:
            assert torch.distributed.is_initialized()
        finally:
            cleanup_distributed()
        cleanup_distributed()  # second call is a no-op


class _DummyDataset(Dataset):
    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.randn(3, 16, 16),
            torch.randn(3, 16, 16),
            torch.rand(1, 16, 16),
            torch.rand(1, 16, 16),
        )


class TestDistributedSamplerBoundary:
    def test_uses_sampler_when_initialized(self) -> None:
        port = _free_port()
        torch.distributed.init_process_group("gloo", rank=0, world_size=1, init_method=f"tcp://127.0.0.1:{port}")
        try:
            dm = PretrainDataModule(_make_config(num_workers=0))
            dm.dataset = _DummyDataset()  # type: ignore[assignment]
            dl = dm.train_dataloader()
        finally:
            torch.distributed.destroy_process_group()

        assert isinstance(dm.sampler, DistributedSampler)
        assert dl.sampler is dm.sampler

    def test_unchanged_when_single_process(self) -> None:
        assert not torch.distributed.is_initialized()

        dm = PretrainDataModule(_make_config(num_workers=0))
        dm.dataset = _DummyDataset()  # type: ignore[assignment]
        dl = dm.train_dataloader()

        assert dm.sampler is None
        assert isinstance(dl.sampler, RandomSampler)


class _Wrapped(torch.nn.Module):
    """Stands in for ``DistributedDataParallel`` so ``unwrap_model`` is exercised."""

    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module


class TestCheckpointFormat:
    """DDP save paths must never emit ``module.``-prefixed keys."""

    def test_checkpoint_no_module_prefix_when_wrapped(self, tmp_path: Path) -> None:
        config = _make_config()
        model = Pretrainer(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        ckpt = tmp_path / "checkpoint.pt"

        _save_checkpoint(ckpt, 3, _Wrapped(model), optimizer, None)

        state = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
        assert not any(k.startswith("module.") for k in state)

        fresh = Pretrainer(config)
        fresh.load_state_dict(state)  # existing resume path loads it unchanged

    def test_encoder_no_module_prefix_when_wrapped(self, tmp_path: Path) -> None:
        config = _make_config()
        model = Pretrainer(config)
        enc = tmp_path / "epoch_3_encoder.pt"

        _save_encoder(enc, _Wrapped(model))

        state = torch.load(enc, map_location="cpu", weights_only=True)
        assert not any(k.startswith("module.") for k in state)
        assert any(k.startswith("patch_embed") for k in state)


class TestRankGating:
    def test_log_epoch_noop_on_nonzero_rank(self, monkeypatch: pytest.MonkeyPatch) -> None:
        writer = MagicMock()
        mlflow_calls: list[object] = []
        monkeypatch.setattr("mlflow.log_metrics", lambda *a, **k: mlflow_calls.append(a))

        result = _log_epoch(
            epoch=0,
            config=Settings(mlflow=True),
            avg_cl=1.0,
            avg_ss=2.0,
            lr=1e-4,
            moco_m=0.99,
            t_epoch=1.0,
            writer=writer,
            rank=1,
        )

        assert result is None
        writer.assert_not_called()
        assert mlflow_calls == []

    def test_finalize_noop_on_nonzero_rank(self, monkeypatch: pytest.MonkeyPatch) -> None:
        writer = MagicMock()
        mlflow_calls: list[object] = []
        monkeypatch.setattr("mlflow.log_metric", lambda *a, **k: mlflow_calls.append(a))

        result = _finalize_training(Timer(), writer, Settings(mlflow=True), rank=1)

        assert result is None
        writer.assert_not_called()
        assert mlflow_calls == []

    def test_log_epoch_still_runs_on_rank_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        logged: dict[str, float] = {}
        monkeypatch.setattr("mlflow.log_metrics", lambda m, step: logged.update(m))

        _log_epoch(
            epoch=0,
            config=Settings(mlflow=True, tensorboard=False),
            avg_cl=0.5,
            avg_ss=0.25,
            lr=1e-4,
            moco_m=0.99,
            t_epoch=1.0,
            writer=None,
            rank=0,
        )

        assert "loss/contrastive" in logged


def _ddp_worker(rank: int, world_size: int, config_dict: dict[str, object], port: int) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)

    config = Settings(**config_dict)  # type: ignore[arg-type]
    ctx = detect_distributed_context("cpu")
    init_distributed(ctx)
    try:
        model = wrap_model(Pretrainer(config), ctx)
        ds = TensorDataset(
            torch.randn(32, 3, 16, 16),
            torch.randn(32, 3, 16, 16),
            torch.rand(32, 1, 16, 16),
            torch.rand(32, 1, 16, 16),
        )
        sampler = DistributedSampler(ds, shuffle=True, drop_last=True)
        dl = DataLoader(ds, batch_size=config.batch_size, sampler=sampler, drop_last=True)

        pretrain(
            model=model,
            train_dataloader=dl,
            config=config,
            device=torch.device("cpu"),
            writer=None,
            rank=ctx.rank,
            world_size=ctx.world_size,
            sampler=sampler,
        )
    finally:
        cleanup_distributed()


class TestDdpSmoke:
    """Real 2-process DDP on the gloo backend exercises actual gradient sync."""

    def test_pretrain_runs_and_checkpoints_are_clean(self, tmp_path: Path) -> None:
        world_size = 2
        port = _free_port()
        config_dict = _ddp_config_dict(str(tmp_path / "ckpts"))

        torch.multiprocessing.start_processes(
            _ddp_worker,
            args=(world_size, config_dict, port),
            nprocs=world_size,
            join=True,
        )

        config = Settings(**config_dict)  # type: ignore[arg-type]
        save_dir = tmp_path / "ckpts" / config.model_name
        ckpt = save_dir / "checkpoint.pt"
        assert ckpt.exists()

        state = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
        assert not any(k.startswith("module.") for k in state)

        encoder = save_dir / f"epoch_{config.max_epochs}_encoder.pt"
        assert encoder.exists()
