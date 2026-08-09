"""Tests for distributed fine-tuning — DDP bootstrap, sampling, and persistence."""

from __future__ import annotations

import os
import socket
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from torch.utils.data import Dataset, DistributedSampler, RandomSampler, TensorDataset

from dr_model.config import Settings
from dr_model.data.finetune_datamodule import FinetuneDataModule
from dr_model.model.finetune import Finetuner
from dr_model.training.distributed import (
    cleanup_distributed,
    detect_distributed_context,
    init_distributed,
    wrap_model,
)
from dr_model.training.finetune_loop import _log_epoch, _save_checkpoint, run


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _finetune_config_dict(checkpoint_dir: str) -> dict[str, object]:
    return {
        "patch_size": 16,
        "embed_dim": 32,
        "depth": 2,
        "num_heads": 4,
        "mlp_ratio": 4.0,
        "num_classes": 5,
        "image_size": (224, 224),
        "finetune_input_size": 384,
        "batch_size": 4,
        "num_workers": 0,
        "seed": -1,
        "finetune_lr": 1e-3,
        "finetune_min_lr": 0.0,
        "finetune_weight_decay": 0.0,
        "finetune_warmup_epochs": 1,
        "finetune_epochs": 1,
        "save_every": 100,
        "gradient_clip_val": 1.0,
        "precision": "32-true",
        "checkpoint_dir": checkpoint_dir,
        "log_dir": checkpoint_dir,
        "pin_memory": False,
        "mlflow": False,
        "tensorboard": False,
    }


class _DummyDataset(Dataset):
    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return torch.randn(3, 384, 384), index % 5


class TestDistributedSamplerBoundary:
    def test_uses_sampler_when_initialized(self) -> None:
        port = _free_port()
        torch.distributed.init_process_group("gloo", rank=0, world_size=1, init_method=f"tcp://127.0.0.1:{port}")
        try:
            dm = FinetuneDataModule(Settings(batch_size=4, num_workers=0, seed=-1))
            dm.train_dataset = _DummyDataset()  # type: ignore[assignment]
            dl = dm.train_dataloader()
        finally:
            torch.distributed.destroy_process_group()

        assert isinstance(dm.sampler, DistributedSampler)
        assert dl.sampler is dm.sampler

    def test_unchanged_when_single_process(self) -> None:
        assert not torch.distributed.is_initialized()

        dm = FinetuneDataModule(Settings(batch_size=4, num_workers=0, seed=-1))
        dm.train_dataset = _DummyDataset()  # type: ignore[assignment]
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
        config = Settings(**{**_finetune_config_dict(str(tmp_path / "ckpts"))})  # type: ignore[arg-type]
        model = Finetuner(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        ckpt = tmp_path / "checkpoint.pt"

        _save_checkpoint(ckpt, 3, _Wrapped(model), optimizer, None)

        state = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
        assert not any(k.startswith("module.") for k in state)

        fresh = Finetuner(config)
        fresh.load_state_dict(state)  # existing resume path loads it unchanged


class TestRankGating:
    def test_log_epoch_noop_on_nonzero_rank(self, monkeypatch: pytest.MonkeyPatch) -> None:
        writer = MagicMock()
        mlflow_calls: list[object] = []
        monkeypatch.setattr("mlflow.log_metrics", lambda *a, **k: mlflow_calls.append(a))

        result = _log_epoch(
            epoch=0,
            config=Settings(mlflow=True),
            train_loss=1.0,
            val_loss=2.0,
            kappa=0.5,
            lr=1e-4,
            t_epoch=1.0,
            writer=writer,
            rank=1,
        )

        assert result is None
        writer.assert_not_called()
        assert mlflow_calls == []

    def test_log_epoch_still_runs_on_rank_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        logged: dict[str, float] = {}
        monkeypatch.setattr("mlflow.log_metrics", lambda m, step: logged.update(m))

        _log_epoch(
            epoch=0,
            config=Settings(mlflow=True, tensorboard=False),
            train_loss=0.5,
            val_loss=0.25,
            kappa=0.9,
            lr=1e-4,
            t_epoch=1.0,
            writer=None,
            rank=0,
        )

        assert "kappa/val" in logged


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
        model = wrap_model(Finetuner(config), ctx)
        ds = TensorDataset(torch.randn(32, 3, 384, 384), torch.randint(0, 5, (32,)))
        dm = FinetuneDataModule(config)
        dm.train_dataset = ds  # type: ignore[assignment]
        dm.val_dataset = ds  # type: ignore[assignment]

        run(
            config=config,
            model=model,
            datamodule=dm,
            device=torch.device("cpu"),
            writer=None,
            rank=ctx.rank,
            world_size=ctx.world_size,
        )
    finally:
        cleanup_distributed()


class TestDdpSmoke:
    """Real 2-process DDP on the gloo backend exercises actual gradient sync."""

    def test_finetune_runs_and_checkpoints_are_clean(self, tmp_path: Path) -> None:
        world_size = 2
        port = _free_port()
        config_dict = _finetune_config_dict(str(tmp_path / "ckpts"))

        torch.multiprocessing.start_processes(
            _ddp_worker,
            args=(world_size, config_dict, port),
            nprocs=world_size,
            join=True,
        )

        config = Settings(**config_dict)  # type: ignore[arg-type]
        save_dir = tmp_path / "ckpts" / "finetune"
        ckpt = save_dir / "epoch_1.pt"
        assert ckpt.exists()

        state = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
        assert not any(k.startswith("module.") for k in state)

        fresh = Finetuner(config)
        fresh.load_state_dict(state)
