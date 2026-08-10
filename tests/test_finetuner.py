from __future__ import annotations

import itertools
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
import torch.nn as nn
import yaml
from PIL import Image

from dr_model.config import Settings
from dr_model.data.finetune_datamodule import FinetuneDataModule
from dr_model.model.backbone import ViTBackbone
from dr_model.model.finetune import Finetuner
from dr_model.training.cli import main as cli_main
from dr_model.training.finetune_loop import adjust_lr, run

if TYPE_CHECKING:
    pass

GRADES = (0, 1, 2, 3, 4)


def _settings(
    root: Path | None,
    *,
    batch_size: int = 4,
    checkpoint_dir: Path | None = None,
    finetune_epochs: int = 2,
) -> Settings:
    return Settings(
        patch_size=16,
        embed_dim=32,
        depth=2,
        num_heads=4,
        image_size=(224, 224),
        num_classes=5,
        finetune_dataset_root=root,
        finetune_input_size=384,
        batch_size=batch_size,
        num_workers=0,
        seed=-1,
        mlflow=False,
        checkpoint_dir=checkpoint_dir or Path("checkpoints"),
        save_every=1,
        finetune_epochs=finetune_epochs,
    )


def _make_split_tree(root: Path, images_per_grade: int = 2) -> Path:
    for grade in GRADES:
        grade_dir = root / str(grade)
        grade_dir.mkdir(parents=True, exist_ok=True)
        for i in range(images_per_grade):
            arr = np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8)
            Image.fromarray(arr).save(grade_dir / f"img_{grade}_{i}.jpeg")
    return root


def _make_dataset_root(tmp_path: Path, images_per_grade: int = 2) -> Path:
    root = tmp_path / "dataset"
    for split in ("train", "val", "test"):
        _make_split_tree(root / split, images_per_grade)
    return root


def _source_encoder(config: Settings) -> ViTBackbone:
    return ViTBackbone(config, input_size=224)


class TestFinetuner:
    def test_forward_logits_shape(self) -> None:
        model = Finetuner(_settings(None))

        out = model(torch.randn(2, 3, 384, 384))

        assert out.shape == (2, 5)

    def test_forward_representation_with_identity_head(self) -> None:
        config = _settings(None)
        model = Finetuner(config)
        model.head = nn.Identity()  # type: ignore[assignment]

        out = model(torch.randn(2, 3, 384, 384))

        assert out.shape == (2, 2 * config.embed_dim)

    def test_trunk_built_at_finetune_input_size(self) -> None:
        config = _settings(None)
        model = Finetuner(config)

        assert model.backbone.pos_embed.shape[1] == 1 + (384 // config.patch_size) ** 2

    def test_loads_checkpoint_pt_format(self, tmp_path: Path) -> None:
        config = _settings(None)
        enc = _source_encoder(config)
        ckpt = {"state_dict": {"base_encoder." + k: v for k, v in enc.state_dict().items()}}
        torch.save(ckpt, tmp_path / "checkpoint.pt")

        model = Finetuner(config, checkpoint_path=str(tmp_path / "checkpoint.pt"))

        assert model.backbone.pos_embed.shape[1] == 577
        for key, value in model.backbone.state_dict().items():
            if key in ("pos_embed", "head.weight", "head.bias"):
                continue
            assert torch.equal(value, enc.state_dict()[key])

    def test_loads_bare_encoder_format(self, tmp_path: Path) -> None:
        config = _settings(None)
        enc = _source_encoder(config)
        torch.save(enc.state_dict(), tmp_path / "epoch_10_encoder.pt")

        model = Finetuner(config, checkpoint_path=str(tmp_path / "epoch_10_encoder.pt"))

        assert model.backbone.pos_embed.shape[1] == 577
        for key, value in model.backbone.state_dict().items():
            if key in ("pos_embed", "head.weight", "head.bias"):
                continue
            assert torch.equal(value, enc.state_dict()[key])

    def test_pos_embed_interpolated_to_finetune_resolution(self, tmp_path: Path) -> None:
        config = _settings(None)
        enc = _source_encoder(config)
        torch.save(enc.state_dict(), tmp_path / "encoder.pt")

        model = Finetuner(config, checkpoint_path=str(tmp_path / "encoder.pt"))

        assert model.backbone.pos_embed.shape == (1, 577, config.embed_dim)

    def test_missing_checkpoint_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            Finetuner(_settings(None), checkpoint_path=str(tmp_path / "nope.pt"))

    def test_checkpoint_with_wrong_keys_raises(self, tmp_path: Path) -> None:
        torch.save({"state_dict": {"head.0.weight": torch.zeros(3)}}, tmp_path / "bad.pt")
        with pytest.raises(RuntimeError, match="Unexpected missing keys"):
            Finetuner(_settings(None), checkpoint_path=str(tmp_path / "bad.pt"))

    def test_loads_finetune_checkpoint(self, tmp_path: Path) -> None:
        config = _settings(None)
        source = Finetuner(config)
        ckpt = {"epoch": 2, "state_dict": source.state_dict(), "optimizer": {}}
        torch.save(ckpt, tmp_path / "finetune.pt")

        model = Finetuner(config, checkpoint_path=str(tmp_path / "finetune.pt"))

        source_sd = source.state_dict()
        for key, value in model.state_dict().items():
            assert torch.equal(value, source_sd[key])

    def test_drc49_head_shape_mismatch_keeps_random_head(self, tmp_path: Path) -> None:
        source_config = _settings(None)
        source = Finetuner(source_config)
        torch.save({"epoch": 1, "state_dict": source.state_dict(), "optimizer": {}}, tmp_path / "finetune.pt")

        target_config = source_config.model_copy(update={"num_classes": 3})
        model = Finetuner(target_config, checkpoint_path=str(tmp_path / "finetune.pt"))

        source_backbone_sd = source.backbone.state_dict()
        for key, value in model.backbone.state_dict().items():
            assert torch.equal(value, source_backbone_sd[key])
        assert model.head.weight.shape == (3, 2 * target_config.embed_dim)
        assert not torch.equal(model.head.weight, source.head.weight)


class TestAdjustLr:
    def _opt(self, config: Settings) -> torch.optim.Optimizer:
        return torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=config.finetune_lr)

    def test_warmup_then_cosine_decay(self) -> None:
        config = _settings(None, finetune_epochs=10)
        config = config.model_copy(update={"finetune_warmup_epochs": 5, "finetune_min_lr": 0.0})
        opt = self._opt(config)

        ratios = [i * 0.5 for i in range(21)]
        lrs = [adjust_lr(opt, config, r) for r in ratios]

        assert lrs[0] == pytest.approx(0.0)
        assert all(b > a for a, b in itertools.pairwise(lrs[:11]))
        assert all(b < a for a, b in itertools.pairwise(lrs[10:]))
        assert lrs[-1] == pytest.approx(config.finetune_min_lr)

    def test_decays_to_min_lr(self) -> None:
        config = _settings(None, finetune_epochs=10)
        config = config.model_copy(update={"finetune_warmup_epochs": 2, "finetune_min_lr": 1e-4})
        opt = self._opt(config)

        lr = adjust_lr(opt, config, 10.0)

        assert lr == pytest.approx(1e-4)

    def test_peak_at_warmup_boundary(self) -> None:
        config = _settings(None, finetune_epochs=10)
        config = config.model_copy(update={"finetune_warmup_epochs": 5, "finetune_min_lr": 0.0})
        opt = self._opt(config)

        assert adjust_lr(opt, config, 5.0) == pytest.approx(config.finetune_lr)


class TestFinetuneDataModule:
    def test_setup_accepts_valid_split_dir(self, tmp_path: Path) -> None:
        root = tmp_path / "dataset"
        for split in ("train", "valid", "test"):
            _make_split_tree(root / split)
        dm = FinetuneDataModule(_settings(root, checkpoint_dir=tmp_path / "ckpts"))
        dm.setup()

        assert dm.val_dataset is not None
        assert dm.test_dataset is not None
        assert len(dm.val_dataset) == len(dm.test_dataset) == len(GRADES) * 2

    def test_setup_missing_val_and_valid_raises(self, tmp_path: Path) -> None:
        root = tmp_path / "dataset"
        for split in ("train", "test"):
            _make_split_tree(root / split)
        dm = FinetuneDataModule(_settings(root, checkpoint_dir=tmp_path / "ckpts"))

        with pytest.raises(FileNotFoundError, match="val"):
            dm.setup()


class _FakeWriter:
    def __init__(self) -> None:
        self.scalars: list[tuple[str, float, int]] = []

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self.scalars.append((tag, float(value), int(step)))


class TestRun:
    def test_run_end_to_end(self, tmp_path: Path) -> None:
        root = _make_dataset_root(tmp_path)
        config = _settings(root, batch_size=2, checkpoint_dir=tmp_path / "ckpts")
        dm = FinetuneDataModule(config)
        dm.setup()
        model = Finetuner(config)
        writer = _FakeWriter()

        run(config=config, model=model, datamodule=dm, device=torch.device("cpu"), writer=writer)  # type: ignore[arg-type]

        tags = {tag for tag, _value, _step in writer.scalars}
        assert {"loss/train", "loss/val", "kappa/val", "lr"} <= tags
        kappa = next(value for tag, value, _step in writer.scalars if tag == "kappa/val")
        assert math.isfinite(kappa)

        ckpt_dir = config.checkpoint_dir / "finetune"
        assert (ckpt_dir / "best_validation_weights.pt").exists()
        assert (ckpt_dir / "epoch_2.pt").exists()

    def test_run_without_writer(self, tmp_path: Path) -> None:
        root = _make_dataset_root(tmp_path)
        config = _settings(root, batch_size=2, checkpoint_dir=tmp_path / "ckpts")
        dm = FinetuneDataModule(config)
        dm.setup()
        model = Finetuner(config)

        run(config=config, model=model, datamodule=dm, device=torch.device("cpu"))

        assert (config.checkpoint_dir / "finetune" / "epoch_2.pt").exists()


class TestScript:
    def test_main_smoke(self, tmp_path: Path) -> None:
        root = _make_dataset_root(tmp_path)
        cfg_path = tmp_path / "finetune_smoke.yaml"
        cfg = {
            "patch_size": 16,
            "embed_dim": 32,
            "depth": 2,
            "num_heads": 4,
            "image_size": [224, 224],
            "num_classes": 5,
            "finetune_dataset_root": str(root),
            "finetune_input_size": 384,
            "batch_size": 2,
            "num_workers": 0,
            "seed": 0,
            "finetune_lr": 3.0e-4,
            "finetune_min_lr": 0.0,
            "finetune_weight_decay": 0.05,
            "finetune_warmup_epochs": 1,
            "finetune_epochs": 1,
            "checkpoint_dir": str(tmp_path / "ckpts"),
            "save_every": 1,
            "mlflow": False,
            "tensorboard": False,
        }
        with open(cfg_path, "w") as f:
            yaml.safe_dump(cfg, f)

        old = os.environ.pop("DR_CONFIG_FILE", None)
        try:
            rc = cli_main(
                argv=["--phase", "finetune", "--config", str(cfg_path), "--device", "cpu", "--num-workers", "0"]
            )
        finally:
            if old is None:
                os.environ.pop("DR_CONFIG_FILE", None)
            else:
                os.environ["DR_CONFIG_FILE"] = old

        assert rc == 0
        assert (tmp_path / "ckpts" / "finetune" / "epoch_1.pt").exists()
