from __future__ import annotations

import argparse
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

from dr_model.config import Settings, _load_yaml_config
from dr_model.data.finetune_datamodule import FinetuneDataModule
from dr_model.model.backbone import ViTBackbone
from dr_model.model.finetune import Finetuner
from dr_model.training.cli import _collect_updates
from dr_model.training.cli import main as cli_main
from dr_model.training.finetune_loop import (
    SquaredCDFLoss,
    SquaredWassersteinLoss,
    adjust_lr,
    build_finetune_criterion,
    run,
)
from dr_model.training.metrics import calculate_classification_metrics

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
    def test_direct_settings_defaults_match_ddr(self) -> None:
        config = Settings()

        assert config.finetune_mean == [0.423737496137619, 0.2609460651874542, 0.128403902053833]
        assert config.finetune_std == [0.29482534527778625, 0.20167365670204163, 0.13668020069599152]

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


class TestFinetuneLoss:
    def test_squared_wasserstein_matches_expected_transport_cost(self) -> None:
        logits = torch.log(torch.tensor([[0.0, 0.5, 0.0, 0.5, 0.0]], dtype=torch.float32) + 1e-8)
        labels = torch.tensor([2])

        loss = SquaredWassersteinLoss()(logits, labels)

        assert loss == pytest.approx(1.0, abs=1e-6)

    def test_squared_cdf_matches_cumulative_distribution_formula(self) -> None:
        logits = torch.log(torch.tensor([[0.1, 0.2, 0.7]], dtype=torch.float32))
        labels = torch.tensor([1])

        loss = SquaredCDFLoss()(logits, labels)
        expected = ((torch.tensor([0.1, 0.3, 1.0]) - torch.tensor([0.0, 1.0, 1.0])) ** 2).mean()

        assert loss == pytest.approx(expected.item())

    def test_ordinal_losses_have_finite_gradients(self) -> None:
        logits = torch.randn(2, 5, requires_grad=True)

        SquaredWassersteinLoss()(logits, torch.tensor([0, 4])).backward()

        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()

    def test_default_and_alternative_criteria(self) -> None:
        config = _settings(None)

        assert isinstance(build_finetune_criterion(config), SquaredWassersteinLoss)
        assert isinstance(
            build_finetune_criterion(config.model_copy(update={"finetune_loss": "squared_cdf"})),
            SquaredCDFLoss,
        )
        assert isinstance(
            build_finetune_criterion(config.model_copy(update={"finetune_loss": "cross_entropy"})),
            nn.CrossEntropyLoss,
        )

    def test_unknown_criterion_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported finetune_loss"):
            build_finetune_criterion(_settings(None).model_copy(update={"finetune_loss": "unknown"}))


class TestClassificationMetrics:
    def test_calculates_aggregate_and_per_class_metrics(self) -> None:
        labels = [0, 1, 2, 3, 4]
        probabilities = torch.eye(5).tolist()
        metrics = calculate_classification_metrics(labels, labels, probabilities, loss=0.25, num_classes=5)

        assert metrics.loss == 0.25
        assert metrics.kappa == pytest.approx(1.0)
        assert metrics.accuracy == pytest.approx(1.0)
        assert metrics.f1_macro == pytest.approx(1.0)
        assert metrics.f1_weighted == pytest.approx(1.0)
        assert metrics.auc_macro == pytest.approx(1.0)
        assert metrics.auc_weighted == pytest.approx(1.0)
        assert metrics.f1_per_class == pytest.approx([1.0] * 5)
        assert metrics.auc_per_class == pytest.approx([1.0] * 5)
        assert metrics.recall_per_class == pytest.approx([1.0] * 5)
        assert metrics.confusion_matrix == np.eye(5, dtype=int).tolist()

    def test_missing_class_keeps_fixed_arrays_and_skips_auc(self) -> None:
        labels = [0, 0, 1, 1]
        predictions = [0, 1, 1, 1]
        probabilities = [
            [0.8, 0.1, 0.05, 0.03, 0.02],
            [0.1, 0.7, 0.1, 0.06, 0.04],
            [0.1, 0.8, 0.04, 0.03, 0.03],
            [0.05, 0.85, 0.04, 0.03, 0.03],
        ]

        metrics = calculate_classification_metrics(labels, predictions, probabilities, loss=1.0, num_classes=5)

        assert len(metrics.recall_per_class) == 5
        assert len(metrics.confusion_matrix) == 5
        assert metrics.auc_macro is None
        assert metrics.auc_weighted is None
        assert metrics.auc_per_class[0] is not None
        assert metrics.auc_per_class[2:] == [None, None, None]


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
        assert {
            "loss/train",
            "loss/val",
            "kappa/val",
            "accuracy/val",
            "f1/macro/val",
            "f1/weighted/val",
            "recall/class_0/val",
            "recall/class_4/val",
            "lr",
        } <= tags
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

    def test_train_on_train_and_valid_without_validation(self, tmp_path: Path) -> None:
        root = _make_dataset_root(tmp_path)
        config = _settings(root, batch_size=2, checkpoint_dir=tmp_path / "ckpts", finetune_epochs=1)
        config = config.model_copy(update={"train_on_train_and_valid": True, "skip_validation": True})
        dm = FinetuneDataModule(config)
        dm.setup()
        assert dm.train_dataset is not None
        assert len(dm.train_dataset) == 20
        assert dm.val_dataset is None

        writer = _FakeWriter()
        run(
            config=config,
            model=Finetuner(config),
            datamodule=dm,
            device=torch.device("cpu"),
            writer=writer,  # type: ignore[arg-type]
        )

        tags = {tag for tag, _value, _step in writer.scalars}
        assert {"loss/train", "lr"} <= tags
        assert "loss/val" not in tags
        assert not (config.checkpoint_dir / "finetune" / "best_validation_weights.pt").exists()
        assert (config.checkpoint_dir / "finetune" / "epoch_1.pt").exists()

    def test_resume_for_additional_epochs(self, tmp_path: Path) -> None:
        root = _make_dataset_root(tmp_path)
        checkpoint_dir = tmp_path / "ckpts"

        first_config = _settings(root, batch_size=2, checkpoint_dir=checkpoint_dir, finetune_epochs=1)
        first_dm = FinetuneDataModule(first_config)
        first_dm.setup()
        run(
            config=first_config,
            model=Finetuner(first_config),
            datamodule=first_dm,
            device=torch.device("cpu"),
        )

        resume_path = checkpoint_dir / "finetune" / "epoch_1.pt"
        assert resume_path.exists()
        legacy_state = torch.load(resume_path, map_location="cpu")
        legacy_state.pop("total_epochs", None)
        legacy_state.pop("kappa", None)
        torch.save(legacy_state, resume_path)

        resumed_config = _settings(root, batch_size=2, checkpoint_dir=checkpoint_dir, finetune_epochs=25)
        resumed_dm = FinetuneDataModule(resumed_config)
        resumed_dm.setup()
        run(
            config=resumed_config,
            model=Finetuner(resumed_config),
            datamodule=resumed_dm,
            device=torch.device("cpu"),
            resume_path=resume_path,
            extra_epochs=1,
        )

        final_path = checkpoint_dir / "finetune" / "epoch_2.pt"
        assert final_path.exists()
        state = torch.load(final_path, map_location="cpu")
        assert state["epoch"] == 1
        assert state["total_epochs"] == 25


class TestScript:
    def test_cli_finetune_overrides(self) -> None:
        args = argparse.Namespace(
            seed=None,
            deterministic=None,
            data_index_path=None,
            data_dir=None,
            finetune_epochs=None,
            batch_size=None,
            finetune_checkpoint=None,
            finetune_checkpoint_dir="checkpoints/wasserstein",
            finetune_loss="squared_wasserstein",
            num_workers=None,
            train_on_train_and_valid=None,
            skip_validation=None,
        )

        assert _collect_updates(args) == {
            "checkpoint_dir": Path("checkpoints/wasserstein"),
            "finetune_loss": "squared_wasserstein",
        }

    def test_yaml_overlay_overrides_base(self, tmp_path: Path) -> None:
        base = tmp_path / "base.yaml"
        overlay = tmp_path / "overlay.yaml"
        base.write_text("finetune_loss: squared_wasserstein\nnum_classes: 5\n")
        overlay.write_text("_base_: base.yaml\nfinetune_loss: cross_entropy\n")

        values = _load_yaml_config(overlay)

        assert values == {"finetune_loss": "cross_entropy", "num_classes": 5}

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
