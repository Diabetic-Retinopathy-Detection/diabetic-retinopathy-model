from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from dr_model.config import Settings
from dr_model.model.finetune import Finetuner
from dr_model.training.evaluate import main

GRADES = (0, 1, 2, 3, 4)
IMAGES_PER_GRADE = 2


def _make_split_tree(root: Path) -> Path:
    for grade in GRADES:
        grade_dir = root / str(grade)
        grade_dir.mkdir(parents=True, exist_ok=True)
        for i in range(IMAGES_PER_GRADE):
            arr = np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8)
            Image.fromarray(arr).save(grade_dir / f"img_{grade}_{i}.jpeg")
    return root


def _make_dataset_root(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    for split in ("train", "val", "test"):
        _make_split_tree(root / split)
    return root


def _config_dict(tmp_path: Path, dataset_root: Path) -> dict[str, object]:
    return {
        "patch_size": 16,
        "embed_dim": 32,
        "depth": 2,
        "num_heads": 4,
        "mlp_ratio": 4.0,
        "num_classes": 5,
        "image_size": (224, 224),
        "finetune_input_size": 64,
        "batch_size": 4,
        "num_workers": 0,
        "seed": -1,
        "mlflow": False,
        "tensorboard": False,
        "checkpoint_dir": str(tmp_path / "ckpts"),
        "artifact_dir": str(tmp_path / "artifacts"),
        "finetune_dataset_root": str(dataset_root),
    }


def _make_checkpoint(tmp_path: Path, config: Settings) -> Path:
    """Save a checkpoint in the exact format produced by the training loop."""
    checkpoint_path = tmp_path / "ckpts" / "finetune" / "MyLossRun"
    checkpoint_path.mkdir(parents=True)
    path = checkpoint_path / "best_validation_weights.pt"
    model = Finetuner(config)
    torch.save({"epoch": 0, "state_dict": model.state_dict(), "optimizer": {}}, path)
    return path


class TestEvaluate:
    def test_evaluate_validation_writes_artifacts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.delenv("DR_CONFIG_FILE", raising=False)
        dataset_root = _make_dataset_root(tmp_path)
        config_dict = _config_dict(tmp_path, dataset_root)
        config_path = tmp_path / "eval_config.yaml"
        config_path.write_text(yaml.safe_dump(config_dict))

        config = Settings(**config_dict)  # type: ignore[arg-type]
        checkpoint = _make_checkpoint(tmp_path, config)

        code = main([
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint),
            "--split",
            "validation",
            "--run-name",
            "my_loss_run",
            "--device",
            "cpu",
        ])

        assert code == 0
        out = capsys.readouterr().out
        assert "validation: kappa=" in out
        assert "Artifacts written to" in out

        outputs = list((tmp_path / "artifacts").iterdir())
        assert len(outputs) == 1
        output = outputs[0]
        assert output.name.startswith("my_loss_run_")

        expected_files = (
            "config.json",
            "run.json",
            "validation_history.csv",
            "validation_metrics.json",
            "validation_confusion_matrix.csv",
            "validation_labels.csv",
            "validation_predictions.csv",
            "validation_probabilities.csv",
        )
        for name in expected_files:
            assert (output / name).exists(), name
        assert not (output / "test_metrics.json").exists()

        n_val = IMAGES_PER_GRADE * len(GRADES)
        with (output / "validation_labels.csv").open() as stream:
            assert sum(1 for _ in csv.DictReader(stream)) == n_val
        metrics_json = json.loads((output / "validation_metrics.json").read_text())
        assert math.isfinite(metrics_json["kappa"])

    def test_default_run_name_from_checkpoint_parent(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DR_CONFIG_FILE", raising=False)
        dataset_root = _make_dataset_root(tmp_path)
        config_dict = _config_dict(tmp_path, dataset_root)
        config_path = tmp_path / "eval_config.yaml"
        config_path.write_text(yaml.safe_dump(config_dict))
        config = Settings(**config_dict)  # type: ignore[arg-type]
        checkpoint = _make_checkpoint(tmp_path, config)

        code = main([
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint),
            "--device",
            "cpu",
        ])

        assert code == 0
        output = next((tmp_path / "artifacts").iterdir())
        assert output.name.startswith("mylossrun_")
