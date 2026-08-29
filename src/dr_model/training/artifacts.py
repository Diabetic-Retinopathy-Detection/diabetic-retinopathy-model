"""Write fine-tuning evaluation data to an external artifact directory.

Artifacts live outside the repository (``config.artifact_dir`` defaults to
``../artifacts``) so reports can consume stable JSON/CSV files without
parsing TensorBoard event files or an MLflow backend.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dr_model.config import Settings
from dr_model.training.metrics import ClassificationMetrics, EvaluationResult

_HISTORY_MISSING = ""


def validation_history_row(
    epoch: int,
    train_loss: float,
    lr: float,
    metrics: ClassificationMetrics,
) -> dict[str, Any]:
    """Flatten one epoch's training context and validation metrics into a row."""
    row: dict[str, Any] = {
        "epoch": epoch,
        "train_loss": train_loss,
        "lr": lr,
        "val_loss": metrics.loss,
        "kappa": metrics.kappa,
        "accuracy": metrics.accuracy,
        "f1_macro": metrics.f1_macro,
        "f1_weighted": metrics.f1_weighted,
        "auc_macro": _defined_or_missing(metrics.auc_macro),
        "auc_weighted": _defined_or_missing(metrics.auc_weighted),
    }
    for index, value in enumerate(metrics.recall_per_class):
        row[f"recall_class_{index}"] = value
    for index, value in enumerate(metrics.f1_per_class):
        row[f"f1_class_{index}"] = value
    for index, auc_value in enumerate(metrics.auc_per_class):
        row[f"auc_class_{index}"] = _defined_or_missing(auc_value)
    return row


def write_finetune_artifacts(
    root: Path,
    config: Settings,
    validation_history: list[dict[str, Any]],
    validation_result: EvaluationResult | None,
    test_result: EvaluationResult | None,
    *,
    run_name: str,
) -> Path:
    """Write resolved configuration and evaluation results outside the repo.

    The run directory is suffixed with a UTC timestamp so repeated runs never
    overwrite earlier artifacts.
    """
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = root / f"{run_name}_{run_id}"
    suffix = 1
    while output.exists():
        output = root / f"{run_name}_{run_id}_{suffix}"
        suffix += 1
    output.mkdir(parents=True)

    _write_json(output / "config.json", _jsonable(config.model_dump()))
    run_info = {"run_name": run_name, "run_id": run_id, "created_at": datetime.now(timezone.utc).isoformat()}
    _write_json(output / "run.json", run_info)
    _write_csv(output / "validation_history.csv", validation_history)
    if validation_result is not None:
        _write_split(output, "validation", validation_result)
    if test_result is not None:
        _write_split(output, "test", test_result)
    return output


def _write_split(output: Path, split: str, result: EvaluationResult) -> None:
    """Write aggregate metrics, confusion matrix, and raw outputs for a split."""
    values = asdict(result.metrics)
    matrix = values.pop("confusion_matrix")
    _write_json(output / f"{split}_metrics.json", _jsonable(values))
    _write_csv(
        output / f"{split}_confusion_matrix.csv",
        [
            {"true_class": row, **{f"predicted_{index}": value for index, value in enumerate(cells)}}
            for row, cells in enumerate(matrix)
        ],
    )
    _write_csv(output / f"{split}_labels.csv", [{"label": label} for label in result.labels])
    _write_csv(output / f"{split}_predictions.csv", [{"prediction": prediction} for prediction in result.predictions])
    _write_csv(
        output / f"{split}_probabilities.csv",
        [
            {f"class_{index}": value for index, value in enumerate(probabilities)}
            for probabilities in result.probabilities
        ],
    )


def _defined_or_missing(value: float | None) -> Any:
    return _HISTORY_MISSING if value is None else value


def _write_json(path: Path, values: Any) -> None:
    path.write_text(json.dumps(values, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
