"""Write fine-tuning evaluation data to an external artifact directory."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dr_model.config import Settings
from dr_model.training.metrics import ClassificationMetrics


def write_finetune_artifacts(
    root: Path,
    config: Settings,
    validation_history: list[dict[str, float]],
    validation_metrics: ClassificationMetrics | None,
    test_metrics: ClassificationMetrics | None,
    *,
    run_name: str,
) -> Path:
    """Write resolved configuration and evaluation metrics outside the repo."""
    output = root / run_name
    output.mkdir(parents=True, exist_ok=True)

    _write_json(output / "config.json", _jsonable(config.model_dump()))
    _write_json(output / "run.json", {"run_name": run_name, "created_at": datetime.now(timezone.utc).isoformat()})
    _write_csv(output / "validation_history.csv", validation_history)
    if validation_metrics is not None:
        _write_metrics(output, "validation", validation_metrics)
    if test_metrics is not None:
        _write_metrics(output, "test", test_metrics)
    return output


def _write_metrics(output: Path, split: str, metrics: ClassificationMetrics) -> None:
    values = asdict(metrics)
    matrix = values.pop("confusion_matrix")
    _write_json(output / f"{split}_metrics.json", _jsonable(values))
    _write_csv(
        output / f"{split}_confusion_matrix.csv",
        [
            {"true_class": row, **{f"predicted_{index}": value for index, value in enumerate(values)}}
            for row, values in enumerate(matrix)
        ],
    )


def _write_json(path: Path, values: Any) -> None:
    path.write_text(json.dumps(values, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
