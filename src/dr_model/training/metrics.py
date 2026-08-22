"""Classification metrics for ordinal diabetic-retinopathy grading."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class ClassificationMetrics:
    """Classification metrics calculated for one dataset split."""

    loss: float
    kappa: float
    accuracy: float
    f1_macro: float
    f1_weighted: float
    auc_macro: float | None
    auc_weighted: float | None
    f1_per_class: list[float]
    auc_per_class: list[float | None]
    recall_per_class: list[float]
    confusion_matrix: list[list[int]]


@dataclass(frozen=True)
class EvaluationResult:
    """Metrics plus raw predictions produced for one evaluated split."""

    metrics: ClassificationMetrics
    labels: list[int]
    predictions: list[int]
    probabilities: list[list[float]]


def calculate_classification_metrics(
    labels: list[int],
    predictions: list[int],
    probabilities: list[list[float]],
    loss: float,
    num_classes: int,
) -> ClassificationMetrics:
    """Calculate aggregate and per-class multiclass metrics.

    ROC-AUC uses one-vs-rest comparisons over the fixed class ordering. It is
    undefined for a class absent from the labels, so those values are ``None``.
    """
    class_labels = list(range(num_classes))
    y_true = np.asarray(labels)
    y_pred = np.asarray(predictions)
    y_prob = np.asarray(probabilities)
    y_true_one_hot = np.eye(num_classes, dtype=float)[y_true]

    auc_per_class = _per_class_auc(y_true_one_hot, y_prob)
    valid_auc = [value for value in auc_per_class if value is not None]
    auc_macro = float(np.mean(valid_auc)) if len(valid_auc) == num_classes else None
    auc_weighted = _multiclass_auc(y_true_one_hot, y_prob, "weighted") if len(valid_auc) == num_classes else None

    return ClassificationMetrics(
        loss=loss,
        kappa=float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        accuracy=float(accuracy_score(y_true, y_pred)),
        f1_macro=float(f1_score(y_true, y_pred, labels=class_labels, average="macro", zero_division=0)),
        f1_weighted=float(f1_score(y_true, y_pred, labels=class_labels, average="weighted", zero_division=0)),
        auc_macro=auc_macro,
        auc_weighted=auc_weighted,
        f1_per_class=[
            float(value) for value in f1_score(y_true, y_pred, labels=class_labels, average=None, zero_division=0)
        ],
        auc_per_class=auc_per_class,
        recall_per_class=[
            float(value) for value in recall_score(y_true, y_pred, labels=class_labels, average=None, zero_division=0)
        ],
        confusion_matrix=confusion_matrix(y_true, y_pred, labels=class_labels).tolist(),
    )


def _per_class_auc(y_true_one_hot: np.ndarray, probabilities: np.ndarray) -> list[float | None]:
    """Calculate one-vs-rest ROC-AUC for each class when defined."""
    values: list[float | None] = []
    for class_index in range(y_true_one_hot.shape[1]):
        try:
            value = float(roc_auc_score(y_true_one_hot[:, class_index], probabilities[:, class_index]))
            values.append(value if np.isfinite(value) else None)
        except ValueError:
            values.append(None)
    return values


def _multiclass_auc(y_true_one_hot: np.ndarray, probabilities: np.ndarray, average: str) -> float | None:
    """Calculate multiclass one-vs-rest ROC-AUC, returning None if undefined."""
    try:
        value = float(roc_auc_score(y_true_one_hot, probabilities, average=average, multi_class="ovr"))
        return value if np.isfinite(value) else None
    except ValueError:
        return None
