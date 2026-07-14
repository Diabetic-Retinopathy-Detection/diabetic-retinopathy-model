"""Per-dataset normalisation statistics.

EyePACS values are precomputed from the training set.  APTOS and Messidor-2
are placeholders — fill them in once the respective datasets are processed.
"""

from __future__ import annotations

DATASET_STATS: dict[str, dict[str, list[float]]] = {
    "eyepacs": {
        "mean": [0.425753653049469, 0.29737451672554016, 0.21293757855892181],
        "std": [0.27670302987098694, 0.20240527391433716, 0.1686241775751114],
    },
    "aptos": {
        "mean": [0.0, 0.0, 0.0],
        "std": [1.0, 1.0, 1.0],
    },
    "messidor2": {
        "mean": [0.0, 0.0, 0.0],
        "std": [1.0, 1.0, 1.0],
    },
}

EYEPACS_MEAN: list[float] = DATASET_STATS["eyepacs"]["mean"]
EYEPACS_STD: list[float] = DATASET_STATS["eyepacs"]["std"]
