"""Per-dataset normalisation statistics.

EyePACS values are precomputed from the training set.  DDR, APTOS 2019, and
Messidor-2 values are sourced from SSiT's ``funcs.py`` (computed from each
dataset's train split).  Keys use SSiT's naming convention (``aptos2019``,
``ddr``, ``messidor2``) so a dataset can be selected by name across both the
pretraining and fine-tuning pipelines.
"""

from __future__ import annotations

DATASET_STATS: dict[str, dict[str, list[float]]] = {
    "eyepacs": {
        "mean": [0.425753653049469, 0.29737451672554016, 0.21293757855892181],
        "std": [0.27670302987098694, 0.20240527391433716, 0.1686241775751114],
    },
    "ddr": {
        "mean": [0.423737496137619, 0.2609460651874542, 0.128403902053833],
        "std": [0.29482534527778625, 0.20167365670204163, 0.13668020069599152],
    },
    "aptos2019": {
        "mean": [0.46100369095802307, 0.246780663728714, 0.07989078760147095],
        "std": [0.24873991310596466, 0.13842609524726868, 0.08025242388248444],
    },
    "messidor2": {
        "mean": [0.48436370491981506, 0.2238118201494217, 0.07583174854516983],
        "std": [0.2939208149909973, 0.14721707999706268, 0.06350880116224289],
    },
}

EYEPACS_MEAN: list[float] = DATASET_STATS["eyepacs"]["mean"]
EYEPACS_STD: list[float] = DATASET_STATS["eyepacs"]["std"]
