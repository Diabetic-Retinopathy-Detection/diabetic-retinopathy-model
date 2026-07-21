"""Create a small pickle index for smoke-testing the pretraining pipeline.

Selects the first 80 images from data/cropped/ that have matching saliency
files in data/saliency/ and writes them to data/smoke_dataset.pkl using
relative paths (resolved against data_dir at runtime).
"""

from __future__ import annotations

import pickle
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
OUTPUT = DATA_DIR / "smoke_dataset.pkl"
N_SAMPLES = 80


def main() -> None:
    cropped_dir = DATA_DIR / "cropped"
    saliency_dir = DATA_DIR / "saliency"

    if not cropped_dir.is_dir():
        msg = f"Cropped directory not found: {cropped_dir}"
        raise SystemExit(msg)
    if not saliency_dir.is_dir():
        msg = f"Saliency directory not found: {saliency_dir}"
        raise SystemExit(msg)

    pairs: list[tuple[Path, Path]] = []
    for img_path in sorted(cropped_dir.glob("*.jpeg")):
        if len(pairs) >= N_SAMPLES:
            break
        sal_path = saliency_dir / img_path.name.replace(".jpeg", ".npy")
        if sal_path.exists():
            pairs.append((
                Path("cropped") / img_path.name,
                Path("saliency") / sal_path.name,
            ))

    if len(pairs) < N_SAMPLES:
        print(f"Warning: only found {len(pairs)} matching pairs (requested {N_SAMPLES})")

    index = {"root": ".", "pairs": pairs}
    with open(OUTPUT, "wb") as f:
        pickle.dump(index, f)

    print(f"Wrote {len(pairs)} pairs to {OUTPUT}")


if __name__ == "__main__":
    main()
