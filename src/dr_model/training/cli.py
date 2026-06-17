from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dr-train",
        description="Train a diabetic retinopathy model (pretrain or fine-tune).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (overrides DR_CONFIG_FILE env).",
    )
    parser.add_argument(
        "--phase",
        type=str,
        choices=["pretrain", "finetune"],
        default="pretrain",
        help="Training phase.",
    )
    args = parser.parse_args(argv)

    print(f"dr-train: phase={args.phase}, config={args.config or '(from env/default)'}")
    print("Training not implemented yet — this is a scaffold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
