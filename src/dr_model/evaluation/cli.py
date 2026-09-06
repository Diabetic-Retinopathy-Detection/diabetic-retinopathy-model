"""CLI entry point for encoder evaluation.

Usage::

    uv run knn-probe \\
        --method knn \\
        --encoder checkpoints/.../epoch_40_encoder.pt \\
        --data-path datasets/DDR-ImageFolder \\
        --dataset ddr
"""

from __future__ import annotations

import argparse
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="knn-probe",
        description="Evaluate a pretrained DR encoder.",
    )
    parser.add_argument(
        "--method",
        choices=["knn"],
        required=True,
        help="Evaluation method.",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        required=True,
        help="Path to pretrained encoder checkpoint (.pt).",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to ImageFolder dataset root (contains train/ and test/).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ddr",
        choices=["ddr", "aptos2019", "messidor2"],
        help="Dataset name for normalisation stats (default: ddr).",
    )
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--knn", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=str, default=None, help="Save results JSON to this path.")

    args = parser.parse_args()

    if args.method == "knn":
        from dr_model.evaluation.knn import evaluate_knn

        results = evaluate_knn(
            encoder_path=args.encoder,
            data_path=args.data_path,
            dataset=args.dataset,
            input_size=args.input_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            nb_knn=args.knn,
            temperature=args.temperature,
            device=args.device,
        )

        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output}")

        sys.exit(0)


if __name__ == "__main__":
    main()
