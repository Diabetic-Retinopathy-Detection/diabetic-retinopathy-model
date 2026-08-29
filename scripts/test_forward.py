#!/usr/bin/env python3
"""Standalone forward-pass test — reproduces the cluster crash without DDP/data.

Builds the Finetuner, loads the pretrain checkpoint, runs one forward pass
on a random tensor (batch 2, 3x384x384) under AMP (16-mixed).  Mirrors the
exact code path that failed on comp040 (finetune.py:65 mean(dim=1)).

Usage:
    uv run python scripts/test_forward.py            # CUDA if available
    uv run python scripts/test_forward.py --device cpu # force CPU
    uv run python scripts/test_forward.py --device mps # force MPS (Mac)

Exit 0 = forward OK, exit 1 = forward failed.
"""

from __future__ import annotations

import argparse
import sys
import traceback

import torch

from dr_model.config import Settings
from dr_model.model.finetune import Finetuner


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto|cuda|cpu|mps")
    parser.add_argument("--batch-size", type=int, default=2, help="batch size")
    parser.add_argument("--no-amp", action="store_true", help="disable AMP (fp32)")
    parser.add_argument("--checkpoint", default="checkpoints/full-pretraining/checkpoint.pt")
    args = parser.parse_args()

    # Resolve device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"torch {torch.__version__} | device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  capability: {torch.cuda.get_device_capability(0)}")
        print(f"  arch list: {torch.cuda.get_arch_list()}")
        print(f"  CUDA build: {torch.version.cuda}")
        print(f"  cuDNN: {torch.backends.cudnn.version()}")

    config = Settings()
    print(f"config: {config.model_name} | input {config.finetune_input_size} | precision {config.precision}")

    print("Building Finetuner and loading checkpoint...")
    model = Finetuner(config, checkpoint_path=args.checkpoint).to(device)
    model.eval()

    n_params = sum(1 for _ in model.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    print(f"param tensors: {n_params} | total params: {total_params:,}")
    print(f"pos_embed shape: {model.backbone.pos_embed.shape}")

    x = torch.randn(args.batch_size, 3, config.finetune_input_size, config.finetune_input_size, device=device)
    print(f"input: {tuple(x.shape)} on {x.device}")

    use_amp = config.precision == "16-mixed" and device.type == "cuda" and not args.no_amp
    print(f"AMP: {'ON (fp16 autocast)' if use_amp else 'OFF (fp32)'}")

    print("Running forward pass...")
    try:
        with torch.no_grad():
            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits = model(x)
            else:
                logits = model(x)
        print(f"FORWARD OK — logits: {tuple(logits.shape)} dtype={logits.dtype}")
        print(f"  sample: {logits[0].tolist()}")
    except Exception:
        print("FORWARD FAILED:")
        traceback.print_exc()
        return 1
    else:
        return 0


if __name__ == "__main__":
    sys.exit(main())
