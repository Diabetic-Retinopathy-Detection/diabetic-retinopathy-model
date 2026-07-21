#!/usr/bin/env bash
# ─── Smoke test: build Docker training image and run a short training ──
#
# Verifies the full pipeline works inside Docker:
#   1. Docker image builds
#   2. Smoke dataset is created (locally, needs Python)
#   3. Training completes without errors
#
# Usage (run from diabetic-retinopathy-model/):
#   bash scripts/smoke_test.sh            # GPU (default)
#   bash scripts/smoke_test.sh --cpu      # CPU only
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

GPU_FLAG="--gpus all"
if [[ "${1:-}" == "--cpu" ]]; then
    GPU_FLAG=""
    echo "=== CPU mode ==="
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
# Data lives in the parent repo's data/ directory
REPO_ROOT="$(dirname "$PROJECT_DIR")"
DATA_DIR="$REPO_ROOT/data"

echo "=== Building Docker training image ==="
docker build -f "$PROJECT_DIR/Dockerfile.training" -t dr-train:local "$PROJECT_DIR"

echo ""
echo "=== Creating smoke dataset (local) ==="
cd "$PROJECT_DIR"
uv run python scripts/create_smoke_dataset.py

echo ""
echo "=== Running smoke test ==="
docker run --rm $GPU_FLAG \
    -v "$DATA_DIR:/app/data" \
    dr-train:local \
    uv run dr-train --phase pretrain --device cpu \
        --config configs/pretrain_smoke.yaml \
        --data-index-path /app/data/smoke_dataset.pkl \
        --data-dir /app/data \
        --seed 42

echo ""
echo "=== Smoke test passed ==="
