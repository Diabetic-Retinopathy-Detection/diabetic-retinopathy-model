# ─── Diabetic Retinopathy Model — Serving Image ─────────────────────
#
# This Dockerfile builds a CPU‑only serving image for the FastAPI API.
# It installs the CPU‑only PyTorch wheel explicitly to keep the image
# as lean as possible (~1.2 GB vs ~2.5 GB with CUDA stubs).
#
# For training on the cluster use a different base image, e.g.
#   nvcr.io/nvidia/pytorch:xx.xx-py3
# ────────────────────────────────────────────────────────────────────

FROM python:3.10-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock /app/

RUN uv sync --frozen --no-install-project

COPY . /app
RUN uv sync --frozen

# Install CPU‑only PyTorch (replaces the default wheel which includes CUDA stubs)
RUN uv pip install torch --extra-index-url https://download.pytorch.org/whl/cpu

EXPOSE 8000

CMD ["uv", "run", "dr-serve"]
