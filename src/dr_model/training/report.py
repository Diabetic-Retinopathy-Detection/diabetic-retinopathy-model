"""Export TensorBoard scalars to a PDF report.

Reads tfevents files from a TensorBoard log directory and generates
a multi-page PDF with loss curves, learning rate schedule, and
training summary statistics.

Usage::

    dr-report --logdir logs/vit_p16_e768_d12_h12_c5/
    dr-report --logdir logs/vit_p16_e768_d12_h12_c5/ --output report.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # type: ignore[import-untyped]

SCALAR_TAGS = [
    "loss/contrastive",
    "loss/saliency",
    "loss/total",
    "lr",
    "momentum_m",
]

TAG_LABELS = {
    "loss/contrastive": "Contrastive Loss",
    "loss/saliency": "Saliency Loss",
    "loss/total": "Total Loss",
    "lr": "Learning Rate",
    "momentum_m": "Momentum",
}


def _load_scalars(logdir: Path) -> dict[str, list[tuple[int, float]]]:
    ea = EventAccumulator(str(logdir))
    ea.Reload()
    available = set(ea.Tags().get("scalars", []))
    scalars: dict[str, list[tuple[int, float]]] = {}
    for tag in SCALAR_TAGS:
        if tag in available:
            events = ea.Scalars(tag)
            scalars[tag] = [(e.step, e.value) for e in events]
    return scalars


def _plot_page(
    fig: plt.Figure,
    tags: list[str],
    scalars: dict[str, list[tuple[int, float]]],
) -> None:
    n = len(tags)
    axes = fig.subplots(1, n, squeeze=False).flatten()

    for ax, tag in zip(axes, tags, strict=True):
        if tag not in scalars:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(TAG_LABELS.get(tag, tag))
            continue

        steps, values = zip(*scalars[tag], strict=True)
        ax.plot(steps, values, linewidth=1.2)
        ax.set_title(TAG_LABELS.get(tag, tag))
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.3)

        if len(values) > 1:
            ax.axhline(y=values[-1], color="red", linestyle="--", alpha=0.4, linewidth=0.8)
            ax.text(
                steps[-1],
                values[-1],
                f" {values[-1]:.4f}",
                fontsize=8,
                color="red",
                va="bottom",
            )

    fig.tight_layout()


def export_report(logdir: Path, output: Path) -> None:
    scalars = _load_scalars(logdir)
    if not scalars:
        msg = f"No scalar data found in {logdir}"
        raise FileNotFoundError(msg)

    page_configs: list[list[str]] = [
        ["loss/contrastive", "loss/saliency", "loss/total"],
        ["lr", "momentum_m"],
    ]

    with plt.ioff():
        pdf_backend = PdfPages(str(output))

        for tags in page_configs:
            tags_with_data = [t for t in tags if t in scalars]
            if not tags_with_data:
                continue
            fig = plt.figure(figsize=(5 * len(tags_with_data), 4))
            fig.suptitle(logdir.name, fontsize=12, fontweight="bold")
            _plot_page(fig, tags_with_data, scalars)
            pdf_backend.savefig(fig)
            plt.close(fig)

        summary_fig = plt.figure(figsize=(8, 4))
        summary_fig.suptitle("Training Summary", fontsize=14, fontweight="bold")
        ax = summary_fig.add_subplot(111)
        ax.axis("off")

        lines = []
        if "loss/total" in scalars:
            values = [v for _, v in scalars["loss/total"]]
            lines.append(f"Final total loss:  {values[-1]:.4f}")
            best_idx = values.index(min(values))
            best_step = scalars["loss/total"][best_idx][0]
            lines.append(f"Best total loss:   {min(values):.4f} (step {best_step})")
        if "loss/contrastive" in scalars:
            lines.append(f"Final contrastive: {scalars['loss/contrastive'][-1][1]:.4f}")
        if "loss/saliency" in scalars:
            lines.append(f"Final saliency:    {scalars['loss/saliency'][-1][1]:.4f}")
        if "lr" in scalars:
            lines.append(f"Final LR:          {scalars['lr'][-1][1]:.6f}")
        if "momentum_m" in scalars:
            lines.append(f"Final momentum:    {scalars['momentum_m'][-1][1]:.4f}")

        total_steps = max(len(v) for v in scalars.values())
        lines.append(f"Total epochs:      {total_steps}")

        ax.text(0.1, 0.9, "\n".join(lines), fontsize=11, fontfamily="monospace", va="top")
        pdf_backend.savefig(summary_fig)
        plt.close(summary_fig)

        pdf_backend.close()

    print(f"Report saved to {output}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dr-report",
        description="Export TensorBoard training data to a PDF report.",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        required=True,
        help="TensorBoard log directory (e.g. logs/vit_p16_e768_d12_h12_c5/).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output PDF path (default: <logdir>/training_report.pdf).",
    )
    args = parser.parse_args(argv)

    logdir = Path(args.logdir)
    if not logdir.exists():
        print(f"Error: log directory not found: {logdir}")
        return 1

    output = Path(args.output) if args.output else logdir / "training_report.pdf"

    export_report(logdir, output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
