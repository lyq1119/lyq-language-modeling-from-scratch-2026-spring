"""Plot the best TinyStories run for each batch size."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


RUNS = {
    16: Path("runs/tinystories-batch-16-lr-1.5e-3/metrics.jsonl"),
    32: Path("runs/tinystories-lr-2e-3/metrics.jsonl"),
    64: Path("runs/tinystories-batch-64-lr-3e-3/metrics.jsonl"),
}
LEARNING_RATES = {16: 1.5e-3, 32: 2e-3, 64: 3e-3}
TOKENS_PER_EXAMPLE = 256


def validation_rows(path: Path) -> list[dict[str, float]]:
    with path.open(encoding="utf-8") as file:
        rows = [json.loads(line) for line in file]
    return [row for row in rows if row["type"] == "validation"]


def main() -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for batch_size, path in RUNS.items():
        rows = validation_rows(path)
        steps = [row["gradient_step"] for row in rows]
        losses = [row["loss"] for row in rows]
        tokens = [step * batch_size * TOKENS_PER_EXAMPLE for step in steps]
        label = f"batch={batch_size}, lr={LEARNING_RATES[batch_size]:g}"
        axes[0].plot(steps, losses, marker="o", markersize=2.5, label=label)
        axes[1].plot(tokens, losses, marker="o", markersize=2.5, label=label)

    axes[0].set(title="Validation loss by gradient step", xlabel="Gradient step")
    axes[1].set(title="Validation loss by tokens processed", xlabel="Tokens processed")
    for axis in axes:
        axis.set_ylabel("Validation loss")
        axis.grid(alpha=0.25)
        axis.legend()
    axes[1].ticklabel_format(axis="x", style="sci", scilimits=(6, 6))

    output = Path("experiment_results/tinystories_batch_size_comparison_wandb.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    print(output)


if __name__ == "__main__":
    main()
