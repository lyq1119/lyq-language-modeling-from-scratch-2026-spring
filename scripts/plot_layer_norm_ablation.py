"""Plot the TinyStories RMSNorm ablation runs using only the standard library."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path


RUNS = {
    "RMSNorm, lr=3e-3": (Path("runs/tinystories-batch-64-lr-3e-3/metrics.jsonl"), "#1f77b4"),
    "No RMSNorm, lr=3e-3": (Path("runs/tinystories-ablation-no-rmsnorm-lr-3e-3/metrics.jsonl"), "#ff7f0e"),
    "No RMSNorm, lr=3e-4": (Path("runs/tinystories-ablation-no-rmsnorm-lr-3e-4/metrics.jsonl"), "#2ca02c"),
}
WIDTH, HEIGHT = 1200, 510
NAN_ANNOTATION_STEP: int | None = 200
Y_MIN, Y_MAX = 1.5, 5.5


def metric_rows(path: Path, metric_type: str) -> list[dict[str, float]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return [row for row in rows if row["type"] == metric_type and math.isfinite(row["loss"])]


def panel(metric_type: str, left: int, title: str) -> list[str]:
    top, width, height = 55, 500, 350
    x_max, y_min, y_max = 2500, Y_MIN, Y_MAX
    parts = [
        f'<text x="{left + width / 2}" y="28" text-anchor="middle" font-size="20">{title}</text>',
        f'<rect x="{left}" y="{top}" width="{width}" height="{height}" fill="white" stroke="#333"/>',
    ]
    for step in range(0, 2501, 500):
        x = left + step / x_max * width
        parts += [
            f'<line x1="{x}" y1="{top}" x2="{x}" y2="{top + height}" stroke="#ddd"/>',
            f'<text x="{x}" y="{top + height + 20}" text-anchor="middle" font-size="12">{step}</text>',
        ]
    tick_step = 1.0
    loss_ticks = [y_min + tick_step * index for index in range(round((y_max - y_min) / tick_step) + 1)]
    for loss in loss_ticks:
        y = top + (y_max - loss) / (y_max - y_min) * height
        parts += [
            f'<line x1="{left}" y1="{y}" x2="{left + width}" y2="{y}" stroke="#ddd"/>',
            f'<text x="{left - 8}" y="{y + 4}" text-anchor="end" font-size="12">{loss:g}</text>',
        ]
    for _label, (path, color) in RUNS.items():
        points = []
        for row in metric_rows(path, metric_type):
            loss = min(max(row["loss"], y_min), y_max)
            x = left + row["gradient_step"] / x_max * width
            y = top + (y_max - loss) / (y_max - y_min) * height
            points.append(f"{x:.1f},{y:.1f}")
        parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"/>')
        if metric_type == "validation":
            parts.extend(
                f'<circle cx="{point.split(",")[0]}" cy="{point.split(",")[1]}" r="2.5" fill="{color}"/>'
                for point in points
            )
    if metric_type == "train" and NAN_ANNOTATION_STEP is not None:
        x = left + NAN_ANNOTATION_STEP / x_max * width
        parts += [
            f'<line x1="{x}" y1="{top}" x2="{x}" y2="{top + height}" stroke="#ff7f0e" stroke-dasharray="6 4"/>',
            f'<text x="{x + 6}" y="{top + 18}" fill="#b55200" font-size="12">first NaN: step {NAN_ANNOTATION_STEP}</text>',
        ]
    parts += [
        f'<text x="{left + width / 2}" y="{top + height + 45}" text-anchor="middle" font-size="14">Gradient step</text>',
        f'<text x="{left - 48}" y="{top + height / 2}" text-anchor="middle" font-size="14" transform="rotate(-90 {left - 48} {top + height / 2})">Loss</text>',
    ]
    return parts


def main(output: Path | None = None) -> None:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="sans-serif" fill="#222">',
        *panel("train", 80, "Training loss"),
        *panel("validation", 650, "Validation loss"),
    ]
    for index, (label, (_, color)) in enumerate(RUNS.items()):
        y = 465 + index * 15
        parts += [
            f'<line x1="390" y1="{y}" x2="420" y2="{y}" stroke="{color}" stroke-width="3"/>',
            f'<text x="428" y="{y + 4}" font-size="12">{html.escape(label)}</text>',
        ]
    parts += ["</g>", "</svg>"]
    output = output or Path("experiment_results/tinystories_layer_norm_ablation.svg")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
