#!/usr/bin/env python3
"""Fit and plot the Chinchilla IsoFLOPs scaling laws.

The script only uses the Python standard library.  It writes two SVG figures and
prints the selected optima, fitted laws, and extrapolated predictions.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable


def select_optima(runs: list[dict[str, float]]) -> list[tuple[float, float, float, float]]:
    """Return (compute, parameters, tokens, loss) for the best run per budget."""
    best: dict[float, dict[str, float]] = {}
    for run in runs:
        compute = float(run["compute_budget"])
        if compute not in best or run["final_loss"] < best[compute]["final_loss"]:
            best[compute] = run
    return [
        (compute, float(run["parameters"]), compute / (6 * run["parameters"]), float(run["final_loss"]))
        for compute, run in sorted(best.items())
    ]


def fit_power_law(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Fit y = coefficient * x**exponent by OLS in log10 space."""
    lx = [math.log10(x) for x in xs]
    ly = [math.log10(y) for y in ys]
    x_bar, y_bar = sum(lx) / len(lx), sum(ly) / len(ly)
    exponent = sum((x - x_bar) * (y - y_bar) for x, y in zip(lx, ly)) / sum(
        (x - x_bar) ** 2 for x in lx
    )
    coefficient = 10 ** (y_bar - exponent * x_bar)
    return coefficient, exponent


def svg_plot(
    path: Path,
    xs: list[float],
    ys: list[float],
    predict: Callable[[float], float],
    title: str,
    y_label: str,
) -> None:
    """Write a dependency-free log-log scaling-law plot as SVG."""
    width, height = 860, 560
    left, right, top, bottom = 105, 35, 55, 80
    x_min, x_max = math.log10(min(xs)) - 0.1, 24.1
    curve_xs = [10 ** (x_min + i * (x_max - x_min) / 200) for i in range(201)]
    all_ys = ys + [predict(x) for x in curve_xs]
    y_min = math.floor(min(map(math.log10, all_ys)) * 2) / 2
    y_max = math.ceil(max(map(math.log10, all_ys)) * 2) / 2

    def px(x: float) -> float:
        return left + (math.log10(x) - x_min) / (x_max - x_min) * (width - left - right)

    def py(y: float) -> float:
        return top + (y_max - math.log10(y)) / (y_max - y_min) * (height - top - bottom)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.tick{font-size:13px}.label{font-size:16px}.title{font-size:20px;font-weight:600}</style>',
        f'<text class="title" x="{width/2}" y="30" text-anchor="middle">{title}</text>',
    ]
    for exponent in range(math.ceil(x_min), math.floor(x_max) + 1):
        x = px(10**exponent)
        parts += [
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height-bottom}" stroke="#ddd"/>',
            f'<text class="tick" x="{x:.1f}" y="{height-bottom+24}" text-anchor="middle">10^{exponent}</text>',
        ]
    half = int(round(y_min * 2))
    while half <= int(round(y_max * 2)):
        exponent = half / 2
        y = py(10**exponent)
        label = f"10^{exponent:g}"
        parts += [
            f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#ddd"/>',
            f'<text class="tick" x="{left-12}" y="{y+5:.1f}" text-anchor="end">{label}</text>',
        ]
        half += 1
    points = " ".join(f"{px(x):.1f},{py(predict(x)):.1f}" for x in curve_xs)
    parts.append(f'<polyline points="{points}" fill="none" stroke="#d55e00" stroke-width="3"/>')
    for x, y in zip(xs, ys):
        parts.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="5" fill="#0072b2" stroke="white"/>')
    parts += [
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#222"/>',
        f'<text class="label" x="{(left+width-right)/2}" y="{height-25}" text-anchor="middle">Compute budget C (FLOPs)</text>',
        f'<text class="label" transform="translate(27 {(top+height-bottom)/2}) rotate(-90)" text-anchor="middle">{y_label}</text>',
        '<circle cx="625" cy="70" r="5" fill="#0072b2"/><text class="tick" x="638" y="75">observed optimum</text>',
        '<line x1="625" y1="92" x2="650" y2="92" stroke="#d55e00" stroke-width="3"/><text class="tick" x="658" y="97">power-law fit</text>',
        '</svg>',
    ]
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/isoflops_curves.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/chinchilla_isoflops"))
    args = parser.parse_args()
    runs = json.loads(args.data.read_text(encoding="utf-8"))
    optima = select_optima(runs)
    computes = [row[0] for row in optima]
    parameters = [row[1] for row in optima]
    tokens = [row[2] for row in optima]
    n_coef, a = fit_power_law(computes, parameters)
    d_coef, b = fit_power_law(computes, tokens)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    svg_plot(args.output_dir / "optimal_model_size.svg", computes, parameters, lambda c: n_coef * c**a,
             "IsoFLOPs scaling law: optimal model size", "Optimal model size N (parameters)")
    svg_plot(args.output_dir / "optimal_dataset_size.svg", computes, tokens, lambda c: d_coef * c**b,
             "IsoFLOPs scaling law: optimal dataset size", "Optimal dataset size D (tokens)")

    print("C,N_opt,D_opt,final_loss")
    for row in optima:
        print(",".join(f"{value:.8g}" for value in row))
    print(f"\nN_opt(C) = {n_coef:.8g} * C^{a:.8f}")
    print(f"D_opt(C) = {d_coef:.8g} * C^{b:.8f}")
    for compute in (1e23, 1e24):
        print(f"C={compute:.0e}: N_opt={n_coef * compute**a:.8g}, D_opt={d_coef * compute**b:.8g}")


if __name__ == "__main__":
    main()
