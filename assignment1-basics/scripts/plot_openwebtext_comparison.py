"""Plot equal-token TinyStories and OpenWebText training runs."""

from pathlib import Path

import plot_layer_norm_ablation as plot


plot.RUNS = {
    "TinyStories (vocab=10k)": (
        Path("runs/tinystories-batch-64-lr-3e-3/metrics.jsonl"),
        "#1f77b4",
    ),
    "OpenWebText (vocab=32k)": (
        Path("runs/openwebtext-baseline-lr-3e-3/metrics.jsonl"),
        "#ff7f0e",
    ),
}
plot.NAN_ANNOTATION_STEP = None
plot.Y_MAX = 7.5


if __name__ == "__main__":
    plot.main(Path("experiment_results/tinystories_openwebtext_comparison.svg"))
