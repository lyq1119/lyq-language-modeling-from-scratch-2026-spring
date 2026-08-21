"""Plot the TinyStories Pre-Norm versus Post-Norm ablation."""

from pathlib import Path

import plot_layer_norm_ablation as plot


plot.RUNS = {
    "Pre-Norm, lr=3e-3": (
        Path("runs/tinystories-batch-64-lr-3e-3/metrics.jsonl"),
        "#1f77b4",
    ),
    "Post-Norm, lr=3e-3": (
        Path("runs/tinystories-ablation-post-norm-lr-3e-3/metrics.jsonl"),
        "#d62728",
    ),
}
plot.NAN_ANNOTATION_STEP = None


if __name__ == "__main__":
    plot.main(Path("experiment_results/tinystories_pre_norm_ablation.svg"))
