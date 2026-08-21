"""Plot the TinyStories RoPE versus NoPE ablation."""

from pathlib import Path

import plot_layer_norm_ablation as plot


plot.RUNS = {
    "RoPE, lr=3e-3": (
        Path("runs/tinystories-batch-64-lr-3e-3/metrics.jsonl"),
        "#1f77b4",
    ),
    "NoPE, lr=3e-3": (
        Path("runs/tinystories-ablation-no-rope-lr-3e-3/metrics.jsonl"),
        "#9467bd",
    ),
}
plot.NAN_ANNOTATION_STEP = None


if __name__ == "__main__":
    plot.main(Path("experiment_results/tinystories_no_pos_emb_ablation.svg"))
