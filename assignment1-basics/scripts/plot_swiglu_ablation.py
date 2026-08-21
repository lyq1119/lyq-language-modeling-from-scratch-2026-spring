"""Plot the TinyStories SwiGLU versus ungated SiLU ablation."""

from pathlib import Path

import plot_layer_norm_ablation as plot


plot.RUNS = {
    "SwiGLU (d_ff=1344)": (
        Path("runs/tinystories-batch-64-lr-3e-3/metrics.jsonl"),
        "#1f77b4",
    ),
    "SiLU (d_ff=2048)": (
        Path("runs/tinystories-ablation-silu-ffn-lr-3e-3/metrics.jsonl"),
        "#e377c2",
    ),
}
plot.NAN_ANNOTATION_STEP = None


if __name__ == "__main__":
    plot.main(Path("experiment_results/tinystories_swiglu_ablation.svg"))
