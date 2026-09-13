"""从训练日志解析验证损失并画学习曲线。

用法：python -m cc2026.plot_learning_curve <train_log> <out.png> [eval_interval]
日志里的行形如：  ... INFO Estimated validation loss: 3.4567
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EVAL_RE = re.compile(r"Estimated validation loss:\s*([0-9.]+)")
TRAIN_RE = re.compile(r"Training step (\d+), Loss:\s*([0-9.]+)")


def main() -> None:
    log_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    eval_interval = int(sys.argv[3]) if len(sys.argv) > 3 else 200

    text = log_path.read_text(errors="replace")
    eval_losses = [float(m) for m in EVAL_RE.findall(text)]
    steps = [(i + 1) * eval_interval for i in range(len(eval_losses))]

    # tqdm 的进度行会带 \r，先按 \r 和 \n 切开
    train_steps, train_losses = [], []
    for line in re.split(r"[\r\n]+", text):
        m = TRAIN_RE.search(line)
        if m:
            train_steps.append(int(m.group(1)))
            train_losses.append(float(m.group(2)))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if train_steps:
        ax.plot(train_steps, train_losses, color="tab:blue", alpha=0.3, linewidth=0.8, label="train loss")
    if eval_losses:
        ax.plot(steps, eval_losses, marker="o", color="tab:red", label="validation loss (Paloma C4-100)")
    ax.set_xlabel("optimization step")
    ax.set_ylabel("cross-entropy loss (nats)")
    ax.set_title("Learning curve on filtered CC data")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}; {len(eval_losses)} eval points")
    if eval_losses:
        best = min(eval_losses)
        print(f"best eval loss: {best:.4f} at step {steps[eval_losses.index(best)]}")


if __name__ == "__main__":
    main()
