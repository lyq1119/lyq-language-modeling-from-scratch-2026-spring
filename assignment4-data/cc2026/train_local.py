"""在本地/Slurm 上启动 scripts/train.py 的训练循环（不修改原训练脚本）。

scripts/train.py 会 import cs336_data.modal_utils，而该模块要求设置 SUNET_ID 并会
连接 Modal。这里在导入前向 sys.modules 注入一个 stub，从而可以在没有 Modal 的
集群上复用完全相同的训练代码。

用法（单机 8 卡）：
  torchrun --standalone --nproc_per_node=8 -m cc2026.train_local \
      --train-bin .../train.bin --valid-bin .../valid.bin --model-output .../your_data \
      --train-steps 16384
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _install_modal_stub() -> None:
    if "cs336_data.modal_utils" in sys.modules:
        return

    class _App:
        def function(self, *a, **k):
            def deco(fn):
                return fn

            return deco

        def local_entrypoint(self, *a, **k):
            def deco(fn):
                return fn

            return deco

    stub = types.ModuleType("cs336_data.modal_utils")
    stub.app = _App()
    stub.VOLUME_MOUNTS = {}
    stub.MODAL_SECRETS = []
    stub.build_image = lambda **k: None
    sys.modules["cs336_data.modal_utils"] = stub


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-bin", required=True)
    parser.add_argument("--valid-bin", required=True)
    parser.add_argument("--model-output", required=True)
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--eval-iterations", type=int, default=None)
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--no-compile", action="store_true", help="禁用 torch.compile（该集群上 inductor 会 SIGSEGV）")
    parser.add_argument("--micro-batch-size", type=int, default=None, help="每卡 micro batch（32G 显存放不下 128，用梯度累积保持等效 batch）")
    parser.add_argument("--grad-accum", type=int, default=None, help="梯度累积步数")
    parser.add_argument("--eval-batch-size", type=int, default=None, help="评估 micro batch")
    args = parser.parse_args()

    _install_modal_stub()

    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )

    from cs336_basics.train_config import Config, PathsConfig

    spec = importlib.util.spec_from_file_location("_a4_train", ROOT / "scripts" / "train.py")
    train = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(train)

    cfg = Config(paths=PathsConfig(train_bin=Path(args.train_bin), valid_bin=Path(args.valid_bin), model_output=Path(args.model_output)))
    if args.train_steps is not None:
        cfg.training.train_steps = args.train_steps
    if args.eval_interval is not None:
        cfg.training.eval_interval = args.eval_interval
    if args.eval_iterations is not None:
        cfg.training.eval_iterations = args.eval_iterations
    if args.save_checkpoints:
        cfg.training.save_checkpoints = True
    if args.no_compile:
        cfg.training.compile = False
    if args.micro_batch_size is not None:
        cfg.training.train_batch_size = args.micro_batch_size
    if args.grad_accum is not None:
        cfg.training.gradient_accumulation_steps = args.grad_accum
    if args.eval_batch_size is not None:
        cfg.training.eval_batch_size = args.eval_batch_size

    train.train_from_config(cfg)


if __name__ == "__main__":
    main()
