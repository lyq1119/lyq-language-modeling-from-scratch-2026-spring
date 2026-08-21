"""Train the assignment Transformer LM on memory-mapped token arrays."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from cs336_basics.experiment import ExperimentLogger
from cs336_basics.model import TransformerLM
from cs336_basics.training import (
    AdamW,
    clip_gradients,
    cosine_learning_rate,
    cross_entropy,
    get_batch,
    load_checkpoint,
    save_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--valid-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--data-dtype", default="uint16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--rope-theta", type=float, default=10_000)
    parser.add_argument(
        "--no-rmsnorm",
        action="store_true",
        help="Remove every RMSNorm layer for the normalization ablation.",
    )
    parser.add_argument(
        "--post-norm",
        action="store_true",
        help="Apply RMSNorm after each residual addition instead of before each sublayer.",
    )
    parser.add_argument(
        "--no-rope",
        action="store_true",
        help="Disable rotary position embeddings for the NoPE ablation.",
    )
    parser.add_argument(
        "--silu-ffn",
        action="store_true",
        help="Use an ungated SiLU feed-forward network instead of SwiGLU.",
    )

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=5_000)
    parser.add_argument("--max-lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-iters", type=int, default=200)
    parser.add_argument("--cosine-cycle-iters", type=int)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument(
        "--overfit-single-batch",
        action="store_true",
        help="Reuse one training batch to verify that the model can overfit it.",
    )
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        "vocab_size", "context_length", "d_model", "d_ff", "num_layers",
        "num_heads", "batch_size", "iterations", "log_every", "eval_every",
        "eval_batches", "checkpoint_every",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.d_model % args.num_heads:
        raise ValueError("--d-model must be divisible by --num-heads")


def open_dataset(path: Path, dtype: str, vocab_size: int, context_length: int) -> np.memmap:
    dataset = np.memmap(path, dtype=np.dtype(dtype), mode="r")
    if len(dataset) <= context_length:
        raise ValueError(f"{path} has too few tokens for context length {context_length}")
    sample = dataset[: min(len(dataset), 1_000_000)]
    if sample.size and int(sample.max()) >= vocab_size:
        raise ValueError(f"{path} contains a sampled token ID outside vocab size {vocab_size}")
    return dataset


@torch.no_grad()
def evaluate(
    model: TransformerLM,
    dataset: np.memmap,
    batch_size: int,
    context_length: int,
    device: torch.device,
    num_batches: int,
) -> float:
    was_training = model.training
    model.eval()
    total = 0.0
    for _ in range(num_batches):
        inputs, targets = get_batch(dataset, batch_size, context_length, device)
        total += cross_entropy(model(inputs), targets).item()
    model.train(was_training)
    return total / num_batches


def main() -> None:
    args = parse_args()
    validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config.update({key: str(value) for key, value in config.items() if isinstance(value, Path)})
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    logger = ExperimentLogger(
        args.output_dir,
        config,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_mode=args.wandb_mode,
    )

    train_data = open_dataset(
        args.train_data, args.data_dtype, args.vocab_size, args.context_length
    )
    valid_data = open_dataset(
        args.valid_data, args.data_dtype, args.vocab_size, args.context_length
    )
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        use_rmsnorm=not args.no_rmsnorm,
        norm_first=not args.post_norm,
        use_rope=not args.no_rope,
        use_swiglu=not args.silu_ffn,
        device=device,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )
    start_iteration = (
        load_checkpoint(args.resume, model, optimizer) if args.resume is not None else 0
    )
    if start_iteration > args.iterations:
        raise ValueError("checkpoint iteration exceeds requested --iterations")

    cycle_iters = args.cosine_cycle_iters or args.iterations
    fixed_batch = (
        get_batch(train_data, args.batch_size, args.context_length, device)
        if args.overfit_single_batch
        else None
    )
    running_loss = 0.0
    model.train()
    try:
        for iteration in range(start_iteration, args.iterations):
            learning_rate = cosine_learning_rate(
                iteration, args.max_lr, args.min_lr, args.warmup_iters, cycle_iters
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate

            if fixed_batch is None:
                inputs, targets = get_batch(
                    train_data, args.batch_size, args.context_length, device
                )
            else:
                inputs, targets = fixed_batch
            optimizer.zero_grad(set_to_none=True)
            loss = cross_entropy(model(inputs), targets)
            loss.backward()
            if args.max_grad_norm > 0:
                clip_gradients(model.parameters(), args.max_grad_norm)
            optimizer.step()
            completed = iteration + 1
            running_loss += loss.item()

            if completed % args.log_every == 0:
                logger.log(
                    "train",
                    completed,
                    running_loss / args.log_every,
                    learning_rate=learning_rate,
                )
                running_loss = 0.0
            if completed % args.eval_every == 0:
                validation_loss = evaluate(
                    model, valid_data, args.batch_size, args.context_length,
                    device, args.eval_batches,
                )
                logger.log("validation", completed, validation_loss)
            if completed % args.checkpoint_every == 0:
                save_checkpoint(
                    model, optimizer, completed,
                    args.output_dir / f"checkpoint_{completed:07d}.pt",
                )

        save_checkpoint(
            model, optimizer, args.iterations, args.output_dir / "checkpoint_final.pt"
        )
    finally:
        logger.finish()


if __name__ == "__main__":
    main()
