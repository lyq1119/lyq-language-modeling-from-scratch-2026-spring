"""Benchmark optimizer state sharding: peak memory and per-iteration time.

Measures, on every rank, the peak allocated CUDA memory at three points:
  1. after model initialization + optimizer construction,
  2. after a forward + backward pass (before the optimizer step),
  3. after the optimizer step (AdamW state has been materialized),
and the wall-clock time of a full training step (forward + backward +
optimizer step) with and without optimizer state sharding.

Both ranks run on the *same* data (the simplified setting used by the
assignment), so the sharded optimizer produces the same updates as the
unsharded one and the comparison isolates the cost of sharding itself.

Usage:
    uv run python sharded_optimizer_benchmark.py --model-size large --gpus 2 --sharded
    uv run python sharded_optimizer_benchmark.py --model-size large --gpus 2
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_basics.model import BasicsTransformerLM
from cs336_systems.sharded_optimizer import ShardedOptimizer

MODEL_CONFIGS = {
    "small": dict(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": dict(d_model=1280, d_ff=5120, num_layers=24, num_heads=20),
    "xl": dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
    "10b": dict(d_model=4608, d_ff=12288, num_layers=50, num_heads=36),
}


def _count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _worker(rank: int, world_size: int, args) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29888"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    config = MODEL_CONFIGS[args.model_size].copy()
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        **config,
    ).to(device=device, dtype=args.dtype)

    # All ranks start from identical weights.
    for p in model.parameters():
        dist.broadcast(p.data, src=0)

    n_params = _count_params(model)
    param_dtype = next(model.parameters()).dtype
    param_bytes = n_params * param_dtype.itemsize

    optimizer_cls = torch.optim.AdamW
    optim_kwargs = dict(lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999), eps=1e-8)
    if args.sharded:
        optimizer = ShardedOptimizer(model.parameters(), optimizer_cls, **optim_kwargs)
    else:
        optimizer = optimizer_cls(model.parameters(), **optim_kwargs)

    # Point 1: peak memory after model initialization + optimizer construction.
    # AdamW materializes its state lazily on the first step, so this is just
    # model parameters (+ CUDA context).
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    peak_after_init = torch.cuda.max_memory_allocated()

    inputs = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)
    labels = torch.randint(0, args.vocab_size, (args.batch_size * args.context_length,), device=device)
    loss_fn = torch.nn.CrossEntropyLoss()

    # warmup
    for _ in range(args.warmup_steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs).float().reshape(-1, args.vocab_size)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

    # Point 2: peak memory during a forward + backward pass (before optimizer step)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    logits = model(inputs).float().reshape(-1, args.vocab_size)
    loss = loss_fn(logits, labels)
    loss.backward()
    torch.cuda.synchronize()
    peak_before_step = torch.cuda.max_memory_allocated()

    # Point 3: peak memory during the optimizer step (AdamW state materialized)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    optimizer.step()
    torch.cuda.synchronize()
    peak_after_step = torch.cuda.max_memory_allocated()

    # Timing: full step over several iterations
    timings = []
    for _ in range(args.measurement_steps):
        torch.cuda.synchronize()
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs).float().reshape(-1, args.vocab_size)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1e3)

    # Gather peak-memory values across ranks (all ranks should agree).
    mem = torch.tensor([peak_after_init, peak_before_step, peak_after_step], dtype=torch.float64, device=device)
    gathered = [torch.empty_like(mem) for _ in range(world_size)]
    dist.all_gather(gathered, mem)
    times = torch.tensor(timings, dtype=torch.float64, device=device)
    t_gathered = [torch.empty_like(times) for _ in range(world_size)]
    dist.all_gather(t_gathered, times)

    if rank == 0:
        all_mem = torch.stack(gathered)  # (world_size, 3)
        all_times = torch.stack(t_gathered)  # (world_size, steps)
        result = {
            "model_size": args.model_size,
            "world_size": world_size,
            "sharded": args.sharded,
            "configuration": {
                **config,
                "vocab_size": args.vocab_size,
                "context_length": args.context_length,
                "batch_size": args.batch_size,
                "dtype": str(args.dtype).split(".")[-1],
                "parameters": n_params,
                "param_bytes_per_rank": param_bytes,
            },
            "peak_memory_mib": {
                "after_init_mean": float(all_mem[:, 0].mean().item() / 1024**2),
                "before_optimizer_step_mean": float(all_mem[:, 1].mean().item() / 1024**2),
                "after_optimizer_step_mean": float(all_mem[:, 2].mean().item() / 1024**2),
            },
            "step_time_ms": {
                "mean": float(all_times.mean().item()),
                "std": float(all_times.std().item()),
            },
        }
        print(json.dumps(result, indent=2))

    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-size", choices=MODEL_CONFIGS, default="large")
    parser.add_argument("--gpus", type=int, default=2, choices=[2, 4])
    parser.add_argument("--sharded", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--measurement-steps", type=int, default=5)
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    args.dtype = dtype
    mp.spawn(_worker, args=(args.gpus, args), nprocs=args.gpus, join=True)


if __name__ == "__main__":
    main()
