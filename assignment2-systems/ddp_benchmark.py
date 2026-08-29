"""Benchmark distributed data parallel (DDP) training variants.

Variant:
  naive   : all-reduce each parameter gradient after the backward pass (no overlap)
  flat    : all-reduce a single tensor of all flattened gradients after the backward pass
  overlap : asynchronously all-reduce each parameter gradient during the backward pass
            (the ``cs336_systems.ddp.DDP`` container used by ``adapters.get_ddp``)

For each training step we report the forward, backward, gradient-communication,
and optimizer times plus the end-to-end step time.  ``comm`` is the wall-clock
time that gradient synchronization adds on top of the forward/backward passes;
for the ``overlap`` variant this is the (small) remainder left after
``finish_gradient_synchronization()``, since most communication is hidden inside
the backward pass.

Usage:
    uv run python ddp_benchmark.py --model-size xl --variant overlap --gpus 2
    uv run python ddp_benchmark.py --model-size xl --variant naive  --gpus 2
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections.abc import Callable
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from cs336_basics.model import BasicsTransformerLM
from cs336_systems.ddp import DDP

MODEL_CONFIGS = {
    "small": dict(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": dict(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
    "10b": dict(d_model=4608, d_ff=12288, num_layers=50, num_heads=36),
}


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed(fn: Callable[[], object], device: torch.device, nvtx_name: str) -> float:
    _synchronize(device)
    start = time.perf_counter()
    nvtx_range = torch.cuda.nvtx.range(nvtx_name) if device.type == "cuda" else nullcontext()
    with nvtx_range:
        fn()
    _synchronize(device)
    return time.perf_counter() - start


def _sync_naive(model: torch.nn.Module) -> None:
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, async_op=False)


def _sync_flat(model: torch.nn.Module) -> None:
    grads = [param.grad for param in model.parameters() if param.grad is not None]
    if not grads:
        return
    flat = _flatten_dense_tensors(grads)
    dist.all_reduce(flat, op=dist.ReduceOp.AVG, async_op=False)
    for grad, updated in zip(grads, _unflatten_dense_tensors(flat, grads)):
        grad.copy_(updated)


def _worker(rank, world_size, args) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29601"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)

    config = MODEL_CONFIGS[args.model_size].copy()
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        **config,
    ).to(device=device, dtype=args.dtype)

    # All ranks must start from identical weights.
    for param in model.parameters():
        dist.broadcast(param.data, src=0)

    if args.variant == "overlap":
        model = DDP(model)  # broadcasts again (idempotent) and registers hooks

    optimizer_cls = {"adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}[args.optimizer]
    optimizer = optimizer_cls(model.parameters(), lr=args.learning_rate)

    inputs = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)
    loss_fn = torch.nn.CrossEntropyLoss()

    def sync_gradients() -> None:
        if args.variant == "naive":
            _sync_naive(model)
        elif args.variant == "flat":
            _sync_flat(model)
        else:
            model.finish_gradient_synchronization()

    samples: dict[str, list[float]] = {"forward": [], "backward": [], "comm": [], "optimizer": [], "step": []}

    def run_step(record: bool) -> None:
        model.zero_grad(set_to_none=True)
        holder: dict[str, torch.Tensor] = {}

        def forward() -> None:
            logits = model(inputs).float().reshape(-1, args.vocab_size)
            labels = torch.randint(0, args.vocab_size, (args.batch_size * args.context_length,), device=device)
            holder["loss"] = loss_fn(logits, labels)

        def backward() -> None:
            holder["loss"].backward()

        def optimizer_step() -> None:
            optimizer.step()

        forward_t = _timed(forward, device, "forward")
        backward_t = _timed(backward, device, "backward")
        comm_t = _timed(sync_gradients, device, "comm")
        optimizer_t = _timed(optimizer_step, device, "optimizer")

        if record:
            samples["forward"].append(forward_t)
            samples["backward"].append(backward_t)
            samples["comm"].append(comm_t)
            samples["optimizer"].append(optimizer_t)
            samples["step"].append(forward_t + backward_t + comm_t + optimizer_t)

    for _ in range(args.warmup_steps):
        run_step(record=False)
    for _ in range(args.measurement_steps):
        run_step(record=True)

    # Aggregate timings across ranks.
    names = list(samples)
    local = torch.zeros((len(names), args.measurement_steps), dtype=torch.float64, device=device)
    for i, name in enumerate(names):
        local[i] = torch.tensor(samples[name], dtype=torch.float64, device=device)
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)

    if rank == 0:
        all_times = torch.stack(gathered)  # (world_size, phases, steps)
        result = {"variant": args.variant, "world_size": world_size, "model_size": args.model_size,
                  "configuration": {**config, "vocab_size": args.vocab_size,
                                    "context_length": args.context_length,
                                    "batch_size": args.batch_size, "dtype": str(args.dtype).split('.')[-1],
                                    "optimizer": args.optimizer,
                                    "parameters": sum(p.numel() for p in model.parameters())}}
        timings = {}
        for i, name in enumerate(names):
            values = all_times[:, i, :]  # (world_size, steps)
            timings[name] = {"mean_ms": float(values.mean().item() * 1e3),
                             "std_ms": float(values.std().item() * 1e3)}
        result["timings"] = timings
        print(json.dumps(result, indent=2))

    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-size", choices=MODEL_CONFIGS, default="xl")
    parser.add_argument("--variant", choices=("naive", "flat", "overlap"), required=True)
    parser.add_argument("--gpus", type=int, default=2, choices=[2, 4])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--optimizer", choices=("adamw", "sgd"), default="adamw")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--measurement-steps", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    args.dtype = dtype
    mp.spawn(_worker, args=(args.gpus, args), nprocs=args.gpus, join=True)


if __name__ == "__main__":
    main()
