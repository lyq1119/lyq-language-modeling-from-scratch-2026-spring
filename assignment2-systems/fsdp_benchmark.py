"""Benchmark FSDP training: per-phase step timing and peak memory.

Each rank trains on a different shard of the batch (true data parallelism).
FSDP shards the Linear/Embedding weights, all-gathers them for forward and
backward, and reduce-scatters gradients back onto the shards; replicated
(norm) gradients are all-reduced inside ``finish_gradient_synchronization``.

Phases are tagged with NVTX ranges so the run can be profiled with
``nsys profile`` to inspect the weight all-gather / forward overlap.

Usage:
    uv run python fsdp_benchmark.py --model-size xl --gpus 2 --optimizer sgd --context-length 128
    nsys profile --trace=cuda,nvtx --nvtx-capture=step -o profiles/fsdp_xl \\
        uv run python fsdp_benchmark.py --model-size xl --gpus 2 --optimizer sgd --context-length 128
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_basics.model import BasicsTransformerLM
from cs336_systems.fsdp import FSDP

MODEL_CONFIGS = {
    "small": dict(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": dict(d_model=1280, d_ff=5120, num_layers=24, num_heads=20),
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


def _worker(rank: int, world_size: int, args) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29999"
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
    model = FSDP(model, compute_dtype=args.compute_dtype)

    optimizer_cls = {"adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}[args.optimizer]
    optimizer = optimizer_cls(model.parameters(), lr=args.learning_rate)

    # Different data per rank: a real data-parallel split.
    torch.manual_seed(0)
    all_inputs = torch.randint(0, args.vocab_size, (args.batch_size * world_size, args.context_length), device=device)
    all_labels = torch.randint(0, args.vocab_size, (args.batch_size * world_size * args.context_length,), device=device)
    inputs = all_inputs[rank * args.batch_size : (rank + 1) * args.batch_size]
    labels = all_labels[rank * args.batch_size * args.context_length : (rank + 1) * args.batch_size * args.context_length]
    loss_fn = torch.nn.CrossEntropyLoss()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # With a compute_dtype, run the forward/backward under autocast so that
    # ops like the attention mask promotion keep consistent dtypes.
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=args.compute_dtype)
        if args.compute_dtype is not None and device.type == "cuda"
        else nullcontext()
    )

    samples: dict[str, list[float]] = {"forward": [], "backward": [], "grad_sync": [], "optimizer": [], "step": []}

    def run_step(record: bool) -> None:
        holder: dict[str, torch.Tensor] = {}

        def forward() -> None:
            with autocast_ctx:
                logits = model(inputs).float().reshape(-1, args.vocab_size)
            holder["loss"] = loss_fn(logits, labels)

        def backward() -> None:
            with autocast_ctx:
                holder["loss"].backward()

        def grad_sync() -> None:
            model.finish_gradient_synchronization()

        def optimizer_step() -> None:
            optimizer.step()

        with torch.cuda.nvtx.range("step") if device.type == "cuda" else nullcontext():
            forward_t = _timed(forward, device, "forward")
            backward_t = _timed(backward, device, "backward")
            grad_sync_t = _timed(grad_sync, device, "grad_sync")
            optimizer_t = _timed(optimizer_step, device, "optimizer")
            optimizer.zero_grad(set_to_none=True)

        if record:
            samples["forward"].append(forward_t)
            samples["backward"].append(backward_t)
            samples["grad_sync"].append(grad_sync_t)
            samples["optimizer"].append(optimizer_t)
            samples["step"].append(forward_t + backward_t + grad_sync_t + optimizer_t)

    for _ in range(args.warmup_steps):
        run_step(record=False)
    for _ in range(args.measurement_steps):
        run_step(record=True)

    peak_mem = torch.cuda.max_memory_allocated()

    # Aggregate across ranks.
    names = list(samples)
    local = torch.zeros((len(names), args.measurement_steps), dtype=torch.float64, device=device)
    for i, name in enumerate(names):
        local[i] = torch.tensor(samples[name], dtype=torch.float64, device=device)
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)
    peak = torch.tensor([peak_mem], dtype=torch.float64, device=device)
    peak_gathered = [torch.empty_like(peak) for _ in range(world_size)]
    dist.all_gather(peak_gathered, peak)

    if rank == 0:
        all_times = torch.stack(gathered)
        all_peaks = torch.stack(peak_gathered)
        result = {
            "variant": "fsdp",
            "model_size": args.model_size,
            "world_size": world_size,
            "configuration": {
                **config,
                "vocab_size": args.vocab_size,
                "context_length": args.context_length,
                "batch_size": args.batch_size,
                "dtype": str(args.dtype).split(".")[-1],
                "compute_dtype": str(args.compute_dtype).split(".")[-1] if args.compute_dtype else None,
                "optimizer": args.optimizer,
                "parameters": sum(p.numel() for p in model.parameters()),
            },
            "peak_memory_mib": float(all_peaks.mean().item() / 1024**2),
        }
        for i, name in enumerate(names):
            values = all_times[:, i, :]
            result[name + "_time_ms"] = {"mean": float(values.mean().item() * 1e3), "std": float(values.std().item() * 1e3)}
        print(json.dumps(result, indent=2))

    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-size", choices=MODEL_CONFIGS, default="xl")
    parser.add_argument("--gpus", type=int, default=2, choices=[2, 4])
    parser.add_argument("--optimizer", choices=("adamw", "sgd"), default="sgd")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--compute-dtype", choices=("float32", "bfloat16"), default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--measurement-steps", type=int, default=5)
    args = parser.parse_args()

    args.dtype = getattr(torch, args.dtype)
    args.compute_dtype = getattr(torch, args.compute_dtype) if args.compute_dtype else None
    mp.spawn(_worker, args=(args.gpus, args), nprocs=args.gpus, join=True)


if __name__ == "__main__":
    main()
