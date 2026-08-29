"""Benchmark all-reduce runtime in a single-node multi-process setting.

Problem (`distributed_communication_single_node`): Distributed Communication
(Single Node) (5 points)

This script benchmarks the runtime of the all-reduce collective in a
single-node multi-process setup, sweeping over the all-reduce data size
(1MB, 10MB, 100MB, 1GB of float32 data) and the number of GPU processes
(2, 4, 6). Timings are measured per call (after warmup) on the NCCL backend,
and aggregated across ranks.

Usage:
    uv run python distributed_comm_benchmark.py --all
    uv run python distributed_comm_benchmark.py --world-size 4 --size 100MB
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Size labels -> number of bytes (binary convention: 1 MB = 2^20, 1 GB = 2^30).
SIZES_BYTES = {"1MB": 2**20, "10MB": 10 * 2**20, "100MB": 100 * 2**20, "1GB": 2**30}


def _setup(rank: int, world_size: int) -> None:
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def _worker(rank, world_size, numel, warmup, iterations, queue):
    _setup(rank, world_size)
    tensor = torch.randn(numel, dtype=torch.float32, device="cuda")

    # Warmup. Especially important for NCCL communication calls.
    for _ in range(warmup):
        dist.all_reduce(tensor, async_op=False)
        torch.cuda.synchronize()

    dist.barrier()

    # Timed iterations. Even with async_op=False we synchronize explicitly,
    # since the call returns once the op is *queued* on the GPU, not when the
    # communication actually finishes.
    local = torch.zeros(iterations, dtype=torch.float64, device="cuda")
    for i in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        dist.all_reduce(tensor, async_op=False)
        torch.cuda.synchronize()
        local[i] = time.perf_counter() - start

    # Aggregate timings across ranks.
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)
    if rank == 0:
        all_times = torch.stack(gathered)  # (world_size, iterations)
        queue.put(
            {
                "world_size": world_size,
                "size_bytes": numel * 4,
                "mean_ms": float(all_times.mean().item() * 1e3),
                "std_ms": float(all_times.std().item() * 1e3),
                "median_ms": float(all_times.median().item() * 1e3),
                "max_ms": float(all_times.max().item() * 1e3),
            }
        )
    dist.destroy_process_group()


def run_config(world_size: int, size_label: str, warmup: int, iterations: int) -> dict:
    numel = SIZES_BYTES[size_label] // 4
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    mp.spawn(
        _worker,
        args=(world_size, numel, warmup, iterations, queue),
        nprocs=world_size,
        join=True,
        start_method="spawn",
    )
    return queue.get()


def _make_plot(results, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_ws = {}
    for r in results:
        by_ws.setdefault(r["world_size"], []).append(r)

    fig, ax = plt.subplots(figsize=(8, 5))
    for ws in sorted(by_ws):
        rs = sorted(by_ws[ws], key=lambda r: r["size_bytes"])
        sizes_mb = [r["size_bytes"] / 2**20 for r in rs]
        mean_ms = [r["mean_ms"] for r in rs]
        std_ms = [r["std_ms"] for r in rs]
        ax.errorbar(sizes_mb, mean_ms, yerr=std_ms, marker="o", capsize=3, label=f"{ws} GPUs")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ax.set_xlabel("all-reduce data size (MB, float32)")
    ax.set_ylabel("time per all-reduce (ms)")
    ax.set_title("NCCL all-reduce latency (single node)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"wrote plot to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-size", type=int, choices=[2, 4, 6], default=2)
    parser.add_argument("--size", choices=list(SIZES_BYTES), default="1MB")
    parser.add_argument("--all", action="store_true", help="run every (world_size, size) combination")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--out", type=Path, default=None, help="write collected results as JSON")
    parser.add_argument("--plot", type=Path, default=None, help="save comparison plot (needs matplotlib)")
    args = parser.parse_args()

    configs = (
        [(ws, s) for ws in (2, 4, 6) for s in SIZES_BYTES]
        if args.all
        else [(args.world_size, args.size)]
    )

    results = []
    for ws, size in configs:
        print(f"running world_size={ws} size={size} ...", flush=True)
        res = run_config(ws, size, args.warmup, args.iterations)
        results.append(res)
        print(json.dumps(res))

    if args.out is not None:
        args.out.write_text(json.dumps(results, indent=2))
        print(f"wrote results to {args.out}")

    if args.plot is not None:
        _make_plot(results, args.plot)


if __name__ == "__main__":
    main()
