"""End-to-end Transformer benchmarking utilities.

Run ``uv run python benchmark.py --help`` for the command-line interface.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from timeit import default_timer

import torch

from cs336_basics.model import BasicsTransformerLM


MODEL_CONFIGS = {
    "small": dict(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": dict(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
    "10b": dict(d_model=4608, d_ff=12288, num_layers=50, num_heads=36),
}


@dataclass(frozen=True)
class Timing:
    mean_ms: float
    std_ms: float


def _summarize(samples: list[float]) -> Timing:
    """Convert samples in seconds to population statistics in milliseconds."""
    return Timing(mean_ms=statistics.fmean(samples) * 1_000, std_ms=statistics.pstdev(samples) * 1_000)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed(operation: Callable[[], object], device: torch.device, nvtx_name: str) -> float:
    # Synchronizing before the clock also prevents earlier unmeasured work from
    # leaking into this interval. The synchronization after the operation is
    # essential because CUDA kernel launches are asynchronous.
    _synchronize(device)
    start = default_timer()
    nvtx_range = torch.cuda.nvtx.range(nvtx_name) if device.type == "cuda" else nullcontext()
    with nvtx_range:
        operation()
    _synchronize(device)
    return default_timer() - start


def benchmark(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    *,
    mode: str,
    warmup_steps: int,
    measurement_steps: int,
    learning_rate: float,
    mixed_precision: bool = False,
    memory_snapshot: Path | None = None,
    memory_stats: dict[str, int] | None = None,
) -> dict[str, Timing]:
    """Benchmark a model and return per-phase and requested end-to-end times."""
    if mode not in {"forward", "forward-backward", "full"}:
        raise ValueError(f"unknown benchmark mode: {mode}")
    if warmup_steps < 0 or measurement_steps < 1:
        raise ValueError("warmup_steps must be >= 0 and measurement_steps must be >= 1")

    device = inputs.device
    needs_backward = mode != "forward"
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) if mode == "full" else None
    samples: dict[str, list[float]] = {"forward": []}
    if needs_backward:
        samples["backward"] = []
        samples["forward_backward"] = []
    if optimizer is not None:
        samples["optimizer"] = []
        samples["full_step"] = []

    def run_step(record: bool) -> None:
        model.zero_grad(set_to_none=True)
        holder: dict[str, torch.Tensor] = {}

        def forward() -> None:
            autocast_context = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if mixed_precision and device.type == "cuda"
                else nullcontext()
            )
            with autocast_context:
                if needs_backward:
                    holder["loss"] = model(inputs).float().mean()
                else:
                    with torch.no_grad():
                        model(inputs)

        forward_time = _timed(forward, device, "forward")
        backward_time = 0.0
        optimizer_time = 0.0
        if needs_backward:
            backward_time = _timed(holder["loss"].backward, device, "backward")
        if optimizer is not None:
            optimizer_time = _timed(optimizer.step, device, "optimizer")

        if record:
            samples["forward"].append(forward_time)
            if needs_backward:
                samples["backward"].append(backward_time)
                samples["forward_backward"].append(forward_time + backward_time)
            if optimizer is not None:
                samples["optimizer"].append(optimizer_time)
                samples["full_step"].append(forward_time + backward_time + optimizer_time)

    model.train(needs_backward)
    if memory_snapshot is not None:
        if device.type != "cuda":
            raise ValueError("memory profiling requires a CUDA device")
        memory_snapshot.parent.mkdir(parents=True, exist_ok=True)
        # Start before warm-up so allocations retain their Python/C++ call stacks.
        torch.cuda.memory._record_memory_history(
            enabled="all",
            context="all",
            stacks="all",
            max_entries=1_000_000,
        )
    try:
        for _ in range(warmup_steps):
            run_step(record=False)
        if memory_snapshot is not None:
            # Report the peak from the measured steps, not from warm-up.
            torch.cuda.reset_peak_memory_stats(device)
        capture_range = torch.cuda.nvtx.range("benchmark") if device.type == "cuda" else nullcontext()
        with capture_range:
            for _ in range(measurement_steps):
                run_step(record=True)
        if memory_snapshot is not None:
            _synchronize(device)
            torch.cuda.memory._dump_snapshot(str(memory_snapshot))
            if memory_stats is not None:
                memory_stats.update(
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                )
    except torch.OutOfMemoryError:
        if memory_snapshot is not None:
            _synchronize(device)
            torch.cuda.memory._dump_snapshot(str(memory_snapshot))
        raise
    finally:
        if memory_snapshot is not None:
            torch.cuda.memory._record_memory_history(enabled=None)
    return {name: _summarize(values) for name, values in samples.items()}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark a cs336 Transformer on random token data.")
    parser.add_argument("--model-size", choices=MODEL_CONFIGS, default="small")
    parser.add_argument("--mode", choices=("forward", "forward-backward", "full"), default="full")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="keep parameters in the selected dtype but run eligible CUDA operations with BF16 autocast",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measurement-steps", type=int, default=10)
    parser.add_argument(
        "--memory-snapshot",
        type=Path,
        help="record the measured steps and write a PyTorch CUDA memory snapshot to this path",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--d-model", type=int, help="override the selected model configuration")
    parser.add_argument("--d-ff", type=int, help="override the selected model configuration")
    parser.add_argument("--num-layers", type=int, help="override the selected model configuration")
    parser.add_argument("--num-heads", type=int, help="override the selected model configuration")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--compile", action="store_true", help="benchmark torch.compile(model)")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.batch_size < 1 or args.context_length < 1 or args.vocab_size < 1:
        raise ValueError("batch size, context length, and vocabulary size must be positive")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available")
    dtype = getattr(torch, args.dtype)
    config = MODEL_CONFIGS[args.model_size].copy()
    for name in ("d_model", "d_ff", "num_layers", "num_heads"):
        override = getattr(args, name)
        if override is not None:
            config[name] = override
    if config["d_model"] % config["num_heads"] != 0:
        raise ValueError("d_model must be divisible by num_heads")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        **config,
    ).to(device=device, dtype=dtype)
    if args.compile:
        model = torch.compile(model)
    inputs = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)
    memory_stats: dict[str, int] = {}
    timings = benchmark(
        model,
        inputs,
        mode=args.mode,
        warmup_steps=args.warmup_steps,
        measurement_steps=args.measurement_steps,
        learning_rate=args.learning_rate,
        mixed_precision=args.mixed_precision,
        memory_snapshot=args.memory_snapshot,
        memory_stats=memory_stats,
    )

    output = {
        "configuration": {
            **config,
            "vocab_size": args.vocab_size,
            "context_length": args.context_length,
            "batch_size": args.batch_size,
            "device": str(device),
            "dtype": args.dtype,
            "mixed_precision": args.mixed_precision,
            "mode": args.mode,
            "warmup_steps": args.warmup_steps,
            "measurement_steps": args.measurement_steps,
            "compiled": args.compile,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        },
        "timings": {name: asdict(timing) for name, timing in timings.items()},
    }
    if args.memory_snapshot is not None:
        output["memory"] = {
            **memory_stats,
            "peak_allocated_mib": memory_stats["peak_allocated_bytes"] / 1024**2,
            "peak_reserved_mib": memory_stats["peak_reserved_bytes"] / 1024**2,
            "snapshot": str(args.memory_snapshot),
        }
    if args.json:
        print(json.dumps(output, indent=2))
    else:
        print(json.dumps(output["configuration"], indent=2))
        print(f"{'phase':<20} {'mean (ms)':>12} {'std (ms)':>12}")
        for name, timing in timings.items():
            print(f"{name:<20} {timing.mean_ms:>12.3f} {timing.std_ms:>12.3f}")


if __name__ == "__main__":
    main()
