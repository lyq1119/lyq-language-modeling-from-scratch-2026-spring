"""Render a compact active-memory timeline from a trusted PyTorch snapshot."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", type=int, default=5)
    parser.add_argument("--title", required=True)
    args = parser.parse_args()

    # Snapshots are pickle files and must only be loaded from trusted sources.
    with args.snapshot.open("rb") as file:
        snapshot = pickle.load(file)
    trace = snapshot["device_traces"][args.device]
    relevant = [event for event in trace if event["action"] in {"alloc", "free_requested"}]
    final_active = sum(
        segment["allocated_size"] for segment in snapshot["segments"] if segment["device"] == args.device
    )
    net_change = sum(
        event["size"] if event["action"] == "alloc" else -event["size"] for event in relevant
    )
    active = final_active - net_change
    start_us = relevant[0]["time_us"]
    times_ms = [0.0]
    active_mib = [active / 1024**2]
    for event in relevant:
        active += event["size"] if event["action"] == "alloc" else -event["size"]
        times_ms.append((event["time_us"] - start_us) / 1000)
        active_mib.append(active / 1024**2)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.plot(times_ms, active_mib, linewidth=1.2)
    axis.fill_between(times_ms, active_mib, alpha=0.2)
    axis.set(title=args.title, xlabel="Time (ms)", ylabel="Active memory (MiB)")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()
