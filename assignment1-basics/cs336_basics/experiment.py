"""Experiment tracking with durable local logs and optional W&B visualization."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class ExperimentLogger:
    """Record step/time loss curves locally and optionally send them to W&B."""

    def __init__(
        self,
        output_dir: Path,
        config: dict[str, Any],
        *,
        wandb_project: str | None = None,
        wandb_entity: str | None = None,
        wandb_run_name: str | None = None,
        wandb_mode: str = "online",
        wandb_module: Any | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = output_dir / "metrics.jsonl"
        self.started_at = time.perf_counter()
        self.run = None
        if wandb_project is not None:
            if wandb_module is None:
                import wandb as wandb_module
            self.run = wandb_module.init(
                project=wandb_project,
                entity=wandb_entity,
                name=wandb_run_name,
                config=config,
                dir=str(output_dir),
                mode=wandb_mode,
            )
            self.run.define_metric("gradient_step")
            self.run.define_metric("wallclock_seconds")
            self.run.define_metric("train/*", step_metric="gradient_step")
            self.run.define_metric("validation/*", step_metric="gradient_step")

    def log(
        self,
        split: str,
        iteration: int,
        loss: float,
        *,
        learning_rate: float | None = None,
    ) -> dict[str, int | float | str]:
        if split not in {"train", "validation"}:
            raise ValueError("split must be 'train' or 'validation'")
        metric: dict[str, int | float | str] = {
            "type": split,
            "gradient_step": iteration,
            "wallclock_seconds": time.perf_counter() - self.started_at,
            "loss": loss,
        }
        if learning_rate is not None:
            metric["learning_rate"] = learning_rate
        with self.metrics_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(metric) + "\n")
        print(json.dumps(metric), flush=True)

        if self.run is not None:
            wandb_metric: dict[str, int | float] = {
                "gradient_step": iteration,
                "wallclock_seconds": float(metric["wallclock_seconds"]),
                f"{split}/loss": loss,
            }
            if learning_rate is not None:
                wandb_metric["train/learning_rate"] = learning_rate
            self.run.log(wandb_metric, step=iteration)
        return metric

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()

    def __enter__(self) -> ExperimentLogger:
        return self

    def __exit__(self, *_: object) -> None:
        self.finish()
