"""Reproduce the learning-rate experiment from assignment section 4.2."""

from __future__ import annotations

import math

import torch


class SGD(torch.optim.Optimizer):
    """SGD whose learning rate decays as lr / sqrt(t + 1)."""

    def __init__(self, params, lr: float = 1e-3) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        super().__init__(params, {"lr": lr})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                iteration = state.get("t", 0)
                parameter.add_(
                    parameter.grad,
                    alpha=-group["lr"] / math.sqrt(iteration + 1),
                )
                state["t"] = iteration + 1
        return loss


def run_experiment(learning_rate: float, num_steps: int = 10) -> list[float]:
    # Resetting the seed makes every learning rate start from identical weights.
    torch.manual_seed(0)
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    optimizer = SGD([weights], lr=learning_rate)
    losses = []

    for _ in range(num_steps):
        optimizer.zero_grad()
        loss = (weights**2).mean()
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    return losses


def main() -> None:
    learning_rates = (1.0, 10.0, 100.0, 1000.0)
    results = {lr: run_experiment(lr) for lr in learning_rates}

    header = "step" + "".join(f" | lr={lr:g}".rjust(17) for lr in learning_rates)
    print(header)
    print("-" * len(header))
    for step in range(10):
        row = f"{step:>4}" + "".join(
            f" | {results[lr][step]:>12.5e}" for lr in learning_rates
        )
        print(row)


if __name__ == "__main__":
    main()
