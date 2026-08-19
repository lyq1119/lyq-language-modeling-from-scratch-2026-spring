"""Utilities used to train a Transformer language model."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable

import torch


def get_batch(
    dataset,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly sample next-token-prediction examples from a 1D token array."""
    if getattr(dataset, "ndim", None) != 1:
        raise ValueError("dataset must be a one-dimensional token array")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if context_length <= 0:
        raise ValueError("context_length must be positive")

    num_start_positions = len(dataset) - context_length
    if num_start_positions <= 0:
        raise ValueError("dataset must contain more tokens than context_length")

    starts = torch.randint(0, num_start_positions, (batch_size,))
    # Copy each slice because np.memmap slices may be read-only. Constructing a
    # tensor this way also converts uint16 token IDs safely to torch.long.
    inputs = torch.stack(
        [torch.as_tensor(dataset[start : start + context_length].copy(), dtype=torch.long)
         for start in starts.tolist()]
    )
    targets = torch.stack(
        [torch.as_tensor(dataset[start + 1 : start + context_length + 1].copy(), dtype=torch.long)
         for start in starts.tolist()]
    )
    return inputs.to(device), targets.to(device)


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return mean cross-entropy over all leading (batch-like) dimensions."""
    if logits.shape[:-1] != targets.shape:
        raise ValueError(
            f"targets must have shape {logits.shape[:-1]}, got {targets.shape}"
        )
    shifted = logits - logits.amax(dim=-1, keepdim=True)
    target_logits = shifted.gather(-1, targets.long().unsqueeze(-1)).squeeze(-1)
    losses = torch.logsumexp(shifted, dim=-1) - target_logits
    return losses.mean()


def clip_gradients(
    parameters: Iterable[torch.nn.Parameter], max_l2_norm: float
) -> None:
    """Clip the joint L2 norm of all present gradients in place."""
    if max_l2_norm < 0:
        raise ValueError("max_l2_norm must be non-negative")
    gradients = [p.grad for p in parameters if p.grad is not None]
    if not gradients:
        return

    # Accumulating individual norms avoids concatenating potentially huge gradients.
    total_norm = torch.linalg.vector_norm(
        torch.stack([torch.linalg.vector_norm(g.detach(), 2) for g in gradients]), 2
    )
    coefficient = max_l2_norm / (total_norm + 1e-6)
    if coefficient < 1:
        for gradient in gradients:
            gradient.mul_(coefficient.to(device=gradient.device, dtype=gradient.dtype))


class AdamW(torch.optim.Optimizer):
    """Adam with decoupled weight decay, following the assignment pseudocode."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError(f"Invalid beta values: {betas}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight decay: {weight_decay}")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                state["step"] += 1
                step = state["step"]
                first_moment = state["exp_avg"]
                second_moment = state["exp_avg_sq"]

                parameter.mul_(1 - lr * weight_decay)
                first_moment.mul_(beta1).add_(gradient, alpha=1 - beta1)
                second_moment.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
                adjusted_lr = lr * math.sqrt(1 - beta2**step) / (1 - beta1**step)
                parameter.addcdiv_(first_moment, second_moment.sqrt().add_(eps), value=-adjusted_lr)
        return loss


def cosine_learning_rate(
    iteration: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """Linear warmup followed by cosine decay and a constant tail."""
    if warmup_iters < 0 or cosine_cycle_iters < warmup_iters:
        raise ValueError("Require 0 <= warmup_iters <= cosine_cycle_iters")
    if iteration < warmup_iters:
        if warmup_iters == 0:
            return max_learning_rate
        return iteration / warmup_iters * max_learning_rate
    if iteration <= cosine_cycle_iters:
        if cosine_cycle_iters == warmup_iters:
            return min_learning_rate
        progress = (iteration - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate + 0.5 * (1 + math.cos(math.pi * progress)) * (
            max_learning_rate - min_learning_rate
        )
    return min_learning_rate
