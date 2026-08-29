"""Distributed data parallel (DDP) training container.

``DDP`` wraps an arbitrary :class:`torch.nn.Module` and takes care of two
things:

1. broadcasting the module's parameters from rank 0 before training, so that
   every rank starts from the same initial weights; and
2. averaging parameter gradients across ranks.

The gradient averaging is overlapped with the backward pass: a
``post_accumulate_grad`` hook is registered on every trainable parameter, and
the hook asynchronously all-reduces the parameter's gradient as soon as it is
ready.  Communication runs on a dedicated CUDA stream so that it can overlap
with the remaining backward computation.  Call :meth:`DDP.finish_gradient_synchronization`
before :func:`torch.optim.Optimizer.step`.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn


class DDP(nn.Module):
    """Wrap a module with overlapped distributed data parallel training."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

        # Every rank must start from the same initial parameters.
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

        # Run communication on a dedicated stream so that gradient all-reduces
        # can overlap with the rest of the backward pass on the default stream.
        self._comm_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self._pending_works: list[dist.Work] = []

        # One hook per parameter tensor (tied weights share a Parameter object,
        # so de-duplicate by tensor identity).
        seen: set[int] = set()
        for param in self.module.parameters():
            if id(param) in seen or not param.requires_grad:
                continue
            seen.add(id(param))
            param.register_post_accumulate_grad_hook(self._make_grad_hook(param))

    def _make_grad_hook(self, param: nn.Parameter):
        def _hook(_param: nn.Parameter) -> None:
            grad = param.grad
            if grad is None:
                return
            if grad.is_cuda and self._comm_stream is not None:
                # Make sure the communication stream waits until the gradient
                # has actually been produced on the current (default) stream.
                self._comm_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self._comm_stream):
                    work = dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True)
            else:
                work = dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True)
            self._pending_works.append(work)

        return _hook

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        """Wait for all pending asynchronous gradient all-reduces to finish."""
        for work in self._pending_works:
            work.wait()
        self._pending_works.clear()
        if self._comm_stream is not None:
            # Order the default stream after the communication stream so that
            # the optimizer step sees the fully-reduced gradients.
            torch.cuda.current_stream().wait_stream(self._comm_stream)
