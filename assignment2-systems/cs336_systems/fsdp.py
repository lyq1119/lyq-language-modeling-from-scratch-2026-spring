"""Fully Sharded Data Parallel (FSDP).

``FSDP`` wraps an arbitrary :class:`torch.nn.Module` and shards the weights of
every :class:`~cs336_basics.model.Linear` and
:class:`~cs336_basics.model.Embedding` layer across ranks.  Other parameters
(normalization layers, biases, ...) are replicated.

Before the forward/backward pass uses a sharded weight, its shards are
all-gathered into a full weight tensor (a custom autograd function keeps the
gradient flow wired back to the shard).  When the gradient for a sharded
weight becomes available, it is reduce-scattered (averaged) back onto the
shards; replicated parameter gradients are all-reduced (averaged).  Master
weights are kept in FP32; when ``compute_dtype`` is provided they are cast
right after gathering, saving communication bandwidth.
"""

from __future__ import annotations

import math

import torch
import torch.distributed as dist
import torch.nn as nn

from cs336_basics.model import Embedding, Linear


class _AllGather(torch.autograd.Function):
    """All-gather shards along the flattened dimension with autograd support.

    Forward:  gather this rank's shard from every rank, concatenate, and trim
    any padding used to make the shard length divide ``world_size``.
    Backward: reduce-scatter the full gradient back onto the shards
    (averaged across ranks, matching DDP semantics).
    """

    @staticmethod
    def forward(ctx, shard: torch.Tensor, world_size: int, numel: int) -> torch.Tensor:
        shard_len = shard.numel()
        pad_len = shard_len * world_size
        gathered = [torch.empty_like(shard) for _ in range(world_size)]
        dist.all_gather(gathered, shard)
        full = torch.cat(gathered)[:numel]
        ctx.world_size = world_size
        ctx.shard_len = shard_len
        ctx.pad_len = pad_len
        ctx.numel = numel
        return full

    @staticmethod
    def backward(ctx, grad_full: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        g = grad_full.new_zeros(ctx.pad_len)
        g[: ctx.numel] = grad_full
        g = g.reshape(ctx.world_size, ctx.shard_len)
        out = g.new_empty(ctx.shard_len)
        dist.reduce_scatter(out, list(g.unbind(0)), op=dist.ReduceOp.SUM)
        out.div_(ctx.world_size)
        return out, None, None


class _ShardedLinear(Linear):
    """Drop-in replacement for a sharded ``Linear``.

    Subclasses :class:`cs336_basics.model.Linear` so that
    ``isinstance(module, Linear)`` checks keep working.  ``self.weight`` holds
    this rank's shard (an ``nn.Parameter``), so it shows up in
    ``named_parameters()`` under the original name (e.g. ``linear1.weight``).
    """

    def __init__(
        self,
        shard: nn.Parameter,
        full_shape: tuple[int, ...],
        compute_dtype: torch.dtype | None,
        world_size: int,
    ):
        # Skip Linear.__init__ so that we do not allocate a full-size weight.
        nn.Module.__init__(self)
        self.weight = shard
        self.full_shape = tuple(full_shape)
        self.numel = math.prod(full_shape)
        self.compute_dtype = compute_dtype
        self.world_size = world_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = _AllGather.apply(self.weight, self.world_size, self.numel)
        w = w.view(self.full_shape)
        if self.compute_dtype is not None:
            w = w.to(self.compute_dtype)
            x = x.to(self.compute_dtype)
        return torch.einsum("...i,oi->...o", x, w)


class _ShardedEmbedding(Embedding):
    """Drop-in replacement for a sharded ``Embedding``."""

    def __init__(
        self,
        shard: nn.Parameter,
        full_shape: tuple[int, ...],
        compute_dtype: torch.dtype | None,
        world_size: int,
    ):
        nn.Module.__init__(self)
        self.weight = shard
        self.full_shape = tuple(full_shape)
        self.numel = math.prod(full_shape)
        self.compute_dtype = compute_dtype
        self.world_size = world_size

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        w = _AllGather.apply(self.weight, self.world_size, self.numel)
        w = w.view(self.full_shape)
        if self.compute_dtype is not None:
            w = w.to(self.compute_dtype)
        return w[token_ids, :]


class FSDP(nn.Module):
    """Wrap a module with fully-sharded data parallel training."""

    def __init__(self, module: nn.Module, compute_dtype: torch.dtype | None = None):
        super().__init__()
        self.module = module
        self.compute_dtype = compute_dtype
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        # Maps sharded parameter id -> metadata needed to rebuild the full tensor.
        self._shard_ids: set[int] = set()
        self._shard_shapes: dict[int, tuple[int, ...]] = {}
        self._shard_numel: dict[int, int] = {}

        # Every remaining (replicated) parameter, e.g. normalization weights.
        self._replicated_params: list[nn.Parameter] = []

        self._shard_modules(module)

        for p in module.parameters():
            if id(p) not in self._shard_ids:
                self._replicated_params.append(p)

    def _shard_modules(self, root: nn.Module) -> None:
        for name, child in list(root._modules.items()):
            if isinstance(child, (Linear, Embedding)):
                root._modules[name] = self._wrap(child)
            else:
                self._shard_modules(child)

    def _wrap(self, m: nn.Module) -> nn.Module:
        full = m.weight.data
        numel = full.numel()
        flat = full.reshape(-1)
        shard_len = (numel + self.world_size - 1) // self.world_size
        pad_len = shard_len * self.world_size
        if pad_len > numel:
            padded = flat.new_zeros(pad_len)
            padded[:numel] = flat
        else:
            padded = flat
        shard = padded.chunk(self.world_size)[self.rank].clone()
        shard_param = nn.Parameter(shard, requires_grad=True)
        self._shard_ids.add(id(shard_param))
        self._shard_shapes[id(shard_param)] = tuple(full.shape)
        self._shard_numel[id(shard_param)] = numel
        if isinstance(m, Embedding):
            return _ShardedEmbedding(shard_param, full.shape, self.compute_dtype, self.world_size)
        return _ShardedLinear(shard_param, full.shape, self.compute_dtype, self.world_size)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        """Average the gradients of replicated (non-sharded) parameters.

        Sharded parameter gradients are already reduced during the backward
        pass (inside :class:`_AllGather`), so this only needs to handle the
        replicated parameters.
        """
        for p in self._replicated_params:
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

    def gather_full_params(self) -> dict[str, torch.Tensor]:
        """Return {name: full unsharded tensor} for every parameter."""
        result: dict[str, torch.Tensor] = {}
        for name, param in self.module.named_parameters():
            key = id(param)
            if key in self._shard_ids:
                gathered = [torch.empty_like(param.data) for _ in range(self.world_size)]
                dist.all_gather(gathered, param.data)
                full = torch.cat(gathered)[: self._shard_numel[key]].view(self._shard_shapes[key])
                result[name] = full
            else:
                result[name] = param.data
        return result
