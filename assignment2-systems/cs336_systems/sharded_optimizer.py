"""Optimizer state sharding.

``ShardedOptimizer`` wraps an arbitrary :class:`torch.optim.Optimizer` and
shards the optimizer state across ranks: each rank's inner optimizer only
maintains optimizer state for the subset of parameters it owns (about
``1 / world_size`` of all parameters).  After every optimizer step the owning
rank broadcasts its updated parameters to every other rank, so the model
parameters stay synchronized across ranks.

This is a simplified version of ZeRO-DP ``P_{os}``: gradients are *not*
sharded (every rank still computes the full gradient of every parameter); only
the optimizer state and the resulting parameter updates are distributed.
"""

from __future__ import annotations

from typing import Any, Iterable, Type

import torch
import torch.distributed as dist
import torch.nn as nn


class ShardedOptimizer(torch.optim.Optimizer):
    """Wrap ``optimizer_cls`` and shard its optimizer state across ranks."""

    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
        optimizer_cls: Type[torch.optim.Optimizer],
        **kwargs: Any,
    ):
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = dict(kwargs)

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        # Shard assignment: every unique parameter is owned by exactly one rank.
        # ``self._all_params`` preserves the (rank-independent) order in which
        # parameters first appear, so the broadcast collectives in ``step`` are
        # issued in a globally consistent order.
        self._shard_of_param: dict[int, int] = {}
        self._sharded_params: list[list[nn.Parameter]] = [[] for _ in range(self.world_size)]
        self._all_params: list[nn.Parameter] = []

        # super().__init__ calls self.add_param_group; guard the inner optimizer
        # construction with a flag so that add_param_group knows not to touch it
        # until the inner optimizer exists.
        self._initializing = True
        super().__init__(params, dict(kwargs))
        self._initializing = False

        # The inner optimizer only ever sees the parameters owned by this rank,
        # preserving per-group hyper-parameters.
        owned_groups = []
        for group in self.param_groups:
            owned = self._owned_params(group["params"])
            if owned:
                g = {k: v for k, v in group.items() if k != "params"}
                g["params"] = owned
                owned_groups.append(g)
        self.optimizer = optimizer_cls(owned_groups, **kwargs)

    def _owned_params(self, params: Iterable[torch.Tensor]) -> list[nn.Parameter]:
        """Return the (deduplicated) parameters of ``params`` owned by this rank."""
        seen: set[int] = set()
        owned: list[nn.Parameter] = []
        for p in params:
            key = id(p)
            if key not in self._shard_of_param or self._shard_of_param[key] != self.rank:
                continue
            if key in seen:
                continue
            seen.add(key)
            owned.append(p)
        return owned

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        # Materialize the (possibly generator) ``params`` list so that it can be
        # iterated more than once (``Optimizer.__init__`` passes generators
        # through as-is).
        param_group = dict(param_group)
        param_group["params"] = list(param_group["params"])
        super().add_param_group(param_group)

        # Assign shards for any new unique parameters.
        for p in param_group["params"]:
            key = id(p)
            if key not in self._shard_of_param:
                self._shard_of_param[key] = len(self._all_params) % self.world_size
                self._sharded_params[self._shard_of_param[key]].append(p)
                self._all_params.append(p)

        # If the inner optimizer already exists (a post-init call, e.g. while
        # gradually unfreezing layers), forward the group's owned parameters.
        if not self._initializing:
            owned = self._owned_params(param_group["params"])
            if owned:
                g = {k: v for k, v in self.param_groups[-1].items() if k != "params"}
                g["params"] = owned
                self.optimizer.add_param_group(g)

    def step(self, closure=None, **kwargs):
        """Run the wrapped optimizer's ``step``, then synchronize parameters."""
        loss = self.optimizer.step(closure=closure, **kwargs)
        # Every rank iterates the full parameter list in the same order, so the
        # broadcast collectives are matched across ranks.  Each parameter is
        # broadcast from its owning rank.
        for p in self._all_params:
            dist.broadcast(p.data, src=self._shard_of_param[id(p)])
        return loss
