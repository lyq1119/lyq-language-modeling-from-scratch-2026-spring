from __future__ import annotations

import torch
import triton
import triton.language as tl

from cs336_systems.ddp import DDP


def _backward_impl(q, k, v, o, do, lse, causal: bool):
    d = q.shape[-1]
    qf, kf, vf, dof = q.float(), k.float(), v.float(), do.float()
    scores = torch.matmul(qf, kf.transpose(-1, -2)) * (d ** -0.5)
    if causal:
        nq, nk = q.shape[-2], k.shape[-2]
        mask = torch.arange(nq, device=q.device)[:, None] >= torch.arange(nk, device=q.device)[None, :]
        scores = scores.masked_fill(~mask, -1e6)
    p = torch.exp(scores - lse.unsqueeze(-1))
    delta = (o.float() * dof).sum(-1, keepdim=True)
    dv = torch.matmul(p.transpose(-1, -2), dof)
    dp = torch.matmul(dof, vf.transpose(-1, -2))
    ds = p * (dp - delta)
    dq = torch.matmul(ds, kf) * (d ** -0.5)
    dk = torch.matmul(ds.transpose(-1, -2), qf) * (d ** -0.5)
    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)


_backward = torch.compile(_backward_impl)


class FlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        tile = 32
        bsz, nq, d = q.shape
        nk = k.shape[-2]
        out = torch.empty_like(q)
        lse = torch.empty((bsz, nq), dtype=torch.float32, device=q.device)
        scale = d ** -0.5
        for qs in range(0, nq, tile):
            qe = min(qs + tile, nq)
            qi = q[:, qs:qe].float()
            m = torch.full((bsz, qe - qs), -float('inf'), device=q.device)
            l = torch.zeros((bsz, qe - qs), device=q.device)
            acc = torch.zeros((bsz, qe - qs, d), device=q.device)
            for ks in range(0, nk, tile):
                ke = min(ks + tile, nk)
                s = torch.matmul(qi, k[:, ks:ke].float().transpose(-1, -2)) * scale
                if is_causal:
                    mask = torch.arange(qs, qe, device=q.device)[:, None] >= torch.arange(ks, ke, device=q.device)[None, :]
                    s = s.masked_fill(~mask, -1e6)
                new_m = torch.maximum(m, s.max(-1).values)
                p = torch.exp(s - new_m.unsqueeze(-1))
                alpha = torch.exp(m - new_m)
                l = alpha * l + p.sum(-1)
                acc = alpha.unsqueeze(-1) * acc + torch.matmul(p, v[:, ks:ke].float())
                m = new_m
            out[:, qs:qe] = (acc / l.unsqueeze(-1)).to(q.dtype)
            lse[:, qs:qe] = m + torch.log(l)
        ctx.is_causal = is_causal
        ctx.save_for_backward(lse, q, k, v, out)
        return out

    @staticmethod
    def backward(ctx, do):
        lse, q, k, v, out = ctx.saved_tensors
        dq, dk, dv = _backward(q, k, v, out, do, lse, ctx.is_causal)
        return dq, dk, dv, None


@triton.jit
def flash_fwd_kernel(q_ptr, k_ptr, v_ptr, o_ptr, l_ptr,
                     sqb, sqq, sqd, skb, skk, skd, svb, svk, svd,
                     sob, soq, sod, slb, slq,
                     NQ: tl.constexpr, NK: tl.constexpr, scale: tl.constexpr,
                     D: tl.constexpr, BQ: tl.constexpr, BK: tl.constexpr,
                     CAUSAL: tl.constexpr):
    q_block = tl.program_id(0)
    b = tl.program_id(1)
    qi = q_block * BQ + tl.arange(0, BQ)
    di = tl.arange(0, D)
    q = tl.load(q_ptr + b * sqb + qi[:, None] * sqq + di[None, :] * sqd)
    m = tl.full((BQ,), -float('inf'), tl.float32)
    l = tl.zeros((BQ,), tl.float32)
    acc = tl.zeros((BQ, D), tl.float32)
    for key_start in range(0, NK, BK):
        ki = key_start + tl.arange(0, BK)
        k = tl.load(k_ptr + b * skb + ki[:, None] * skk + di[None, :] * skd)
        v = tl.load(v_ptr + b * svb + ki[:, None] * svk + di[None, :] * svd)
        s = tl.dot(q, tl.trans(k)) * scale
        if CAUSAL:
            s = tl.where(qi[:, None] >= ki[None, :], s, -1.0e6)
        new_m = tl.maximum(m, tl.max(s, axis=1))
        p = tl.exp(s - new_m[:, None])
        alpha = tl.exp(m - new_m)
        l = alpha * l + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(v.dtype), v, acc=acc)
        m = new_m
    tl.store(o_ptr + b * sob + qi[:, None] * soq + di[None, :] * sod, acc / l[:, None])
    tl.store(l_ptr + b * slb + qi * slq, m + tl.log(l))


class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        if not q.is_cuda:
            raise ValueError('Triton implementation needs CUDA tensors')
        bsz, nq, d = q.shape
        nk = k.shape[-2]
        if nq % 32 or nk % 32:
            raise ValueError('sequence lengths must be multiples of 32')
        out = torch.empty_like(q)
        lse = torch.empty((bsz, nq), dtype=torch.float32, device=q.device)
        flash_fwd_kernel[(triton.cdiv(nq, 32), bsz)](
            q, k, v, out, lse,
            *q.stride(), *k.stride(), *v.stride(), *out.stride(), *lse.stride(),
            nq, nk, d ** -0.5, D=d, BQ=32, BK=32, CAUSAL=is_causal, num_warps=4)
        ctx.is_causal = is_causal
        ctx.save_for_backward(lse, q, k, v, out)
        return out

    @staticmethod
    def backward(ctx, do):
        lse, q, k, v, out = ctx.saved_tensors
        dq, dk, dv = _backward(q, k, v, out, do, lse, ctx.is_causal)
        return dq, dk, dv, None


def get_flashattention_autograd_function_pytorch() -> type:
    return FlashAttentionPytorch


def get_flashattention_autograd_function_triton() -> type:
    return FlashAttentionTriton


def get_ddp(module: torch.nn.Module) -> torch.nn.Module:
    """
    Returns a torch.nn.Module container that handles
    parameter broadcasting and gradient synchronization for
    distributed data parallel training.

    This container should overlaps communication with backprop computation
    by asynchronously communicating gradients as they are ready
    in the backward pass. The gradient for each parameter tensor
    is individually communicated.

    Args:
        module: torch.nn.Module
            Underlying model to wrap with DDP.
    Returns:
        Instance of a DDP class.
    """
    return DDP(module)


def ddp_on_after_backward(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    Code to run after the backward pass is completed, but before we take
    an optimizer step.

    Args:
        ddp_model: torch.nn.Module
            DDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the DDP-wrapped model.
    """
    ddp_model.finish_gradient_synchronization()


def get_fsdp(module: torch.nn.Module, compute_dtype: torch.dtype | None = None) -> torch.nn.Module:
    """
    Returns a torch.nn.Module container that handles
    fully-sharded data parallel training, including weight sharding,
    all-gather for forward/backward, and gradient reduce-scatter.

    Args:
        module: torch.nn.Module
            Underlying model to wrap with FSDP.
        compute_dtype: optional torch.dtype
            If provided, weights are cast to this dtype before communication
            and compute, saving bandwidth. Master weights stay in fp32.
    Returns:
        Instance of an FSDP class.
    """
    from cs336_systems.fsdp import FSDP

    return FSDP(module, compute_dtype=compute_dtype)


def fsdp_on_after_backward(fsdp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    Code to run after the backward pass is completed, but before we take
    an optimizer step.

    Args:
        fsdp_model: torch.nn.Module
            FSDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the FSDP-wrapped model.
    """
    fsdp_model.finish_gradient_synchronization()


def fsdp_gather_full_params(fsdp_model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """
    All-gather sharded parameters from the FSDP model to reconstruct full
    parameter tensors. Replicated parameters are returned as-is.

    Args:
        fsdp_model: torch.nn.Module
            FSDP-wrapped model.
    Returns:
        State dictionary mapping parameter names to full (unsharded) tensors.
    """
    return fsdp_model.gather_full_params()


def get_sharded_optimizer(params, optimizer_cls: type[torch.optim.Optimizer], **kwargs) -> torch.optim.Optimizer:
    """
    Returns a torch.optim.Optimizer that handles optimizer state sharding
    of the given optimizer_cls on the provided parameters.

    Arguments:
        params (``Iterable``): an ``Iterable`` of :class:`torch.Tensor` s
            or :class:`dict` s giving all parameters, which will be sharded
            across ranks.
        optimizer_class (:class:`torch.nn.Optimizer`): the class of the local
            optimizer.
    Keyword arguments:
        kwargs: keyword arguments to be forwarded to the optimizer constructor.
    Returns:
        Instance of sharded optimizer.
    """
    from cs336_systems.sharded_optimizer import ShardedOptimizer

    return ShardedOptimizer(params, optimizer_cls, **kwargs)
