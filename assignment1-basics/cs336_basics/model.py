from __future__ import annotations

import math

import torch
from einops import einsum, rearrange, reduce
from torch import nn


class Linear(nn.Module):
    """A linear transformation without a bias term."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        std = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    """A learned lookup table that maps token IDs to embedding vectors."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """Root mean square layer normalization over the final dimension."""

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(reduce(x.square(), "... d_model -> ... 1", "mean") + self.eps)
        result = x / rms * self.weight
        return result.to(in_dtype)


def silu(x: torch.Tensor) -> torch.Tensor:
    """Apply the SiLU activation element-wise."""

    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    """Position-wise feed-forward network using a SwiGLU activation."""

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_ff is None:
            d_ff = math.ceil((8 / 3 * d_model) / 64) * 64

        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))


class SiLUFeedForward(nn.Module):
    """Ungated position-wise feed-forward network with a SiLU activation."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)))


class RotaryPositionalEmbedding(nn.Module):
    """Apply rotary position embeddings to adjacent feature pairs."""

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError(f"d_k must be even, got {d_k}")

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        pair_indices = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)
        inverse_frequencies = theta ** (-pair_indices / d_k)
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        angles = einsum(positions, inverse_frequencies, "position, pair -> position pair")
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.d_k:
            raise ValueError(f"expected input dimension {self.d_k}, got {x.shape[-1]}")

        cos = self.cos[token_positions].to(dtype=x.dtype)
        sin = self.sin[token_positions].to(dtype=x.dtype)
        while cos.ndim < x.ndim:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)

        x_pairs = rearrange(x, "... (pair two) -> ... pair two", two=2)
        x_even, x_odd = x_pairs.unbind(dim=-1)
        rotated = torch.stack(
            (x_even * cos - x_odd * sin, x_even * sin + x_odd * cos),
            dim=-1,
        )
        return rearrange(rotated, "... pair two -> ... (pair two)")


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Compute a numerically stable softmax along ``dim``."""

    shifted = x - x.max(dim=dim, keepdim=True).values
    exponentials = shifted.exp()
    return exponentials / exponentials.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute scaled dot-product attention over arbitrary leading dimensions."""

    d_k = query.shape[-1]
    scores = einsum(
        query,
        key,
        "... query d_k, ... key d_k -> ... query key",
    ) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, -torch.inf)
    attention = softmax(scores, dim=-1)
    return einsum(
        attention,
        value,
        "... query key, ... key d_v -> ... query d_v",
    )


class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention, optionally with RoPE."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float | None = None,
        max_seq_len: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        if (theta is None) != (max_seq_len is None):
            raise ValueError("theta and max_seq_len must either both be provided or both be omitted")
        self.rope = (
            RotaryPositionalEmbedding(theta, self.d_head, max_seq_len, device=device)
            if theta is not None and max_seq_len is not None
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query, key, value = (
            rearrange(
                projection(x),
                "... sequence (head d_head) -> ... head sequence d_head",
                head=self.num_heads,
            )
            for projection in (self.q_proj, self.k_proj, self.v_proj)
        )
        sequence_length = x.shape[-2]
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(sequence_length, device=x.device)
            query = self.rope(query, token_positions)
            key = self.rope(key, token_positions)

        causal_mask = torch.ones(
            sequence_length,
            sequence_length,
            dtype=torch.bool,
            device=x.device,
        ).tril()
        attended = scaled_dot_product_attention(query, key, value, mask=causal_mask)
        attended = rearrange(attended, "... head sequence d_head -> ... sequence (head d_head)")
        return self.output_proj(attended)


class TransformerBlock(nn.Module):
    """A pre-norm Transformer block with causal attention and SwiGLU."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        use_rmsnorm: bool = True,
        norm_first: bool = True,
        use_rope: bool = True,
        use_swiglu: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.ln1 = (
            RMSNorm(d_model, device=device, dtype=dtype) if use_rmsnorm else nn.Identity()
        )
        self.attn = MultiHeadSelfAttention(
            d_model,
            num_heads,
            theta=theta if use_rope else None,
            max_seq_len=max_seq_len if use_rope else None,
            device=device,
            dtype=dtype,
        )
        self.ln2 = (
            RMSNorm(d_model, device=device, dtype=dtype) if use_rmsnorm else nn.Identity()
        )
        ffn_type = SwiGLU if use_swiglu else SiLUFeedForward
        self.ffn = ffn_type(d_model, d_ff, device=device, dtype=dtype)
        self.norm_first = norm_first

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.norm_first:
            x = x + self.attn(self.ln1(x), token_positions)
            return x + self.ffn(self.ln2(x))
        x = self.ln1(x + self.attn(x, token_positions))
        return self.ln2(x + self.ffn(x))


class TransformerLM(nn.Module):
    """A decoder-only Transformer language model."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        use_rmsnorm: bool = True,
        norm_first: bool = True,
        use_rope: bool = True,
        use_swiglu: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    context_length,
                    rope_theta,
                    use_rmsnorm=use_rmsnorm,
                    norm_first=norm_first,
                    use_rope=use_rope,
                    use_swiglu=use_swiglu,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = (
            RMSNorm(d_model, device=device, dtype=dtype) if use_rmsnorm else nn.Identity()
        )
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        sequence_length = token_ids.shape[-1]
        if sequence_length > self.context_length:
            raise ValueError(
                f"sequence length {sequence_length} exceeds context length {self.context_length}"
            )

        token_positions = torch.arange(sequence_length, device=token_ids.device)
        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x, token_positions)
        return self.lm_head(self.ln_final(x))
