"""Autoregressive decoding utilities for language models."""

from __future__ import annotations

import torch


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample token IDs from one or more next-token logit vectors."""
    if logits.ndim < 1:
        raise ValueError("logits must have at least one dimension")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if temperature == 0:
        return logits.argmax(dim=-1)

    scaled_logits = logits / temperature
    scaled_logits = scaled_logits - scaled_logits.amax(dim=-1, keepdim=True)
    probabilities = scaled_logits.exp()
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)

    if top_p < 1:
        sorted_probabilities, sorted_indices = probabilities.sort(dim=-1, descending=True)
        cumulative = sorted_probabilities.cumsum(dim=-1)
        remove = cumulative - sorted_probabilities >= top_p
        sorted_probabilities = sorted_probabilities.masked_fill(remove, 0)
        sorted_probabilities = sorted_probabilities / sorted_probabilities.sum(
            dim=-1, keepdim=True
        )
        sampled_sorted_index = torch.multinomial(
            sorted_probabilities, 1, generator=generator
        ).squeeze(-1)
        return sorted_indices.gather(-1, sampled_sorted_index.unsqueeze(-1)).squeeze(-1)

    return torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    prompt_token_ids: torch.Tensor,
    max_new_tokens: int,
    *,
    eos_token_id: int | None = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Append sampled tokens to a one-dimensional tokenized prompt."""
    if prompt_token_ids.ndim != 1 or prompt_token_ids.numel() == 0:
        raise ValueError("prompt_token_ids must be a non-empty one-dimensional tensor")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")

    device = next(model.parameters()).device
    tokens = prompt_token_ids.to(device=device, dtype=torch.long)
    context_length = getattr(model, "context_length", None)
    was_training = model.training
    model.eval()
    try:
        for _ in range(max_new_tokens):
            model_input = tokens[-context_length:] if context_length is not None else tokens
            logits = model(model_input.unsqueeze(0))[0, -1]
            next_token = sample_next_token(
                logits,
                temperature=temperature,
                top_p=top_p,
                generator=generator,
            )
            tokens = torch.cat((tokens, next_token.reshape(1)))
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
    finally:
        model.train(was_training)
    return tokens
