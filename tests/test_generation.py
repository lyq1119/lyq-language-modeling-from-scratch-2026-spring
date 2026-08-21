import torch

from cs336_basics.generation import generate, sample_next_token


class _NextTokenModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 8, context_length: int = 3):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.seen_lengths: list[int] = []

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        self.seen_lengths.append(token_ids.shape[-1])
        logits = torch.full(
            (*token_ids.shape, self.vocab_size),
            -100.0,
            device=token_ids.device,
        )
        next_ids = (token_ids + 1) % self.vocab_size
        return logits.scatter(-1, next_ids.unsqueeze(-1), 100.0) + self.anchor


def test_sample_next_token_greedy_and_top_p():
    logits = torch.tensor([1.0, 4.0, 2.0])
    assert sample_next_token(logits, temperature=0).item() == 1
    generator = torch.Generator().manual_seed(0)
    assert sample_next_token(logits, top_p=0.01, generator=generator).item() == 1


def test_generate_stops_at_eos_and_crops_context():
    model = _NextTokenModel()
    result = generate(
        model,
        torch.tensor([6, 7, 0, 1]),
        max_new_tokens=10,
        eos_token_id=3,
        temperature=0,
    )
    assert result.tolist() == [6, 7, 0, 1, 2, 3]
    assert model.seen_lengths == [3, 3]


def test_generate_restores_training_mode():
    model = _NextTokenModel()
    model.train()
    generate(model, torch.tensor([0]), max_new_tokens=1, temperature=0)
    assert model.training
