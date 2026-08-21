"""Generate text from a trained assignment Transformer checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cs336_basics.generation import generate
from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import Tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--merges", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--eos-token", default="<|endoftext|>")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    tokenizer = Tokenizer.from_files(
        args.vocab,
        args.merges,
        special_tokens=[args.eos_token] if args.eos_token else None,
    )
    device = torch.device(args.device)
    model = TransformerLM(
        vocab_size=int(config["vocab_size"]),
        context_length=int(config["context_length"]),
        d_model=int(config["d_model"]),
        num_layers=int(config["num_layers"]),
        num_heads=int(config["num_heads"]),
        d_ff=int(config["d_ff"]),
        rope_theta=float(config["rope_theta"]),
        use_rmsnorm=not bool(config.get("no_rmsnorm", False)),
        norm_first=not bool(config.get("post_norm", False)),
        use_rope=not bool(config.get("no_rope", False)),
        use_swiglu=not bool(config.get("silu_ffn", False)),
        device=device,
    )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])

    prompt_ids = tokenizer.encode(args.prompt)
    eos_token_id = (
        tokenizer._special_to_id.get(args.eos_token) if args.eos_token else None
    )
    random_generator = torch.Generator(device=device).manual_seed(args.seed)
    generated_ids = generate(
        model,
        torch.tensor(prompt_ids, dtype=torch.long, device=device),
        args.max_new_tokens,
        eos_token_id=eos_token_id,
        temperature=args.temperature,
        top_p=args.top_p,
        generator=random_generator,
    )
    print(tokenizer.decode(generated_ids.tolist()))


if __name__ == "__main__":
    main()
