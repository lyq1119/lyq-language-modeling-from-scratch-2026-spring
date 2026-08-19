"""Compare serialized TinyStories and OpenWebText BPE vocabularies."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tinystories", type=Path, default=Path("artifacts/tinystories_bpe/vocab.pkl"))
    parser.add_argument("--owt", type=Path, default=Path("artifacts/owt_bpe/vocab.pkl"))
    return parser.parse_args()


def readable(tokens: set[bytes], limit: int = 30) -> list[str]:
    ordered = sorted(tokens, key=lambda token: (len(token), token), reverse=True)
    return [token.decode("utf-8", errors="replace") for token in ordered[:limit]]


def main() -> None:
    args = parse_args()
    with args.tinystories.open("rb") as file:
        tinystories_vocab: dict[int, bytes] = pickle.load(file)
    with args.owt.open("rb") as file:
        owt_vocab: dict[int, bytes] = pickle.load(file)

    tinystories = set(tinystories_vocab.values())
    owt = set(owt_vocab.values())
    shared = tinystories & owt
    print(f"TinyStories vocab size: {len(tinystories)}")
    print(f"OpenWebText vocab size: {len(owt)}")
    print(f"Shared token count: {len(shared)}")
    print(f"TinyStories tokens also in OWT: {len(shared) / len(tinystories):.1%}")
    print(f"Mean token byte length (TinyStories): {sum(map(len, tinystories)) / len(tinystories):.2f}")
    print(f"Mean token byte length (OWT): {sum(map(len, owt)) / len(owt):.2f}")
    print(f"Longest TinyStories-only tokens: {readable(tinystories - owt)}")
    print(f"Longest OWT-only tokens: {readable(owt - tinystories)}")


if __name__ == "__main__":
    main()
