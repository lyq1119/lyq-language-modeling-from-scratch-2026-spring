"""Stream the TinyStories and OpenWebText splits into uint16 token arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cs336_basics.tokenizer import Tokenizer


DATASETS = (
    ("tinystories_train", "data/TinyStoriesV2-GPT4-train.txt", "artifacts/tinystories_bpe", ["<|endoftext|>"]),
    ("tinystories_valid", "data/TinyStoriesV2-GPT4-valid.txt", "artifacts/tinystories_bpe", ["<|endoftext|>"]),
    ("owt_train", "data/owt_train.txt", "artifacts/owt_bpe", None),
    ("owt_valid", "data/owt_valid.txt", "artifacts/owt_bpe", None),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/tokenized"))
    parser.add_argument("--buffer-tokens", type=int, default=1_000_000)
    return parser.parse_args()


def encode_file(tokenizer: Tokenizer, input_path: Path, output_path: Path, buffer_tokens: int) -> int:
    count = 0
    buffer: list[int] = []
    with input_path.open(encoding="utf-8") as source, output_path.open("wb") as destination:
        for token_id in tokenizer.encode_iterable(source):
            if token_id >= 2**16:
                raise ValueError(f"token ID {token_id} does not fit in uint16")
            buffer.append(token_id)
            if len(buffer) >= buffer_tokens:
                np.asarray(buffer, dtype=np.uint16).tofile(destination)
                count += len(buffer)
                buffer.clear()
        if buffer:
            np.asarray(buffer, dtype=np.uint16).tofile(destination)
            count += len(buffer)
    return count


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for name, input_name, artifact_name, special_tokens in DATASETS:
        artifact_dir = Path(artifact_name)
        tokenizer = Tokenizer.from_files(
            artifact_dir / "vocab.pkl", artifact_dir / "merges.pkl", special_tokens
        )
        output_path = args.output_dir / f"{name}.uint16.bin"
        count = encode_file(tokenizer, Path(input_name), output_path, args.buffer_tokens)
        report[name] = {
            "input": input_name,
            "output": str(output_path),
            "dtype": "uint16",
            "token_count": count,
        }
        print(f"{name}: {count} tokens -> {output_path}", flush=True)

    (args.output_dir / "metadata.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
