"""Train and inspect an OpenWebText byte-level BPE vocabulary."""

from __future__ import annotations

import argparse
import json
import pickle
import resource
import time
from pathlib import Path

from cs336_basics.bpe import train_bpe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/owt_train.txt"))
    parser.add_argument("--vocab-size", type=int, default=32_000)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/owt_bpe"))
    return parser.parse_args()


def token_description(token: bytes) -> dict[str, object]:
    return {
        "bytes": repr(token),
        "utf8": token.decode("utf-8", errors="replace"),
        "byte_length": len(token),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    vocab, merges = train_bpe(args.input, args.vocab_size, special_tokens=[])
    elapsed_seconds = time.perf_counter() - start

    with (args.output_dir / "vocab.pkl").open("wb") as file:
        pickle.dump(vocab, file)
    with (args.output_dir / "merges.pkl").open("wb") as file:
        pickle.dump(merges, file)

    tokens_by_length = sorted(vocab.values(), key=lambda token: (len(token), token), reverse=True)
    report = {
        "input": str(args.input),
        "vocab_size": len(vocab),
        "merge_count": len(merges),
        "elapsed_seconds": elapsed_seconds,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
        "longest_token": token_description(tokens_by_length[0]),
        "twenty_longest_tokens": [token_description(token) for token in tokens_by_length[:20]],
        "mean_token_byte_length": sum(map(len, vocab.values())) / len(vocab),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
