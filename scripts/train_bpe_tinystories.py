"""Train and inspect a TinyStories byte-level BPE vocabulary."""

from __future__ import annotations

import argparse
import cProfile
import json
import pickle
import resource
import time
from pathlib import Path

from cs336_basics.bpe import train_bpe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/TinyStoriesV2-GPT4-train.txt"))
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/tinystories_bpe"))
    parser.add_argument("--profile", action="store_true", help="Save cProfile data alongside the tokenizer.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    special_tokens = ["<|endoftext|>"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    profiler = cProfile.Profile() if args.profile else None
    start = time.perf_counter()
    if profiler is not None:
        profiler.enable()
    vocab, merges = train_bpe(args.input, args.vocab_size, special_tokens)
    if profiler is not None:
        profiler.disable()
    elapsed_seconds = time.perf_counter() - start

    with (args.output_dir / "vocab.pkl").open("wb") as file:
        pickle.dump(vocab, file)
    with (args.output_dir / "merges.pkl").open("wb") as file:
        pickle.dump(merges, file)
    if profiler is not None:
        profiler.dump_stats(args.output_dir / "train.prof")

    special_bytes = {token.encode("utf-8") for token in special_tokens}
    longest = max((token for token in vocab.values() if token not in special_bytes), key=len)
    report = {
        "input": str(args.input),
        "vocab_size": len(vocab),
        "merge_count": len(merges),
        "elapsed_seconds": elapsed_seconds,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
        "longest_token_bytes": repr(longest),
        "longest_token_utf8": longest.decode("utf-8", errors="replace"),
        "longest_token_byte_length": len(longest),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
