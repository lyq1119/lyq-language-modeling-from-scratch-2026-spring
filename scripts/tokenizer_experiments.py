"""Measure tokenizer compression ratios on TinyStories and OpenWebText samples."""

from __future__ import annotations

import argparse
from pathlib import Path

from cs336_basics.tokenizer import Tokenizer


END_OF_TEXT = "<|endoftext|>"
DATASETS = (
    (
        "TinyStories",
        Path("data/TinyStoriesV2-GPT4-train.txt"),
        Path("artifacts/tinystories_bpe"),
    ),
    (
        "OpenWebText",
        Path("data/owt_train.txt"),
        Path("artifacts/owt_bpe"),
    ),
)


def read_first_documents(path: Path, count: int) -> str:
    """Read the first ``count`` documents without loading the whole corpus."""
    documents: list[str] = []
    current_document: list[str] = []

    with path.open(encoding="utf-8") as source:
        for line in source:
            parts = line.split(END_OF_TEXT)
            for index, part in enumerate(parts):
                current_document.append(part)
                if index < len(parts) - 1:
                    documents.append("".join(current_document))
                    current_document.clear()
                    if len(documents) == count:
                        return END_OF_TEXT.join(documents)

    if current_document and len(documents) < count:
        documents.append("".join(current_document))

    if len(documents) < count:
        raise ValueError(f"{path} contains only {len(documents)} documents; requested {count}")
    return END_OF_TEXT.join(documents)


def measure_compression(
    data_path: Path,
    tokenizer_dir: Path,
    document_count: int,
) -> tuple[int, int, float]:
    sample = read_first_documents(data_path, document_count)
    return measure_text(sample, tokenizer_dir)


def measure_text(text: str, tokenizer_dir: Path) -> tuple[int, int, float]:
    """Return UTF-8 bytes, token count, and bytes per token for some text."""
    tokenizer = Tokenizer.from_files(
        tokenizer_dir / "vocab.pkl",
        tokenizer_dir / "merges.pkl",
        special_tokens=[END_OF_TEXT],
    )

    byte_count = len(text.encode("utf-8"))
    token_count = len(tokenizer.encode(text))
    return byte_count, token_count, byte_count / token_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute UTF-8 bytes per token for the two training corpora."
    )
    parser.add_argument(
        "part",
        choices=("a", "b"),
        help="experiment part to run",
    )
    parser.add_argument(
        "--documents",
        type=int,
        default=10,
        help="number of documents to take from the beginning of each corpus (default: 10)",
    )
    args = parser.parse_args()
    if args.documents <= 0:
        parser.error("--documents must be positive")
    return args


def main() -> None:
    args = parse_args()
    if args.part == "a":
        run_part_a(args.documents)
    else:
        run_part_b(args.documents)


def run_part_a(document_count: int) -> None:
    for name, data_path, tokenizer_dir in DATASETS:
        byte_count, token_count, ratio = measure_compression(
            data_path, tokenizer_dir, document_count
        )
        print(
            f"{name}: documents={document_count}, bytes={byte_count:,}, "
            f"tokens={token_count:,}, compression_ratio={ratio:.2f} bytes/token"
        )


def run_part_b(document_count: int) -> None:
    owt_sample = read_first_documents(Path("data/owt_train.txt"), document_count)
    _, native_tokens, native_ratio = measure_text(owt_sample, Path("artifacts/owt_bpe"))
    byte_count, tiny_tokens, tiny_ratio = measure_text(
        owt_sample, Path("artifacts/tinystories_bpe")
    )
    token_increase = (tiny_tokens / native_tokens - 1) * 100
    ratio_decrease = (1 - tiny_ratio / native_ratio) * 100

    print(
        f"OWT tokenizer:       bytes={byte_count:,}, tokens={native_tokens:,}, "
        f"compression_ratio={native_ratio:.2f} bytes/token"
    )
    print(
        f"TinyStories tokenizer: bytes={byte_count:,}, tokens={tiny_tokens:,}, "
        f"compression_ratio={tiny_ratio:.2f} bytes/token"
    )
    print(
        f"Change: tokens +{token_increase:.1f}%, "
        f"compression ratio -{ratio_decrease:.1f}%"
    )
    print(
        "Qualitative result: the TinyStories vocabulary is less suited to web text, "
        "so technical terms, proper nouns, code, and other rare strings are split "
        "into more and shorter byte-level subwords."
    )


if __name__ == "__main__":
    main()
