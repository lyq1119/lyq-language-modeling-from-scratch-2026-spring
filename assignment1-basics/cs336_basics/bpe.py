"""Byte-pair encoding training."""

from __future__ import annotations

from collections import Counter, defaultdict
from os import PathLike

import regex


# This is the pre-tokenizer used by GPT-2.  Keeping the optional leading space in
# the token is important: BPE is not allowed to merge across these boundaries.
GPT2_PATTERN = regex.compile(
    r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)


def _pretoken_counts(text: str, special_tokens: list[str]) -> Counter[tuple[bytes, ...]]:
    """Return byte-token tuples and their multiplicities.

    Special tokens are separators rather than ordinary text.  Splitting once
    with an alternation also handles adjacent special tokens and prevents a
    pre-token from spanning one.
    """
    if special_tokens:
        # Longest first makes overlapping special tokens behave predictably.
        alternatives = "|".join(regex.escape(token) for token in sorted(set(special_tokens), key=len, reverse=True))
        chunks = regex.split(f"(?:{alternatives})", text)
    else:
        chunks = (text,)

    counts: Counter[tuple[bytes, ...]] = Counter()
    for chunk in chunks:
        for match in GPT2_PATTERN.finditer(chunk):
            raw = match.group().encode("utf-8")
            counts[tuple(bytes((byte,)) for byte in raw)] += 1
    return counts


def train_bpe(
    input_path: str | PathLike[str],
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a byte-level BPE vocabulary and return it with ordered merges."""
    special_tokens = list(dict.fromkeys(special_tokens))
    minimum_size = 256 + len(special_tokens)
    if vocab_size < minimum_size:
        raise ValueError(f"vocab_size must be at least {minimum_size}")

    with open(input_path, encoding="utf-8") as corpus:
        word_counts = _pretoken_counts(corpus.read(), special_tokens)

    vocab: dict[int, bytes] = {i: bytes((i,)) for i in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    words = list(word_counts)
    frequencies = [word_counts[word] for word in words]
    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    pair_words: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)

    for word_id, word in enumerate(words):
        local = Counter(zip(word, word[1:]))
        for pair, occurrences in local.items():
            pair_counts[pair] += frequencies[word_id] * occurrences
            pair_words[pair].add(word_id)

    merges: list[tuple[bytes, bytes]] = []
    while len(vocab) < vocab_size and pair_counts:
        # The assignment specifies lexicographically greatest as the tie-break.
        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
        affected = tuple(pair_words[best_pair])
        merged_token = best_pair[0] + best_pair[1]

        for word_id in affected:
            old_word = words[word_id]
            old_pairs = Counter(zip(old_word, old_word[1:]))
            for pair, occurrences in old_pairs.items():
                pair_counts[pair] -= frequencies[word_id] * occurrences
                pair_words[pair].discard(word_id)
                if pair_counts[pair] == 0:
                    del pair_counts[pair]
                    del pair_words[pair]

            new_word: list[bytes] = []
            index = 0
            while index < len(old_word):
                if index + 1 < len(old_word) and (old_word[index], old_word[index + 1]) == best_pair:
                    new_word.append(merged_token)
                    index += 2
                else:
                    new_word.append(old_word[index])
                    index += 1
            words[word_id] = tuple(new_word)

            new_pairs = Counter(zip(new_word, new_word[1:]))
            for pair, occurrences in new_pairs.items():
                pair_counts[pair] += frequencies[word_id] * occurrences
                pair_words[pair].add(word_id)

        merges.append(best_pair)
        vocab[len(vocab)] = merged_token

    return vocab, merges
