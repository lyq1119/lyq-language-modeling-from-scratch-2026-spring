"""Byte-level BPE tokenization."""

from __future__ import annotations

import pickle
from collections.abc import Iterable, Iterator
from os import PathLike

import regex

from cs336_basics.bpe import GPT2_PATTERN


class Tokenizer:
    """A byte-level BPE tokenizer with optional indivisible special tokens."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(dict.fromkeys(special_tokens or []))

        token_to_id = {token: token_id for token_id, token in self.vocab.items()}
        next_id = max(self.vocab, default=-1) + 1
        for special_token in self.special_tokens:
            token_bytes = special_token.encode("utf-8")
            if token_bytes not in token_to_id:
                self.vocab[next_id] = token_bytes
                token_to_id[token_bytes] = next_id
                next_id += 1

        self._token_to_id = token_to_id
        self._merge_rank = {pair: rank for rank, pair in enumerate(self.merges)}
        self._special_to_id = {
            token: self._token_to_id[token.encode("utf-8")] for token in self.special_tokens
        }
        if self.special_tokens:
            # Alternation is leftmost-first, so put longer overlapping tokens first.
            alternatives = "|".join(
                regex.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True)
            )
            self._special_pattern = regex.compile(f"({alternatives})")
        else:
            self._special_pattern = None

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | PathLike[str],
        merges_filepath: str | PathLike[str],
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        """Load pickle-serialized vocabulary and merge files."""
        with open(vocab_filepath, "rb") as file:
            vocab = pickle.load(file)
        with open(merges_filepath, "rb") as file:
            merges = pickle.load(file)
        return cls(vocab, merges, special_tokens)

    def _apply_bpe(self, raw: bytes) -> Iterator[int]:
        pieces = [bytes((byte,)) for byte in raw]
        while len(pieces) > 1:
            best_index = -1
            best_rank = len(self._merge_rank)
            for index, pair in enumerate(zip(pieces, pieces[1:])):
                rank = self._merge_rank.get(pair)
                if rank is not None and rank < best_rank:
                    best_index = index
                    best_rank = rank
            if best_index < 0:
                break
            pieces[best_index : best_index + 2] = [pieces[best_index] + pieces[best_index + 1]]

        for piece in pieces:
            yield self._token_to_id[piece]

    def _encode_ordinary(self, text: str) -> Iterator[int]:
        for match in GPT2_PATTERN.finditer(text):
            yield from self._apply_bpe(match.group().encode("utf-8"))

    def _encode_piece(self, text: str) -> Iterator[int]:
        if self._special_pattern is None:
            yield from self._encode_ordinary(text)
            return

        for chunk in self._special_pattern.split(text):
            special_id = self._special_to_id.get(chunk)
            if special_id is not None:
                yield special_id
            elif chunk:
                yield from self._encode_ordinary(chunk)

    def encode(self, text: str) -> list[int]:
        """Encode a complete string into token IDs."""
        return list(self._encode_piece(text))

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Lazily encode successive text chunks without retaining prior chunks."""
        for text in iterable:
            yield from self._encode_piece(text)

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs, replacing byte sequences that are invalid UTF-8."""
        return b"".join(self.vocab[token_id] for token_id in ids).decode("utf-8", errors="replace")
