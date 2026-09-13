"""4 tokenize_data：把过滤后的数据用 GPT-2 tokenizer 编码成 uint16 序列。

用法：
  python -m cc2026.tokenize_data \
      --jsonl-dir local-shared-data/filtered_jsonl \
      --out local-shared-data/tokenized/train.bin

可选 --minhash 对文档做模糊去重（会先写到临时文件再调用 run_minhash_deduplication）。
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_docs(jsonl_dir: Path, limit: int = 0) -> list[dict]:
    docs = []
    for path in sorted(jsonl_dir.glob("*.jsonl.gz")):
        with gzip.open(path, "rt") as f:
            for line in f:
                docs.append(json.loads(line))
                if limit and len(docs) >= limit:
                    return docs
    return docs


def exact_doc_dedup(docs: list[dict], min_chars: int = 200) -> list[dict]:
    """按“空白归一化后的整篇文本”做精确文档级去重（比逐行去重快得多、内存小）。"""
    seen: set[bytes] = set()
    out = []
    for doc in docs:
        text = doc["text"]
        if len(text.strip()) < min_chars:
            continue
        key = hashlib.blake2b(" ".join(text.split()).encode("utf-8"), digest_size=16).digest()
        if key in seen:
            continue
        seen.add(key)
        out.append(doc)
    return out


def exact_line_dedup(docs: list[dict]) -> list[dict]:
    """移除在整个语料库中重复出现的行，返回去重后的文档。"""
    counts: Counter[bytes] = Counter()
    for doc in docs:
        for line in doc["text"].splitlines(keepends=True):
            counts[hashlib.blake2b(line.encode("utf-8"), digest_size=16).digest()] += 1
    out = []
    for doc in docs:
        kept = [
            line
            for line in doc["text"].splitlines(keepends=True)
            if counts[hashlib.blake2b(line.encode("utf-8"), digest_size=16).digest()] == 1
        ]
        text = "".join(kept)
        if text.strip():
            out.append({"url": doc["url"], "text": text})
    return out


def minhash_dedup(docs: list[dict], tmp_dir: Path, num_hashes: int = 128, num_bands: int = 16, ngrams: int = 5, threshold: float = 0.8) -> list[dict]:
    from tests.adapters import run_minhash_deduplication

    tmp_in = tmp_dir / "minhash_in"
    tmp_out = tmp_dir / "minhash_out"
    tmp_in.mkdir(parents=True, exist_ok=True)
    tmp_out.mkdir(parents=True, exist_ok=True)
    mapping = {}
    for i, doc in enumerate(docs):
        name = f"{i:08d}.txt"
        (tmp_in / name).write_text(doc["text"])
        mapping[name] = doc
    run_minhash_deduplication(
        input_files=[tmp_in / n for n in mapping],
        output_directory=tmp_out,
        num_hashes=num_hashes,
        num_bands=num_bands,
        ngrams=ngrams,
        jaccard_threshold=threshold,
    )
    kept = []
    for path in sorted(tmp_out.glob("*.txt")):
        doc = mapping[path.name]
        if path.read_text().strip():
            kept.append(doc)
    return kept


_TOKENIZER = None


def _init_worker():
    global _TOKENIZER
    from transformers import AutoTokenizer

    import os

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    _TOKENIZER = AutoTokenizer.from_pretrained("gpt2")


def _tokenize(text: str) -> list[int]:
    ids = _TOKENIZER.encode(text)
    return ids + [_TOKENIZER.eos_token_id]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl-dir", type=Path, default=ROOT / "local-shared-data/filtered_jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "local-shared-data/tokenized/train.bin")
    parser.add_argument("--dedup", action="store_true", help="exact line dedup (慢，内存高)")
    parser.add_argument("--doc-dedup", action="store_true", help="exact document dedup (快)")
    parser.add_argument("--minhash", action="store_true", help="minhash doc dedup")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--limit-docs", type=int, default=0, help="只处理前 N 篇（0=全部），上限用于控制 token 总量")
    parser.add_argument("--max-tokens", type=int, default=0, help="写满这么多 token 就停止（0=不限）")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    docs = load_docs(args.jsonl_dir, args.limit_docs)
    n_raw = len(docs)
    print(f"loaded {n_raw} docs in {time.time() - t0:.0f}s", flush=True)
    if args.limit_docs and n_raw >= args.limit_docs:
        print(f"capped to {len(docs)} docs", flush=True)

    if args.dedup:
        docs = exact_line_dedup(docs)
        print(f"after exact line dedup: {len(docs)} docs", flush=True)
    if args.doc_dedup:
        docs = exact_doc_dedup(docs)
        print(f"after exact doc dedup: {len(docs)} docs", flush=True)
    if args.minhash:
        tmp_dir = ROOT / "local-shared-data/tmp_dedup"
        docs = minhash_dedup(docs, tmp_dir)
        print(f"after minhash dedup: {len(docs)} docs", flush=True)

    import numpy as np
    from transformers import AutoTokenizer

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.model_max_length = int(1e12)  # 关闭 >1024 token 的逐条告警（不截断）
    eos = tokenizer.eos_token_id
    eos_arr = np.asarray([eos], dtype=np.uint16)
    eos_bytes = eos_arr.tobytes()

    texts = [d["text"] for d in docs]
    n_tokens = 0
    batch_size = 512
    # 直接用 HF 的多线程 encode_batch 在父进程内完成，避免每个文档跨进程 pickle。
    t1 = time.time()
    stop = False
    with open(args.out, "wb") as fout:
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            # 用 fast tokenizer 的批量接口（底层 Rust 多线程），不逐篇 pickle。
            for ids in tokenizer(chunk, add_special_tokens=False)["input_ids"]:
                arr = np.asarray(ids, dtype=np.uint16)
                fout.write(arr.tobytes())
                fout.write(eos_bytes)
                n_tokens += len(arr) + 1
                if args.max_tokens and n_tokens >= args.max_tokens:
                    stop = True
                    break
            done = start + len(chunk)
            if done % 50000 == 0 or done >= len(texts) or stop:
                rate = done / max(1e-9, time.time() - t1)
                print(f"  tokenized {done}/{len(texts)} docs, {n_tokens} tokens, {rate:.0f} docs/s", flush=True)
            if stop:
                break

    print(f"docs: {len(docs)}  tokens: {n_tokens}  -> {args.out}", flush=True)
    summary = {"raw_docs": n_raw, "final_docs": len(docs), "tokens": int(n_tokens), "bin": str(args.out)}
    (args.out.parent / "tokenize_summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
