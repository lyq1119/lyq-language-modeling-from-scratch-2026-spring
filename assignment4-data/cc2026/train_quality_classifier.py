"""2.7(a) 质量分类器训练：用高质量参考文本作正样本、Common Crawl WET 页面作负样本，
训练一个 fastText 二分类器（__label__wiki=高质量 / __label__cc=低质量）。

正样本使用两类：
  1. Paloma C4 100 域名验证集里的文本（题目明确允许用 Paloma 验证数据构造过滤器/
     分类器，只是不允许把验证数据逐字复制进训练集）——这一路让分类器学到“接近
     目标评测分布的高质量网页文本”；
  2. Wikipedia 正文（enwiki parquet）——这一路提供通用的百科式高质量文本。
只把 Wikipedia 正文当作正样本会把分类器训成“是不是维基百科”的探测器（实测对
CC 文本几乎全部判为 __label__cc、置信度≈1.0，保留率仅 0.7%），因此这里引入
Paloma 正样本来拓宽“高质量”的定义。

用法：
  python -m cc2026.train_quality_classifier \
      --wiki-parquet local-shared-data/wiki/enwiki_train-00000.parquet \
      --paloma-bin local-shared-data/tokenized_paloma_c4_100_domains_validation.bin \
      --wet-dir local-shared-data/raw-wet \
      --out local-shared-data/classifiers/quality_classifier.bin
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import fasttext  # noqa: E402
from fastwarc.warc import ArchiveIterator, WarcRecordType  # noqa: E402

from tests.adapters import run_identify_language  # noqa: E402

MAX_CHARS = 2000
MIN_NEG_CHARS = 200


def load_paloma_positives(bin_path: Path, limit: int) -> list[str]:
    """从 Paloma C4-100 验证集的 uint16 token 流中解码出文档文本作正样本。"""
    import numpy as np
    from transformers import AutoTokenizer

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    data = np.fromfile(bin_path, dtype=np.uint16)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    eos = tokenizer.eos_token_id
    end_positions = np.flatnonzero(data == eos)

    texts: list[str] = []
    start = 0
    for end in end_positions:
        text = tokenizer.decode(data[start:end]).replace("\n", " ").strip()
        start = int(end) + 1
        if len(text) >= MIN_NEG_CHARS:
            texts.append(text[:MAX_CHARS])
        if len(texts) >= limit:
            break
    return texts


def load_wiki_positives(parquet_path: Path, limit: int) -> list[str]:
    import polars as pl

    df = pl.read_parquet(parquet_path, columns=["text"])
    texts = []
    for text in df["text"].to_list():
        if text and len(text) >= MIN_NEG_CHARS:
            texts.append(text[:MAX_CHARS].replace("\n", " "))
        if len(texts) >= limit:
            break
    return texts


def load_cc_negatives(wet_dir: Path, limit: int, min_lang_score: float = 0.7) -> list[str]:
    """从 WET 文件中抽取英文页面作为负样本。"""
    texts: list[str] = []
    rng = random.Random(336)
    files = sorted(wet_dir.glob("*.warc.wet.gz"))
    rng.shuffle(files)
    for path in files:
        with open(path, "rb") as f:
            for rec in ArchiveIterator(f, record_types=WarcRecordType.conversion):
                payload = rec.reader.read()
                text = payload.decode("utf-8", errors="replace")
                if len(text.strip()) < MIN_NEG_CHARS:
                    continue
                lang, score = run_identify_language(text)
                if lang != "en" or score < min_lang_score:
                    continue
                texts.append(text[:MAX_CHARS].replace("\n", " "))
                if len(texts) >= limit:
                    return texts
        print(f"  negatives so far: {len(texts)} (after {path.name})", flush=True)
    return texts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wiki-parquet", type=Path, default=ROOT / "local-shared-data/wiki/enwiki_train-00000.parquet")
    parser.add_argument("--paloma-bin", type=Path, default=ROOT / "local-shared-data/tokenized_paloma_c4_100_domains_validation.bin")
    parser.add_argument("--paloma-limit", type=int, default=14000)
    parser.add_argument("--wet-dir", type=Path, default=ROOT / "local-shared-data/raw-wet")
    parser.add_argument("--out", type=Path, default=ROOT / "local-shared-data/classifiers/quality_classifier.bin")
    parser.add_argument("--per-class", type=int, default=60000)
    parser.add_argument("--epoch", type=int, default=8)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    data_dir = ROOT / "local-shared-data/quality_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    positives: list[str] = []
    if args.paloma_bin.exists():
        print("Loading Paloma C4-100 positives ...", flush=True)
        paloma = load_paloma_positives(args.paloma_bin, args.paloma_limit)
        print(f"paloma positives: {len(paloma)}", flush=True)
        positives.extend(paloma)
    n_wiki = max(0, args.per_class - len(positives))
    print(f"Loading Wikipedia positives (up to {n_wiki}) ...", flush=True)
    positives.extend(load_wiki_positives(args.wiki_parquet, n_wiki))
    print(f"positives: {len(positives)}", flush=True)

    print("Loading CC negatives (English WET pages) ...", flush=True)
    negatives = load_cc_negatives(args.wet_dir, len(positives))
    print(f"negatives: {len(negatives)}", flush=True)

    rng = random.Random(336)
    rng.shuffle(positives)
    rng.shuffle(negatives)
    n_valid = min(2000, len(positives) // 10, len(negatives) // 10)
    valid_lines = [f"__label__wiki {t}" for t in positives[:n_valid]] + [
        f"__label__cc {t}" for t in negatives[:n_valid]
    ]
    train_lines = [f"__label__wiki {t}" for t in positives[n_valid:]] + [
        f"__label__cc {t}" for t in negatives[n_valid:]
    ]

    train_path = data_dir / "train.txt"
    valid_path = data_dir / "valid.txt"
    train_path.write_text("\n".join(train_lines), encoding="utf-8")
    valid_path.write_text("\n".join(valid_lines), encoding="utf-8")
    print(f"train: {len(train_lines)}  valid: {len(valid_lines)}", flush=True)

    model = fasttext.train_supervised(
        input=str(train_path),
        epoch=args.epoch,
        lr=0.5,
        wordNgrams=2,
        dim=100,
        minCount=2,
        loss="softmax",
        thread=32,
    )
    model.save_model(str(args.out))
    print(f"saved model to {args.out}", flush=True)

    # 用题目提供的两个 fixture 做健全性检查
    fixtures = ROOT / "tests" / "fixtures"
    for name, label in [("low_quality_cc.txt", "cc"), ("high_quality_wiki_reference.txt", "wiki")]:
        text = (fixtures / name).read_text()
        labels, scores = model.predict(text.replace("\n", " "), k=1)
        print(f"fixture {name}: pred={labels[0]} (want {label}) score={scores[0]:.3f}", flush=True)


if __name__ == "__main__":
    main()
