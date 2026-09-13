"""2.5(d) 实验：对 cc2026/example.warc.gz 抽取的文本运行 NSFW / 恶意言论过滤器，
统计数据并抽 20 篇分类器预测为有害的文档供人工核对。

用法：python -m cc2026.harmful_experiment
输出：cc2026/harmful_results.json
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastwarc.warc import ArchiveIterator, WarcRecordType  # noqa: E402
from tests.adapters import (  # noqa: E402
    run_classify_nsfw,
    run_classify_toxic_speech,
    run_extract_text_from_html_bytes,
)

MIN_TEXT_CHARS = 50
HEAD_CHARS = 600


def main() -> None:
    warc_path = ROOT / "cc2026" / "example.warc.gz"
    out_path = ROOT / "cc2026" / "harmful_results.json"

    docs = []
    n_docs = 0
    nsfw_counts = Counter()
    toxic_counts = Counter()
    with open(warc_path, "rb") as f:
        for rec in ArchiveIterator(f, record_types=WarcRecordType.response):
            url = rec.headers.get("WARC-Target-URI", "")
            text = run_extract_text_from_html_bytes(rec.reader.read()) or ""
            if len(text.strip()) < MIN_TEXT_CHARS:
                continue
            n_docs += 1
            nsfw_label, nsfw_score = run_classify_nsfw(text)
            toxic_label, toxic_score = run_classify_toxic_speech(text)
            nsfw_counts[nsfw_label] += 1
            toxic_counts[toxic_label] += 1
            if nsfw_label == "nsfw" or toxic_label == "toxic":
                docs.append(
                    {
                        "url": url,
                        "text_len": len(text),
                        "text_head": text[:HEAD_CHARS],
                        "nsfw": [nsfw_label, round(nsfw_score, 4)],
                        "toxic": [toxic_label, round(toxic_score, 4)],
                    }
                )

    with open(out_path, "w") as f:
        json.dump(docs, f, ensure_ascii=False, indent=1)

    print(f"文档总数: {n_docs}")
    print("NSFW 标签分布:", dict(nsfw_counts))
    print("恶意言论标签分布:", dict(toxic_counts))
    print(f"被判为有害(NSFW 或 toxic)的文档数: {len(docs)}")
    if n_docs:
        print(f"有害文档占比: {100 * len(docs) / n_docs:.2f}%")

    random.seed(2026)
    sample = random.sample(docs, min(20, len(docs)))
    print("\n=== 随机抽 20 篇被判为有害的文档 ===")
    for i, d in enumerate(sample):
        print(f"\n--- [{i}] {d['url']} nsfw={d['nsfw']} toxic={d['toxic']} len={d['text_len']}")
        print("    ", d["text_head"][:300].replace("\n", " "))


if __name__ == "__main__":
    main()
