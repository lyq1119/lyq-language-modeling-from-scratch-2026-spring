"""2.4(e) 实验：对 cc2026/example.warc.gz 抽取的文本运行三个 PII 掩蔽器，
收集发生了替换的文档与匹配片段（含上下文），供人工检查假阳性/假阴性。

用法：python -m cc2026.pii_experiment
输出：cc2026/pii_results.json
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastwarc.warc import ArchiveIterator, WarcRecordType  # noqa: E402
from tests.adapters import (  # noqa: E402
    _EMAIL_RE,
    _IPV4_RE,
    _PHONE_RE,
    run_extract_text_from_html_bytes,
)

MIN_TEXT_CHARS = 50
CTX = 50  # 每个匹配前后保留的上下文长度


def main() -> None:
    warc_path = ROOT / "cc2026" / "example.warc.gz"
    out_path = ROOT / "cc2026" / "pii_results.json"

    maskers = {
        "email": (_EMAIL_RE, "|||EMAIL_ADDRESS|||"),
        "phone": (_PHONE_RE, "|||PHONE_NUMBER|||"),
        "ip": (_IPV4_RE, "|||IP_ADDRESS|||"),
    }

    hits = []  # 至少发生一次替换的文档
    totals = {"email": 0, "phone": 0, "ip": 0}
    n_docs = 0

    with open(warc_path, "rb") as f:
        for rec in ArchiveIterator(f, record_types=WarcRecordType.response):
            url = rec.headers.get("WARC-Target-URI", "")
            text = run_extract_text_from_html_bytes(rec.reader.read()) or ""
            if len(text.strip()) < MIN_TEXT_CHARS:
                continue
            n_docs += 1
            doc = {"url": url, "text_len": len(text), "matches": [], "counts": {"email": 0, "phone": 0, "ip": 0}}
            for kind, (pattern, _mask) in maskers.items():
                for m in pattern.finditer(text):
                    s, e = m.span()
                    doc["matches"].append(
                        {
                            "kind": kind,
                            "match": m.group(0),
                            "before": text[max(0, s - CTX): s].replace("\n", " "),
                            "after": text[e: e + CTX].replace("\n", " "),
                        }
                    )
                    doc["counts"][kind] += 1
                    totals[kind] += 1
            if any(doc["counts"].values()):
                hits.append(doc)

    with open(out_path, "w") as f:
        json.dump(hits, f, ensure_ascii=False, indent=1)

    print(f"文档总数: {n_docs}")
    print(f"发生替换的文档数: {len(hits)}")
    print("各类型替换总数:", totals)

    random.seed(2026)
    sample = random.sample(hits, min(20, len(hits)))
    print("\n=== 随机抽 20 篇含替换文档，打印前几条匹配与上下文 ===")
    for i, d in enumerate(sample):
        print(f"\n--- [{i}] {d['url']} counts={d['counts']} len={d['text_len']}")
        for m in d["matches"][:6]:
            print(f"  <{m['kind']}> {m['match']!r} || ...{m['before'][-25:]}«{m['match']}»{m['after'][:25]}...")


if __name__ == "__main__":
    main()
