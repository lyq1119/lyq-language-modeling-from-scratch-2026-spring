"""2.6(b) 实验：对 cc2026/example.warc.gz 抽取的文本运行 Gopher 质量规则过滤器，
统计各规则命中情况，并抽 20 篇被移除 / 保留的文档供人工核对。

用法：python -m cc2026.gopher_experiment
输出：cc2026/gopher_results.json
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastwarc.warc import ArchiveIterator, WarcRecordType  # noqa: E402
from tests.adapters import run_extract_text_from_html_bytes, run_gopher_quality_filter  # noqa: E402

MIN_TEXT_CHARS = 50
HEAD_CHARS = 600
_WORD_RE = re.compile(r"\S+")


def fail_reasons(text: str) -> list[str]:
    """复刻 run_gopher_quality_filter 的判定，返回未通过的规则名。"""
    reasons = []
    words = _WORD_RE.findall(text)
    n = len(words)
    if n < 50 or n > 100_000:
        reasons.append("word_count")
    mean_wl = sum(len(w) for w in words) / n if n else 0
    if n and (mean_wl < 3 or mean_wl > 10):
        reasons.append("mean_word_length")
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if lines:
        frac = sum(1 for ln in lines if ln.rstrip().endswith("...")) / len(lines)
        if frac > 0.30:
            reasons.append("ellipsis")
    if n and sum(1 for w in words if any(c.isalpha() for c in w)) / n < 0.80:
        reasons.append("alpha_fraction")
    return reasons


def main() -> None:
    warc_path = ROOT / "cc2026" / "example.warc.gz"
    out_path = ROOT / "cc2026" / "gopher_results.json"

    kept, removed = [], []
    reason_counter = Counter()
    with open(warc_path, "rb") as f:
        for rec in ArchiveIterator(f, record_types=WarcRecordType.response):
            url = rec.headers.get("WARC-Target-URI", "")
            text = run_extract_text_from_html_bytes(rec.reader.read()) or ""
            if len(text.strip()) < MIN_TEXT_CHARS:
                continue
            passed = run_gopher_quality_filter(text)
            reasons = [] if passed else fail_reasons(text)
            for r in reasons:
                reason_counter[r] += 1
            entry = {"url": url, "text_len": len(text), "text_head": text[:HEAD_CHARS], "reasons": reasons}
            (kept if passed else removed).append(entry)

    with open(out_path, "w") as f:
        json.dump({"kept": kept, "removed": removed}, f, ensure_ascii=False, indent=1)

    total = len(kept) + len(removed)
    print(f"文档总数: {total}")
    print(f"通过 Gopher 过滤器: {len(kept)} ({100 * len(kept) / total:.1f}%)")
    print(f"被移除: {len(removed)} ({100 * len(removed) / total:.1f}%)")
    print("被移除文档触发的规则计数:", dict(reason_counter))

    print("\n=== 随机抽 20 篇被移除的文档 ===")
    import random

    random.seed(2026)
    for i, d in enumerate(random.sample(removed, min(20, len(removed)))):
        print(f"[{i}] {d['url']} reasons={d['reasons']} len={d['text_len']}")
        print("    ", d["text_head"][:250].replace("\n", " "))


if __name__ == "__main__":
    main()
