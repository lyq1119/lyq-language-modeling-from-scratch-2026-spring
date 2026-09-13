"""2.3(c) 实验：对 cc2026/example.warc.gz 的每个 response 文档
1) 用 run_extract_text_from_html_bytes 抽取纯文本
2) 用 run_identify_language 做语言识别
3) 保存 (url, 文本长度, 文本头部, 预测语言, 置信度) 供人工抽检与统计

用法：python -m cc2026.langid_experiment
输出：cc2026/langid_results.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastwarc.warc import ArchiveIterator, WarcRecordType  # noqa: E402
from tests.adapters import run_extract_text_from_html_bytes, run_identify_language  # noqa: E402

MIN_TEXT_CHARS = 50  # 过滤空页面 / 几乎无文本的页面
HEAD_CHARS = 500  # 存多少文本用于人工抽检


def main() -> None:
    warc_path = ROOT / "cc2026" / "example.warc.gz"
    out_path = ROOT / "cc2026" / "langid_results.json"

    t0 = time.time()
    docs = []
    n_response = 0
    n_skipped = 0
    with open(warc_path, "rb") as f:
        for rec in ArchiveIterator(f, record_types=WarcRecordType.response):
            n_response += 1
            url = rec.headers.get("WARC-Target-URI", "")
            html = rec.reader.read()
            text = run_extract_text_from_html_bytes(html) or ""
            if len(text.strip()) < MIN_TEXT_CHARS:
                n_skipped += 1
                continue
            lang, score = run_identify_language(text)
            docs.append(
                {
                    "url": url,
                    "text_len": len(text),
                    "text_head": text[:HEAD_CHARS],
                    "lang": lang,
                    "score": round(score, 4),
                }
            )

    with open(out_path, "w") as f:
        json.dump(docs, f, ensure_ascii=False, indent=1)

    n = len(docs)
    n_en = sum(1 for d in docs if d["lang"] == "en")
    print(f"response 记录数: {n_response}")
    print(f"文本 >= {MIN_TEXT_CHARS} 字符的文档数: {n} (跳过 {n_skipped})")
    print(f"fastText 预测为英语: {n_en} ({100 * n_en / n:.1f}%)")
    print(f"用时 {time.time() - t0:.1f}s, 结果已保存到 {out_path}")


if __name__ == "__main__":
    main()
