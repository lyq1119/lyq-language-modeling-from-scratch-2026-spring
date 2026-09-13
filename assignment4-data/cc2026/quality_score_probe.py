"""探测质量分类器在“已通过语言/gopher/有害内容过滤”的英文 CC 文档上的分数分布。

用法：python -m cc2026.quality_score_probe [--files N]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MIN_TEXT_CHARS = 200


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wet-dir", type=Path, default=ROOT / "local-shared-data/raw-wet")
    parser.add_argument("--files", type=int, default=2)
    parser.add_argument("--lang-min", type=float, default=0.70)
    args = parser.parse_args()

    from fastwarc.warc import ArchiveIterator, WarcRecordType

    from tests.adapters import (
        run_classify_nsfw,
        run_classify_quality,
        run_classify_toxic_speech,
        run_gopher_quality_filter,
        run_identify_language,
    )

    wet_files = sorted(args.wet_dir.glob("*.warc.wet.gz"))[: args.files]
    label_counts = Counter()
    wiki_scores = []
    cc_scores = []
    for path in wet_files:
        with open(path, "rb") as f:
            for rec in ArchiveIterator(f, record_types=WarcRecordType.conversion):
                text = rec.reader.read().decode("utf-8", errors="replace")
                if len(text.strip()) < MIN_TEXT_CHARS:
                    continue
                lang, ls = run_identify_language(text)
                if lang != "en" or ls < args.lang_min:
                    continue
                if not run_gopher_quality_filter(text):
                    continue
                nsfw, ns = run_classify_nsfw(text)
                if nsfw == "nsfw" and ns >= 0.5:
                    continue
                tox, ts = run_classify_toxic_speech(text)
                if tox == "toxic" and ts >= 0.5:
                    continue
                label, score = run_classify_quality(text)
                label_counts[label] += 1
                (wiki_scores if label == "wiki" else cc_scores).append(score)

    total = sum(label_counts.values())
    print(f"survivors before quality filter: {total}")
    print(f"label distribution: {dict(label_counts)}")
    if total:
        print(f"wiki fraction: {100 * label_counts.get('wiki', 0) / total:.1f}%")
    for name, scores in [("wiki", wiki_scores), ("cc", cc_scores)]:
        if not scores:
            continue
        scores.sort()
        n = len(scores)
        qs = [scores[int(p * (n - 1))] for p in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)]
        print(f"{name}: n={n} quantiles(0,10,25,50,75,90,100)={[round(q,3) for q in qs]}")
    for thr in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        kept = sum(1 for s in wiki_scores if s >= thr)
        pct = 100 * kept / total if total else 0
        print(f"threshold wiki>={thr}: kept={kept} ({pct:.2f}% of survivors)")


if __name__ == "__main__":
    main()
