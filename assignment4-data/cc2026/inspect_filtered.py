"""inspect_filtered_data：抽取 5 条保留样本和 5 条被移除样本（含移除阶段）。

保留样本：从 out-dir/*.jsonl.gz（filter_wet 的输出）中做蓄水池抽样。
移除样本：对 raw-wet 中若干 WET 文件重放同一套过滤流水线，记录每条文档
          首次被哪个阶段移除，再做蓄水池抽样。

用法：
  python -m cc2026.inspect_filtered --jsonl-dir DIR --wet-dir DIR [--files 8] [--out JSON]
输出：cc2026/inspect_results.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MIN_TEXT_CHARS = 200
HEAD_CHARS = 400


def sample_kept(jsonl_dir: Path, k: int, max_files: int) -> list[dict]:
    files = sorted(jsonl_dir.glob("*.jsonl.gz"))
    rng = random.Random(336)
    rng.shuffle(files)
    reservoir: list[dict] = []
    seen = 0
    for path in files[:max_files]:
        with gzip.open(path, "rt") as f:
            for line in f:
                doc = json.loads(line)
                seen += 1
                text = doc["text"]
                entry = {
                    "url": doc["url"],
                    "text_len": len(text),
                    "text_head": text[:HEAD_CHARS],
                    "source_file": path.name,
                }
                if len(reservoir) < k:
                    reservoir.append(entry)
                else:
                    j = rng.randrange(seen)
                    if j < k:
                        reservoir[j] = entry
    return reservoir


def classify_stage(text: str, params: dict) -> str | None:
    """返回文档被移除的阶段名；通过全部过滤器则返回 None。"""
    from tests.adapters import (
        run_classify_nsfw,
        run_classify_toxic_speech,
        run_gopher_quality_filter,
        run_identify_language,
    )

    if len(text.strip()) < MIN_TEXT_CHARS:
        return "length"
    lang, lang_score = run_identify_language(text)
    if lang != "en" or lang_score < params["lang_min"]:
        return "language"
    if not run_gopher_quality_filter(text):
        return "gopher"
    nsfw_label, nsfw_score = run_classify_nsfw(text)
    if nsfw_label == "nsfw" and nsfw_score >= params["nsfw_min"]:
        return "nsfw"
    toxic_label, toxic_score = run_classify_toxic_speech(text)
    if toxic_label == "toxic" and toxic_score >= params["toxic_min"]:
        return "toxic"
    if params.get("skip_quality"):
        return None
    from tests.adapters import run_classify_quality

    q_label, q_score = run_classify_quality(text)
    if q_label != "wiki" or q_score < params["quality_min"]:
        return "quality"
    return None


def sample_removed(wet_dir: Path, k: int, max_files: int, params: dict, seed: int = 336) -> list[dict]:
    from fastwarc.warc import ArchiveIterator, WarcRecordType

    files = sorted(wet_dir.glob("*.warc.wet.gz"))
    rng = random.Random(seed)
    rng.shuffle(files)
    per_stage: dict[str, list[dict]] = {}
    stage_counts = Counter()
    for path in files[:max_files]:
        with open(path, "rb") as f:
            for rec in ArchiveIterator(f, record_types=WarcRecordType.conversion):
                url = rec.headers.get("WARC-Target-URI", "")
                text = rec.reader.read().decode("utf-8", errors="replace")
                stage = classify_stage(text, params)
                if stage is None:
                    continue
                stage_counts[stage] += 1
                bucket = per_stage.setdefault(stage, [])
                if len(bucket) < 40:  # 每个阶段最多存 40 个候选，最后再分层抽样
                    bucket.append(
                        {
                            "url": url,
                            "stage": stage,
                            "text_len": len(text),
                            "text_head": text[:HEAD_CHARS],
                            "source_file": path.name,
                        }
                    )

    # 分层轮询：每个阶段轮流取一个，保证 5 个样本覆盖不同过滤器
    order = ["length", "language", "gopher", "nsfw", "toxic", "quality"]
    for s in per_stage:
        if s not in order:
            order.append(s)
    rng.shuffle(per_stage.get("language", []))  # language 样本最多，打乱避免同一来源聚集
    picked: list[dict] = []
    round_idx = 0
    while len(picked) < k:
        added = False
        for s in order:
            bucket = per_stage.get(s, [])
            if round_idx < len(bucket):
                picked.append(bucket[round_idx])
                added = True
                if len(picked) >= k:
                    break
        if not added:
            break
        round_idx += 1
    return picked, dict(stage_counts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl-dir", type=Path, default=ROOT / "local-shared-data/filtered_jsonl")
    parser.add_argument("--wet-dir", type=Path, default=ROOT / "local-shared-data/raw-wet")
    parser.add_argument("--out", type=Path, default=ROOT / "cc2026/inspect_results.json")
    parser.add_argument("--kept-max-files", type=int, default=20)
    parser.add_argument("--files", type=int, default=6, help="重放多少个 WET 文件来收集被移除样本")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--lang-min", type=float, default=0.70)
    parser.add_argument("--nsfw-min", type=float, default=0.50)
    parser.add_argument("--toxic-min", type=float, default=0.50)
    parser.add_argument("--quality-min", type=float, default=0.50)
    parser.add_argument("--skip-quality", action="store_true", help="与 filter_wet 保持一致：不应用质量分类器")
    args = parser.parse_args()

    params = {
        "lang_min": args.lang_min,
        "nsfw_min": args.nsfw_min,
        "toxic_min": args.toxic_min,
        "quality_min": args.quality_min,
        "skip_quality": args.skip_quality,
    }

    print("sampling kept docs ...", flush=True)
    kept = sample_kept(args.jsonl_dir, args.k, args.kept_max_files)
    print(f"kept samples: {len(kept)}", flush=True)

    print("replaying pipeline to collect removed docs ...", flush=True)
    removed, stage_counts = sample_removed(args.wet_dir, args.k, args.files, params)
    print(f"removed samples: {len(removed)}; stage counts: {stage_counts}", flush=True)

    result = {"kept": kept, "removed": removed, "removed_stage_counts": stage_counts}
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1))

    for kind in ("kept", "removed"):
        print(f"\n=== {kind} ===")
        for i, d in enumerate(result[kind]):
            stage = d.get("stage", "-")
            print(f"[{i}] stage={stage} len={d['text_len']} {d['url']}")
            print("    ", d["text_head"][:250].replace("\n", " "))


if __name__ == "__main__":
    main()
