"""4 filter_data：并行过滤 Common Crawl WET 文件以生成语言建模数据。

每个 WET 文件依次经过以下过滤器（顺序即流水线顺序）：
  1. 长度过滤：文本 < 200 字符直接丢弃
  2. 语言识别：仅保留 fastText lid.176 判为英语且置信度 >= 0.70 的文档
  3. Gopher 质量规则（2.6）
  4. NSFW 过滤（2.5）
  5. 恶意言论过滤（2.5）
  6. 质量分类器（2.7）：仅保留预测为 wiki 且置信度 >= 阈值的文档

用法：
  python -m cc2026.filter_wet --wet-dir DIR --out-dir DIR --workers 32
输出：
  out-dir/*.jsonl.gz       每个输入 WET 文件对应一个，内含保留下来的文档
  out-dir/filter_stats.json 各阶段保留计数
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MIN_TEXT_CHARS = 200
HEAD_CHARS = 0  # 完整保存文本


def filter_one_file(args_tuple):
    wet_path_str, out_dir_str, params = args_tuple
    wet_path = Path(wet_path_str)
    out_dir = Path(out_dir_str)

    from fastwarc.warc import ArchiveIterator, WarcRecordType

    from tests.adapters import (
        run_classify_nsfw,
        run_classify_toxic_speech,
        run_gopher_quality_filter,
        run_identify_language,
    )

    skip_quality = params.get("skip_quality", False)
    if not skip_quality:
        from tests.adapters import run_classify_quality

    lang_min = params["lang_min"]
    nsfw_min = params["nsfw_min"]
    toxic_min = params["toxic_min"]
    quality_min = params["quality_min"]

    counts = {"total": 0, "length": 0, "language": 0, "gopher": 0, "nsfw": 0, "toxic": 0, "quality": 0}
    out_path = out_dir / (wet_path.name.replace(".warc.wet.gz", "") + ".jsonl.gz")
    n_written = 0
    with open(wet_path, "rb") as f, gzip.open(out_path, "wt") as out:
        for rec in ArchiveIterator(f, record_types=WarcRecordType.conversion):
            counts["total"] += 1
            url = rec.headers.get("WARC-Target-URI", "")
            payload = rec.reader.read()
            text = payload.decode("utf-8", errors="replace")
            if len(text.strip()) < MIN_TEXT_CHARS:
                continue
            counts["length"] += 1

            lang, lang_score = run_identify_language(text)
            if lang != "en" or lang_score < lang_min:
                continue
            counts["language"] += 1

            if not run_gopher_quality_filter(text):
                continue
            counts["gopher"] += 1

            nsfw_label, nsfw_score = run_classify_nsfw(text)
            if nsfw_label == "nsfw" and nsfw_score >= nsfw_min:
                continue
            counts["nsfw"] += 1

            toxic_label, toxic_score = run_classify_toxic_speech(text)
            if toxic_label == "toxic" and toxic_score >= toxic_min:
                continue
            counts["toxic"] += 1

            if not skip_quality:
                q_label, q_score = run_classify_quality(text)
                if q_label != "wiki" or q_score < quality_min:
                    continue
            counts["quality"] += 1

            out.write(json.dumps({"url": url, "text": text}, ensure_ascii=False) + "\n")
            n_written += 1
    return {"file": wet_path.name, "written": n_written, "counts": counts}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wet-dir", type=Path, default=ROOT / "local-shared-data/raw-wet")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "local-shared-data/filtered_jsonl")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 个文件（0=全部）")
    parser.add_argument("--shard-index", type=int, default=0, help="分片索引（配合 --num-shards 把文件拆分到多个作业）")
    parser.add_argument("--num-shards", type=int, default=1, help="分片总数")
    parser.add_argument("--stats-out", type=Path, default=None, help="统计 JSON 输出路径（默认 out-dir/filter_stats.json）")
    parser.add_argument("--lang-min", type=float, default=0.70)
    parser.add_argument("--nsfw-min", type=float, default=0.50)
    parser.add_argument("--toxic-min", type=float, default=0.50)
    parser.add_argument("--quality-min", type=float, default=0.50)
    parser.add_argument(
        "--skip-quality",
        action="store_true",
        help="不把质量分类器当作硬过滤（该分类器对随机 CC 文档高度饱和，保留率仅约 2%%）",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    wet_files = sorted(args.wet_dir.glob("*.warc.wet.gz"))
    if args.limit:
        wet_files = wet_files[: args.limit]
    if args.num_shards > 1:
        wet_files = wet_files[args.shard_index :: args.num_shards]
    print(f"processing {len(wet_files)} WET files with {args.workers} workers", flush=True)

    params = {
        "lang_min": args.lang_min,
        "nsfw_min": args.nsfw_min,
        "toxic_min": args.toxic_min,
        "quality_min": args.quality_min,
        "skip_quality": args.skip_quality,
    }
    tasks = [(str(p), str(args.out_dir), params) for p in wet_files]

    import concurrent.futures
    import multiprocessing

    # 在 fork 之前把模型加载进父进程，让子进程通过 copy-on-write 共享，
    # 避免每个 worker 各加载 ~1.7GB 的 fastText 模型（分区每节点内存上限 64GB）。
    from tests.adapters import (
        run_classify_nsfw,
        run_classify_toxic_speech,
        run_identify_language,
    )

    print("preloading models ...", flush=True)
    run_identify_language("hello world")
    run_classify_nsfw("hello world")
    run_classify_toxic_speech("hello world")
    if not args.skip_quality:
        from tests.adapters import run_classify_quality

        run_classify_quality("hello world")
    print("models loaded", flush=True)

    t0 = time.time()
    stage_order = ["total", "length", "language", "gopher", "nsfw", "toxic", "quality"]
    totals = {s: 0 for s in stage_order}
    results = []
    ctx = multiprocessing.get_context("fork")
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        for i, res in enumerate(pool.map(filter_one_file, tasks), 1):
            for s in stage_order:
                totals[s] += res["counts"][s]
            results.append(res)
            if i % 5 == 0 or i == len(tasks):
                print(f"[{i}/{len(tasks)}] {res['file']} written={res['written']} elapsed={time.time() - t0:.0f}s", flush=True)

    stats = {"stage_totals": totals, "per_file": results, "elapsed_seconds": time.time() - t0}
    stats_path = args.stats_out or (args.out_dir / "filter_stats.json")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=1))
    print("\n=== per-stage retention ===")
    for i, s in enumerate(stage_order):
        denom = totals["total"]
        print(f"{s:9s} {totals[s]:9d}  ({100 * totals[s] / denom:.2f}% of total)" if denom else s)
    print(f"elapsed {time.time() - t0:.0f}s")
    print(f"stats saved to {stats_path}")


if __name__ == "__main__":
    main()
