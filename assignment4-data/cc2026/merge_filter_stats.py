"""合并 filter_wet.py 各分片的 filter_stats_shard*.json 为一个 filter_stats.json。

用法：python -m cc2026.merge_filter_stats <stats_dir> <out_json>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

STAGES = ["total", "length", "language", "gopher", "nsfw", "toxic", "quality"]


def main() -> None:
    stats_dir = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    shard_files = sorted(stats_dir.glob("filter_stats_shard*.json"))
    if not shard_files:
        shard_files = [stats_dir / "filter_stats.json"]

    totals = {s: 0 for s in STAGES}
    per_file: list[dict] = []
    elapsed = 0.0
    for path in shard_files:
        data = json.loads(path.read_text())
        for s in STAGES:
            totals[s] += data["stage_totals"].get(s, 0)
        per_file.extend(data.get("per_file", []))
        elapsed = max(elapsed, data.get("elapsed_seconds", 0.0))

    merged = {"stage_totals": totals, "per_file": per_file, "elapsed_seconds": elapsed, "shards": [p.name for p in shard_files]}
    out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=1))
    print(f"merged {len(shard_files)} shards -> {out_path}")
    for i, s in enumerate(STAGES):
        denom = totals["total"]
        pct = 100 * totals[s] / denom if denom else 0
        print(f"{s:9s} {totals[s]:10d}  ({pct:.2f}% of total)")


if __name__ == "__main__":
    main()
