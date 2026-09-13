"""Download a subset of Common Crawl WET files in parallel.

Usage: python download_wet.py <paths_file> <out_dir> [workers]

The paths file contains one crawl-relative path per line, e.g.
    crawl-data/CC-MAIN-2026-17/segments/.../wet/...warc.wet.gz
"""

from __future__ import annotations

import concurrent.futures
import sys
import urllib.request
from pathlib import Path

BASE_URL = "https://data.commoncrawl.org/"


def download_one(rel_path: str, out_dir: Path) -> tuple[str, int]:
    out = out_dir / Path(rel_path).name
    if out.exists() and out.stat().st_size > 0:
        return out.name, out.stat().st_size
    tmp = out.with_suffix(out.suffix + ".part")
    urllib.request.urlretrieve(BASE_URL + rel_path, tmp)
    tmp.rename(out)
    return out.name, out.stat().st_size


def main() -> None:
    paths_file = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    out_dir.mkdir(parents=True, exist_ok=True)
    rel_paths = [line.strip() for line in paths_file.read_text().splitlines() if line.strip()]

    done = 0
    total_bytes = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_one, rp, out_dir): rp for rp in rel_paths}
        for future in concurrent.futures.as_completed(futures):
            name, size = future.result()
            done += 1
            total_bytes += size
            if done % 10 == 0 or done == len(rel_paths):
                print(f"[{done}/{len(rel_paths)}] {name} ({size/1e6:.1f} MB, total {total_bytes/1e9:.2f} GB)", flush=True)
    print("WET_DOWNLOAD_DONE", flush=True)


if __name__ == "__main__":
    main()
