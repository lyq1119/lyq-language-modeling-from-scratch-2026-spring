"""Minimal WARC/WET record iterator over a gzipped WARC file.

Usage:
  python3 parse_warc.py <file.warc.wet.gz> [max_records] [--json]
Prints each record as: separator line + headers (WARC-Type/URL/date/...) +
first N chars of content.
"""
import gzip
import sys


def split_headers(block: bytes):
    text = block.decode("utf-8", errors="replace")
    lines = text.splitlines()
    headers = {}
    for ln in lines:
        if not ln.strip():
            continue
        if ":" in ln:
            k, v = ln.split(":", 1)
            headers[k.strip().lower()] = v.strip()
        else:
            headers.setdefault("_raw", []).append(ln)
    return headers


def iter_records(fileobj):
    """Yield (headers_dict, content_bytes) for each WARC record."""
    buf = b""
    while True:
        # read until end of header block
        while b"\r\n\r\n" not in buf and b"\n\n" not in buf:
            chunk = fileobj.read(1 << 16)
            if not chunk:
                return
            buf += chunk
        if b"\r\n\r\n" in buf:
            header_part, buf = buf.split(b"\r\n\r\n", 1)
            sep = b"\r\n\r\n"
        else:
            header_part, buf = buf.split(b"\n\n", 1)
            sep = b"\n\n"
        headers = split_headers(header_part)
        try:
            clen = int(headers.get("content-length", "0"))
        except ValueError:
            clen = 0
        # read content of clen bytes
        while len(buf) < clen:
            chunk = fileobj.read(1 << 16)
            if not chunk:
                break
            buf += chunk
        content = buf[:clen]
        buf = buf[clen:]
        # consume record separator after content (\r\n\r\n typically)
        for s in (b"\r\n\r\n", b"\n\n"):
            if buf.startswith(s):
                buf = buf[len(s):]
                break
        yield headers, content


def main():
    path = sys.argv[1]
    max_rec = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    want_json = "--json" in sys.argv
    with gzip.open(path, "rb") as f:
        for i, (headers, content) in enumerate(iter_records(f)):
            if i >= max_rec:
                break
            if want_json:
                import json
                print(json.dumps({
                    "idx": i,
                    "type": headers.get("warc-type"),
                    "url": headers.get("warc-target-uri"),
                    "date": headers.get("warc-date"),
                    "conv_from": headers.get("warc-converted-from-uri"),
                    "content_length": len(content),
                    "content_head": content[:400].decode("utf-8", errors="replace"),
                }, ensure_ascii=False)[:2000])
            else:
                print(f"===== RECORD {i} =====")
                for k, v in headers.items():
                    print(f"{k}: {v}")
                print(f"--- content ({len(content)} bytes) head ---")
                print(content[:600].decode("utf-8", errors="replace"))
                print()


if __name__ == "__main__":
    main()
