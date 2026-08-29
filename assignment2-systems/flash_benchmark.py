"""Benchmark eager attention against the 4.2.2 FlashAttention implementation.

Run on a GPU, e.g.:
  srun --gres=gpu:1 uv run python flash_benchmark.py --max-seq 8192
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch
import triton.testing
from tests.adapters import get_flashattention_autograd_function_triton


def eager_attention(q, k, v, causal=True):
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    if causal:
        n = q.shape[-2]
        mask = torch.arange(n, device=q.device)[:, None] >= torch.arange(n, device=q.device)[None, :]
        scores = scores.masked_fill(~mask, -1e6)
    return torch.matmul(torch.softmax(scores, dim=-1), v)


def measure(fn):
    torch.cuda.synchronize()
    return float(triton.testing.do_bench(fn, warmup=25, rep=100))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-seq', type=int, default=65536)
    parser.add_argument('--output', type=Path, default=Path('profiles/flash_benchmark.csv'))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    flash = get_flashattention_autograd_function_triton().apply
    rows = []
    for dtype in (torch.bfloat16, torch.float32):
        for d in (16, 32, 64, 128):
            for seq in (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536):
                if seq > args.max_seq:
                    continue
                for name, impl in [('eager', eager_attention), ('flash', flash)]:
                    row = dict(implementation=name, dtype=str(dtype).removeprefix('torch.'), d_model=d, seq_len=seq)
                    try:
                        q = torch.randn(1, seq, d, device='cuda', dtype=dtype, requires_grad=True)
                        k = torch.randn_like(q, requires_grad=True)
                        v = torch.randn_like(q, requires_grad=True)
                        do = torch.randn_like(q)
                        forward = lambda: impl(q, k, v, True)
                        def full_backward():
                            q.grad = k.grad = v.grad = None
                            impl(q, k, v, True).backward(do)
                        torch.cuda.reset_peak_memory_stats()
                        row['forward_ms'] = measure(forward)
                        row['forward_backward_ms'] = measure(full_backward)
                        # Each repetition needs a fresh autograd graph.  Thus direct backward
                        # timing is the measured full step minus separately measured forward.
                        row['backward_ms'] = row['forward_backward_ms'] - row['forward_ms']
                        row['peak_mib'] = torch.cuda.max_memory_allocated() / 1024**2
                        row['status'] = 'ok'
                    except torch.OutOfMemoryError:
                        row.update(status='oom', forward_ms='', backward_ms='', forward_backward_ms='', peak_mib='')
                        torch.cuda.empty_cache()
                    rows.append(row)
                    print(row, flush=True)
    with args.output.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['implementation','dtype','d_model','seq_len','forward_ms','backward_ms','forward_backward_ms','peak_mib','status'])
        writer.writeheader(); writer.writerows(rows)


if __name__ == '__main__':
    main()
