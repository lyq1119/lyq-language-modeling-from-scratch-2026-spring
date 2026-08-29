from pathlib import Path
import argparse
import json, math, time
import torch

BATCH = 8
DIMS = (16, 32, 64, 128)
LENGTHS = (256, 1024, 4096, 8192, 16384)
WARMUP = 5
REPS = 100

def attention(q, k, v):
    scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(q.shape[-1])
    mask = torch.ones(q.shape[1], q.shape[1], device=q.device, dtype=torch.bool).triu(1)
    scores = scores.masked_fill(mask, float("-inf"))
    return torch.bmm(torch.softmax(scores, dim=-1), v)

def sync(): torch.cuda.synchronize()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true", help="benchmark torch.compile(attention)")
    parser.add_argument("--output", type=Path, default=Path("profiles/attention_benchmark.json"))
    args = parser.parse_args()
    device = torch.device("cuda")
    attention_fn = torch.compile(attention) if args.compile else attention
    rows = []
    for d in DIMS:
      for s in LENGTHS:
        row = {"batch_size": BATCH, "d_model": d, "sequence_length": s}
        try:
          q = torch.randn(BATCH,s,d,device=device,requires_grad=True)
          k = torch.randn_like(q,requires_grad=True); v = torch.randn_like(q,requires_grad=True)
          for _ in range(WARMUP):
            attention_fn(q,k,v).sum().backward(); q.grad=k.grad=v.grad=None
          sync(); forward=[]; backward=[]; peak=0
          for _ in range(REPS):
            torch.cuda.reset_peak_memory_stats(device); sync(); t=time.perf_counter()
            out=attention_fn(q,k,v); sync(); forward.append((time.perf_counter()-t)*1e3)
            peak=max(peak, torch.cuda.max_memory_allocated(device))
            t=time.perf_counter(); out.sum().backward(); sync(); backward.append((time.perf_counter()-t)*1e3)
            q.grad=k.grad=v.grad=None
          row.update(status="ok", forward_ms=sum(forward)/REPS, backward_ms=sum(backward)/REPS, pre_backward_mib=peak/1024**2)
        except torch.OutOfMemoryError:
          row.update(status="oom")
          torch.cuda.empty_cache()
        print(json.dumps(row), flush=True); rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
if __name__ == "__main__": main()
