import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group("nccl")
lr = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(lr)
print("rank", dist.get_rank(), "dev", lr, flush=True)
m = torch.nn.Linear(1024, 1024).cuda()
m = DDP(m, device_ids=[lr])
x = torch.randn(128, 1024, device=f"cuda:{lr}")
y = m(x).sum()
y.backward()
torch.cuda.synchronize()
print("ddp ok", dist.get_rank(), float(y), flush=True)
dist.destroy_process_group()
