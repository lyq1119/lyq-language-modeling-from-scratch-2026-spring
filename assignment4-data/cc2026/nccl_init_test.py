import os

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29517")

import torch.distributed as dist

dist.init_process_group("nccl", rank=0, world_size=1)
print("NCCL_INIT_OK", flush=True)
