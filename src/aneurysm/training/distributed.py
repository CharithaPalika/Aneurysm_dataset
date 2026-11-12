"""Multi-GPU helpers (PyTorch DistributedDataParallel, one process per GPU).

Started with torchrun, e.g. on 4 GPUs:
    torchrun --standalone --nproc_per_node=4 scripts/run_train.py --config configs/train/lefusion_train_a100.yaml
torchrun sets RANK / LOCAL_RANK / WORLD_SIZE. Without torchrun everything runs as one process
on one device (notebooks, the Windows pilot), and these helpers do nothing.

Rules used in the trainer:
- `batch_size` is per GPU: effective batch = batch_size x grad_accum x number of GPUs.
- Only the main process (rank 0) writes logs, checkpoints, samples and runs validation.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist


@dataclass
class DistInfo:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    device: torch.device | None = None

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def enabled(self) -> bool:
        return self.world_size > 1


def setup(default_device: torch.device) -> DistInfo:
    """Join the process group if started by torchrun (safe to call more than once)."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world <= 1:
        return DistInfo(device=default_device)
    rank, local_rank = int(os.environ["RANK"]), int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device, backend = torch.device("cuda", local_rank), "nccl"
    else:                                                  # CPU test runs
        device, backend = torch.device("cpu"), "gloo"
    if not dist.is_initialized():
        # long timeout: rank 0 alone writes checkpoints / makes preview samples while the others wait
        dist.init_process_group(backend=backend, timeout=timedelta(hours=2))
    return DistInfo(rank=rank, local_rank=local_rank, world_size=world, device=device)


def barrier(info: DistInfo) -> None:
    if info.enabled:
        dist.barrier()


def mean_over_gpus(value: float, info: DistInfo) -> float:
    """Average a number over all processes (e.g. the step loss for logging)."""
    if not info.enabled:
        return value
    t = torch.tensor([value], dtype=torch.float64, device=info.device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item()) / info.world_size


def cleanup() -> None:
    """Leave the process group (call once at the end of a script)."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def broadcast_object(obj, info: DistInfo):
    """Give every process rank 0's value (e.g. a run name with a time stamp)."""
    if not info.enabled:
        return obj
    box = [obj]
    dist.broadcast_object_list(box, src=0)
    return box[0]
