from __future__ import annotations

import os
from datetime import timedelta

import torch
import torch.distributed as dist


def _process_group_timeout() -> timedelta:
    sec = int(os.environ.get("DIST_TIMEOUT_SECONDS", str(6 * 3600)))
    return timedelta(seconds=max(sec, 300))


def ddp_enabled() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_main_process() -> bool:
    return not dist.is_initialized() or get_rank() == 0


def setup_distributed() -> torch.device:
    local_rank = get_local_rank()
    timeout = _process_group_timeout()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", timeout=timeout)
        return torch.device("cuda", local_rank)
    dist.init_process_group(backend="gloo", timeout=timeout)
    return torch.device("cpu")


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def dist_barrier() -> None:
    if not dist.is_initialized():
        return
    if torch.cuda.is_available():
        local_rank = get_local_rank()
        torch.cuda.set_device(local_rank)
        try:
            dist.barrier(device_ids=[local_rank])
        except TypeError:
            dist.barrier()
    else:
        dist.barrier()


def broadcast_scalar_tensor(values: list[float], device: torch.device) -> list[float]:
    if not dist.is_initialized():
        return values
    t = torch.tensor(values, dtype=torch.float32, device=device)
    dist.broadcast(t, src=0)
    return [float(x) for x in t.tolist()]
