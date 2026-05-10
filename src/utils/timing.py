import time
from collections.abc import Callable

import torch


def batch_average_inference_seconds(
    forward_fn: Callable[[], None],
    device: torch.device,
    num_warmup: int = 2,
    num_batches: int = 10,
) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize()
    for _ in range(num_warmup):
        forward_fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_batches):
        forward_fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return elapsed / max(num_batches, 1)
