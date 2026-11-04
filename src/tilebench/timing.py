"""GPU timing shared by every target, so all paths are measured the same way."""

import time
from collections.abc import Callable
from typing import Any

import torch

# Larger than L2, so each timed launch starts cold. Zeroing it also keeps the GPU busy
# while the host queues the launch. Without that, short kernels measure launch latency:
# a 2^16 vector add read 41 us instead of 7 us under WSL2.
FLUSH_BYTES = 256 * 1024 * 1024

# Long enough to leave idle clocks behind: 50 ms of load already reached full clocks.
WARMUP_S = 0.3

REPEATS = 100


def measure_gpu_ms(launch: Callable[[], Any], repeats: int = REPEATS) -> list[float]:
    """Return the GPU time of each launch, in milliseconds, with a cold L2."""
    flush = torch.empty(FLUSH_BYTES, dtype=torch.int8, device="cuda")

    # Synchronizing each pass keeps the warmup to its real duration. Otherwise the host
    # runs far ahead of the GPU and the backlog outlasts the deadline.
    deadline = time.perf_counter() + WARMUP_S
    while time.perf_counter() < deadline:
        flush.zero_()
        launch()
        torch.cuda.synchronize()

    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(repeats)
    ]
    for start, end in events:
        flush.zero_()
        start.record()
        launch()
        end.record()
    torch.cuda.synchronize()
    return [start.elapsed_time(end) for start, end in events]

