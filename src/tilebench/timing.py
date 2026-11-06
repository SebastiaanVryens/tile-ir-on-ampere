"""GPU timing shared by every target, so all paths are measured the same way."""

import time
from collections.abc import Callable
from typing import Any

import torch

# Larger than L2, so each timed launch starts cold. Zeroing it also keeps the GPU busy
# while the host queues the launch. Without that, short kernels measure launch latency:
# a 2^16 vector add read 41 us instead of 7 us under WSL2.
FLUSH_BYTES = 256 * 1024 * 1024

# Clocks only stay up under continuous load. Timing one kernel at a time with 100 ms gaps
# ran at about 580 MHz and read up to 2.5x slow, and nvidia-smi calls count as gaps. So
# kernels are timed in bursts: one warmup, then launches back to back. Getting clocks back
# from idle took up to 0.53 s.
FULL_WARMUP_S = 0.75

# A burst has to end before the GPU overheats, so this bounds it in launches.
LAUNCHES_PER_BURST = 640

# Untimed launches before each kernel, which load its code onto the GPU.
SETTLE_LAUNCHES = 3

REPEATS = 100


def burst_length(repeats: int) -> int:
    """Return how many kernels fit in one burst at this many repeats each."""
    return max(1, LAUNCHES_PER_BURST // repeats)


class Timer:
    """Times launches with a cold L2."""

    def __init__(self) -> None:
        self.flush = torch.empty(FLUSH_BYTES, dtype=torch.int8, device="cuda")

    def warm_up(self, launch: Callable[[], Any]) -> None:
        """Bring the GPU to full clocks. Call this at the start of every burst."""
        # Synchronizing each pass keeps the warmup to its real duration. Otherwise the host
        # runs far ahead of the GPU and the backlog outlasts the deadline.
        deadline = time.perf_counter() + FULL_WARMUP_S
        while time.perf_counter() < deadline:
            self.flush.zero_()
            launch()
            torch.cuda.synchronize()

    def gpu_ms(self, launch: Callable[[], Any], repeats: int = REPEATS) -> list[float]:
        """Return the GPU time of each launch, in milliseconds."""
        for _ in range(SETTLE_LAUNCHES):
            self.flush.zero_()
            launch()

        events = [
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(repeats)
        ]
        for start, end in events:
            self.flush.zero_()
            start.record()
            launch()
            end.record()
        torch.cuda.synchronize()
        return [start.elapsed_time(end) for start, end in events]
