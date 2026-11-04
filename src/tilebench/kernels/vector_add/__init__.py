"""Vector add: a memory-bound baseline.

Every path should reach the same DRAM bandwidth here, so this workload validates the
harness rather than separating the backends.
"""

import torch

from tilebench.workload import Inputs, Size, Workload

FLOAT32_BYTES = 4


def make_inputs(size: Size) -> Inputs:
    (n,) = size
    return torch.randn(n, device="cuda"), torch.randn(n, device="cuda")


def bytes_moved(size: Size) -> float:
    (n,) = size
    return 3 * n * FLOAT32_BYTES


WORKLOAD = Workload(
    name="vector_add",
    sizes=tuple((1 << power,) for power in (16, 20, 22, 24, 26)),
    # Sizes that are not multiples of any tile exercise the masking and bounds handling.
    check_sizes=((1,), (1000,), (1023,), (1025,), (4097,), (1_000_003,)),
    make_inputs=make_inputs,
    reference=torch.add,
    work=bytes_moved,
    rate_unit="GB/s",
    rate_scale=1e9,
    rtol=0.0,
    atol=0.0,
)
