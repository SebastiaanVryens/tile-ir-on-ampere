"""Layer normalization over the last dimension of an fp16 matrix.

Each row needs two reductions before it can be normalized, so a kernel either holds the
whole row at once or reads it several times. That choice gives the compilers something
to differ on, which vector add does not.
"""

import torch
import torch.nn.functional as F

from tilebench.workload import Inputs, Size, Workload

EPS = 1e-5
FLOAT16_BYTES = 2
TOTAL_ELEMENTS = 1 << 24
ROW_LENGTHS = tuple(1 << power for power in range(10, 17))
LOOP_BLOCKS = (1024, 4096)

# Descriptors need 16-byte aligned rows, which is 8 fp16 elements.
DESCRIPTOR_ALIGNMENT = 8


def whole_row_block(cols: int) -> int:
    """Return the smallest power-of-two tile that holds a whole row."""
    return 1 << (cols - 1).bit_length()


def loop_blocks(cols: int) -> tuple[int, ...]:
    return tuple(sorted({min(block, whole_row_block(cols)) for block in LOOP_BLOCKS}))


def make_inputs(size: Size) -> Inputs:
    rows, cols = size
    x = torch.randn(rows, cols, device="cuda", dtype=torch.float16)
    weight = torch.randn(cols, device="cuda", dtype=torch.float16)
    bias = torch.randn(cols, device="cuda", dtype=torch.float16)
    return x, weight, bias


def reference(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(x, (x.shape[-1],), weight, bias, EPS)


def bytes_moved(size: Size) -> float:
    """Return the minimum traffic of one read and one write per element.

    Looped variants read each row more than once, so their GB/s is an effective rate
    rather than the bandwidth they actually use.
    """
    rows, cols = size
    return 2 * rows * cols * FLOAT16_BYTES


WORKLOAD = Workload(
    name="layernorm",
    sizes=tuple((TOTAL_ELEMENTS // cols, cols) for cols in ROW_LENGTHS),
    # Row lengths that are not powers of two exercise the masks, and 1025 and 4097 are
    # not aligned for descriptors.
    check_sizes=((1, 8), (3, 1000), (5, 1024), (7, 1025), (4, 2048), (2, 4097), (2, 65536)),
    make_inputs=make_inputs,
    reference=reference,
    work=bytes_moved,
    rate_unit="GB/s",
    rate_scale=1e9,
    rtol=1e-2,
    atol=2e-2,
)
