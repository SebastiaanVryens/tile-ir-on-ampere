"""Tiled matrix multiply, C = A @ B, on fp16 with fp32 accumulation.

Vector add and layernorm do little math per byte, so memory bandwidth limits them. A large
GEMM reuses every tile many times, which moves the limit to the tensor cores and to how
well the compiled code keeps them fed.

Sizes are (M, N, K), with A of shape (M, K) and B of shape (K, N).
"""

import torch

from tilebench.workload import Config, Inputs, Size, Workload

SQUARE_SIZES = (1024, 2048, 4096, 8192)
FLOAT16_BYTES = 2

# The Ampere matmul configurations of the Triton tutorial, numbers 1 to 8 and the 64-deep
# tiles 14 to 16 (python/tutorials/03-matrix-multiplication.py). A published list keeps the
# choice of tiles out of the comparison. The warp counts are the tutorial's own pairing for
# each tile. Triton uses all of them, and cuTile only those above its own choice of 4.
TUTORIAL_TILES = (
    # tile_m, tile_n, tile_k, num_warps
    (128, 256, 64, 8),
    (64, 256, 32, 4),
    (128, 128, 32, 4),
    (128, 64, 32, 4),
    (64, 128, 32, 4),
    (128, 32, 32, 4),
    (64, 32, 32, 2),
    (32, 64, 32, 2),
    (128, 64, 64, 4),
    (64, 128, 64, 4),
    (128, 32, 64, 4),
)

# Both Tile IR paths sweep the same values, which match layernorm's. Occupancy 4 capped
# registers at 128 and spilled on 128x128 tiles in both, so the sweep stops at 2.
LATENCY_HINTS = (1, 3, 5, 10)
OCCUPANCIES = (1, 2)

# Blocks walk the output in bands of this many block rows, so that blocks running at the
# same time share tiles of B in L2. Every path uses the same order.
GROUP_ROWS = 8

# Descriptors need 16-byte aligned rows, which is 8 fp16 elements.
DESCRIPTOR_ALIGNMENT = 8

# Tensor cores reach the power limit within a second, and the chip jumps about 18 C. From
# 70 C, 5 s bursts at 8192 reached 87 C and slowed for heat once the heatsink was warm.
# From 60 C they peaked at 85 C, limited by power alone.
START_TEMP_C = 60


def tile_fields(tile_m: int, tile_n: int, tile_k: int) -> Config:
    return {"tile_m": tile_m, "tile_n": tile_n, "tile_k": tile_k}


def make_inputs(size: Size) -> Inputs:
    m, n, k = size
    a = torch.randn(m, k, device="cuda", dtype=torch.float16)
    b = torch.randn(k, n, device="cuda", dtype=torch.float16)
    return a, b


def reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Multiply in fp32, so that cuBLAS is checked too rather than being the answer."""
    return (a.float() @ b.float()).to(a.dtype)


def flops(size: Size) -> float:
    """Return the floating-point work, counting one multiply and one add per term."""
    m, n, k = size
    return 2 * m * n * k


WORKLOAD = Workload(
    name="gemm",
    sizes=tuple((n, n, n) for n in SQUARE_SIZES),
    # Tiles do not divide most of these, which exercises the edges of M, N and K, and 600
    # leaves a last band shorter than GROUP_ROWS. 129 and 1025 are not aligned for
    # descriptors, and 8192 is a long sum.
    check_sizes=(
        (1, 8, 8),
        (64, 64, 64),
        (128, 256, 512),
        (600, 200, 1000),
        (65, 129, 1025),
        (256, 256, 8192),
    ),
    make_inputs=make_inputs,
    reference=reference,
    work=flops,
    rate_unit="TFLOPS",
    rate_scale=1e12,
    # Two correct fp16 results can differ by one rounding step, at most 2^-10 of their value.
    # Near zero, the worst measured error on any path was 0.003, at 8192x8192x8192. This is
    # tight enough to catch a kernel that accumulates in fp16.
    rtol=1e-3,
    atol=1e-2,
    max_start_temp_c=START_TEMP_C,
)
