"""Triton GEMM, shared by the PTX and Tile IR targets.

The two kernels differ only in how they load tiles, which is the variable the
tensor-of-pointer investigation depends on.
"""

from typing import Any

import torch
import triton
import triton.language as tl

from tilebench.kernels.gemm import (
    DESCRIPTOR_ALIGNMENT,
    FLOAT16_BYTES,
    GROUP_ROWS,
    LATENCY_HINTS,
    OCCUPANCIES,
    TUTORIAL_TILES,
    tile_fields,
)
from tilebench.workload import Config, Inputs, Prepared, Size


@triton.jit
def block_position(M, N, BM: tl.constexpr, BN: tl.constexpr, GROUP: tl.constexpr):
    """Return the output tile of this block, walking bands of GROUP block rows."""
    pid = tl.program_id(0)
    blocks_m = tl.cdiv(M, BM)
    blocks_n = tl.cdiv(N, BN)
    band_size = GROUP * blocks_n
    first_m = (pid // band_size) * GROUP
    band_rows = min(blocks_m - first_m, GROUP)
    in_band = pid % band_size
    return first_m + in_band % band_rows, in_band // band_rows


@triton.jit
def gemm_pointer(
    a_ptr, b_ptr, c_ptr, M, N, K,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr,
):
    block_m, block_n = block_position(M, N, BM, BN, GROUP)
    rows = block_m * BM + tl.arange(0, BM)
    cols = block_n * BN + tl.arange(0, BN)
    steps = tl.arange(0, BK)
    # Wrapping keeps edge tiles in bounds, so loads only need a mask along K. The wrapped
    # rows and columns produce results that the store below drops.
    a_ptrs = a_ptr + (rows % M)[:, None] * K + steps[None, :]
    b_ptrs = b_ptr + steps[:, None] * N + (cols % N)[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(a_ptrs, mask=steps[None, :] < K - k, other=0.0)
        b = tl.load(b_ptrs, mask=steps[:, None] < K - k, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BK
        b_ptrs += BK * N
    c_ptrs = c_ptr + rows[:, None] * N + cols[None, :]
    tl.store(c_ptrs, acc.to(tl.float16), mask=(rows[:, None] < M) & (cols[None, :] < N))


@triton.jit
def gemm_descriptor(
    a_ptr, b_ptr, c_ptr, M, N, K,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr,
):
    block_m, block_n = block_position(M, N, BM, BN, GROUP)
    a_desc = tl.make_tensor_descriptor(a_ptr, shape=[M, K], strides=[K, 1], block_shape=[BM, BK])
    b_desc = tl.make_tensor_descriptor(b_ptr, shape=[K, N], strides=[N, 1], block_shape=[BK, BN])
    c_desc = tl.make_tensor_descriptor(c_ptr, shape=[M, N], strides=[N, 1], block_shape=[BM, BN])
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    # Descriptors read zeros past the edge of the array, so no masks are needed.
    for k in range(0, K, BK):
        acc = tl.dot(a_desc.load([block_m * BM, k]), b_desc.load([k, block_n * BN]), acc)
    c_desc.store([block_m * BM, block_n * BN], acc.to(tl.float16))


KERNELS = {"pointer": gemm_pointer, "descriptor": gemm_descriptor}

# On PTX, num_stages sets the depth of the async copy pipeline: from 1 to 5 stages, the
# 128x128x32 kernel went from plain loads to 40 async copies and 64 KB of shared memory.
PTX_STAGES = (2, 3, 4, 5)

# The PTX backend keeps num_stages - 1 copies of both input tiles in shared memory, and
# sm_86 allows 99 KB per block. 128x256x64 fit at 3 stages and asked for 144 KB at 4.
SHARED_LIMIT_BYTES = 99 * 1024

TILE_KEYS = ("style", "tile_m", "tile_n", "tile_k")


def scratch_alloc(size: int, alignment: int, stream: int | None) -> torch.Tensor:
    return torch.empty(size, device="cuda", dtype=torch.int8)


triton.set_allocator(scratch_alloc)


def pipeline_bytes(tile_m: int, tile_n: int, tile_k: int, stages: int) -> int:
    return (stages - 1) * (tile_m * tile_k + tile_k * tile_n) * FLOAT16_BYTES


def knobs(backend: str, tile_m: int, tile_n: int, tile_k: int, warps: int) -> list[Config]:
    if backend == "cuda":
        return [
            {"num_warps": warps, "num_stages": stages}
            for stages in PTX_STAGES
            if pipeline_bytes(tile_m, tile_n, tile_k, stages) <= SHARED_LIMIT_BYTES
        ]
    # On Tile IR, num_stages becomes the latency hint, and the compiler sizes the pipeline.
    return [
        {"num_warps": warps, "num_stages": hint, "occupancy": occupancy}
        for hint in LATENCY_HINTS
        for occupancy in OCCUPANCIES
    ]


def configs(backend: str, size: Size) -> tuple[Config, ...]:
    _, n, k = size
    aligned = n % DESCRIPTOR_ALIGNMENT == 0 and k % DESCRIPTOR_ALIGNMENT == 0
    styles = tuple(KERNELS) if aligned else ("pointer",)
    return tuple(
        {"style": style, **tile_fields(tile_m, tile_n, tile_k), **knob}
        for style in styles
        for tile_m, tile_n, tile_k, warps in TUTORIAL_TILES
        for knob in knobs(backend, tile_m, tile_n, tile_k, warps)
    )


def prepare(inputs: Inputs, config: Config) -> Prepared:
    a, b = inputs
    m, k = a.shape
    n = b.shape[1]
    c = torch.empty(m, n, device=a.device, dtype=a.dtype)
    kernel = KERNELS[config["style"]]
    tile_m, tile_n, tile_k = config["tile_m"], config["tile_n"], config["tile_k"]
    options = {key: value for key, value in config.items() if key not in TILE_KEYS}
    grid = (triton.cdiv(m, tile_m) * triton.cdiv(n, tile_n),)

    def launch() -> Any:
        return kernel[grid](
            a, b, c, m, n, k, BM=tile_m, BN=tile_n, BK=tile_k, GROUP=GROUP_ROWS, **options
        )

    return Prepared(launch, c)
