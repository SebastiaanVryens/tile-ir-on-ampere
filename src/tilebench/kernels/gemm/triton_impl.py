"""Triton GEMM, shared by the PTX and Tile IR targets.

The two kernels differ only in how they load tiles, which is the variable the
tensor-of-pointer investigation depends on.
"""

from typing import Any

import torch
import triton
import triton.language as tl

from tilebench.kernels.gemm import DESCRIPTOR_ALIGNMENT, GROUP_ROWS, TILES
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


def scratch_alloc(size: int, alignment: int, stream: int | None) -> torch.Tensor:
    return torch.empty(size, device="cuda", dtype=torch.int8)


triton.set_allocator(scratch_alloc)


def configs(backend: str, size: Size) -> tuple[Config, ...]:
    _, n, k = size
    aligned = n % DESCRIPTOR_ALIGNMENT == 0 and k % DESCRIPTOR_ALIGNMENT == 0
    styles = tuple(KERNELS) if aligned else ("pointer",)
    return tuple({"style": style, **tile} for style in styles for tile in TILES)


def prepare(inputs: Inputs, config: Config) -> Prepared:
    a, b = inputs
    m, k = a.shape
    n = b.shape[1]
    c = torch.empty(m, n, device=a.device, dtype=a.dtype)
    kernel = KERNELS[config["style"]]
    tile_m, tile_n, tile_k = config["tile_m"], config["tile_n"], config["tile_k"]
    grid = (triton.cdiv(m, tile_m) * triton.cdiv(n, tile_n),)

    def launch() -> Any:
        return kernel[grid](
            a, b, c, m, n, k, BM=tile_m, BN=tile_n, BK=tile_k, GROUP=GROUP_ROWS
        )

    return Prepared(launch, c)
