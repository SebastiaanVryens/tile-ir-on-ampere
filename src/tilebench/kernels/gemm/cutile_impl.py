"""cuTile GEMM. ct.load addresses whole tiles, so there is no pointer or descriptor split."""

import functools
from collections.abc import Callable
from typing import Any

import cuda.tile as ct
import torch

from tilebench.kernels.gemm import (
    GROUP_ROWS,
    LATENCY_HINTS,
    OCCUPANCIES,
    TUTORIAL_TILES,
    tile_fields,
)
from tilebench.workload import Config, Inputs, Prepared, Size

ZERO = ct.PaddingMode.ZERO

# None sends no hint, as in layernorm, where it was not always the same as any value.
LATENCIES = (None, *LATENCY_HINTS)

# The warp count cuTile chose by itself for every tile we inspected.
COMPILER_WARPS = 4


def worker_warps(tutorial_warps: int) -> int | None:
    """Return the warp count to request, or None to leave the choice to the compiler.

    Only tiles that the tutorial gives more warps than the compiler's choice get a request.
    On 128x256x64 the compiler's 4 warps left each thread a 256-register fp32 accumulator,
    so it spilled 1112 bytes and ran at 6.6 TFLOPS at 4096, where 8 warps ran at 22.9 with
    no spill. On 128x128x32 the compiler's choice gave the same code as 4 warps. Requests
    below 4 need tileiras 13.5, so the 2-warp tiles keep the compiler's choice as well.
    """
    return tutorial_warps if tutorial_warps > COMPILER_WARPS else None


def block_position(m, n, tile_m, tile_n, group):
    """Return the output tile of this block, walking bands of `group` block rows."""
    bid = ct.bid(0)
    blocks_m = ct.cdiv(m, tile_m)
    blocks_n = ct.cdiv(n, tile_n)
    band_size = group * blocks_n
    first_m = (bid // band_size) * group
    band_rows = min(blocks_m - first_m, group)
    in_band = bid % band_size
    return first_m + in_band % band_rows, in_band // band_rows


# The hint is captured rather than passed, because ct.Constant arguments cannot be None.
def gemm_body(latency: int | None) -> Callable[..., None]:
    def gemm(
        a,
        b,
        c,
        tile_m: ct.Constant[int],
        tile_n: ct.Constant[int],
        tile_k: ct.Constant[int],
        group: ct.Constant[int],
    ):
        block_m, block_n = block_position(a.shape[0], b.shape[1], tile_m, tile_n, group)
        steps = ct.num_tiles(a, axis=1, shape=(tile_m, tile_k))
        acc = ct.full((tile_m, tile_n), 0, dtype=ct.float32)
        for k in range(steps):
            # The padding matters along K: anything but zero would enter the sum.
            a_tile = ct.load(
                a, index=(block_m, k), shape=(tile_m, tile_k), padding_mode=ZERO, latency=latency
            )
            b_tile = ct.load(
                b, index=(k, block_n), shape=(tile_k, tile_n), padding_mode=ZERO, latency=latency
            )
            acc = ct.mma(a_tile, b_tile, acc)
        ct.store(c, index=(block_m, block_n), tile=ct.astype(acc, c.dtype), latency=latency)

    return gemm


@functools.cache
def build_kernel(occupancy: int, latency: int | None, num_worker_warps: int | None) -> Any:
    """Build each variant once, because every new kernel object compiles again."""
    return ct.kernel(gemm_body(latency), occupancy=occupancy, num_worker_warps=num_worker_warps)


def configs(backend: str, size: Size) -> tuple[Config, ...]:
    return tuple(
        {
            **tile_fields(tile_m, tile_n, tile_k),
            "num_worker_warps": worker_warps(warps),
            "occupancy": occupancy,
            "latency": latency,
        }
        for tile_m, tile_n, tile_k, warps in TUTORIAL_TILES
        for occupancy in OCCUPANCIES
        for latency in LATENCIES
    )


def prepare(inputs: Inputs, config: Config) -> Prepared:
    a, b = inputs
    m = a.shape[0]
    n = b.shape[1]
    c = torch.empty(m, n, device=a.device, dtype=a.dtype)
    kernel = build_kernel(config["occupancy"], config["latency"], config["num_worker_warps"])
    tile_m, tile_n, tile_k = config["tile_m"], config["tile_n"], config["tile_k"]
    grid = (ct.cdiv(m, tile_m) * ct.cdiv(n, tile_n), 1, 1)
    args = (a, b, c, tile_m, tile_n, tile_k, GROUP_ROWS)

    def launch() -> None:
        ct.launch(torch.cuda.current_stream(), grid, kernel, args)

    return Prepared(launch, c, (kernel, args))
