"""The cuTile GEMM, plus a variant that tells a just-in-time launch the sizes divide by 8.

cuTile records an array's shape as divisible by 16 or by nothing. ct.assume_divisible_by
only applies to integer scalars, so assuming it on a.shape[0] alone changes nothing: the
loads still check the array's own shape. The variant slices each array to its own length,
with that length assumed divisible by 8, and loads from the slices. The slices cover the
whole array, so only what the compiler knows changes. The caller must check that the sizes
are multiples of 8; otherwise the behaviour is undefined.

This module has the shape of a benchmark implementation: configurations with the style
"assume" build the variant, and all others the repository's kernel.
"""

import functools
from collections.abc import Callable
from typing import Any

import cuda.tile as ct
import torch

from tilebench.kernels.gemm import GROUP_ROWS, cutile_impl
from tilebench.kernels.gemm.cutile_impl import ZERO, block_position
from tilebench.workload import Config, Inputs, Prepared, Size

STYLE = "assume"
DIVISOR = 8


# The hint is captured rather than passed, as in cutile_impl, because ct.Constant
# arguments cannot be None.
def gemm_assume_body(latency: int | None) -> Callable[..., None]:
    def gemm_assume(
        a,
        b,
        c,
        tile_m: ct.Constant[int],
        tile_n: ct.Constant[int],
        tile_k: ct.Constant[int],
        group: ct.Constant[int],
    ):
        m = ct.assume_divisible_by(a.shape[0], DIVISOR)
        k_size = ct.assume_divisible_by(a.shape[1], DIVISOR)
        n = ct.assume_divisible_by(b.shape[1], DIVISOR)
        a_view = a.slice(0, 0, m).slice(1, 0, k_size)
        b_view = b.slice(0, 0, k_size).slice(1, 0, n)
        c_view = c.slice(0, 0, m).slice(1, 0, n)
        block_m, block_n = block_position(m, n, tile_m, tile_n, group)
        steps = ct.num_tiles(a_view, axis=1, shape=(tile_m, tile_k))
        acc = ct.full((tile_m, tile_n), 0, dtype=ct.float32)
        for k in range(steps):
            a_tile = ct.load(
                a_view, index=(block_m, k), shape=(tile_m, tile_k), padding_mode=ZERO, latency=latency
            )
            b_tile = ct.load(
                b_view, index=(k, block_n), shape=(tile_k, tile_n), padding_mode=ZERO, latency=latency
            )
            acc = ct.mma(a_tile, b_tile, acc)
        ct.store(c_view, index=(block_m, block_n), tile=ct.astype(acc, c.dtype), latency=latency)

    return gemm_assume


@functools.cache
def build_assume_kernel(occupancy: int, latency: int | None, num_worker_warps: int | None) -> Any:
    return ct.kernel(gemm_assume_body(latency), occupancy=occupancy, num_worker_warps=num_worker_warps)


def configs(backend: str, size: Size) -> tuple[Config, ...]:
    return cutile_impl.configs(backend, size)


def prepare(inputs: Inputs, config: Config) -> Prepared:
    if config.get("style") != STYLE:
        return cutile_impl.prepare(inputs, config)
    a, b = inputs
    m = a.shape[0]
    n = b.shape[1]
    if any(dim % DIVISOR for dim in (m, n, a.shape[1])):
        raise ValueError(f"the sizes must be multiples of {DIVISOR}, got {m}x{n}x{a.shape[1]}")
    c = torch.empty(m, n, device=a.device, dtype=a.dtype)
    kernel = build_assume_kernel(config["occupancy"], config["latency"], config["num_worker_warps"])
    tile_m, tile_n, tile_k = config["tile_m"], config["tile_n"], config["tile_k"]
    grid = (ct.cdiv(m, tile_m) * ct.cdiv(n, tile_n), 1, 1)
    args = (a, b, c, tile_m, tile_n, tile_k, GROUP_ROWS)

    def launch() -> None:
        ct.launch(torch.cuda.current_stream(), grid, kernel, args)

    return Prepared(launch, c, (kernel, args))
