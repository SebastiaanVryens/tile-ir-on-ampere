"""cuTile layer norm, with the same two strategies as the Triton version.

cuTile has one load style, so there is no pointer or descriptor split here.
"""

import functools
from collections.abc import Callable
from typing import Any

import cuda.tile as ct
import torch

from tilebench.kernels.layernorm import EPS, LATENCY_HINTS, loop_blocks, whole_row_block
from tilebench.workload import Config, Inputs, Prepared, Size

ZERO = ct.PaddingMode.ZERO
OCCUPANCIES = (None, 1, 2, 4, 8)
# None leaves the choice to the compiler, which picked 4 here. tileiras 13.4 accepts 4 or 8.
WORKER_WARPS = (None, 8)
# None sends no hint. That is not always the same as 3, which is what Triton sends by
# default: they gave different SASS on looped kernels with 1024-element tiles.
LATENCIES = (None, *LATENCY_HINTS)


# The hint is captured rather than passed, because ct.Constant arguments cannot be None.
def whole_body(latency: int | None) -> Callable[..., None]:
    def whole(x, w, b, y, eps, tile: ct.Constant[int]):
        bid = ct.bid(0)
        n = x.shape[1]
        row = ct.load(x, index=(bid, 0), shape=(1, tile), padding_mode=ZERO, latency=latency)
        row = ct.astype(row, ct.float32)
        mean = ct.sum(row, axis=1) / n
        mask = ct.arange(tile, dtype=ct.int32) < n
        centered = ct.where(mask, row - mean, 0)
        rstd = 1 / ct.sqrt(ct.sum(centered * centered, axis=1) / n + eps)
        w_tile = ct.load(w, index=(0,), shape=(tile,), padding_mode=ZERO, latency=latency)
        b_tile = ct.load(b, index=(0,), shape=(tile,), padding_mode=ZERO, latency=latency)
        result = centered * rstd * ct.astype(w_tile, ct.float32) + ct.astype(b_tile, ct.float32)
        ct.store(y, index=(bid, 0), tile=ct.astype(result, y.dtype), latency=latency)

    return whole


def looped_body(latency: int | None) -> Callable[..., None]:
    def looped(x, w, b, y, eps, tile: ct.Constant[int]):
        bid = ct.bid(0)
        n = x.shape[1]
        num_tiles = ct.num_tiles(x, axis=1, shape=(1, tile))

        total = ct.full((1, tile), 0, dtype=ct.float32)
        for j in range(num_tiles):
            row = ct.load(x, index=(bid, j), shape=(1, tile), padding_mode=ZERO, latency=latency)
            total += ct.astype(row, ct.float32)
        mean = ct.sum(total, axis=1) / n

        squares = ct.full((1, tile), 0, dtype=ct.float32)
        for j in range(num_tiles):
            row = ct.load(x, index=(bid, j), shape=(1, tile), padding_mode=ZERO, latency=latency)
            mask = (j * tile + ct.arange(tile, dtype=ct.int32)) < n
            centered = ct.where(mask, ct.astype(row, ct.float32) - mean, 0)
            squares += centered * centered
        rstd = 1 / ct.sqrt(ct.sum(squares, axis=1) / n + eps)

        for j in range(num_tiles):
            row = ct.load(x, index=(bid, j), shape=(1, tile), padding_mode=ZERO, latency=latency)
            w_tile = ct.load(w, index=(j,), shape=(tile,), padding_mode=ZERO, latency=latency)
            b_tile = ct.load(b, index=(j,), shape=(tile,), padding_mode=ZERO, latency=latency)
            normalized = (ct.astype(row, ct.float32) - mean) * rstd
            result = normalized * ct.astype(w_tile, ct.float32) + ct.astype(b_tile, ct.float32)
            ct.store(y, index=(bid, j), tile=ct.astype(result, y.dtype), latency=latency)

    return looped


BODIES = {"whole": whole_body, "looped": looped_body}


@functools.cache
def build_kernel(
    style: str, occupancy: int | None, num_worker_warps: int | None, latency: int | None
) -> Any:
    """Build each variant once, because every new kernel object compiles again."""
    body = BODIES[style](latency)
    return ct.kernel(body, occupancy=occupancy, num_worker_warps=num_worker_warps)


def configs(backend: str, size: Size) -> tuple[Config, ...]:
    _, cols = size
    tiles = {"whole": (whole_row_block(cols),), "looped": loop_blocks(cols)}
    return tuple(
        {
            "style": style,
            "tile": tile,
            "occupancy": occupancy,
            "num_worker_warps": warps,
            "latency": latency,
        }
        for style, style_tiles in tiles.items()
        for tile in style_tiles
        for occupancy in OCCUPANCIES
        for warps in WORKER_WARPS
        for latency in LATENCIES
    )


def prepare(inputs: Inputs, config: Config) -> Prepared:
    x, weight, bias = inputs
    y = torch.empty_like(x)
    kernel = build_kernel(
        config["style"], config["occupancy"], config["num_worker_warps"], config["latency"]
    )
    args = (x, weight, bias, y, EPS, config["tile"])
    grid = (x.shape[0], 1, 1)

    def launch() -> None:
        ct.launch(torch.cuda.current_stream(), grid, kernel, args)

    return Prepared(launch, y, (kernel, args))
