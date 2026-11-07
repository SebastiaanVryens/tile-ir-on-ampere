"""cuTile GEMM. ct.load addresses whole tiles, so there is no pointer or descriptor split."""

import cuda.tile as ct
import torch

from tilebench.kernels.gemm import GROUP_ROWS, TILES
from tilebench.workload import Config, Inputs, Prepared, Size

ZERO = ct.PaddingMode.ZERO


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


@ct.kernel
def gemm_kernel(
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
        a_tile = ct.load(a, index=(block_m, k), shape=(tile_m, tile_k), padding_mode=ZERO)
        b_tile = ct.load(b, index=(k, block_n), shape=(tile_k, tile_n), padding_mode=ZERO)
        acc = ct.mma(a_tile, b_tile, acc)
    ct.store(c, index=(block_m, block_n), tile=ct.astype(acc, c.dtype))


def configs(backend: str, size: Size) -> tuple[Config, ...]:
    return tuple(dict(tile) for tile in TILES)


def prepare(inputs: Inputs, config: Config) -> Prepared:
    a, b = inputs
    m = a.shape[0]
    n = b.shape[1]
    c = torch.empty(m, n, device=a.device, dtype=a.dtype)
    tile_m, tile_n, tile_k = config["tile_m"], config["tile_n"], config["tile_k"]
    grid = (ct.cdiv(m, tile_m) * ct.cdiv(n, tile_n), 1, 1)
    args = (a, b, c, tile_m, tile_n, tile_k, GROUP_ROWS)

    def launch() -> None:
        ct.launch(torch.cuda.current_stream(), grid, gemm_kernel, args)

    return Prepared(launch, c, (gemm_kernel, args))
