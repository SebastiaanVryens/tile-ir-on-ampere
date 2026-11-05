"""cuTile vector add."""

import cuda.tile as ct
import torch

from tilebench.workload import Config, Inputs, Prepared, Size

TILES = (1024, 4096)


@ct.kernel
def add_kernel(a, b, c, tile: ct.Constant[int]):
    pid = ct.bid(0)
    a_tile = ct.load(a, index=(pid,), shape=(tile,))
    b_tile = ct.load(b, index=(pid,), shape=(tile,))
    ct.store(c, index=(pid,), tile=a_tile + b_tile)


def configs(backend: str, size: Size) -> tuple[Config, ...]:
    return tuple({"tile": tile} for tile in TILES)


def prepare(inputs: Inputs, config: Config) -> Prepared:
    a, b = inputs
    c = torch.empty_like(a)
    tile = config["tile"]
    grid = (ct.cdiv(a.numel(), tile), 1, 1)

    def launch() -> None:
        ct.launch(torch.cuda.current_stream(), grid, add_kernel, (a, b, c, tile))

    return Prepared(launch, c)
