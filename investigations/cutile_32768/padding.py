"""check_bounds=False with a padding mode: compiles on tileiras 13.5, fails on 13.4.

The docs say the padding mode is ignored in this case. On 13.4 the frontend still puts it on
the view, and tileiras refuses a padded view with any inbounds promise, without saying why.
"""

from collections.abc import Callable
from importlib.metadata import version

import cuda.tile as ct
import torch

PAD = ct.PaddingMode
MODES = (PAD.UNDETERMINED, PAD.ZERO, PAD.NEG_ZERO, PAD.NAN, PAD.POS_INF, PAD.NEG_INF)
TILE, TILES = 256, 4


def flat(pad: PAD, check: bool) -> Callable[..., None]:
    def copy(x, y, tile: ct.Constant[int]):
        bid = ct.bid(0)
        values = ct.load(x, index=(bid,), shape=(tile,), padding_mode=pad, check_bounds=check)
        ct.store(y, index=(bid,), tile=values)

    return copy


def rows(pad: PAD, check: bool) -> Callable[..., None]:
    def copy(x, y, tile: ct.Constant[int]):
        bid = ct.bid(0)
        values = ct.load(x, index=(bid, 0), shape=(1, tile), padding_mode=pad, check_bounds=check)
        ct.store(y, index=(bid, 0), tile=values)

    return copy


def tiled_view(pad: PAD, check: bool) -> Callable[..., None]:
    def copy(x, y, tile: ct.Constant[int]):
        bid = ct.bid(0)
        values = x.tiled_view((tile,), padding_mode=pad).load((bid,), check_bounds=check)
        ct.store(y, index=(bid,), tile=values)

    return copy


FORMS = {
    "ct.load 1D": (flat, (TILE * TILES,)),
    "ct.load 2D": (rows, (TILES, TILE)),
    "TiledView": (tiled_view, (TILE * TILES,)),
}


def attempt(
    build: Callable[[PAD, bool], Callable[..., None]], shape: tuple[int, ...], pad: PAD, check: bool
) -> str:
    x = torch.arange(TILE * TILES, device="cuda", dtype=torch.float32).reshape(shape)
    y = torch.zeros_like(x)
    try:
        kernel = ct.kernel(build(pad, check))
        ct.launch(torch.cuda.current_stream(), (TILES, 1, 1), kernel, (x, y, TILE))
        torch.cuda.synchronize()
    except ct.TileCompilerExecutionError:
        return "compiler refused"
    return "ok" if torch.equal(x, y) else "WRONG OUTPUT"


def main() -> None:
    print(f"cuda-tile {ct.__version__}, tileiras {version('nvidia-cuda-tileiras')}")
    print(f"{'form':<12}{'padding':<14}{'checked':<18}{'unchecked':<18}")
    for form, (build, shape) in FORMS.items():
        for pad in MODES:
            checked = attempt(build, shape, pad, True)
            unchecked = attempt(build, shape, pad, False)
            print(f"{form:<12}{pad.name:<14}{checked:<18}{unchecked:<18}", flush=True)


if __name__ == "__main__":
    main()
