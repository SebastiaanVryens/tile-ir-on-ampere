"""The main loop of each GEMM kernel, per step along K, at 4,096.

Usage, from the activated venv: python investigations/tensor_of_pointer/main_loop.py

On the reference setting (128x256x64, 8 warps, 3 stages or hint 3, occupancy 1 on Tile IR),
on both Triton backends:
  pointer       the repository's kernel: rows and columns wrap, so only K is masked
  descriptor    the repository's descriptor kernel
  masked        pointer loads written the plain way: masked along both dimensions
  ca            the pointer kernel with loads that ask to be cached in L1
On 64x256x32 at occupancy 2, the tile the GEMM results compare cuTile on: both Triton Tile
IR kernels (4 warps, hint 3) and cuTile (its own warp count, latency 1).
Nothing is timed. The loop is the backward branch with the most tensor core instructions,
and every count is divided by the steps along K that one iteration covers.
"""

from typing import Any

from common import (
    REFERENCE_KNOBS,
    REFERENCE_TILE,
    compiled,
    instructions,
    main,
    per_step,
)

from tilebench.kernels.gemm import tile_fields
from tilebench.targets import get_target
from tilebench.worker import load_impl, load_workload
from tilebench.workload import Config

TARGETS = ("triton-ptx", "triton-tileir", "cutile")
SIZE = (4096, 4096, 4096)
KERNELS = {"pointer": "pointer", "descriptor": "descriptor", "masked": "pointer-masked",
           "ca": "pointer-ca"}
WIDE_TILE = (64, 256, 32)
WIDE_KNOBS = {"tileir": {"num_warps": 4, "num_stages": 3, "occupancy": 2},
              "cutile": {"num_worker_warps": None, "occupancy": 2, "latency": 1}}
COLUMNS = ("total", "HMMA", "LDGSTS", "ISETP", "SEL", "LEA", "IMAD", "LDL", "STL")


def builds(backend: str) -> list[tuple[str, tuple[int, int, int], Config]]:
    """Return the label, tile and configuration of every kernel to build on a backend."""
    plans = []
    if backend in REFERENCE_KNOBS:
        reference = {**tile_fields(*REFERENCE_TILE), **REFERENCE_KNOBS[backend]}
        plans += [(label, REFERENCE_TILE, {"style": style, **reference}) for label, style in KERNELS.items()]
    if backend == "tileir":
        wide = {**tile_fields(*WIDE_TILE), **WIDE_KNOBS["tileir"]}
        plans += [(style, WIDE_TILE, {"style": style, **wide}) for style in ("pointer", "descriptor")]
    if backend == "cutile":
        plans.append(("tile loads", WIDE_TILE, {**tile_fields(*WIDE_TILE), **WIDE_KNOBS["cutile"]}))
    return plans


def run(name: str) -> list[dict[str, Any]]:
    target = get_target(name)
    if target.impl == "triton_impl":
        import kernels

        kernels.register()
    workload = load_workload("gemm")
    impl = load_impl("gemm", target)
    loops = []
    for label, tile, config in builds(target.backend):
        _, kernel = compiled(target, workload, impl, SIZE, config)
        counts, copies = per_step(kernel.sass, tile, kernel.threads // 32)
        loops.append({"label": label, "tile": tile, "warps": kernel.threads // 32, "counts": counts,
                      "copies": copies, "code": instructions(kernel.sass)})
    return loops


def report(results: dict[str, Any]) -> None:
    for tile in (REFERENCE_TILE, WIDE_TILE):
        print(f"\nMain loop per step along K, {'x'.join(map(str, tile))} at 4,096")
        print(f"  {'kernel':<28}{'warps':>6}" + "".join(f"{column:>8}" for column in COLUMNS) + "   copies")
        for name, loops in results.items():
            for loop in loops:
                if tuple(loop["tile"]) != tile or loop["label"] == "ca":
                    continue
                cells = "".join(f"{loop['counts'].get(column, 0):>8.0f}" for column in COLUMNS)
                print(f"  {name + ' ' + loop['label']:<28}{loop['warps']:>6}{cells}   {', '.join(loop['copies'])}")
    print()
    for name, loops in results.items():
        code = {loop["label"]: loop["code"] for loop in loops if tuple(loop["tile"]) == REFERENCE_TILE}
        if "ca" in code:
            same = code["ca"] == code["pointer"]
            print(f"  {name}: asking for L1 caching gives {'the same' if same else 'different'} SASS")


if __name__ == "__main__":
    main(__file__, TARGETS, run, report)
