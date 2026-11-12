"""The pointer kernel written the plain way, against the repository's, at every setting at 4,096.

Usage, from the activated venv: python investigations/tensor_of_pointer/masked.py

The plain kernel masks each load along both of its dimensions. The repository's kernel
wraps rows and columns, as Triton's tutorial does, so only K needs a mask. Every setting
the benchmark tunes is built for both kernels, checked, and timed with 20 launches. The
best 3 settings of each kernel and the reference setting are then timed again in 3 rounds
of 100 launches, rotating the order each round, as the benchmark's rounds do.
Takes about 9 minutes, longer the first time while every kernel compiles.
"""

import statistics
from collections import defaultdict
from typing import Any

from common import (
    MAX_START_C,
    REFERENCE_KNOBS,
    REFERENCE_TILE,
    compiled,
    conditions,
    main,
    setting,
    tflops,
    tile_of,
)

from tilebench.__main__ import TUNE_REPEATS
from tilebench.kernels.gemm import tile_fields
from tilebench.targets import get_target
from tilebench.worker import bench, load_impl, load_workload, start_limit

TARGETS = ("triton-ptx", "triton-tileir")
SIZE = (4096, 4096, 4096)
STYLES = ("pointer", "pointer-masked")
TOP = 3
ROUNDS = 3
ROUND_REPEATS = 100


def run(name: str) -> dict[str, Any]:
    import kernels

    kernels.register()
    target = get_target(name)
    workload = load_workload("gemm")
    impl = load_impl("gemm", target)
    limit = start_limit(workload, MAX_START_C)
    configs = [{**config, "style": style} for config in impl.configs(target.backend, SIZE)
               if config["style"] == "pointer" for style in STYLES]
    builds = []
    for config in configs:
        _, kernel = compiled(target, workload, impl, SIZE, config)
        builds.append({"config": config, "registers": kernel.registers, "stack_bytes": kernel.stack_bytes})
    sweep = bench(workload, impl, target, [(SIZE, config) for config in configs], TUNE_REPEATS, limit)

    reference = {**tile_fields(*REFERENCE_TILE), **REFERENCE_KNOBS[target.backend]}
    chosen = [{**reference, "style": style} for style in STYLES]
    for style in STYLES:
        ranked = sorted((r for r in sweep if r["config"]["style"] == style), key=rate, reverse=True)
        chosen += [r["config"] for r in ranked[:TOP] if r["config"] not in chosen]
    rounds = []
    for index in range(ROUNDS):
        shift = index % len(chosen)
        order = chosen[shift:] + chosen[:shift]
        rounds += bench(workload, impl, target, [(SIZE, config) for config in order], ROUND_REPEATS, limit)
    return {"builds": builds, "sweep": sweep, "rounds": rounds, "reference": reference}


def rate(record: dict[str, Any]) -> float:
    return tflops(record["size"], record["gpu_ms"])


def report(results: dict[str, Any]) -> None:
    for name, result in results.items():
        print(f"\n===== {name}")
        print(f"Sweep: {conditions(result['sweep'])}")
        print(f"Rounds: {conditions(result['rounds'])}")

        print(f"Every setting, {TUNE_REPEATS} launches each")
        reference = result["reference"]
        for style in STYLES:
            mine = [r for r in result["sweep"] if r["config"]["style"] == style]
            best = max(mine, key=rate)
            at_reference = next(r for r in mine if r["config"] == {**reference, "style": style})
            spilling = [b for b in result["builds"] if b["config"]["style"] == style and b["stack_bytes"]]
            registers = sorted({b["registers"] for b in spilling})
            print(f"  {style:<16}best {rate(best):5.1f} TFLOPS at {setting(best['config']):<44}"
                  f"reference setting {rate(at_reference):5.1f}   "
                  f"{len(spilling)} of {len(mine)} builds spill, at {registers} registers")
        per_tile: dict[tuple[int, int, int], dict[str, float]] = defaultdict(dict)
        for record in result["sweep"]:
            tile, style = tile_of(record["config"]), record["config"]["style"]
            per_tile[tile][style] = max(per_tile[tile].get(style, 0.0), rate(record))
        cells = ", ".join(f"{'x'.join(map(str, tile))} {(best['pointer'] / best['pointer-masked'] - 1) * 100:+.0f}%"
                          for tile, best in sorted(per_tile.items(), reverse=True))
        print(f"  Extra time of the best plain kernel per tile: {cells}")

        print(f"Rounds, {ROUNDS} x {ROUND_REPEATS} launches")
        pooled: dict[str, list[float]] = defaultdict(list)
        by_round: dict[str, list[float]] = defaultdict(list)
        configs = {}
        for record in result["rounds"]:
            key = f"{record['config']['style']} {setting(record['config'])}"
            pooled[key] += record["gpu_ms"]
            by_round[key].append(statistics.median(record["gpu_ms"]))
            configs[key] = record["config"]
        medians = {key: statistics.median(samples) for key, samples in pooled.items()}
        best_pointer = min(ms for key, ms in medians.items() if configs[key]["style"] == "pointer")
        for key, ms in sorted(medians.items(), key=lambda item: item[1]):
            spread = (max(by_round[key]) - min(by_round[key])) / ms * 100
            mark = "  (reference setting)" if {**configs[key], "style": "x"} == {**reference, "style": "x"} else ""
            print(f"  {key:<62}{tflops(SIZE, pooled[key]):5.1f} TFLOPS {(ms / best_pointer - 1) * 100:+6.1f}% "
                  f"time against the best pointer setting, rounds within {spread:.1f}%{mark}")


if __name__ == "__main__":
    main(__file__, TARGETS, run, report)
