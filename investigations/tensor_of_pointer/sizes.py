"""GEMM at 4,096 and 4,088 on every path, and the Triton kernels told what the sizes divide by.

Usage, from the activated venv: python investigations/tensor_of_pointer/sizes.py

4,088 is 8 x 511, so every row of the matrices starts 16-byte aligned, but it is not
divisible by 16, the only divisibility the compilers record. Every setting of the tiles
128x256x64 and 128x128x32 is checked against the reference and timed like the benchmark's
tuning, with 20 launches in bursts from 60 C, and the table shows the best setting of each
path. The harness times one size after the other. The variants:
  no spec.   Triton's size specialization turned off, at 4,096
  assume     Triton: tl.assume(K % 8 == 0), and the same for M and N
             cuTile: every array sliced to its own length, assumed divisible by 8
  // 8 * 8   Triton: K // 8 * 8 passed in place of K, and the same for M and N
Takes about 11 minutes, longer the first time while every kernel compiles.
"""

import json
from collections import defaultdict
from typing import Any, cast

from common import (
    MAX_START_C,
    compiled,
    conditions,
    describe_loads,
    load_forms,
    main,
    setting,
    tflops,
    tile_of,
)

from tilebench.__main__ import TUNE_REPEATS
from tilebench.targets import get_target
from tilebench.worker import bench, load_impl, load_workload, start_limit
from tilebench.workload import Config, Impl

TARGETS = ("torch", "cutile", "triton-ptx", "triton-tileir")
TILES = {(128, 256, 64), (128, 128, 32)}
# The columns of the table, as size and variant, and the variants each path has.
COLUMNS = ((4096, ""), (4096, "nospec"), (4088, ""), (4088, "assume"), (4088, "round"))
VARIANTS = {"torch_impl": ("",), "cutile_impl": ("", "assume"),
            "triton_impl": ("", "nospec", "assume", "round")}
HEADINGS = {"": "", "nospec": "no spec.", "assume": "assume", "round": "// 8 * 8"}


def run(name: str) -> dict[str, Any]:
    target = get_target(name)
    workload = load_workload("gemm")
    impl = load_impl("gemm", target)
    if target.impl == "triton_impl":
        import kernels

        kernels.register()
    if target.impl == "cutile_impl":
        import cutile_kernels

        impl = cast(Impl, cutile_kernels)
    plan = []
    for n, variant in COLUMNS:
        if variant not in VARIANTS[target.impl]:
            continue
        size = (n, n, n)
        for config in impl.configs(target.backend, size):
            if config and tile_of(config) not in TILES:
                continue
            # register() adds the variants to the styles the benchmark lists; keep its own two.
            if config.get("style", "pointer") not in ("pointer", "descriptor"):
                continue
            if variant:
                style = f"{config['style']}-{variant}" if "style" in config else variant
                config = {**config, "style": style}
            plan.append((size, config))
    builds = []
    if name != "torch":
        for size, config in plan:
            _, kernel = compiled(target, workload, impl, size, config)
            builds.append({"size": size[0], "config": config, "registers": kernel.registers,
                           "stack_bytes": kernel.stack_bytes, "loads": load_forms(kernel.sass)})
    records = bench(workload, impl, target, plan, TUNE_REPEATS, start_limit(workload, MAX_START_C))
    return {"records": records, "builds": builds}


def report(results: dict[str, Any]) -> None:
    best: dict[tuple[str, int, str], dict[str, Any]] = {}
    builds = {}
    for name, result in results.items():
        for build in result["builds"]:
            builds[(name, build["size"], json.dumps(build["config"], sort_keys=True))] = build
        for record in result["records"]:
            base, variant = split_style(name, record["config"])
            key = (f"{name} {base}".strip(), record["size"][0], variant)
            if key not in best or tflops(record["size"], record["gpu_ms"]) > rate(best[key]):
                best[key] = record
    print()
    for name, result in results.items():
        print(f"{name}: {conditions(result['records'])}")

    rows = list(dict.fromkeys(row for row, _, _ in best))
    print("\nTFLOPS, best over every setting of 128x256x64 and 128x128x32, 20 launches per setting")
    print(f"{'path':<26}{'4,096':>8}{'no spec.':>10}{'4,088':>8}{'change':>8}{'assume':>8}{'// 8 * 8':>10}")
    for row in rows:
        cells = []
        for n, variant in COLUMNS:
            width = 10 if variant in ("nospec", "round") else 8
            record = best.get((row, n, variant))
            cells.append(f"{rate(record):>{width}.1f}" if record else f"{'-':>{width}}")
            if (n, variant) == (4088, ""):
                change = rate(record) / rate(best[(row, 4096, "")]) - 1
                cells.append(f"{change * 100:>+7.0f}%")
        print(f"{row:<26}{''.join(cells)}")

    print("\nThe best setting of each cell, and what it compiled to")
    for row in rows:
        if row == "torch":
            continue
        print(f"  {row}")
        for n, variant in COLUMNS:
            if not (record := best.get((row, n, variant))):
                continue
            build = builds[(record["target"], n, json.dumps(record["config"], sort_keys=True))]
            column = f"{n:,}" + (f" {HEADINGS[variant]}" if variant else "")
            print(f"    {column:<16}{setting(record['config']):<58}{build['registers']} regs, "
                  f"{build['stack_bytes']} B stack: {describe_loads(build['loads'])}")

    print("\nSettings that spill, out of all built")
    spills: dict[tuple[str, int, str], list[int]] = defaultdict(lambda: [0, 0])
    for (name, n, config), build in builds.items():
        variant = split_style(name, json.loads(config))[1]
        spills[(name, n, variant)][0] += build["stack_bytes"] > 0
        spills[(name, n, variant)][1] += 1
    for (name, n, variant), (spilling, total) in spills.items():
        print(f"  {name:<16}{n:>6,} {HEADINGS[variant]:<10}{spilling:>4} of {total}")


def split_style(target: str, config: Config) -> tuple[str, str]:
    """Return the load style and the variant: Triton styles read 'pointer-round', cuTile's 'assume'."""
    style = config.get("style", "")
    if target.startswith("triton"):
        base, _, variant = style.partition("-")
        return base, variant
    return "", style


def rate(record: dict[str, Any]) -> float:
    return tflops(record["size"], record["gpu_ms"])


if __name__ == "__main__":
    main(__file__, TARGETS, run, report)
