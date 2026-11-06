"""Run one target on one workload in this process.

Launched with the environment from targets.worker_env, because the backend is fixed when
Triton is imported. --check verifies correctness only, and --inspect-size saves what the
compiler produced for one size. Otherwise this benchmarks every configuration, or only the
ones listed in --plan. Results go to --out.
"""

import argparse
import importlib
import json
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import torch

from tilebench.artifacts import config_slug, inspect_build
from tilebench.system import MAX_START_TEMP_C, gpu_state, wait_until_cool
from tilebench.targets import Target, get_target
from tilebench.timing import REPEATS, Timer, burst_length
from tilebench.workload import Config, Impl, Prepared, Size, Workload


def load_workload(name: str) -> Workload:
    return importlib.import_module(f"tilebench.kernels.{name}").WORKLOAD


def load_impl(workload: str, target: Target) -> Impl:
    module = importlib.import_module(f"tilebench.kernels.{workload}.{target.impl}")
    return cast(Impl, module)


def require_backend(target: Target, launch_result: Any) -> None:
    """Fail if a Triton kernel compiled on anything but the backend under test."""
    if target.impl != "triton_impl":
        return
    backend = launch_result.metadata.target.backend
    if backend != target.backend:
        raise RuntimeError(f"{target.name} compiled on backend {backend}")


def verified(
    workload: Workload, impl: Impl, target: Target, size: Size, config: Config
) -> Prepared:
    inputs = workload.make_inputs(size)
    prepared = impl.prepare(inputs, config)
    require_backend(target, prepared.launch())
    torch.testing.assert_close(
        prepared.output,
        workload.reference(*inputs),
        rtol=workload.rtol,
        atol=workload.atol,
    )
    return prepared


def check(workload: Workload, impl: Impl, target: Target) -> None:
    """Cover the awkward sizes. bench verifies the benchmark sizes before timing them."""
    for size in workload.check_sizes:
        for config in impl.configs(target.backend, size):
            verified(workload, impl, target, size, config)


def full_sweep(workload: Workload, impl: Impl, target: Target) -> list[tuple[Size, Config]]:
    return [
        (size, config)
        for size in workload.sizes
        for config in impl.configs(target.backend, size)
    ]


def by_size(plan: Iterable[tuple[Sequence[int], Config]]) -> dict[Size, list[Config]]:
    groups: dict[Size, list[Config]] = defaultdict(list)
    for dims, config in plan:
        groups[tuple(dims)].append(config)
    return groups


def bursts(configs: Sequence[Config], length: int) -> Iterator[Sequence[Config]]:
    for start in range(0, len(configs), length):
        yield configs[start : start + length]


def bench(
    workload: Workload,
    impl: Impl,
    target: Target,
    plan: Iterable[tuple[Sequence[int], Config]],
    repeats: int,
    max_temp: int,
) -> list[dict[str, Any]]:
    timer = Timer()
    records = []
    for size, configs in by_size(plan).items():
        # Compiling and checking leave the GPU idle, so all of it happens before timing.
        for config in configs:
            verified(workload, impl, target, size, config)

        inputs = workload.make_inputs(size)
        for burst in bursts(configs, burst_length(repeats)):
            start_temp = wait_until_cool(max_temp)
            timer.warm_up(impl.prepare(inputs, burst[0]).launch)
            timings = [timer.gpu_ms(impl.prepare(inputs, c).launch, repeats) for c in burst]
            # Only read here, because querying the GPU inside a burst would slow its clocks.
            state = gpu_state()
            records += [
                {
                    "workload": workload.name,
                    "target": target.name,
                    "size": list(size),
                    "config": config,
                    "gpu_ms": gpu_ms,
                    "start_temp": start_temp,
                    "gpu_state": state,
                    "work": workload.work(size),
                    "rate_unit": workload.rate_unit,
                    "rate_scale": workload.rate_scale,
                }
                for config, gpu_ms in zip(burst, timings)
            ]
    return records


def parse_size(text: str) -> Size:
    return tuple(int(dim) for dim in text.split("x"))


def selected(config: Config, style: str, where: Mapping[str, str]) -> bool:
    """Match style text and exact fields, ignoring fields this target does not have."""
    fields = all(str(config[key]) == value for key, value in where.items() if key in config)
    return fields and style in config.get("style", "")


def inspect_kernels(
    workload: Workload,
    impl: Impl,
    target: Target,
    size: Size,
    style: str,
    where: Mapping[str, str],
    artifact_dir: Path,
) -> list[dict[str, Any]]:
    records = []
    for config in impl.configs(target.backend, size):
        if selected(config, style, where):
            prepared = verified(workload, impl, target, size, config)
            directory = artifact_dir / target.name / config_slug(config)
            summary = inspect_build(target, prepared, directory)
            records.append({"target": target.name, "config": config} | summary)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--inspect-size", type=parse_size)
    parser.add_argument("--style", default="")
    parser.add_argument("--where", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--plan", type=json.loads, help="JSON list of [size, config] pairs")
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--max-temp", type=int, default=MAX_START_TEMP_C)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    target = get_target(args.target)
    workload = load_workload(args.workload)
    impl = load_impl(args.workload, target)
    if args.check:
        check(workload, impl, target)
    elif args.inspect_size:
        where = dict(field.split("=", 1) for field in args.where)
        records = inspect_kernels(
            workload, impl, target, args.inspect_size, args.style, where, args.artifacts
        )
        args.out.write_text(json.dumps(records))
    else:
        plan = full_sweep(workload, impl, target) if args.plan is None else args.plan
        records = bench(workload, impl, target, plan, args.repeats, args.max_temp)
        args.out.write_text(json.dumps(records))


if __name__ == "__main__":
    main()
