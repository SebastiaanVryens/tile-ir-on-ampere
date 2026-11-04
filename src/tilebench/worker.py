"""Run one target on one workload in this process.

Launched with the environment from targets.worker_env, because the backend is fixed when
Triton is imported. --check verifies correctness only; otherwise results go to --out.
"""

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, cast

import torch

from tilebench.system import gpu_state
from tilebench.targets import Target, get_target
from tilebench.timing import measure_gpu_ms
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
    for size in workload.check_sizes + workload.sizes:
        for config in impl.configs(target.backend):
            verified(workload, impl, target, size, config)


def bench(workload: Workload, impl: Impl, target: Target) -> list[dict[str, Any]]:
    records = []
    for size in workload.sizes:
        for config in impl.configs(target.backend):
            prepared = verified(workload, impl, target, size, config)
            gpu_ms = measure_gpu_ms(prepared.launch)
            state = gpu_state()
            records.append(
                {
                    "workload": workload.name,
                    "target": target.name,
                    "size": list(size),
                    "config": config,
                    "gpu_ms": gpu_ms,
                    "gpu_state": state,
                    "work": workload.work(size),
                    "rate_unit": workload.rate_unit,
                    "rate_scale": workload.rate_scale,
                }
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    target = get_target(args.target)
    workload = load_workload(args.workload)
    impl = load_impl(args.workload, target)
    if args.check:
        check(workload, impl, target)
    else:
        args.out.write_text(json.dumps(bench(workload, impl, target)))


if __name__ == "__main__":
    main()
