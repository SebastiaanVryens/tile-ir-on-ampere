"""Facts about the machine, recorded with results so they can be interpreted later."""

import platform
import subprocess
from importlib import metadata
from typing import Any

from tilebench.targets import TILEIR_SITE

GPU_STATE_FIELDS = ("clocks.sm", "clocks.mem", "temperature.gpu", "power.draw", "pstate")
PACKAGES = ("torch", "cuda-tile", "nvidia-cuda-tileiras")


def query_gpu(fields: tuple[str, ...]) -> dict[str, str]:
    proc = subprocess.run(
        ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    return dict(zip(fields, (value.strip() for value in proc.stdout.split(","))))


def gpu_state() -> dict[str, str]:
    return query_gpu(GPU_STATE_FIELDS)


def describe_environment() -> dict[str, Any]:
    packages = {name: metadata.version(name) for name in PACKAGES}
    for dist in metadata.distributions(path=[str(TILEIR_SITE)]):
        packages[dist.name] = dist.version
    return {
        "gpu": query_gpu(("name", "compute_cap", "driver_version")),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": packages,
    }
