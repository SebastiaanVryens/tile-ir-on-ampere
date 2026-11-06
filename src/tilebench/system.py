"""Facts about the machine, recorded with results so they can be interpreted later."""

import platform
import subprocess
import time
from importlib import metadata
from typing import Any

from tilebench.targets import TILEIR_SITE

GPU_STATE_FIELDS = ("clocks.sm", "clocks.mem", "temperature.gpu", "power.draw", "pstate")
PACKAGES = ("torch", "cuda-tile", "nvidia-cuda-tileiras")

# This GPU held a median of 1950 MHz up to 82 C, then fell to 1575 MHz at 86 C. A burst adds
# up to 13 degrees when measuring, so starting at 70 C keeps it on the flat part.
MAX_START_TEMP_C = 70
COOL_POLL_S = 1.0
COOL_TIMEOUT_S = 120.0


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


def gpu_temperature() -> int:
    return int(query_gpu(("temperature.gpu",))["temperature.gpu"])


def wait_until_cool(limit_c: int, timeout_s: float = COOL_TIMEOUT_S) -> int:
    """Wait for the GPU to reach limit_c or below, and return the temperature it reached.

    Gives up after timeout_s rather than failing a long run. The caller records the result,
    so a measurement that started hot stays visible in the report.
    """
    deadline = time.monotonic() + timeout_s
    temperature = gpu_temperature()
    while temperature > limit_c and time.monotonic() < deadline:
        time.sleep(COOL_POLL_S)
        temperature = gpu_temperature()
    return temperature


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
