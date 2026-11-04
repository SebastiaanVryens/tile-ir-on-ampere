"""The kernel paths under comparison, and the environment each one needs.

Triton reads ENABLE_TILE when it is imported, so one process cannot switch backends and
every target runs in its own process. This module avoids importing torch or triton so
the parent process can use it safely.
"""

import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

TILEIR_SITE = Path(sys.prefix) / "opt" / "triton_tileir"

# Inherited settings that would silently change which backend a target exercises.
SCRUBBED_ENV = (
    "PYTHONPATH",
    "ENABLE_TILE",
    "TRITON_DEFAULT_BACKEND",
    "TRITON_TILEIR_RUNTIME_FALLBACK",
)


@dataclass(frozen=True)
class Target:
    name: str
    impl: str
    backend: str
    env: Mapping[str, str] = field(default_factory=dict)


# Both Triton targets load the Tile IR wheel, so they differ only in the backend.
TARGETS = (
    Target("torch", "torch_impl", "torch"),
    Target("cutile", "cutile_impl", "cutile"),
    Target("triton-ptx", "triton_impl", "cuda", {"PYTHONPATH": str(TILEIR_SITE)}),
    Target(
        "triton-tileir",
        "triton_impl",
        "tileir",
        {"PYTHONPATH": str(TILEIR_SITE), "ENABLE_TILE": "1"},
    ),
)


def get_target(name: str) -> Target:
    return next(target for target in TARGETS if target.name == name)


def worker_env(target: Target, base: Mapping[str, str] | None = None) -> dict[str, str]:
    source = os.environ if base is None else base
    env = {key: value for key, value in source.items() if key not in SCRUBBED_ENV}
    env.update(target.env)
    return env
