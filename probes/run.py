"""Gate check: can every kernel path compile and run correctly on this GPU?

Run `python probes/run.py` from the activated venv. Exits non-zero if a required case fails.

Each (target, case) pair gets a fresh subprocess. Triton reads ENABLE_TILE when it is
imported, so one process cannot switch backends, and a CUDA fault would otherwise poison
the context for every later case.
"""

import json
import os
import subprocess
import sys
import sysconfig
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROBE_DIR = Path(__file__).resolve().parent
TILEIR_SITE = Path(sys.prefix) / "opt" / "triton_tileir"
CACHE_ROOT = Path.home() / ".cache" / "cutile" / "probe"
TIMEOUT_S = 300
STDERR_TAIL_LINES = 12

# Inherited settings that would silently change which backend a case exercises.
SCRUBBED_ENV = (
    "PYTHONPATH",
    "ENABLE_TILE",
    "TRITON_DEFAULT_BACKEND",
    "TRITON_TILEIR_RUNTIME_FALLBACK",
)

TRITON_CASES = ("add", "matmul_pointers", "matmul_block_ptr", "matmul_descriptor")
CUTILE_CASES = ("add", "matmul")
# Block pointers are deprecated upstream, so a failure there is reported but not fatal.
OPTIONAL_CASES = frozenset({"matmul_block_ptr"})


@dataclass(frozen=True)
class Target:
    name: str
    script: str
    cases: tuple[str, ...]
    backend: str
    env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Outcome:
    target: str
    case: str
    passed: bool
    report: dict[str, Any] = field(default_factory=dict)
    detail: str = ""


# triton-ptx and triton-tileir share one wheel, so they differ only in the backend.
# triton-upstream is the separate, newer release that torch installs.
TARGETS = (
    Target("triton-upstream", "triton_cases.py", TRITON_CASES, "cuda"),
    Target(
        "triton-ptx",
        "triton_cases.py",
        TRITON_CASES,
        "cuda",
        {"PYTHONPATH": str(TILEIR_SITE)},
    ),
    Target(
        "triton-tileir",
        "triton_cases.py",
        TRITON_CASES,
        "tileir",
        {"PYTHONPATH": str(TILEIR_SITE), "ENABLE_TILE": "1"},
    ),
    Target("cutile", "cutile_cases.py", CUTILE_CASES, "cutile"),
)


def run_case(target: Target, case: str) -> Outcome:
    env = {key: value for key, value in os.environ.items() if key not in SCRUBBED_ENV}
    env.update(target.env)
    env["TRITON_CACHE_DIR"] = str(CACHE_ROOT / target.name)
    env["TRITON_ALWAYS_COMPILE"] = "1"

    command = [sys.executable, str(PROBE_DIR / target.script), case]
    try:
        proc = subprocess.run(
            command, env=env, capture_output=True, text=True, timeout=TIMEOUT_S
        )
    except subprocess.TimeoutExpired:
        return Outcome(target.name, case, False, detail=f"timed out after {TIMEOUT_S}s")

    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-STDERR_TAIL_LINES:])
        return Outcome(target.name, case, False, detail=tail)

    report = json.loads(proc.stdout.strip().splitlines()[-1])
    if report["backend"] != target.backend:
        detail = f"expected backend {target.backend}, got {report['backend']}"
        return Outcome(target.name, case, False, report, detail)
    return Outcome(target.name, case, True, report)


def tileiras_version(binary: Path) -> str:
    if not binary.exists():
        return f"missing ({binary})"
    proc = subprocess.run([str(binary), "--version"], capture_output=True, text=True)
    lines = proc.stdout.splitlines()
    return next((line.strip() for line in lines if "release" in line), "unknown")


def print_toolchain() -> None:
    query = "--query-gpu=name,compute_cap,driver_version"
    gpu = subprocess.run(
        ["nvidia-smi", query, "--format=csv,noheader"], capture_output=True, text=True
    )
    pip_tileiras = Path(sysconfig.get_paths()["purelib"]) / "nvidia/cu13/bin/tileiras"
    wheel_tileiras = TILEIR_SITE / "triton/backends/nvidia/tileir_cuda/bin/tileiras"
    print(f"GPU:              {gpu.stdout.strip()}")
    print(f"tileiras (pip):   {tileiras_version(pip_tileiras)}")
    print(f"tileiras (wheel): {tileiras_version(wheel_tileiras)}")
    print()


def format_row(outcome: Outcome) -> str:
    report = outcome.report
    status = "PASS" if outcome.passed else "FAIL"
    if outcome.case in OPTIONAL_CASES and not outcome.passed:
        status = "fail*"
    loads = "-"
    if report.get("backend") == "tileir":
        loads = f"{report['ptr_loads']} / {report['view_loads']}"
    return (
        f"{outcome.target:<16}{outcome.case:<20}{status:<7}"
        f"{report.get('backend', '-'):<9}{report.get('triton', '-'):<9}{loads}"
    )


def main() -> int:
    if not TILEIR_SITE.is_dir():
        print(f"error: {TILEIR_SITE} not found; run scripts/setup_wsl.sh first", file=sys.stderr)
        return 2

    print_toolchain()
    print(f"{'target':<16}{'case':<20}{'result':<7}{'backend':<9}{'triton':<9}ptr / view loads")

    outcomes = []
    for target in TARGETS:
        for case in target.cases:
            outcome = run_case(target, case)
            outcomes.append(outcome)
            print(format_row(outcome), flush=True)

    failures = [outcome for outcome in outcomes if not outcome.passed]
    for outcome in failures:
        print(f"\n--- {outcome.target} / {outcome.case} ---\n{outcome.detail}")

    required = [outcome for outcome in failures if outcome.case not in OPTIONAL_CASES]
    print(f"\nGATE: {'FAIL' if required else 'PASS'} ({len(required)} required failures)")
    if len(failures) > len(required):
        print("fail* = optional case, reported but not counted")
    return 1 if required else 0


if __name__ == "__main__":
    sys.exit(main())
