"""Run a script once per target, and read loads and main loops out of SASS.

Each target needs its own process: the Triton targets load the Tile IR wheel, and only
one of them enables the Tile IR backend. A script calls `main` with a function that does
the work for one target and a function that prints the results of all of them. Without
arguments, `main` starts the script again for every target, in that target's environment,
and passes what each run returned to the report.
"""

import argparse
import json
import re
import statistics
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from tilebench.artifacts import Artifacts, collect
from tilebench.cubin import Kernel, count_opcodes, read_kernel
from tilebench.targets import Target, get_target, worker_env
from tilebench.worker import verified
from tilebench.workload import Config, Impl, Size, Workload

# The benchmark's GEMM bursts start at 60 C; worker.start_limit keeps the cooler of the two.
MAX_START_C = 60

# The setting where the repository's pointer kernel won at 4,096, used wherever a script
# compares kernels at one setting. occupancy only exists on Tile IR.
REFERENCE_TILE = (128, 256, 64)
REFERENCE_KNOBS = {"cuda": {"num_warps": 8, "num_stages": 3},
                   "tileir": {"num_warps": 8, "num_stages": 3, "occupancy": 1}}

LOAD = re.compile(r"\b(LDGSTS|LDG|STG)((?:\.[A-Z0-9_]+)*)")
LINE = re.compile(r"^\s*/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;")
LABEL = re.compile(r"^\s*(\.L_x_\d+):")
BRANCH = re.compile(r"BRA\s+`?\(?(\.L_x_\d+|0x[0-9a-f]+)")
OPCODE = re.compile(r"^(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9]*)((?:\.[A-Z0-9_]+)*)")


def main(script: str, targets: Sequence[str], run: Callable[[str], Any],
         report: Callable[[dict[str, Any]], None]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep", type=Path, metavar="DIR",
                        help="also save what each target's run returned, as DIR/TARGET.json")
    parser.add_argument("--target", help=argparse.SUPPRESS)
    parser.add_argument("--out", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.target:
        args.out.write_text(json.dumps(run(args.target)))
        return
    results = {}
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_dir = args.keep or Path(tmp_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in targets:
            out = out_dir / f"{name}.json"
            command = [sys.executable, script, "--target", name, "--out", str(out)]
            subprocess.run(command, env=worker_env(get_target(name)), check=True)
            results[name] = json.loads(out.read_text())
    report(results)


def conditions(records: Sequence[Mapping[str, Any]]) -> str:
    """Describe the GPU around the bursts, read by the harness right after each one ended.

    The readings come before the GPU has settled, so they show the end of the load. In the
    benchmark's GEMM run they had a median of 1,845 to 1,909 MHz and 110 to 127 W, against
    1,702 MHz and 30 W at idle. Lower clocks and power at normal temperatures point to a
    power limit outside the GPU, such as a battery that is charging.
    """
    def values(field: str) -> list[float]:
        return sorted(float(r["gpu_state"][field]) for r in records)

    clocks, power, temps = values("clocks.sm"), values("power.draw"), values("temperature.gpu")
    starts = sorted({r["start_temp"] for r in records})
    return (f"bursts started at {starts} C and ended at {temps[0]:.0f} to {temps[-1]:.0f} C; "
            f"after each burst the SM clock was {clocks[0]:.0f} to {clocks[-1]:.0f} MHz "
            f"(median {statistics.median(clocks):.0f}) and the power {statistics.median(power):.0f} W")


def compiled(target: Target, workload: Workload, impl: Impl, size: Size,
             config: Config) -> tuple[Artifacts, Kernel]:
    """Build a configuration, check its result against the reference, and read its cubin."""
    artifacts = collect(target, verified(workload, impl, target, size, config))
    return artifacts, read_kernel(artifacts.cubin)


def tile_of(config: Mapping[str, Any]) -> tuple[int, int, int]:
    return config["tile_m"], config["tile_n"], config["tile_k"]


def setting(config: Mapping[str, Any]) -> str:
    """Name a configuration by its tile and knobs, leaving out the style."""
    knobs = " ".join(f"{key}={value}" for key, value in config.items()
                     if key not in ("style", "tile_m", "tile_n", "tile_k"))
    return f"{'x'.join(map(str, tile_of(config)))} {knobs}" if "tile_m" in config else knobs


def tflops(size: Sequence[int], gpu_ms: Sequence[float]) -> float:
    m, n, k = size
    return 2 * m * n * k / (statistics.median(gpu_ms) * 1e-3) / 1e12


def load_forms(sass: str) -> dict[str, int]:
    """Count global loads, stores and asynchronous copies by their full form, such as LDG.E.U16."""
    return dict(Counter(name + modifiers for name, modifiers in LOAD.findall(sass)))


def describe_loads(forms: Mapping[str, int]) -> str:
    return ", ".join(f"{count} {form}" for form, count in sorted(forms.items(), key=lambda x: -x[1]))


def instruction_count(sass: str) -> int:
    """Count instructions the way tilebench.artifacts.summarize does, without the padding NOPs."""
    opcodes = count_opcodes(sass)
    return opcodes.total() - opcodes["NOP"]


def instructions(sass: str) -> list[str]:
    """Return the instruction text alone, so two builds can be compared line by line."""
    return [match.group(2) for line in sass.splitlines() if (match := LINE.match(line))]


def main_loop(sass: str) -> list[str]:
    """Return the body of the backward branch that holds the most HMMA instructions.

    Branches name their target either by label or by absolute address, depending on the
    compiler, so both forms are resolved.
    """
    listing: list[tuple[int, str]] = []
    labels: dict[str, int] = {}
    pending: list[str] = []
    for line in sass.splitlines():
        if label := LABEL.match(line):
            pending.append(label.group(1))
        elif match := LINE.match(line):
            address = int(match.group(1), 16)
            labels.update((name, address) for name in pending)
            pending = []
            listing.append((address, match.group(2)))
    best: list[str] = []
    for index, (address, text) in enumerate(listing):
        if not (branch := BRANCH.search(text)):
            continue
        target = branch.group(1)
        start = int(target, 16) if target.startswith("0x") else labels.get(target, address + 1)
        if start > address:
            continue
        body = [text for at, text in listing[: index + 1] if at >= start]
        if count_hmma(body) > count_hmma(best):
            best = body
    return best


def count_hmma(body: Sequence[str]) -> int:
    return sum(opcode(text)[0] == "HMMA" for text in body)


def opcode(text: str) -> tuple[str, str]:
    match = OPCODE.match(text)
    return (match.group(1), match.group(2)) if match else ("", "")


def per_step(sass: str, tile: tuple[int, int, int], warps: int) -> tuple[dict[str, float], list[str]]:
    """Return the main loop's opcodes per step along K, with "total", and its copy forms.

    Each warp issues its share of the m16n8k16 tensor core instructions for one step, so
    dividing the loop's HMMA count by that share gives the steps per iteration, which keeps
    unrolled loops comparable. Opcodes are counted without modifiers, so spills count as
    LDL and STL whatever their width; the forms of the asynchronous copies are listed apart.
    """
    tile_m, tile_n, tile_k = tile
    hmma_per_step = tile_m * tile_n * tile_k // (16 * 8 * 16) // warps
    body = main_loop(sass)
    steps = count_hmma(body) / hmma_per_step
    counts = Counter(opcode(text)[0] for text in body)
    counts["total"] = len(body)
    copies = sorted({"".join(opcode(text)) for text in body if opcode(text)[0] == "LDGSTS"})
    return {name: count / steps for name, count in counts.items()}, copies
