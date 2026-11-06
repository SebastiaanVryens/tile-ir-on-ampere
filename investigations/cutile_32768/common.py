"""The case under study, timed the way the benchmark times it."""

import os
import statistics
from collections import defaultdict
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, NamedTuple

import torch

from tilebench import artifacts, cubin
from tilebench.cubin import Kernel, count_opcodes
from tilebench.kernels.layernorm import bytes_moved
from tilebench.schedule import read_schedule
from tilebench.system import MAX_START_TEMP_C, wait_until_cool
from tilebench.timing import REPEATS, Timer, burst_length

SIZE = (512, 32768)
ROWS, COLS = SIZE
HINTS = tuple(range(1, 11))
ROUNDS = 3

# A disassembler cannot read what a newer release wrote, so a newer toolchain brings its own.
if tools := os.environ.get("TOOL_BIN"):
    cubin.CUOBJDUMP = Path(tools) / "cuobjdump"
    artifacts.TILEIRDISASM = Path(tools) / "tileirdisasm"


class Timing(NamedTuple):
    gbs: float
    iqr_us: float


def verified(launch: Callable[[], Any], output: torch.Tensor, expected: torch.Tensor) -> Any:
    """Launch once and check the result. NaN first, so a kernel that writes nothing fails."""
    output.fill_(float("nan"))
    result = launch()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, expected, rtol=1e-2, atol=2e-2)
    return result


def measure(launches: Mapping[str, Callable[[], Any]]) -> dict[str, Timing]:
    """Time every kernel in bursts, rotating the order each round."""
    timer = Timer()
    labels = list(launches)
    length = burst_length(REPEATS)
    samples: dict[str, list[float]] = defaultdict(list)
    for round_index in range(ROUNDS):
        shift = round_index % len(labels)
        rotated = labels[shift:] + labels[:shift]
        for start in range(0, len(rotated), length):
            burst = rotated[start : start + length]
            wait_until_cool(MAX_START_TEMP_C)
            timer.warm_up(launches[burst[0]])
            for label in burst:
                samples[label] += timer.gpu_ms(launches[label], REPEATS)
    return {label: summarize(values) for label, values in samples.items()}


def summarize(values: list[float]) -> Timing:
    median_ms = statistics.median(values)
    quartiles = statistics.quantiles(values, n=4)
    return Timing(bytes_moved(SIZE) / (median_ms * 1e-3) / 1e9, (quartiles[2] - quartiles[0]) * 1e3)


def print_by_hint(variants: tuple[str, ...], timings: Mapping[str, Timing]) -> None:
    print(f"{'GB/s by hint':<24}" + "".join(f"{hint:>6}" for hint in HINTS) + f"{'best':>8}")
    for variant in variants:
        values = [timings[f"{variant} h{hint}"].gbs for hint in HINTS]
        cells = "".join(f"{value:>6.0f}" for value in values)
        print(f"{variant:<24}{cells}{max(values):>8.1f}")


def print_kernels(kernels: Mapping[str, Kernel], timings: Mapping[str, Timing]) -> None:
    print(f"{'kernel':<28}{'regs':>6}{'BSSY':>6}{'FCHK':>6}{'row':>5}{'tail':>6}{'spills':>8}{'GB/s':>8}")
    for label, kernel in kernels.items():
        schedule = read_schedule(kernel.sass)
        print(
            f"{label:<28}{kernel.registers:>6}{schedule.branch_regions:>6}"
            f"{count_opcodes(kernel.sass)['FCHK']:>6}{schedule.first_peak:>5}"
            f"{schedule.last_peak:>6}{schedule.spills:>8}{timings[label].gbs:>8.1f}"
        )
