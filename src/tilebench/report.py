"""Turn raw benchmark records into a comparison table."""

import statistics
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from tilebench.stats import Summary, summarize

Record = dict[str, Any]

TIMER_NOTE = (
    "CUDA event timing moves in steps of about 1 us here, "
    "so differences of a few us at small sizes are not meaningful."
)


@dataclass(frozen=True)
class Row:
    workload: str
    size: tuple[int, ...]
    variant: str
    config: str
    gpu_ms: Summary
    rate: float
    rate_unit: str
    drift_pct: float


def variant_name(record: Record) -> str:
    style = record["config"].get("style")
    return f"{record['target']}/{style}" if style else record["target"]


def config_label(config: dict[str, Any]) -> str:
    parts = [f"{key}={value}" for key, value in config.items() if key != "style"]
    return " ".join(parts) or "-"


def summarize_rounds(records: Sequence[Record]) -> Row:
    """Pool one configuration across rounds; drift is how far the round medians spread."""
    first = records[0]
    gpu_ms = summarize([sample for record in records for sample in record["gpu_ms"]])
    round_medians = [statistics.median(record["gpu_ms"]) for record in records]
    return Row(
        workload=first["workload"],
        size=tuple(first["size"]),
        variant=variant_name(first),
        config=config_label(first["config"]),
        gpu_ms=gpu_ms,
        rate=first["work"] / (gpu_ms.median * 1e-3) / first["rate_scale"],
        rate_unit=first["rate_unit"],
        drift_pct=(max(round_medians) - min(round_medians)) / gpu_ms.median * 100,
    )


def best_rows(records: Iterable[Record]) -> list[Row]:
    """Keep the fastest configuration of each variant at each size."""
    groups: dict[tuple[Any, ...], list[Record]] = defaultdict(list)
    for record in records:
        key = (
            record["workload"],
            tuple(record["size"]),
            variant_name(record),
            config_label(record["config"]),
        )
        groups[key].append(record)

    best: dict[tuple[Any, ...], Row] = {}
    for group in groups.values():
        row = summarize_rounds(group)
        key = (row.workload, row.size, row.variant)
        if key not in best or row.gpu_ms.median < best[key].gpu_ms.median:
            best[key] = row
    # Reversed so that 2D sizes read in order of row length, the dimension that matters.
    return sorted(best.values(), key=lambda row: (row.workload, row.size[::-1], row.variant))


def gpu_state_note(records: Sequence[Record]) -> str:
    """Summarize the conditions of a run, because a laptop GPU slows down as it heats up."""
    clocks = [int(record["gpu_state"]["clocks.sm"]) for record in records]
    temperatures = [int(record["gpu_state"]["temperature.gpu"]) for record in records]
    return (
        f"SM clock {min(clocks)}-{max(clocks)} MHz, "
        f"temperature {min(temperatures)}-{max(temperatures)} C across the run."
    )


def format_table(rows: Sequence[Row]) -> str:
    lines: list[str] = []
    previous: tuple[str, tuple[int, ...]] | None = None
    for row in rows:
        if (row.workload, row.size) != previous:
            previous = (row.workload, row.size)
            size = "x".join(str(dim) for dim in row.size)
            lines.append(f"\n{row.workload}  size={size}")
            lines.append(
                f"  {'variant':<33}{'median_us':>11}{'iqr_us':>9}"
                f"{row.rate_unit:>9}{'drift%':>8}  config"
            )
        lines.append(
            f"  {row.variant:<33}{row.gpu_ms.median * 1e3:>11.1f}{row.gpu_ms.iqr * 1e3:>9.1f}"
            f"{row.rate:>9.1f}{row.drift_pct:>8.1f}  {row.config}"
        )
    lines.append(f"\n{TIMER_NOTE}")
    return "\n".join(lines)
