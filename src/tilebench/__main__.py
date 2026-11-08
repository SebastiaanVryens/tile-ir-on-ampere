"""Run the benchmarks and print a comparison: `python -m tilebench`.

A run has two stages. Tuning times every configuration briefly to choose one per variant.
The rounds then measure only those, so the reported numbers are not the same timings that
picked the winner, and a cool GPU stays cool for long enough to measure them.
"""

import argparse
import json
import sys
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tilebench.kernels import WORKLOAD_NAMES
from tilebench.report import fastest_configs, format_table, gpu_state_note, pooled_rows
from tilebench.system import MAX_START_TEMP_C, describe_environment
from tilebench.targets import TARGETS, TILEIR_SITE, Target, get_target, run_worker

Record = dict[str, Any]

TUNE_REPEATS = 20


def tune(
    targets: Sequence[Target], workloads: Sequence[str], max_temp: int
) -> Iterator[list[Record]]:
    for workload in workloads:
        for target in targets:
            print(f"tuning: {workload} on {target.name}", flush=True)
            options = ("--repeats", str(TUNE_REPEATS), "--max-temp", str(max_temp))
            yield run_worker(target, workload, *options)


def plan_for(winners: Sequence[Record], target: Target, workload: str) -> list[list[Any]]:
    """List the sizes and configurations that one worker should measure."""
    return [
        [record["size"], record["config"]]
        for record in winners
        if record["target"] == target.name and record["workload"] == workload
    ]


def run_rounds(
    targets: Sequence[Target],
    workloads: Sequence[str],
    winners: Sequence[Record],
    rounds: int,
    max_temp: int,
) -> Iterator[list[Record]]:
    for round_index in range(rounds):
        # Rotating the order stops one target always running on a cooler GPU.
        offset = round_index % len(targets)
        order = [*targets[offset:], *targets[:offset]]
        for workload in workloads:
            for target in order:
                print(f"round {round_index + 1}/{rounds}: {workload} on {target.name}", flush=True)
                plan = json.dumps(plan_for(winners, target, workload))
                options = ("--plan", plan, "--max-temp", str(max_temp))
                records = run_worker(target, workload, *options)
                yield [record | {"round": round_index} for record in records]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--targets", nargs="+", default=[target.name for target in TARGETS])
    parser.add_argument("--workloads", nargs="+", default=list(WORKLOAD_NAMES))
    parser.add_argument(
        "--max-temp",
        type=int,
        default=MAX_START_TEMP_C,
        help="start bursts at or below this; a workload can ask for cooler",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    if not TILEIR_SITE.is_dir():
        print(f"error: {TILEIR_SITE} not found; run scripts/setup_wsl.sh first", file=sys.stderr)
        return 2

    targets = [get_target(name) for name in args.targets]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = args.out_dir / f"{stamp}.json"
    results: dict[str, Any] = {"environment": describe_environment(), "tuning": [], "records": []}

    # Saved after every worker, so that a run that fails late keeps what it measured.
    for batch in tune(targets, args.workloads, args.max_temp):
        results["tuning"] += batch
        path.write_text(json.dumps(results))

    winners = fastest_configs(results["tuning"])
    for batch in run_rounds(targets, args.workloads, winners, args.rounds, args.max_temp):
        results["records"] += batch
        path.write_text(json.dumps(results))

    print(format_table(pooled_rows(results["records"])))
    print(gpu_state_note(results["records"]))
    print(f"\nRaw results: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
